"""読書コーナーのライフサイクル管理（v4 §10.6 / §10.7 / §10.8）。

ScriptThread はこのクラスの ``produce_batch()`` を時間帯内で繰り返し呼ぶだけ。
セッションの選定・復元・完読・失敗フォールバックはすべてここに閉じ込める。

「読書コーナーが動かないことは異常ではない。放送が止まることだけが異常。」（§10.8）

原典は言語で選ぶ
---------------
``[locale] lang`` が ``ja`` なら青空文庫（``aozora/``）、``en`` なら
Project Gutenberg（``gutenberg/``）。取得プロトコルも本文の書式も権利の判断も
まったく別なので、共通の抽象を被せず**取得層を丸ごと差し替える**
（理由は ``gutenberg/__init__.py`` のヘッダ）。このクラスから下は
``WorkMeta`` / ``Candidate`` / ``Chunk`` の3つの型だけで書いてあり、
どちらの corpus でもそのまま通る。
"""

from __future__ import annotations

import json
import logging
import random
import threading

from .. import language
from ..aozora import Chunk
from ..aozora.corpus import AozoraCorpus, WorkMeta
from ..aozora.parser import parse_work_text
from ..config import CastMember, ContentConfig, LLMConfig, LiteraryReadingParams
from ..gutenberg.corpus import GutenbergCorpus
from ..gutenberg.parser import parse_work_text as parse_gutenberg_text
from ..db import TopicStore
from ..script import Script, ScriptLine
from ..script.ollama_client import pick_speakers
from ..state import SharedState
from .comment import generate_comment
from .curator import select_work_llm
from .session import LiteraryReadingSession

logger = logging.getLogger(__name__)

_MAX_WORK_FAILURES = 3   # 作品の読み込み/パース失敗がこれ続いたらコーナー無効化（§10.8）
_MAX_OLLAMA_FAILURES = 3  # 感想生成の連続失敗がこれでコーナー終了（§10.8）


class LiteraryReadingCorner:
    def __init__(
        self,
        content: ContentConfig,
        store: TopicStore,
        state: SharedState,
        cast: list[CastMember],
        llm_config: LLMConfig,
    ):
        self._content = content
        self._rp: LiteraryReadingParams = content.literary_reading or LiteraryReadingParams()
        self._cast = cast
        self._llm_config = llm_config
        self._store = store
        self._state = state
        self._en = language.current() == "en"
        if self._en:
            self._corpus = GutenbergCorpus(
                self._rp.corpus_dir,
                auto_fetch=self._rp.auto_fetch,
                allow_translations=self._rp.allow_translations,
                lang=language.current(),
                catalog_url=self._rp.catalog_url,
                text_url_template=self._rp.text_url_template,
            )
        else:
            self._corpus = AozoraCorpus(
                self._rp.corpus_dir,
                auto_fetch=self._rp.auto_fetch,
                allow_translations=self._rp.allow_translations,
            )

        # 朗読・つっこみ担当はセッション開始時に [[cast]] から抽選する（§10.5）。
        self._narrator: CastMember | None = None
        self._commentator: CastMember | None = None

        self._session: LiteraryReadingSession | None = None
        self._disabled = False
        self._work_failures = 0
        self._ollama_failures = 0
        self._intro_pending = False

        # 次作品のプリフェッチ（§10.2）。朗読中に1回だけバックグラウンドで取得し、
        # 次のセッション選定でこれを優先する。
        self._prefetched_work_id: str | None = None
        self._prefetch_started = False

    def _assign_cast(self) -> None:
        picked = pick_speakers(self._content, self._cast)
        self._narrator = picked[0]
        self._commentator = picked[1] if len(picked) > 1 else picked[0]

    # --- 公開 API ---------------------------------------------------------

    @property
    def disabled(self) -> bool:
        return self._disabled

    def produce_batch(self) -> list[Script]:
        """次に script_queue へ積むべき Script のリストを返す（時間帯内で繰り返し呼ばれる）。"""
        if self._disabled:
            return []

        if self._session is None and not self._start_session():
            return []

        session = self._session
        assert session is not None
        out: list[Script] = []

        if self._intro_pending:
            self._intro_pending = False
            out.append(self._script(self._intro_lines(session), title="literary_reading:intro"))

        self._maybe_prefetch_next(session)

        batch, is_beat = session.peek_batch()
        if not batch:
            self._finish_session(session)
            return out

        read_lines = [
            ScriptLine(
                speaker=self._narrator.id,
                text=ch.speech_text,
                reading_chunk_index=ch.index,
                pad_ms=self._pad_for(ch, batch, is_beat),
                # 朗読中だけ BGM を深く絞る（§10.4）。感想パートの行には付けないので、
                # AnnounceThread 側で自動的に通常の深さへ戻る。
                duck_db=self._rp.duck_db,
            )
            for ch in batch
        ]
        out.append(self._script(read_lines, title=f"literary_reading:{session.work_id}:{session.cursor}"))
        self._store.record_literary_reading_log(session.work_id, batch[0].index, "read")

        # cursor（enqueue 位置）はここで進める。再生位置（DB cursor）は AnnounceThread が
        # 実際に流れた時点で進める（§10.6）。再起動時は DB cursor から復元する。
        session.cursor += len(batch)

        if is_beat:
            comment = self._run_beat(session)
            if comment is not None:
                out.append(comment)

        self._publish_state(session)
        return out

    def end_corner(self) -> list[Script]:
        """時間帯を抜けたときに1回だけ呼ぶ。締めのアナウンスを返し、状態をクリアする。"""
        lines: list[Script] = []
        if self._session is not None and not self._disabled:
            lines.append(self._script(self._outro_lines(self._session), title="literary_reading:outro"))
        self._session = None
        self._intro_pending = False
        self._ollama_failures = 0
        self._clear_state()
        return lines

    # --- セッション選定・復元 -------------------------------------------

    def _start_session(self) -> bool:
        # まず未完のセッションがあれば再開する（再起動をまたいだ継続・§10.6）。
        resume = self._store.latest_unfinished_literary_reading_session()
        if resume is not None:
            meta = self._corpus.ensure_text(resume["work_id"])
            if meta is not None and meta.path.exists():
                if self._load_session(meta, resume["cursor"], resume["rolling_summary"]):
                    return True
            else:
                logger.warning(
                    "literary_reading: body text for unfinished session work_id=%s not found. Selecting a new work",
                    resume["work_id"],
                )

        for _ in range(_MAX_WORK_FAILURES):
            meta = self._select_work()
            if meta is None:
                logger.warning("literary_reading: no target works available. Disabling the reading corner (§10.8)")
                self._disabled = True
                return False
            saved = self._store.load_literary_reading_session(meta.work_id)
            cursor = saved["cursor"] if saved and not saved["finished_at"] else 0
            summary = saved["rolling_summary"] if saved and not saved["finished_at"] else ""
            if self._load_session(meta, cursor, summary):
                return True
            if self._disabled:
                return False
        return False

    def _load_session(self, meta: WorkMeta, cursor: int, summary: str) -> bool:
        try:
            text = self._corpus.load_text(meta)
            if self._en:
                # ルビが無いので ruby_mode は渡さない（英語には対応するものが無い）。
                chunks = parse_gutenberg_text(
                    text,
                    chunk_target_chars=self._rp.chunk_target_chars,
                    chunk_max_chars=self._rp.chunk_max_chars,
                    chunk_min_chars=self._rp.chunk_min_chars,
                )
            else:
                chunks = parse_work_text(
                    text,
                    ruby_mode=self._rp.ruby_mode,
                    chunk_target_chars=self._rp.chunk_target_chars,
                    chunk_max_chars=self._rp.chunk_max_chars,
                    chunk_min_chars=self._rp.chunk_min_chars,
                )
        except Exception:
            logger.exception("literary_reading: failed to load/parse work work_id=%s. Moving to next candidate", meta.work_id)
            chunks = []

        if len(chunks) < 2:
            self._work_failures += 1
            if self._work_failures >= _MAX_WORK_FAILURES:
                logger.error("literary_reading: work failures reached %d. Stopping the reading corner and returning to normal broadcast (§10.8)", self._work_failures)
                self._disabled = True
            return False

        self._work_failures = 0
        self._prefetch_started = False  # このセッション用に次作品プリフェッチをやり直す
        self._assign_cast()  # このセッションの朗読・つっこみ担当を抽選
        cursor = max(0, min(cursor, len(chunks)))
        self._store.open_literary_reading_session(meta.work_id, meta.title, meta.author, len(chunks))
        self._session = LiteraryReadingSession(
            meta.work_id, meta.title, meta.author, chunks,
            cursor=cursor,
            rolling_summary=summary,
            beat_interval=self._rp.beat_interval,
            summary_max_chars=self._rp.summary_max_chars,
        )
        self._intro_pending = cursor == 0  # 冒頭からのときだけ導入を入れる
        self._publish_state(self._session)
        logger.info(
            "literary_reading: starting \"%s\" (%s) cursor=%d/%d",
            meta.title, meta.author, cursor, len(chunks),
        )
        return True

    def _select_work(self) -> WorkMeta | None:
        exclude = self._store.recently_read_work_ids(self._rp.reread_after_days)

        # 1) 明示指定（[[content.works]]）があればそこから（無ければ取得する）。
        explicit_ids = [
            str(w.get("work_id", "")).strip() for w in self._rp.works if w.get("work_id")
        ]
        if explicit_ids:
            fresh = [i for i in explicit_ids if i not in exclude]
            for wid in fresh or explicit_ids:
                meta = self._corpus.ensure_text(wid)
                if meta is not None:
                    return meta
            return None

        # 2) 朗読中にプリフェッチした次作品があれば優先する（§10.2）。
        if self._prefetched_work_id and self._prefetched_work_id not in exclude:
            wid, self._prefetched_work_id = self._prefetched_work_id, None
            meta = self._corpus.ensure_text(wid)
            if meta is not None:
                return meta

        # 3) 候補リストから選ぶ（random / LLM キュレーション・§10.2）。
        candidates = self._corpus.candidates(
            exclude_ids=exclude, allow_translations=self._rp.allow_translations
        )
        if not candidates:
            # 全候補が最近読んだ扱いなら除外を無視して選び直す（無音より再読を優先）。
            candidates = self._corpus.candidates(
                exclude_ids=set(), allow_translations=self._rp.allow_translations
            )
        if not candidates:
            return None

        chosen_id: str | None = None
        if self._rp.work_selector == "llm":
            shortlist = candidates
            if len(shortlist) > self._rp.llm_candidate_count:
                shortlist = random.sample(shortlist, self._rp.llm_candidate_count)
            chosen_id = select_work_llm(
                shortlist,
                recent_titles=self._store.recent_literary_reading_titles(5),
                llm_config=self._llm_config,
                tone_hint=self._content.tone_hint,
            )

        # 選ばれた1本を先頭に、取得できなければ他候補を数本試す（§10.2-4 / §10.8）。
        order = [c.work_id for c in candidates]
        random.shuffle(order)
        if chosen_id is not None:
            order = [chosen_id] + [w for w in order if w != chosen_id]
        for wid in order[:3]:
            meta = self._corpus.ensure_text(wid)
            if meta is not None:
                return meta
        return None

    def _maybe_prefetch_next(self, session: LiteraryReadingSession) -> None:
        """朗読中に1回だけ、次に読みそうな作品をバックグラウンドで取得しておく（§10.2）。"""
        if not self._rp.prefetch_next or self._prefetch_started or self._disabled:
            return
        self._prefetch_started = True
        threading.Thread(
            target=self._prefetch_worker,
            args=(session.work_id,),
            name="LiteraryReadingPrefetch",
            daemon=True,
        ).start()

    def _prefetch_worker(self, current_work_id: str) -> None:
        try:
            exclude = set(self._store.recently_read_work_ids(self._rp.reread_after_days))
            exclude.add(current_work_id)
            cands = self._corpus.candidates(
                exclude_ids=exclude, allow_translations=self._rp.allow_translations
            )
            if not cands:
                return
            pick = random.choice(cands)
            meta = self._corpus.ensure_text(pick.work_id)
            if meta is not None:
                self._prefetched_work_id = pick.work_id
                logger.info("literary_reading: prefetched next work \"%s\" (%s)", meta.title, meta.author)
        except Exception:
            logger.exception("literary_reading: failed to prefetch next work (ignoring and continuing)")

    def _finish_session(self, session: LiteraryReadingSession) -> None:
        self._store.finish_literary_reading_session(session.work_id)
        self._store.record_literary_reading_log(session.work_id, session.total_chunks, "read", "finished")
        logger.info("literary_reading: finished reading \"%s\". Selecting a different work next time", session.title)
        self._session = None
        self._intro_pending = False
        self._clear_state()

    # --- Beat（感想） ---------------------------------------------------

    def _run_beat(self, session: LiteraryReadingSession) -> Script | None:
        result = generate_comment(
            title=session.title,
            author=session.author,
            rolling_summary=session.rolling_summary,
            recent_chunks=session.recent_chunks(self._rp.beat_interval),
            narrator=self._narrator,
            commentator=self._commentator,
            llm_config=self._llm_config,
            comment_lines=self._rp.comment_lines,
            summary_max_chars=self._rp.summary_max_chars,
        )
        if result is None:
            self._ollama_failures += 1
            if self._ollama_failures >= _MAX_OLLAMA_FAILURES:
                logger.error("literary_reading: comment generation failed %d times in a row. Ending the reading corner (§10.8)", self._ollama_failures)
                self._disabled = True
            return None

        self._ollama_failures = 0
        lines, summary_update = result
        session.apply_summary(summary_update)
        self._store.save_literary_reading_summary(session.work_id, session.rolling_summary)

        if lines:
            lines[0].pad_ms = self._rp.pause_beat_ms
            lines[-1].pad_ms = self._rp.pause_beat_ms
        script = self._script(lines, title=f"literary_reading:{session.work_id}:comment")
        self._store.record_literary_reading_log(
            session.work_id, session.cursor, "comment",
            json.dumps({"lines": [{"speaker": l.speaker, "text": l.text} for l in lines]}, ensure_ascii=False),
        )
        return script

    # --- 補助 -----------------------------------------------------------

    def _pad_for(self, ch: Chunk, batch: list[Chunk], is_beat: bool) -> int:
        if ch.is_chapter_head:
            return self._rp.pause_chapter_ms
        if is_beat and ch is batch[-1]:
            return self._rp.pause_beat_ms
        return self._rp.pause_ms

    def _script(self, lines: list[ScriptLine], title: str) -> Script:
        return Script(topic_id=None, topic_title=title, lines=lines, is_literary_reading=True)

    def _intro_lines(self, session: LiteraryReadingSession) -> list[ScriptLine]:
        """コーナーの導入（LLM は通さない・固定文）。

        **LLM を通らないので language.LANGUAGE_GUIDANCE が効かない。** 言語ごとの
        定型文をここに持つ（filler.py のテンプレートと同じ扱い）。
        """
        narrator = self._narrator
        if self._en:
            by = f"{session.author}'s " if session.author else ""
            return [
                ScriptLine(narrator.id, "Right. This is the reading."),
                ScriptLine(
                    narrator.id,
                    f"Tonight, {by}{session.title}, from Project Gutenberg.",
                ),
                ScriptLine(self._commentator.id, "Settle in."),
            ]
        return [
            ScriptLine(narrator.id, "さて、ここからは読書のお時間です。"),
            ScriptLine(narrator.id, f"今夜は、{session.author}の「{session.title}」。青空文庫からお届けします。"),
            ScriptLine(self._commentator.id, "では、ゆっくり聴いていきましょう。"),
        ]

    def _outro_lines(self, session: LiteraryReadingSession) -> list[ScriptLine]:
        if self._en:
            return [
                ScriptLine(self._commentator.id, "And that is about as far as we go tonight."),
                ScriptLine(
                    self._narrator.id,
                    f"{session.title}. We pick it up again tomorrow. That was the reading.",
                ),
            ]
        return [
            ScriptLine(self._commentator.id, "……今夜はこのあたりまでですね。"),
            ScriptLine(self._narrator.id, f"「{session.title}」、続きはまた明日。読書のお時間でした。"),
        ]

    def _publish_state(self, session: LiteraryReadingSession) -> None:
        self._state.reading_work_id = session.work_id
        source_label = "Project Gutenberg" if self._en else "青空文庫"
        self._state.reading_now_playing = (
            f"{session.title} / {session.author} / {source_label}"
        )
        self._state.reading_progress = session.progress_label()
        # 読書コーナー中は表示を朗読・つっこみの2名に絞る（§10.11）。
        # 両者が host に代行された場合は重複を潰す。
        self._state.reading_cast_ids = tuple(
            dict.fromkeys(
                (self._narrator.id, self._commentator.id)
            )
        )

    def _clear_state(self) -> None:
        self._state.reading_work_id = None
        self._state.reading_now_playing = ""
        self._state.reading_progress = ""
        self._state.reading_cast_ids = ()
        self._state.duck_db_override = None
