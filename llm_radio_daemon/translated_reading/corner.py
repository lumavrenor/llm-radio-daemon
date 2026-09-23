"""翻訳朗読コーナーのライフサイクル管理。docs/idea-translated-reading.md 案1。

literary_reading/corner.py と同じ骨格（セッションの選定・復元・完読・失敗
フォールバックをここに閉じ込める）だが、原文をそのままTTSへ渡さず、
biography_reading と同じく**チャンクごとにLLMを通す**（今回は「解説」ではなく
「翻訳ナレーション」）。原典は常に Project Gutenberg（英語）―― 翻訳先の言語に
かかわらず、このコーナーの前提は「英語の原文を、放送言語へ訳しながら読む」こと。

「翻訳朗読コーナーが動かないことは異常ではない。放送が止まることだけが異常。」
（literary_reading / biography_reading と同じ考え方）
"""

from __future__ import annotations

import json
import logging
import random
import threading

from ..aozora.corpus import WorkMeta
from ..config import CastMember, ContentConfig, LLMConfig, TranslatedReadingParams
from ..gutenberg.corpus import GutenbergCorpus
from ..gutenberg.parser import parse_work_text as parse_gutenberg_text
from ..literary_reading.curator import select_work_llm
from ..db import TopicStore
from ..script import Script, ScriptLine
from ..script.ollama_client import pick_speakers
from ..state import SharedState
from .comment import generate_beat, generate_translation, translate_title
from .session import TranslatedReadingSession

logger = logging.getLogger(__name__)

_MAX_WORK_FAILURES = 3    # 作品の読み込み/パース失敗がこれ続いたらコーナー無効化
_MAX_OLLAMA_FAILURES = 3  # 翻訳生成の連続失敗がこれでコーナー終了（Beatの失敗はカウントしない）


class TranslatedReadingCorner:
    def __init__(
        self,
        content: ContentConfig,
        store: TopicStore,
        state: SharedState,
        cast: list[CastMember],
        llm_config: LLMConfig,
    ):
        self._content = content
        self._tp: TranslatedReadingParams = content.translated_reading or TranslatedReadingParams()
        self._cast = cast
        self._llm_config = llm_config
        self._store = store
        self._state = state
        self._corpus = GutenbergCorpus(
            self._tp.corpus_dir,
            auto_fetch=self._tp.auto_fetch,
            allow_translations=True,  # Gutenbergのカタログには訳者の役割が無く判別できない（gutenberg/corpus.py参照）
            lang="en",
            catalog_url=self._tp.catalog_url,
            text_url_template=self._tp.text_url_template,
        )

        # ナレーター・つっこみ担当はセッション開始時に [[cast]] から抽選する（literary_reading と同じ）。
        self._narrator: CastMember | None = None
        self._commentator: CastMember | None = None

        self._session: TranslatedReadingSession | None = None
        self._disabled = False
        self._work_failures = 0
        self._ollama_failures = 0
        self._intro_pending = False

        # 次作品のプリフェッチ。翻訳中に1回だけバックグラウンドで取得し、次のセッション選定で優先する。
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
            out.append(self._script(self._intro_lines(session), title="translated_reading:intro"))

        self._maybe_prefetch_next(session)

        chunk = session.next_chunk()
        if chunk is None:
            self._finish_session(session)
            return out

        result = generate_translation(
            title=session.title,
            author=session.author,
            rolling_summary=session.rolling_summary,
            chunk=chunk,
            narrator=self._narrator,
            llm_config=self._llm_config,
            tone_hint=self._content.tone_hint,
            translate_lines=self._tp.translate_lines,
            summary_max_chars=self._tp.summary_max_chars,
        )
        if result is None:
            self._ollama_failures += 1
            if self._ollama_failures >= _MAX_OLLAMA_FAILURES:
                logger.error(
                    "translated_reading: translation generation failed %d times in a row. Ending the corner", self._ollama_failures
                )
                self._disabled = True
            self._publish_state(session)
            return out

        self._ollama_failures = 0
        lines, summary_update = result
        session.apply_summary(summary_update)
        session.record_translation(" ".join(l.text for l in lines))
        session.cursor += 1
        self._store.save_translated_reading_summary(session.work_id, session.rolling_summary)

        lines[-1].pad_ms = self._tp.pause_chapter_ms if chunk.is_chapter_head else self._tp.pause_ms
        lines[-1].translated_chunk_index = chunk.index
        script = self._script(lines, title=f"translated_reading:{session.work_id}:{chunk.index}")
        self._store.record_translated_reading_log(
            session.work_id, chunk.index, "translate",
            json.dumps({"lines": [{"speaker": l.speaker, "text": l.text} for l in lines]}, ensure_ascii=False),
        )
        out.append(script)

        if session.is_beat_due():
            beat = self._run_beat(session, chunk.index)
            if beat is not None:
                out.append(beat)

        self._publish_state(session)
        return out

    def end_corner(self) -> list[Script]:
        """時間帯を抜けたときに1回だけ呼ぶ。締めのアナウンスを返し、状態をクリアする。"""
        lines: list[Script] = []
        if self._session is not None and not self._disabled:
            lines.append(self._script(self._outro_lines(self._session), title="translated_reading:outro"))
        self._session = None
        self._intro_pending = False
        self._ollama_failures = 0
        self._clear_state()
        return lines

    # --- セッション選定・復元 -------------------------------------------

    def _start_session(self) -> bool:
        # まず未完のセッションがあれば再開する（再起動をまたいだ継続）。
        resume = self._store.latest_unfinished_translated_reading_session()
        if resume is not None:
            meta = self._corpus.ensure_text(resume["work_id"])
            if meta is not None and meta.path.exists():
                if self._load_session(
                    meta, resume["cursor"], resume["rolling_summary"],
                    display_title=resume["title"], display_author=resume["author"],
                ):
                    return True
            else:
                logger.warning(
                    "translated_reading: body text for unfinished session work_id=%s not found. Selecting a new work",
                    resume["work_id"],
                )

        for _ in range(_MAX_WORK_FAILURES):
            meta = self._select_work()
            if meta is None:
                logger.warning("translated_reading: no target works available. Disabling the translated reading corner")
                self._disabled = True
                return False
            saved = self._store.load_translated_reading_session(meta.work_id)
            cursor = saved["cursor"] if saved and not saved["finished_at"] else 0
            summary = saved["rolling_summary"] if saved and not saved["finished_at"] else ""
            display_title = saved["title"] if saved else None
            display_author = saved["author"] if saved else None
            if self._load_session(meta, cursor, summary, display_title, display_author):
                return True
            if self._disabled:
                return False
        return False

    def _load_session(
        self,
        meta: WorkMeta,
        cursor: int,
        summary: str,
        display_title: str | None = None,
        display_author: str | None = None,
    ) -> bool:
        try:
            text = self._corpus.load_text(meta)
            chunks = parse_gutenberg_text(
                text,
                chunk_target_chars=self._tp.chunk_target_chars,
                chunk_max_chars=self._tp.chunk_max_chars,
                chunk_min_chars=self._tp.chunk_min_chars,
            )
        except Exception:
            logger.exception("translated_reading: failed to load/parse work work_id=%s. Moving to next candidate", meta.work_id)
            chunks = []

        if len(chunks) < 2:
            self._work_failures += 1
            if self._work_failures >= _MAX_WORK_FAILURES:
                logger.error("translated_reading: work failures reached %d. Stopping the translated reading corner and returning to normal broadcast", self._work_failures)
                self._disabled = True
            return False

        # 書名・著者名の日本語化は作品ごとに1回だけ（DBに保存済みならそれを使い回す）。
        # これが無いと英語の原題がそのままアルファベットで台本に混ざり、TTSが正しく読めない。
        if display_title is None:
            display_title, display_author = translate_title(meta.title, meta.author, self._llm_config)

        self._work_failures = 0
        self._prefetch_started = False  # このセッション用に次作品プリフェッチをやり直す
        self._assign_cast()  # このセッションのナレーター・つっこみ担当を抽選
        cursor = max(0, min(cursor, len(chunks)))
        self._store.open_translated_reading_session(meta.work_id, display_title, display_author or "", len(chunks))
        self._session = TranslatedReadingSession(
            meta.work_id, display_title, display_author or "", chunks,
            cursor=cursor,
            rolling_summary=summary,
            beat_interval=self._tp.beat_interval,
            summary_max_chars=self._tp.summary_max_chars,
        )
        self._intro_pending = cursor == 0  # 冒頭からのときだけ導入を入れる
        self._publish_state(self._session)
        logger.info(
            "translated_reading: starting \"%s\" (%s, original title: %s / %s) cursor=%d/%d",
            display_title, display_author, meta.title, meta.author, cursor, len(chunks),
        )
        return True

    def _select_work(self) -> WorkMeta | None:
        exclude = self._store.recently_read_translated_reading_work_ids(self._tp.reread_after_days)

        # 1) 明示指定（[[content.works]]）があればそこから（無ければ取得する）。
        explicit_ids = [
            str(w.get("work_id", "")).strip() for w in self._tp.works if w.get("work_id")
        ]
        if explicit_ids:
            fresh = [i for i in explicit_ids if i not in exclude]
            for wid in fresh or explicit_ids:
                meta = self._corpus.ensure_text(wid)
                if meta is not None:
                    return meta
            return None

        # 2) 翻訳中にプリフェッチした次作品があれば優先する。
        if self._prefetched_work_id and self._prefetched_work_id not in exclude:
            wid, self._prefetched_work_id = self._prefetched_work_id, None
            meta = self._corpus.ensure_text(wid)
            if meta is not None:
                return meta

        # 3) 候補リストから選ぶ（random / LLM キュレーション）。
        candidates = self._corpus.candidates(exclude_ids=exclude)
        if not candidates:
            # 全候補が最近読んだ扱いなら除外を無視して選び直す（無音より再読を優先）。
            candidates = self._corpus.candidates(exclude_ids=set())
        if not candidates:
            return None

        chosen_id: str | None = None
        if self._tp.work_selector == "llm":
            shortlist = candidates
            if len(shortlist) > self._tp.llm_candidate_count:
                shortlist = random.sample(shortlist, self._tp.llm_candidate_count)
            chosen_id = select_work_llm(
                shortlist,
                recent_titles=self._store.recent_translated_reading_titles(5),
                llm_config=self._llm_config,
                tone_hint=self._content.tone_hint,
            )

        # 選ばれた1本を先頭に、取得できなければ他候補を数本試す。
        order = [c.work_id for c in candidates]
        random.shuffle(order)
        if chosen_id is not None:
            order = [chosen_id] + [w for w in order if w != chosen_id]
        for wid in order[:3]:
            meta = self._corpus.ensure_text(wid)
            if meta is not None:
                return meta
        return None

    def _maybe_prefetch_next(self, session: TranslatedReadingSession) -> None:
        """翻訳中に1回だけ、次に読みそうな作品をバックグラウンドで取得しておく。"""
        if not self._tp.prefetch_next or self._prefetch_started or self._disabled:
            return
        self._prefetch_started = True
        threading.Thread(
            target=self._prefetch_worker,
            args=(session.work_id,),
            name="TranslatedReadingPrefetch",
            daemon=True,
        ).start()

    def _prefetch_worker(self, current_work_id: str) -> None:
        try:
            exclude = set(self._store.recently_read_translated_reading_work_ids(self._tp.reread_after_days))
            exclude.add(current_work_id)
            cands = self._corpus.candidates(exclude_ids=exclude)
            if not cands:
                return
            pick = random.choice(cands)
            meta = self._corpus.ensure_text(pick.work_id)
            if meta is not None:
                self._prefetched_work_id = pick.work_id
                logger.info("translated_reading: prefetched next work \"%s\" (%s)", meta.title, meta.author)
        except Exception:
            logger.exception("translated_reading: failed to prefetch next work (ignoring and continuing)")

    def _finish_session(self, session: TranslatedReadingSession) -> None:
        self._store.finish_translated_reading_session(session.work_id)
        logger.info("translated_reading: finished translating \"%s\". Selecting a different work next time", session.title)
        self._session = None
        self._intro_pending = False
        self._clear_state()

    # --- Beat（感想） ---------------------------------------------------

    def _run_beat(self, session: TranslatedReadingSession, chunk_index: int) -> Script | None:
        lines = generate_beat(
            title=session.title,
            author=session.author,
            recent_translated_text=session.recent_translated_text(),
            narrator=self._narrator,
            commentator=self._commentator,
            llm_config=self._llm_config,
            comment_lines=self._tp.comment_lines,
        )
        if not lines:
            return None  # Beatは無くても翻訳朗読は続くので失敗をカウントしない

        lines[0].pad_ms = self._tp.pause_beat_ms
        lines[-1].pad_ms = self._tp.pause_beat_ms
        script = self._script(lines, title=f"translated_reading:{session.work_id}:comment")
        self._store.record_translated_reading_log(
            session.work_id, chunk_index, "comment",
            json.dumps({"lines": [{"speaker": l.speaker, "text": l.text} for l in lines]}, ensure_ascii=False),
        )
        return script

    # --- 補助 -----------------------------------------------------------

    def _script(self, lines: list[ScriptLine], title: str) -> Script:
        return Script(topic_id=None, topic_title=title, lines=lines, is_translated_reading=True)

    def _intro_lines(self, session: TranslatedReadingSession) -> list[ScriptLine]:
        narrator = self._narrator
        by = f"{session.author}の" if session.author else ""
        return [
            ScriptLine(narrator.id, "さて、ここからは翻訳朗読のお時間です。"),
            ScriptLine(
                narrator.id,
                f"今夜は、{by}「{session.title}」。海外の作品を訳しながらお届けします。",
            ),
            ScriptLine(self._commentator.id, "では、ゆっくり聴いていきましょう。"),
        ]

    def _outro_lines(self, session: TranslatedReadingSession) -> list[ScriptLine]:
        return [
            ScriptLine(self._commentator.id, "……今夜はこのあたりまでですね。"),
            ScriptLine(self._narrator.id, f"「{session.title}」、続きはまた明日。翻訳朗読のお時間でした。"),
        ]

    def _publish_state(self, session: TranslatedReadingSession) -> None:
        self._state.reading_work_id = session.work_id
        self._state.reading_now_playing = (
            f"{session.title} / {session.author} / Project Gutenberg（翻訳）"
        )
        self._state.reading_progress = session.progress_label()
        self._state.reading_cast_ids = tuple(
            dict.fromkeys((self._narrator.id, self._commentator.id))
        )

    def _clear_state(self) -> None:
        self._state.reading_work_id = None
        self._state.reading_now_playing = ""
        self._state.reading_progress = ""
        self._state.reading_cast_ids = ()
