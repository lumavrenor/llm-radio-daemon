"""偉人伝コーナーのライフサイクル管理。

ScriptThread はこのクラスの ``produce_batch()`` を時間帯内で繰り返し呼ぶだけ。
人物の選定・復元・紹介済み・失敗フォールバックはすべてここに閉じ込める
（literary_reading/corner.py と同じ構造）。

対象人物は figures_file に明示指定されたものだけを使う。「政治・宗教に
無関係で1950年より前に活動した人物」という条件は自動選定だと守り切れないため、
選定そのものをLLM任せにしない方針にした（09-05相談）。

「偉人伝コーナーが動かないことは異常ではない。放送が止まることだけが異常。」
（literary_reading と同じ考え方）
"""

from __future__ import annotations

import json
import logging
import random

from .. import language
from ..config import BiographyReadingParams, CastMember, ContentConfig, LLMConfig
from ..db import TopicStore
from ..script import Script, ScriptLine
from ..script.ollama_client import pick_speakers
from ..state import SharedState
from ..wiki_bio.fetch import FigureMeta, WikiBioCorpus
from ..wiki_bio.parser import parse_article_text
from .comment import generate_segment
from .session import BiographyReadingSession

logger = logging.getLogger(__name__)

_MAX_FIGURE_FAILURES = 3   # 記事の取得/パース失敗がこれ続いたらコーナー無効化
_MAX_OLLAMA_FAILURES = 3   # 生成の連続失敗がこれでコーナー終了


class BiographyReadingCorner:
    def __init__(
        self,
        content: ContentConfig,
        store: TopicStore,
        state: SharedState,
        cast: list[CastMember],
        llm_config: LLMConfig,
    ):
        self._content = content
        self._bp: BiographyReadingParams = content.biography_reading or BiographyReadingParams()
        self._cast = cast
        self._llm_config = llm_config
        self._store = store
        self._state = state
        self._en = language.current() == "en"
        self._corpus = WikiBioCorpus(self._bp.data_dir)

        # MC・その他の出演者はセッション開始時に [[cast]] から抽選する
        # （host_id を指定していれば pick_speakers がその人を先頭＝MCへ寄せる）。
        self._mc: CastMember | None = None
        self._others: list[CastMember] = []

        self._session: BiographyReadingSession | None = None
        self._disabled = False
        self._figure_failures = 0
        self._ollama_failures = 0
        self._intro_pending = False

    def _assign_cast(self) -> None:
        # pick_speakers() は [debug] pinned_cast_ids 固定時、その id 並びを
        # 人数抽選なしでそのまま返す（デバッグ用に他コーナーと同じ顔ぶれで
        # 固定するため）。そのため pinned の人数が max_speakers を超えると
        # ひな壇（_reading_poses、朗読・つっこみ用の単一横列）に全員並びきらず
        # 画面端で見切れる。ここで max_speakers に切り詰めておく。
        picked = pick_speakers(self._content, self._cast)[: self._content.max_speakers]
        self._mc = picked[0]
        self._others = picked[1:] or [picked[0]]

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
            out.append(self._script(self._intro_lines(session), title="biography_reading:intro"))

        chunk = session.next_chunk()
        if chunk is None:
            self._finish_session(session)
            return out

        result = generate_segment(
            figure_name=session.title,
            rolling_summary=session.rolling_summary,
            chunk=chunk,
            mc=self._mc,
            others=self._others,
            llm_config=self._llm_config,
            tone_hint=self._content.tone_hint,
            segment_lines=self._bp.segment_lines,
            summary_max_chars=self._bp.summary_max_chars,
        )
        if result is None:
            self._ollama_failures += 1
            if self._ollama_failures >= _MAX_OLLAMA_FAILURES:
                logger.error(
                    "biography_reading: generation failed %d times in a row. Ending the corner", self._ollama_failures
                )
                self._disabled = True
            self._publish_state(session)
            return out

        self._ollama_failures = 0
        lines, summary_update = result
        session.apply_summary(summary_update)
        session.cursor += 1
        self._store.save_biography_reading_summary(session.figure_id, session.rolling_summary)

        lines[-1].biography_chunk_index = chunk.index
        script = self._script(lines, title=f"biography_reading:{session.figure_id}:{chunk.index}")
        self._store.record_biography_reading_log(
            session.figure_id, chunk.index,
            json.dumps({"lines": [{"speaker": l.speaker, "text": l.text} for l in lines]}, ensure_ascii=False),
        )
        out.append(script)
        self._publish_state(session)
        return out

    def end_corner(self) -> list[Script]:
        """時間帯を抜けたときに1回だけ呼ぶ。締めのアナウンスを返し、状態をクリアする。"""
        lines: list[Script] = []
        if self._session is not None and not self._disabled:
            lines.append(self._script(self._outro_lines(self._session), title="biography_reading:outro"))
        self._session = None
        self._intro_pending = False
        self._ollama_failures = 0
        self._clear_state()
        return lines

    # --- セッション選定・復元 -------------------------------------------

    def _start_session(self) -> bool:
        # まず未完のセッションがあれば再開する（再起動をまたいだ継続）。
        resume = self._store.latest_unfinished_biography_reading_session()
        if resume is not None:
            fig = self._figure_by_id(resume["figure_id"])
            meta = self._fetch_for(fig) if fig is not None else None
            if meta is not None:
                if self._load_session(fig, meta, resume["cursor"], resume["rolling_summary"]):
                    return True
            else:
                logger.warning(
                    "biography_reading: article for unfinished session figure_id=%s not found. Selecting the next figure",
                    resume["figure_id"],
                )

        for _ in range(_MAX_FIGURE_FAILURES):
            fig = self._select_figure()
            if fig is None:
                logger.warning("biography_reading: no figures found in figures_file. Disabling the corner")
                self._disabled = True
                return False
            meta = self._fetch_for(fig)
            if meta is None:
                self._figure_failures += 1
                if self._figure_failures >= _MAX_FIGURE_FAILURES:
                    logger.error("biography_reading: repeated article fetch failures. Disabling the corner")
                    self._disabled = True
                    return False
                continue
            saved = self._store.load_biography_reading_session(fig["id"])
            cursor = saved["cursor"] if saved and not saved["finished_at"] else 0
            summary = saved["rolling_summary"] if saved and not saved["finished_at"] else ""
            if self._load_session(fig, meta, cursor, summary):
                return True
            if self._disabled:
                return False
        return False

    def _fetch_for(self, fig: dict | None) -> FigureMeta | None:
        if fig is None:
            return None
        return self._corpus.ensure_text(fig["id"], fig["wiki_title"], self._bp.lang)

    def _load_session(self, fig: dict, meta: FigureMeta, cursor: int, summary: str) -> bool:
        try:
            text = WikiBioCorpus.load_text(meta)
            chunks = parse_article_text(
                text,
                chunk_target_chars=self._bp.chunk_target_chars,
                chunk_max_chars=self._bp.chunk_max_chars,
                chunk_min_chars=self._bp.chunk_min_chars,
            )
        except Exception:
            logger.exception(
                "biography_reading: failed to load/parse article figure_id=%s. Moving to the next candidate", fig["id"]
            )
            chunks = []

        if len(chunks) < 2:
            self._figure_failures += 1
            if self._figure_failures >= _MAX_FIGURE_FAILURES:
                logger.error(
                    "biography_reading: figure failures reached %d. Aborting the corner and returning to normal broadcast", self._figure_failures
                )
                self._disabled = True
            return False

        self._figure_failures = 0
        self._assign_cast()
        cursor = max(0, min(cursor, len(chunks)))
        display_name = fig["wiki_title"]
        self._store.open_biography_reading_session(fig["id"], display_name, len(chunks))
        self._session = BiographyReadingSession(
            fig["id"], display_name, chunks,
            cursor=cursor,
            rolling_summary=summary,
            summary_max_chars=self._bp.summary_max_chars,
        )
        self._intro_pending = cursor == 0  # 冒頭からのときだけ導入を入れる
        self._publish_state(self._session)
        logger.info(
            "biography_reading: starting \"%s\" cursor=%d/%d", display_name, cursor, len(chunks)
        )
        return True

    def _figure_by_id(self, figure_id: str) -> dict | None:
        return next((f for f in self._bp.figures if f["id"] == figure_id), None)

    def _select_figure(self) -> dict | None:
        if not self._bp.figures:
            return None
        exclude = self._store.recently_read_biography_figure_ids(self._bp.reread_after_days)
        fresh = [f for f in self._bp.figures if f["id"] not in exclude]
        pool = fresh or self._bp.figures
        return random.choice(pool)

    def _finish_session(self, session: BiographyReadingSession) -> None:
        self._store.finish_biography_reading_session(session.figure_id)
        logger.info("biography_reading: finished introducing \"%s\". Will pick a different figure next time", session.title)
        self._session = None
        self._intro_pending = False
        self._clear_state()

    # --- 補助 -----------------------------------------------------------

    def _script(self, lines: list[ScriptLine], title: str) -> Script:
        return Script(topic_id=None, topic_title=title, lines=lines, is_biography_reading=True)

    def _intro_lines(self, session: BiographyReadingSession) -> list[ScriptLine]:
        mc = self._mc
        if self._en:
            return [
                ScriptLine(mc.id, "Right. Time for the biography corner."),
                ScriptLine(
                    mc.id,
                    f"Tonight's subject is {session.title}, from the Wikipedia article.",
                ),
            ]
        return [
            ScriptLine(mc.id, "さて、ここからは偉人伝のコーナーです。"),
            ScriptLine(mc.id, f"今夜取り上げるのは、{session.title}。Wikipediaの記事をもとにお届けします。"),
        ]

    def _outro_lines(self, session: BiographyReadingSession) -> list[ScriptLine]:
        if self._en:
            return [
                ScriptLine(
                    self._mc.id,
                    f"That's as far as we go with {session.title} tonight. That was the biography corner.",
                ),
            ]
        return [
            ScriptLine(
                self._mc.id,
                f"……{session.title}については、今夜はこのあたりで。偉人伝のコーナーでした。",
            ),
        ]

    def _publish_state(self, session: BiographyReadingSession) -> None:
        self._state.reading_work_id = session.figure_id
        label = "Biography" if self._en else "偉人伝"
        self._state.reading_now_playing = f"{session.title} / {label} / Wikipedia"
        self._state.reading_progress = session.progress_label()
        # 偉人伝コーナー中は表示をMC＋その他出演者に絞る（読書コーナーと共用の仕組み）。
        self._state.reading_cast_ids = tuple(
            dict.fromkeys([self._mc.id] + [m.id for m in self._others])
        )

    def _clear_state(self) -> None:
        self._state.reading_work_id = None
        self._state.reading_now_playing = ""
        self._state.reading_progress = ""
        self._state.reading_cast_ids = ()
