"""1人物ぶんの偉人伝トーク進行状態。

既存の ``Source`` プロトコル（1件返して終わり）は流用しない。位置と要約という
連続した状態を持つ別クラスにする（literary_reading/session.py と同じ考え方）。
再起動をまたいだ永続化は ``db.TopicStore`` が担当。
"""

from __future__ import annotations

import logging

from ..wiki_bio import SourceChunk

logger = logging.getLogger(__name__)

_MIN_SANE_SUMMARY = 20  # これ未満なら要約が壊れたとみなす


class BiographyReadingSession:
    def __init__(
        self,
        figure_id: str,
        title: str,
        chunks: list[SourceChunk],
        *,
        cursor: int = 0,
        rolling_summary: str = "",
        summary_max_chars: int = 800,
    ):
        self.figure_id = figure_id
        self.title = title
        self.chunks = chunks
        self.cursor = max(0, min(cursor, len(chunks)))
        self.rolling_summary = rolling_summary
        self.summary_max_chars = summary_max_chars

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    def is_finished(self) -> bool:
        return self.cursor >= len(self.chunks)

    def next_chunk(self) -> SourceChunk | None:
        """次にLLMへ渡すチャンク。cursor はここでは進めない（呼び出し側が生成成功後に進める）。"""
        if self.cursor >= len(self.chunks):
            return None
        return self.chunks[self.cursor]

    def progress_label(self) -> str:
        """「生涯 42%」相当。節名は直近に通過したものを使う。"""
        if not self.chunks:
            return ""
        pct = int(100 * self.cursor / len(self.chunks))
        section = next(
            (c.section for c in reversed(self.chunks[: self.cursor]) if c.section), ""
        )
        return f"{section} {pct}%" if section else f"{pct}%"

    def apply_summary(self, new_summary: str | None) -> None:
        """要約を上書きする。空・極端に短い・None のときは前回の要約を保持する。"""
        if not new_summary:
            logger.warning("biography_reading: summary update is empty. Keeping previous summary")
            return
        cleaned = new_summary.strip()
        if len(cleaned) < _MIN_SANE_SUMMARY:
            logger.warning("biography_reading: summary update too short (%d chars). Keeping previous", len(cleaned))
            return
        self.rolling_summary = cleaned[: self.summary_max_chars]
