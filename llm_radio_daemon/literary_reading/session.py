"""1作品の読書進行状態（v4 §10.6）。

既存の ``Source`` プロトコル（1件返して終わり）は流用しない。位置と要約という
連続した状態を持つ別クラスにする。再起動をまたいだ永続化は ``db.TopicStore`` が担当。
"""

from __future__ import annotations

import logging

from .. import language
from ..aozora import Chunk

logger = logging.getLogger(__name__)

_MIN_SANE_SUMMARY = 20  # これ未満なら要約が壊れたとみなす（§10.3）


class LiteraryReadingSession:
    def __init__(
        self,
        work_id: str,
        title: str,
        author: str,
        chunks: list[Chunk],
        *,
        cursor: int = 0,
        rolling_summary: str = "",
        beat_interval: int = 3,
        summary_max_chars: int = 800,
    ):
        self.work_id = work_id
        self.title = title
        self.author = author
        self.chunks = chunks
        self.cursor = max(0, min(cursor, len(chunks)))
        self.rolling_summary = rolling_summary
        self.beat_interval = max(1, beat_interval)
        self.summary_max_chars = summary_max_chars
        self._beats_done = self.cursor // self.beat_interval

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    def is_finished(self) -> bool:
        return self.cursor >= len(self.chunks)

    def peek_batch(self) -> tuple[list[Chunk], bool]:
        """次に朗読するチャンク列と、その後に Beat（感想生成）を挟むかを返す。

        cursor はここでは進めない。実際に再生が始まった時点で
        ``TopicStore.advance_reading_cursor`` により前進する（§10.6）。
        章題（is_chapter_head）の直前でバッチを切る。
        """
        start = self.cursor
        if start >= len(self.chunks):
            return [], False

        batch: list[Chunk] = []
        i = start
        while i < len(self.chunks) and len(batch) < self.beat_interval:
            ch = self.chunks[i]
            if batch and ch.is_chapter_head:
                break  # 章題の直前で切る
            batch.append(ch)
            i += 1

        reached_end = i >= len(self.chunks)
        is_beat = (len(batch) == self.beat_interval) or (reached_end and bool(batch))
        return batch, is_beat

    def progress_label(self) -> str:
        """「42%」相当。章番号は is_chapter_head の数から概算する。"""
        if not self.chunks:
            return ""
        pct = int(100 * self.cursor / len(self.chunks))
        chapter = sum(1 for c in self.chunks[: self.cursor] if c.is_chapter_head)
        if chapter:
            return language.pick(ja=f"第{chapter}章 {pct}%", en=f"ch. {chapter} · {pct}%")
        return f"{pct}%"

    def recent_chunks(self, n: int) -> list[Chunk]:
        end = self.cursor
        return self.chunks[max(0, end - n) : end]

    def apply_summary(self, new_summary: str | None) -> None:
        """要約を上書きする。空・極端に短い・None のときは前回の要約を保持（§10.3）。"""
        if not new_summary:
            logger.warning("reading: summary update is empty. Keeping previous summary")
            return
        cleaned = new_summary.strip()
        if len(cleaned) < _MIN_SANE_SUMMARY:
            logger.warning("reading: summary update too short (%d chars). Keeping previous", len(cleaned))
            return
        self.rolling_summary = cleaned[: self.summary_max_chars]
