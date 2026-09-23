"""1作品の翻訳朗読進行状態。

literary_reading/session.py と同じ考え方（既存の ``Source`` プロトコルは流用しない）。
違いは「原文チャンクを1つずつ順にLLMへ渡して訳す」点（biography_reading と同じ）。
再起動をまたいだ永続化は ``db.TopicStore`` が担当するが、直近に訳した日本語文
（Beat の材料）だけは持ち越さない ―― 復帰直後にBeatの間合いが1回ぶん軽くなる
程度の劣化で済み、コーナーを止めるよりましなため（§10.8 と同じ判断）。
"""

from __future__ import annotations

import logging

from ..aozora import Chunk

logger = logging.getLogger(__name__)

_MIN_SANE_SUMMARY = 20  # これ未満なら要約が壊れたとみなす


class TranslatedReadingSession:
    def __init__(
        self,
        work_id: str,
        title: str,
        author: str,
        chunks: list[Chunk],
        *,
        cursor: int = 0,
        rolling_summary: str = "",
        beat_interval: int = 4,
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
        self._recent_ja: list[str] = []

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    def is_finished(self) -> bool:
        return self.cursor >= len(self.chunks)

    def next_chunk(self) -> Chunk | None:
        """次に訳す原文チャンク。cursor はここでは進めない（生成成功後に呼び出し側が進める）。"""
        if self.cursor >= len(self.chunks):
            return None
        return self.chunks[self.cursor]

    def is_beat_due(self) -> bool:
        """今の cursor で Beat（感想パート）を挟むタイミングか。

        literary_reading の peek_batch と違い、ここでは cursor は「訳し終えた
        チャンク数」そのものなので、割り切れる／読了のときが合図になる。
        """
        if not self.cursor:
            return False
        return self.cursor % self.beat_interval == 0 or self.is_finished()

    def progress_label(self) -> str:
        """「42%」相当。章番号は is_chapter_head の数から概算する（literary_reading と同じ）。"""
        if not self.chunks:
            return ""
        pct = int(100 * self.cursor / len(self.chunks))
        chapter = sum(1 for c in self.chunks[: self.cursor] if c.is_chapter_head)
        return f"第{chapter}章 {pct}%" if chapter else f"{pct}%"

    def record_translation(self, ja_text: str) -> None:
        """Beat で使う「直近に訳した日本語文」を積む（beat_interval 件で頭打ち）。"""
        text = ja_text.strip()
        if not text:
            return
        self._recent_ja.append(text)
        if len(self._recent_ja) > self.beat_interval:
            self._recent_ja = self._recent_ja[-self.beat_interval :]

    def recent_translated_text(self) -> str:
        return "\n".join(self._recent_ja)

    def apply_summary(self, new_summary: str | None) -> None:
        """要約を上書きする。空・極端に短い・None のときは前回の要約を保持する。"""
        if not new_summary:
            logger.warning("translated_reading: summary update is empty. Keeping previous summary")
            return
        cleaned = new_summary.strip()
        if len(cleaned) < _MIN_SANE_SUMMARY:
            logger.warning("translated_reading: summary update too short (%d chars). Keeping previous", len(cleaned))
            return
        self.rolling_summary = cleaned[: self.summary_max_chars]
