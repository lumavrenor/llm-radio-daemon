"""青空文庫テキストの取得と前処理（v4 §10.2）。

朗読パートは LLM を一切通さない。青空文庫の原文テキストをここでパースし、
TTS 投入用の ``speech_text`` と画面字幕用の ``display_text`` に分けて出力する。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    """朗読の1単位。段落ベースで 300〜500 文字程度に束ねたもの（§10.1）。"""

    index: int          # 作品内の通し番号（0始まり）
    display_text: str   # 画面字幕用（漢字のまま。ルビは括弧書き）
    speech_text: str    # TTS 投入用（ruby_mode="kana" ならルビ語をかなに置換済み）
    char_offset: int    # 前処理前の原文先頭からの文字位置（進捗表示・レジューム用）
    is_chapter_head: bool = False


@dataclass
class Work:
    """パース済みの1作品。"""

    work_id: str
    title: str
    author: str
    translator: str
    chunks: list[Chunk]
