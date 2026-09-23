"""Wikipedia記事本文の取得と前処理（偉人伝トーク・type = "biography_reading"）。

朗読コーナー（aozora）と違い、ここで得た原文はTTSへ直行しない。MCの解説トーク自体を
LLMが生成するための素材（コンテキスト）として使う。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SourceChunk:
    """偉人伝トークの1区切り。Wikipedia本文を段落ベースで束ねたもの（LLMへ渡す生素材）。"""

    index: int          # 記事内の通し番号（0始まり）
    text: str            # 原文（プレーンテキスト）
    section: str          # 直近の見出し名（冒頭リード文なら空文字）
    is_section_head: bool = False  # そのセクションに入って最初のチャンクか
