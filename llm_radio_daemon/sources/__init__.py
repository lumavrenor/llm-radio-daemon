"""ネタ源の共通インターフェース。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol

USER_AGENT = "llm-radio-daemon/0.1 (https://github.com/lumavrenor/llm-radio-daemon; personal 24h local-LLM radio project)"


@dataclass
class Topic:
    source: str          # "wikimedia" | "hackernews" | "arxiv" | "sports" | "nowplaying" | ...
    external_id: str     # 重複排除キー（URL または一意ID）
    title: str
    body: str            # LLMに渡す本文（最大2000文字程度に切る）
    url: str | None
    hint: str            # LLMへの切り口ヒント


class Source(Protocol):
    name: str

    # 意味的重複排除（embedding 類似度）を掛けるか。既定は掛ける。
    # 天気のように「内容がほぼ毎回同じでも、そのつど改めて成立する」ネタだけ
    # False を宣言する（宣言しないソースは SourceThread が True 扱いにする）。
    semantic_dedup: bool

    def fetch(self) -> Iterator[Topic]: ...
