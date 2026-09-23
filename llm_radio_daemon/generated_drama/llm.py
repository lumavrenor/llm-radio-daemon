"""執筆バッチ用の LLM 呼び出し（v6 §4.7.1）。

**放送プロセスからは呼ばれない。** 別プロセスの ``generated_drama_writer`` からだけ使う。
1回の生成が長い（数千字）ため、放送側（``script/ollama_client.py``）より
タイムアウトを長く取り、structured output で JSON を受け取る。
実際の HTTP は engine を見て llm_http が振り分ける（ollama / lmstudio 等）。
"""

from __future__ import annotations

import json
import logging

import requests

from .. import llm_http
from ..config import LLMConfig

logger = logging.getLogger(__name__)


def chat_json(
    llm_config: LLMConfig,
    prompt: str,
    schema: dict,
    *,
    timeout_sec: int = 600,
    temperature: float | None = None,
    attempts: int = 2,
    max_tokens: int = 4096,
) -> dict | None:
    """プロンプトを投げて JSON を1つ受け取る。失敗したら None。

    ``max_tokens``（num_predict）は暴走対策。上限を切らないと、モデルが同じ指摘を
    延々と並べ続けて数万字の JSON を吐き、途中で切れてパースできないことがある。
    """
    for attempt in range(max(1, attempts)):
        try:
            raw = llm_http.chat(
                llm_config, prompt, schema=schema,
                temperature=(
                    llm_config.temperature if temperature is None else temperature
                ),
                num_predict=max_tokens,
                timeout_sec=timeout_sec,
            )
            return json.loads(raw)
        except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError) as e:
            logger.warning(
                "generated_drama_writer: generation failed (%d/%d): %s", attempt + 1, max(1, attempts), e
            )
    return None
