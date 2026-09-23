"""embeddingによる類似トピック重複排除（v3）。SPEC.md 4.2 第2段階。

external_id が完全一致しない限り通過してしまう第1段階（db.pyのUNIQUE制約）だけでは、
「表現が違うだけの同じ話」（似た論文が続けて出た、同じ事件を別記事が報じた等）を
弾けない。ここでは title+body を埋め込みモデル（既定 nomic-embed-text）でベクトル化し、
直近N件とのコサイン類似度が閾値を超えたら「もう話した内容とほぼ同じ」として破棄する。

埋め込み取得はOllama常駐プロセスへのHTTP呼び出しであり、失敗しても
「無音を作らない」方針を優先し、重複ではない（＝通す）ものとして扱う。
"""

from __future__ import annotations

import logging

import numpy as np

from . import llm_http
from .config import EmbeddingConfig
from .db import TopicStore

logger = logging.getLogger(__name__)


def check_embedding(config: EmbeddingConfig) -> bool:
    """起動時の疎通確認。実際に1回だけ埋め込みを取り、成否を分かりやすくログへ出す。

    失敗しても放送は止めない（重複排除が働かないだけ）。詳しいエラー内容は
    ``llm_http.embed`` 側の warning に出るので、ここでは「つながったか」と
    「つながらないと何が起きるか」だけを人間向けに記録する。
    """
    vec = llm_http.embed(config, "接続確認テスト")
    if vec:
        logger.info(
            "connected to embedding (%s, engine=%s, model=%s, %d dims)",
            config.host, config.engine, config.model, len(vec),
        )
        return True
    logger.warning(
        "could not connect to embedding (%s, engine=%s, model=%s). "
        "Similar-topic deduplication won't work, so the same story may repeat with different wording "
        "(the broadcast itself will continue).",
        config.host, config.engine, config.model,
    )
    return False


def _embed(text: str, config: EmbeddingConfig) -> np.ndarray | None:
    vec = llm_http.embed(config, text)
    if not vec:
        return None
    return np.asarray(vec, dtype=np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class EmbeddingDeduplicator:
    def __init__(self, store: TopicStore, config: EmbeddingConfig):
        self._store = store
        self._config = config

    def is_duplicate(self, topic_id: int, title: str, body: str | None) -> bool:
        """埋め込みを計算してDBに保存し、直近トピックと似すぎていれば True を返す。

        埋め込み取得に失敗した場合は破棄せず False を返す（放送を止めないため）。
        """
        text = f"{title}\n{body or ''}"[:4000]
        vec = _embed(text, self._config)
        if vec is None:
            return False
        self._store.save_embedding(topic_id, vec)

        recent = self._store.recent_embeddings(exclude_id=topic_id, limit=self._config.max_recent)
        for _, other in recent:
            if _cosine_similarity(vec, other) > self._config.similarity_threshold:
                return True
        return False
