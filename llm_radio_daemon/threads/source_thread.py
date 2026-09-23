"""ネタ収集スレッド。ソースを1つ回して重複排除後 topic_queue に投入する。

v3ではソースが複数（Wikimedia/Hacker News/arXiv/RSS）になるため、
SourceThread はどのソースを回すか(source引数)を受け取る汎用ワーカーにし、
main.py 側でソースごとに1つずつインスタンス化する。

例外は握りつぶして次のイテレーションへ進む（24時間止めない）。
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable

from ..db import TopicStore
from ..dedup import EmbeddingDeduplicator
from ..source_status import DUPLICATE, KNOWN, NEW, SourceStatus
from ..sources import Source, Topic

logger = logging.getLogger(__name__)


def enqueue_topic(
    topic: Topic,
    store: TopicStore,
    topic_queue: "queue.Queue[tuple[int, Topic]]",
    stop_event: threading.Event,
    deduper: EmbeddingDeduplicator | None = None,
    semantic_dedup: bool = True,
) -> str:
    """重複排除してキューへ投入する共通処理。SourceThreadとMusicBrainz連携の両方から使う。

    ``semantic_dedup=False`` で embedding による意味的重複排除だけを外す
    （external_id の完全一致チェックは常に効く）。同じ曲がまた流れたときの選曲紹介の
    ように、「内容が前と同じでも改めて成立する」ネタのための逃げ道。

    戻り値は :mod:`..source_status` の ``NEW`` / ``KNOWN`` / ``DUPLICATE``。
    捨てた理由を呼び出し側（SourceStatus）が数えられるようにするためのもの。
    """
    topic_id = store.try_insert_topic(
        topic.source, topic.external_id, topic.title, topic.body, topic.url
    )
    if topic_id is None:
        return KNOWN  # 既知のトピック（完全一致）。捨てる

    if deduper is not None and semantic_dedup:
        try:
            if deduper.is_duplicate(topic_id, topic.title, topic.body):
                logger.info("topic %r discarded: too similar to a recent topic (embedding)", topic.title)
                return DUPLICATE
        except Exception:
            logger.exception("embedding dedup crashed for topic %r; keeping topic", topic.title)

    while not stop_event.is_set():
        try:
            topic_queue.put((topic_id, topic), timeout=1.0)
            return NEW
        except queue.Full:
            continue
    return KNOWN  # 停止要求で投入できずに抜けた。新着としては数えない


class SourceThread(threading.Thread):
    def __init__(
        self,
        source: Source,
        topic_queue: "queue.Queue[tuple[int, Topic]]",
        store: TopicStore,
        stop_event: threading.Event | None = None,
        deduper: EmbeddingDeduplicator | None = None,
        is_active: Callable[[], bool] | None = None,
        # 出番待ちの確認間隔。判定は属性を読むだけなので短くてよく、長いとコーナー
        # 切り替えの準備中（暗転中）にネタ集めを始めるのが遅れる。
        idle_check_sec: float = 1.0,
        status: SourceStatus | None = None,
    ):
        super().__init__(name=f"SourceThread:{source.name}", daemon=True)
        self._source = source
        self._topic_queue = topic_queue
        self._store = store
        self._stop_event = stop_event or threading.Event()
        self._deduper = deduper
        # 番組表型（排他）：このソースの type が今アクティブなときだけネタを収集する。
        # None なら常時アクティブ扱い（テスト・単体利用向け）。
        self._is_active = is_active or (lambda: True)
        self._idle_check_sec = idle_check_sec
        # 1周ぶんの集計。ソース側が cycle_end を呼んでログ・画面表示に出す。
        self._status = status

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        backoff = 1.0
        while not self._stop_event.is_set():
            if not self._is_active():
                if self._stop_event.wait(self._idle_check_sec):
                    return
                continue
            try:
                for topic in self._source.fetch():
                    if self._stop_event.is_set():
                        return
                    if not self._is_active():
                        # 時間帯を抜けた。ジェネレータを畳んで待機に戻る。
                        # 周の途中なので集計はここで捨てる（cycle_end は来ない）。
                        if self._status is not None:
                            self._status.idle()
                        break
                    outcome = enqueue_topic(
                        topic, self._store, self._topic_queue, self._stop_event, self._deduper,
                        # 天気のように「毎回ほぼ同じ文面でも、そのつど改めて成立する」
                        # ソースは意味的重複排除を外す（宣言が無ければ従来どおり掛ける）。
                        semantic_dedup=getattr(self._source, "semantic_dedup", True),
                    )
                    if self._status is not None:
                        self._status.record(outcome)
                    backoff = 1.0
            except Exception:
                logger.exception("source %s failed; will retry", self._source.name)
                if self._status is not None:
                    self._status.crashed()
            if self._stop_event.is_set():
                return
            if self._stop_event.wait(backoff):
                return
            backoff = min(backoff * 2, 60.0)
