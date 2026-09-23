"""ワーカースレッドの生存監視。死んでいたら再起動する。

「止まらないこと」が最優先という設計方針（8. 非機能要件）の最後の砦。
Pythonのthreading.Threadは一度runが終わると再開できないため、
再起動は「同じ構成で新しいインスタンスを作って差し替える」形で行う。
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

logger = logging.getLogger(__name__)


class WatchdogThread(threading.Thread):
    def __init__(
        self,
        factories: dict[str, Callable[[], threading.Thread]],
        check_interval_sec: float = 5.0,
        stop_event: threading.Event | None = None,
    ):
        super().__init__(name="WatchdogThread", daemon=True)
        self._factories = factories
        self._threads: dict[str, threading.Thread] = {}
        self._check_interval_sec = check_interval_sec
        self._stop_event = stop_event or threading.Event()

    def register(self, name: str, thread: threading.Thread) -> None:
        self._threads[name] = thread

    def stop(self) -> None:
        self._stop_event.set()

    def stop_all(self) -> None:
        """自分自身に加え、現在監視下にある全ワーカースレッドの停止も指示する。"""
        self.stop()
        for thread in list(self._threads.values()):
            stop_fn = getattr(thread, "stop", None)
            if callable(stop_fn):
                try:
                    stop_fn()
                except Exception:
                    logger.exception("failed to stop %s cleanly", thread.name)

    def run(self) -> None:
        while not self._stop_event.wait(self._check_interval_sec):
            for name, factory in self._factories.items():
                thread = self._threads.get(name)
                if thread is not None and thread.is_alive():
                    continue
                logger.error("%s is not running; restarting", name)
                try:
                    new_thread = factory()
                    new_thread.start()
                    self._threads[name] = new_thread
                except Exception:
                    logger.exception("failed to restart %s", name)
