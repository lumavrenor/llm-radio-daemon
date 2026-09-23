"""番組進行（director.py の状態機械）を一定間隔で回すスレッド。

状態はすべて ProgramDirector 側が持つので、ウォッチドッグに作り直されても
切り替えの途中から続きを進められる。
"""

from __future__ import annotations

import logging
import threading

from ..director import ProgramDirector

logger = logging.getLogger(__name__)

_TICK_SEC = 0.1


class DirectorThread(threading.Thread):
    def __init__(self, director: ProgramDirector, stop_event: threading.Event | None = None):
        super().__init__(name="DirectorThread", daemon=True)
        self._director = director
        self._stop_event = stop_event or threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.wait(_TICK_SEC):
            try:
                self._director.tick()
            except Exception:
                logger.exception("director tick crashed; continuing")
