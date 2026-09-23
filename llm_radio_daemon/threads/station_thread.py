"""アクティブなコンテンツに合わせてネットラジオ局を切り替える。

[[content]] の ``stream = ["id", ...]`` に書かれた [[streams]] の id から 1 局を
ランダムに選び、MusicThread / IcyMetadataReader の接続先を差し替える。``stream``
未指定のコーナーでは既定局（[[streams]] の先頭）に戻す。抽選するのはコーナーが
切り替わった瞬間だけで、同じコーナーが続いている間は局を変えない。

選んだ局が ``fallback_after_sec`` のあいだ音を返さなければ、同じ候補の中の別局
（尽きたら既定局）へ落とす。局ごとにフォールバック先を書かせるのは設定が複雑に
なるわりに使いどころが無いので、「候補の中で融通する」だけに留めている。
"""

from __future__ import annotations

import logging
import random
import threading
from typing import Callable

from ..config import Config, ContentConfig, StreamConfig
from .. import schedule

logger = logging.getLogger(__name__)


class StationHolder:
    """いま流している局と、それを鳴らしているスレッド。

    ウォッチドッグが MusicThread / IcyMetadataReader を作り直しても局が起動時の
    値に戻らないよう、「今どこに繋ぐか」はスレッドではなくここが持つ。ファクトリは
    ``stream.url`` を見て新しいスレッドを作り、自分をここへ登録する。
    """

    def __init__(self, stream: StreamConfig, tuned_content: ContentConfig | None = None):
        self.stream = stream
        self.music = None  # MusicThread | None
        self.icy = None    # IcyMetadataReader | None
        # どのコーナー向けに局を選び終えたか。番組進行が切り替えの準備完了を判定するのに使う。
        self.tuned_content = tuned_content


class StationThread(threading.Thread):
    def __init__(
        self,
        config: Config,
        holder: StationHolder,
        initial_content: ContentConfig | None = None,
        on_change: Callable[[StreamConfig], None] | None = None,
        # コーナー切り替えの準備（暗転中）はこの局の切り替えを待つので、短めに見る。
        check_interval_sec: float = 1.0,
        fallback_after_sec: float = 60.0,
        stop_event: threading.Event | None = None,
    ):
        super().__init__(name="StationThread", daemon=True)
        self._config = config
        self._holder = holder
        self._on_change = on_change
        self._check_interval_sec = check_interval_sec
        self._fallback_after_sec = fallback_after_sec
        self._stop_event = stop_event or threading.Event()
        # 起動時の局は main が initial_content から抽選済み。ここで選び直すと
        # 起動 5 秒後に無意味な繋ぎ直しが入るので、選んだ状態から始める。
        self._current_content = initial_content
        self._candidates: list[StreamConfig] = config.streams_for(initial_content)
        self._tried: list[str] = [holder.stream.id]

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.wait(self._check_interval_sec):
            try:
                self._tick()
            except Exception:
                logger.exception("station switch failed")

    def _tick(self) -> None:
        content = schedule.active_content(self._config.content)
        if content is not self._current_content:
            self._current_content = content
            self._candidates = self._config.streams_for(content)
            self._tried = []
            self._switch_to(random.choice(self._candidates), "コーナー切替")
            self._holder.tuned_content = content
            return
        self._maybe_fallback()

    def _maybe_fallback(self) -> None:
        """選んだ局が黙ったままなら候補の別局へ。候補が1つなら再接続に任せる。"""
        music = self._holder.music
        if music is None or music.seconds_since_data() < self._fallback_after_sec:
            return
        rest = [s for s in self._candidates if s.id not in self._tried]
        if not rest:
            default = self._config.primary_stream
            if default.id in self._tried:
                return  # 打つ手なし。MusicThread の再接続に任せる
            rest = [default]
        self._switch_to(
            random.choice(rest),
            f"{self._holder.stream.label} が {self._fallback_after_sec:.0f}秒 無音",
        )

    def _switch_to(self, stream: StreamConfig, reason: str) -> None:
        if stream.id not in self._tried:
            self._tried.append(stream.id)
        holder = self._holder
        if holder.stream.id == stream.id and holder.music is not None:
            return  # 同じ局。無駄に繋ぎ直して音を切らさない
        holder.stream = stream
        if holder.music is not None:
            holder.music.set_url(stream.url)
        if holder.icy is not None:
            holder.icy.set_url(stream.url)
        logger.info("station -> %s（%s）", stream.label, reason)
        if self._on_change is not None:
            self._on_change(stream)
