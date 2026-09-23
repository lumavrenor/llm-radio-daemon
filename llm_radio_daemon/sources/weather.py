"""お天気コーナー（``[[content]] type = "weather"``）。

天気そのものの取得と鮮度管理は :mod:`..weather`（``WeatherProvider``）が持つ。
ここはその値を1件の :class:`Topic` に仕立てるだけで、外部通信も判断もしない
—— フィラー雑談とお天気コーナーが別々に API を叩いて別々の値を喋る、という
ずれを起こさないため、取得口はプロセスにひとつだけにしてある。

``external_id`` は「地点＋日付＋時」でまとめる。天気は変化が遅いので、コーナーの
時間帯に何度 fetch() が回っても、同じ時間帯のあいだは 4.2 節・第1段階（完全一致）で
落ちて1本しか放送されない。日をまたげば・時が変われば新しいネタになる。

意味的重複排除（embedding）はこのソースには掛けない（``semantic_dedup = False``）。
「昨日の東京の天気」と「今日の東京の天気」は文面がほとんど同じなので、掛けると
2日目以降が全部消え、コーナーの時間帯が無言になる。選曲紹介（musicbrainz）を
``semantic_dedup=False`` で通しているのと同じ理由。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Iterator

from .. import language
from ..source_status import SourceStatus
from ..weather import WeatherProvider
from . import Topic

logger = logging.getLogger(__name__)

# hint / title は LLM が書いた文ではなく、こちらが台本プロンプトへ差し込む定型文。
# 日本語固定のままだと英語放送でも日本語のまま渡り、そのまま字幕へ出る。
_HINT = {
    "ja": (
        "お天気コーナーです。司会が本文の観測値をそのまま読み上げて伝え、"
        "ひな壇はその天気での過ごし方・服装・気分を各自の立場から話してください。"
        "本文に無い数値・地名・予報を足さないこと。明日以降の予報は持っていないので"
        "「この先どうなるか」を断定しないこと。"
    ),
    "en": (
        "This is the weather slot. The host reads out the observations in the copy as they "
        "are, and the rest of the panel talk about what the day is good for, what to wear, "
        "how it makes them feel. Do not add numbers, place names or forecasts that are not "
        "in the copy. There is no forecast beyond today, so do not state what happens next."
    ),
}

_TITLE = {
    "ja": "{location}の空模様（{month}月{day}日 {hour}時台）",
    # コロンやスラッシュを入れない。タイトルは LLM が書き直す前提だが、そのまま
    # 引き写されたときに TTS が読めない表記だけは残さない（language.py 参照）。
    "en": "The sky over {location} (month {month}, day {day}, the {hour} o'clock hour)",
}


class WeatherSource:
    name = "weather"

    # 意味的重複排除を外す（モジュール docstring 参照）。SourceThread が見る。
    semantic_dedup = False

    def __init__(
        self,
        provider: WeatherProvider,
        poll_interval_sec: float = 600.0,
        stop_event: threading.Event | None = None,
        status: SourceStatus | None = None,
    ):
        self._provider = provider
        self._poll_interval_sec = max(poll_interval_sec, 60.0)
        self._stop_event = stop_event or threading.Event()
        self._status = status

    def fetch(self) -> Iterator[Topic]:
        """今の天気を1件ネタ化しては poll_interval 待つ。時間帯ゲートは SourceThread が担当。"""
        while not self._stop_event.is_set():
            topic = self._to_topic()
            if topic is not None:
                yield topic

            if self._status is not None:
                self._status.cycle_end(1 if topic is not None else 0, self._poll_interval_sec)
            if self._stop_event.wait(self._poll_interval_sec):
                return

    def _to_topic(self) -> Topic | None:
        weather = self._provider.current()
        if weather is None:
            # 取れていない／古すぎる。天気を推測で喋らせるくらいならネタを出さず、
            # この時間帯はフィラーに任せる（weather.py の方針と同じ）。
            logger.info("weather corner: no usable observation; skipping this cycle")
            return None

        now = time.localtime()
        lang = language.current()
        return Topic(
            source=self.name,
            external_id=(
                f"weather:{weather.location}:{time.strftime('%Y-%m-%d %H', now)}"
            ),
            title=_TITLE.get(lang, _TITLE["ja"]).format(
                location=weather.location,
                month=now.tm_mon, day=now.tm_mday, hour=now.tm_hour,
            ),
            body="\n".join(weather.detail_lines),
            url=None,
            hint=_HINT.get(lang, _HINT["ja"]),
        )
