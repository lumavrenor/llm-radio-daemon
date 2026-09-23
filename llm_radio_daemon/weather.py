"""現在の天気（config.toml の ``[weather]``）。プロンプトへ差し込む「状況」を1つ取ってくる。

なぜ ``sources/`` ではなくここに置くのか
----------------------------------------
天気は「ネタ」ではなく「文脈」だから。フィラー雑談とお天気コーナーの両方が同じ値を
見るし、内容は1時間経ってもほとんど変わらない。これを毎回 :class:`~.sources.Topic`
として流すと 4.2 節の重複排除（embedding 類似度）に片端から消され、しかも消える
たびに埋め込みの呼び出しだけが走る。取得はここに1つ置き、コーナー側
（``sources/weather.py``）は「その値を Topic に仕立てるだけ」にする。

場所はどこから来るか
--------------------
``[weather]`` に緯度経度を明示する。IP ジオロケーションも OS の位置情報 API も
使わない —— この番組にとって天気は「リスナーの現在地」ではなく
**「スタジオの所在地」** という番組設定であり、勝手に推測する筋合いのものではない。
書かれていなければ ``[weather] enabled = false`` と同じ＝天気には一切触れない。

これは ``[llm] placement`` と同じ流儀（filler.py の ``_PLACEMENT_FACTS`` 参照）。
取得に失敗したとき・値が古すぎるときに :meth:`WeatherProvider.current` が ``None``
を返すのも同じ理由で、「たぶん晴れ」を喋らせるより黙らせるほうが事故が小さい。

取得先は Open-Meteo（https://open-meteo.com/ ）。API キーが要らず、緯度経度だけで
現在の天気・気温・当日の最高最低・降水確率が1リクエストで揃い、日本以外でも同じ
コードで動く（気象庁の予報 JSON は地域コード制で日本専用になる）。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import requests

from . import language  # 属性参照で使う（起動時に差し替わる）
from .sources import USER_AGENT

logger = logging.getLogger(__name__)

DEFAULT_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# 取得に失敗してから次に試すまで。放送を止めないための最低限のバックオフ。
_RETRY_AFTER_SEC = 120.0

# キャッシュをここまで古くなったら捨てる（= 天気に触れさせない）。ネットが落ちた
# まま半日流し続けると「今の天気」が朝の値のままになるので、refresh_sec とは
# 別に絶対の上限を置く。
_STALE_LIMIT_SEC = 3 * 3600.0

# WMO weather code → (日本語, English, 天候タグ)。
# https://open-meteo.com/en/docs のコード表。番組で喋る粒度に丸めてある
# （「弱い雨」「並の雨」を分けても台本では使いようがないため）。
_WMO_CODES: dict[int, tuple[str, str, str]] = {
    0: ("快晴", "clear sky", "clear"),
    1: ("晴れ", "mainly clear", "clear"),
    2: ("薄曇り", "partly cloudy", "cloudy"),
    3: ("曇り", "overcast", "cloudy"),
    45: ("霧", "fog", "cloudy"),
    48: ("霧（霧氷）", "freezing fog", "cloudy"),
    51: ("霧雨", "light drizzle", "rain"),
    53: ("霧雨", "drizzle", "rain"),
    55: ("強い霧雨", "dense drizzle", "rain"),
    56: ("凍える霧雨", "freezing drizzle", "rain"),
    57: ("凍える霧雨", "dense freezing drizzle", "rain"),
    61: ("小雨", "light rain", "rain"),
    63: ("雨", "rain", "rain"),
    65: ("強い雨", "heavy rain", "rain"),
    66: ("凍雨", "freezing rain", "rain"),
    67: ("強い凍雨", "heavy freezing rain", "rain"),
    71: ("小雪", "light snow", "snow"),
    73: ("雪", "snow", "snow"),
    75: ("大雪", "heavy snow", "snow"),
    77: ("霧雪", "snow grains", "snow"),
    80: ("にわか雨", "light rain showers", "rain"),
    81: ("にわか雨", "rain showers", "rain"),
    82: ("激しいにわか雨", "violent rain showers", "rain"),
    85: ("にわか雪", "light snow showers", "snow"),
    86: ("にわか雪", "heavy snow showers", "snow"),
    95: ("雷雨", "thunderstorm", "storm"),
    96: ("ひょうをともなう雷雨", "thunderstorm with hail", "storm"),
    99: ("ひょうをともなう激しい雷雨", "severe thunderstorm with hail", "storm"),
}

_UNKNOWN_CODE = ("天気は不明", "unknown", "cloudy")


def describe_code(code: int, lang: str = "ja") -> tuple[str, str]:
    """WMO weather code → (読み上げ用の天気名, 天候タグ)。未知のコードは曇り扱い。"""
    ja, en, tag = _WMO_CODES.get(code, _UNKNOWN_CODE)
    return (en if (lang or "").strip().lower() == "en" else ja), tag


def _round(value: float | None) -> int | None:
    return None if value is None else int(round(value))


@dataclass(frozen=True)
class Weather:
    """ある時点の天気1件。プロンプトへ渡すのはこの中身だけで、推測は足させない。"""

    location: str          # トークで呼ぶ地名（[weather] location_name）
    description: str       # 「晴れ」「にわか雨」など、読み上げ用の天気名
    condition: str         # 天候タグ（clear / cloudy / rain / snow / storm）
    temp_c: int | None
    feels_like_c: int | None
    high_c: int | None     # 当日の予想最高気温
    low_c: int | None      # 当日の予想最低気温
    precip_prob: int | None  # 当日の降水確率（％）
    wind_kph: int | None
    fetched_at: float      # time.time()

    @property
    def fact_line(self) -> str:
        """フィラーのプロンプトへ1行で渡す用。書いていない数字は喋らせない。

        ラベルは ``[locale] lang`` ぶんを出す。ここが日本語固定だと、英語版の
        フィラープロンプトに日本語のお天気が1行だけ混ざる（description 自体は
        describe_code() が lang を見て英語で入っている）。
        """
        if language.current() == "en":
            parts = [f"{self.location} is currently {self.description}"]
            if self.temp_c is not None:
                felt = ""
                if self.feels_like_c is not None and self.feels_like_c != self.temp_c:
                    felt = f" (feels like {self.feels_like_c})"
                parts.append(f"{self.temp_c} degrees{felt}")
            if self.high_c is not None and self.low_c is not None:
                parts.append(f"today's high {self.high_c} and low {self.low_c}")
            if self.precip_prob is not None:
                parts.append(f"{self.precip_prob} percent chance of rain")
            return ", ".join(parts) + "."
        parts = [f"{self.location}は今「{self.description}」"]
        if self.temp_c is not None:
            felt = ""
            if self.feels_like_c is not None and self.feels_like_c != self.temp_c:
                felt = f"・体感{self.feels_like_c}度"
            parts.append(f"気温{self.temp_c}度{felt}")
        if self.high_c is not None and self.low_c is not None:
            parts.append(f"今日の予想最高{self.high_c}度／最低{self.low_c}度")
        if self.precip_prob is not None:
            parts.append(f"降水確率{self.precip_prob}パーセント")
        return "、".join(parts) + "。"

    @property
    def detail_lines(self) -> list[str]:
        """お天気コーナーの本文用。1行1項目で、ここに無い数字は台本にも出させない。

        fact_line と同じ理由で ``[locale] lang`` ぶんのラベルを出す。単位語
        （度・パーセント）まで綴ってあるのは、Kokoro が記号を読めないため
        （language.py 参照）—— "12C" ではなく "12 degrees" と書いておく。
        """
        if language.current() == "en":
            out = [f"Location: {self.location}", f"Conditions: {self.description}"]
            if self.temp_c is not None:
                out.append(f"Temperature: {self.temp_c} degrees")
            if self.feels_like_c is not None:
                out.append(f"Feels like: {self.feels_like_c} degrees")
            if self.high_c is not None:
                out.append(f"High today: {self.high_c} degrees")
            if self.low_c is not None:
                out.append(f"Low today: {self.low_c} degrees")
            if self.precip_prob is not None:
                out.append(f"Chance of rain today: {self.precip_prob} percent")
            if self.wind_kph is not None:
                out.append(f"Wind: {self.wind_kph} kilometres per hour")
            return out
        out = [f"観測地点: {self.location}", f"今の天気: {self.description}"]
        if self.temp_c is not None:
            out.append(f"気温: {self.temp_c}度")
        if self.feels_like_c is not None:
            out.append(f"体感温度: {self.feels_like_c}度")
        if self.high_c is not None:
            out.append(f"今日の予想最高気温: {self.high_c}度")
        if self.low_c is not None:
            out.append(f"今日の予想最低気温: {self.low_c}度")
        if self.precip_prob is not None:
            out.append(f"今日の降水確率: {self.precip_prob}パーセント")
        if self.wind_kph is not None:
            out.append(f"風速: 時速{self.wind_kph}キロ")
        return out


class WeatherProvider:
    """Open-Meteo から現在の天気を取ってきて TTL つきで持っておく。

    フィラースレッドとお天気コーナー（SourceThread）の両方から呼ばれるので、
    取得はロックで直列化する。呼ぶ側は常に :meth:`current` だけを見ればよく、
    ``None`` は「今回は天気に触れない」を意味する。

    専用スレッドは作らない。天気は「聞かれたときに答えられればよい」情報で、
    誰も見ていない間まで定期的に外へ出ていく必要がないため。
    """

    def __init__(
        self,
        latitude: float,
        longitude: float,
        location_name: str,
        *,
        lang: str = "ja",
        refresh_sec: float = 1800.0,
        timeout_sec: float = 5.0,
        url: str = DEFAULT_FORECAST_URL,
    ):
        self._latitude = latitude
        self._longitude = longitude
        self._location_name = location_name
        self._lang = lang
        self._refresh_sec = max(float(refresh_sec), 60.0)
        self._timeout_sec = float(timeout_sec)
        self._url = url
        self._lock = threading.Lock()
        self._cache: Weather | None = None
        self._last_attempt = 0.0

    @property
    def location_name(self) -> str:
        return self._location_name

    def current(self) -> Weather | None:
        """今の天気。取れていない・古すぎるときは ``None``（＝天気に触れさせない）。"""
        with self._lock:
            now = time.time()
            fresh = self._cache is not None and now - self._cache.fetched_at < self._refresh_sec
            if not fresh and now - self._last_attempt >= _RETRY_AFTER_SEC:
                self._last_attempt = now
                got = self._fetch()
                if got is not None:
                    self._cache = got
            if self._cache is None:
                return None
            if now - self._cache.fetched_at > _STALE_LIMIT_SEC:
                # ネットが落ちたまま流し続けている。朝の値で「今の天気」を
                # 語らせるくらいなら黙らせる。
                return None
            return self._cache

    def _fetch(self) -> Weather | None:
        params = {
            "latitude": self._latitude,
            "longitude": self._longitude,
            "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "timezone": "auto",
            "forecast_days": 1,
        }
        try:
            resp = requests.get(
                self._url,
                params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=self._timeout_sec,
            )
            resp.raise_for_status()
            data = resp.json()
            current = data["current"]
            daily = data.get("daily", {})
            description, condition = describe_code(int(current["weather_code"]), self._lang)
            weather = Weather(
                location=self._location_name,
                description=description,
                condition=condition,
                temp_c=_round(current.get("temperature_2m")),
                feels_like_c=_round(current.get("apparent_temperature")),
                high_c=_round(_first(daily.get("temperature_2m_max"))),
                low_c=_round(_first(daily.get("temperature_2m_min"))),
                precip_prob=_round(_first(daily.get("precipitation_probability_max"))),
                wind_kph=_round(current.get("wind_speed_10m")),
                fetched_at=time.time(),
            )
        except (requests.RequestException, ValueError, KeyError, TypeError) as e:
            logger.warning("weather fetch failed for %s: %s", self._location_name, e)
            return None
        logger.info(
            "weather: %s %s (%s degrees)",
            weather.location, weather.description, weather.temp_c,
        )
        return weather


def _first(values: object) -> float | None:
    """Open-Meteo の daily は「日ごとの配列」で返る。forecast_days=1 なので先頭だけ見る。"""
    if isinstance(values, list) and values and isinstance(values[0], (int, float)):
        return float(values[0])
    return None
