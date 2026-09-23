"""ICYメタデータの読み取り（now_playing 表示用）。

ffmpegにストリームを渡して再生する経路とは別に、Icy-MetaData: 1 ヘッダを
付けた軽量な専用コネクションを張り、StreamTitle だけを抜き出す。
実際の曲替わり検知からのMusicBrainz連携（ネタ化）はv3。v1では表示のみに使う。
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Callable

import requests

from ..sources import USER_AGENT

logger = logging.getLogger(__name__)

_STREAM_TITLE_RE = re.compile(r"StreamTitle='([^']*)';")


class IcyMetadataReader(threading.Thread):
    def __init__(
        self,
        url: str,
        on_title_change: Callable[[str], None],
        stop_event: threading.Event | None = None,
        timeout_sec: int = 10,
        max_backoff_sec: float = 60.0,
        no_meta_retry_sec: float = 60.0,
    ):
        super().__init__(name="IcyMetadataReader", daemon=True)
        self._url = url
        self._on_title_change = on_title_change
        self._stop_event = stop_event or threading.Event()
        self._timeout_sec = timeout_sec
        self._max_backoff_sec = max_backoff_sec
        self._no_meta_retry_sec = no_meta_retry_sec
        self._last_title: str | None = None

    def set_url(self, url: str) -> None:
        """読み取り先の局を切り替える。読み取りループは次のブロックで抜けて繋ぎ直す。"""
        if url == self._url:
            return
        self._url = url
        self._last_title = None  # 局が変われば同じ曲名でも「新しい曲」として通知する

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        backoff = 1.0
        while not self._stop_event.is_set():
            no_meta = False
            try:
                no_meta = not self._read_once()
                backoff = 1.0
            except Exception as e:
                logger.warning("icy metadata read failed: %s", e)
            if self._stop_event.is_set():
                return
            # ICY メタデータを持たない局に 1 秒間隔で繋ぎ直しても意味がないので、
            # 局が変わるのを待つくらいの間隔まで落とす。
            wait_sec = self._no_meta_retry_sec if no_meta else backoff
            if self._stop_event.wait(wait_sec):
                return
            backoff = min(backoff * 2, self._max_backoff_sec)

    def _read_once(self) -> bool:
        """接続してメタデータを読む。ICY メタデータ非対応の局なら False。"""
        headers = {"Icy-MetaData": "1", "User-Agent": USER_AGENT}
        url = self._url
        with requests.get(
            url, headers=headers, stream=True, timeout=(self._timeout_sec, None)
        ) as resp:
            resp.raise_for_status()
            meta_int = int(resp.headers.get("icy-metaint", 0) or 0)
            if meta_int <= 0:
                logger.info(
                    "stream has no icy-metaint header; now_playing is unavailable for this stream (%s)",
                    url,
                )
                return False

            raw = resp.raw
            while not self._stop_event.is_set():
                if self._url != url:
                    return True  # 局が切り替わった。抜けて新しいURLへ繋ぎ直す
                audio_chunk = raw.read(meta_int)
                if not audio_chunk:
                    raise ConnectionError("stream ended while reading audio block")
                length_byte = raw.read(1)
                if not length_byte:
                    raise ConnectionError("stream ended while reading metadata length")
                meta_len = length_byte[0] * 16
                if meta_len > 0:
                    meta_bytes = raw.read(meta_len)
                    self._handle_meta(meta_bytes)
        return True

    def _handle_meta(self, meta_bytes: bytes) -> None:
        text = meta_bytes.rstrip(b"\x00").decode("utf-8", errors="ignore")
        match = _STREAM_TITLE_RE.search(text)
        if not match:
            return
        title = match.group(1).strip()
        if title and title != self._last_title:
            self._last_title = title
            self._on_title_change(title)
