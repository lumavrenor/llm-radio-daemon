"""ネットラジオの音声を ffmpeg 経由でデコードし、リングバッファに供給する。

切断時は指数バックオフで自動再接続する（1s, 2s, 4s, ... 最大60s）。
ここが止まってもプロセス全体は落とさない。呼び出し側スレッドを専有するだけ。
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time

import numpy as np

from .ringbuffer import AudioRingBuffer

logger = logging.getLogger(__name__)

_BYTES_PER_SAMPLE = 4  # float32
_CHANNELS = 2
_BYTES_PER_FRAME = _BYTES_PER_SAMPLE * _CHANNELS


def _resolve_ffmpeg() -> str | None:
    """ffmpeg 実行ファイルのパスを返す。

    imageio-ffmpeg が同梱するバイナリを優先する（pip install 時点で wheel に
    含まれるバイナリが展開済みなので、ユーザーの手動インストール・PATH設定が
    不要になる）。
    imageio-ffmpeg が使えない場合は PATH 上の ffmpeg にフォールバックする
    （ユーザーが自前の ffmpeg を使いたい場合もこちらで拾える）。
    """
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        logger.warning(
            "Failed to obtain ffmpeg via imageio-ffmpeg. Searching for ffmpeg on PATH.",
            exc_info=True,
        )
    return shutil.which("ffmpeg")


class MusicThread(threading.Thread):
    def __init__(
        self,
        url: str,
        ring: AudioRingBuffer,
        samplerate: int = 48000,
        stop_event: threading.Event | None = None,
        max_backoff_sec: float = 60.0,
    ):
        super().__init__(name="MusicThread", daemon=True)
        self._url = url
        self._ring = ring
        self._samplerate = samplerate
        self._stop_event = stop_event or threading.Event()
        self._max_backoff_sec = max_backoff_sec
        self._proc: subprocess.Popen | None = None
        self._ffmpeg_path: str | None = None
        # 局の切り替え要求。立っていたらバックオフを待たずに繋ぎ直す。
        self._switch_event = threading.Event()
        # 最後に音声データを受け取った時刻。StationThread が「この局は鳴っていない」の
        # 判定に使う（切り替え直後は繋ぎに行っている最中なので、そこを起点にする）。
        self._last_data_at = time.monotonic()

    def set_url(self, url: str) -> None:
        """再生する局を切り替える。ffmpeg を落とせば再接続ループが新しいURLへ繋ぎ直す。"""
        if url == self._url:
            return
        self._url = url
        self._last_data_at = time.monotonic()
        self._switch_event.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.kill()

    def seconds_since_data(self) -> float:
        """最後に音が来てからの経過秒。繋がっていれば常に 0 付近になる。"""
        return time.monotonic() - self._last_data_at

    def stop(self) -> None:
        self._stop_event.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.kill()

    def run(self) -> None:
        self._ffmpeg_path = _resolve_ffmpeg()
        if self._ffmpeg_path is None:
            logger.error(
                "ffmpeg not found (the imageio-ffmpeg download also failed). "
                "Check your network connection, or install ffmpeg manually and add it to PATH. "
                "The music track will remain silent."
            )

        backoff = 1.0
        while not self._stop_event.is_set():
            try:
                got_any_data = self._play_once()
                if got_any_data:
                    backoff = 1.0
            except Exception:
                logger.exception("music stream error (url=%s)", self._url)

            if self._stop_event.is_set():
                return
            if self._switch_event.is_set():
                # 局の切り替えで自分から切ったので、バックオフは待たずに繋ぎに行く。
                self._switch_event.clear()
                backoff = 1.0
                continue
            logger.info("reconnecting to music stream in %.0fs", backoff)
            if self._stop_event.wait(backoff):
                return
            backoff = min(backoff * 2, self._max_backoff_sec)

    def _play_once(self) -> bool:
        if self._ffmpeg_path is None:
            # run() で見つからなかった場合。バックオフしながら再試行はするが、
            # 見つからない状態そのものは変わらないので毎回同じ警告で返す。
            raise RuntimeError("ffmpeg が見つからないため再生できません")

        cmd = [
            self._ffmpeg_path,
            "-i", self._url,
            "-f", "f32le",
            "-ar", str(self._samplerate),
            "-ac", str(_CHANNELS),
            "-loglevel", "error",
            "-",
        ]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        got_any_data = False
        try:
            chunk_frames = 4096
            chunk_bytes = chunk_frames * _BYTES_PER_FRAME
            while not self._stop_event.is_set():
                data = self._proc.stdout.read(chunk_bytes)
                if not data:
                    break
                got_any_data = True
                self._last_data_at = time.monotonic()
                usable_frames = len(data) // _BYTES_PER_FRAME
                usable_bytes = usable_frames * _BYTES_PER_FRAME
                if usable_frames == 0:
                    continue
                pcm = np.frombuffer(data[:usable_bytes], dtype=np.float32).reshape(-1, _CHANNELS)
                self._ring.write(pcm)
            if not got_any_data:
                raise RuntimeError(
                    "ffmpeg がデータを返しませんでした（URL / ネットワーク / ffmpeg のインストールを確認してください）"
                )
            return got_any_data
        finally:
            proc = self._proc
            self._proc = None
            if proc.poll() is None:
                proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("ffmpeg did not exit within timeout after kill")
