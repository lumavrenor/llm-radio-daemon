"""音楽ストリーム用のスレッドセーフなリングバッファ。

MusicThread が書き込み、AudioCallback（sounddeviceのスレッド）が読み出す。
読み出し側は絶対にブロックしてはいけないので、データが足りなければ
無音で埋めて返す（アンダーラン）。
"""

from __future__ import annotations

import threading

import numpy as np


class AudioRingBuffer:
    def __init__(self, capacity_frames: int, channels: int = 2):
        self._buf = np.zeros((capacity_frames, channels), dtype=np.float32)
        self._capacity = capacity_frames
        self._channels = channels
        self._write_pos = 0
        self._read_pos = 0
        self._available = 0
        self._lock = threading.Lock()

    def write(self, data: np.ndarray) -> int:
        """(n, channels) の float32 を書き込む。空きがなければ書けるだけ書いて残りは捨てる。"""
        with self._lock:
            n = data.shape[0]
            free = self._capacity - self._available
            to_write = min(n, free)
            if to_write <= 0:
                return 0
            end_space = self._capacity - self._write_pos
            if to_write <= end_space:
                self._buf[self._write_pos : self._write_pos + to_write] = data[:to_write]
            else:
                self._buf[self._write_pos :] = data[:end_space]
                self._buf[: to_write - end_space] = data[end_space:to_write]
            self._write_pos = (self._write_pos + to_write) % self._capacity
            self._available += to_write
            return to_write

    def read(self, n: int) -> tuple[np.ndarray, bool]:
        """n frame 読み出す。不足分は無音で埋め、アンダーランしたかを合わせて返す。"""
        with self._lock:
            out = np.zeros((n, self._channels), dtype=np.float32)
            to_read = min(n, self._available)
            if to_read > 0:
                end_space = self._capacity - self._read_pos
                if to_read <= end_space:
                    out[:to_read] = self._buf[self._read_pos : self._read_pos + to_read]
                else:
                    out[:end_space] = self._buf[self._read_pos :]
                    out[end_space:to_read] = self._buf[: to_read - end_space]
                self._read_pos = (self._read_pos + to_read) % self._capacity
                self._available -= to_read
            return out, to_read < n

    @property
    def available_frames(self) -> int:
        with self._lock:
            return self._available
