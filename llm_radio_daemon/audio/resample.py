"""TTSバックエンドが返すネイティブサンプルレートの音声を、
ミキサーのサンプルレート(48kHz)に揃えるための小さなユーティリティ。

TTSBackend はどんなサンプルレートで返してきてもよい設計にしてあるため
（将来 Style-Bert-VITS2 等に差し替える前提）、リサンプルはここに集約する。
"""

from __future__ import annotations

import numpy as np


def resample_mono(pcm: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr or len(pcm) == 0:
        return pcm.astype(np.float32, copy=False)
    duration = len(pcm) / orig_sr
    target_len = max(1, int(round(duration * target_sr)))
    orig_idx = np.linspace(0.0, len(pcm) - 1, num=len(pcm))
    target_idx = np.linspace(0.0, len(pcm) - 1, num=target_len)
    return np.interp(target_idx, orig_idx, pcm).astype(np.float32)


def mono_to_stereo(pcm: np.ndarray) -> np.ndarray:
    return np.repeat(pcm.reshape(-1, 1), 2, axis=1).astype(np.float32, copy=False)
