"""VOICEVOX ENGINE (HTTP API) を叩く TTSBackend 実装。

POST /audio_query?text=...&speaker=...  → クエリJSON
POST /synthesis?speaker=...  (body: 上のJSON)  → WAVバイト列
"""

from __future__ import annotations

import io
import logging
import wave
from typing import TYPE_CHECKING

import numpy as np
import requests

from ..config import ConfigError

if TYPE_CHECKING:
    from ..config import CastMember

logger = logging.getLogger(__name__)


class VoicevoxError(RuntimeError):
    pass


class VoicevoxBackend:
    def __init__(self, host: str, timeout_sec: int = 10):
        self._host = host.rstrip("/")
        self._timeout_sec = timeout_sec

    def health_check(self) -> str:
        """起動時の疎通確認。ENGINE のバージョン文字列を返す。落ちていれば例外。

        VOICEVOX が居ないまま走らせると、生成済みの台本が TTSThread で
        1行ずつ捨てられて（リトライ後に諦める）音にならないまま消える。
        起動時に気付けるよう、ここで一度だけ叩いておく。
        """
        resp = requests.get(f"{self._host}/version", timeout=self._timeout_sec)
        resp.raise_for_status()
        return resp.text.strip().strip('"')

    def list_speakers(self) -> dict[str, dict[str, int]]:
        """/speakers から (キャラ名 → {スタイル名: 話者ID}) を取得する。

        config.toml の [[cast]] は話者IDを持たず、キャラ名とスタイル名だけを書く。
        起動時にここで実データと突き合わせて解決する（resolve_cast_voices）。
        """
        resp = requests.get(f"{self._host}/speakers", timeout=self._timeout_sec)
        resp.raise_for_status()
        out: dict[str, dict[str, int]] = {}
        for speaker in resp.json():
            out[speaker["name"]] = {s["name"]: s["id"] for s in speaker["styles"]}
        return out

    def resolve_cast_voices(self, cast: list[CastMember]) -> None:
        """[[cast]] の声設定を /speakers の実データと突き合わせて話者IDへ解決する。"""
        resolve_voicevox_voices(cast, self.list_speakers())

    def synth(self, text: str, speaker: int) -> tuple[np.ndarray, int]:
        query = self._audio_query(text, speaker)
        wav_bytes = self._synthesis(query, speaker)
        return _wav_bytes_to_float32(wav_bytes)

    def _audio_query(self, text: str, speaker: int) -> dict:
        resp = requests.post(
            f"{self._host}/audio_query",
            params={"text": text, "speaker": speaker},
            timeout=self._timeout_sec,
        )
        resp.raise_for_status()
        return resp.json()

    def _synthesis(self, query: dict, speaker: int) -> bytes:
        resp = requests.post(
            f"{self._host}/synthesis",
            params={"speaker": speaker},
            json=query,
            timeout=self._timeout_sec,
        )
        resp.raise_for_status()
        return resp.content


def resolve_voicevox_voices(
    cast: list[CastMember], speakers: dict[str, dict[str, int]]
) -> None:
    """[[cast]] の voicevox_speaker_name / voicevox_speaker_styles を、VOICEVOX ENGINE の
    /speakers から取れる実データ（``VoicevoxBackend.list_speakers()``）と突き合わせて
    実際の話者IDへ解決し、各 CastMember に書き込む（インプレース）。

    名前・スタイル名が実在しなければ、放送を始めずにここで即座に落とす
    （聞こえない声で番組を続けるより、起動時に気付けたほうがわかりやすい）。
    """
    for m in cast:
        if not m.voicevox_speaker_name:
            raise ConfigError(
                f"[[cast]] {m.id!r}: voicevox_speaker_name は必須です"
                "（VOICEVOX ENGINE の /speakers に実在するキャラ名）。"
            )
        styles_for_name = speakers.get(m.voicevox_speaker_name)
        if styles_for_name is None:
            raise ConfigError(
                f"[[cast]] {m.id!r}: voicevox_speaker_name={m.voicevox_speaker_name!r} は"
                f" VOICEVOX ENGINE に存在しません。"
                f"実在するキャラ名: {', '.join(sorted(speakers)) or 'なし'}"
            )
        style_names = list(m.voicevox_speaker_styles or ["ノーマル"])
        resolved: dict[str, int] = {}
        for style_name in style_names:
            style_id = styles_for_name.get(style_name)
            if style_id is None:
                raise ConfigError(
                    f"[[cast]] {m.id!r}: {m.voicevox_speaker_name!r} に"
                    f" スタイル {style_name!r} は存在しません。"
                    f"実在するスタイル: {', '.join(sorted(styles_for_name)) or 'なし'}"
                )
            resolved[style_name] = style_id
        m.resolved_voices = resolved
        m.resolved_style_names = style_names


def _wav_bytes_to_float32(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        samplerate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        pcm = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise VoicevoxError(f"unsupported WAV sample width: {sampwidth}")

    if n_channels > 1:
        pcm = pcm.reshape(-1, n_channels).mean(axis=1)

    return pcm.astype(np.float32, copy=False), samplerate
