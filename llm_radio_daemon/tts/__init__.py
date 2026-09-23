"""TTSバックエンドの共通インターフェース。

将来 Style-Bert-VITS2 等に差し替えられるよう、ここは VOICEVOX 固有の
知識を一切持たない。ネイティブのサンプルレートで返すだけでよく、
ミキサーのサンプルレートへのリサンプルは呼び出し側（TTSThread）で行う。

声の指定はバックエンドごとに形が違う（VOICEVOX は話者ID＝int、Kokoro は
スタイルベクトル）。そこで「起動時にバックエンドが [[cast]] を読んで
不透明なハンドルを作り、合成時にそれを受け取る」という形に統一している。
ハンドルの中身を知っているのはそれを作ったバックエンドだけでよい。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from ..config import CastMember, Config

# バックエンド不透明な声のハンドル。resolve_cast_voices() が作り synth() が食う。
# VOICEVOX なら話者ID(int)、Kokoro なら (スタイルベクトル, 速度)。
VoiceHandle = Any


class TTSBackend(Protocol):
    def health_check(self) -> str:
        """起動時の疎通確認。バージョン等の短い文字列を返す。使えなければ例外。"""
        ...

    def resolve_cast_voices(self, cast: list[CastMember]) -> None:
        """[[cast]] の声設定を実データと突き合わせ、各 CastMember へ
        解決済みハンドル（resolved_voices / resolved_style_names）を書き込む。

        名前やレシピが1つでも不正なら、聞こえない声のまま放送を始めるより先に
        ConfigError で落とすこと（全バックエンド共通の規約）。
        """
        ...

    def synth(self, text: str, voice: VoiceHandle) -> tuple[np.ndarray, int]:
        """mono float32 PCM（-1.0〜1.0）と、そのサンプルレートを返す。"""
        ...


def create_backend(config: Config) -> TTSBackend:
    """[tts] backend の名前から実装を選ぶ。

    重い依存（onnxruntime, spacy 等）を持つバックエンドがあるので、
    import は選ばれたものだけ行う。日本語版しか使わない人に
    英語TTSの依存をインストールさせないため。
    """
    name = (config.tts.backend or "").strip().lower()
    if name == "voicevox":
        from .voicevox import VoicevoxBackend

        return VoicevoxBackend(config.tts.host, timeout_sec=config.tts.timeout_sec)
    if name == "kokoro":
        from .kokoro import KokoroBackend

        return KokoroBackend(config.tts)
    raise ValueError(
        f"[tts] backend={config.tts.backend!r} は未知です。"
        ' "voicevox"（日本語）か "kokoro"（英語）を指定してください。'
    )
