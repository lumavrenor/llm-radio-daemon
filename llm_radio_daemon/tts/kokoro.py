"""Kokoro-82M (ONNX) を CPU で直接叩く TTSBackend 実装（英語版）。

なぜ ``kokoro-onnx`` パッケージを使わないか
-------------------------------------------
``kokoro-onnx`` は ``phonemizer-fork``(GPL-3.0) と ``espeakng-loader`` を extra ではなく
**必須依存**として引く。本プロジェクトは MIT/Apache で揃える方針なので採れない。
一方、モデルの ONNX インターフェースは

    tokens: int64[1, N] / style: float32[1, 256] / speed: float32[1]  ->  audio: float32[T]

の3入力だけで、onnxruntime を直接呼べば済む。llm_http.py で SDK を使わず HTTP を
直に書いているのと同じ流儀。

声の作り分け
------------
Kokoro には VOICEVOX のような感情スタイルが無い。代わりに 28 種の英語ボイスの
スタイルベクトルを**加重平均でブレンド**でき、キャストごとの固有の声を config の
配合レシピとして設計できる。既定の af_heart 自体が Bella+Sarah の 50:50。

GPU は LLM が占有しているので、ここは常に CPU で動かす（SPEC §4.4 と同じ理由）。
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .. import sentences
from ..config import ConfigError, KokoroStyle, VoiceBlend

if TYPE_CHECKING:
    from ..config import CastMember, TTSConfig

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000  # Kokoro のネイティブ。VOICEVOX と同じなので既存のリサンプル経路に乗る

# style 配列の第0次元。系列長で引くので、これがトークン数の上限でもある。
_MAX_STYLE_LEN = 510
# 開始・終了の 0 を入れるぶん 2 つ余裕を見る
_MAX_TOKENS = _MAX_STYLE_LEN - 2

_MODEL_NAME = "kokoro-v1.0.onnx"
_VOICES_NAME = "voices-v1.0.bin"
_CONFIG_NAME = "config.json"

_MODEL_URLS = {
    _MODEL_NAME: (
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
        "kokoro-v1.0.onnx"
    ),
    _VOICES_NAME: (
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
        "voices-v1.0.bin"
    ),
    _CONFIG_NAME: "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/config.json",
}

USER_AGENT = "llm-radio-daemon/0.1 (personal 24h local-LLM radio project; kokoro model fetch)"
TIMEOUT = (10, 600)  # モデルが 310MB あるので読み取りは長めに


class KokoroError(RuntimeError):
    pass


class KokoroVoice:
    """解決済みの声。style は (510, 1, 256) のブレンド済み配列のまま持つ。

    合成時の系列長で引く必要があるので、1行に潰さずここで抱えておく。
    """

    __slots__ = ("label", "style", "speed")

    def __init__(self, label: str, style: np.ndarray, speed: float) -> None:
        self.label = label
        self.style = style
        self.speed = speed

    def __repr__(self) -> str:  # ログ用
        return f"KokoroVoice({self.label!r}, speed={self.speed})"


class KokoroBackend:
    def __init__(self, cfg: TTSConfig) -> None:
        self._model_dir = Path(cfg.model_dir)
        self._threads = max(0, int(cfg.threads))
        self._lexicon_file = cfg.lexicon_file
        self._session = None
        self._voices = None
        self._vocab: dict[str, int] = {}
        self._g2p = None
        self._token_input = "tokens"
        # onnxruntime のセッションはスレッド安全だが、G2P（spaCy）は安全でないため
        # 合成全体を直列化する。TTSThread は1本なので実害はない。
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ 起動

    def health_check(self) -> str:
        """モデルを（無ければ取得して）読み込み、1回だけ試し合成する。

        起動時に落としておかないと、放送が始まってから毎行失敗して無音になる。
        """
        self._ensure_loaded()
        with self._lock:
            pcm = self._synth_chunk("Ready.", self._default_style(), 1.0)
        if pcm.size == 0:
            raise KokoroError("Kokoro の試し合成が空でした")
        import onnxruntime as ort

        return f"Kokoro-82M (onnxruntime {ort.__version__}, threads={self._threads or 'auto'})"

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        import json

        import onnxruntime as ort

        _fetch_models(self._model_dir)
        self._vocab = json.loads(
            (self._model_dir / _CONFIG_NAME).read_text(encoding="utf-8")
        )["vocab"]
        self._voices = np.load(self._model_dir / _VOICES_NAME, allow_pickle=True)

        so = ort.SessionOptions()
        if self._threads > 0:
            so.intra_op_num_threads = self._threads
            so.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(self._model_dir / _MODEL_NAME),
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )
        # 配布元によって入力名が "tokens" だったり "input_ids" だったりする
        self._token_input = self._session.get_inputs()[0].name
        self._g2p = _build_g2p(self._lexicon_file)
        logger.info(
            "Loaded Kokoro: %s (voices=%d, threads=%s)",
            self._model_dir,
            len(self._voice_names()),
            self._threads or "auto",
        )

    def _voice_names(self) -> list[str]:
        v = self._voices
        return list(v.keys()) if hasattr(v, "keys") else list(v.files)

    def _default_style(self) -> np.ndarray:
        return self._voices["af_heart"]

    # ------------------------------------------------------------- 声の解決

    def resolve_cast_voices(self, cast: list[CastMember]) -> None:
        """[[cast]] の kokoro_voice / kokoro_styles を解決して各 CastMember に書き込む。

        存在しないボイス名や重み合計 0 のレシピは、聞こえない声で放送を始めるより先に
        ここで落とす（VOICEVOX バックエンドと同じ規約）。
        """
        self._ensure_loaded()
        available = self._voice_names()
        for m in cast:
            # kokoro_voice だけ書いた場合は "normal" 1つだけのスタイルとみなす
            styles = m.kokoro_styles or [KokoroStyle(name="normal")]
            resolved: dict[str, KokoroVoice] = {}
            names: list[str] = []
            for st in styles:
                # blend 省略時は [[cast]] の kokoro_voice を単体で使う
                recipe = st.blend or ([VoiceBlend(m.kokoro_voice)] if m.kokoro_voice else [])
                if not recipe:
                    raise ConfigError(
                        f"[[cast]] {m.id!r} の kokoro_styles {st.name!r}: blend が空です。"
                        " blend を書くか、[[cast]] に kokoro_voice を書いてください。"
                    )
                total = float(sum(b.weight for b in recipe))
                if total <= 0:
                    raise ConfigError(
                        f"[[cast]] {m.id!r} の kokoro_styles {st.name!r}: weight の合計が"
                        f" {total} です。正の値にしてください。"
                    )
                blended = None
                for b in recipe:
                    if b.voice not in available:
                        raise ConfigError(
                            f"[[cast]] {m.id!r} の kokoro_styles {st.name!r}:"
                            f" voice={b.voice!r} は存在しません。"
                            f"使えるボイス: {', '.join(sorted(available))}"
                        )
                    part = self._voices[b.voice].astype(np.float32) * (b.weight / total)
                    blended = part if blended is None else blended + part
                label = "+".join(f"{b.voice}:{b.weight:g}" for b in recipe)
                resolved[st.name] = KokoroVoice(label, blended, st.speed)
                names.append(st.name)
            m.resolved_voices = resolved
            m.resolved_style_names = names
            logger.debug("cast %s voices: %s", m.id, {k: v.label for k, v in resolved.items()})

    # --------------------------------------------------------------- 合成

    def synth(self, text: str, voice: KokoroVoice) -> tuple[np.ndarray, int]:
        self._ensure_loaded()
        if not isinstance(voice, KokoroVoice):
            raise KokoroError(
                f"声のハンドルが Kokoro のものではありません: {type(voice).__name__}"
                "（[tts] backend と [[cast]] の設定が食い違っていませんか）"
            )
        with self._lock:
            chunks = [
                self._synth_chunk(part, voice.style, voice.speed)
                for part in self._split_for_model(text)
            ]
        chunks = [c for c in chunks if c.size]
        if not chunks:
            return np.zeros(0, dtype=np.float32), SAMPLE_RATE
        return np.concatenate(chunks), SAMPLE_RATE

    def _to_ids(self, text: str) -> list[int]:
        phonemes, _ = self._g2p(text)
        return [self._vocab[c] for c in phonemes if c in self._vocab]

    def _split_for_model(self, text: str) -> list[str]:
        """音素にすると 510 トークンを超える長文を、文→句の順に切って収める。

        朗読コーナーは1行が長くなりうる。ここで切らないとモデルに入らない。
        """
        if len(self._to_ids(text)) <= _MAX_TOKENS:
            return [text]
        out: list[str] = []
        for sentence in _split_sentences(text):
            if len(self._to_ids(sentence)) <= _MAX_TOKENS:
                out.append(sentence)
                continue
            # 1文でも長すぎる場合は語で詰めていく
            buf: list[str] = []
            for word in sentence.split():
                trial = " ".join(buf + [word])
                if buf and len(self._to_ids(trial)) > _MAX_TOKENS:
                    out.append(" ".join(buf))
                    buf = [word]
                else:
                    buf.append(word)
            if buf:
                out.append(" ".join(buf))
        return out or [text]

    def _synth_chunk(self, text: str, style: np.ndarray, speed: float) -> np.ndarray:
        ids = self._to_ids(text)
        if not ids:
            return np.zeros(0, dtype=np.float32)
        row = style[len(ids) + 2]  # style は系列長で引く（ここを誤ると声が変わる）
        audio = self._session.run(
            None,
            {
                self._token_input: np.array([[0, *ids, 0]], dtype=np.int64),
                "style": np.asarray(row, dtype=np.float32).reshape(1, -1),
                "speed": np.array([speed], dtype=np.float32),
            },
        )[0]
        return np.asarray(audio, dtype=np.float32).flatten()


# ---------------------------------------------------------------- 補助


def _split_sentences(text: str) -> list[str]:
    """英語の文分割。510トークン超の長文を文単位へ割るのに使う。

    以前はここに ``(?<=[.!?])\\s+`` の素朴版を持っていたが、``Mr.`` や ``a.m.`` で
    切れてしまう。朗読コーナー・ラジオドラマも同じ判定を要るので ``sentences`` へ寄せた。
    """
    return [s for s in (p.strip() for p in sentences.split_en(text)) if s]


def _build_g2p(lexicon_file: str):
    """misaki の英語 G2P を espeak 抜きで組む。

    fallback を必ず差すこと。差さないと辞書外語で TypeError が飛び、
    固有名詞を含む行が丸ごと無音になる（g2p_fallback.py の説明を参照）。
    """
    from misaki import en

    from .g2p_fallback import ArpabetFallback

    g2p = en.G2P(trf=False, british=False, fallback=ArpabetFallback())
    _apply_lexicon(g2p, lexicon_file)
    return g2p


def _apply_lexicon(g2p, lexicon_file: str) -> None:
    """[tts] lexicon_file（1行 "word<TAB>IPA"）を misaki の辞書へ流し込む。

    指定が無い・ファイルが無ければ何もしない。
    """
    if not lexicon_file:
        return
    path = Path(lexicon_file)
    if not path.exists():
        logger.info("[tts] lexicon_file not found; skipping pronunciation overrides: %s", path)
        return
    golds = getattr(getattr(g2p, "lexicon", None), "golds", None)
    if golds is None:
        logger.warning("misaki's dictionary structure differs from what was expected; cannot apply lexicon_file")
        return
    count = 0
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t") if "\t" in line else line.split(None, 1)
        if len(parts) != 2:
            logger.warning("%s:%d: not in 'word<TAB>IPA' format: %r", path, lineno, raw)
            continue
        word, ipa = parts[0].strip(), parts[1].strip()
        if word and ipa:
            golds[word.lower()] = ipa
            count += 1
    logger.info("[tts] loaded %d pronunciation entries from lexicon_file: %s", count, path)


def _fetch_models(model_dir: Path) -> None:
    """モデル一式を（無ければ）取得する。青空文庫の lazy fetch と同じ方針。"""
    import requests

    model_dir.mkdir(parents=True, exist_ok=True)
    for name, url in _MODEL_URLS.items():
        dest = model_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        logger.info("Fetching Kokoro model (first time only): %s", name)
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            with requests.get(
                url, stream=True, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}
            ) as resp:
                resp.raise_for_status()
                with tmp.open("wb") as f:
                    for block in resp.iter_content(chunk_size=1 << 20):
                        f.write(block)
            tmp.replace(dest)  # 途中で落ちた不完全なファイルを掴まないように
        except Exception as e:
            tmp.unlink(missing_ok=True)
            raise KokoroError(f"{name} を取得できませんでした（{url}）: {e}") from e
        logger.info("Fetched: %s (%.1f MB)", dest, dest.stat().st_size / (1 << 20))
