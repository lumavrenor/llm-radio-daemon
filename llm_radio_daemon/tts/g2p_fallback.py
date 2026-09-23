"""辞書外語（OOV）のための英語 G2P フォールバック。

なぜ必要か
----------
Kokoro の標準 G2P である misaki は、辞書に無い語を espeak-ng に投げて解決する。
だが espeak-ng は GPL で、本プロジェクトは MIT/Apache で揃える方針なので使えない。
かといって ``fallback=None`` にすると、misaki は辞書外語を unk 文字に落とさず
``TypeError`` を投げる（misaki/en.py の ``t.phonemes + t.whitespace``）。

実際の放送文体で測ると OOV は全トークンの約 4.6%、しかし **15 文中 10 文**に
1語以上現れた。アーティスト名（musicbrainz）や選手名（MLB Stats API）は実行時に
無限に来るので静的な辞書では塞げない。フォールバックが無いと、固有名詞を含む行が
ことごとく例外になり TTSThread に捨てられて無音になる。

そこで g2p_en（Apache-2.0。CMUdict＋辞書外語用の小さなニューラルネット。torch 不要）
を使い、その出力の ARPAbet を Kokoro の IPA 語彙へ写像する。

これは「破綻しない読み」を保証するだけで「正しい読み」ではない
（例: Ryuichi → ɹijˈuki）。番組の看板になる語は [tts] lexicon_file で明示的に上書きする。
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

# ARPAbet -> misaki/Kokoro の US English IPA。
# A / I / O / W / Y は misaki が二重母音（eɪ / aɪ / oʊ / aʊ / ɔɪ）に割り当てている1文字表記。
_ARPA_TO_IPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "W", "AY": "I",
    "B": "b", "CH": "ʧ", "D": "d", "DH": "ð", "EH": "ɛ", "ER": "ɜɹ",
    "EY": "A", "F": "f", "G": "ɡ", "HH": "h", "IH": "ɪ", "IY": "i",
    "JH": "ʤ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ",
    "OW": "O", "OY": "Y", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ",
    "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u", "V": "v", "W": "w",
    "Y": "j", "Z": "z", "ZH": "ʒ",
}

# 強勢記号を前置する母音（ARPAbet の数字サフィックスが付くもの）
_VOWELS = frozenset(
    {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
)

_STRESS = {"1": "ˈ", "2": "ˌ"}  # 0（無強勢）は記号なし

# g2p_en が要求する nltk データ。g2p_en は 2019 年から更新が止まっており、
# nltk 側でのリソース改名（averaged_perceptron_tagger → *_eng）に追随していないため、
# 新旧どちらの名前も取りに行く。
_NLTK_RESOURCES = ("averaged_perceptron_tagger_eng", "averaged_perceptron_tagger", "cmudict")


def _ensure_nltk_data() -> None:
    import nltk

    for name in _NLTK_RESOURCES:
        try:
            nltk.download(name, quiet=True)
        except Exception as e:  # ネットワークが無くても既に入っていれば動く
            logger.debug("nltk.download(%s) failed (continuing if existing data is present): %s", name, e)


class ArpabetFallback:
    """misaki の ``fallback`` プロトコルに適合する OOV 解決器。

    ``fallback(token) -> (phonemes, rating)`` として呼ばれる。
    g2p_en の初期化は重い（CMUdict の読み込み）ので初回呼び出しまで遅延させる。
    """

    def __init__(self) -> None:
        self._g2p = None
        self._lock = threading.Lock()
        self._warned: set[str] = set()

    def _ensure_loaded(self):
        if self._g2p is None:
            with self._lock:
                if self._g2p is None:
                    _ensure_nltk_data()
                    from g2p_en import G2p

                    self._g2p = G2p()
        return self._g2p

    def phonemize(self, word: str) -> str:
        """単語 -> Kokoro の IPA 文字列。解決できなければ空文字。"""
        out: list[str] = []
        for ph in self._ensure_loaded()(word):
            if not ph or not ph[0].isalpha():
                continue  # 区切りや記号は捨てる（呼び出し側が空白を持っている）
            base, stress = ph, ""
            if ph[-1].isdigit():
                base, digit = ph[:-1], ph[-1]
                stress = _STRESS.get(digit, "")
                # 無強勢の AH / ER は曖昧母音になる（misaki の辞書もそう振る舞う）
                if digit == "0" and base == "AH":
                    out.append("ə")
                    continue
                if digit == "0" and base == "ER":
                    out.append("ɚ")
                    continue
            ipa = _ARPA_TO_IPA.get(base)
            if ipa is None:
                continue
            out.append((stress if base in _VOWELS else "") + ipa)
        return "".join(out)

    def __call__(self, token) -> tuple[str, int]:
        text = getattr(token, "text", None) or str(token)
        phonemes = self.phonemize(text)
        key = text.lower()
        if key not in self._warned:
            self._warned.add(key)
            if phonemes:
                # 辞書を育てられるように、1語につき1回だけ知らせる
                logger.warning(
                    "TTS: %r is not in the dictionary, estimating the reading (%s). "
                    "To pin the reading for a word, add it to [tts] lexicon_file.",
                    text,
                    phonemes,
                )
            else:
                logger.warning("TTS: could not estimate reading for %r (will be silent)", text)
        # rating は misaki が付ける確信度。辞書ヒット(4)より低い 2 を返す。
        return phonemes, 2
