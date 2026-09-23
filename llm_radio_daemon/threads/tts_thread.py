"""TTS合成スレッド。script_queue → VOICEVOX → speech_queue (PCM)。

VOICEVOX が落ちていても放送を止めない。失敗した行は数回リトライしてから
諦めて次の行へ進む（音楽だけが流れる状態にはなるが、無音にはしない）。

字幕・発話ログはここでは出さない（speech_queue のぶんだけ音声より先行するため）。
実際に再生が始まった行を AudioMixer → AnnounceThread が拾って表示する。

NG ワード（sensitive.blocked_term）: LLM が生成したセリフに NG ワードが入っていたら
キャラクターの声で読ませず、短い間（無音）に差し替える。行そのものは捨てずに
speech_queue へ流す。朗読の進行位置・リクエストの記録・ひな壇の切り替えは「その行の
再生開始」を合図にしているので、行ごと落とすと同じチャンクを延々と読み直してしまうため。

コーナー切り替え（director.py）: 台本の取り出しは director 経由で行い、「今は合成
中か」を director が確実に知れるようにする。切り替え待ちの間は「いま流れている
台本」と締めの台本だけを合成し、次のトークは取り出した時点／行の合間で捨てる。
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

from .. import sensitive
from ..audio.resample import mono_to_stereo, resample_mono
from ..config import CastMember
from ..script import Script, ScriptLine
from ..state import SharedState
from ..tts import TTSBackend, VoiceHandle

if TYPE_CHECKING:
    from ..director import ProgramDirector

logger = logging.getLogger(__name__)

_MAX_RETRIES_PER_LINE = 2
_RETRY_DELAY_SEC = 1.0
_IDLE_POLL_SEC = 0.05

# フィラー・通常トーク（天気・ニュース等）は台本の切れ目に何も「間」が無く、続けて
# 流れると発話がベタ付きに聞こえる。朗読系コーナーは pad_ms を自前で細かく制御して
# いるので、それらが既に何か設定している場合は上書きしない。
_INTER_SEGMENT_PAD_MS = 3000

# NG ワードで読み上げを止めた行の代わりに流す無音の長さと、字幕に出す代わりの文言。
_BLOCKED_LINE_SILENCE_MS = 800
_BLOCKED_LINE_TEXT = "……"


def _mute_if_blocked(line: ScriptLine) -> bool:
    """NG ワードを含む行なら、字幕・読み上げのテキストを伏せて True を返す。

    読書コーナーの本文（reading_chunk_index 付き）は青空文庫・Project Gutenberg の原文を
    そのまま読んでいる行で、LLM の生成物ではないので対象外（古典には当時の差別語が普通に
    出てくるため、照合すると作品が読めなくなる）。
    """
    if line.reading_chunk_index is not None:
        return False
    term = sensitive.blocked_term(line.text) or sensitive.blocked_term(line.speech_text)
    if term is None:
        return False
    logger.warning(
        "NG word %r found in a generated line; muting it instead of speaking [%s] %s",
        term, line.speaker, line.text,
    )
    line.text = _BLOCKED_LINE_TEXT
    line.speech_text = None
    return True


def synthesize_line(
    backend: TTSBackend,
    cast_by_id: dict[str, CastMember],
    mixer_samplerate: int,
    line: ScriptLine,
) -> np.ndarray | None:
    """1行を合成してミキサー用のステレオ PCM にする。諦めたら None。"""
    member = cast_by_id.get(line.speaker)
    if member is None:
        logger.warning("unknown speaker %r; skipping line", line.speaker)
        return None
    speaker_id = member.resolve_style(line.style)

    if _mute_if_blocked(line):
        pcm = np.zeros(int(mixer_samplerate * _BLOCKED_LINE_SILENCE_MS / 1000), dtype=np.float32)
    else:
        pcm = _synth_with_retry(backend, line, speaker_id, mixer_samplerate)
    if pcm is None:
        return None

    stereo = mono_to_stereo(pcm)

    # 朗読の「間」（§10.4）：合成音の末尾に無音を足すだけ。ミキサー側は変更しない。
    if line.pad_ms:
        pad_frames = int(mixer_samplerate * line.pad_ms / 1000)
        if pad_frames > 0:
            stereo = np.concatenate(
                [stereo, np.zeros((pad_frames, stereo.shape[1]), dtype=stereo.dtype)]
            )
    return stereo


def _synth_with_retry(
    backend: TTSBackend, line: ScriptLine, speaker_id: VoiceHandle, mixer_samplerate: int
) -> np.ndarray | None:
    for attempt in range(_MAX_RETRIES_PER_LINE + 1):
        try:
            mono, sr = backend.synth(line.speech_text or line.text, speaker_id)
            return resample_mono(mono, sr, mixer_samplerate)
        except Exception as e:
            logger.warning(
                "TTS synth failed (attempt %d/%d): %s",
                attempt + 1,
                _MAX_RETRIES_PER_LINE + 1,
                e,
            )
            if attempt < _MAX_RETRIES_PER_LINE:
                time.sleep(_RETRY_DELAY_SEC)
    return None


class TTSThread(threading.Thread):
    def __init__(
        self,
        script_queue: "queue.Queue[Script]",
        speech_queue: "queue.Queue[tuple[ScriptLine, bool, np.ndarray]]",
        backend: TTSBackend,
        cast_by_id: dict[str, CastMember],
        mixer_samplerate: int,
        director: "ProgramDirector",
        stop_event: threading.Event | None = None,
        state: SharedState | None = None,
    ):
        super().__init__(name="TTSThread", daemon=True)
        self._script_queue = script_queue
        self._speech_queue = speech_queue
        self._backend = backend
        self._cast_by_id = cast_by_id
        self._mixer_samplerate = mixer_samplerate
        self._director = director
        self._stop_event = stop_event or threading.Event()
        self._state = state

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            script = self._director.tts_take(self._script_queue)
            if script is None:
                self._stop_event.wait(_IDLE_POLL_SEC)
                continue
            try:
                self._process(script)
            finally:
                self._director.tts_done()

    def _process(self, script: Script) -> None:
        if not self._director.tts_should_run(script):
            logger.info("Not playing script because waiting for corner switch: %s", script.topic_title)
            return
        self._tag_active_cast(script)
        for line in script.lines:
            line.script_seq = script.seq
            line.closing = script.is_closing
        self._apply_inter_segment_pad(script)
        for i, line in enumerate(script.lines):
            if self._stop_event.is_set():
                return
            if not self._director.tts_should_run(script):
                logger.info(
                    "Discarding the remaining %d script lines while waiting for a corner switch: %s",
                    len(script.lines) - i, script.topic_title,
                )
                return
            self._speak_line(line, script.is_filler)

    def _apply_inter_segment_pad(self, script: Script) -> None:
        """台本の最後の行に、次の台本との間に空ける無音を足す（朗読系コーナーは対象外）。

        暗転をまたぐ締め（is_closing）はどのみちフェードで間が空くので触らない。
        """
        if (
            not script.lines
            or script.is_closing
            or script.is_literary_reading
            or script.is_translated_reading
            or script.is_biography_reading
            or script.is_generated_drama
        ):
            return
        last = script.lines[-1]
        if not last.pad_ms:
            last.pad_ms = _INTER_SEGMENT_PAD_MS

    def _tag_active_cast(self, script: Script) -> None:
        """通常トークの台本の各行に「再生開始時に並べるひな壇の顔ぶれ」を貼る。

        以前はここ（合成時＝台本を取り出した時）で ``state.active_cast_ids`` を直接
        書いていたが、合成は再生より数行先行するため、前の話題のトークがまだ流れて
        いるうちに次の話題のキャストへ入れ替わってしまう。そこで顔ぶれは行に貼るだけに
        して、実際にその行の再生が始まった時に AnnounceThread が適用する。

        フィラー・読書コーナー・翻訳朗読コーナー・偉人伝トーク・ラジオドラマ朗読は対象外
        （読書・翻訳朗読・偉人伝は reading_cast_ids、小説はシーン単位で
        GeneratedDramaCorner が active_cast_ids を出す。フィラーは直前のひな壇構成を保つ）。
        """
        if (
            self._state is None
            or script.is_filler
            or script.is_literary_reading
            or script.is_translated_reading
            or script.is_biography_reading
            or script.is_generated_drama
        ):
            return
        # 抽選された出演者ぜんぶ（min/max_speakers ぶん）を並べる。台本で実際に
        # セリフが割り振られるのは小モデルだと数人に収束するが、ひな壇は抽選人数
        # どおりに立たせたい。appearer_ids が無い古い経路は実際の話者へフォールバック。
        ids = tuple(script.appearer_ids) or tuple(
            dict.fromkeys(l.speaker for l in script.lines)
        )
        if not ids:
            return
        # 全行に貼る。先頭行の合成が失敗して speech_queue に載らなくても、
        # 次の行で顔ぶれが適用されるようにするため（AnnounceThread は変化時のみ反映）。
        for line in script.lines:
            line.talk_cast_ids = ids

    def _speak_line(self, line: ScriptLine, is_filler: bool) -> None:
        stereo = synthesize_line(self._backend, self._cast_by_id, self._mixer_samplerate, line)
        if stereo is None:
            return  # このセリフは諦めて次へ

        while not self._stop_event.is_set():
            try:
                # 行そのものを渡す。字幕・朗読の進行記録は「実際に再生が始まった行」を
                # 見て AnnounceThread が行うため、行に付いた情報を落とさず運ぶ。
                self._speech_queue.put((line, is_filler, stereo), timeout=1.0)
                return
            except queue.Full:
                continue
