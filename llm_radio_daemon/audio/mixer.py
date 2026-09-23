"""オーディオミキサー（最重要コンポーネント）。

すべての音（音楽 + 音声）をここで1本の sounddevice.OutputStream に混ぜる。
ダッキング、RMS計算（リップシンク用）もこのコールバックの中で完結させる。
コールバックはリアルタイムスレッドで呼ばれるため、ここでは
ネットワークI/O・ファイルI/O・ロックの長時間保持を一切行わない。

発話ゲート（コーナー切り替え・director.py）:
  - OPEN   : speech_queue を順に流す（通常）
  - FINISH : 「いま流れている台本」と締めの台本の行だけ流し、それ以外は捨てる
  - HOLD   : speech_queue には触らない（次のコーナーの作り置きを溜めたまま待たせる）
どのモードでも、director が差し込む優先レーン（リクエストの受けアナウンス）が先に流れる。
"""

from __future__ import annotations

import collections
import logging
import queue
import threading
import time

import numpy as np
import sounddevice as sd

from ..script import ScriptLine
from ..state import SharedState
from .ringbuffer import AudioRingBuffer

logger = logging.getLogger(__name__)

_SILENCE_LOG_THRESHOLD_SEC = 3.0

# 起動音量フェードインの長さ。画面側 (display/app.py の _STARTUP_FADE_SEC) と
# 揃えて、明転と同じ速さで音量が上がっているように聞こえる長さにすること。
_STARTUP_FADE_SEC = 1.4

GATE_OPEN = "open"
GATE_FINISH = "finish"
GATE_HOLD = "hold"


class AudioMixer:
    def __init__(
        self,
        state: SharedState,
        music_ring: AudioRingBuffer,
        speech_queue: "queue.Queue[tuple[ScriptLine, bool, np.ndarray]]",
        samplerate: int = 48000,
        blocksize: int = 1024,
        channels: int = 2,
        device: str = "",
        duck_db: float = -12.0,
        duck_attack_ms: float = 150.0,
        duck_release_ms: float = 500.0,
        duck_hold_ms: float = 300.0,
        announce_queue: "queue.Queue[tuple[ScriptLine, bool]] | None" = None,
    ):
        self._state = state
        self._music_ring = music_ring
        self._speech_queue = speech_queue
        # 「今から再生する行」を字幕・ログに反映するための通知先。
        # 字幕を TTS 合成時に更新すると speech_queue のぶんだけ音声より先行してしまう
        # （数人ぶん字幕が先に出る）ため、実際に再生が始まるこの層から通知する。
        self._announce_queue = announce_queue
        self._samplerate = samplerate
        self._blocksize = blocksize
        self._channels = channels
        self._device = device or None

        self._attack_frames = max(1, int(samplerate * duck_attack_ms / 1000))
        self._release_frames = max(1, int(samplerate * duck_release_ms / 1000))
        self._hold_frames = max(1, int(samplerate * duck_hold_ms / 1000))
        # 読書コーナー（§10.4）は朗読中だけ深いダッキングにする。SharedState.duck_db_override
        # を毎ブロック見て、変化したときだけステップを再計算する（変化は分に数回程度）。
        self._base_duck_db = duck_db
        self._active_duck_db = duck_db
        self._set_duck_db(duck_db)

        self._music_gain = 1.0
        self._hold_counter = 0

        self._current_speech: np.ndarray | None = None
        self._current_speech_pos = 0
        self._current_speaker: str | None = None
        self._current_is_filler = False
        self._current_is_priority = False

        # 「次の行を取り出して再生中にする」までを他スレッドの is_idle() / flush() と
        # 原子的にするためのロック。保持は数行ぶんの間だけ。
        self._gate_lock = threading.Lock()
        self._gate = GATE_OPEN
        self._gate_seq: int | None = None
        self._last_seq: int | None = None  # 最後に再生を始めた行の台本番号
        self._priority: collections.deque = collections.deque()

        self._silence_since: float | None = None
        self._ever_had_sound = False
        self._stream: sd.OutputStream | None = None

        # 起動音量フェードイン。state.startup_fade_started_at が立つまでは無音
        # （接続直後のストリーミングノイズを聞かせない）。立ったら画面の明転と
        # 同じ長さでゆっくり 0→1 に上げる。一度 1.0 に達したら以降は計算を省く。
        self._startup_fade_done = False

    def _set_duck_db(self, duck_db: float) -> None:
        self._active_duck_db = duck_db
        self._duck_target_gain = float(10 ** (duck_db / 20))
        duck_range = 1.0 - self._duck_target_gain
        self._attack_step = duck_range / self._attack_frames
        self._release_step = duck_range / self._release_frames

    def start(self) -> None:
        self._stream = sd.OutputStream(
            samplerate=self._samplerate,
            channels=self._channels,
            blocksize=self._blocksize,
            dtype="float32",
            device=self._device,
            callback=self._callback,
        )
        self._stream.start()
        logger.info("audio mixer started (device=%s)", self._device or "default")

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    # --- 発話ゲート（director.py から呼ぶ） ------------------------------

    def begin_finish(self) -> int | None:
        """今の台本だけ最後まで流し、次のトークは捨てるモードへ。対象の台本番号を返す。

        「今の台本」は最後に再生を始めた行の台本。コールバックと同じロックの中で
        決めるので、判定の直後に次のトークの1行目が滑り込むことはない。
        """
        with self._gate_lock:
            self._gate = GATE_FINISH
            self._gate_seq = self._last_seq
            return self._gate_seq

    def hold(self) -> None:
        with self._gate_lock:
            self._gate = GATE_HOLD

    def open(self) -> None:
        with self._gate_lock:
            self._gate = GATE_OPEN
            self._gate_seq = None

    def flush(self) -> int:
        """再生中の行・待ち行・優先レーンをすべて捨てる。捨てた行数を返す。"""
        with self._gate_lock:
            n = len(self._priority)
            self._priority.clear()
            while True:
                try:
                    self._speech_queue.get_nowait()
                except queue.Empty:
                    break
                n += 1
            if self._current_speech is not None:
                n += 1
            self._current_speech = None
            self._current_speech_pos = 0
            self._current_speaker = None
            self._current_is_priority = False
            return n

    def push_priority(self, items: list) -> None:
        """(ScriptLine, is_filler, pcm) を優先レーンへ。ゲートに関係なく先に流れる。"""
        with self._gate_lock:
            self._priority.extend(items)

    def is_idle(self) -> bool:
        """今は何も流しておらず、ゲートを通る待ち行も無いか。"""
        with self._gate_lock:
            if self._current_speech is not None or self._priority:
                return False
            return self._gate == GATE_HOLD or self._speech_queue.empty()

    def priority_idle(self) -> bool:
        """優先レーンを流し終えたか（HOLD 中に受けアナウンスの完了を待つ用）。"""
        with self._gate_lock:
            return not self._priority and not (
                self._current_speech is not None and self._current_is_priority
            )

    def _next_item_locked(self) -> bool:
        """ゲートに従って次の行を再生中にする。流すものが無ければ False。"""
        if self._priority:
            item = self._priority.popleft()
            is_priority = True
        elif self._gate == GATE_HOLD:
            return False
        else:
            is_priority = False
            while True:
                try:
                    item = self._speech_queue.get_nowait()
                except queue.Empty:
                    return False
                line = item[0]
                if (
                    self._gate == GATE_FINISH
                    and line.script_seq != self._gate_seq
                    and not line.closing
                ):
                    continue  # 次のトーク。切り替え待ちなので流さない
                break
        line, is_filler, pcm = item
        self._current_speech = pcm
        self._current_speech_pos = 0
        self._current_speaker = line.speaker
        self._current_is_filler = is_filler
        self._current_is_priority = is_priority
        if not is_priority and line.script_seq is not None:
            self._last_seq = line.script_seq
        self._announce(line, is_filler)
        return True

    def _has_pending_speech(self) -> bool:
        if self._priority:
            return True
        return self._gate == GATE_OPEN and not self._speech_queue.empty()

    def _announce(self, line: ScriptLine, is_filler: bool) -> None:
        """再生開始した行を字幕・ログ担当スレッドへ渡す（コールバックからは put_nowait のみ）。"""
        if self._announce_queue is None:
            return
        try:
            self._announce_queue.put_nowait((line, is_filler))
        except queue.Full:
            pass

    def _pull_speech(self, n: int) -> tuple[np.ndarray, bool, str | None, bool]:
        # 1ブロックぶんの配列コピーだけなのでロックはごく短い。ここを丸ごと囲むと
        # 別スレッドの flush() / is_idle() と「行の取り出し〜再生中にする」が食い違わない。
        with self._gate_lock:
            return self._pull_speech_locked(n)

    def _pull_speech_locked(self, n: int) -> tuple[np.ndarray, bool, str | None, bool]:
        out = np.zeros((n, self._channels), dtype=np.float32)
        filled = 0
        speaking = False
        speaker: str | None = None
        is_filler = False

        while filled < n:
            if self._current_speech is None and not self._next_item_locked():
                break

            remaining = len(self._current_speech) - self._current_speech_pos
            take = min(remaining, n - filled)
            out[filled : filled + take] = self._current_speech[
                self._current_speech_pos : self._current_speech_pos + take
            ]
            self._current_speech_pos += take
            filled += take
            speaking = True
            speaker = self._current_speaker
            is_filler = self._current_is_filler

            if self._current_speech_pos >= len(self._current_speech):
                self._current_speech = None
                self._current_speech_pos = 0
                self._current_speaker = None
                self._current_is_priority = False

        return out, speaking, speaker, is_filler

    def _duck_gains(self, frames: int, want_duck: bool) -> np.ndarray:
        target = self._duck_target_gain if want_duck else 1.0
        gains = np.empty(frames, dtype=np.float32)
        g = self._music_gain
        for i in range(frames):
            if target < g:
                g = max(target, g - self._attack_step)
                self._hold_counter = self._hold_frames
            elif target > g:
                if self._hold_counter > 0:
                    self._hold_counter -= 1
                else:
                    g = min(target, g + self._release_step)
            gains[i] = g
        self._music_gain = g
        return gains

    def _startup_gain_now(self) -> float:
        """起動フェードインの現在ゲイン（0.0〜1.0）。完了後は毎回 1.0 を軽く返す。"""
        if self._startup_fade_done:
            return 1.0
        started_at = self._state.startup_fade_started_at
        if started_at is None:
            return 0.0  # 画面（or main.py）がまだフェード開始時刻を立てていない＝無音で待つ
        elapsed = time.monotonic() - started_at
        if elapsed >= _STARTUP_FADE_SEC:
            self._startup_fade_done = True
            return 1.0
        return max(0.0, elapsed / _STARTUP_FADE_SEC)

    def _callback(self, outdata, frames, time_info, status) -> None:
        if status:
            logger.warning("sounddevice status: %s", status)

        speech, speaking, speaker, is_filler = self._pull_speech(frames)

        # 読書コーナー（§10.4）：朗読中は BGM をさらに落とす。AnnounceThread が
        # 朗読チャンクの再生開始/終了に合わせて duck_db_override を出し入れする。
        want_db = self._state.duck_db_override
        if want_db is None:
            want_db = self._base_duck_db
        if want_db != self._active_duck_db:
            self._set_duck_db(want_db)

        # 先読みダッキング：次の行が既にキューにあれば、このブロックで発話が
        # 始まっていなくても下げ始める（HOLD 中の作り置きでは下げない）。
        want_duck = speaking or self._has_pending_speech()
        gains = self._duck_gains(frames, want_duck)

        music, _underrun = self._music_ring.read(frames)
        mixed = music * gains[:, None] + speech
        np.clip(mixed, -1.0, 1.0, out=mixed)
        outdata[:] = mixed * self._startup_gain_now()

        rms = float(np.sqrt(np.mean(speech**2))) if speaking else 0.0
        self._state.current_rms = self._state.current_rms * 0.7 + rms * 0.3
        self._state.set_speaking(speaker)
        self._state.filler_active = speaking and is_filler

        block_has_sound = speaking or float(np.max(np.abs(music))) > 1e-4
        now = time.monotonic()
        if block_has_sound:
            self._ever_had_sound = True
            self._silence_since = None
            self._state.on_air = True
        elif self._ever_had_sound:
            # 起動直後（ffmpegの接続・デコード開始待ち）はまだ一度も音が出ていないので、
            # ここでの無音判定は「一度でも音が出た後」に限定する。
            if self._silence_since is None:
                self._silence_since = now
            silence_duration = now - self._silence_since
            if silence_duration > _SILENCE_LOG_THRESHOLD_SEC:
                if self._state.on_air:
                    logger.error("silence has continued for over %.0fs", _SILENCE_LOG_THRESHOLD_SEC)
                self._state.on_air = False
            # 猶予時間内はまだ on_air=True のまま（頻繁なランプ点滅を避ける）
