"""番組進行（コーナー切り替えの状態機械）。

「今どのコーナーを放送しているか」（on-air）を決めるのはここだけ。番組表・リクエストが
示す「放送したいコーナー」（``schedule.scheduled_content()``）とは分けて持ち、両者が
食い違ったら、次の手順で1段ずつ確かめながら on-air を切り替える。各スレッド・画面は
``schedule.active_content()`` 経由で on-air を読むので、切り替わるのは暗転している間だけ。

    ON_AIR      通常放送。番組表の変化・新しいリクエストを見張る
      ↓ 変化あり
    DRAINING    いま流れているトークだけ最後まで流す（次のトークは捨てる）。前のコーナーの
                締め（読書コーナーの「続きはまた明日」など）もここで流す。ScriptThread /
                FillerThread が手を止め、TTS・ミキサーが空になり、ラジオドラマの裏執筆も
                止まるまで待つ
      ↓ 全部止まった
    FADING_OUT  画面を暗転させ、暗くなりきるのを待つ
      ↓ （ここで一気に入れ替える）作り置きの破棄・左上のコーナー名・字幕の消去・キャスト交換
    PREPARING   暗いまま、次のコーナーの準備を待つ（局の切り替え・最初のトークの合成、
                リクエストなら受けアナウンスの生成と合成）。ミキサーは HOLD で、合成済みの
                トークは流さずに溜めておく
      ↓ 整った
    FADING_IN   明転し、明るくなりきるのを待つ
      ↓
    ANNOUNCING  （リクエストのときだけ）「リクエストをいただきました」を流し切るのを待つ
      ↓
    ON_AIR      溜めておいた次のコーナーのトークから流し始める

どの待ちにも上限があり、超えたら警告を出して先へ進む（「放送が止まることだけが異常」）。
"""

from __future__ import annotations

import enum
import logging
import queue
import threading
import time
from typing import TYPE_CHECKING

from . import schedule
from .config import CastMember, ContentConfig, LLMConfig
from .db import TopicStore
from .script import Script
from .script.ollama_client import pick_speakers
from .script.request_announce import generate_request_announce
from .sources import Topic
from .state import SharedState
from .threads.tts_thread import synthesize_line
from .tts import TTSBackend

if TYPE_CHECKING:
    from .audio.mixer import AudioMixer
    from .threads.station_thread import StationHolder

logger = logging.getLogger(__name__)

_SCHEDULE_CHECK_SEC = 0.5
# 今のトーク（読書コーナーのひとまとまり等は数分ある）＋締め＋裏の LLM 呼び出しの完走まで。
_DRAIN_TIMEOUT_SEC = 300.0
_FADE_ACK_TIMEOUT_SEC = 5.0
# 準備が一瞬で整っても、暗転がチラつきに見えないよう最低これだけは暗いままにする。
_MIN_DARK_SEC = 0.8
_PREPARE_TIMEOUT_SEC = 90.0
# 受けアナウンスの話者は「明転後にひな壇に立っている人」から選ぶ。読書系コーナーは
# 担当の抽選が準備の途中で決まるので、それを待つ上限。
_STAGE_WAIT_SEC = 20.0
_ANNOUNCE_TIMEOUT_SEC = 90.0
_WAIT_LOG_INTERVAL_SEC = 15.0

# 出演者を自前で抽選するコーナー。それ以外のトーク系は切り替え時にここで抽選する。
_CORNER_TYPES = frozenset(
    {"literary_reading", "translated_reading", "biography_reading", "generated_drama"}
)


class Phase(enum.Enum):
    ON_AIR = "on_air"
    DRAINING = "draining"
    FADING_OUT = "fading_out"
    PREPARING = "preparing"
    FADING_IN = "fading_in"
    ANNOUNCING = "announcing"


def topic_belongs_to(topic: Topic, content: ContentConfig | None) -> bool:
    """このネタは content のコーナーで話すものか。radio の曲ネタは musicbrainz 由来。"""
    if content is None:
        return False
    if content.type == "radio":
        return topic.source == "musicbrainz"
    return topic.source == content.type


def _label(content: ContentConfig | None) -> str:
    return content.display_label if content is not None else "(none)"


class ProgramDirector:
    def __init__(
        self,
        content: list[ContentConfig],
        cast: list[CastMember],
        llm_config: LLMConfig,
        state: SharedState,
        store: TopicStore,
        mixer: "AudioMixer",
        script_queue: "queue.Queue[Script]",
        topic_queue: "queue.Queue[tuple[int, Topic]]",
        speech_queue: queue.Queue,
        tts_backend: TTSBackend,
        samplerate: int,
        has_filler: bool,
    ):
        self._content = content
        self._cast = cast
        self._cast_by_id = {m.id: m for m in cast}
        self._llm_config = llm_config
        self._state = state
        self._store = store
        self._mixer = mixer
        self._script_queue = script_queue
        self._topic_queue = topic_queue
        self._speech_queue = speech_queue
        self._tts_backend = tts_backend
        self._samplerate = samplerate
        self._has_filler = has_filler
        self._station: "StationHolder | None" = None

        self._lock = threading.Lock()
        self._phase = Phase.ON_AIR
        self._phase_since = time.monotonic()
        self._tid = 0  # 切り替えの通し番号。各スレッドの「止まった」報告の宛先
        self._target: ContentConfig | None = None
        self._request: dict | None = None
        self._drain_seq: int | None = None
        self._transition_started = 0.0

        self._script_parked_tid = 0
        self._filler_parked_tid = 0
        self._nothing_tid = 0
        self._tts_busy = False

        self._display_attached = False
        self._fade: tuple[str, int] | None = None
        self._fade_acked: tuple[str, int] | None = None

        self._announce_started = False
        self._announce_items: list | None = None

        self._last_schedule_check = 0.0
        self._last_wait_log = 0.0

        self._accept_request_at_startup()
        self._on_air = schedule.scheduled_content(content)
        self._session_cast = self._pick_session_cast(self._on_air)
        if self._session_cast:
            state.active_cast_ids = tuple(m.id for m in self._session_cast)
        logger.info(
            "on-air at startup: %s%s",
            _label(self._on_air),
            f" (cast: {', '.join(state.active_cast_ids)})" if self._session_cast else "",
        )

    # --- 組み立て --------------------------------------------------------

    def bind_station(self, station: "StationHolder") -> None:
        self._station = station

    def attach_display(self) -> None:
        """暗転・明転を実際に描く画面がある（Ursina）。無ければ暗転は即完了扱い。"""
        self._display_attached = True

    # --- 各スレッド・画面から読む ------------------------------------------

    @property
    def phase(self) -> Phase:
        return self._phase

    def on_air_content(self) -> ContentConfig | None:
        return self._on_air

    def transition_id(self) -> int:
        return self._tid

    def is_steady(self) -> bool:
        return self._phase is Phase.ON_AIR

    def audio_idle(self) -> bool:
        """今、音声パイプラインに何も流れておらず、待ち行列も無いか（コーナー内ローカル暗転用）。"""
        return self._mixer.is_idle()

    @property
    def display_attached(self) -> bool:
        return self._display_attached

    def in_closing_window(self) -> bool:
        """前のコーナーを畳んでいる最中（ScriptThread は締めを積んだら手を止める）。"""
        return self._phase in (Phase.DRAINING, Phase.FADING_OUT)

    def is_switching(self) -> bool:
        """暗転しきってから明転が始まるまで（画面は真っ黒で中身を入れ替えている最中）。

        画面はこの間だけ通常表示の代わりに「切り替え中」の演出を出してよい
        （どのみち camera.overlay の不透明な黒に隠れて見えないので、他のスレッド・
        画面の状態が多少入れ替わり中でも構わない）。
        """
        return self._phase is Phase.PREPARING

    def accepts_script(self, closing: bool) -> bool:
        """今 script_queue へ積んでよい台本か。切り替え待ちの間は締めだけ。"""
        ph = self._phase
        if ph is Phase.DRAINING:
            return closing
        if ph is Phase.FADING_OUT:
            return False
        return not closing

    def collecting(self, content_type: str) -> bool:
        """SourceThread がネタを集めてよいか（on-air のコーナーのものだけ、畳んでいる間は止める）。"""
        ac = self._on_air
        return (
            not self.in_closing_window()
            and ac is not None
            and ac.type == content_type
        )

    def session_cast(self, content: ContentConfig) -> list[CastMember] | None:
        """切り替え時に抽選した、このコーナーの出演者（ScriptThread が使い回す）。"""
        with self._lock:
            if content is self._on_air and self._session_cast:
                return list(self._session_cast)
        return None

    def fade_command(self) -> tuple[str, int] | None:
        """画面へのフェード指示 ("out" | "in", 切り替え番号)。無ければ None。"""
        return self._fade

    # --- 各スレッドからの報告 ------------------------------------------------

    def ack_script_parked(self, tid: int) -> None:
        self._script_parked_tid = tid

    def ack_filler_parked(self, tid: int) -> None:
        self._filler_parked_tid = tid

    def report_nothing_to_air(self, tid: int) -> None:
        """次のコーナーに今すぐ流せるものが無い（在庫切れ・ネタ待ち）。準備完了として扱う。"""
        if self._nothing_tid != tid:
            logger.info("Nothing to air right now for the next corner. Cutting preparation short and fading in.")
        self._nothing_tid = tid

    def ack_fade(self, kind: str, tid: int) -> None:
        with self._lock:
            if self._fade == (kind, tid):
                self._fade_acked = (kind, tid)

    # --- TTSThread ---------------------------------------------------------

    def tts_take(self, script_queue: "queue.Queue[Script]") -> Script | None:
        """台本を1本取り出して「合成中」にする。キューが空なら None。

        取り出しと「合成中」の印を同じロックの中で行う。こうしないと、取り出した直後の
        一瞬に「キューは空・TTS は暇」と見えて、切り替え待ちが早合点で明けてしまう。
        """
        with self._lock:
            try:
                script = script_queue.get_nowait()
            except queue.Empty:
                return None
            self._tts_busy = True
            return script

    def tts_done(self) -> None:
        with self._lock:
            self._tts_busy = False

    def tts_should_run(self, script: Script) -> bool:
        """この台本を（続けて）合成してよいか。行の合間ごとに聞かれる。"""
        ph = self._phase
        if ph is Phase.DRAINING:
            return script.is_closing or (
                self._drain_seq is not None and script.seq == self._drain_seq
            )
        if ph is Phase.FADING_OUT:
            return False
        return not script.is_closing

    # --- 状態機械 -----------------------------------------------------------

    def tick(self) -> None:
        ph = self._phase
        if ph is Phase.ON_AIR:
            self._tick_on_air()
        elif ph is Phase.DRAINING:
            self._tick_draining()
        elif ph is Phase.FADING_OUT:
            self._tick_fading_out()
        elif ph is Phase.PREPARING:
            self._tick_preparing()
        elif ph is Phase.FADING_IN:
            self._tick_fading_in()
        elif ph is Phase.ANNOUNCING:
            self._tick_announcing()

    def _set_phase_locked(self, phase: Phase) -> None:
        self._phase = phase
        self._phase_since = time.monotonic()
        self._last_wait_log = 0.0

    def _elapsed(self) -> float:
        return time.monotonic() - self._phase_since

    def _log_waiting(self, what: str, waiting: list[str]) -> None:
        now = time.monotonic()
        if now - self._last_wait_log < _WAIT_LOG_INTERVAL_SEC:
            return
        self._last_wait_log = now
        logger.info("%s (%.0f sec elapsed): waiting on %s", what, self._elapsed(), ", ".join(waiting))

    # ON_AIR -----------------------------------------------------------------

    def _tick_on_air(self) -> None:
        now = time.monotonic()
        if now - self._last_schedule_check < _SCHEDULE_CHECK_SEC:
            return
        self._last_schedule_check = now

        # 番組表を先に読み、リクエストはその後で DB から直接読む（逆順だと、番組表の
        # キャッシュが先にリクエストを拾い、受けアナウンス抜きで切り替わる隙ができる）。
        target = schedule.scheduled_content(self._content)
        req = self._take_new_request()
        if req is not None:
            requested = self._request_content(req)
            if requested is None:
                return
            if requested is self._on_air:
                logger.info(
                    "Request (%s) is the corner currently on air. Continuing without switching (%.0f min remaining)",
                    requested.display_label,
                    max(0.0, req["expires_at"] - time.time()) / 60.0,
                )
                return
            self._begin(requested, req, "request")
            return
        if target is not self._on_air:
            self._begin(target, None, "schedule")

    def _accept_request_at_startup(self) -> None:
        """起動時点でまだ受けていないリクエストは、読み上げずにそのまま有効にする。"""
        try:
            req = self._store.pending_request_announcement()
            if req is not None:
                self._store.mark_request_announced(req["id"])
                schedule.invalidate_request_cache()
                logger.info("Marking the startup-time request as accepted: %s", req["label"])
        except Exception:
            logger.exception("Failed to check the request at startup. Starting according to the schedule.")

    def _take_new_request(self) -> dict | None:
        """新しく届いたリクエストを受け付け済みにして返す（1本につき1回だけ）。

        受け付け済み（announced_at あり）になった瞬間から ``scheduled_content()`` が
        リクエスト先を返すようになる（db.py active_request_target）。
        """
        try:
            req = self._store.pending_request_announcement()
            if req is None:
                return None
            self._store.mark_request_announced(req["id"])
        except Exception:
            logger.exception("Failed to check the request. Continuing the broadcast.")
            return None
        schedule.invalidate_request_cache()
        return req

    def _request_content(self, req: dict) -> ContentConfig | None:
        idx = req["content_index"]
        if 0 <= idx < len(self._content) and self._content[idx].type == req["content_type"]:
            return self._content[idx]
        logger.warning(
            "Request (#%d %s) does not match the current [[content]]. Ignoring it.", idx, req["content_type"]
        )
        return None

    def _begin(self, target: ContentConfig | None, req: dict | None, reason: str) -> None:
        # 先にミキサーを「今の台本だけ流し切る」へ。どの台本が「今の」かはミキサーが
        # 自分のロックの中で決める。
        drain_seq = self._mixer.begin_finish()
        with self._lock:
            self._tid += 1
            self._target = target
            self._request = req
            self._drain_seq = drain_seq
            self._announce_started = False
            self._announce_items = None
            self._fade = None
            self._fade_acked = None
            self._transition_started = time.monotonic()
            self._set_phase_locked(Phase.DRAINING)
        logger.info(
            "Starting corner switch (%s): %s -> %s. Finishing the current talk before switching.",
            reason, _label(self._on_air), _label(target),
        )

    # DRAINING ---------------------------------------------------------------

    def _tick_draining(self) -> None:
        tid = self._tid
        waiting: list[str] = []
        if self._script_parked_tid != tid:
            waiting.append("ScriptThread stopping")
        if self._has_filler and self._filler_parked_tid != tid:
            waiting.append("FillerThread stopping")
        with self._lock:
            tts_idle = not self._tts_busy and self._script_queue.empty()
        if not tts_idle:
            waiting.append("TTS synthesis")
        if not self._mixer.is_idle():
            waiting.append("talk currently playing")
        if self._state.generated_drama_writing:
            waiting.append("radio drama writing stopping")

        if waiting:
            if self._elapsed() < _DRAIN_TIMEOUT_SEC:
                self._log_waiting("waiting for the previous corner to stop", waiting)
                return
            logger.warning(
                "The previous corner still hasn't fully stopped after %.0f sec (%s). Cutting it off and switching.",
                self._elapsed(), ", ".join(waiting),
            )
            self._mixer.flush()
        else:
            logger.info("Everything in the previous corner has stopped (%.1f sec). Fading out.", self._elapsed())
        self._mixer.hold()
        self._start_fade("out", Phase.FADING_OUT)

    # FADING_OUT / FADING_IN ------------------------------------------------

    def _start_fade(self, kind: str, phase: Phase) -> None:
        with self._lock:
            self._fade = (kind, self._tid)
            self._fade_acked = None
            self._set_phase_locked(phase)

    def _fade_done(self) -> bool:
        if not self._display_attached or self._fade_acked == self._fade:
            return True
        if self._elapsed() >= _FADE_ACK_TIMEOUT_SEC:
            logger.warning("No report of fade completion from the screen. Proceeding without waiting.")
            return True
        return False

    def _tick_fading_out(self) -> None:
        if self._fade_done():
            self._swap()

    def _swap(self) -> None:
        """暗転しきった状態で、前のコーナーから次のコーナーへ一気に入れ替える。"""
        target = self._target
        dropped_lines = self._mixer.flush()
        dropped_scripts = self._drain_script_queue()
        dropped_topics = self._purge_topics(target)

        st = self._state
        st.subtitle = ""
        # 読書系コーナーは締めの時点で自分で片付けているが、打ち切りで進んだ場合に
        # 前のコーナーの表示が残らないよう、ここでも消す。
        st.reading_work_id = None
        st.reading_now_playing = ""
        st.reading_progress = ""
        st.reading_cast_ids = ()
        st.duck_db_override = None

        session = self._pick_session_cast(target)
        # 読書系・ラジオドラマは準備の途中で各コーナーが自分の出演者を出す。
        st.active_cast_ids = tuple(m.id for m in session) if session else ()
        with self._lock:
            self._on_air = target
            self._session_cast = session
            self._set_phase_locked(Phase.PREPARING)
        logger.info(
            "Switching during fade-out: CONTENT=%s cast=%s (discarded stock: %d scripts, %d audio lines, %d topics)",
            _label(target),
            ", ".join(st.active_cast_ids) or "(decided by the corner itself)",
            dropped_scripts, dropped_lines, dropped_topics,
        )

    def _pick_session_cast(self, content: ContentConfig | None) -> list[CastMember] | None:
        if content is None or not content.is_talk or content.type in _CORNER_TYPES:
            return None
        return pick_speakers(content, self._cast)

    def _drain_script_queue(self) -> int:
        n = 0
        while True:
            try:
                self._script_queue.get_nowait()
            except queue.Empty:
                return n
            n += 1

    def _purge_topics(self, target: ContentConfig | None) -> int:
        """topic_queue から次のコーナーのネタ以外を捨てる。"""
        kept: list = []
        dropped = 0
        while True:
            try:
                item = self._topic_queue.get_nowait()
            except queue.Empty:
                break
            if topic_belongs_to(item[1], target):
                kept.append(item)
            else:
                dropped += 1
        for item in kept:
            try:
                self._topic_queue.put_nowait(item)
            except queue.Full:
                break
        return dropped

    # PREPARING --------------------------------------------------------------

    def _tick_preparing(self) -> None:
        ac = self._on_air
        elapsed = self._elapsed()
        station_ok = self._station is None or self._station.tuned_content is ac
        content_ok = self._content_ready(ac)

        if self._request is not None and not self._announce_started:
            if self._stage_known() or content_ok or elapsed >= _STAGE_WAIT_SEC:
                self._start_announcement()
        announce_ok = self._request is None or self._announce_items is not None

        waiting: list[str] = []
        if not station_ok:
            waiting.append("station switch")
        if not content_ok:
            waiting.append("synthesis of the first talk")
        if not announce_ok:
            waiting.append("synthesis of the request-received announcement")

        if waiting or elapsed < _MIN_DARK_SEC:
            if elapsed < _PREPARE_TIMEOUT_SEC:
                if waiting:
                    self._log_waiting("waiting for the next corner to be ready", waiting)
                return
            logger.warning(
                "The next corner isn't ready after %.0f sec (%s). Fading in anyway.",
                elapsed, ", ".join(waiting),
            )
            if not announce_ok:
                self._give_up_announcement()
        else:
            logger.info("The next corner is ready (%.1f sec). Fading in.", elapsed)
        self._start_fade("in", Phase.FADING_IN)

    def _content_ready(self, ac: ContentConfig | None) -> bool:
        # radio は曲の切れ目で喋るので、最初のトークを待たない（局が繋がれば始められる）。
        if ac is None or ac.type == "radio":
            return True
        if not self._speech_queue.empty():
            return True  # 最初の行が合成済み（ミキサーは HOLD で待たせている）
        return self._nothing_tid == self._tid

    def _stage_known(self) -> bool:
        return bool(self._state.reading_cast_ids or self._state.active_cast_ids)

    def _stage_members(self, content: ContentConfig | None) -> list[CastMember]:
        ids = self._state.reading_cast_ids or self._state.active_cast_ids
        members = [self._cast_by_id[i] for i in ids if i in self._cast_by_id]
        if members:
            return members
        return pick_speakers(content, self._cast) if content is not None else []

    def _start_announcement(self) -> None:
        self._announce_started = True
        content = self._on_air
        req = self._request
        if content is None or req is None:
            self._announce_items = []
            return
        members = self._stage_members(content)
        threading.Thread(
            target=self._build_announcement,
            args=(self._tid, content, req, members),
            name="RequestAnnounce",
            daemon=True,
        ).start()

    def _build_announcement(
        self, tid: int, content: ContentConfig, req: dict, members: list[CastMember]
    ) -> None:
        """受けアナウンスを生成・合成して優先レーン用の行にする（明転後に流す）。"""
        items: list = []
        try:
            script = generate_request_announce(self._llm_config, members, content, req["id"])
        except Exception:
            logger.exception("Generating the request-received announcement crashed. Entering the corner in silence.")
            script = None
        if script is not None:
            # ひな壇はもう次のコーナーの顔ぶれなので talk_cast_ids は貼らない（キャストを動かさない）。
            for line in script.lines:
                line.request_id = None
                pcm = synthesize_line(self._tts_backend, self._cast_by_id, self._samplerate, line)
                if pcm is not None:
                    items.append((line, False, pcm))
        if items:
            # 最後に流れる行で entered を記録する（AnnounceThread）。
            items[-1][0].request_id = req["id"]
        else:
            self._mark_entered(req["id"])
        with self._lock:
            # 待ちきれずに諦めた後（_give_up_announcement）に届いた分は使わない。
            if self._tid != tid or self._announce_items is not None:
                return
            self._announce_items = items
        logger.info("Prepared the request-received announcement (%d lines)", len(items))

    def _give_up_announcement(self) -> None:
        with self._lock:
            req = self._request
            self._announce_items = []
        if req is not None:
            self._mark_entered(req["id"])

    def _mark_entered(self, request_id: int) -> None:
        try:
            self._store.mark_request_entered(request_id)
        except Exception:
            logger.exception("Failed to update the request's entered status")

    # FADING_IN / ANNOUNCING ------------------------------------------------

    def _tick_fading_in(self) -> None:
        if not self._fade_done():
            return
        items = self._announce_items if self._request is not None else None
        if items:
            self._mixer.push_priority(items)
            with self._lock:
                self._set_phase_locked(Phase.ANNOUNCING)
            logger.info("Playing the request-received announcement")
            return
        self._go_on_air()

    def _tick_announcing(self) -> None:
        if self._mixer.priority_idle():
            self._go_on_air()
        elif self._elapsed() >= _ANNOUNCE_TIMEOUT_SEC:
            logger.warning("The received-request announcement hasn't finished after %.0f sec. Starting the corner.", self._elapsed())
            self._go_on_air()

    def _go_on_air(self) -> None:
        self._mixer.open()
        with self._lock:
            self._fade = None
            self._fade_acked = None
            self._request = None
            self._target = None
            self._set_phase_locked(Phase.ON_AIR)
        self._last_schedule_check = 0.0
        logger.info(
            "Corner switch complete: %s (%.1f sec total)",
            _label(self._on_air), time.monotonic() - self._transition_started,
        )
