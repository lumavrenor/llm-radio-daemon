"""フィラー投入スレッド。SPEC.md 8章「無音を作らない」の第1段階。

script_queue が一定時間空のままだと、ScriptThread（LLM呼び出し）が詰まっているか
topic_queue にネタが来ていない状態。そのまま放っておくと TTSThread が空振りし続けて
トークの無い時間が伸びるので、フィラー原稿を script_queue に流し込んで場をつなぐ。

まず LLM に短い雑談を書かせ（generate_llm_filler）、失敗したら事前テンプレート
（generate_filler）へフォールバックする。LLM 呼び出しは最大 30 秒待つが、呼ばれるのは
「どのみち script_queue が空」の状況なので、その間の待ちは許容する。

コーナー切り替え中（director.py）はフィラーを出さない。生成の途中で切り替えが
始まったら結果は捨て、手が空いた時点で番組進行へ「止まった」と報告する。
"""

from __future__ import annotations

import logging
import queue
import random
import threading
import time
from collections import deque

from ..config import CastMember, ContentConfig, FillerConfig, LLMConfig
from ..director import ProgramDirector
from ..schedule import active_content
from ..weather import WeatherProvider
from ..script.filler import (
    generate_filler,
    generate_llm_filler,
    generate_llm_mid_song_chat,
    generate_llm_topic,
    generate_mid_song_chat,
    summarize_filler,
)
from ..script.ollama_client import pick_speakers
from ..state import SharedState
from ..script import Script

logger = logging.getLogger(__name__)

# 曲中のひとことを、曲が始まってから何秒後に入れるか（[[content]] で上書き可）。
_DEFAULT_MID_SONG_AFTER_SEC = 120.0

# 直近フィラーを何本ぶん覚えて次回プロンプトへ渡すか（反復を避けさせる参考情報）。
_RECENT_FILLER_MEMORY = 5

# 直近に使ったお題を何本ぶん覚えて再抽選から外すか。
_RECENT_TOPIC_MEMORY = 15

# フィラー1本にLLM生成のお題を仕込む確率。残りは従来どおり時刻・曲・楽屋ネタから
# 組む（お題ばかりだと今度はその型に飽きるため）。
_TOPIC_USE_PROBABILITY = 0.65


class FillerThread(threading.Thread):
    def __init__(
        self,
        script_queue: "queue.Queue[Script]",
        state: SharedState,
        content: list[ContentConfig],
        cast: list[CastMember],
        llm_config: LLMConfig,
        director: ProgramDirector,
        idle_threshold_sec: float = 20.0,
        check_interval_sec: float = 2.0,
        filler_config: FillerConfig | None = None,
        weather: WeatherProvider | None = None,
        stop_event: threading.Event | None = None,
    ):
        super().__init__(name="FillerThread", daemon=True)
        self._script_queue = script_queue
        self._state = state
        self._content = content
        self._cast = cast
        self._llm_config = llm_config
        self._director = director
        self._idle_threshold_sec = idle_threshold_sec
        self._check_interval_sec = check_interval_sec
        # お題の抽選は毎回ここから引き直す。実行場所は起動時に確定するが、天気は
        # 放送中に変わる（雨がやんだのに「@rain」のお題を引き続ける、を避ける）。
        self._filler_config = filler_config or FillerConfig()
        self._weather = weather
        self._stop_event = stop_event or threading.Event()
        self._idle_for = 0.0
        # 直近に流したフィラーの一行要約。次の generate_llm_filler へ渡して反復を避けさせる。
        self._recent_fillers: deque[str] = deque(maxlen=_RECENT_FILLER_MEMORY)
        # 直近に使ったお題。連続で同じお題を引かないよう再抽選から外す。
        self._recent_topics: deque[str] = deque(maxlen=_RECENT_TOPIC_MEMORY)
        # 曲中のひとこと用。「今どの曲か」「その曲でもう喋ったか」を1曲ぶんだけ覚える。
        self._song_title: str = ""
        self._song_started = time.monotonic()
        self._song_chatted = False

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.wait(self._check_interval_sec):
            try:
                self._tick()
            except Exception:
                logger.exception("filler thread crashed on tick; continuing")

    def _put(self, script: Script) -> bool:
        """放送が落ち着いているときだけ積む（生成中に切り替えが始まったら捨てる）。"""
        if not self._director.is_steady():
            logger.info("Discarding filler because a corner switch is in progress: %s", script.topic_title)
            return False
        self._script_queue.put_nowait(script)
        return True

    def _tick(self) -> None:
        if not self._director.is_steady():
            self._idle_for = 0.0
            self._director.ack_filler_parked(self._director.transition_id())
            return

        ac = active_content(self._content)
        if ac is not None and ac.type == "radio":
            # 音楽のみの区間（song_talk = "off"）ではフィラーを出さない。
            if ac.song_talk == "off":
                self._idle_for = 0.0
                return
            # song_talk = "back" は「いま終わった曲の振り返り」だけを曲間に流し、
            # それ以外は静かにしておく。通常のフィラーで場をつながない代わりに、
            # 曲の途中で軽いひとことを1回だけ挟む（曲の長さぶん無音が続くため）。
            if ac.song_talk == "back":
                self._idle_for = 0.0
                self._tick_mid_song_chat(ac)
                return
            # song_talk = "intro" は曲紹介トークの合間を通常のフィラーでつなぐ。

        # 読書コーナー中はフィラーを出さない（朗読の合間にひな壇の雑談が割り込むのを防ぐ）。
        # Beat の感想生成で script_queue が一時的に空でも、ここは待つ。
        if self._state.reading_now_playing or self._state.generated_drama_filler_hold:
            self._idle_for = 0.0
            return

        if not self._script_queue.empty():
            self._idle_for = 0.0
            return

        self._idle_for += self._check_interval_sec
        if self._idle_for < self._idle_threshold_sec:
            return

        a, b = self._pick_two()
        now_playing = self._state.now_playing or None
        # 天気は「取れていれば渡す」だけ。None なら台本側が天気に一切触れない。
        weather = self._weather.current() if self._weather is not None else None
        topic_seed = self._pick_topic()
        script = generate_llm_filler(
            self._llm_config, now_playing, a, b, list(self._recent_fillers),
            topic_seed, weather,
        )
        if script is None:
            script = generate_filler(
                now_playing, a.id, b.id,
                self._llm_config.model, self._llm_config.engine_label, topic_seed,
                self._llm_config.placement, weather,
            )
        try:
            if not self._put(script):
                self._idle_for = 0.0
                return
            self._recent_fillers.append(summarize_filler(script))
            if topic_seed is not None:
                self._recent_topics.append(topic_seed)
            logger.info("script_queue idle for %.0fs; injected filler (%s)", self._idle_for, script.topic_title)
        except queue.Full:
            pass
        self._idle_for = 0.0

    def _tick_mid_song_chat(self, content: ContentConfig) -> None:
        """song_talk = "back" の曲中に、軽いひとことを1曲につき1回だけ挟む。

        曲の長さは事前に分からないので「曲が変わってから N 秒後」で判定する
        （N = ``mid_song_chat_after_sec``）。N より短い曲では何も挟まずに終わる。
        """
        if not content.params.get("mid_song_chat", False):
            return

        song = self._state.now_playing
        if song != self._song_title:  # 曲が変わった。タイマーと「もう喋った」を仕切り直す
            self._song_title = song
            self._song_started = time.monotonic()
            self._song_chatted = False
        if self._song_chatted or not song:
            return

        after = float(
            content.params.get("mid_song_chat_after_sec", _DEFAULT_MID_SONG_AFTER_SEC)
        )
        elapsed = time.monotonic() - self._song_started
        if elapsed < after:
            return
        if not self._script_queue.empty():
            return  # 曲頭の振り返りがまだ残っている。重ねずに次の tick へ回す

        # 出演者は通常のトークと同じく [[content]] の min/max_speakers から抽選する。
        appearers = pick_speakers(content, self._cast)
        script = generate_llm_mid_song_chat(
            self._llm_config, appearers, content.tone_hint
        )
        if script is None:
            script = generate_mid_song_chat(appearers)
        try:
            if not self._put(script):
                return
        except queue.Full:
            return  # 次の tick で入れ直す（_song_chatted は立てない）
        self._song_chatted = True
        logger.info(
            "mid-song chat injected %.0fs into %r (%s, %d lines)",
            elapsed, song, script.topic_title, len(script.lines),
        )

    def _pick_topic(self) -> str | None:
        """今回のお題を1つ用意する。無ければ／確率を外せば None。

        台本を書くのと同じ LLM にお題そのものを考えさせる（ネタだし。
        ``generate_llm_topic``）。固定リストと違って有限にならず、お題の質もそのモデルの
        地力を映す。失敗／タイムアウトしたときは None（お題なしで進める）。

        直近 ``_RECENT_TOPIC_MEMORY`` 本で使ったお題は参考として渡し、同じ切り口の
        繰り返しを避けさせる。
        """
        if random.random() > _TOPIC_USE_PROBABILITY:
            return None
        return generate_llm_topic(self._llm_config, list(self._recent_topics))

    def _pick_two(self) -> tuple[CastMember, CastMember]:
        """フィラーの2人（A/B）を選ぶ。

        フィラーは ``is_filler`` なのでひな壇の顔ぶれ（``active_cast_ids``）を
        更新しない＝直前のトークの並びがそのまま残る。そこから外れた人を
        フィラーで喋らせると「画面にいないのに喋っている」状態になるため、
        いま並んでいる出演者の中から選ぶ。まだ誰も並んでいない起動直後などは
        ロースター全体から選ぶ（ロースターは config で1人以上が保証されている）。
        """
        active = [m for m in self._cast if m.id in self._state.active_cast_ids]
        pool = active if active else self._cast
        if len(pool) >= 2:
            a, b = random.sample(pool, 2)
            return a, b
        only = pool[0]
        return only, only
