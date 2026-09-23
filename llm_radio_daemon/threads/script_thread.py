"""原稿生成スレッド。topic_queue → Ollama → script_queue。

全履歴は渡さず、直前2〜3トピックの一行要約だけをコンテキストとして使う
（24時間目にコンテキストが破裂しないように）。

コーナー切り替え（director.py）: 番組進行が前のコーナーを畳み始めたら、今いる
コーナーの締め（読書コーナーの「続きはまた明日」など）を積んで手を止め、
番組進行へ「止まった」と報告する。LLM 呼び出しの途中なら、それが返ってきてから
（結果は捨てて）報告する ―― 番組進行はこの報告を待ってから暗転する。
"""

from __future__ import annotations

import collections
import json
import logging
import queue
import threading
import time

from ..biography_reading.corner import BiographyReadingCorner
from ..config import CastMember, ContentConfig, LLMConfig
from ..db import TopicStore
from ..director import Phase, ProgramDirector, topic_belongs_to
from ..generated_drama.corner import GeneratedDramaCorner
from ..literary_reading.corner import LiteraryReadingCorner
from ..mood import CastMoodProvider
from ..script import Script
from ..script.ollama_client import generate_script, pick_speakers
from ..sources import Topic
from ..translated_reading.corner import TranslatedReadingCorner

logger = logging.getLogger(__name__)

_RECENT_HISTORY = 3
_PARK_POLL_SEC = 0.2
# 切り替えの準備中、次のコーナーのネタがこれだけ届かなければ「今は流せるものが無い」と
# 番組進行へ報告する（暗いまま待たせ続けない。明転後はフィラーが場をつなぐ）。
_PREPARE_STARVE_SEC = 30.0


def _format_comparison(topic: Topic, script: Script, names: dict[str, str]) -> str:
    """元ネタ（ネットニュース等の原文）と LLM が作った台本を並べてログに出す。

    後で「元ネタ→生成物」でどう化けたかを眺めて比較するための出力。
    複数スレッドが同時に書いても1レコードとしてまとまるよう、
    1回の logger 呼び出しで複数行を出す。
    """
    generated = "\n".join(
        f"  [{names.get(line.speaker, line.speaker)}"
        f"{'/' + line.style if line.style else ''}] {line.text}"
        for line in script.lines
    )
    return (
        "\n===== 元ネタ vs 生成台本 =====\n"
        f"[source] {topic.source}  (扱い: {topic.hint})\n"
        f"[title ] {topic.title}\n"
        f"[url   ] {topic.url or '(なし)'}\n"
        "----- 元ネタ本文 -----\n"
        f"{topic.body.strip()}\n"
        "----- 生成台本 -----\n"
        f"{generated}\n"
        "============================="
    )


class ScriptThread(threading.Thread):
    def __init__(
        self,
        topic_queue: "queue.Queue[tuple[int, Topic]]",
        script_queue: "queue.Queue[Script]",
        store: TopicStore,
        llm_config: LLMConfig,
        cast: list[CastMember],
        director: ProgramDirector,
        stop_event: threading.Event | None = None,
        literary_reading_corner: LiteraryReadingCorner | None = None,
        translated_reading_corner: TranslatedReadingCorner | None = None,
        biography_reading_corner: BiographyReadingCorner | None = None,
        generated_drama_corner: GeneratedDramaCorner | None = None,
        mood_provider: CastMoodProvider | None = None,
    ):
        super().__init__(name="ScriptThread", daemon=True)
        self._topic_queue = topic_queue
        self._script_queue = script_queue
        self._store = store
        self._llm_config = llm_config
        self._cast = cast
        self._director = director
        # 日替わりの「今日のキャラごとの気分」（持ちネタキャラバリエーション §2）。
        # 既定 off。[llm] daily_mood = true のときだけ main.py が渡す（それ以外は None）。
        self._mood_provider = mood_provider
        self._names = {m.id: m.name for m in cast}
        self._stop_event = stop_event or threading.Event()
        self._recent_summaries: collections.deque = collections.deque(maxlen=_RECENT_HISTORY)
        # 自前でセッションを持つコーナー（読書 v4 §10.7 / 翻訳朗読 / 偉人伝 / ラジオドラマ v6 §4.7）。
        # ウォッチドッグでこのスレッドが作り直されてもセッション状態を失わないよう、
        # インスタンスは main.py 側で1つずつ作って渡す。
        self._corners = {
            ctype: corner
            for ctype, corner in (
                ("literary_reading", literary_reading_corner),
                ("translated_reading", translated_reading_corner),
                ("biography_reading", biography_reading_corner),
                ("generated_drama", generated_drama_corner),
            )
            if corner is not None
        }
        self._in_corner = None  # 今 produce_batch を回しているコーナー
        # トピック型コンテンツ（arxiv/rss/…）のキャストを、そのコンテンツ区間が
        # 続く間は固定するためのキャッシュ（読書系コーナーは各コーナーが自前で
        # セッション開始時に抽選・固定しているのでここでは扱わない）。
        self._session_content: ContentConfig | None = None
        self._session_cast: list[CastMember] | None = None
        self._parked_tid = 0  # 切り替え番号は1から。作り直された直後でも報告は必ず出す
        self._topic_wait_since: float | None = None

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            if self._director.in_closing_window():
                self._park()
                self._stop_event.wait(_PARK_POLL_SEC)
                continue

            ac = self._director.on_air_content()
            if ac is not self._session_content:
                self._session_content = None
                self._session_cast = None

            corner = self._corner_for(ac)
            if self._in_corner is not None and self._in_corner is not corner:
                # 番組進行の段取りを経ずにコーナーを抜けた（打ち切りで進んだ・コーナーが
                # 自分で無効化した）。締めを流す場面ではないので状態だけ降ろす。
                self._end_corner(emit_outro=False)
            if corner is not None:
                self._in_corner = corner
                self._tick_corner(corner, ac.type)
                continue

            if ac is not None and not ac.is_talk:
                # 音楽のみの区間（radio の song_talk = "off"）。トークは生成しない
                # （フィラーも FillerThread 側で抑止）。
                self._stop_event.wait(1.0)
                continue

            try:
                topic_id, topic = self._topic_queue.get(timeout=1.0)
            except queue.Empty:
                self._maybe_report_starved()
                continue
            self._topic_wait_since = None
            if not topic_belongs_to(topic, ac):
                logger.info(
                    "topic %r (%s) isn't material for the current corner (%s); discarding",
                    topic.title, topic.source, ac.type if ac else "(none)",
                )
                continue
            try:
                self._handle_topic(topic_id, topic, ac)
            except Exception:
                logger.exception("script generation crashed for topic %r", topic.title)

    # --- コーナー切り替え（director.py） -----------------------------------

    def _park(self) -> None:
        """今のコーナーの締めを積み、手を止めたことを番組進行へ報告する（切り替え1回につき1度）。"""
        tid = self._director.transition_id()
        if self._parked_tid == tid:
            return
        if self._in_corner is not None:
            self._end_corner(emit_outro=True)
        self._session_content = None
        self._session_cast = None
        self._topic_wait_since = None
        self._parked_tid = tid
        self._director.ack_script_parked(tid)
        logger.info("ScriptThread: stopped script generation for the previous corner")

    def _maybe_report_starved(self) -> None:
        if self._director.phase is not Phase.PREPARING:
            self._topic_wait_since = None
            return
        now = time.monotonic()
        if self._topic_wait_since is None:
            self._topic_wait_since = now
        elif now - self._topic_wait_since >= _PREPARE_STARVE_SEC:
            self._director.report_nothing_to_air(self._director.transition_id())

    # --- 自前セッションを持つコーナー -------------------------------------

    def _corner_for(self, ac: ContentConfig | None):
        if ac is None:
            return None
        corner = self._corners.get(ac.type)
        if corner is None or corner.disabled:
            return None
        return corner

    def _tick_corner(self, corner, ctype: str) -> None:
        try:
            scripts = corner.produce_batch()
        except Exception:
            logger.exception("%s corner produce_batch crashed; not stopping regular broadcast", ctype)
            scripts = []
        for script in scripts:
            self._put_script(script)
        if not scripts:
            if self._director.phase is Phase.PREPARING:
                # 在庫切れ（ラジオドラマ）・作品の準備失敗など。暗いまま待たせない。
                self._director.report_nothing_to_air(self._director.transition_id())
            # ローカル暗転（フィラー⇄朗読・シーンまたぎ）の途中は、各段階の検知が
            # 2秒ずつ遅れて積み重なると演出全体が間延びするので、細かくポーリングする。
            wait = _PARK_POLL_SEC if getattr(corner, "transition_pending", False) else 2.0
            self._stop_event.wait(wait)  # セッション準備中/失敗時に忙しく回さない

    def _end_corner(self, emit_outro: bool) -> None:
        """今のコーナーを降ろす。朗読系は再生位置（DB）から再開するので読み飛ばしは出ない。"""
        corner, self._in_corner = self._in_corner, None
        try:
            outros = corner.end_corner()
        except Exception:
            logger.exception("corner end_corner crashed")
            return
        if not emit_outro:
            return
        for script in outros:
            script.is_closing = True
            self._put_script(script, closing=True)

    def _put_script(self, script: Script, closing: bool = False) -> None:
        while not self._stop_event.is_set():
            if not self._director.accepts_script(closing):
                logger.info("Not queuing script during corner transition: %s", script.topic_title)
                return
            try:
                self._script_queue.put(script, timeout=1.0)
                return
            except queue.Full:
                continue

    # --- トピック型コンテンツ ------------------------------------------------

    def _session_appearers(self, content: ContentConfig) -> list[CastMember]:
        """このコンテンツ区間が続く間は同じ出演者を使い回す。

        出演者は切り替え時に番組進行が抽選してひな壇へ並べたもの（暗転中に交換済み）。
        起動直後など番組進行が持っていないときだけここで抽選する。
        """
        if self._session_cast is not None and self._session_content is content:
            return self._session_cast
        appearers = self._director.session_cast(content) or pick_speakers(content, self._cast)
        self._session_content = content
        self._session_cast = appearers
        return appearers

    def _handle_topic(self, topic_id: int, topic: Topic, content: ContentConfig) -> None:
        logger.info("calling Ollama (%s) for topic %r ...", self._llm_config.model, topic.title)
        start = time.monotonic()
        mood = self._mood_provider.current() if self._mood_provider is not None else None
        appearers = self._session_appearers(content)
        script = generate_script(
            topic, list(self._recent_summaries), self._llm_config, content, self._cast,
            mood=mood, appearers=appearers,
        )
        elapsed = time.monotonic() - start
        if script is None:
            logger.warning("Ollama call for topic %r failed after %.1fs", topic.title, elapsed)
            return  # パース失敗など。このトピックは捨てて次へ

        logger.info(
            "Ollama generated %d lines for topic %r in %.1fs",
            len(script.lines),
            topic.title,
            elapsed,
        )
        if self._director.on_air_content() is not content or not self._director.accepts_script(False):
            # 生成している間にコーナー切り替えが始まった。前のコーナーのトークなので流さない。
            logger.info("Discarding generated script because corner transition has started: %r", topic.title)
            return
        logger.info("%s", _format_comparison(topic, script, self._names))

        script.topic_id = topic_id
        self._recent_summaries.append(f"{topic.title}（{topic.hint}）")

        script_json = json.dumps(
            {
                "lines": [
                    {"speaker": l.speaker, "text": l.text, "style": l.style}
                    for l in script.lines
                ]
            },
            ensure_ascii=False,
        )
        self._store.record_broadcast(topic_id, script_json)
        self._store.mark_used(topic_id)

        self._put_script(script)
