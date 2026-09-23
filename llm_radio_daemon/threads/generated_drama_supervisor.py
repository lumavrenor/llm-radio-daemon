"""ラジオドラマの自動執筆スーパーバイザ（SPEC.md §4.7.6）。

``[[content]] type = "generated_drama"`` の ``auto_write = true`` のときだけ起動する。
放送が LLM をほとんど使っていない隙を見て、執筆バッチ
（``python -m llm_radio_daemon.generated_drama_writer run --auto --scenes 1``）を
**子プロセス**として1シーンぶんだけ走らせる。これでユーザーは VOICEVOX と
放送本体を起動するだけでよく、タスクスケジューラ登録も手動実行も要らなくなる。

放送と執筆で GPU・Ollama 常駐モデルを取り合わないための2段構え:

1. **アイドル判定（中間）** — アクティブなコーナーが LLM をほぼ使わない種類
   （``generated_drama`` の朗読中、または ``radio``）のときだけ起動する。話芸コーナー
   （rss/hackernews/arxiv/sports/weather/worry_consultation/wikimedia）や ``reading``
   （ビート感想で LLM を使う）の最中は起動しない。
2. **走行中の監視と中断** — 子プロセスの実行中も数秒ごとにアイドル判定を続け、
   条件が崩れたら子プロセスを ``terminate()`` する。既に Ollama へ投げ済みの
   1リクエストだけはサーバ側で完走するが、その間は script_queue の作り置きと
   フィラーが場をつなぐ（§4.7.4「放送が止まることだけが異常」）。

バックオフは「まだ放送していない ready 済みシーンの在庫」で刻む（適応式）:

    在庫 20 本以上 → 60 分あける（十分先まで書けている）
    在庫  5〜19 本 → 20 分
    在庫   5 本未満 → 1 分（在庫が薄い。急いで補充する）

新規に小説を立ち上げた直後は在庫ゼロなので自然と 1 分刻みで書き溜め、
追いついたら勝手に間隔が伸びる。
"""

from __future__ import annotations

import logging
import queue
import subprocess
import sys
import threading

from ..config import ContentConfig
from ..db import TopicStore
from ..director import ProgramDirector
from ..schedule import active_content
from ..script import Script
from ..state import SharedState

logger = logging.getLogger(__name__)

# 起動直後は放送の立ち上げ（初回の台本生成・ソース取得）で GPU が混むので、
# 少し待ってから最初の執筆を試す。
_STARTUP_DELAY_SEC = 120.0

# アイドルでない間、次にアイドル判定をやり直すまでの待ち。
_IDLE_POLL_SEC = 30.0

# 子プロセスの実行中、アイドル判定をやり直す間隔。コーナー切り替えは執筆の停止を
# 待ってから暗転するので、ここが長いとそのぶん切り替えが遅れる。
_RUN_POLL_SEC = 2.0

# terminate() したあと SIGKILL 相当へ切り替えるまでの猶予。
_TERM_GRACE_SEC = 15.0

# 適応バックオフの在庫しきい値（本数）と各段の待ち秒。
_STOCK_HIGH = 20
_STOCK_LOW = 5
_BACKOFF_HIGH_SEC = 3600.0
_BACKOFF_MID_SEC = 1200.0
_BACKOFF_LOW_SEC = 60.0
# run --auto が「書くものが無い」で返したとき（全部 finished かつ auto_concept=false 等）。
_BACKOFF_NOTHING_SEC = 1800.0
# 子プロセスの起動そのものに失敗したとき。
_BACKOFF_LAUNCH_FAIL_SEC = 1800.0


class GeneratedDramaSupervisorThread(threading.Thread):
    def __init__(
        self,
        config_path: str,
        content: list[ContentConfig],
        state: SharedState,
        script_queue: "queue.Queue[Script]",
        store: TopicStore,
        director: ProgramDirector,
        stop_event: threading.Event | None = None,
        *,
        log_file: str = "generated_drama_writer.log",
    ):
        super().__init__(name="GeneratedDramaSupervisorThread", daemon=True)
        self._config_path = config_path
        self._content = content
        self._director = director
        self._state = state
        self._script_queue = script_queue
        self._store = store
        self._stop_event = stop_event or threading.Event()
        self._log_file = log_file
        self._proc: subprocess.Popen | None = None

    def stop(self) -> None:
        self._stop_event.set()
        self._terminate_child()

    # --- メインループ ---------------------------------------------------

    def run(self) -> None:
        if self._stop_event.wait(_STARTUP_DELAY_SEC):
            return
        while not self._stop_event.is_set():
            try:
                if not self._is_idle():
                    self._stop_event.wait(_IDLE_POLL_SEC)
                    continue
                outcome = self._run_one_scene()
                self._stop_event.wait(self._backoff_for(outcome))
            except Exception:
                logger.exception("generated drama supervisor crashed on tick; continuing")
                self._terminate_child()
                self._stop_event.wait(_BACKOFF_MID_SEC)

    # --- アイドル判定（中間）-------------------------------------------

    def _is_idle(self) -> bool:
        """今、放送が LLM をほとんど使っていない＝執筆を走らせてよいか。"""
        if not self._director.is_steady():
            # コーナー切り替え中。前のコーナーを止め切る・次のコーナーを準備する間は譲る。
            return False
        ac = active_content(self._content)
        if ac is None:
            return False
        if ac.type == "generated_drama":
            # 確定本文の朗読中は GPU が完全に空く。在庫切れでフィラーが場を
            # つないでいる場合も「まさに書くべき時」なのでここで許可する。
            return True
        if ac.type == "radio":
            # 曲トークは1曲あたり数えるほどしか LLM を呼ばない。ただし作り置きの
            # 台本が尽きている・フィラーが動いているときは譲る。
            return not self._script_queue.empty() and not self._state.filler_active
        # rss / hackernews / arxiv / sports / weather / worry_consultation / wikimedia /
        # literary_reading / translated_reading / biography_reading（いずれもトークの合間にLLMを使う。
        # translated_reading はチャンクごとに毎回LLMを呼ぶので biography_reading と同じ扱い）
        return False

    # --- 子プロセスの起動と監視 ---------------------------------------

    def _run_one_scene(self) -> str:
        """generated_drama_writer を1シーンぶん走らせる。結果ラベルを返す。

        戻り値: "wrote"（書けた）/ "nothing"（書くものが無い）/
                "interrupted"（放送が忙しくなって中断）/ "launch_failed"。
        """
        cmd = [
            sys.executable,
            "-m",
            "llm_radio_daemon.generated_drama_writer",
            "--config",
            self._config_path,
            "--log-file",
            self._log_file,
            "run",
            "--auto",
            "--scenes",
            "1",
        ]
        logger.info("generated_drama_writer: broadcast slot is open, launching the writing batch")
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            logger.exception("generated_drama_writer: failed to launch child process")
            return "launch_failed"

        self._proc = proc
        self._state.generated_drama_writing = True
        try:
            while True:
                try:
                    rc = proc.wait(timeout=_RUN_POLL_SEC)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if self._stop_event.is_set():
                    self._terminate_child()
                    return "interrupted"
                if not self._is_idle():
                    logger.info(
                        "generated_drama_writer: broadcast started using the LLM; interrupting writing"
                    )
                    self._terminate_child()
                    return "interrupted"
        finally:
            self._proc = None
            self._state.generated_drama_writing = False

        # generated_drama_writer の run: 何か書けたら 0、書くものが無ければ 1。
        if rc == 0:
            stock = self._store.count_broadcastable_scenes()
            logger.info("generated_drama_writer: wrote 1 scene (%d in stock awaiting broadcast)", stock)
            return "wrote"
        logger.info("generated_drama_writer: nothing to write this time (rc=%s)", rc)
        return "nothing"

    def _terminate_child(self) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=_TERM_GRACE_SEC)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:
            logger.exception("generated_drama_writer: failed to stop child process")

    # --- 適応バックオフ ------------------------------------------------

    def _backoff_for(self, outcome: str) -> float:
        if outcome == "launch_failed":
            return _BACKOFF_LAUNCH_FAIL_SEC
        if outcome == "interrupted":
            return _BACKOFF_LOW_SEC  # 少し置いてまた隙を探す
        if outcome == "nothing":
            return _BACKOFF_NOTHING_SEC

        stock = self._store.count_broadcastable_scenes()
        if stock >= _STOCK_HIGH:
            return _BACKOFF_HIGH_SEC
        if stock >= _STOCK_LOW:
            return _BACKOFF_MID_SEC
        return _BACKOFF_LOW_SEC
