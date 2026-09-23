"""ネタ源1つぶんの取得状況。

ログだけを見ていると「SourceThread が死んだ」のか「単に新着が無い」のかが
区別できない（重複で捨てた分は無言で消えるため）。ここで1周ぶんの結果を
集計して、周回のたびに必ず1行ログを出し、同じ内容を短い英語で画面左上へ流す。

役割分担:
  - SourceThread が1件ごとの結果を :meth:`record` で積む（新規/既知/重複の別は
    重複排除まで通してみないと分からないので、スレッド側でしか数えられない）
  - ソース本体（RssSource など）が1周の終わりに :meth:`cycle_end` を呼ぶ
    （「どこが1周か」を知っているのはソース側だけ）

SourceThread とそのソースは同じスレッドで動くのでカウンタにロックは要らない。
SharedState への書き込みだけ SharedState 側のロックで守られる。
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from .state import SharedState

logger = logging.getLogger(__name__)

NEW = "new"              # 採用してキューへ入れた
KNOWN = "known"          # external_id が既知（過去に取り込み済み）
DUPLICATE = "duplicate"  # embedding 的に直近のネタと似すぎ


class SourceStatus:
    def __init__(
        self,
        content_type: str,
        state: SharedState | None = None,
        pending_check: Callable[[], bool] | None = None,
    ):
        self._type = content_type
        self._state = state
        # 「まだ読み上げ待ちの台本・ネタが残っているか」。残っているうちは新着ゼロでも
        # 番組は普通に続いているので、画面の「no new articles」で紛らわしく見せない
        # （フィラーに落ちて初めて出す）。None なら常に「残っていない」扱い。
        self._pending_check = pending_check or (lambda: False)
        self._new = 0
        self._known = 0
        self._duplicate = 0
        self._failed_feeds = 0
        self._last_new_at: float | None = None
        self._started_at = time.time()

    # --- SourceThread から ---------------------------------------------

    def record(self, outcome: str) -> None:
        if outcome == NEW:
            self._new += 1
            self._last_new_at = time.time()
            # 周回のあるソースは直後の cycle_end がこれを上書きする。上書きが来ない
            # ソース（wikimedia の SSE など）でも、最低限これだけは画面に出る。
            self._publish(f"last new {self._since_text()}")
        elif outcome == DUPLICATE:
            self._duplicate += 1
        else:
            self._known += 1

    def crashed(self) -> None:
        """周の途中で例外が出た（cycle_end は来ない）。集計を捨てて画面にも出す。

        例外そのものは SourceThread が traceback 付きで ERROR に出しているので、
        ここは「ネタが無いのではなく壊れている」と分かる短い表示だけにする。
        """
        self._new = self._known = self._duplicate = self._failed_feeds = 0
        self._publish("error - see log")

    def idle(self) -> None:
        """番組表の時間帯を抜けて収集をやめた。周の途中なのでカウンタは捨てる。"""
        self._new = self._known = self._duplicate = self._failed_feeds = 0
        self._publish("off schedule")

    # --- ソース本体から -------------------------------------------------

    def fetch_failed(self, url: str, err: object) -> None:
        """フィード1本の取得に失敗した。周回自体は続くので数えるだけ。"""
        self._failed_feeds += 1
        logger.warning("%s: fetch failed for %s: %s", self._type, url, err)

    def cycle_end(self, entries: int, next_poll_sec: float = 0.0) -> None:
        """1周ぶんの集計をログと画面へ出し、カウンタを畳む。

        ``entries`` はソースが読み取った記事数（重複排除より前の生の件数）。
        """
        nxt = f"; next poll in {next_poll_sec:.0f}s" if next_poll_sec else ""
        if self._new:
            logger.info(
                "%s cycle: %d entries -> %d new (known %d, dup %d, failed feeds %d)%s",
                self._type, entries, self._new, self._known,
                self._duplicate, self._failed_feeds, nxt,
            )
            self._publish(f"{self._new} new / {entries} entries")
        else:
            # 新着ゼロは「異常ではないが放っておくとフィラーだけになる」状態。
            # 起動直後の切り分けで一番知りたいので WARNING で出す。
            logger.warning(
                "%s cycle: %d entries -> 0 new (known %d, dup %d, failed feeds %d); "
                "nothing new since %s%s",
                self._type, entries, self._known, self._duplicate,
                self._failed_feeds, self._since_text(), nxt,
            )
            if self._pending_check():
                # 読み上げ待ちの作り置きがまだある＝フィラーに落ちてはいない。ここで
                # 「no new articles」に書き換えると、台本はまだ続いているのに
                # ネタ切れしたように見えてしまうので、画面表示は前回の内容のまま触らない。
                pass
            elif entries == 0 and self._failed_feeds:
                self._publish(f"fetch failed ({self._failed_feeds} feed(s))")
            elif entries == 0:
                self._publish("no entries fetched")
            else:
                self._publish(f"no new articles since {self._since_text()}")

        self._new = self._known = self._duplicate = self._failed_feeds = 0

    # --- 内部 -----------------------------------------------------------

    def _since_text(self) -> str:
        """最後に新着を採用した時刻（まだ無ければ起動時刻）を HH:MM で。"""
        return time.strftime("%H:%M", time.localtime(self._last_new_at or self._started_at))

    def _publish(self, text: str) -> None:
        if self._state is not None:
            self._state.set_source_status(self._type, text)
