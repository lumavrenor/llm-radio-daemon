"""番組編成スケジューラ。config.toml の [[content]] を現在時刻で解決する。

番組表型（排他）: ある時刻に「アクティブなコンテンツ」は 1 つだけ。[[content]] を
記述順に評価し、``enabled`` かつ ``schedule`` の窓に入っている最初のエントリを選ぶ。
``schedule`` を省略したエントリは常時対象（フォールバック用に末尾へ置く）。

SourceThread は自分の type がアクティブなときだけネタを収集し、ScriptThread は
アクティブなコンテンツの種類に応じて通常トーク / 読書コーナー / 無音（radio）を
切り替える。プロンプトのトーンは ``tone_hint()`` から差し込む。

「放送したいコーナー」と「放送中のコーナー」は分けて持つ:

- ``scheduled_content()`` … 番組表＋リクエストが今示しているコーナー（放送したいもの）。
  読むのは番組進行（director.py）だけ。
- ``active_content()`` … 今実際に放送中のコーナー（on-air）。各スレッド・画面はこちらを
  読む。``set_on_air_provider()`` で登録した番組進行が返す値で、番組進行は両者が
  食い違ったら「今のトークを流し切る → 暗転 → 入れ替え → 準備 → 明転」の順に
  on-air を切り替える。プロバイダ未登録（テスト・下見）なら番組表そのもの。

リクエスト（§6.2）: 「今このコーナーが聴きたい」は、番組表の判定そのものを一時的に
上書きする形で入れる。``set_request_provider()`` で登録したプロバイダ（実体は
``TopicStore.active_request_target``）が受け付け済みのリクエストを返している間、
``scheduled_content()`` は番組表を無視してそのエントリを返す。
"""

from __future__ import annotations

import datetime
import logging
import time
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .config import ContentConfig

logger = logging.getLogger(__name__)

# リクエストの問い合わせ結果をこの秒数だけ使い回す。active_content() は5つの
# スレッドから毎ループ（およそ1秒間隔）呼ばれるので、素通しだと SQLite を
# 秒間5回叩くことになる。display/app.py が同じ理由で 2 秒キャッシュしている。
_REQUEST_CACHE_TTL_SEC = 1.0

_request_provider: "Callable[[], tuple[int, str] | None] | None" = None
_request_cache: dict = {"t": 0.0, "value": None}
# 添字ズレを警告するのは1リクエストにつき1回だけ（毎秒ログに出さない）。
_warned_targets: set = set()

_on_air_provider: "Callable[[], ContentConfig | None] | None" = None


def set_on_air_provider(provider: "Callable[[], ContentConfig | None] | None") -> None:
    """放送中のコーナーを返す関数（ProgramDirector.on_air_content）を登録する。"""
    global _on_air_provider
    _on_air_provider = provider


def invalidate_request_cache() -> None:
    """リクエストを受け付けた直後に呼ぶ。次の scheduled_content() で DB を読み直させる。"""
    _request_cache["t"] = 0.0


def set_request_provider(
    provider: "Callable[[], tuple[int, str] | None] | None",
) -> None:
    """有効なリクエストの (content_index, content_type) を返す関数を登録する。

    main.py が TopicStore を作った直後に一度だけ呼ぶ。None を渡すと解除
    （テストや、DB を持たない経路では登録しないまま＝従来どおり番組表だけで動く）。
    """
    global _request_provider
    _request_provider = provider
    _request_cache["t"] = 0.0
    _request_cache["value"] = None
    _warned_targets.clear()


def _requested_target() -> "tuple[int, str] | None":
    if _request_provider is None:
        return None
    now = time.monotonic()
    if now - _request_cache["t"] >= _REQUEST_CACHE_TTL_SEC:
        try:
            _request_cache["value"] = _request_provider()
        except Exception:
            logger.exception("Failed to read request. Proceeding according to the schedule")
            _request_cache["value"] = None
        _request_cache["t"] = now
    return _request_cache["value"]


def requested_content(content: list["ContentConfig"]) -> "ContentConfig | None":
    """有効なリクエストが指しているコーナー。無効・無ければ None。

    **必ず ``content`` の中の既存インスタンスを返す**（複製を作らない）。
    StationThread が ``content is not self._current_content`` と同一性で
    コーナー切替を判定しているため、毎回別オブジェクトを返すと局の繋ぎ直しが
    延々と走ってしまう。
    """
    target = _requested_target()
    if target is None:
        return None
    idx, ctype = target
    if 0 <= idx < len(content) and content[idx].type == ctype:
        return content[idx]
    if target not in _warned_targets:
        _warned_targets.add(target)
        logger.warning(
            "Request (#%d %s) does not match the current [[content]]. "
            "config_content.toml may have been edited after it was issued; proceeding per the schedule",
            idx, ctype,
        )
    return None


def _parse_hm(s: str) -> int:
    """"HH:MM" → 0時からの分。"""
    h, _, m = s.strip().partition(":")
    return int(h) * 60 + int(m or 0)


def parse_window(spec: str) -> tuple[int, int]:
    """"HH:MM-HH:MM" → (開始分, 終了分)。書式不正なら ValueError。"""
    start_s, sep, end_s = spec.partition("-")
    if not sep:
        raise ValueError(f"invalid window spec: {spec!r}")
    return _parse_hm(start_s), _parse_hm(end_s)


def within_window(spec: str, now: datetime.datetime | None = None) -> bool:
    """現在時刻が spec("HH:MM-HH:MM") の窓に入っているか。パース不能なら False。"""
    now = now or datetime.datetime.now()
    try:
        start, end = parse_window(spec)
    except (ValueError, AttributeError):
        logger.warning("schedule format is invalid: %r (HH:MM-HH:MM)", spec)
        return False
    if start == end:
        return False
    cur = now.hour * 60 + now.minute
    if start < end:
        return start <= cur < end
    return cur >= start or cur < end  # 日跨ぎ


def within_any(specs: list[str], now: datetime.datetime | None = None) -> bool:
    """specs のいずれかの窓に入っていれば True。specs が空なら常時 True（時間帯指定なし）。"""
    if not specs:
        return True
    return any(within_window(s, now) for s in specs)


def scheduled_content(
    content: list["ContentConfig"], now: datetime.datetime | None = None
) -> "ContentConfig | None":
    """番組表（＋リクエスト）が示すコンテンツ。記述順が先で、enabled かつ時間帯に入っている
    最初のもの。無ければ None。

    受け付け済みのリクエスト（§6.2）があれば番組表より優先する。ただし ``now`` を明示
    されたときは掛けない ―― その呼び方は「その時刻の番組表はどうなっているか」を問うもので
    （テストや編成の下見）、「今なにが流れているか」ではないため。
    """
    if now is None:
        requested = requested_content(content)
        if requested is not None:
            return requested
    for c in content:
        if c.enabled and within_any(c.schedule, now):
            return c
    return None


def active_content(
    content: list["ContentConfig"], now: datetime.datetime | None = None
) -> "ContentConfig | None":
    """今放送中のコンテンツ（番組進行が切り替えを済ませたもの）。

    番組進行が未登録、または ``now`` を明示されたときは番組表そのもの。
    """
    if now is None and _on_air_provider is not None:
        return _on_air_provider()
    return scheduled_content(content, now)


def is_active(
    content: list["ContentConfig"],
    content_type: str,
    now: datetime.datetime | None = None,
) -> bool:
    """指定 type のコンテンツが今アクティブか。"""
    ac = active_content(content, now)
    return ac is not None and ac.type == content_type


def tone_hint(
    content: list["ContentConfig"], now: datetime.datetime | None = None
) -> str:
    """アクティブなコンテンツの tone_hint（無ければ空文字）。"""
    ac = active_content(content, now)
    return ac.tone_hint if ac is not None else ""
