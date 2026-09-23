"""スポーツ結果ダイジェスト（MLB のみ）ソース。docs/sports-news.md 参照。

情報源を「事実」と「色付け」の二系統に分離する方針のうち、本モジュールは
**事実（スコア・勝敗・主要選手成績）**だけを担当する。色付け（世間の反応）は
type = "rss" の枠で別途スポーツメディアの RSS を拾う想定で、ここでは扱わない。

- 事実は MLB Stats API（``statsapi.mlb.com``、APIキー不要・無料）から取得する。
- 試合終了後（``status.abstractGameState == "Final"``）の直近の試合だけを Topic 化する
  （開始から48時間以内。試合中の逐次実況＝1球速報レベルは対象外）。日本の昼から見た
  「昨夜の結果」は米国日付では1〜2日前になるため、schedule は数日ぶん広めに取得して
  古い試合を落とす。
- 掲示板・X の実況スレは情報源にしない（適法性を情報源そのもので担保する方針）。
- 数値は API から取った確定済みの事実だけを body に入れ、LLM に新しい数値を
  創作させない（SPEC.md 4.3 ハルシネーション対策と同じ方針。数値は事実・
  コメントは自由）。

config.toml の ``[[content]]`` type = "sports" 配下に ``[[content.sports_watch]]``
を並べると、そこで指定した選手 / チームが出場した試合を優先ピックアップする
（何も指定しなければその日の Final 試合をすべて拾う）。日本人選手（大谷翔平・
山本由伸・今永昇太 等）を登録しておくと番組との親和性が高い。

    [[content.sports_watch]]
    player = "大谷翔平"        # body 内の表示名（任意。省略時は API の英語名）
    player_en = "Shohei Ohtani"  # API の person.fullName との照合に使う（部分一致）
    # mlb_id = 660271           # person.id が分かるなら照合はこちらが確実
    # team_id = 119             # チーム単位で拾う（119 = ドジャース）
"""

from __future__ import annotations

import datetime
import logging
import threading
from typing import Iterator

import requests

from ..source_status import SourceStatus
from . import USER_AGENT, Topic

logger = logging.getLogger(__name__)

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
BOXSCORE_URL = "https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"

_SPORT_ID_MLB = 1
_MAX_NEW_PER_POLL = 5  # 再起動直後に大量投入しないための上限（他ソースの _MAX_* に倣う）
_SKIP_STATES = {"Postponed", "Cancelled", "Canceled", "Suspended"}
# 「直近の試合結果」だけを扱う。MLB の schedule は officialDate（現地の暦日）で返るため、
# 米国の日付は UTC より 4〜7 時間遅れる。日本の昼に「昨夜の結果」を出すには UTC 日付で
# 数日ぶん引いて拾い、開始時刻が古すぎる試合をここで落とす。
_LOOKBACK_DAYS = 3
_MAX_AGE_HOURS = 48

# MLB 30 球団の日本語通称（team.id → 通称）。API は英語名しか返さないため添える。
_TEAM_JP: dict[int, str] = {
    108: "エンゼルス", 109: "ダイヤモンドバックス", 110: "オリオールズ",
    111: "レッドソックス", 112: "カブス", 113: "レッズ", 114: "ガーディアンズ",
    115: "ロッキーズ", 116: "タイガース", 117: "アストロズ", 118: "ロイヤルズ",
    119: "ドジャース", 120: "ナショナルズ", 121: "メッツ", 133: "アスレチックス",
    134: "パイレーツ", 135: "パドレス", 136: "マリナーズ", 137: "ジャイアンツ",
    138: "カージナルス", 139: "レイズ", 140: "レンジャーズ", 141: "ブルージェイズ",
    142: "ツインズ", 143: "フィリーズ", 144: "ブレーブス", 145: "ホワイトソックス",
    146: "マーリンズ", 147: "ヤンキース", 158: "ブルワーズ",
}


def _team_name(team: dict) -> str:
    tid = team.get("id")
    return _TEAM_JP.get(tid) or team.get("name") or team.get("teamName") or "?"


def _game_age_hours(game_date: str | None, now: datetime.datetime) -> float:
    """試合開始（gameDate, ISO8601 UTC）から現在までの経過時間。不明なら 0（＝新しい扱い）。"""
    if not game_date:
        return 0.0
    try:
        started = datetime.datetime.fromisoformat(game_date.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    return (now - started).total_seconds() / 3600.0


def _format_ip(ip: str | float | None) -> str:
    """MLB の投球回表記（"6.0" / "6.1" / "6.2"）を日本語の回表記に直す。"""
    s = str(ip or "0.0")
    whole, _, frac = s.partition(".")
    tail = {"1": "と1/3", "2": "と2/3"}.get(frac, "")
    return f"{whole}{tail}回"


class SportsSource:
    name = "sports"

    def __init__(
        self,
        watch: list[dict] | None = None,
        poll_interval_sec: float = 300.0,
        stop_event: threading.Event | None = None,
        timeout_sec: int = 10,
        status: SourceStatus | None = None,
    ):
        self._status = status
        self._watch = [dict(w) for w in (watch or [])]
        self._watch_ids = {
            int(w["mlb_id"]) for w in self._watch if w.get("mlb_id") is not None
        }
        self._watch_names = [
            str(w["player_en"]).lower() for w in self._watch if w.get("player_en")
        ]
        self._watch_teams = {
            int(w["team_id"]) for w in self._watch if w.get("team_id") is not None
        }
        self._base_poll_interval_sec = max(poll_interval_sec, 60.0)
        self._stop_event = stop_event or threading.Event()
        self._timeout_sec = timeout_sec
        # 一度 Topic 化した gamePk。同じ試合の boxscore を毎周期取りに行かないため。
        self._seen: set[int] = set()

    # -- fetch ---------------------------------------------------------------

    def fetch(self) -> Iterator[Topic]:
        headers = {"User-Agent": USER_AGENT}
        while not self._stop_event.is_set():
            games = 0
            try:
                for topic in self._poll_once(headers):
                    games += 1
                    yield topic
            except requests.RequestException as e:
                if self._status is not None:
                    self._status.fetch_failed("MLB Stats API", e)
                else:
                    logger.warning("mlb schedule poll failed: %s", e)

            if self._status is not None:
                self._status.cycle_end(games, self._base_poll_interval_sec)
            if self._stop_event.wait(self._base_poll_interval_sec):
                return

    def _poll_once(self, headers: dict) -> Iterator[Topic]:
        now = datetime.datetime.now(datetime.timezone.utc)
        today = now.date()
        # 日本の昼から見た「昨夜の試合」は米国日付では 1〜2 日前で、UTC 日付でも前日に
        # またがる。広めに拾って、開始が古すぎる試合は _maybe_topic の年齢チェックで落とす。
        params = {
            "sportId": _SPORT_ID_MLB,
            "startDate": (today - datetime.timedelta(days=_LOOKBACK_DAYS)).isoformat(),
            "endDate": (today + datetime.timedelta(days=1)).isoformat(),
            "hydrate": "decisions",
        }
        resp = requests.get(
            SCHEDULE_URL, params=params, headers=headers, timeout=self._timeout_sec
        )
        resp.raise_for_status()
        data = resp.json()

        games = [g for d in data.get("dates", []) for g in d.get("games", [])]
        new_finals = sum(
            1
            for g in games
            if g.get("status", {}).get("abstractGameState") == "Final"
            and g.get("gamePk") not in self._seen
            and _game_age_hours(g.get("gameDate"), now) <= _MAX_AGE_HOURS
        )

        emitted = 0
        for game in games:
            if self._stop_event.is_set() or emitted >= _MAX_NEW_PER_POLL:
                return
            topic = self._maybe_topic(game, headers, now)
            if topic is not None:
                emitted += 1
                logger.info("mlb topic: %s", topic.title)
                yield topic

        if emitted == 0 and new_finals and self._watch:
            # Final はあるが watch 対象の出場試合が無い（＝深夜帯や登板日以外は普通）。
            # 無音の原因調査で「ソースは動いているが素材が無い」と分かるようログに残す。
            logger.info(
                "mlb: %d new Final game(s) in window but none match sports_watch",
                new_finals,
            )

    def _maybe_topic(
        self, game: dict, headers: dict, now: datetime.datetime
    ) -> Topic | None:
        game_pk = game.get("gamePk")
        if game_pk is None or game_pk in self._seen:
            return None

        status = game.get("status", {})
        if status.get("abstractGameState") != "Final":
            return None
        if status.get("detailedState") in _SKIP_STATES:
            self._seen.add(game_pk)
            return None

        if _game_age_hours(game.get("gameDate"), now) > _MAX_AGE_HOURS:
            self._seen.add(game_pk)  # 古い試合。以後この周期で見直さない
            return None

        teams = game.get("teams", {})
        home = teams.get("home", {})
        away = teams.get("away", {})
        home_team = home.get("team", {})
        away_team = away.get("team", {})
        home_id = home_team.get("id")
        away_id = away_team.get("id")

        team_match = bool(self._watch_teams & {home_id, away_id})
        # watch にチームしか無く、この試合に無関係なら boxscore を取る前に落とす。
        if self._watch and not team_match and not self._watch_names and not self._watch_ids:
            return None

        box = self._fetch_boxscore(game_pk, headers)
        if box is None:
            return None  # 一時的な失敗。_seen に入れず次周期で再試行する

        self._seen.add(game_pk)

        highlights, player_hit = self._collect_highlights(box)
        if self._watch and not team_match and not player_hit:
            return None  # 監視対象が出ていない試合

        home_runs = home.get("score")
        away_runs = away.get("score")
        if home_runs is None or away_runs is None:
            return None

        home_name = _team_name(home_team)
        away_name = _team_name(away_team)
        title = f"{away_name} {away_runs}-{home_runs} {home_name}"

        if home_runs == away_runs:
            lead = f"{away_name}と{home_name}は{away_runs}-{home_runs}で引き分け。"
        else:
            win_name, lose_name = (
                (home_name, away_name) if home_runs > away_runs else (away_name, home_name)
            )
            hi, lo = max(home_runs, away_runs), min(home_runs, away_runs)
            lead = f"{win_name}が{hi}-{lo}で{lose_name}に勝利。"

        decision = self._decision_line(game.get("decisions") or {})
        parts = [lead]
        if decision:
            parts.append(decision)
        parts.extend(highlights)
        body = "\n".join(parts)

        return Topic(
            source=self.name,
            external_id=f"mlb:{game_pk}",
            title=title,
            body=body[:2000],
            url=None,
            hint="この試合結果についてひな壇で盛り上がってコメントして",
        )

    def _decision_line(self, decisions: dict) -> str | None:
        """勝利投手 / 敗戦投手 / セーブ（schedule の hydrate=decisions から）。"""
        seg: list[str] = []
        for key, label in (("winner", "勝利投手"), ("loser", "敗戦投手"), ("save", "セーブ")):
            person = decisions.get(key)
            if person and person.get("fullName"):
                seg.append(f"{label} {self._display_name(person)}")
        return "、".join(seg) if seg else None

    # -- MLB Stats API helpers --------------------------------------------------

    def _fetch_boxscore(self, game_pk: int, headers: dict) -> dict | None:
        try:
            resp = requests.get(
                BOXSCORE_URL.format(game_pk=game_pk),
                headers=headers,
                timeout=self._timeout_sec,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.warning("mlb boxscore %s fetch failed: %s", game_pk, e)
            return None

    def _display_name(self, person: dict) -> str:
        full = person.get("fullName", "")
        pid = person.get("id")
        low = full.lower()
        for w in self._watch:
            if not w.get("player"):
                continue
            if (w.get("mlb_id") is not None and pid == int(w["mlb_id"])) or (
                w.get("player_en") and str(w["player_en"]).lower() in low
            ):
                return str(w["player"])
        return full

    def _is_watched(self, person: dict) -> bool:
        pid = person.get("id")
        low = person.get("fullName", "").lower()
        if pid in self._watch_ids:
            return True
        return any(n in low for n in self._watch_names)

    def _collect_highlights(self, box: dict) -> tuple[list[str], bool]:
        """主要選手の成績行と、監視対象が出場していたかを返す。

        watch 指定があればその選手だけを、無ければ目立った成績の選手を数人拾う。
        """
        lines: list[str] = []
        player_hit = False
        for side in ("away", "home"):
            players = box.get("teams", {}).get(side, {}).get("players", {})
            for entry in players.values():
                person = entry.get("person", {})
                stats = entry.get("stats", {})
                watched = self._is_watched(person)
                if watched:
                    player_hit = True

                if self._watch and not watched:
                    continue

                line = self._stat_line(person, stats, notable_only=not self._watch)
                if line:
                    lines.append(line)

        if not self._watch:
            lines = lines[:4]  # body が長くなりすぎないよう抑える
        return lines, player_hit

    def _stat_line(self, person: dict, stats: dict, notable_only: bool) -> str | None:
        name = self._display_name(person)
        bat = stats.get("batting", {}) or {}
        pit = stats.get("pitching", {}) or {}

        parts: list[str] = []
        at_bats = bat.get("atBats")
        if at_bats:
            hits = bat.get("hits", 0)
            hr = bat.get("homeRuns", 0)
            rbi = bat.get("rbi", 0)
            notable = hits >= 2 or hr >= 1 or rbi >= 2
            if not notable_only or notable:
                seg = f"{hits}安打{at_bats}打数"
                if hr:
                    seg += f"（{hr}本塁打）"
                if rbi:
                    seg += f"、{rbi}打点"
                parts.append(seg)

        ip = pit.get("inningsPitched")
        if ip and float(str(ip)) > 0:
            er = pit.get("earnedRuns", 0)
            so = pit.get("strikeOuts", 0)
            h = pit.get("hits", 0)
            notable = float(str(ip)) >= 4.0
            if not notable_only or notable:
                parts.append(
                    f"{_format_ip(ip)}を投げ被安打{h}・{so}奪三振・自責点{er}"
                )

        if not parts:
            return None
        return f"{name}: " + "、".join(parts)
