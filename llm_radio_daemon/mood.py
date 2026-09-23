"""日替わりの「今日のキャラごとの気分・トーン」（持ちネタキャラバリエーション §2）。

なぜ ``weather.py`` と同型で、別プロセスにしないのか
--------------------------------------------------
mood は「ネタ」ではなく「今日という日の一回性」を足すための軽い文脈情報。
SPEC 4.7.1 が NovelWriter を別プロセスにするのは「常駐モデル・単一 GPU で数分かかる
多段生成を放送プロセスで走らせない」ためで、mood は **1日1回・1リクエスト・数秒**。
``ScriptThread``（ワーカー）が既にトピック毎に回している ``generate_script()`` に対して
誤差の範囲であり、2.1 が禁じる *メインループ* のブロックには当たらない。
``weather.py`` も同種の軽い文脈情報について専用スレッドを作らず遅延キャッシュを
選んでいる（``WeatherProvider`` の docstring 参照）。よって mood も:

- 専用スレッド・専用プロセス・タスクスケジューラ登録を作らない
- :meth:`CastMoodProvider.current` 呼び出し時にローカル日付の変化を検知して再生成
- 生成結果はメモリ＋ ``data/mood_<lang>.json`` に atomic write（昼間の再起動で作り直さない）。
  再利用は「日付＝今日 かつ ``[llm] model`` が一致」のときだけ（モデルを切り替えて
  試すとき前モデルの mood を引きずらない）。失敗時のフォールバックはモデル違いでも使う
- **SQLite テーブルは作らない**（1日1行に対して過剰）

フォールバック（SPEC 8章）
------------------------
生成に失敗したら「前日分 → 既定の中立（＝空 dict）」の順に落ちる。mood が無くても
放送は無影響（``generate_script`` は mood 行を1行も足さないだけ）。生成が同じ日に
何度も失敗して LLM を叩き続けないよう、``weather.py`` と同じくリトライ間隔を空ける。
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

import requests

from . import language, llm_http
from .config import CastMember, LLMConfig
from .weather import Weather, WeatherProvider

logger = logging.getLogger(__name__)

# 生成に失敗してから次に試すまで（同じ日に LLM を叩き続けないための最低限のバックオフ）。
_RETRY_AFTER_SEC = 900.0

# 気分の1行が長くなりすぎたときの安全網（§4-2: 各行は短く）。文字数で機械的に丸める。
# 英語は1文字あたりの情報量が低く、プロンプトで頼んでいる「15語まで」が 60 字に
# 収まらないので言語ごとに変える（日本語 40 字 ≒ 英語 100 字の感覚）。
_MAX_MOOD_CHARS = {"ja": 60, "en": 120}

# 月 → 季節（北半球・機械判定。LLM は使わない）。
_SEASONS_JA = {12: "冬", 1: "冬", 2: "冬", 3: "春", 4: "春", 5: "春",
               6: "夏", 7: "夏", 8: "夏", 9: "秋", 10: "秋", 11: "秋"}
_SEASONS_EN = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
               6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}
_WEEKDAYS_JA = ["月曜", "火曜", "水曜", "木曜", "金曜", "土曜", "日曜"]


def _season(month: int, lang: str) -> str:
    table = _SEASONS_EN if lang == "en" else _SEASONS_JA
    return table.get(month, "")


def _weekday(date: _dt.date, lang: str) -> str:
    if lang == "en":
        return date.strftime("%A")
    return _WEEKDAYS_JA[date.weekday()]


def _weather_line(w: Weather, lang: str) -> str:
    """天気を1行に丸める。

    以前はここに英語版の要約を手で組んでいたが、``Weather.fact_line`` 自体が
    ``[locale] lang`` を見るようになったのでそちらへ寄せた（"12C" のような
    記号混じりを英語の TTS が読めない問題も向こうで直っている）。lang 引数は
    呼び出し側の都合で残してあるが、判断は fact_line に任せる。
    """
    return w.fact_line


def _mood_schema(cast_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "moods": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        # cast_id は enum。綴り崩れ・知らない id を弾く。
                        "cast_id": {"type": "string", "enum": cast_ids},
                        "mood": {"type": "string"},
                    },
                    "required": ["cast_id", "mood"],
                },
            }
        },
        "required": ["moods"],
    }


def _build_prompt(
    cast: list[CastMember], weather: Weather | None, date: _dt.date, lang: str
) -> str:
    season = _season(date.month, lang)
    weekday = _weekday(date, lang)
    roster = "\n".join(f"- {m.id}: {m.name} — {m.desc}" for m in cast)
    if lang == "en":
        context = [f"Date: {date.isoformat()} ({weekday})", f"Season: {season}"]
        if weather is not None:
            context.append("Weather: " + _weather_line(weather, lang))
        ctx = "\n".join(context)
        return f"""You write a short daily "mood" note for each cast member of a radio talk show.
It is a gentle day-to-day wobble in tone, not a personality change. Keep each member's
core character (below) intact and only add how today feels for them.

## Today
{ctx}

## Cast
{roster}

## What to write
- One short line per cast member (at most ~15 words), in natural spoken English
- Describe only a slight shift in mood or tone for today, grounded in the weather / season /
  weekday above. Do not invent events, plans, news, or facts
- If a member has no noteworthy wobble today, omit them from the list
- Return JSON: {{ "moods": [ {{ "cast_id": "...", "mood": "..." }}, ... ] }}
{language.LANGUAGE_GUIDANCE}
"""
    context = [f"日付: {date.isoformat()}（{weekday}）", f"季節: {season}"]
    if weather is not None:
        context.append("天気: " + _weather_line(weather, lang))
    ctx = "\n".join(context)
    return f"""ラジオのトーク番組の出演者それぞれに、「今日の気分・トーン」を一言だけ書いてください。
これはキャラを変える設定ではなく、日替わりのごく軽い揺らぎです。下に書いた
そのキャラの根本（説明文）はそのままに、「今日はこう感じている」だけを足します。

## 今日
{ctx}

## 出演者
{roster}

## 書き方
- 出演者1人につき短い1行（40文字以内のめやす）。話し言葉で
- 上の天気・季節・曜日から自然に導ける、気分やトーンのわずかな傾きだけを書く。
  出来事・予定・ニュース・事実をでっち上げないこと
- 特筆すべき揺らぎが無い出演者は、リストから省いてよい
- JSON で返すこと: {{ "moods": [ {{ "cast_id": "...", "mood": "..." }}, ... ] }}
{language.LANGUAGE_GUIDANCE}
"""


class CastMoodProvider:
    """日替わりのキャラごとの気分・トーンを持っておく（``weather.py`` 型の遅延キャッシュ）。

    ``ScriptThread`` が ``generate_script()`` のプロンプト構築前に :meth:`current` を1回
    引く。その日の初回だけ数秒ブロックするが、フィラーが埋めるので許容する。
    既定 off。``[llm] daily_mood = true`` のときだけ main.py がこのインスタンスを作る。
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        cast: list[CastMember],
        *,
        lang: str = "ja",
        weather: WeatherProvider | None = None,
        cache_path: str | Path = "",
        timeout_sec: int = 90,
        _today: Callable[[], _dt.date] | None = None,
    ):
        self._llm_config = llm_config
        self._cast = cast
        self._lang = (lang or "ja").strip().lower()
        self._weather = weather
        self._cache_path = Path(cache_path) if cache_path else Path(f"data/mood_{self._lang}.json")
        self._timeout_sec = int(timeout_sec)
        self._today = _today or _dt.date.today
        self._lock = threading.Lock()
        self._cache: dict[str, str] | None = None
        self._cache_date: str | None = None
        # 「その日ぶんの生成をいつ試したか」。日付が変われば必ず1回試し、同じ日に
        # 失敗が続いても _RETRY_AFTER_SEC 間隔までしか LLM を叩かない（weather.py と同じ）。
        self._attempted_date: str | None = None
        self._last_attempt = 0.0

    def current(self) -> dict[str, str]:
        """今日の ``{cast_id: 気分の1行}``。生成前・失敗時は前日分、無ければ ``{}``。"""
        with self._lock:
            today = self._today().isoformat()
            if self._cache is not None and self._cache_date == today:
                return self._cache

            disk = self._load_disk()
            # 日付が今日で、かつ同じモデルで作ったものだけ再利用する。モデルを
            # 切り替えて試すとき（小型モデルの検証など）に前モデルの mood を
            # 引きずらないため。失敗時のフォールバックでは別モデル分でも使う（下）。
            if disk is not None and disk[0] == today and disk[1] == self._llm_config.model:
                self._cache_date, self._cache = today, disk[2]
                return self._cache

            now = time.monotonic()
            if (
                self._attempted_date == today
                and now - self._last_attempt < _RETRY_AFTER_SEC
            ):
                # 今日ぶんは直近で試したばかり（成功していればキャッシュで返っている）。
                # LLM を叩き直さず、前日分（無ければ中立）で凌ぐ。
                return self._cache if self._cache is not None else {}
            self._attempted_date = today
            self._last_attempt = now

            generated = self._generate(today)
            if generated is not None:
                self._cache_date, self._cache = today, generated
                self._write_disk(today, generated)
                return generated

            # 生成失敗 → 前日分（メモリ or ディスク）→ 中立（空）。
            # フォールバックではモデル違いの mood でも「無音より前日分」で拾う。
            if self._cache is not None:
                return self._cache
            if disk is not None:
                self._cache_date, self._cache = today, disk[2]
                logger.info(
                    "mood: today's generation failed. Substituting with %s's (%s) mood", disk[0], disk[1] or "?",
                )
                return self._cache
            self._cache_date, self._cache = today, {}
            return self._cache

    # --- 内部 -----------------------------------------------------------------

    def _generate(self, today: str) -> dict[str, str] | None:
        cast_ids = [m.id for m in self._cast]
        weather = self._weather.current() if self._weather is not None else None
        prompt = _build_prompt(self._cast, weather, self._today(), self._lang)
        schema = _mood_schema(cast_ids)
        try:
            raw = llm_http.chat(
                self._llm_config, prompt, schema=schema,
                temperature=self._llm_config.temperature,
                timeout_sec=max(self._timeout_sec, self._llm_config.timeout_sec),
            )
            parsed = json.loads(raw)
            known = set(cast_ids)
            out: dict[str, str] = {}
            for item in parsed.get("moods", []):
                cid = item.get("cast_id")
                mood = (item.get("mood") or "").strip().replace("\n", " ")
                if cid in known and mood:
                    out[cid] = mood[:_MAX_MOOD_CHARS.get(self._lang, 60)]
        except (requests.RequestException, ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            logger.warning("mood generation failed: %s", e)
            return None
        logger.info(
            "mood: generated %s's mood (%d/%d people, weather=%s)",
            today, len(out), len(cast_ids),
            weather.description if weather is not None else "none",
        )
        return out

    def _load_disk(self) -> tuple[str, str, dict[str, str]] | None:
        """``(date, model, moods)``。``model`` は旧フォーマットでは ``""``。"""
        try:
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
            date = str(data["date"])
            model = str(data.get("model", ""))
            moods = {str(k): str(v) for k, v in dict(data["moods"]).items()}
        except FileNotFoundError:
            return None
        except (ValueError, KeyError, TypeError, OSError) as e:
            logger.warning("Failed to load mood cache %s: %s", self._cache_path, e)
            return None
        return date, model, moods

    def _write_disk(self, date: str, moods: dict[str, str]) -> None:
        payload = json.dumps(
            {"date": date, "lang": self._lang, "model": self._llm_config.model, "moods": moods},
            ensure_ascii=False, indent=2,
        )
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(
                dir=str(self._cache_path.parent), prefix=".tmp_mood_", suffix=".json"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                    f.write(payload)
                os.replace(tmp, self._cache_path)  # 同一ボリューム内なので原子的
            except Exception:
                Path(tmp).unlink(missing_ok=True)
                raise
        except OSError as e:
            logger.warning("Failed to write mood cache %s: %s", self._cache_path, e)
