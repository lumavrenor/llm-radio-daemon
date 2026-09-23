"""読書コーナーの選書（LLM キュレーション）— v4 §10.2「作品の選定」。

候補リスト（すべて事前フィルタ済み：PD・翻訳除外・直近読了除外）を Ollama に渡し、
**その中から1つ選ばせる**。時刻・季節・直前に読んだ作品を添えて、深夜の枠に合った
「短め・静かめ」等のキュレーションを効かせる。

- structured output の enum を候補 work_id に固定し、さらに戻り値も候補集合で検証する
  （ハルシネーションで権利存続作品・存在しない作品を掴む事故を防ぐ・§10.2-3）
- 失敗（パース不能・候補外・タイムアウト）したら None を返す。呼び出し側が random に落ちる
"""

from __future__ import annotations

import datetime
import json
import logging

import requests

from .. import language, llm_http
from ..config import LLMConfig
from ..aozora.corpus import Candidate

logger = logging.getLogger(__name__)

_SEASON = {
    12: "冬", 1: "冬", 2: "冬", 3: "春", 4: "春", 5: "春",
    6: "夏", 7: "夏", 8: "夏", 9: "秋", 10: "秋", 11: "秋",
}

_SEASON_EN = {
    12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn",
}

_NDC_TOP = {
    "0": "総記", "1": "哲学・宗教", "2": "歴史・地理", "3": "社会科学",
    "4": "自然科学", "5": "技術・工学", "6": "産業", "7": "芸術・芸能",
    "8": "言語", "9": "文学",
}


def _topic_label(c: Candidate) -> str:
    """候補の「分野」。青空文庫は日本十進分類、Gutenberg は Bookshelves。

    分類体系が違うので同じ列に詰めず、Candidate の別フィールドから選ぶ
    （aozora.corpus.Candidate のコメント参照）。
    """
    if language.current() == "en":
        return c.subjects or "no subject given"
    digits = "".join(ch for ch in c.ndc if ch.isdigit())
    if not digits:
        return "分類なし"
    return _NDC_TOP.get(digits[0], f"分類{digits[0]}")


def _size_label(approx_chars: int | None) -> str:
    if language.current() == "en":
        # 英語は取得済みのときだけ実サイズが分かる（未取得は None）。
        # 語数のほうが尺の見当がつくので、ざっくり 5 文字/語で割って出す。
        if not approx_chars:
            return "length unknown"
        words = approx_chars // 5
        if words < 7500:
            return f"short, about {words} words"
        if words < 40000:
            return f"novella, about {words} words"
        return f"novel-length, about {words} words"
    if not approx_chars:
        return "長さ不明"
    if approx_chars < 8000:
        return f"短編・約{approx_chars}字"
    if approx_chars < 30000:
        return f"中編・約{approx_chars}字"
    return f"長編・約{approx_chars}字"


def select_work_llm(
    candidates: list[Candidate],
    *,
    recent_titles: list[str],
    llm_config: LLMConfig,
    tone_hint: str = "",
    now: datetime.datetime | None = None,
) -> str | None:
    """候補から Ollama に選ばせた work_id を返す。失敗・候補外なら None。"""
    if not candidates:
        return None
    now = now or datetime.datetime.now()
    ids = [c.work_id for c in candidates]
    id_set = set(ids)

    en = language.current() == "en"
    if en:
        listing = "\n".join(
            f"- {c.work_id}: \"{c.title}\" by {c.author or 'an unknown author'}"
            f" / {_topic_label(c)} / {_size_label(c.approx_chars)}"
            for c in candidates
        )
        recent = ", ".join(t for t in recent_titles if t) or "(none)"
    else:
        listing = "\n".join(
            f"- {c.work_id}: 「{c.title}」{c.author or '作者不詳'}"
            f"／{_topic_label(c)}／{_size_label(c.approx_chars)}"
            for c in candidates
        )
        recent = "、".join(t for t in recent_titles if t) or "（なし）"

    prompt = language.pick(
        ja=f"""あなたはラジオ番組「エルエルエム・ラジオ・デーモン」深夜の読書コーナーの選書担当です。
今夜朗読する青空文庫の作品を、下の候補リストから **1つだけ** 選んでください。

## いまの状況
時刻: {now.strftime('%H:%M')}／季節: {_SEASON.get(now.month, '')}
コーナーの雰囲気: {tone_hint or '深夜にひとりで静かに聴く、落ち着いた朗読'}
最近この番組で読んだ作品: {recent}

## 候補（work_id: 作品名 著者／分類／長さ）
{listing}

## 選び方
- 深夜の時間帯に合う、静かめ・短めの作品を優先する
- 最近読んだ作品と作風・ジャンルが続かないよう変化をつける
- 季節に合う題材があれば軽く考慮してよい
- **必ず上の候補の work_id の中から選ぶこと。** リストに無い作品名・IDは絶対に出さない

work_id と、その作品を選んだ短い理由（reason）を返してください。
""",
        en=f"""You choose the books for the reading on a radio show called "LLM Radio Daemon".
Pick **exactly one** work from the list below to read from Project Gutenberg tonight.

## Where things stand
Time: {now.strftime('%H:%M')} / season: {_SEASON_EN.get(now.month, '')}
The feel of the slot: {tone_hint or 'a quiet, unhurried reading, for someone listening alone late at night'}
Read recently on this show: {recent}

## The candidates (work_id: title by author / subject / length)
{listing}

## How to choose
- Prefer something quiet and on the shorter side, suited to the late slot
- Vary it: do not follow a recent pick with something of the same kind or period
- A subject that suits the season is a mild plus
- Reference works, dictionaries, catalogues, collections of documents and anything
  built out of tables or lists do not work read aloud. Prefer prose meant to be read
  straight through: fiction, essays, letters, memoir
- **Choose from the work_id values above and nothing else.** Never give a title or an
  id that is not in the list

Return the work_id and a short reason for choosing it.
""",
    )

    schema = {
        "type": "object",
        "properties": {
            "work_id": {"type": "string", "enum": ids},
            "reason": {"type": "string"},
        },
        "required": ["work_id"],
    }
    try:
        raw = llm_http.chat(
            llm_config, prompt, schema=schema, temperature=llm_config.temperature
        )
        parsed = json.loads(raw)
        work_id = (parsed.get("work_id") or "").strip()
        reason = (parsed.get("reason") or "").strip()
    except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError) as e:
        logger.warning("literary_reading: LLM selection failed (falling back to random): %s", e)
        return None

    if work_id not in id_set:
        logger.warning(
            "literary_reading: LLM returned work_id=%r outside the candidates. Discarding and falling back to random", work_id
        )
        return None

    logger.info("literary_reading: LLM selection -> work_id=%s (reason: %s)", work_id, reason or "―")
    return work_id
