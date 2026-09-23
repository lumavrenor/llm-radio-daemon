"""ステージ0「企画立案」（v6 §4.7.1）。

``generated_drama_writer new`` でタイトルを省略したとき、または ``run --auto`` の自動補充で
呼ばれる。トロープのプールから乱数でシード（ネタの素）を組み、LLM に「なろう系／
ラブコメのラノベ企画」の候補を数本出させて自己採点で1本選ぶ。返すのは
``(title, premise)`` のペアだけで、下流の ``GeneratedDramaWriterService.create()`` は無改修。

**放送プロセスからは呼ばれない。** 執筆バッチ（別プロセス）専用。
"""

from __future__ import annotations

import logging
import random

from .. import language
from ..config import LLMConfig, GeneratedDramaParams
from .llm import chat_json

logger = logging.getLogger(__name__)


def _concept_schema(count: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "premise": {"type": "string"},
                        "hook": {"type": "string"},
                    },
                    "required": ["title", "premise", "hook"],
                },
            },
            "pick": {"type": "integer"},
            "reason": {"type": "string"},
        },
        "required": ["candidates", "pick"],
    }


def _build_seed(p: GeneratedDramaParams) -> dict:
    """トロープのプールから乱数で企画のシードを組む。プールが空でも落ちない。"""

    def take(pool: list[str], k: int) -> list[str]:
        pool = [s for s in (pool or []) if s.strip()]
        if not pool:
            return []
        return random.sample(pool, min(k, len(pool)))

    return {
        "genre": (take(p.genres, 1) or [""])[0],
        "tropes": take(p.trope_pool, random.randint(2, 3)),
        "relationship": (take(p.relationship_pool, 1) or [""])[0],
        "twist": (take(p.twist_pool, 1) or [""])[0],
    }


_SEED_LABELS = {
    "ja": ("ジャンル", "取り入れる要素", "主要な関係性", "ひねり", "、",
           "（指定なし。なろう系ラブコメとして自由に発想する）"),
    "en": ("Genre", "Elements to work in", "The central relationship", "The twist", ", ",
           "(nothing specified; invent a genre story freely)"),
}


def _seed_text(seed: dict) -> str:
    genre_l, tropes_l, rel_l, twist_l, joiner, empty = _SEED_LABELS.get(
        language.current(), _SEED_LABELS["ja"]
    )
    lines = []
    if seed.get("genre"):
        lines.append(f"{genre_l}: {seed['genre']}")
    if seed.get("tropes"):
        lines.append(f"{tropes_l}: {joiner.join(seed['tropes'])}")
    if seed.get("relationship"):
        lines.append(f"{rel_l}: {seed['relationship']}")
    if seed.get("twist"):
        lines.append(f"{twist_l}: {seed['twist']}")
    return "\n".join(lines) or empty


def _title_rule(title_style: str) -> str:
    if title_style == "short":
        return language.pick(
            ja="体言中心の短くキャッチーなタイトル（15字以内目安）。",
            en="A short, striking noun-phrase title. Four or five words at most.",
        )
    # なろう系の長いタイトルは日本語のウェブ小説の慣習で、英語圏には対応する型が
    # 無い（config/en は title_style = "short" にしてある）。英語で指定された場合は
    # 「一文の説明的なタイトル」という近い形へ寄せる。
    return language.pick(
        ja=(
            "「主人公の状況／転機／今の立場」を説明的な一文にした、なろう系の長いタイトル"
            "（40〜60字目安）。読点で区切ってよい。体言止めで終えること。"
        ),
        en=(
            "A long, explanatory title that states the hero's situation or the turn their "
            "life has just taken, in one phrase of about twelve to twenty words. "
            "Commas are fine. It must not be a complete sentence with a full stop."
        ),
    )


def _existing_text(existing: list[dict]) -> str:
    if not existing:
        return language.pick(ja="（まだ1本もない）", en="(there are none yet)")
    return "\n".join(
        f"- {e.get('title', '')}: {(e.get('premise', '') or '')[:120]}" for e in existing
    )


def _prompt(p: GeneratedDramaParams, seed: dict, existing: list[dict], steer: str) -> str:
    steer_ja = (
        f"## 追加の狙い（人間からの指定）{chr(10)}{steer}{chr(10)}" if steer.strip() else ""
    )
    steer_en = (
        f"## Extra steer (specified by a human){chr(10)}{steer}{chr(10)}" if steer.strip() else ""
    )
    return language.pick(
        ja=f"""あなたはラノベレーベルの編集者です。深夜ラジオで1シーンずつ朗読される
オリジナルのライトノベルを新しく1本立ち上げます。日本語話者のオタク層に刺さり、
かつ isekai / romance-comedy / villainess のように英語圏のオタクにも鉄板の題材で
考えてください。企画の候補を{p.concept_candidates}本出し、その中で最も面白い1本を選んでください。

## 必ず織り込むシード
{_seed_text(seed)}
{steer_ja}
## 既存の連載（主題・設定がかぶらないこと）
{_existing_text(existing)}

## 各候補に書くこと
- title: {_title_rule(p.title_style)}
- premise: どんな話か、何が読みどころか、を3〜5文で。主人公と目的とヒロイン格の関係が分かること
- hook: この企画がオタクに刺さる理由を一文で

## 制約
- 記号・絵文字・アルファベットを使わないこと（タイトルも本文もすべて音声で読み上げられる）
- 起承転結が付けられる、全8章程度で完結できる規模にすること
- pick は1から始まる番号で、candidates の何本目を選んだか
""",
        # 日本語版は「なろう系ラブコメ」という的が定まっているので候補が散らからない。
        # 英語にはそれに当たる単一の型が無いので、代わりに config の genres プールが
        # シードとして的を絞る（config/en/config_content.toml の genres を参照）。
        en=f"""You are a commissioning editor for a fiction imprint. You are starting a new
original serial that will be read out one scene at a time on a late-night radio show.
It should be the kind of story that keeps someone listening in the dark with the
lights off: a clear hook, characters worth following, and a reason to come back
tomorrow night. Propose {p.concept_candidates} candidates, then pick the best one.

## Seed you must work in
{_seed_text(seed)}
{steer_en}
## Serials that already exist (do not overlap with their premise or setting)
{_existing_text(existing)}

## What to write for each candidate
- title: {_title_rule(p.title_style)}
- premise: three to five sentences on what the story is and what makes it worth
  hearing. Make the protagonist, what they want, and who matters most to them clear
- hook: one sentence on why a listener would stay for this

## Constraints
- Every word of this, the title included, will be spoken by a speech synthesiser.
  Use no symbols, no emoji, no digits, no abbreviations. Write numbers out as words
- Keep it to a scale that resolves in about eight chapters, with a beginning,
  a turn and an ending. It is not an open-ended series
- pick is one-based: which of the candidates you chose
""",
    )


def generate_concept(
    llm_config: LLMConfig,
    params: GeneratedDramaParams,
    existing: list[dict],
    *,
    steer: str = "",
    timeout_sec: int | None = None,
) -> tuple[str, str] | None:
    """企画を1本立てて ``(title, premise)`` を返す。失敗したら ``None``。

    ``premise`` にはシード要素と選ばれた候補の premise / hook をまとめて詰める。
    ``GeneratedDramaWriterService._design_prompt`` がそのまま「狙い・題材」として使う。
    """
    seed = _build_seed(params)
    logger.info("generated_drama_writer: concept planning seed %s", seed)

    result = chat_json(
        llm_config,
        _prompt(params, seed, existing, steer),
        _concept_schema(params.concept_candidates),
        timeout_sec=timeout_sec or params.writer_timeout_sec,
        max_tokens=3072,
    )
    if result is None:
        logger.error("generated_drama_writer: failed to generate concept")
        return None

    candidates = [c for c in result.get("candidates", []) if str(c.get("title", "")).strip()]
    if not candidates:
        logger.error("generated_drama_writer: no concept candidates were returned")
        return None

    try:
        pick = int(result.get("pick", 1))
    except (TypeError, ValueError):
        pick = 1
    chosen = candidates[pick - 1] if 1 <= pick <= len(candidates) else candidates[0]

    title = str(chosen.get("title", "")).strip()
    premise = _compose_premise(seed, chosen, steer)
    logger.info(
        "generated_drama_writer: selected the concept (out of %d candidates, picked #%d): %s",
        len(candidates), pick if 1 <= pick <= len(candidates) else 1, title,
    )
    return title, premise


_PREMISE_LABELS = {
    "ja": ("読みどころ", "企画のシード", "人間からの狙い"),
    "en": ("The hook", "Seed this came from", "Steer from a human"),
}


def _compose_premise(seed: dict, chosen: dict, steer: str) -> str:
    hook_l, seed_l, steer_l = _PREMISE_LABELS.get(
        language.current(), _PREMISE_LABELS["ja"]
    )
    parts = [str(chosen.get("premise", "")).strip()]
    hook = str(chosen.get("hook", "")).strip()
    if hook:
        parts.append(f"{hook_l}: {hook}")
    parts.append(f"{seed_l}: {_seed_text(seed).replace(chr(10), ' / ')}")
    if steer.strip():
        parts.append(f"{steer_l}: {steer.strip()}")
    return "\n".join(p for p in parts if p)
