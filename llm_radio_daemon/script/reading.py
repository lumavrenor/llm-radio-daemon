"""台本テキストから読み上げ用テキストをLLMで生成する（TTSの誤読対策）。

字幕（AnnounceThread）は常に ScriptLine.text（生成された原文）を表示する。
このモジュールは、その text を読み間違えやすい漢字だけ読みが一意に
決まる表記へ書き換えた speech_text を別途LLMで作り、ScriptLine に足す。
TTSThread は speech_text があればそちらを合成に使う（tts_thread.py）。

台本生成に使う本体モデルへもう1往復投げるだけなので、有効にすると台本生成が
その分遅くなる（2B級の小型モデルでは「一部の語だけ直す」という選択的編集が
そもそも成立しない実測あり。行数を1行ずつに割っても改善しなかった＝
バッチサイズではなくモデルの instruction-following 能力の問題）。

失敗時（LLM応答が壊れている・行数が合わない等）は何もしない。
speech_text が None のままの行は従来どおり text がそのままTTSに渡る。

【2026-09-20 実測メモ】台本生成と同じ1回の構造化出力へ speech_text を統合する案を
試したが、qwen3.8:27b でも本編と同じ temperature（既定0.9・創作寄り）で生成すると
「一部の語だけ直す」指示を無視して行全体をかな化し、しかも一部で誤変換（例:
「大胆な彩色」→「だれんなさいしき」、「特別な機会」→「とくべなきかい」）まで発生した。
このモジュールの低温（0.2固定）の別呼び出し方式に戻し、統合版は不採用とした。
"""

from __future__ import annotations

import json
import logging
import re

from .. import llm_http
from ..config import LLMConfig
from . import ScriptLine

logger = logging.getLogger(__name__)

# 機械的な書き換えタスクなので、台本生成に使う llm_config.temperature
# （創作寄りに高め、既定0.9）とは別に低温で固定する。
_TEMPERATURE = 0.2

# プロンプト側で行に振っている「1: 」「2: 」の番号を、モデルがそのまま出力へ
# 混ぜて返してくることがある（gemma4:12b, qwen3.5:4b で実測）。中身は壊れて
# いないので、この場合は番号だけ剥がして使う。1〜2桁＋コロンに絞り、
# 「3、2、1、それでは」のような台詞中のカウントダウンとは区別する。
_LEAKED_NUMBERING = re.compile(r"^\s*\d{1,2}\s*[:：]\s*")

_PROMPT_TEMPLATE = """次の{n}行のセリフを、日本語音声合成（VOICEVOX）が正しく読めるように書き換えてください。

ルール（これだけ）:
1. 「今日」「方」「一分」のように、読み方が複数ある漢字を見つけたら、
   文脈から正しい読みを判断してひらがなに書き換える
   （今日→きょう/こんにち、方→かた/ほう、一分→いっぷん/いちぶん、など）
2. それ以外の漢字・言い回し・すでにひらがなの部分は絶対に変更しない
3. 該当する漢字が無い行は、入力と完全に同じ文字列をそのまま出力する
4. 行数・順序は入力どおり{n}行のまま

セリフ:
{lines}
"""


def _build_prompt(texts: list[str]) -> str:
    numbered = "\n".join(f"{i + 1}: {t}" for i, t in enumerate(texts))
    return _PROMPT_TEMPLATE.format(n=len(texts), lines=numbered)


def _schema(n: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "speech_lines": {
                "type": "array",
                "minItems": n,
                "maxItems": n,
                "items": {"type": "string"},
            }
        },
        "required": ["speech_lines"],
    }


def apply_speech_reading(lines: list[ScriptLine], llm_config: LLMConfig) -> None:
    """lines の各 ScriptLine.speech_text を in-place でセットする。失敗時は無処理。"""
    if not llm_config.speech_reading_pass or not lines:
        return

    texts = [line.text for line in lines]
    prompt = _build_prompt(texts)
    try:
        raw = llm_http.chat(
            llm_config, prompt, schema=_schema(len(texts)), temperature=_TEMPERATURE
        )
        parsed = json.loads(raw)
        speech_lines = parsed["speech_lines"]
        if not isinstance(speech_lines, list) or len(speech_lines) != len(texts):
            raise ValueError(
                f"line count mismatch: got {len(speech_lines) if isinstance(speech_lines, list) else 'non-list'}, "
                f"want {len(texts)}"
            )
    except Exception as e:
        logger.warning("speech reading pass failed, falling back to text as-is: %s", e)
        return

    # ollama_client.sanitize_text と循環importになるためローカルimport。
    from .ollama_client import sanitize_text

    for line, speech in zip(lines, speech_lines):
        if not isinstance(speech, str):
            continue
        speech = _LEAKED_NUMBERING.sub("", speech)
        cleaned = sanitize_text(speech)
        if cleaned:
            line.speech_text = cleaned
