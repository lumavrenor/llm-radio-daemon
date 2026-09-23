"""リクエスト受付のアナウンス（§6.2）。

``request now <コーナー>`` で番組表を上書きしたとき、黙って番組が切り替わるので
はなく、DJ が一度「リクエストをいただきました」と受けてからコーナーへ入る。
ここが作るのは**その入り口の短い台本1本だけ**。番組進行（director.py）が、
暗転中に次のコーナーの準備と並べてこれを生成・合成しておき、明転した直後・
コーナーの最初のトークより前に流す。台本の最後の行に request_id を立てておき、
その行が再生され始めた時点で AnnounceThread が DB へ entered_at を記録する。

LLM に書かせ、失敗したらテンプレートへ落とす（filler.py と同じ構え）。
テンプレートは LLM を通らない＝言語指示が効かないので、定型文を言語ごとに持つ。

**リクエスト主の名前は絶対に作らせない。** ラジオの型に寄せると LLM は
「ラジオネーム○○さんから」と平気ででっち上げる。誰が送ったかはこちらも
知らないので、プロンプトで明示的に禁じている（この番組の「知らないことを
断定させない」方針どおり）。
"""

from __future__ import annotations

import json
import logging

import requests

from .. import language  # LANGUAGE_GUIDANCE は起動時に差し替わるので属性参照する
from .. import llm_http
from ..config import CastMember, ContentConfig, LLMConfig
from . import Script, ScriptLine
from .ollama_client import sanitize_text

logger = logging.getLogger(__name__)

# 「今すぐ聴きたい」に応える場面なので、フィラーよりさらに短く待つ。
# ここで待たされるとリクエストの体感が鈍る（超えたらテンプレートで即座に流す）。
_LLM_TIMEOUT_SEC = 15
_MIN_LINES = 2
_MAX_LINES = 5


def _prompt(appearers: list[CastMember], content: ContentConfig) -> str:
    roster = "\n".join(f"- {m.id}（{m.name}）: {m.desc}" for m in appearers)
    tone = content.tone_hint or "（指定なし）"
    return f"""あなたはラジオ番組「エルエルエム・ラジオ・デーモン」の放送作家です。
放送中にリスナーからリクエストが届き、番組表を変更して「{content.display_label}」の
コーナーへ入ります。その切り替えの入り口になる、ごく短い台本を作ってください。

## 出演者（speaker にはこの id をそのまま使うこと。先頭が進行役）
{roster}

## これから入るコーナー
「{content.display_label}」
このコーナーのトーン: {tone}

## 書くこと
- リクエストが届いたことを進行役が受け、これから「{content.display_label}」へ入る、と渡すだけ
- {_MIN_LINES}〜{_MAX_LINES}行。短くてよい。一往復で足りるならそれでよい

## 書いてはいけないこと
- **リクエスト主の名前・ラジオネーム・地域・メッセージ本文を作らないこと。**
  誰が送ったかは分かっていない。「リクエストをいただきました」までにとどめること
- コーナーの中身（どの記事・どの曲・どの作品か）に触れないこと。まだ決まっていない
- リクエストの受付方法・番号・メールアドレス・SNSの案内をしないこと。この番組には無い
- 大げさに盛り上げないこと。深夜でも昼でも成立する、落ち着いた受け方にすること

## 台本の書き方
{language.LANGUAGE_GUIDANCE}
- 1行はひと息で読める長さにすること
"""


def _schema(speaker_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "lines": {
                "type": "array",
                "minItems": _MIN_LINES,
                "maxItems": _MAX_LINES,
                "items": {
                    "type": "object",
                    "properties": {
                        "speaker": {"type": "string", "enum": speaker_ids},
                        "text": {"type": "string"},
                    },
                    "required": ["speaker", "text"],
                },
            }
        },
        "required": ["lines"],
    }


def _generate_llm(
    llm_config: LLMConfig, appearers: list[CastMember], content: ContentConfig, request_id: int
) -> Script | None:
    ids = [m.id for m in appearers]
    try:
        raw = llm_http.chat(
            llm_config,
            _prompt(appearers, content),
            schema=_schema(ids),
            temperature=llm_config.temperature,
            timeout_sec=min(llm_config.timeout_sec, _LLM_TIMEOUT_SEC),
        )
        parsed = json.loads(raw)
        lines = [
            ScriptLine(speaker=l["speaker"], text=sanitize_text(l["text"]))
            for l in parsed["lines"]
            if l.get("speaker") in ids and sanitize_text(l.get("text", ""))
        ]
    except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError) as e:
        logger.warning("Failed to generate request announcement (falling back to template): %s", e)
        return None

    if len(lines) < _MIN_LINES:
        return None
    return _script(lines, appearers, content, "llm", request_id)


# LLM を通らない保険。ここだけは言語指示が効かないので定型文を言語ごとに持つ。
_TEMPLATES = {
    "ja": (
        "さて、ここでリクエストをいただきました。",
        "ありがとうございます。",
        "それでは、{label}、いってみましょう。",
    ),
    "en": (
        "And we've just had a request come in.",
        "Thank you for that.",
        "So, let's head over to {label}.",
    ),
}


def _generate_template(
    appearers: list[CastMember], content: ContentConfig, request_id: int
) -> Script:
    """LLM なしでアナウンスを1本作る（_generate_llm 失敗時の保険）。"""
    texts = _TEMPLATES.get(language.current(), _TEMPLATES["ja"])
    host = appearers[0]
    other = appearers[1] if len(appearers) > 1 else host
    # 受ける・相づち・渡す、の3行。受けと渡しは同じ進行役に持たせる
    # （3人目に締めを任せると、誰が仕切っているのか分からなくなる）。
    speakers = (host, other, host)
    lines = [
        ScriptLine(speaker=s.id, text=t.format(label=content.display_label))
        for s, t in zip(speakers, texts)
    ]
    return _script(lines, appearers, content, "template", request_id)


def _script(
    lines: list[ScriptLine],
    appearers: list[CastMember],
    content: ContentConfig,
    origin: str,
    request_id: int,
) -> Script:
    # is_filler は立てない。フィラー扱いにすると ON AIR ランプが点滅する。
    # これはリクエストされたコーナーの本編の入り口。
    #
    # 最後の行にだけ request_id を立てる。その行の再生が始まった時点で
    # AnnounceThread が mark_request_entered() で記録する（§6.2）。
    lines[-1].request_id = request_id
    return Script(
        topic_id=None,
        topic_title=f"request:{origin} ({content.display_label})",
        lines=lines,
        appearer_ids=tuple(m.id for m in appearers),
    )


def generate_request_announce(
    llm_config: LLMConfig, appearers: list[CastMember], content: ContentConfig, request_id: int
) -> Script | None:
    """「リクエストをいただきました」の台本。出演者が空なら None。"""
    if not appearers:
        return None
    script = _generate_llm(llm_config, appearers, content, request_id)
    return script if script is not None else _generate_template(appearers, content, request_id)
