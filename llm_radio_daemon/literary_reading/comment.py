"""Beat での感想生成とローリング要約更新（v4 §10.3）。

Ollama へ渡すのは常に「要約1本 ＋ 直近3チャンクの原文 ＋ 2人のキャラ設定」だけ。
作品が何十万字あってもプロンプト長は増えない（§4.3 のコンテキスト管理と同じ）。

朗読キャラ ⇔ つっこみキャラの掛け合いにする（2人の speaker）。
narrator は朗読専用だが、Beat では短い相づち・疑問で会話に加わる。
パース失敗時は感想を捨てて朗読を続行する（要約は更新しない）。リトライ1回。
"""

from __future__ import annotations

import json
import logging

import requests

from .. import language, llm_http  # LANGUAGE_GUIDANCE は起動時に差し替わるので属性参照する
from ..config import CastMember, LLMConfig
from ..aozora import Chunk
from ..script import ScriptLine
from ..script.ollama_client import sanitize_text

logger = logging.getLogger(__name__)


def _schema(speaker_ids: list[str], style_names: list[str], min_lines: int, max_lines: int) -> dict:
    line_props: dict = {
        "speaker": {"type": "string", "enum": speaker_ids},
        "text": {"type": "string"},
    }
    if style_names:
        line_props["style"] = {"type": "string", "enum": style_names}
    return {
        "type": "object",
        "properties": {
            "summary_update": {"type": "string"},
            "lines": {
                "type": "array",
                "minItems": min_lines,
                "maxItems": max_lines,
                "items": {
                    "type": "object",
                    "properties": line_props,
                    "required": ["speaker", "text"],
                },
            },
        },
        "required": ["summary_update", "lines"],
    }


def _build_prompt(
    *,
    title: str,
    author: str,
    rolling_summary: str,
    recent_chunks: list[Chunk],
    narrator: CastMember,
    commentator: CastMember,
    summary_max_chars: int,
    min_lines: int,
    max_lines: int,
) -> str:
    en = language.current() == "en"
    if not rolling_summary.strip():
        summary = "(still at the very beginning; no summary yet)" if en else "（まだ冒頭。要約なし）"
    else:
        summary = rolling_summary.strip()
    passage = "\n".join(c.display_text for c in recent_chunks)
    return language.pick(
        ja=f"""あなたはラジオ番組「エルエルエム・ラジオ・デーモン」の放送作家です。
いまは深夜の読書コーナーで、青空文庫の作品を朗読しています。
朗読の区切り（Beat）に来たので、出演者2人の短い感想の掛け合いを書いてください。

## 作品
「{title}」（{author}）※ 青空文庫

## ここまでのあらすじ（LLM用の内部メモ。番組では読み上げない）
{summary}

## いま朗読したばかりの部分（原文）
{passage}

## 出演者（speaker にはこの id をそのまま使う）
- {narrator.id}（{narrator.name}）: 朗読担当。基本は聞き役で、短い相づちや素朴な疑問だけ言う
- {commentator.id}（{commentator.name}）: {commentator.desc} 感想・ツッコミ・脱線・時代背景の補足を担当

## 書き方
- {min_lines}〜{max_lines}行、合計300〜600文字。朗読の流れを切りすぎないよう短めに
- 掛け合いにする。{commentator.id} が感想を振り、{narrator.id} が短く受ける
- **ネタバレ厳禁。** 「いま朗読した部分まで」で分かることだけを話す。
  この作品の結末や後の展開を知っていても、絶対に先の話をしない
- あらすじの復唱をしない（リスナーは今聴いたばかり）。感想・疑問・脱線・時代背景の補足に寄せる
- 原文の言い回しをそのまま繰り返さない（朗読と重複して冗長になる）
- 原文にない固有名詞・数値を断定しない。確証がないことは「〜らしい」「〜かも」と言う
- 記号・絵文字・Markdown・アルファベットを使わない。すべて音声で読み上げられる

## summary_update
「ここまでのあらすじ」に、いま朗読した部分の内容を追記・圧縮した新しい版を書く。
古い内容も保持したうえで{summary_max_chars}文字以内にまとめること。ネタバレは書かない。
""",
        en=f"""You are the writer for a radio show called "LLM Radio Daemon".
This is the late-night reading: they are working through a book from Project Gutenberg.
The reading has reached a break, so write the short exchange the two of them have about it.

## The work
"{title}" by {author or "an unknown author"} (from Project Gutenberg)

## The story so far (an internal note for you; it is never read out on air)
{summary}

## The passage they have just read (the original text)
{passage}

## Who is on (use these ids exactly as the speaker)
- {narrator.id} ({narrator.name}): does the reading. Mostly listens here, and only comes
  in with a short agreement or a plain question
- {commentator.id} ({commentator.name}): {commentator.desc} Handles the reactions, the
  needling, the tangents, and any background on the period

## How to write it
- {min_lines} to {max_lines} lines, three hundred to six hundred words in total. Keep it
  short enough that it does not break the thread of the reading
- Make it an exchange: {commentator.id} opens with a reaction, {narrator.id} takes it briefly
- **Never give away what comes later.** Talk only about what can be known from the passage
  just read. Even if you know how this book ends, say nothing about anything ahead
- Do not recap the plot — the listener heard it a moment ago. Stay on reactions,
  questions, tangents and background
- Do not quote the passage back. It has only just been read and repeating it drags
- Do not state a name, a date or a number that is not in the passage. Where you are not
  certain, say so ("I think", "something like")
- Use no symbols, no emoji, no Markdown. Every word is read aloud by a speech synthesiser
{language.LANGUAGE_GUIDANCE}
- Write it as speech, not as prose. Contractions, false starts and short sentences

## summary_update
Write a new version of "The story so far" with the passage just read folded in and
compressed. Keep what was already there and stay under {summary_max_chars} characters.
Nothing about what comes later.
""",
    )


def generate_comment(
    *,
    title: str,
    author: str,
    rolling_summary: str,
    recent_chunks: list[Chunk],
    narrator: CastMember,
    commentator: CastMember,
    llm_config: LLMConfig,
    comment_lines: tuple[int, int] = (4, 8),
    summary_max_chars: int = 800,
) -> tuple[list[ScriptLine], str] | None:
    """感想の行リストと要約更新文字列を返す。失敗時は None（感想を捨てて朗読続行）。"""
    min_lines, max_lines = comment_lines
    speakers = [narrator.id, commentator.id]
    by_id = {narrator.id: narrator, commentator.id: commentator}
    # 声色バリエーションは generated_drama 専用。それ以外の朗読コンテンツでは
    # 既定（先頭）の声だけを使う（違和感が出やすいため）。
    style_names: list[str] = []
    prompt = _build_prompt(
        title=title,
        author=author,
        rolling_summary=rolling_summary,
        recent_chunks=recent_chunks,
        narrator=narrator,
        commentator=commentator,
        summary_max_chars=summary_max_chars,
        min_lines=min_lines,
        max_lines=max_lines,
    )
    schema = _schema(speakers, style_names, min_lines, max_lines)

    for attempt in range(2):
        try:
            raw = llm_http.chat(
                llm_config, prompt, schema=schema, temperature=llm_config.temperature
            )
            parsed = json.loads(raw)
            summary_update = (parsed.get("summary_update") or "").strip()

            lines: list[ScriptLine] = []
            for raw in parsed["lines"]:
                speaker = raw.get("speaker")
                if speaker not in by_id:
                    continue
                text = sanitize_text(raw.get("text") or "")
                if not text:
                    continue
                style = raw.get("style") or None
                if style is not None and style not in style_names:
                    style = None
                lines.append(ScriptLine(speaker=speaker, text=text, style=style))

            if not lines:
                raise ValueError("no valid comment lines")
            return lines, summary_update

        except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError) as e:
            logger.warning("literary_reading comment generation failed (attempt %d/2): %s", attempt + 1, e)

    logger.error("literary_reading: giving up on comment after 2 attempts; continuing reading")
    return None
