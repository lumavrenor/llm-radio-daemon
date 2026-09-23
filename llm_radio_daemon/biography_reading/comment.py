"""偉人伝トーク1チャンクぶんの台本生成（MCの解説＋出演者の反応）。

Ollamaへ渡すのは常に「ローリング要約1本 ＋ いまのチャンクの原文 ＋ 出演者設定」だけ。
記事がどれだけ長くてもプロンプト長は増えない（literary_reading/comment.py と同じ
コンテキスト管理）。literary_reading と違い朗読パートを持たない――このチャンクの
内容そのものをMCがかみ砕いて話す行が毎回のLLM出力になる。
"""

from __future__ import annotations

import json
import logging

import requests

from .. import language, llm_http  # LANGUAGE_GUIDANCE は起動時に差し替わるので属性参照する
from ..config import CastMember, LLMConfig
from ..script import ScriptLine
from ..script.ollama_client import sanitize_text
from .. import sensitive  # SENSITIVE_TOPICS_GUIDANCE も起動時に差し替わるので属性参照する
from ..wiki_bio import SourceChunk

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


_ROLE_NOTES = {
    "ja": ("この記事の内容をかみ砕いて解説する進行役", "感想・質問・軽いツッコミで絡む"),
    "en": ("presents this one, unpacking the article as they go",
           "come in with reactions, questions and the odd bit of needling"),
}


def _roster_entry(m: CastMember, role_note: str) -> str:
    # 声色バリエーションは generated_drama 専用。ここ（偉人伝コーナー）では
    # 既定の声だけを使うので、LLM への「声色」ヒントは出さない。
    en = language.current() == "en"
    return (
        f"- {m.id} ({m.name}): {m.desc} {role_note}" if en
        else f"- {m.id}（{m.name}）: {m.desc} {role_note}"
    )


def _build_prompt(
    *,
    figure_name: str,
    rolling_summary: str,
    chunk: SourceChunk,
    mc: CastMember,
    others: list[CastMember],
    tone_hint: str,
    summary_max_chars: int,
    min_lines: int,
    max_lines: int,
) -> str:
    en = language.current() == "en"
    if not rolling_summary.strip():
        summary = "(this is the very start of the item)" if en else "（まだ紹介の冒頭）"
    else:
        summary = rolling_summary.strip()
    host_note, other_note = _ROLE_NOTES.get(language.current(), _ROLE_NOTES["ja"])
    roster = "\n".join(
        [_roster_entry(mc, host_note)]
        + [_roster_entry(m, other_note) for m in others]
    )
    if chunk.is_section_head and chunk.section:
        section_note = (
            f' (this is where the section "{chunk.section}" begins)' if en
            else f"（ここから見出し「{chunk.section}」に入ったところ）"
        )
    else:
        section_note = ""
    return language.pick(
        ja=f"""あなたはラジオ番組「エルエルエム・ラジオ・デーモン」の放送作家です。
いまは偉人伝コーナーで、Wikipediaの記事をもとに歴史上の人物を紹介しています。

## 今回の人物
{figure_name}（出典: Wikipedia）

## ここまでの紹介内容（LLM用の内部メモ。番組では読み上げない）
{summary}

## 今回取り上げる記事の一部{section_note}
{chunk.text}

## 出演者（speaker にはこの id をそのまま使う）
{roster}

## 今の時間帯のトーン
{tone_hint or "落ち着いたトーンで、じっくりと。"}

## 書き方
- {min_lines}〜{max_lines}行、合計500〜900文字程度で書くこと
- 1人の発言は1〜3文程度。進行役も含め、1つの発言に何段落も詰め込まないこと
- {mc.id} が「今回取り上げる記事の一部」の内容を、自分の言葉でかみ砕いて説明する
- 1行目は「さて、ここからは」「さて、今回は」のような前口上や定型の切り出しで
  始めないこと。前のセグメントからの続きなので、今回のチャンクにある具体的な
  出来事・事実からいきなり入るか、他の出演者の一言を受けて話し出すこと
- 単なる要約の読み上げにしない。{mc.id} は今回のチャンクの中に一つ「山場」や
  「意外な展開」を見つけ、その結末を明かす前に他の出演者へクイズとして振ること。
  例:「さて、ここでニュートンはどうしたと思いますか?」「この状況、二人ならどう動く?」
  他の出演者に一言ずつ予想させ、そのあとで {mc.id} が記事の実際の展開を明かして
  「実は——」と受ける。この振り→予想→答え合わせの流れを毎回1回は入れる
- 全体に前のめりで、驚き・感心・笑いのリアクションをはっきり声に出す。
  相づちだけで流さない
- 他の出演者はクイズへの予想のほか、素朴な疑問・感想・軽いツッコミで絡む。
  全員が均等に話す必要はない
- **この記事に書かれている範囲の事実だけを話すこと。** まだ紹介していない
  この先の出来事を先取りして話さない。クイズの「答え」も必ず今回のチャンクの
  記述の範囲に収めること（この先の展開を答えにしない）
- **記事の一部に書かれていない経緯・エピソード・動機・数値を作り足さないこと。**
  「実は種から育て直した」「親族の家に泊まり込んだ」のような、記述にない具体を
  さも事実のように語らせない。書かれていない部分は「記事にはそこまで書かれていませんが」
  と断るか、触れないこと。予想役の発言も、突飛な作り話ではなく人物像から自然に出る範囲に
- 同じ出演者が前と同じ言い回し・同じ一文を繰り返さないこと。話は毎行進める
- 「ここまでの紹介内容」を復唱しない（文脈として使うだけで、聴いている人に
  向けて繰り返さない）
- 記事に書かれていない固有名詞・数値を断定的に話さないこと。
  確証がないことは「〜らしい」「〜だそうです」のように話すこと
{sensitive.SENSITIVE_TOPICS_GUIDANCE}
- 記号・絵文字・Markdown記法（*, #, ` など）を一切使わないこと。すべて音声で読み上げられる
{language.LANGUAGE_GUIDANCE}
- 英語などの固有名詞（人名・地名・団体名）の扱い:
  発音が確実に分かるものだけ、素直なカタカナ表記にすること。
  読み方に自信がないものは、固有名詞を出さずに一般的な言い方へ言い換えること
  （綴りから発音を推測して不正確なカタカナ読みをでっち上げないこと）
- 話し言葉で書くこと。書き言葉的な硬い表現は避けること

## summary_update
「ここまでの紹介内容」に、今回取り上げた部分の内容を追記・圧縮した新しい版を書く。
古い内容も保持したうえで{summary_max_chars}文字以内にまとめること。
""",
        en=f"""You are the writer for a radio show called "LLM Radio Daemon".
This is the biography corner: they are going through the life of a historical figure,
working from the Wikipedia article.

## Who it is this time
{figure_name} (source: Wikipedia)

## Where the item has got to so far (internal note. Not to be read out)
{summary}

## The part of the article to cover this time{section_note}
{chunk.text}

## On air (use these ids verbatim in speaker)
{roster}

## Tone for this time of day
{tone_hint or "Settled and unhurried."}

## How to write it
- {min_lines} to {max_lines} lines, around 350 to 650 words in total
- One to three sentences per turn, the presenter included. Do not pack several paragraphs into one turn
- {mc.id} explains what is in "the part of the article to cover this time" in their own words
- Do not open with a set phrase like "right, so today" or "now then". This carries on from the
  previous segment, so start straight in on something specific from this part of the article,
  or pick up on what someone else just said
- Do not simply read out a summary. {mc.id} finds one turning point or one surprise in this
  part, and before revealing how it went, puts it to the others as a question.
  For example: "so what do you think Newton did next?" or "in that position, what would you two do?"
  Each of the others has a guess, and then {mc.id} reveals what the article actually says and
  picks it up with "well, in fact...". Work this ask, guess, reveal shape in once every time
- Keep everyone leaning in. Surprise, admiration and laughter are audible, not just murmured agreement
- Besides guessing, the others come in with plain questions, reactions and light needling.
  Nobody has to get equal time
- **Stay inside the facts in this part of the article.** Do not jump ahead to events that have
  not been covered yet. The answer to the question must also be within this part (not something
  that happens later)
- **Do not invent background, anecdotes, motives or numbers that are not in this part of the
  article.** Nothing like "he actually grew it again from seed" or "she stayed with relatives for
  months" unless it is written there. Where the article does not go that far, either say so
  ("the article doesn't say") or leave it alone. The guesses should follow naturally from the
  person as described, not be wild invention
- Nobody repeats a phrase or a sentence they already used. Every line moves it along
- Do not recite "where the item has got to so far" back to the listener. It is context only
- Do not state proper nouns or numbers that are not in the article as if they were fact.
  Where you are not certain, say it as "apparently" or "by all accounts"
{sensitive.SENSITIVE_TOPICS_GUIDANCE}
- No symbols, emoji or Markdown (*, #, ` and so on). Every line is read aloud
{language.LANGUAGE_GUIDANCE}
- Write it as speech. Avoid anything that reads like written prose

## summary_update
Write a new version of "where the item has got to so far", with this part folded in and
compressed. Keep what was already there and bring the whole thing in under
{summary_max_chars} characters.
""",
    )


def generate_segment(
    *,
    figure_name: str,
    rolling_summary: str,
    chunk: SourceChunk,
    mc: CastMember,
    others: list[CastMember],
    llm_config: LLMConfig,
    tone_hint: str = "",
    segment_lines: tuple[int, int] = (5, 10),
    summary_max_chars: int = 800,
) -> tuple[list[ScriptLine], str] | None:
    """行リストと要約更新文字列を返す。失敗時は None（このチャンクは次回リトライ）。"""
    min_lines, max_lines = segment_lines
    appearers = [mc, *others]
    speakers = [m.id for m in appearers]
    by_id = {m.id: m for m in appearers}
    # 声色バリエーションは generated_drama 専用。それ以外の朗読コンテンツでは
    # 既定（先頭）の声だけを使う（違和感が出やすいため）。
    style_names: list[str] = []
    prompt = _build_prompt(
        figure_name=figure_name,
        rolling_summary=rolling_summary,
        chunk=chunk,
        mc=mc,
        others=others,
        tone_hint=tone_hint,
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
                raise ValueError("no valid lines")
            return lines, summary_update

        except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError) as e:
            logger.warning("biography_reading segment generation failed (attempt %d/2): %s", attempt + 1, e)

    logger.error("biography_reading: giving up on segment after 2 attempts")
    return None
