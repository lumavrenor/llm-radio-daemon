"""翻訳朗読コーナーのLLM呼び出し3種。

1. :func:`translate_title` ―― セッション開始時に1回だけ。英語の書名・著者名を、
   音声で読める日本語（訳題・カタカナ表記）に変換する。**これが無いと、原文の
   英語タイトルが番組内でそのままアルファベット読み上げにされてしまう**
   （language.LANGUAGE_GUIDANCE の「原稿にアルファベットを残さない」に反する）。
   失敗時は原題をそのまま使う（多少発音が崩れても、コーナーを止めるよりまし）。
2. :func:`generate_translation` ―― 原文チャンク1つを日本語ナレーションへ翻訳する
   （biography_reading/comment.py と同じ「要約1本＋今回のチャンク原文」の構成）。
   speaker は常にナレーター1人（朗読役のセリフ化・docs/idea-translated-reading.md）。
3. :func:`generate_beat` ―― literary_reading と同じ、数チャンクごとに挟む感想パート。
   ただし「いま朗読した部分」は原文（英語）ではなく、直近に訳した日本語文を渡す
   （英語へ戻らずに済み、感想パートも一貫して日本語だけで完結する）。
"""

from __future__ import annotations

import json
import logging
import re

import requests

from .. import language, llm_http  # LANGUAGE_GUIDANCE は起動時に差し替わるので属性参照する
from ..config import CastMember, LLMConfig
from ..aozora import Chunk
from ..script import ScriptLine
from ..script.ollama_client import sanitize_text

logger = logging.getLogger(__name__)


# --- 1. 書名・著者名の翻訳（セッション開始時に1回） -------------------------

def translate_title(title: str, author: str, llm_config: LLMConfig) -> tuple[str, str]:
    """(title_ja, author_ja) を返す。失敗時は原題・原著者名をそのまま返す。"""
    prompt = f"""次の英語の書名と著者名を、ラジオ番組で声に出して読めるように日本語へ変換してください。

書名: {title}
著者: {author or "unknown"}

- 書名は、定訳（既存の邦題）があればそれを使うこと。無ければ意味の通る日本語訳、
  もしくは自然なカタカナ表記にすること
- 著者名は素直なカタカナ表記にすること（例: Charles Dickens → チャールズ・ディケンズ）
- 読み方に自信がない固有名詞を、綴りから推測してでっち上げないこと
- 記号・括弧・引用符を付けず、変換した文字列だけを返すこと
"""
    schema = {
        "type": "object",
        "properties": {
            "title_ja": {"type": "string"},
            "author_ja": {"type": "string"},
        },
        "required": ["title_ja", "author_ja"],
    }
    try:
        raw = llm_http.chat(llm_config, prompt, schema=schema, temperature=llm_config.temperature)
        parsed = json.loads(raw)
        title_ja = (parsed.get("title_ja") or "").strip()
        author_ja = (parsed.get("author_ja") or "").strip()
        if title_ja:
            return title_ja, (author_ja or author)
    except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError) as e:
        logger.warning("translated_reading: failed to translate title/author (continuing with original title): %s", e)
    return title, author


# --- 2. チャンクの翻訳ナレーション ------------------------------------------

def _translate_schema(style_names: list[str], min_lines: int, max_lines: int) -> dict:
    line_props: dict = {"text": {"type": "string"}}
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
                    "required": ["text"],
                },
            },
        },
        "required": ["summary_update", "lines"],
    }


def _build_translate_prompt(
    *,
    title: str,
    author: str,
    rolling_summary: str,
    chunk: Chunk,
    narrator: CastMember,
    tone_hint: str,
    summary_max_chars: int,
    min_lines: int,
    max_lines: int,
) -> str:
    summary = rolling_summary.strip() or "（まだ翻訳の冒頭）"
    chapter_note = "（ここから新しい章に入るところ）" if chunk.is_chapter_head else ""
    return f"""あなたはラジオ番組「エルエルエム・ラジオ・デーモン」の放送作家です。
いまは翻訳朗読コーナーで、英語の作品（Project Gutenberg）を原文のニュアンスを
保ちながら日本語に訳してナレーションしています。

## 作品
「{title}」（{author or "著者不詳"}、Project Gutenberg）

## ここまでの翻訳のあらすじ（LLM用の内部メモ。番組では読み上げない）
{summary}

## 今回訳す原文の一節{chapter_note}
{chunk.display_text}

## ナレーター（speaker は常にこの1人）
- {narrator.id}（{narrator.name}）: {narrator.desc}

## 今の時間帯のトーン
{tone_hint or "落ち着いたトーンで、じっくりと。"}

## 書き方
- {min_lines}〜{max_lines}行。原文の分量に見合うだけの日本語で書くこと（大胆に
  端折らない。ただし逐語訳のような硬い直訳にもしないこと）
- **これは翻訳であり要約ではない。** 原文にある出来事・セリフ・描写を、物語の
  順番どおりに漏らさず訳すこと
- 地の文は落ち着いた朗読口調、会話文は自然な話し言葉のセリフにすること
- 人名・地名は「ここまでの翻訳のあらすじ」に出ている表記があれば、それに揃えること
  （章をまたいで同じ人物の呼び方が変わらないように）
- 原文に書かれていない出来事・セリフ・心情を書き足さないこと
- この一節の内容を先取りしすぎたり、省略しすぎたりしないこと（続きは次のチャンクへ渡る）
- 記号・絵文字・Markdown記法（*, #, ` など）を一切使わないこと。すべて音声で読み上げられる
{language.LANGUAGE_GUIDANCE}

## summary_update
「ここまでの翻訳のあらすじ」に、今回訳した部分の内容を追記・圧縮した新しい版を書く。
人名・地名の表記もできれば残すこと。古い内容も保持したうえで{summary_max_chars}文字
以内にまとめること。
"""


def _is_untranslated_leak(text: str) -> bool:
    """原文の英語をほぼそのままコピーしてきていないか。

    小型モデル（gemma4:e2b など）は稀に「翻訳」と指示しても原文の英語をそのまま
    text に流し込んでくる（プロンプトの「アルファベットを残さない」指示を無視する）。
    サニタイズでは救えないので、文字種の比率で検出して翻訳失敗として扱い、
    リトライさせる（放送言語が日本語のときだけ。放送言語が英語の設定では原文と
    重なって当然なので対象外）。
    """
    letters = sum(1 for c in text if c.isalpha())
    if letters < 6:
        return False
    latin = sum(1 for c in text if c.isalpha() and c.isascii())
    return latin / letters > 0.5


def generate_translation(
    *,
    title: str,
    author: str,
    rolling_summary: str,
    chunk: Chunk,
    narrator: CastMember,
    llm_config: LLMConfig,
    tone_hint: str = "",
    translate_lines: tuple[int, int] = (1, 4),
    summary_max_chars: int = 800,
) -> tuple[list[ScriptLine], str] | None:
    """翻訳ナレーションの行リストと要約更新文字列を返す。失敗時は None（次回リトライ）。"""
    min_lines, max_lines = translate_lines
    # 声色バリエーションは generated_drama 専用。それ以外の朗読コンテンツでは
    # 既定（先頭）の声だけを使う（違和感が出やすいため）。
    style_names: list[str] = []
    prompt = _build_translate_prompt(
        title=title,
        author=author,
        rolling_summary=rolling_summary,
        chunk=chunk,
        narrator=narrator,
        tone_hint=tone_hint,
        summary_max_chars=summary_max_chars,
        min_lines=min_lines,
        max_lines=max_lines,
    )
    schema = _translate_schema(style_names, min_lines, max_lines)

    for attempt in range(2):
        try:
            raw = llm_http.chat(
                llm_config, prompt, schema=schema, temperature=llm_config.temperature
            )
            parsed = json.loads(raw)
            summary_update = (parsed.get("summary_update") or "").strip()

            lines: list[ScriptLine] = []
            for raw_line in parsed["lines"]:
                text = sanitize_text(raw_line.get("text") or "")
                if not text:
                    continue
                if language.current() != "en" and _is_untranslated_leak(text):
                    raise ValueError(f"untranslated English text leaked: {text[:60]!r}")
                style = raw_line.get("style") or None
                if style is not None and style not in style_names:
                    style = None
                lines.append(ScriptLine(speaker=narrator.id, text=text, style=style))

            if not lines:
                raise ValueError("no valid translation lines")
            return lines, summary_update

        except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError) as e:
            logger.warning("translated_reading translation failed (attempt %d/2): %s", attempt + 1, e)

    logger.error("translated_reading: giving up on translation after 2 attempts")
    return None


# --- 3. Beat（感想パート） --------------------------------------------------

def _beat_schema(speaker_ids: list[str], style_names: list[str], min_lines: int, max_lines: int) -> dict:
    line_props: dict = {
        "speaker": {"type": "string", "enum": speaker_ids},
        "text": {"type": "string"},
    }
    if style_names:
        line_props["style"] = {"type": "string", "enum": style_names}
    return {
        "type": "object",
        "properties": {
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
        "required": ["lines"],
    }


def _build_beat_prompt(
    *,
    title: str,
    author: str,
    recent_translated_text: str,
    narrator: CastMember,
    commentator: CastMember,
    min_lines: int,
    max_lines: int,
) -> str:
    return f"""あなたはラジオ番組「エルエルエム・ラジオ・デーモン」の放送作家です。
いまは翻訳朗読コーナーで、英語の作品（Project Gutenberg）を日本語に訳しながら
ナレーションしています。朗読の区切り（Beat）に来たので、出演者2人の短い感想の
掛け合いを書いてください。

## 作品
「{title}」（{author or "著者不詳"}、Project Gutenberg。日本語訳でお届け中）

## いま訳して読んだばかりの部分（日本語訳）
{recent_translated_text}

## 出演者（speaker にはこの id をそのまま使う）
- {narrator.id}（{narrator.name}）: 翻訳ナレーション担当。基本は聞き役で、短い相づちや素朴な疑問だけ言う
- {commentator.id}（{commentator.name}）: {commentator.desc} 感想・ツッコミ・脱線を担当

## 書き方
- {min_lines}〜{max_lines}行、合計300〜600文字。翻訳の流れを切りすぎないよう短めに
- 掛け合いにする。{commentator.id} が感想を振り、{narrator.id} が短く受ける
- **ネタバレ厳禁。** 「いま訳して読んだ部分まで」で分かることだけを話す
- あらすじの復唱をしない（リスナーは今聴いたばかり）。感想・疑問・脱線に寄せる
- 翻訳文の言い回しをそのまま繰り返さない（朗読と重複して冗長になる）
- 書かれていない固有名詞・数値を断定しない。確証がないことは「〜らしい」「〜かも」と言う
- 記号・絵文字・Markdown・アルファベットを使わない。すべて音声で読み上げられる
{language.LANGUAGE_GUIDANCE}
"""


def generate_beat(
    *,
    title: str,
    author: str,
    recent_translated_text: str,
    narrator: CastMember,
    commentator: CastMember,
    llm_config: LLMConfig,
    comment_lines: tuple[int, int] = (4, 8),
) -> list[ScriptLine] | None:
    """感想の行リストを返す。失敗時は None（感想を捨てて翻訳朗読を続行）。"""
    if not recent_translated_text.strip():
        return None
    min_lines, max_lines = comment_lines
    speakers = [narrator.id, commentator.id]
    by_id = {narrator.id: narrator, commentator.id: commentator}
    # 声色バリエーションは generated_drama 専用。それ以外の朗読コンテンツでは
    # 既定（先頭）の声だけを使う（違和感が出やすいため）。
    style_names: list[str] = []
    prompt = _build_beat_prompt(
        title=title,
        author=author,
        recent_translated_text=recent_translated_text,
        narrator=narrator,
        commentator=commentator,
        min_lines=min_lines,
        max_lines=max_lines,
    )
    schema = _beat_schema(speakers, style_names, min_lines, max_lines)

    for attempt in range(2):
        try:
            raw = llm_http.chat(
                llm_config, prompt, schema=schema, temperature=llm_config.temperature
            )
            parsed = json.loads(raw)

            lines: list[ScriptLine] = []
            for raw_line in parsed["lines"]:
                speaker = raw_line.get("speaker")
                if speaker not in by_id:
                    continue
                text = sanitize_text(raw_line.get("text") or "")
                if not text:
                    continue
                style = raw_line.get("style") or None
                if style is not None and style not in style_names:
                    style = None
                lines.append(ScriptLine(speaker=speaker, text=text, style=style))

            if not lines:
                raise ValueError("no valid comment lines")
            return lines

        except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError) as e:
            logger.warning("translated_reading beat generation failed (attempt %d/2): %s", attempt + 1, e)

    logger.error("translated_reading: giving up on beat after 2 attempts; continuing translated reading")
    return None
