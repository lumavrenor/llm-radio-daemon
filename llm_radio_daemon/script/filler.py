"""フィラートーク（原稿が枯れたときのつなぎ）。

SPEC.md 8章「無音を作らない」の第1段階：script_queue が空のまま放置されたら、
ここでつなぎのトークを流し込む。

2 段構え：

1. まず LLM に短い雑談を書かせる（``generate_llm_filler``）。テンプレートだけだと
   24 時間で反復感が強すぎるため、バリエーションは LLM に任せる。
2. LLM 呼び出しが失敗／タイムアウトしたら、事前に用意したテンプレート
   （``generate_filler``）へフォールバックする。こちらはネットワーク I/O なしで
   即座に返るので、「詰まったからフィラーを出したいのに生成自体が詰まる」状況でも
   最低限の場つなぎは保証される。

雑談の「お題」自体も同じ構え。``generate_llm_topic`` が台本を書くのと同じ LLM に
一言だけ考えさせ（ネタだし）、その結果を上の ``generate_llm_filler`` の
``topic_seed`` に渡す（トーク組み立ては引き続きそちら側の役目）。固定リストだと
いくら行を増やしても有限で、どのモデルに喋らせても入り口が同じになってしまうため
――お題の質もモデルの地力の差として出したい。失敗／タイムアウトしたときは
お題なし（呼び出し元は ``threads/filler_thread.py::FillerThread._pick_topic``）。

どちらの経路でも、番組が動いている LLM のモデル名と推論エンジン名に一度は触れる
（楽屋ネタとして）。「このパソコンの中で動いている」かどうかは決め打ちにせず、
``LLMConfig.placement``（起動時に llm_http.detect_placement() が判定）に従う ——
ollama の `<model>-cloud` はローカルと同じ host から使えてしまうため、決め打ちだと
クラウド実行に切り替えた瞬間に台詞が嘘になる。判定できないときは実行場所に触れさせない。

今の天気（``[weather]``）も同じ扱いで、``weather.WeatherProvider`` が取れた値だけを
「渡した事実」としてプロンプトに書き、取れなければ（``None``）天気に一切触れさせない。
数字を渡す以上、書いていない予報を推測で足されるのがいちばん困るため。

言語（``[locale] lang``）
------------------------
**テンプレート側は LLM を通らないので language.LANGUAGE_GUIDANCE が効かない。**
放送開始直後はモデルのロードに数十秒かかり、その間の LLM フィラーは
``_LLM_FILLER_TIMEOUT_SEC`` で必ず落ちてここへ来るため、英語放送の一発目の字幕が
まるごと日本語になっていた。定型文は ``request_announce._TEMPLATES`` と同じく
言語ごとのテーブルに分け、``_tbl()`` で引く。

英語の定型文で時刻や気温を出すときは数字のままにせず語に開くこと（``_spell_clock`` /
``_spell_number``）。Kokoro の G2P はコロン混じりの "2:30" を音素化できない
（language.py 参照）。プロンプト経由なら LANGUAGE_GUIDANCE が同じことを指示して
いるが、テンプレートはそこを通らないので自分で開く。
"""

from __future__ import annotations

import json
import logging
import random
import time

import requests

from .. import llm_http
from ..config import CastMember, LLMConfig
from .. import language  # LANGUAGE_GUIDANCE は起動時に差し替わるので属性参照する
from .. import sensitive  # SENSITIVE_TOPICS_GUIDANCE も起動時に差し替わるので属性参照する
from ..weather import Weather
from . import Script, ScriptLine
from .ollama_client import sanitize_text

logger = logging.getLogger(__name__)

# LLM フィラーは「詰まっている」状況で呼ばれるので、通常の台本生成より短く待つ。
_LLM_FILLER_TIMEOUT_SEC = 30
_LLM_MIN_LINES = 6
_LLM_MAX_LINES = 14

# ネタだし（generate_llm_topic）は一言だけ返させる軽い呼び出しなので、トーク組み立て
# より短いタイムアウトで見切る（失敗したらお題なしで進める）。
_LLM_TOPIC_TIMEOUT_SEC = 15


def _tbl(table: dict):
    """言語ごとの定型文テーブルから今の言語ぶんを引く（無ければ日本語）。"""
    return table.get(language.current(), table["ja"])


# --- 英語テンプレート用の数詞（LLM を通らないので自前で開く）-------------------

_ONES = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty")


def _spell_number(n: int) -> str:
    """0〜59 を語に開く。時刻・気温のテンプレート用（範囲外はそのまま数字で返す）。"""
    if not 0 <= n < 60:
        return str(n)
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] if ones == 0 else f"{_TENS[tens]}-{_ONES[ones]}"


def _spell_clock(hour: int, minute: int) -> str:
    """24時間の時分を英語の話し言葉に開く（例: 14:05 → "two oh five in the afternoon"）。

    Kokoro が "2:05" を読めないので、テンプレート側で語にしておく（モジュール
    docstring 参照）。
    """
    h12 = hour % 12 or 12
    if 5 <= hour < 12:
        part = "in the morning"
    elif 12 <= hour < 18:
        part = "in the afternoon"
    elif 18 <= hour < 22:
        part = "in the evening"
    else:
        part = "at night"
    if minute == 0:
        return f"{_spell_number(h12)} o'clock {part}"
    if minute < 10:
        return f"{_spell_number(h12)} oh {_spell_number(minute)} {part}"
    return f"{_spell_number(h12)} {_spell_number(minute)} {part}"


# --- LLM 生成フィラー -------------------------------------------------------


_RECENT_EMPTY = {
    "ja": "（まだ無し。今回が最初のフィラー）",
    "en": "(none yet - this is the first filler of the run)",
}


def _recent_block(recent: list[str]) -> str:
    if not recent:
        return _tbl(_RECENT_EMPTY)
    return "\n".join(f"- {s}" for s in recent)


# --- ネタだし（お題そのものを LLM に考えさせる） ----------------------------
#
# 固定リストだと、いくら行を増やしても有限＝どのモデルで喋らせても入り口が
# 同じになってしまう。トーク組み立て（_llm_prompt 以下）は元から LLM 任せなので、
# お題の方も同じモデルに考えさせれば「ネタの引き出し」自体がモデルの地力の差として
# 出る。ここで作った一言を generate_llm_filler の topic_seed にそのまま渡す
# （トーク組み立ては従来どおり）。失敗／タイムアウト時は None を返し、
# 呼び出し側（FillerThread._pick_topic）はお題なしで進める。

_TOPIC_GUIDANCE = {
    "ja": (
        "深夜ラジオのフリートークで使う「お題」を1つだけ考えてください。\n"
        "お題は完結した小噺ではなく、二人が掛け合いで膨らませるための短い種です。\n"
        "日常のふとした場面・気分・あるある（家事、時間帯、記憶、ことば、季節、天気、"
        "生活の小さな謎、番組そのものの楽屋ネタ、など）が向いています。\n"
        "避けるもの: 政治・宗教・時事、特定の実在人物いじり、センシティブな話題、"
        "オチが要る作り込んだエピソード。\n"
        "書き方: 独り言や気づきのような短い一文にすること。疑問文（「〜は？」）にしない、"
        "かぎ括弧で囲んでタイトルのように見せない、クイズの出題文のような体裁にしない。"
        "説明や前置きなしに、お題そのものだけを書くこと。"
    ),
    "en": (
        "Come up with exactly one 'topic' for late-night radio free talk.\n"
        "It should not be a finished anecdote - just a short seed the two hosts can riff on "
        "back and forth.\n"
        "Small everyday moments, moods, and 'you know how...' observations work well "
        "(chores, time of day, memory, words, seasons, weather, small domestic mysteries, "
        "backstage jokes about the show itself).\n"
        "Avoid: politics, religion, current events, teasing real named individuals, "
        "sensitive subjects, or anything that needs a punchline to land.\n"
        "Style: write it as a short stray thought or observation. Do not phrase it as a "
        "question, do not wrap it in quotes like a title, do not make it read like a quiz "
        "prompt. Write only the topic itself, with no preamble or explanation."
    ),
}


def _llm_topic_prompt(recent: list[str]) -> str:
    return language.pick(
        ja=f"""{_tbl(_TOPIC_GUIDANCE)}

## 直近で使ったお題（参考。今回はこれらと違う切り口にすること）
{_recent_block(recent)}
""",
        en=f"""{_tbl(_TOPIC_GUIDANCE)}

## Recently used topics (for reference. Come at it from a different angle this time)
{_recent_block(recent)}
""",
    )


def _llm_topic_schema() -> dict:
    return {
        "type": "object",
        "properties": {"topic": {"type": "string"}},
        "required": ["topic"],
    }


def generate_llm_topic(llm_config: LLMConfig, recent_topics: list[str] | None = None) -> str | None:
    """フィラーの「お題」自体を LLM に考えさせる（ネタだし）。失敗時は None。

    ``recent_topics`` は直近に使ったお題。同じ切り口の繰り返しを避けさせる
    参考情報として渡すだけで、照合や再生成はしない。
    """
    try:
        raw = llm_http.chat(
            llm_config, _llm_topic_prompt(recent_topics or []),
            schema=_llm_topic_schema(),
            temperature=llm_config.temperature,
            timeout_sec=min(llm_config.timeout_sec, _LLM_TOPIC_TIMEOUT_SEC),
        )
        topic = sanitize_text(json.loads(raw)["topic"]).strip()
    except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError) as e:
        logger.warning("LLM topic generation failed: %s", e)
        return None
    return topic or None


def _topic_block(topic_seed: str | None) -> str:
    if language.current() == "en":
        if topic_seed:
            return (
                f"## Today's prompt\n"
                f'"{topic_seed}"\n'
                f"This is a writer's note, not a line either of them has seen. One of them should "
                f"open with it as if they just thought of it themselves. Never say things like "
                f"\"today's topic is...\" or \"so the prompt is...\", never quote this line back "
                f"verbatim, and never let on that a topic was handed to them at all - as far as "
                f"they know, nobody assigned this. Do not just explain or summarise it and stop; "
                f"let the two of them wander off it or dig into it."
            )
        return (
            "## Today's prompt\n"
            "(none given. Work with whatever is already in the room: the time of day, the "
            "track that is playing, the fact that they are waiting on the next item)"
        )
    if topic_seed:
        return (
            f"## 今回のお題（放送作家のメモ。出演者は見ていない）\n"
            f"「{topic_seed}」\n"
            f"二人のどちらかが、自分でふと思いついたことのように自然に話し始める入り口として使うこと。"
            f"「お題は〜ですね」「今回のお題は〜」のように、これがお題として与えられたものであることや、"
            f"文言そのものを引用して読み上げることは絶対にしないこと（出演者は「お題」という仕組みの"
            f"存在自体を知らない）。そのまま説明・要約して終わらせず、二人のやりとりで脱線したり"
            f"掘り下げたりすること。"
        )
    return (
        "## 今回のお題\n"
        "（指定なし。今の時刻・かかっている曲・「次のネタ待ち」の楽屋など、"
        "その場にあるものから軽く広げること）"
    )


# 実行場所（LLMConfig.placement）ごとの「楽屋ネタ」の事実。プロンプトにはここに
# 書いたことだけを渡し、LLM には「書いていないことを足すな」と併せて指示する。
_PLACEMENT_FACTS = {
    "ja": {
        "local": (
            "この番組は、このパソコンの中だけで動いているローカルの生成AIが台本を書いている"
            "（外部のAPIには一切つないでいない）。"
        ),
        "cloud": (
            "この番組の台本は、クラウドで動いている生成AIが書いている"
            "（手元のパソコンで動かしているわけではなく、ネットの向こうに投げて返してもらっている）。"
        ),
    },
    "en": {
        "local": (
            "The scripts for this show are written by a generative model running entirely on "
            "this one machine (nothing is sent to an outside API)."
        ),
        "cloud": (
            "The scripts for this show are written by a generative model running in the cloud "
            "(not on the machine in the room - the request goes out over the network and comes back)."
        ),
    },
}


def _placement_fact(placement: str) -> str:
    """実行場所の事実を1文で返す。判定できていなければ空文字（触れさせない）。"""
    return _tbl(_PLACEMENT_FACTS).get(placement, "")


_PLACEMENT_INSTRUCTIONS = {
    "ja": {
        "known": (
            "- 生成AIが「どこで動いているか」に触れるなら、上の『状況・楽屋ネタ』に書いたとおりに言うこと。"
            "ローカルとクラウドを取り違えたり、書いていない話（課金額・通信量・スペックなど）を"
            "推測で足したりしないこと"
        ),
        "unknown": (
            "- 生成AIが「どこで動いているか」（このパソコンの中か、クラウドか）には触れないこと。"
            "今回は分かっていないため"
        ),
    },
    "en": {
        "known": (
            "- If anyone brings up where the model is running, say it the way it is written under "
            "'Behind the scenes' above. Do not mix up local and cloud, and do not add anything that "
            "is not written there (costs, bandwidth, hardware specs)"
        ),
        "unknown": (
            "- Do not bring up where the model is running (on this machine or in the cloud). "
            "It is not known this time"
        ),
    },
}


def _placement_instruction(placement: str) -> str:
    key = "known" if _placement_fact(placement) else "unknown"
    return _tbl(_PLACEMENT_INSTRUCTIONS)[key]


# 天気は placement と同じく「渡した事実だけ喋らせる」枠。取れていなければ触れさせない
# （weather.py 参照）。数字を渡す以上、書いていない予報を足されるのがいちばん困る。
def _weather_block(weather: Weather | None) -> str:
    if weather is None:
        return ""
    if language.current() == "en":
        return f"\n## The weather right now\n{weather.fact_line}\n"
    return f"\n## 今の天気\n{weather.fact_line}\n"


_WEATHER_INSTRUCTIONS = {
    "ja": {
        "known": (
            "- 天気に触れるなら、上の『今の天気』に書いてある内容だけを使うこと。"
            "そこに無い予報（明日・週末・この先の見通し）や、湿度・風向きのような数値を"
            "推測で足さないこと。天気は話の入口や相づちに軽く使う程度でよく、"
            "無理に触れなくてよい"
        ),
        "unknown": "- 天気の話には触れないこと（今回は分かっていないため）",
    },
    "en": {
        "known": (
            "- If the weather comes up, use only what is written under 'The weather right now'. "
            "Do not add a forecast that is not there (tomorrow, the weekend, the week ahead) or "
            "numbers like humidity and wind direction. The weather is a way in or a bit of "
            "small talk; there is no need to force it"
        ),
        "unknown": "- Do not talk about the weather (it is not known this time)",
    },
}


def _weather_instruction(weather: Weather | None) -> str:
    return _tbl(_WEATHER_INSTRUCTIONS)["unknown" if weather is None else "known"]


def _selfref_instruction(topic_seed: str | None, model: str, engine_label: str) -> str:
    if language.current() == "en":
        if topic_seed:
            return (
                f'- The model name "{model}" and the inference engine "{engine_label}" only need a '
                f"mention if the conversation happens to go there (no need to force it in this time)"
            )
        return (
            f'- Work the model name "{model}" and the inference engine "{engine_label}" into the '
            f"conversation once, naturally, as a self-deprecating aside. Do not make it sound like an advert"
        )
    if topic_seed:
        return (
            f"- 番組を動かしている生成AIのモデル名「{model}」や推論エンジン「{engine_label}」の"
            f"楽屋ネタは、話の流れで自然に触れられそうなときだけでよい（今回は無理に入れなくてよい）"
        )
    return (
        f"- モデル名「{model}」と推論エンジン「{engine_label}」に、会話の中で一度は自然に触れること"
        f"（自虐や楽屋ネタとして。宣伝くさくしないこと）"
    )


def _llm_prompt(now_playing: str | None, model: str, engine_label: str,
                a: CastMember, b: CastMember, recent: list[str],
                topic_seed: str | None = None, placement: str = "unknown",
                weather: Weather | None = None) -> str:
    song = f"「{now_playing}」" if now_playing else "（今は不明）"
    song_en = f'"{now_playing}"' if now_playing else "(not known)"
    placement_fact = _placement_fact(placement)
    return language.pick(
        ja=f"""あなたはラジオ番組「エルエルエム・ラジオ・デーモン」の放送作家です。
次のネタがまだ用意できていません。間をつなぐ短い雑談（フィラートーク）の台本を作ってください。

## 出演者（speaker にはこの id をそのまま使うこと）
- {a.id}（{a.name}）: {a.desc}
- {b.id}（{b.name}）: {b.desc}

{_topic_block(topic_seed)}
{_weather_block(weather)}
## 状況・楽屋ネタ
- {placement_fact}台本を書いている LLM のモデル名は「{model}」、
  動かしている推論エンジンは「{engine_label}」。
- 今かかっている曲: {song}

## 直近のフィラー（参考。今回はこれらと違う入り方・話題・オチにすること）
{_recent_block(recent)}

## 内容の指示
- {_LLM_MIN_LINES}〜{_LLM_MAX_LINES}行の短い雑談。テンポよく、肩の力を抜いた雰囲気で
- 上に挙げた直近のフィラーと似た切り口・話題・締め方を避け、別の角度から始めること
- 「次のネタが来るまでのつなぎ」であること自体をネタにしてよい
{_selfref_instruction(topic_seed, model, engine_label)}
{_placement_instruction(placement)}
{_weather_instruction(weather)}
- モデルや推論エンジンをいじるのは親しみを込めた楽屋ネタに留め、開発元の企業や、その姿勢を否定・批判する方向には広げないこと
- 今かかっている曲があるなら一言触れてもよい（無理に触れなくてよい）
- 1人の発言は1〜2文程度
{sensitive.SENSITIVE_TOPICS_GUIDANCE}
- 記号・絵文字・Markdown記法を使わないこと
{language.LANGUAGE_GUIDANCE}
- 本文に無い固有名詞や数値を断定的に話さないこと
- 話し言葉で書くこと
""",
        en=f"""You are the writer for a radio show called "LLM Radio Daemon".
The next item is not ready yet. Write a short piece of filler chat to cover the gap.

## Who is on air (use these ids verbatim in speaker)
- {a.id} ({a.name}): {a.desc}
- {b.id} ({b.name}): {b.desc}

{_topic_block(topic_seed)}
{_weather_block(weather)}
## Behind the scenes
- {placement_fact}The model writing the scripts is called "{model}",
  and it runs on an inference engine called "{engine_label}".
- Track playing right now: {song_en}

## Recent filler (for reference. Start somewhere else this time)
{_recent_block(recent)}

## What to write
- {_LLM_MIN_LINES} to {_LLM_MAX_LINES} lines of short chat. Quick, easy, nobody is trying hard
- Avoid the opening, the subject and the sign-off of the recent filler above; come at it from another angle
- The fact that this is padding until the next item is fair game as a joke
{_selfref_instruction(topic_seed, model, engine_label)}
{_placement_instruction(placement)}
{_weather_instruction(weather)}
- Teasing the model or the engine stays affectionate and behind-the-scenes. Do not push it into
  criticising the company that built it or the way they work
- If a track is playing they may mention it in passing (no need to force it)
- One or two sentences per turn
{sensitive.SENSITIVE_TOPICS_GUIDANCE}
- No symbols, emoji or Markdown
{language.LANGUAGE_GUIDANCE}
- Do not state proper nouns or numbers that are not given above as if they were fact
- Write it as speech, not prose
""",
    )


def _llm_schema(
    speaker_ids: list[str],
    min_lines: int = _LLM_MIN_LINES,
    max_lines: int = _LLM_MAX_LINES,
) -> dict:
    return {
        "type": "object",
        "properties": {
            "lines": {
                "type": "array",
                "minItems": min_lines,
                "maxItems": max_lines,
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


def generate_llm_filler(
    llm_config: LLMConfig,
    now_playing: str | None,
    speaker_a: CastMember,
    speaker_b: CastMember,
    recent_talks: list[str] | None = None,
    topic_seed: str | None = None,
    weather: Weather | None = None,
) -> Script | None:
    """LLM に短いフィラー雑談を書かせる。失敗時は None（呼び出し側がテンプレへフォールバック）。

    ``recent_talks`` は直近に流したフィラーの短い要約。プロンプトに載せて、
    同じ入り方・話題・オチの繰り返しを避けさせる（照合や再生成はしない）。

    ``topic_seed`` は ``generate_llm_topic`` が考えた「お題」の一言。渡すと
    それを入り口に雑談を広げさせる（毎回同じ楽屋ネタに収束するのを防ぐ）。

    ``weather`` は [weather] から取れた今の天気。``None`` なら天気には触れさせない。
    """
    ids = [speaker_a.id, speaker_b.id]
    prompt = _llm_prompt(
        now_playing, llm_config.model, llm_config.engine_label,
        speaker_a, speaker_b, recent_talks or [], topic_seed,
        llm_config.placement, weather,
    )
    try:
        raw = llm_http.chat(
            llm_config, prompt, schema=_llm_schema(ids),
            temperature=llm_config.temperature,
            timeout_sec=min(llm_config.timeout_sec, _LLM_FILLER_TIMEOUT_SEC),
        )
        parsed = json.loads(raw)
        lines = [
            ScriptLine(speaker=l["speaker"], text=sanitize_text(l["text"]))
            for l in parsed["lines"]
            if l.get("speaker") in ids and sanitize_text(l.get("text", ""))
        ]
    except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError) as e:
        logger.warning("LLM filler generation failed: %s", e)
        return None

    if len(lines) < 2:
        return None
    title = f"filler:llm ({topic_seed})" if topic_seed else "filler:llm"
    return Script(topic_id=None, topic_title=title, lines=lines, is_filler=True)


def summarize_filler(script: Script, max_chars: int = 160) -> str:
    """流したフィラーを1行に潰す。次回のプロンプトへ「直近のフィラー」として渡す用。"""
    joined = " ".join(" ".join(l.text.split()) for l in script.lines)
    return joined[:max_chars].strip()


# --- 曲中のひとこと（radio の song_talk = "back"） -------------------------
#
# このモードはトークが「曲が終わった瞬間の振り返り」1回だけなので、
# 曲の長さぶん（ネットラジオだと4〜7分）まるごと無音になる。その途中に一度だけ
# 軽いひとことを挟んで間を持たせる。**曲名・アーティスト名はここでは出さない**
# ―― 曲が終わってから振り返って紹介するのが back_announce の型であり、
# 先に名前を出すとその段取りが崩れるため。プロンプトにも曲情報自体を渡さない。

_MID_SONG_MIN_LINES = 2
_MID_SONG_MAX_LINES = 3

_MID_SONG_DEFAULT_TONE = {
    "ja": "肩の力を抜いた、静かめのトーンで。",
    "en": "Relaxed and on the quiet side.",
}


def _mid_song_prompt(appearers: list[CastMember], tone_hint: str) -> str:
    roster = "\n".join(f"- {m.id}（{m.name}）: {m.desc}" for m in appearers)
    roster_en = "\n".join(f"- {m.id} ({m.name}): {m.desc}" for m in appearers)
    now = time.localtime()
    return language.pick(
        ja=f"""あなたはラジオ番組「エルエルエム・ラジオ・デーモン」の放送作家です。
今は曲が流れている最中です。その途中に軽くひとことだけ挟む、ごく短いトークの台本を作ってください。

## 出演者（speaker にはこの id をそのまま使うこと）
{roster}

## 今の時刻
{now.tm_hour}時{now.tm_min}分（24時間表記）。24時間放送なので、時間帯に触れるなら必ずこの時刻に
合わせること。実際と違う時間帯（深夜なのに「お昼」など）を口にしないこと。

## 台本の形
- {_MID_SONG_MIN_LINES}〜{_MID_SONG_MAX_LINES}行だけ。ひとこと言って、すぐ終わる
- 曲名・アーティスト名には触れないこと（曲が終わってから改めて紹介する段取りのため）
- 曲を解説しない。今の雰囲気や気分、ふと思いついたことを軽く漏らす程度にとどめる
- 次の曲や別の話題の予告はしない。話を広げない
- 全員が話す必要はない。流れとして自然な人だけが話す

## 今の時間帯のトーン
{tone_hint or _MID_SONG_DEFAULT_TONE["ja"]}

## 制約
- 記号・絵文字・Markdown記法（*, #, ` など）を一切使わないこと。すべて音声で読み上げられる
{language.LANGUAGE_GUIDANCE}
- 1人の発言は1〜2文程度。話し言葉で書くこと""",
        en=f"""You are the writer for a radio show called "LLM Radio Daemon".
A track is playing right now. Write a very short exchange to drop in over the top of it.

## Who is on air (use these ids verbatim in speaker)
{roster_en}

## The time right now
{_spell_clock(now.tm_hour, now.tm_min)}. The station runs around the clock, so if anyone
refers to the time of day it has to match this. Do not say "this afternoon" in the middle of the night.

## Shape of it
- {_MID_SONG_MIN_LINES} to {_MID_SONG_MAX_LINES} lines only. Someone says one thing and it ends
- Do not name the track or the artist (it gets introduced properly after it finishes)
- Do not analyse the music. Just let slip how it feels right now, or a stray thought
- Do not trail the next track or another subject. Do not open anything up
- Not everyone has to speak. Only whoever it falls to naturally

## Tone for this time of day
{tone_hint or _MID_SONG_DEFAULT_TONE["en"]}

## Constraints
- No symbols, emoji or Markdown (*, #, ` and so on). Every line is read aloud
{language.LANGUAGE_GUIDANCE}
- One or two sentences per turn. Write it as speech""",
    )


def generate_llm_mid_song_chat(
    llm_config: LLMConfig, appearers: list[CastMember], tone_hint: str = ""
) -> Script | None:
    """LLM に曲中のひとことを書かせる。失敗時は None（呼び出し側がテンプレへフォールバック）。"""
    ids = [m.id for m in appearers]
    try:
        raw = llm_http.chat(
            llm_config, _mid_song_prompt(appearers, tone_hint),
            schema=_llm_schema(ids, _MID_SONG_MIN_LINES, _MID_SONG_MAX_LINES),
            temperature=llm_config.temperature,
            timeout_sec=min(llm_config.timeout_sec, _LLM_FILLER_TIMEOUT_SEC),
        )
        parsed = json.loads(raw)
        lines = [
            ScriptLine(speaker=l["speaker"], text=sanitize_text(l["text"]))
            for l in parsed["lines"]
            if l.get("speaker") in ids and sanitize_text(l.get("text", ""))
        ]
    except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError) as e:
        logger.warning("LLM mid-song chat generation failed: %s", e)
        return None

    if not lines:
        return None
    # is_filler は立てない。曲中のひとことは「原稿が枯れたつなぎ」ではなく番組の一部なので、
    # ON AIR ランプは点灯のまま（橙点滅にしない）、ひな壇も喋る顔ぶれに更新させる。
    return Script(topic_id=None, topic_title="mid_song:llm", lines=lines)


# LLM が使えないときの保険。曲名に触れない当たり障りのない一言だけを用意する。
_MID_SONG_CHATS: dict[str, list[list[str]]] = {
    "ja": [
        ["……いい感じですね、これ。", "うん。しばらく黙って聴いていたいやつ。"],
        ["こういう時間、けっこう好きです。", "分かります。"],
        ["ちょっと眠くなってきました。", "寝ないでくださいよ。"],
        ["今、外は静かなんでしょうね。", "たぶんね。ここも静かですけど。"],
        ["……なんか、いいですね。", "語彙が消えてる。", "曲がいいので許してください。"],
    ],
    "en": [
        ["This one is rather nice, actually.", "Mm. One to just sit with for a bit."],
        ["I like this part of the night.", "Yeah, me too."],
        ["I am getting a bit sleepy here.", "Don't you dare."],
        ["It must be quiet outside right now.", "Probably. It is quiet in here as well."],
        ["That is, um. That is good.", "You have lost all your words.", "The track is good. Let me off."],
    ],
}


def generate_mid_song_chat(appearers: list[CastMember]) -> Script:
    """LLM なしで曲中のひとことを1本作る（generate_llm_mid_song_chat 失敗時の保険）。"""
    texts = random.choice(_tbl(_MID_SONG_CHATS))
    lines = [
        ScriptLine(appearers[i % len(appearers)].id, text)
        for i, text in enumerate(texts)
    ]
    return Script(topic_id=None, topic_title="mid_song:template", lines=lines)


# --- テンプレートフィラー（LLM 失敗時のフォールバック） ---------------------

_TIME_OF_DAY_REMARKS = {
    "ja": {
        "morning": ["こんな朝早くから聞いてくれてる人、物好きですね〜。", "眠くなったら一緒に喋りましょう。"],
        "day": ["こんな時間まで何してるんですか、なんて聞きませんけどね。", "お昼寝したくなる感じの時間ですね。"],
        "night": ["こんな遅くまで、お疲れさまです。", "夜更かし組、今日も付き合ってくれてありがとうございます。"],
    },
    "en": {
        "morning": [
            "Anyone up with us this early is a bit of an odd one, aren't they.",
            "If you start nodding off, just talk to us instead.",
        ],
        "day": [
            "I won't ask what you are all doing at this hour.",
            "It is the time of day that makes you want a nap, isn't it.",
        ],
        "night": [
            "Still up at this hour. Long day, was it.",
            "To everyone on the late shift, thanks for sitting with us again.",
        ],
    },
}


def _time_of_day() -> str:
    hour = time.localtime().tm_hour
    if 5 <= hour < 11:
        return "morning"
    if 11 <= hour < 18:
        return "day"
    return "night"


_TIME_REPORT = {
    "ja": {
        "open": "さて、只今{clock}になりました。",
        "close": [
            "というわけで、このまま気ままにお喋り続けます。",
            "次のネタが入るまで、少しだけ雑談にお付き合いください。",
        ],
    },
    "en": {
        "open": "Right, it has just gone {clock}.",
        "close": [
            "So we will keep rambling on for a bit, if that is alright.",
            "Stay with us a minute while we wait on the next item.",
        ],
    },
}


def _clock_text() -> str:
    """テンプレートに差し込む「今の時刻」。英語は語に開く（モジュール docstring 参照）。"""
    now = time.localtime()
    if language.current() == "en":
        return _spell_clock(now.tm_hour, now.tm_min)
    return f"{now.tm_hour}時{now.tm_min}分"


def _time_report(host: str, asst: str) -> list[ScriptLine]:
    t = _tbl(_TIME_REPORT)
    remark = random.choice(_tbl(_TIME_OF_DAY_REMARKS)[_time_of_day()])
    return [
        ScriptLine(host, t["open"].format(clock=_clock_text())),
        ScriptLine(asst, remark),
        ScriptLine(host, random.choice(t["close"])),
    ]


_SONG_INTRO = {
    "ja": {
        "open": "そういえば、今流れているのは『{title}』だそうです。",
        "remarks": [
            "これ、なんか耳に残りますね。",
            "お、この曲いいですね。",
            "こういうのが流れてると、なんか集中できる気がします。",
        ],
        "followups": [
            "選曲は完全にネットラジオ任せなんですけどね。",
            "曲についてはこっちも詳しくは分からないんですが、良い雰囲気です。",
        ],
    },
    "en": {
        "open": "Oh, apparently what is playing right now is {title}.",
        "remarks": [
            "This one sticks in your head a bit, doesn't it.",
            "Oh, this is a good one.",
            "Something like this on in the background and I can actually concentrate.",
        ],
        "followups": [
            "Mind you, the whole playlist is whatever the stream feels like.",
            "We do not know much more about it than you do, but it is a nice mood.",
        ],
    },
}


def _song_intro(now_playing: str, host: str, asst: str) -> list[ScriptLine]:
    t = _tbl(_SONG_INTRO)
    return [
        ScriptLine(host, t["open"].format(title=now_playing)),
        ScriptLine(asst, random.choice(t["remarks"])),
        ScriptLine(host, random.choice(t["followups"])),
    ]


# 天気のテンプレ台詞。相づちだけを天候タグごとに分ける（気温・降水確率のような
# 数値は1行目で読み上げた値しか使わない ―― テンプレは LLM を通らないので、
# 渡された Weather に無いことは書きようがない、という作りにしておく）。
_WEATHER_REMARKS = {
    "ja": {
        "clear": [
            "いい天気じゃないですか。ここからは見えませんけど。",
            "外に出たほうがいい日ですね、たぶん。",
        ],
        "cloudy": [
            "はっきりしない空ですねえ。",
            "こういう日、嫌いじゃないです。",
        ],
        "rain": [
            "傘、持って出ました？",
            "雨の音を聞きながらというのも、悪くないですけどね。",
        ],
        "snow": [
            "積もると聞いてちょっとわくわくしています。",
            "足元、気をつけてくださいね。",
        ],
        "storm": [
            "外、けっこう荒れてるみたいですよ。",
            "こういう日は無理に出かけないのがいちばんです。",
        ],
    },
    "en": {
        "clear": [
            "That is a nice day, that is. Can't see it from in here, mind.",
            "Sounds like a day to actually go outside.",
        ],
        "cloudy": [
            "Can't make its mind up, that sky.",
            "I do not mind a day like that, to be honest.",
        ],
        "rain": [
            "Did you take an umbrella out with you?",
            "There are worse things than listening to the rain, though.",
        ],
        "snow": [
            "I will admit I am a bit excited if it settles.",
            "Watch your footing out there.",
        ],
        "storm": [
            "Sounds fairly rough out there.",
            "That is a day to stay in if you possibly can.",
        ],
    },
}

_WEATHER_CLOSERS = {
    "ja": [
        "……という、聞かれてもいないお天気でした。",
        "スタジオからは外が見えないので、これも人づてです。",
        "まあ、この番組はどんな天気でも流れているんですけどね。",
    ],
    "en": [
        "And that was the weather, which nobody asked for.",
        "We cannot see outside from the studio, so we are taking someone's word for it.",
        "Either way, this show goes out whatever it is doing out there.",
    ],
}

_WEATHER_OPEN = {
    "ja": {"line": "今、{location}は{description}{temp}。", "temp": "。気温は{deg}度だそうです"},
    "en": {
        "line": "Right now it is {description} in {location}{temp}.",
        "temp": ", and about {deg} degrees",
    },
}


def _weather_chat(host: str, asst: str, weather: Weather) -> list[ScriptLine]:
    """今の天気に軽く触れるだけのつなぎ（LLM 失敗時のフォールバック）。"""
    t = _tbl(_WEATHER_OPEN)
    # 英語は気温も語に開く（_spell_clock と同じ理由。0〜59 の外はそのまま数字）。
    if weather.temp_c is None:
        temp = ""
    elif language.current() == "en":
        temp = t["temp"].format(deg=_spell_number(int(round(weather.temp_c))))
    else:
        temp = t["temp"].format(deg=weather.temp_c)
    remarks = _tbl(_WEATHER_REMARKS)
    return [
        ScriptLine(host, t["line"].format(
            location=weather.location, description=weather.description, temp=temp,
        )),
        ScriptLine(asst, random.choice(remarks.get(weather.condition, remarks["cloudy"]))),
        ScriptLine(host, random.choice(_tbl(_WEATHER_CLOSERS))),
    ]


# _ai_chat の2行目。実行場所を言い切る行なので placement ごとに分ける。
# "unknown"（判定できなかった）のときは場所に触れず、エンジン名だけで済ませる。
_AI_CHAT = {
    "ja": {
        "openers": [
            "この番組の台本、{model} っていうモデルが書いてるんですよね。",
            "喋ってる中身を考えてるのは {model} っていう生成AIらしいです。",
        ],
        "followups": {
            "local": [
                "しかも {engine} で、このパソコンの中だけで動いてるんですよね。",
                "{engine} っていうので、外に一切つながず回してるそうです。",
            ],
            "cloud": [
                "しかも {engine} 経由で、クラウドの向こうで動いてるらしいです。",
                "{engine} から呼んでるだけで、中身はこのパソコンにはいないんですって。",
            ],
            "unknown": [
                "{engine} っていうやつで動かしてるらしいです。",
                "{engine} から呼んでるそうですよ。詳しいことは聞かされてません。",
            ],
        },
        "closers": [
            "だからたまに言葉に詰まっても、大目に見てください。",
            "無課金でここまでやってるので、そこは褒めてほしいところです。",
        ],
    },
    "en": {
        "openers": [
            "The scripts for this show are written by a model called {model}, you know.",
            "Whatever we are saying, a model called {model} thought of it first.",
        ],
        "followups": {
            "local": [
                "And it is all running on {engine}, inside this one machine.",
                "Something called {engine}, with nothing going out to the internet at all.",
            ],
            "cloud": [
                "Running out in the cloud somewhere, through {engine}.",
                "We just call it through {engine}. The thing itself is not in this room.",
            ],
            "unknown": [
                "Runs on something called {engine}, I am told.",
                "We call it through {engine}. Nobody has told us much beyond that.",
            ],
        },
        "closers": [
            "So if it loses its thread now and then, go easy on it.",
            "It is doing all this without anyone paying for it, which deserves some credit.",
        ],
    },
}


def _ai_chat(
    host: str, asst: str, model: str, engine_label: str, placement: str = "unknown"
) -> list[ScriptLine]:
    """この番組を動かしている LLM のモデル名・推論エンジン名に触れる楽屋ネタ。"""
    t = _tbl(_AI_CHAT)
    followups = t["followups"].get(placement, t["followups"]["unknown"])
    return [
        ScriptLine(host, random.choice(t["openers"]).format(model=model)),
        ScriptLine(asst, random.choice(followups).format(engine=engine_label)),
        ScriptLine(host, random.choice(t["closers"])),
    ]


_IDLE_CHATS: dict[str, list[list[tuple[str, str]]]] = {
    "ja": [
        [
            ("B", "ところで、こういうのっていつまで喋り続けるんですか？"),
            ("A", "さあ……次のネタが来るまで、としか言いようがないですね。"),
            ("B", "身も蓋もない。"),
        ],
        [
            ("A", "そういえば、聞いてる人がいるかどうかは分からないんですよね。"),
            ("B", "いても言わないでほしいです、緊張するので。"),
        ],
        [
            ("B", "ちょっと喉が渇いてきました。"),
            ("A", "生放送じゃないんだから水分補給していいんですよ。"),
            ("B", "そうでした、これずっと流れてるだけでしたね。"),
        ],
        [
            ("A", "少し間が空きましたが、番組は止まっていませんのでご安心を。"),
            ("B", "止まったら止まったで気づいてほしいですけどね。"),
        ],
    ],
    "en": [
        [
            ("B", "How long do we actually keep talking for, out of interest?"),
            ("A", "Until the next item turns up. That is genuinely the whole answer."),
            ("B", "Well, that is bleak."),
        ],
        [
            ("A", "It occurs to me we have no idea whether anyone is listening."),
            ("B", "If you are, please do not tell me. I will get nervous."),
        ],
        [
            ("B", "I am getting a bit thirsty here."),
            ("A", "It is not live. You can go and get a drink."),
            ("B", "Oh, that is true. This just runs, doesn't it."),
        ],
        [
            ("A", "Bit of a gap there, but nothing is broken, I promise."),
            ("B", "Although if it did break, I would like someone to notice."),
        ],
    ],
}


def _idle_chat(host: str, asst: str) -> list[ScriptLine]:
    chat = random.choice(_tbl(_IDLE_CHATS))
    role_to_id = {"A": host, "B": asst}
    return [ScriptLine(role_to_id[speaker], text) for speaker, text in chat]


# LLM なしなのでお題を膨らませられない。読み上げて相づちを打つだけの薄いつなぎ。
_SEED_CHAT = {
    "ja": {
        "open": "ちょっと聞いてください。{seed}。",
        "reactions": [
            "あー、なんか分かります。",
            "急にどうしたんですか。",
            "それ、いま言うことですか。",
            "で、オチはあるんですか。",
            "ふふ、いいですね。",
        ],
        "followups": [
            "……という感じで、次のネタが来るまで持たせております。",
            "深く考えないでください。つなぎなので。",
            "そんな話をしているうちに、そろそろ本編が戻ってくるはずです。",
        ],
    },
    "en": {
        "open": "Listen to this a second. {seed}.",
        "reactions": [
            "Yeah, no, I get that.",
            "Where has this come from?",
            "And you are bringing this up now?",
            "Right. Is there a punchline coming?",
            "Ha. No, that is good.",
        ],
        "followups": [
            "And that is how we fill the time until the next item.",
            "Do not read too much into it. It is padding.",
            "By the time we are done with this, the show proper should be back.",
        ],
    },
}


def _seed_chat(seed: str, host: str, asst: str) -> list[ScriptLine]:
    t = _tbl(_SEED_CHAT)
    if language.current() == "en" and seed:
        # お題は小文字始まりで返ってくることがあるので、文の途中ではなく
        # 1文として読ませる英語版では頭を大文字に起こす。
        seed = seed[0].upper() + seed[1:]
    return [
        ScriptLine(host, t["open"].format(seed=seed)),
        ScriptLine(asst, random.choice(t["reactions"])),
        ScriptLine(host, random.choice(t["followups"])),
    ]


def generate_filler(
    now_playing: str | None,
    speaker_a_id: str,
    speaker_b_id: str,
    model: str = "",
    engine_label: str = "",
    topic_seed: str | None = None,
    placement: str = "unknown",
    weather: Weather | None = None,
) -> Script:
    """即座に（LLM呼び出しなしで）フィラー原稿を1本生成する。

    2人進行（A が主に進行、B が相づち）。id は呼び出し側が [[cast]] から渡す。
    ``topic_seed`` があれば、それを読み上げて相づちを打つだけの薄いつなぎを優先する。
    ``placement`` は ``[llm] placement``（local / cloud / unknown）。楽屋ネタの
    「どこで動いているか」の言い回しがこれで変わる。
    ``weather`` があれば「今の天気」もつなぎのネタ候補に加わる（無ければ触れない）。

    定型文は ``[locale] lang`` ぶんを引く（モジュール docstring 参照）。
    """
    if topic_seed and random.random() < 0.7:
        lines = _seed_chat(topic_seed, speaker_a_id, speaker_b_id)
        return Script(
            topic_id=None, topic_title=f"filler:seed ({topic_seed})",
            lines=lines, is_filler=True,
        )

    choices: list[str] = ["time", "idle"]
    if now_playing:
        choices.append("song")
    if model and engine_label:
        choices.append("ai")
    if weather is not None:
        choices.append("weather")
    kind = random.choice(choices)

    if kind == "time":
        lines = _time_report(speaker_a_id, speaker_b_id)
    elif kind == "song":
        lines = _song_intro(now_playing or "", speaker_a_id, speaker_b_id)
    elif kind == "weather":
        lines = _weather_chat(speaker_a_id, speaker_b_id, weather)
    elif kind == "ai":
        lines = _ai_chat(speaker_a_id, speaker_b_id, model, engine_label, placement)
    else:
        lines = _idle_chat(speaker_a_id, speaker_b_id)

    return Script(topic_id=None, topic_title=f"filler:{kind}", lines=lines, is_filler=True)
