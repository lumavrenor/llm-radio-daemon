"""Ollama で原稿を生成する。structured output (JSON Schema) を使う。

パース失敗時はそのトピックを捨てて次へ進む（リトライは1回まで）。
全履歴は渡さない。直前2〜3トピックの一行要約だけを渡す。
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import deque

import requests

from .. import llm_http
from ..config import CastMember, ContentConfig, LLMConfig
from .. import language  # LANGUAGE_GUIDANCE は起動時に差し替わるので属性参照する
from .. import sensitive  # SENSITIVE_TOPICS_GUIDANCE も起動時に差し替わるので属性参照する
from ..sources import Topic
from . import Script, ScriptLine
from . import reading

logger = logging.getLogger(__name__)

_MIN_LINES = 10
_MAX_LINES = 20

# radio の song_talk = "back"（曲間で「いま終わった曲」を振り返るだけ）。
# 雑談に発展させず、短いひとことで終わらせたいので行数を絞る。
_BACK_ANNOUNCE_MIN_LINES = 2
_BACK_ANNOUNCE_MAX_LINES = 6

# tone_hint 未設定のコンテンツで使う汎用のトーン指示。
_DEFAULT_TONE_HINT = "その時間帯に合った自然なトーンで、テンポよく話すこと。"
_DEFAULT_TONE_HINT_EN = "Whatever tone suits the hour, and keep it moving."


def _script_schema(
    speaker_ids: list[str],
    style_names: list[str],
    min_lines: int = _MIN_LINES,
    max_lines: int = _MAX_LINES,
) -> dict:
    line_props: dict = {
        "speaker": {"type": "string", "enum": speaker_ids},
        "text": {"type": "string"},
    }
    # style は任意。出演者に声色の選択肢がある場合だけ enum を出す（綴りを固定するため）。
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
            }
        },
        "required": ["lines"],
    }


def warm_up(llm_config: LLMConfig) -> bool:
    """モデルをメモリへロードさせるだけの最小リクエスト。

    Ollama / LM Studio(JIT) は初回リクエスト時にモデルをロードするため、起動直後は
    数十秒〜1分ほど応答が返らないことがある。この関数の完了（成否問わず）をもって
    画面左上の "Loading..." 表示を解除する（state.llm_ready）ので、通常の
    timeout_sec より長めに待つ。
    """
    try:
        llm_http.chat(
            llm_config, "hi", temperature=0,
            timeout_sec=max(llm_config.timeout_sec, 120),
        )
        return True
    except (requests.RequestException, ValueError) as e:
        logger.warning("LLM warm-up failed (%s): %s", llm_config.model, e)
        return False


# デバッグ用の出演者固定（config の [debug] pinned_cast_ids）。
# main.py が起動時に set_pinned_cast_ids() でセットする。None なら通常の抽選。
_pinned_cast_ids: tuple[str, ...] | None = None


def set_pinned_cast_ids(ids: list[str] | tuple[str, ...] | None) -> None:
    """pick_speakers が毎回この id 並びだけを返すよう固定する（デバッグ用）。

    ``None`` または空を渡すと固定を解除して通常の抽選に戻す。
    """
    global _pinned_cast_ids
    _pinned_cast_ids = tuple(ids) if ids else None


# --- 持ちネタキャラバリエーション（§1 角度選択）-------------------------------
#
# 出演者ごとの angle_variants（config_cast.toml・静的）から、トピック毎に「今回の
# 角度」を1つ選んでプロンプトへ短い1行として上乗せする。desc（根本人格）は壊さない。
#
# 直近回避：cast_id 毎に「直近に使ったインデックス」を覚えておいて除外する。DB には
# 永続化しない（再起動でリセットされても実害が薄い）。ウォッチドッグが ScriptThread を
# 作り直しても選択履歴が生き残るよう、_pinned_cast_ids と同じくモジュールに置く。
#
# 2（直前2回分）にしているのは、4択だと maxlen=1 では2回に1回同じ角度へ戻ってしまい、
# 毎日長時間聴くと同じ言い回しの再来が早く感じられたため（ユーザー報告）。2にすると
# 残り2択から選ぶため間隔が空く。全員 angle_variants は4件想定（3件以下でも
# _pick_angle の fallback で候補が尽きた回は全件から選び直すだけで壊れない）。

_ANGLE_HISTORY = 2  # 直近2回に使ったインデックスを除外する（maxlen）
_recent_angle_idx: dict[str, deque[int]] = {}

# デバッグ用の角度固定（config の [debug] pin_angle_index）。main.py が起動時に
# set_pin_angle_index() でセットする。None なら通常どおりトピック毎にランダム選択。
_pin_angle_index: int | None = None


def set_pin_angle_index(index: int | None) -> None:
    """全 cast の「今回の角度」を angle_variants のこのインデックスに固定する（デバッグ用）。

    ``None`` を渡すと固定を解除して通常のランダム選択に戻す。範囲外のインデックスは
    「角度なし」扱いになる（その cast の追加行そのものが出ない）。
    """
    global _pin_angle_index
    _pin_angle_index = index


def _pick_angle(member: CastMember) -> str | None:
    """出演者の angle_variants から「今回の角度」を1つ選ぶ。無ければ ``None``。

    ``[debug] pin_angle_index`` 指定時は全員そのインデックスに固定（範囲外は ``None``）。
    通常は直前に使ったインデックスを避けてランダムに選ぶ。
    """
    variants = member.angle_variants
    if not variants:
        return None
    if _pin_angle_index is not None:
        if 0 <= _pin_angle_index < len(variants):
            return variants[_pin_angle_index]
        return None
    history = _recent_angle_idx.setdefault(member.id, deque(maxlen=_ANGLE_HISTORY))
    candidates = [i for i in range(len(variants)) if i not in history] or list(
        range(len(variants))
    )
    idx = random.choice(candidates)
    history.append(idx)
    return variants[idx]


# roster の desc の下に足す1行の見出し（「今日の気分」「今回の角度」）。
# プロンプトへそのまま出るので言語ごとに持つ。
_NOTE_LABELS = {
    "ja": ("今日の気分", "今回の角度"),
    "en": ("Today", "This time"),
}


def _appearer_notes(
    appearers: list[CastMember],
    content: ContentConfig,
    mood: dict[str, str] | None,
) -> dict[str, list[str]]:
    """出演者ごとに roster へ足す短い1行（「今日の気分」「今回の角度」）を組む（§3・§4）。

    mood（日次・全コーナー共通のベースライン）と angle（静的・トピック毎の振れ）は
    どちらも「desc のすぐ下に足す1行」でしかないので同じ dict にまとめ、
    :func:`_build_prompt` は両者を区別せず扱う。中身のある出演者にしかキーを作らない
    （§4-1: 基本メンバーは素の roster のまま）。

    ``worry_consultation`` は通常の topic→script 経路に乗るが、角度も mood も
    tone_hint（このコーナー専用の笑い話ノリ）と衝突するため、まるごと注入しない（§1 スコープ）。

    注記が付く出演者は最大でも出演者数の半分（切り上げ）に間引く。小型モデルは
    roster の注記を「読み上げるセリフ」として全行に効かせがちで、出演者全員に
    付くと台本が戯画の羅列になる（実機観察。gemma3n e2b/e4b 級で顕著）。mood/angle
    生成側のプロンプト品質と無関係に効く安全弁として、ここで頭数を絞る。
    """
    if content.type == "worry_consultation":
        return {}
    mood = mood or {}
    mood_label, angle_label = _NOTE_LABELS.get(
        language.current(), _NOTE_LABELS["ja"]
    )
    notes: dict[str, list[str]] = {}
    for m in appearers:
        lines: list[str] = []
        today_mood = mood.get(m.id, "").strip()
        if today_mood:
            lines.append(f"{mood_label}: {today_mood}")
        angle = _pick_angle(m)
        if angle:
            lines.append(f"{angle_label}: {angle}")
        if lines:
            notes[m.id] = lines

    cap = max(1, (len(appearers) + 1) // 2)
    if len(notes) > cap:
        keep = set(random.sample(list(notes), cap))
        notes = {cid: lines for cid, lines in notes.items() if cid in keep}
    return notes


def pick_speakers(content: ContentConfig, cast: list[CastMember]) -> list[CastMember]:
    """コンテンツの min/max から人数を抽選し、[[cast]] ロースターからその人数を選ぶ。

    ``host_id`` 指定時は、その role（例: ``"host"``）を持つ人を必ず含めて先頭
    （進行役）に置く。順番＝発言の主導権ではないが、プロンプトでは先頭の出演者を
    進行役として扱う。

    ``host_id`` 未指定でも、抽選された中に role が ``host`` / ``assistant`` の人が
    いれば、その人を先頭へ寄せる（進行役ガード）。一発ネタ用のキャラが締めを
    任される事故を避けるため。適任者がいなければ抽選順のまま。

    ``set_pinned_cast_ids()`` で固定されている場合は、人数抽選も host ガードも
    行わず、その id 並びのメンバーをそのまま返す（デバッグ用）。
    """
    if _pinned_cast_ids:
        by_id = {m.id: m for m in cast}
        pinned = [by_id[cid] for cid in _pinned_cast_ids if cid in by_id]
        if pinned:
            return pinned

    upper = min(content.max_speakers, len(cast))
    lower = min(content.min_speakers, upper)
    n = max(1, random.randint(lower, upper))
    picked = random.sample(cast, n)

    if content.host_id:
        host = next((m for m in cast if m.role == content.host_id), None)
        if host is not None:
            picked = [host] + [m for m in picked if m.id != host.id]
            return picked[:n]

    if picked and picked[0].role not in ("host", "assistant"):
        lead = next((m for m in picked if m.role == "host"), None) or next(
            (m for m in picked if m.role == "assistant"), None
        )
        if lead is not None:
            picked = [lead] + [m for m in picked if m.id != lead.id]
    return picked


# TTSが記号や絵文字をそのまま読み上げてしまうため、後処理でも除去する。
_MARKDOWN_CHARS = re.compile(r"[*_`#>\[\]{}~^|]")

# gemma-4 系（Unsloth 経由）が、enable_thinking=False でもまれにチャット
# テンプレートの制御マーカー（"<channelthought"、"<start_of_turn>"、"channel" だけ、
# "<channel>" 等）や、モデル向けの英語メタ文を本文へ漏らす。iter00〜01 の実測:
#   - back_announce#3: "<channelthought <channelthought No change in tone needed for os …"
#   - talk_worry#2:    "channel 眠れない夜、それは本当にお辛いですよね"（角括弧なし）
# 角括弧の有無・閉じ ">" の有無いずれもあり得るので緩めに拾う。マーカーに続く
# 非日本語のゴミ（"No change in tone needed …" 等）も一緒に落とし、日本語の実文
# （かな・カナ・漢字）が始まったらそこで止める。
_CONTROL_LEAK = re.compile(
    r"<*\s*/?\s*(?:channel\w*|start_of_turn|end_of_turn|thought)\b"
    r"[^>\n぀-ヿ一-鿿]*>?",
    re.IGNORECASE,
)

# 英語版。上の日本語版はそのままでは使えない。「日本語の実文（かな・カナ・漢字）が
# 始まったら止まる」という終端が英語には無いので、[^>\n…]* が行末まで飲み込んで
# **正常な行が丸ごと消える**（実測: "channel Right, so what happened there." → ""）。
# しかも "channel" は英語では普通の単語（"a YouTube channel"）なので、単独で拾うと
# 誤爆する側の損害のほうが大きい。
# そこで英語では「角括弧で囲まれた形」と「英語の単語としては現れない綴り」だけを落とす。
# 稀な漏れを1つ見逃すほうが、無事な台詞を1行消すより安い。
_CONTROL_LEAK_EN = re.compile(
    r"<\s*/?\s*(?:channel\w*|start_of_turn|end_of_turn|thought)\b[^>\n]*>?"
    r"|\bchannelthought\b"
    r"|\b(?:start_of_turn|end_of_turn)\b",
    re.IGNORECASE,
)

# 本来 style フィールドへ入れるべきスタイル名を、行頭のラベル付きで本文へ書いて
# くることがある。実測した形:
#   iter00: "スタイル：実況風 橋の上から…"   iter01: "style: ノーマル"
#   iter01b: 'style="実況風"'（コロンでなく = ＋引用符）
# ラベル＋区切り（：: =）＋値（引用符ありも可）を落とし、後ろに本文が続けばそれは残す。
# 行の本文がラベルだけなら除去後に空になり、呼び出し側が行ごと捨てる。
_META_PREFIX = re.compile(
    r"""^\s*(?:スタイル|style|声色|tone|口調)\s*[：:=]\s*["'”「]?\S+?["'”」]?(?:[ 　]+|$)""",
    re.IGNORECASE,
)

# モデルが台本本文の頭に付ける見出しラベル（iter01b drama#1: "【summary】ハクは…"）と
# コードフェンス。ラベル部分だけ落として後続の本文は残す。
_LABEL_OR_FENCE = re.compile(r"^\s*(?:```+\w*|【[^】\n]{1,12}】)\s*", re.MULTILINE)

# JSON の別フィールド（summary）を本文の末尾へ書き足すもの。その行を丸ごと落とす。
#   iter03 drama#2: 'summary: ハクは異変を…'  drama#4: '"summary": "ハクは異変を…"'
_LEAKED_SUMMARY_LINE = re.compile(
    r"^\s*[\"'”]?summary[\"'”]?\s*[:：].*$", re.MULTILINE | re.IGNORECASE
)

# drama のト書きに「ト書き：」という見出しラベルを付けてくる（iter03b で発生）。
# 丸括弧は残し、ラベルだけ落とす: （ト書き：夕暮れの静寂）→（夕暮れの静寂）
_TOGAKI_LABEL = re.compile(r"（\s*ト\s*書\s*き?\s*[：:]\s*")

# 行末に紛れ込む HTML 風の断片（iter01 talk#3: "…価値がありますよね。/div"）。
_HTML_FRAGMENT = re.compile(
    r"\s*(?:<\s*/?\s*|/)\s*(?:div|p|span|br|li|ul)\b\s*/?>?\s*$", re.IGNORECASE
)
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002190-\U000021FF"
    "\U00002B00-\U00002BFF"
    "\U0000FE0F"
    "]+",
    flags=re.UNICODE,
)


# 中国語由来のモデル（qwen 等）がまれに混ぜる簡体字を、対応する日本語の漢字へ寄せる。
# プロンプトでも禁じているが完全には止まらないので、後処理の保険。
# 日本語の正書法にまず現れない簡体字だけを 1:1 で置換する（旦那の「那」のように
# 日本語で使う字は入れない）。中国語の文末語気助詞（啦・呗）は読み上げると
# 不自然なので落とす。判断に迷う字は触らない（VOICEVOX 側で拾える方に賭ける）。
_CJK_FIXUPS = str.maketrans({
    "谁": "誰", "标": "標", "线": "線", "话": "話", "说": "説", "见": "見",
    "关": "関", "对": "対", "实": "実", "变": "変", "觉": "覚", "过": "過",
    "还": "還", "选": "選", "样": "様", "书": "書", "长": "長", "门": "門",
    "问": "問", "间": "間", "阳": "陽", "开": "開", "应": "応",
    "啦": "", "呗": "",
})


# ひな壇トーク等のセリフ行頭に紛れ込む短いステージ指示・声色名
# （iter00: "（ヒソヒソ）まあ…"、iter05: "（のんびり）そうなんです。" ×3）。
# VOICEVOX が「かっこ…」と読むので落とす。8 字以内・句点や長文を含まないものだけ。
# drama では「（ト書き）」が正規の書式なので keep_stage_directions=True で保護する。
_LEADING_PAREN_NOTE = re.compile(r"^\s*（[^）\n。、！？]{1,8}）\s*")
# 英語版は半角括弧（"(quietly) I didn't expect that."）。句点を含まない短いものだけ。
_LEADING_PAREN_NOTE_EN = re.compile(r"^\s*\([^)\n.!?]{1,20}\)\s*")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def sanitize_text(text: str, *, keep_stage_directions: bool = False) -> str:
    en = language.current() == "en"
    text = (_CONTROL_LEAK_EN if en else _CONTROL_LEAK).sub("", text)
    text = _META_PREFIX.sub("", text)
    text = _LABEL_OR_FENCE.sub("", text)
    text = _LEAKED_SUMMARY_LINE.sub("", text)
    text = _TOGAKI_LABEL.sub("（", text)
    text = _HTML_FRAGMENT.sub("", text)
    if not keep_stage_directions:
        text = _LEADING_PAREN_NOTE.sub("", text)
        if en:
            text = _LEADING_PAREN_NOTE_EN.sub("", text)
    text = _EMOJI_PATTERN.sub("", text)
    text = _MARKDOWN_CHARS.sub("", text)
    text = text.translate(_CJK_FIXUPS)
    # _CONTROL_LEAK で拾い切れなかった孤立した山括弧（台本では使わないし、
    # どちらの TTS も記号としては読めない）。
    text = text.replace("<", "").replace(">", "")
    if en:
        # 記号を落とした跡に空白が二重に残る（"5 > 3" → "5  3"）。英語は語の
        # 区切りが空白なので、ここで詰めておかないと G2P の分割がぶれる。
        text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def _roster_entry(m: CastMember, notes: list[str] | None = None) -> str:
    en = language.current() == "en"
    entry = f"- {m.id} ({m.name}): {m.desc}" if en else f"- {m.id}（{m.name}）: {m.desc}"
    # desc 本体のすぐ下に「今日の気分」「今回の角度」を続ける（§3・§4-2: 見出しや
    # 箇条書き階層を増やさない）。notes は中身のある出演者にだけ渡ってくる。
    for note in notes or ():
        entry += f"\n  {note}"
    if _has_style_choices([m]):
        names = " / ".join(s.name for s in m.styles)
        if en:
            entry += (
                f"\n  voices: {names}"
                f' (default is "{m.default_style_name}"; set style only on the lines you want changed)'
            )
        else:
            entry += (
                f"\n  声色: {names}"
                f"（既定は「{m.default_style_name}」。変えたい行だけ style に指定）"
            )
    return entry


def _has_style_choices(appearers: list[CastMember]) -> bool:
    # 声色バリエーションは generated_drama 専用（そちらは別経路で声を固定割り当てする）。
    # ここ（ひな壇トーク等）では違和感が出やすいので、複数スタイルがあっても常に不使用。
    del appearers
    return False


def _build_prompt(
    topic: Topic,
    recent_summaries: list[str],
    appearers: list[CastMember],
    tone_hint: str,
    extra_notes: dict[str, list[str]] | None = None,
) -> str:
    # extra_notes: cast_id → その出演者の roster エントリへ足す短い1行のリスト
    # （「今日の気分: …」「今回の角度: …」）。中身のある出演者にだけ入っている（§4-1）。
    extra_notes = extra_notes or {}
    en = language.current() == "en"
    if recent_summaries:
        recent = "\n".join(f"- {s}" for s in recent_summaries)
    else:
        recent = "(none. This is the first item of the run)" if en else "（なし。今回が最初の話題）"
    roster = "\n".join(_roster_entry(m, extra_notes.get(m.id)) for m in appearers)
    host = appearers[0]
    style_guide = (
        (
            "\n## Choosing a voice\n"
            "- Some of the people above have a list of voices. On a line where the feeling shifts,"
            " you may put that name in style\n"
            "- A line with no style uses their default voice. Save it for the moments that matter;"
            " do not use it often\n"
            "- Do not put a style on anyone who has no voices listed\n"
            if en else
            "\n## 声の出し分け\n"
            "- 出演者によっては上の一覧に「声色」の候補がある。感情が動く行では、その名前を style に入れてよい\n"
            "- style を指定しない行は既定の声になる。ここぞという場面だけに絞り、多用しないこと\n"
            "- 「声色」の候補がない出演者には style を付けないこと\n"
        )
        if _has_style_choices(appearers)
        else ""
    )
    # 誰かに「今日の気分」「今回の角度」が付いているときだけ、扱い方を1文で添える
    # （§4-3: style_guide と同じ出し分け方式）。ソフトな示唆でありハード制約ではない（§4-4）。
    variation_guide = (
        (
            "\n## How to treat Today / This time\n"
            "- Some of the people above have a Today or a This time line. It is the faintest"
            " nudge, and it sits underneath their actual description and the tone above, both of"
            " which win. It is not an instruction, so do not try to make it show in everyone or"
            " in every line\n"
            if en else
            "\n## 今回の気分・角度の扱い\n"
            "- 一部の出演者に「今日の気分」「今回の角度」がある。これはキャラ本来の説明と"
            "上のトーンを優先したうえでの、ごく軽い揺らぎの示唆にすぎない。"
            "ハードな指示ではないので、全員・全行に効かせようとしないこと\n"
        )
        if any(extra_notes.get(m.id) for m in appearers)
        else ""
    )
    return language.pick(
        ja=_build_prompt_ja(topic, recent, roster, host, tone_hint, style_guide, variation_guide),
        en=_build_prompt_en(topic, recent, roster, host, tone_hint, style_guide, variation_guide),
    )


def _build_prompt_ja(topic, recent, roster, host, tone_hint, style_guide, variation_guide) -> str:
    return f"""あなたはラジオ番組「エルエルエム・ラジオ・デーモン」の放送作家です。
スタジオのひな壇に出演者が並ぶ、ワイドショー風のトーク番組です。
今回登場するのは以下の出演者で、この人たちだけで以下のネタについてのトークの台本を作ってください。

## 今回の出演者（speaker にはこの id をそのまま使うこと）
{roster}
{style_guide}{variation_guide}
## 進行の形
- {host.name}（{host.id}）が進行役。ネタを振って話を回し、事実部分を押さえ、最後に軽く落として締める
- ほかの出演者は対等に、ボケ・ツッコミ・脱線・素朴な疑問・客席目線の感想で絡む
- 全員が均等に話す必要はない。話の流れとして自然な人だけが話す
- 1人の発言は1〜3文程度。短いテンポのやり取りを混ぜる
- 話は前へ進めること。同じ出演者が前に言ったのと同じ言い回しや同じ主張を繰り返さない。
  一度出た論点は、別の人が引き取って掘り下げるか、次の論点へ渡す
- 各出演者の持ち味・口ぐせ・持ちネタ（角度）は、最初の1〜2発言でそれと分かれば十分。
  以降はその人も、持ち味のフィルター（「バズるか映えるか」「何でもスポーツに例える」など）を
  毎回かけず、ネタの中身そのものに素直に反応すること。同じ枕詞・同じ切り口を3回以上使わない
- 今回のネタの中身に全員が一度は具体的に触れること。茶化すだけ・自分の話だけで終わらせない

## 今回のネタ（この話題の扱い方: {topic.hint}）
タイトル: {topic.title}
本文:
{topic.body}

## 直前の話題（参考情報。話をつなげる義務はない）
{recent}
※ 直前の話題に無理に言及しないこと。触れるとしても、上のリストに実際に書かれている内容だけを使い、
  書かれていないこと（登場人物・ジャンル・固有名詞など）を推測で補わないこと。
  自然につながらないなら、いきなり今回のネタから話し始めてよい。

## 今の時間帯のトーン
{tone_hint or _DEFAULT_TONE_HINT}

## 制約
- {_MIN_LINES}〜{_MAX_LINES}行、合計1200〜2000文字程度で書くこと。1文ずつ細切れの行にして
  行数を稼がず、1発言1〜3文でまとめる。行数より、やり取りを最後まで展開して締めることを優先する
- 記号・絵文字・Markdown記法（*, #, ` など）を一切使わないこと。すべて音声で読み上げられる
- ネタの出どころ（RSS／Hacker News／arXiv／Wikipedia など）や、この指示文の見出し・ラベル
  （「本文」「この話題の扱い方」など）をそのまま口に出さないこと。「〜で見つけた記事です」の
  ような前置きをせず、いきなりネタの中身から話し始める
- 本文が「視聴者の反応まとめ」のように引用の寄せ集めでも、引用をそのまま台詞にしないこと。
  出演者それぞれの言葉で受け止めて話す
- 本文に書かれていない固有名詞や数値を断定的に話さないこと。
  確証がないことは「〜らしい」「〜だそうです」のように話すこと
- ただし、話題が広く知られた作品・人物・出来事（有名な映画・音楽・小説・歴史上のできごとなど）の
  場合は、世間一般によく知られている範囲の中身（主要な登場人物、物語の大筋、代表曲、時代背景など）に
  自然に触れてよい。マニアックな細部や、少しでも自信が持てないことは口にしないこと。
  年号・数値・受賞歴などの検証しづらい事実は、本文に無ければ相変わらず断定しないこと
{sensitive.SENSITIVE_TOPICS_GUIDANCE}
{language.LANGUAGE_GUIDANCE}
- 話し言葉で書くこと。書き言葉的な硬い表現は避けること
"""


def _build_prompt_en(topic, recent, roster, host, tone_hint, style_guide, variation_guide) -> str:
    """_build_prompt_ja の英語版。段落構成と制約は1対1で対応させてある。

    日本語版は docs/main-tuning-log.md の iter00〜08 で実測しながら詰めたものなので、
    こちらを直すときも「日本語版の対応する段落がなぜあるか」を先に確認すること
    （行数・文字数の目安だけは英語の情報密度に合わせて置き換えてある）。
    """
    return f"""You are the writer for a radio show called "LLM Radio Daemon".
It is a panel show: the presenters sit in a row in the studio and talk over the day's items.
Write the script for one item, using only the people listed below.

## On air this time (use these ids verbatim in speaker)
{roster}
{style_guide}{variation_guide}
## How it runs
- {host.name} ({host.id}) is presenting. Introduces the item, moves it around the panel, keeps the facts straight, and lands a light closing line
- Everyone else comes in as equals: the daft angle, the correction, the tangent, the naive question, the view from the cheap seats
- Nobody has to get equal time. Only whoever it falls to naturally
- One to three sentences per turn. Mix in some short, quick exchanges
- Keep it moving forward. Nobody repeats a line or a point they already made.
  Once a point has been made, someone else takes it further or hands on to the next one
- One or two turns is enough to establish what a person is like. After that they stop running
  everything through their own filter ("will it go viral", "it's like a football match") and
  just react to what the item actually says. Do not use the same lead-in or the same angle three times
- Everyone touches on the substance of the item at least once, specifically. Nobody gets away
  with only sending it up or only talking about themselves

## The item (how to handle it: {topic.hint})
Title: {topic.title}
Copy:
{topic.body}

## The previous item (for reference. There is no obligation to connect to it)
{recent}
Note: do not force a reference back to it. If you do mention it, use only what is actually
written in the list above; do not fill in anything that is not there (characters, genre,
proper nouns) by guessing. If it does not join up naturally, just start on this item.

## Tone for this time of day
{tone_hint or _DEFAULT_TONE_HINT_EN}

## Constraints
- {_MIN_LINES} to {_MAX_LINES} lines, around 900 to 1500 words in total. Do not pad the line
  count by chopping single sentences into separate turns; keep each turn to one to three
  sentences. Getting the exchange all the way to a close matters more than hitting the line count
- No symbols, emoji or Markdown (*, #, ` and so on). Every line is read aloud
- Do not say out loud where the item came from (RSS, Hacker News, arXiv, Wikipedia) or the
  headings and labels in these instructions ("Copy", "how to handle it"). No "so I found this
  article" preamble; start on the substance
- If the copy is a pile of quoted reactions, do not put the quotes straight into anyone's mouth.
  Each of them takes it in and says it in their own words
- Do not state proper nouns or numbers that are not in the copy as if they were fact.
  Where you are not certain, say it as "apparently" or "I gather"
- That said, when the subject is a widely known work, person or event (a famous film, a piece of
  music, a novel, something from history), it is fine to touch naturally on what is generally
  known about it: the main characters, the broad shape of the story, the best-known songs, the
  period. Stay off the obscure detail and anything you are less than sure of. Dates, figures and
  awards are still not to be asserted unless they are in the copy
{sensitive.SENSITIVE_TOPICS_GUIDANCE}
{language.LANGUAGE_GUIDANCE}
- Write it as speech. Avoid anything that reads like written prose
"""


def _build_back_announce_prompt(
    topic: Topic, appearers: list[CastMember], tone_hint: str
) -> str:
    """back_announce モード用。いま終わった曲を軽く振り返るだけの短い台本。"""
    roster = "\n".join(_roster_entry(m) for m in appearers)
    style_guide = (
        (
            "\n## Choosing a voice\n"
            "- Some of the people above have a list of voices. On a line where the feeling shifts,"
            " you may put that name in style\n"
            "- Do not put a style on anyone who has no voices listed\n"
            if language.current() == "en" else
            "\n## 声の出し分け\n"
            "- 出演者によっては上の一覧に「声色」の候補がある。感情が動く行では、その名前を style に入れてよい\n"
            "- 「声色」の候補がない出演者には style を付けないこと\n"
        )
        if _has_style_choices(appearers)
        else ""
    )
    return language.pick(
        ja=f"""あなたはラジオ番組「エルエルエム・ラジオ・デーモン」の放送作家です。
いま流れていた曲が終わり、次の曲へ切り替わったところです。
「いま終わった曲」を軽く振り返るだけの、ごく短いひとことトークの台本を作ってください。
雑談に広げたり、別の話題へ脱線したりしないこと。振り返ったらすぐ終わること。

## 今回の出演者（speaker にはこの id をそのまま使うこと）
{roster}
{style_guide}
## 台本の形
- {_BACK_ANNOUNCE_MIN_LINES}〜{_BACK_ANNOUNCE_MAX_LINES}行の短いやり取り
- いま終わった曲の曲名とアーティストに触れ、ひとこと感想を添えるだけ
- 曲の豆知識（初出年・収録アルバムなど）は、下の本文にあれば一つだけ拾ってよい。
  本文に無ければ触れないこと（曲名とアーティストだけで成立させる）
- 全員が話す必要はない。流れとして自然な人だけが話す
- 次の曲や他の話題には触れない。「では次の曲どうぞ」のような繋ぎもいらない

## いま終わった曲
{topic.body}

## 今の時間帯のトーン
{tone_hint or _DEFAULT_TONE_HINT}

## 制約
- 記号・絵文字・Markdown記法（*, #, ` など）を一切使わないこと。すべて音声で読み上げられる
- 本文に書かれていない固有名詞や数値を断定的に話さないこと
{sensitive.SENSITIVE_TOPICS_GUIDANCE}
{language.LANGUAGE_GUIDANCE}
- 話し言葉で書くこと。書き言葉的な硬い表現は避けること
""",
        en=f"""You are the writer for a radio show called "LLM Radio Daemon".
The track that was playing has just finished and the next one has started.
Write a very short exchange that does nothing but look back at the track that just ended.
Do not open it out into chat or wander onto another subject. Once it has been mentioned, stop.

## On air this time (use these ids verbatim in speaker)
{roster}
{style_guide}
## Shape of it
- A short exchange, {_BACK_ANNOUNCE_MIN_LINES} to {_BACK_ANNOUNCE_MAX_LINES} lines
- Name the track and the artist that just finished and add one thought about it. That is all
- One piece of trivia about the track (the year, the album) is allowed if it is in the copy below.
  If it is not there, leave it out and let the title and the artist carry it
- Not everyone has to speak. Only whoever it falls to naturally
- Do not mention the next track or any other subject. No "and here's the next one" hand-off

## The track that just finished
{topic.body}

## Tone for this time of day
{tone_hint or _DEFAULT_TONE_HINT_EN}

## Constraints
- No symbols, emoji or Markdown (*, #, ` and so on). Every line is read aloud
- Do not state proper nouns or numbers that are not in the copy as if they were fact
{sensitive.SENSITIVE_TOPICS_GUIDANCE}
{language.LANGUAGE_GUIDANCE}
- Write it as speech. Avoid anything that reads like written prose
""",
    )


def _is_back_announce(topic: Topic, content: ContentConfig) -> bool:
    """radio の song_talk = "back"（曲間で「いま終わった曲」を振り返る）か。

    topic.source は曲情報の出どころ（MusicBrainz）で、コーナー種別ではない。
    """
    return content.is_back_announce and topic.source == "musicbrainz"


def generate_script(
    topic: Topic,
    recent_summaries: list[str],
    llm_config: LLMConfig,
    content: ContentConfig,
    cast: list[CastMember],
    mood: dict[str, str] | None = None,
    appearers: list[CastMember] | None = None,
) -> Script | None:
    back_announce = _is_back_announce(topic, content)
    # 出演者数は back_announce でも通常どおり [[content]] の min/max_speakers に従う。
    # ``appearers`` を渡された場合はそれをそのまま使う（呼び出し側でコンテンツの
    # 区間内キャスト固定を管理している場合。省略時は毎回ここで抽選する＝従来どおり）。
    if appearers is None:
        appearers = pick_speakers(content, cast)
    by_id = {m.id: m for m in appearers}
    valid_ids = set(by_id)
    style_names = (
        sorted({s.name for m in appearers for s in m.styles if len(m.styles) > 1})
        if _has_style_choices(appearers) else []
    )
    if back_announce:
        min_lines, max_lines = _BACK_ANNOUNCE_MIN_LINES, _BACK_ANNOUNCE_MAX_LINES
        prompt = _build_back_announce_prompt(topic, appearers, content.tone_hint)
    else:
        min_lines, max_lines = _MIN_LINES, _MAX_LINES
        prompt = _build_prompt(
            topic, recent_summaries, appearers, content.tone_hint,
            _appearer_notes(appearers, content, mood),
        )
    schema = _script_schema(
        [m.id for m in appearers], style_names, min_lines, max_lines
    )

    for attempt in range(2):  # 初回 + リトライ1回まで
        try:
            raw_content = llm_http.chat(
                llm_config, prompt, schema=schema,
                temperature=llm_config.temperature,
            )
            parsed = json.loads(raw_content)
            lines_raw = parsed["lines"]
            if len(lines_raw) < min_lines:
                raise ValueError(f"too few lines: {len(lines_raw)}")

            lines = []
            for line in lines_raw:
                speaker = line["speaker"]
                if speaker not in valid_ids:
                    # 想定外の話者IDが混ざった行だけ捨てて、残りは活かす
                    logger.warning("dropping line with unknown speaker %r", speaker)
                    continue
                text = sanitize_text(line["text"])
                if not text:
                    continue
                # style は任意。その出演者が持たないスタイル名なら捨てる（TTS側で既定へフォールバック）。
                style = line.get("style") or None
                if style is not None and style not in style_names:
                    logger.debug("ignoring style %r not available for %r", style, speaker)
                    style = None
                lines.append(ScriptLine(speaker=speaker, text=text, style=style))

            if not lines:
                raise ValueError("no valid lines after sanitization")

            reading.apply_speech_reading(lines, llm_config)

            return Script(
                topic_id=None,
                topic_title=topic.title,
                lines=lines,
                appearer_ids=tuple(m.id for m in appearers),
            )

        except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError) as e:
            logger.warning(
                "script generation failed (attempt %d/2) for topic %r: %s",
                attempt + 1,
                topic.title,
                e,
            )

    logger.error("giving up on topic %r after 2 attempts", topic.title)
    return None
