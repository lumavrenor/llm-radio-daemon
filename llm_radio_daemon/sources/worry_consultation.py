"""悩み相談トークショー（v3 追加ソース）。docs/spec-worry-consultation.md 参照。

config.toml の [[content]] type = "worry_consultation" で時間帯（schedule）と出演人数を
指定する。時間帯ゲートは番組表型スケジューラ（schedule.active_content）が担当し、
SourceThread がアクティブなときだけこの fetch() を回す。このソース自体は時間帯を
意識せず「生成 → yield → poll_interval 待ち」を繰り返すだけでよい。

他のソースと違い外部通信は行わず、Ollamaに「架空の相談」を生成させる。実在の相談者を
一切介在させないことで、「特定個人の私的な悩みを無断で番組ネタにする」懸念そのものを
発生させない（詳細は spec のなぜこの設計かを参照）。

生成は2段階に分ける:
  1. ペルソナ（年代・属性・お題）をランダムに振る。ここは値の候補が少なく列挙できる
     カテゴリなので、Python側の重み付き抽選で決める（LLM呼び出しなし）。ここをLLMに
     任せると、v1のWikimediaソースで分かった「切り口を渡すと収束する」のと同種の
     問題で結局似た属性ばかりになりやすい。抽選なら多様性を機械的に保証できる
  2. 振られたペルソナを条件に、LLMへ悩み本文（タイトル＋本文）を書かせる
     （ここだけがLLM呼び出し）
1回のプロンプトで属性決定と本文執筆の両方を頼むと、属性の多様性が出にくく似た
相談ばかりになる傾向があったため、この2段階に分けている。

external_id は生成本文のハッシュ（完全な架空データのためURL/IDが存在しない）。
このため4.2節・第1段階（完全一致排除）はほぼ機能せず、4.2節・第2段階の embedding
類似度チェックが実質必須になる。ただしそのチェックは SourceThread.enqueue_topic 側で
ソースを問わず既に一律に効いている（config.toml [embedding] enabled=true が既定）ため、
このソース側で個別対応する必要はない。
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import threading
from dataclasses import dataclass
from typing import Iterator

import requests

from .. import llm_http
from ..config import LLMConfig
from ..script.ollama_client import sanitize_text
from .. import language
from ..source_status import SourceStatus
from . import Topic

logger = logging.getLogger(__name__)

# 相談者のペルソナ候補。spec では「config 化するかコード内固定か」を未確定にしたまま
# コード内固定にしていたが、英語版を足す段で config へ出した ——
#   * 候補そのものが**番組の中身**（どんな相談が流れるか）で、放送中に差し替えたい種類の値
#   * 言語ごとに違う（「主婦/主夫」「50代以上」は英語の相談コーナーの区分ではない）
# の2点から、コーナーの設定として config_content.toml に置くほうが素直だった。
#
# 下は config に何も書かなかったときの既定。日本語版の設定を書かずに動かしても
# 従来とまったく同じ抽選になるよう、ja の値は以前の定数のまま。
_DEFAULT_PERSONAS = {
    "ja": {
        "age_groups": ("20代", "30代", "40代", "50代以上"),
        "roles": ("会社員", "学生", "主婦/主夫", "自営業"),
        # 人間関係・仕事はありがちな相談として厚めに重み付け（spec参照）
        "themes": ("人間関係", "仕事", "家族", "将来", "お金", "恋愛"),
        "theme_weights": (3, 3, 2, 2, 1, 2),
    },
    "en": {
        "age_groups": ("in their twenties", "in their thirties", "in their forties",
                       "fifty or over"),
        "roles": ("works in an office", "is a student", "is at home with the family",
                  "is self-employed"),
        "themes": ("people around them", "work", "family", "the future", "money",
                   "someone they like"),
        "theme_weights": (3, 3, 2, 2, 1, 2),
    },
}


def _defaults() -> dict:
    return _DEFAULT_PERSONAS.get(language.current(), _DEFAULT_PERSONAS["ja"])

# 台本プロンプトへ渡す「この話題の扱い方」。LLM が書いた文ではないので言語別に持つ。
_HINT = {
    "ja": (
        "これは架空の相談です。司会が相談に乗り、ひな壇がそれぞれの"
        "立場から意見を言ってください。"
    ),
    "en": (
        "This letter is made up. The host takes it seriously and answers it, and the "
        "rest of the panel weigh in from wherever they each stand."
    ),
}

_WORRY_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "body": {"type": "string"},
    },
    "required": ["title", "body"],
}


@dataclass
class WorryPersona:
    """相談者の属性。値は _DEFAULT_PERSONAS か [[content]] の設定から来る。"""

    age_group: str
    role: str
    theme: str


class WorryConsultationSource:
    name = "worry_consultation"

    def __init__(
        self,
        llm_config: LLMConfig,
        poll_interval_sec: float = 600.0,
        stop_event: threading.Event | None = None,
        status: SourceStatus | None = None,
        age_groups: list[str] | None = None,
        roles: list[str] | None = None,
        themes: list[dict] | None = None,
    ):
        """``age_groups`` / ``roles`` / ``themes`` は [[content]] の設定。

        ``themes`` は ``[{"name": "...", "weight": 3}, ...]``（weight 省略で 1）。
        どれも省略・空なら :data:`_DEFAULT_PERSONAS` の今の言語ぶんを使う。
        """
        self._llm_config = llm_config
        self._poll_interval_sec = max(poll_interval_sec, 60.0)
        self._stop_event = stop_event or threading.Event()
        self._status = status

        d = _defaults()
        self._age_groups = list(age_groups) if age_groups else list(d["age_groups"])
        self._roles = list(roles) if roles else list(d["roles"])
        if themes:
            # 名前が空の項目は落とす（config の書き間違いで空文字を抽選しないため）
            pairs = [
                (str(t.get("name", "")).strip(), max(1, int(t.get("weight", 1))))
                for t in themes
                if str(t.get("name", "")).strip()
            ]
        else:
            pairs = list(zip(d["themes"], d["theme_weights"]))
        self._themes = [name for name, _ in pairs]
        self._theme_weights = [w for _, w in pairs]

    def fetch(self) -> Iterator[Topic]:
        """架空の相談を1件生成しては poll_interval 待つ。時間帯ゲートは SourceThread が担当。"""
        while not self._stop_event.is_set():
            topic = self._generate_topic()
            if topic is not None:
                yield topic

            if self._status is not None:
                self._status.cycle_end(1 if topic is not None else 0, self._poll_interval_sec)
            if self._stop_event.wait(self._poll_interval_sec):
                return

    def _generate_topic(self) -> Topic | None:
        persona = self._generate_persona()
        worry = self._generate_worry(persona)
        if worry is None:
            return None
        title, body = worry
        return Topic(
            source=self.name,
            external_id=hashlib.sha256(body.encode("utf-8")).hexdigest(),
            title=title,
            body=body,
            url=None,
            hint=_HINT.get(language.current(), _HINT["ja"]),
        )

    def _generate_persona(self) -> WorryPersona:
        """相談者属性をランダムに振る（LLM不使用。理由はモジュールdocstring参照）。"""
        return WorryPersona(
            age_group=random.choice(self._age_groups),
            role=random.choice(self._roles),
            theme=random.choices(self._themes, weights=self._theme_weights, k=1)[0],
        )

    def _build_prompt(self, persona: WorryPersona) -> str:
        return language.pick(
            ja=f"""あなたはラジオ番組「エルエルエム・ラジオ・デーモン」の放送作家です。
深夜帯の「悩み相談コーナー」向けに、ありそうな架空の相談を1件でっち上げてください。
実在の人物・実際にあった出来事を元にせず、完全な創作として書くこと。

## 相談者の設定
- 年代: {persona.age_group}
- 属性: {persona.role}
- 悩みのテーマ: {persona.theme}

## トーン（最重要）
- これは深夜ラジオの「くすっと笑える」悩み相談コーナーです。聞き手が思わず
  笑ってしまう、身近で軽い悩みにすること。しんみりさせる話・深刻な話は禁止
- 例えるなら「靴下がいつも片方だけなくなる」「相方のいびきがうるさすぎる」
  「推し活にお金をかけすぎて貯金がない」のような、あるあるネタ・ちょっとした
  困りごとのレベルに留めること
- 次のような重い内容は絶対に書かないこと: 希死念慮・自殺・自傷、生死に関わる
  悩み、深刻な病気、虐待、いじめ、ハラスメント、貧困、孤独死、将来を悲観して
  絶望するような内容。人生の意味や生きる価値を問うような重い哲学的な悩みも禁止
- 「将来」「お金」のテーマであっても、深刻な将来不安ではなく、「貯金が全然
  増えない」「気づいたら年齢だけ重ねていた（でも軽い自虐）」程度の、笑って
  流せる範囲に軽くとどめること

## 書き方
- title には相談の一言タイトルを短く書くこと（10〜20文字程度）
- body には相談本文を、相談者の一人称で200〜400文字程度で書くこと
- 深夜ラジオへの投稿らしい、少しくだけた文体で書くこと
- 具体的すぎる固有名詞（実在の会社名・学校名・地名等）は出さず、
  「職場の先輩」「行きつけの店」のように一般化して書くこと
- 相談の中身を宗教・宗派・政党・政治思想がテーマの悩みにしないこと。
  日常の人間関係・仕事・家族・お金・将来といった範囲にとどめること
- 実在の人物名・団体名を挙げて非難したり持ち上げたりしないこと
- 記号・絵文字・Markdown記法（*, #, ` など）を一切使わないこと
""",
            en=f"""You are the writer for a radio show called "LLM Radio Daemon".
Invent one plausible letter for the late-night problems slot. It is entirely made up:
do not base it on a real person or something that actually happened.

## The person writing in
- Age: {persona.age_group}
- Situation: {persona.role}
- What the letter is about: {persona.theme}

## Tone (this matters most)
- This is the gently funny kind of problems slot, the sort that makes someone driving
  home at night laugh out loud. Keep it small, ordinary and recognisable.
  Nothing sad. Nothing heavy
- The register to aim for: socks that only ever come back one at a time, a partner who
  snores loudly enough to be heard through a wall, spending so much on a hobby that
  there is nothing left over. Everyday annoyances, nothing more
- Never write any of the following: suicidal thoughts, self-harm, anything life
  threatening, serious illness, abuse, bullying, harassment, poverty, dying alone,
  or despair about the future. No heavy philosophical letters about whether life
  is worth living either
- Even when the subject is money or the future, keep it light enough to laugh off:
  savings that never seem to grow, suddenly being older than you had noticed.
  Not real financial fear

## How to write it
- title: a short one-line title for the letter (under about eight words)
- body: the letter itself, in the writer's own voice, roughly 120 to 220 words
- Write it the way someone actually writes in to a late-night show: loose, a bit
  self-deprecating, not polished
- No real company names, schools or places. Keep it general: "someone senior at work",
  "the place I always go"
- Do not make the letter about religion, denomination, a political party or a political
  position. Keep it to everyday life: people, work, family, money, the future
- Do not name a real person or organization to criticize or praise them
- No symbols, emoji or Markdown (*, #, ` and so on). Every line is read aloud, so spell
  out numbers rather than leaving digits and colons in the text
""",
        )

    def _generate_worry(self, persona: WorryPersona) -> tuple[str, str] | None:
        prompt = self._build_prompt(persona)
        # 他ソースは失敗時しかログを出さないが、このソースは生成にOllama呼び出しが
        # 1〜数分かかることがあり、進行中かどうかログだけでは分からないと運用時に
        # 困る（無音を作らない方針の監視上も重要）。開始・完了をINFOで残す。
        logger.info(
            "generating worry consultation (%s, persona=%s/%s/%s) ...",
            self._llm_config.model, persona.age_group, persona.role, persona.theme,
        )
        try:
            content = llm_http.chat(
                self._llm_config, prompt, schema=_WORRY_SCHEMA,
                temperature=self._llm_config.temperature,
            )
            parsed = json.loads(content)
            title = sanitize_text(parsed["title"])
            body = sanitize_text(parsed["body"])
            if not title or not body:
                raise ValueError("empty title/body")
            logger.info("worry consultation generated: %r", title)
            return title, body[:2000]
        except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError) as e:
            logger.warning("worry consultation generation failed: %s", e)
            return None
