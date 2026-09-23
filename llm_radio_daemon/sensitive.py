"""政治・宗教・実在の個人/団体まわりなど、扱いに気を付けたい話題を避けるための共有ヘルパー。

使い方は 3 つ:

* ``SENSITIVE_TOPICS_GUIDANCE`` … 台本生成プロンプトへ差し込む「踏み込まない」指示
  （ScriptThread 共通ガード）。既存のトピックを与えて喋らせる場面では話題そのものは
  変えられないので、せめて扱い方を穏当にさせるための複数行の箇条書き。実在の個人・
  団体・国家・民族・集団への非難・批判・応援、政治・宗教・思想への肩入れ、誹謗中傷、
  性的・R-18・グロテスクな表現を避けさせるほか、医療・健康は断定を避けさせ、投資・
  金融商品の推奨/勧誘もしないよう釘を刺す（VOICEVOX の一部キャラクターの利用規約が、
  この種の用途を禁じているため）。**language.LANGUAGE_GUIDANCE と同じく起動時に
  set_language() で差し替わる**ので、``from .sensitive import SENSITIVE_TOPICS_GUIDANCE``
  と書くと import 時の値で固定されて効かない。必ず ``from .. import sensitive`` して
  ``{sensitive.SENSITIVE_TOPICS_GUIDANCE}`` と属性参照すること。挿入先では既に複数行の
  箇条書きになっているので、呼び出し側で先頭に ``- `` を付け足さないこと。
* ``looks_sensitive()`` … トピックを配信キューに載せる前のふるい。話題そのものが
  政治／宗教に寄っている記事（Wikipedia のランダム記事、Hacker News の新着）を、
  台本生成にかける前に丸ごと落とす。SENSITIVE_TOPICS_GUIDANCE とは独立していて、
  ここでの追加項目（医療・投資など）はふるいの対象語には加えていない
  （話題そのものを弾くのではなく、扱い方をガードする側で対応する方針のため）。
* ``blocked_term()`` … 生成済みのセリフを読み上げ直前に照合する NG ワードフィルタ
  （TTSThread が使う）。プロンプトの指示を LLM が無視した場合の最後の層で、性的表現・
  差別語・直接的な罵倒を含む行をキャラクターの声で読ませない。

完全な検閲ではなく「取りこぼしはあっても、明らかなものは避ける」程度のもの。
個人利用（非配信）向けなので、過剰に絞って供給を止めるより、明白なものだけ弾く
バランスにしてある。
"""

from __future__ import annotations

import re

_JA_GUIDANCE = (
    "- 実在の個人・団体・国家・民族・集団を、非難・批判・応援の対象にしない。事実と話題の"
    "面白さだけを話すこと\n"
    "- 特定の宗教・宗派・政党・政治家・政治思想について、その是非や優劣には踏み込まないこと。"
    "触れる場合は、対立や優劣を煽らず、歴史的・一般的な事実の側面にとどめること\n"
    "- 誹謗中傷や、第三者が不快になる言い方をしないこと（出演者同士の軽いツッコミは可）\n"
    "- 性的・R-18な表現、過度にグロテスクな表現をしないこと\n"
    "- 医療・健康の話は断定せず「〜らしい」「〜だそうです」のように話し、誤解を生む"
    "言い方をしないこと\n"
    "- 投資・金融商品の推奨や勧誘をしないこと。相場や企業の話題は事実の紹介にとどめること"
)

_EN_GUIDANCE = (
    "- Do not single out a real individual, organization, nation, ethnicity or group for "
    "criticism or support. Stick to facts and to what makes the topic interesting\n"
    "- Do not take a side on any religion, denomination, political party, politician or "
    "political ideology, or rank one above another. If it comes up, keep it to the plain "
    "historical or factual side of it, without playing up the conflict\n"
    "- No personal attacks, and nothing that would make a third party uncomfortable (light "
    "ribbing between the cast is fine)\n"
    "- No sexual or R-18 content, and nothing gratuitously graphic or gross\n"
    "- Don't state medical or health claims as fact. Hedge them as \"apparently\" or \"by all "
    "accounts\", and don't say anything that could be misleading\n"
    "- Don't recommend or push investments or financial products. Keep market and company "
    "talk to plain facts"
)

_GUIDANCE_BY_LANG = {"ja": _JA_GUIDANCE, "en": _EN_GUIDANCE}

# 既定は日本語版。set_language() を呼ばなければ従来どおりの挙動になる。
SENSITIVE_TOPICS_GUIDANCE = _JA_GUIDANCE


def set_language(lang: str) -> None:
    """[locale] lang に合わせて差し込む文言を切り替える（起動時に一度だけ呼ぶ）。

    language.set_language() と同じ約束で、未知の言語コードなら日本語版のまま据え置く。
    """
    global SENSITIVE_TOPICS_GUIDANCE
    SENSITIVE_TOPICS_GUIDANCE = _GUIDANCE_BY_LANG.get(
        (lang or "").strip().lower(), _JA_GUIDANCE
    )

# ASCII 側は単語境界で判定する（software / culture / warehouse などへの誤爆を防ぐ）。
_ASCII_TERMS = (
    # 政治
    r"politic\w*", r"election\w*", r"president\w*", r"prime minister",
    r"parliament\w*", r"congress\w*", r"senator\w*", r"lawmaker\w*",
    r"democrats?", r"republicans?", r"left-wing", r"right-wing",
    r"geopolit\w*", r"sanctions?", r"sanctioned", r"coup", r"regime",
    r"referendum", r"legislation", r"impeach\w*", r"dictator\w*",
    # 宗教
    r"relig\w*", r"church\w*", r"mosque\w*", r"synagogue\w*",
    r"islam\w*", r"muslims?", r"christian\w*", r"christianity",
    r"catholic\w*", r"protestant\w*", r"buddhis\w*", r"hindus?",
    r"judaism", r"jewish", r"bible", r"biblical", r"quran", r"koran",
    r"gospel", r"clergy", r"priests?", r"pope", r"papal", r"cult",
    r"cults", r"theolog\w*", r"worship", r"prophet", r"scriptures?",
    r"sermons?",
)

# CJK 側は単語境界が無いので素直に部分一致。誤爆しやすい語（保守 = maintenance など）は
# 入れていない。
_CJK_TERMS = (
    # 政治
    "政治", "政党", "政権", "与党", "野党", "選挙", "首相", "大統領",
    "国会", "議員", "内閣", "右翼", "左翼", "共産党", "自民党", "民主党",
    "政治家", "政策", "外交", "経済制裁", "侵攻", "戦争", "紛争",
    "クーデター", "抗議デモ",
    # 宗教
    "宗教", "宗派", "教会", "神社", "寺院", "仏教", "キリスト教", "イスラム",
    "ムスリム", "ユダヤ教", "ヒンドゥー", "神道", "教義", "聖書", "コーラン",
    "クルアーン", "礼拝", "布教", "信仰", "教祖", "カルト", "新興宗教",
    "教皇", "法王", "司教", "牧師", "僧侶", "聖職者", "経典",
)

_ASCII_RE = re.compile(r"\b(?:" + "|".join(_ASCII_TERMS) + r")\b", re.IGNORECASE)


def looks_sensitive(*texts: str | None) -> bool:
    """渡したテキストのどれかが政治／宗教に寄っていそうなら True。"""
    for text in texts:
        if not text:
            continue
        if _ASCII_RE.search(text):
            return True
        if any(term in text for term in _CJK_TERMS):
            return True
    return False


# ---------------------------------------------------------------------------
# 出力側の NG ワードフィルタ（TTSThread が VOICEVOX / Kokoro へ渡す直前に使う）
#
# SENSITIVE_TOPICS_GUIDANCE はプロンプトでの「お願い」なので、LLM が無視すれば素通り
# する。こちらは生成済みのセリフを機械的に照合する最後の層で、VOICEVOX の各キャラクター
# 利用規約が禁じがちな性的表現・差別語・直接的な罵倒を、キャラクターの声で読ませない
# ためのもの。一致した行は読み上げず（無音に差し替え）、ログにだけ残す。
#
# looks_sensitive() とは目的が違う（あちらは話題のふるい、こちらはセリフのふるい）。
# 政治・宗教の語は入れていない。ニュースや偉人伝で事実として触れるのは許しているため。
# ラジオドラマ（ミステリー等）の筋を壊さないよう、「殺す」「死」のような物語で普通に
# 出る語も入れていない。明白に不適切なものだけに絞り、取りこぼしは許容する方針。
# 配信などで音声を公開する人は、必要に応じてここへ語を足すこと。
# ---------------------------------------------------------------------------

# 誤爆を避けるための約束:
# * ASCII 側は単語境界つき・大文字小文字無視で照合する（部分一致させない）。
#   Moby-Dick の dick、堤防の dyke、Super Bowl XXX、美術の nude、鳥の tit のように
#   普通の意味がある語は入れていない。
# * CJK 側は単語境界が無いので、他の語の一部になりにくい長さ・形のものだけにし、
#   必要なら前後の文字で除外する（フェラーリ・かたわら・めくらない・「田中氏ね」・
#   「生きる価値がある」・インドシナ人・グレイプバイン などを弾かないため）。
_NG_ASCII_TERMS = (
    # 性的表現
    r"porn\w*", r"hentai", r"blowjobs?", r"handjobs?", r"cumshots?", r"dildos?",
    r"masturbat\w*", r"orgasms?", r"pussy", r"sluts?", r"whores?",
    # 差別語
    r"nigg(?:er|a)s?", r"faggots?", r"fags", r"spics?", r"kikes?", r"trann(?:y|ies)",
    # 直接的な罵倒・加害の呼びかけ
    r"kill yourself", r"kys", r"fuck\w*", r"motherfuck\w*", r"cunts?",
)

_NG_CJK_TERMS = (
    # 性的表現
    r"オナニー", r"自慰", r"フェラチオ", r"パイズリ", r"手コキ", r"おっぱいを?揉",
    r"ちんぽ", r"ちんこ(?!う)", r"まんこ(?!う)", r"(?<!グ)レイプ", r"強姦",
    r"アダルトビデオ", r"[AＡ][VＶ]女優", r"エロ動画", r"風俗嬢",
    # 差別語
    r"ガイジ(?!ン)", r"キチガイ", r"気違い", r"基地外", r"つんぼ", r"支那人",
    r"部落民", r"ホモ野郎", r"オカマ野郎",
    # 直接的な罵倒・加害の呼びかけ（「死ねない」「死ねば」「死ねた」などは物語で普通に出る）
    r"死ね(?![なばるずたまそ])", r"殺すぞ", r"ぶっ殺", r"消えろカス", r"ゴミ人間",
    r"生きる価値(?:が|も|なんて)?(?:ない|無い)", r"自殺しろ", r"首(?:を)?吊れ",
)

_NG_RE = re.compile(
    r"\b(?:" + "|".join(_NG_ASCII_TERMS) + r")\b|" + "|".join(_NG_CJK_TERMS),
    re.IGNORECASE,
)


def blocked_term(text: str | None) -> str | None:
    """セリフに NG ワードが含まれていれば、最初に見つかった語を返す（無ければ None）。"""
    if not text:
        return None
    m = _NG_RE.search(text)
    return m.group(0) if m else None
