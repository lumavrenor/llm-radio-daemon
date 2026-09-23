"""台本を書かせる言語の指示（プロンプトへ差し込む共通文言）。

なぜ必要か
----------
日本語版のプロンプトには「原稿にアルファベットを残さず、必ず日本語で書くこと」という
制約が各所に埋まっている。英語版でこれをそのまま送ると、英語の台本が生成できない。
[locale] lang に応じてこの一段落だけを差し替える。

使い方は sensitive.py の SENSITIVE_TOPICS_GUIDANCE と同じで、プロンプトの
f-string に ``{LANGUAGE_GUIDANCE}`` として埋める。f-string は呼び出し時に
評価されるので、起動時に set_language() で一度差し替えれば全プロンプトに効く
（main.py の set_pinned_cast_ids と同じ流儀）。

TTS 固有の注意もここに入れている。英語の Kokoro は "2:30" のような記号混じりの
表記を音素化できない（コロンが未知トークンになる）ので、数字や時刻を語に開かせる。
プロンプトで予防するのは日本語版と同じ方針（記号・絵文字をここで止めているのと同じ）。

プロンプト本文そのものの言語（pick）
----------------------------------
LANGUAGE_GUIDANCE が差し替えるのは「何語で書け」の一段落だけで、プロンプト**本文**は
日本語のままだった。英語版では本文も英語で送りたいが、日本語版のプロンプトは
docs/main-tuning-log.md の iter00〜08 で実測しながら詰めたものなので、共通の英語
プロンプトへ一本化すると、そのチューニングを丸ごと捨てることになる。

そこで「日本語の本文はそのまま残し、英語版を隣に足して実行時に選ぶ」という形にした。
``mood._build_prompt`` と ``request_announce._TEMPLATES`` が先に同じことを個別に
やっていたので、その分岐を1か所へまとめたのが :func:`pick`:

    return language.pick(
        ja=f\"\"\"（従来どおりの日本語プロンプト）\"\"\",
        en=f\"\"\"(the English rewrite)\"\"\",
    )

未知の言語コードや、その言語ぶんを書いていないプロンプトでは ja へ落ちる
（set_language() と同じ「黙って英語に倒すより今までどおり動く」方針）。
"""

from __future__ import annotations

_JA_GUIDANCE = """- すべて音声で読み上げるため、原稿にアルファベットを残さず、必ず日本語(かな漢字・カタカナ)で書くこと
- 中国語の漢字・簡体字を混ぜないこと。日本語の常用漢字だけを使う（「谁」→「誰」、「标」→「標」、
  「那个」「也觉得」のような中国語表現を書かない）
- ハングル（韓国語の文字）を混ぜないこと。「情緒 있죠」のように語尾や単語を韓国語にしない。
  すべて日本語で書く（中国語の漢字と同じ扱い）
- 英語などの固有名詞（人名・団体名・略語）の扱い:
  発音が確実に分かるものだけ、素直なカタカナ表記にすること（例: Joe Gibbs → ジョー・ギブス）。
  読み方に自信がないもの、無理にカタカナ化すると珍妙になりそうなものは、固有名詞を出さずに
  「アメリカのあるストックカーレースのチーム」のように一般的な言い方へ言い換えること。
  綴りから発音を推測して不正確なカタカナ読みをでっち上げないこと（これがいちばんやってはいけないこと）"""

_EN_GUIDANCE = """- Write everything in natural spoken English. Every line is read aloud by a speech synthesiser
- Spell out anything that would otherwise be read as a symbol or a bare digit string:
  "two thirty in the morning" not "2:30 AM", "seven to three" not "7-3", "nineteen seventy nine"
  not "1979". Do not leave colons, slashes or hyphens inside numbers
- Proper nouns are pronounced by a guessing model and unusual names are often mangled.
  If a name is not widely known, prefer a general description ("a stock car racing team out of
  North Carolina") over the name itself. Never invent a spelling to force a pronunciation"""

_GUIDANCE_BY_LANG = {"ja": _JA_GUIDANCE, "en": _EN_GUIDANCE}

# 既定は日本語版。set_language() を呼ばなければ従来どおりの挙動になる。
LANGUAGE_GUIDANCE = _JA_GUIDANCE

_current_lang = "ja"


def set_language(lang: str) -> None:
    """[locale] lang に合わせて台本の言語指示を差し替える（起動時に一度だけ呼ぶ）。

    未知の言語コードなら日本語版のまま据え置く（黙って英語に倒すより、
    今までどおり動いたほうが事故が小さい）。
    """
    global LANGUAGE_GUIDANCE, _current_lang
    key = (lang or "").strip().lower()
    LANGUAGE_GUIDANCE = _GUIDANCE_BY_LANG.get(key, _JA_GUIDANCE)
    _current_lang = key if key in _GUIDANCE_BY_LANG else "ja"


def current() -> str:
    """今の言語コード（"ja" / "en"）。LLM を通さない定型文の出し分けに使う。"""
    return _current_lang


def is_supported(lang: str) -> bool:
    return (lang or "").strip().lower() in _GUIDANCE_BY_LANG


def pick(*, ja: str, en: str | None = None) -> str:
    """今の言語のプロンプト本文（定型文）を返す。無ければ日本語版。

    呼び出し側は両方の文字列を組んでから渡すことになる（f-string は引数評価時に
    展開される）が、プロンプト1本ぶんの文字列連結でしかないので実行時コストは
    無視できる。分岐を書く側が ``if language.current() == "en":`` を毎回書かずに
    済むほうを取った。
    """
    if _current_lang == "en" and en is not None:
        return en
    return ja
