"""英語の文分割（日本語の「。！？」で切るループに対応するもの）。

なぜ共有モジュールにしたか
--------------------------
日本語版は「。！？」という**文字の一覧**で切れるので、必要な実装は3行のループで済み、
``aozora/parser.py`` と ``generated_drama/parser.py`` がそれぞれ自前に持っていた。
英語で同じことをやると省略記号（``Mr.`` ``a.m.`` ``J. R. R.``）と小数点で誤爆し、
「1文」が壊れる。朗読コーナー・ラジオドラマ・Kokoro の510トークン分割の3か所が同じ
判定を要るので、英語版だけここへ一本化した。

**日本語版のループは各パーサに残してある。** 共通化すると ``。！？`` と ``。！？!?``
という既存の差（aozora と generated_drama で違う）を1つに寄せることになり、
日本語版の出力が変わる。英語を足すためにそれを動かす理由が無い（language.py §5.1 と
同じ方針）。呼び出し側は

    if language.current() == "en":
        return sentences.split_en(text)
    （従来どおりの日本語ループ）

という形で分岐する。

契約
----
``"".join(split_en(text)) == text``。区切りの空白は直前の文の末尾に付けて返すので、
呼び出し側が ``buf += sent`` で連結すれば原文がそのまま戻る（``aozora`` 側は
``char_offset`` を ``len(seg)`` の累計で数えているので、ここを削ると進捗位置がずれる）。
"""

from __future__ import annotations

import re

# 文末記号＋閉じ括弧・閉じ引用符＋空白。ここが「文の切れ目かもしれない」候補。
_BREAK_RE = re.compile(r"""([.!?][.!?]*)(["'”’»)\]]*)(\s+)""")

# 直前の語がこれなら、ピリオドは文末ではなく省略記号。小文字化して末尾の "." を
# 落とした形で引く（"a.m" → "am"）。判断に迷うものは入れない（入れすぎると
# 本当の文末で切れなくなり、1文が長くなって Kokoro の510トークンに収まらなくなる）。
_ABBREVIATIONS = frozenset(
    """
    mr mrs ms mx dr prof rev fr st sgt capt lt col gen adm hon gov sen rep
    jr sr esq
    no nos vs etc al inc ltd co corp dept est fig figs vol vols ch chap pp approx
    jan feb mar apr jun jul aug sep sept oct nov dec
    mon tue tues wed weds thu thur thurs fri sat sun
    am pm ie eg cf ca viz ibid
    """.split()
)

# ピリオドの手前にある語（"Mr." の "Mr"、"a.m." の "a.m"）。
_WORD_BEFORE_RE = re.compile(r"([A-Za-z][A-Za-z.]*)$")


def _is_boundary(text: str, m: re.Match[str]) -> bool:
    tail = m.end()
    if tail >= len(text):
        return True  # 末尾の空白。ここで切っても失うものが無い

    # 次の文が小文字で始まるなら文末ではない。省略記号の取りこぼしと、
    # 引用のあとに続く伝達節（``“Go!” he shouted.``）の両方をここで止める。
    if text[tail].islower():
        return False

    if m.group(1) != ".":
        return True  # "!" "?" "..." は略記に使われない

    before = _WORD_BEFORE_RE.search(text[: m.start()])
    if before is None:
        return True
    # 語中のピリオドも落として引く（"a.m" → "am"、"i.e" → "ie"）。strip(".") だと
    # 語中の点が残って表を引けない。
    word = before.group(1).lower().replace(".", "")
    if len(word) == 1:
        return False  # イニシャル（"J. R. R. Tolkien"）
    return word not in _ABBREVIATIONS


def split_en(text: str) -> list[str]:
    """英語の文へ割る。区切りの空白は直前の文に付く（``"".join()`` で原文に戻る）。"""
    out: list[str] = []
    start = 0
    for m in _BREAK_RE.finditer(text):
        if not _is_boundary(text, m):
            continue
        out.append(text[start : m.end()])
        start = m.end()
    if start < len(text):
        out.append(text[start:])
    return [s for s in out if s.strip()]
