"""ラジオドラマの本文を「誰が読む何行か」に割るパーサー（v6 §4.7.3）。

朗読は基本的に地の文（ナレーター）の連続読み上げで、そこにセリフが挟まる形になる。
4.3 のひな壇掛け合い（LLM が speaker を振った JSON）とは別物なので、確定済み本文を
この軽量パーサーで割り、``ScriptLine`` を直接組み立てて TTSThread へ渡す。

**ここは複雑にしすぎない**（§4.7.3）。話者が確実に分からないセリフはナレーター読みへ
フォールバックする。違和感が大きければ ``characters.json`` 側にタグを増やす方向で拡張する。

英語版（``[locale] lang = "en"``）
---------------------------------
日本語の書式に依存している判定が3つあるので、``language.current()`` で英語版へ分岐する。
**日本語版の判定は1行も変えていない**（language.py §5.1 と同じ方針）。

1. セリフの括弧 ``「」`` → ``"…"`` / ``“…”``。直引用符は開きと閉じが同じ文字なので
   入れ子の深さを数えられない。英語側は「次の閉じ引用符まで」で1セリフとする
2. 効果音の行。日本語は「カタカナだけの短い行」で見分かるが、英語は地の文と同じ
   アルファベットなので文字種では分からない。脚本の慣習に合わせて
   **全部大文字の短い行**（``CRASH!`` / ``A DOOR SLAMS.``）と、``SFX:`` のラベル付き、
   擬音語だけで組まれた行の3つを拾う（詳細は :func:`_sfx_text_en`）
3. 呼び名の探索。日本語は部分一致で足りるが、英語は語境界を見ないと
   ``Ann`` が ``announced`` に当たる。伝達節（``"…" said Haku.``）の判定も
   「と」に相当するものが無いので、発話動詞の一覧で見る

単体確認（``--lang en`` で英語の判定に切り替える）:
    python -m llm_radio_daemon.generated_drama.parser data/generated_drama_data_<lang>/001/scenes/ch01_sc01.txt \\
        --characters data/generated_drama_data_<lang>/001/characters.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .. import language, sentences
from . import GeneratedDramaChunk, GeneratedDramaCharacter

# セリフは「」。中の『』は入れ子として扱い、外側の「」だけを1セリフとみなす。
_OPEN, _CLOSE = "「", "」"

_SENTENCE_END = "。！？!?"

# 「健太「……」」のような台本記法（呼び名だけの地の文）を落とすための判定。
_NAME_TAG_TAIL = "：: 　"

# セリフの直前・直後で話者名を探す範囲。長く取ると別人の名前を拾いやすくなる。
_LOOKBEHIND = 18
_LOOKAHEAD = 24

# 擬音・効果音だけの行（「ゴゴゴゴゴーッ！」「ドオオオン」「バタン」など）。地の文と混ぜず
# 単独チャンクにし、前後に厚い「間」を取って紙芝居的なメリハリを出す（§4.7.3）。
# カタカナ（全角・半角）と長音・促音・感嘆符・括弧だけで組まれた短い行を効果音とみなす。
_KATAKANA = "ァ-ヿㇰ-ㇿｦ-ﾟ"
_SFX_MARKS = "ーーｰ〜～ッッ！!？?。.、,…‥・「」『』（）() 　\n"
_ONOMATOPOEIA_RE = re.compile(rf"^[{_KATAKANA}{_SFX_MARKS}]+$")


def _is_onomatopoeia(text: str) -> bool:
    """擬音・効果音だけで組まれた短い行か。"""
    s = text.strip()
    core = s.strip("（）() 　「」『』").strip()
    if not (2 <= len(core) <= 24):
        return False
    if not _ONOMATOPOEIA_RE.match(s):
        return False
    kana = sum(
        1 for ch in core if "ァ" <= ch <= "ヿ" or "ｦ" <= ch <= "ﾟ"
    )
    return kana >= 2


# --- 効果音行の判定（英語）--------------------------------------------------

# ラジオ脚本の慣習。``SFX: a door slams`` のラベルは読み上げてはいけないので落とす。
# 丸括弧で囲まれた形（``(SFX: a door slams)``）も来る。
_SFX_LABEL_RE = re.compile(r"^[(（]?\s*(?:SFX|SOUND|FX|SE)\s*[:：.\-—]\s*", re.IGNORECASE)

# 擬音語の見出し。モデルが小文字で ``crash!`` と書いてきたときの拾い網。
# 同じ文字の連打は畳んでから引く（``craaaash`` → ``crash``、``booooom`` → ``bom``）。
_ONOMATOPOEIA_EN = frozenset(
    _squashed
    for _word in """
    bang boom bam blam crash smash slam thud thump bump crack craack snap pop
    clang clank clatter rattle creak screech squeal squeak shriek hiss
    whoosh whump whoomph swish swoosh splash splosh plop drip patter
    sizzle crackle crunch zap buzz beep bleep ding dong clink chink chime
    ring jingle knock rap tap click clack tick tock rumble roar growl howl
    wail thwack whack smack wham kaboom kapow thunk clonk plink hoot honk
    vroom stomp tramp shush whir whirr hum twang boing ka-thunk clip-clop
    """.split()
    for _squashed in (re.sub(r"(.)\1+", r"\1", _word),)
)

_SFX_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


def _sfx_text_en(text: str) -> str | None:
    """効果音の行なら「読み上げるテキスト」を返す。違えば None（＝ふつうの地の文）。

    ``SFX:`` のラベルだけで中身が無い行は空文字を返す（＝効果音だがテキストは無い）。

    日本語版と同じく、迷ったら地の文へ倒す（誤検知して間を厚くするより、
    ふつうに読まれるほうが事故が小さい）。返すテキストは日本語版にならって
    **書かれたまま**にする。``CRASH!`` の大文字は字幕の勢いになるし、Kokoro の
    G2P も大文字の単語を綴り読みしない（頭字語の ``SFX`` だけ綴り読みになるので、
    そのラベルはここで落としてある）。
    """
    s = text.strip()

    m = _SFX_LABEL_RE.match(s)
    if m:
        # ラベルだけで中身が無ければ空文字を返す。呼び出し側がその行を捨てる
        # （``SFX:`` をそのまま渡すと頭字語として「エスエフエックス」と読まれる）。
        return s[m.end() :].strip().rstrip(")）").strip()

    core = s.strip("（）() 　\t").strip()
    if not (2 <= len(core) <= 48):
        return None
    if ":" in core or "：" in core:
        return None  # 話者名タグの取りこぼし（``HAKU: WHAT``）を効果音にしない
    words = _SFX_WORD_RE.findall(core)
    if not words:
        return None

    # 1) 全部大文字の短い行。脚本での効果音・音の指示の書式で、実際に生成させると
    #    「A FEEDBACK ROAR TEARS THE AIR.」のような主語＋動詞の一文になりがちで、
    #    擬音語1つには収まらない（実測: qwen3.8:27b）。全部大文字という書式そのものが
    #    強い合図なので、語数はここだけ緩める。
    letters = [ch for ch in core if ch.isalpha()]
    if len(letters) >= 2 and len(words) <= 10 and all(ch.isupper() for ch in letters):
        return core

    # 2) 擬音語だけで組まれた行（モデルが小文字で書いたとき）。大文字化のような
    #    強い合図が無いので、こちらは短い語数のままにして誤検知を避ける。
    if len(words) <= 5 and all(
        re.sub(r"(.)\1+", r"\1", w.lower()) in _ONOMATOPOEIA_EN for w in words
    ):
        return core

    return None


def _narration_segment(narration: str) -> tuple[str, bool]:
    """地の文セグメントを ``(読み上げるテキスト, 効果音か)`` にする。"""
    if language.current() == "en":
        sfx = _sfx_text_en(narration)
        if sfx is not None:
            return sfx, True
        return narration.strip(), False
    return narration.strip(), _is_onomatopoeia(narration)


def _split_sentences(text: str) -> list[str]:
    if language.current() == "en":
        return sentences.split_en(text)
    out: list[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in _SENTENCE_END:
            out.append(buf)
            buf = ""
    if buf:
        out.append(buf)
    return out


def _split_quotes(paragraph: str) -> list[tuple[str, bool]]:
    """1段落を [(テキスト, セリフか)] の並びへ割る。閉じ括弧が無ければ地の文扱い。"""
    if language.current() == "en":
        return _split_quotes_en(paragraph)
    out: list[tuple[str, bool]] = []
    buf = ""
    depth = 0
    quote = ""
    for ch in paragraph:
        if depth == 0:
            if ch == _OPEN:
                if buf:
                    out.append((buf, False))
                    buf = ""
                quote = ch
                depth = 1
            else:
                buf += ch
            continue
        quote += ch
        if ch == _OPEN:
            depth += 1
        elif ch == _CLOSE:
            depth -= 1
            if depth == 0:
                out.append((quote, True))
                quote = ""
    if quote:  # 閉じられなかった。地の文として読む（放送は止めない）
        buf += quote
    if buf:
        out.append((buf, False))
    return out


# 英語のセリフを囲む引用符。直引用符 ``"`` は開きと閉じが同じ文字なので、
# 日本語版のように入れ子の深さを数えられない（``"He said "no" again."`` を
# 正しく割る方法は無い）。「開いたら次の閉じまで」で1セリフとする。
_QUOTE_CLOSE_EN = {'"': '"', "“": "”", "„": "”", "«": "»"}


def _split_quotes_en(paragraph: str) -> list[tuple[str, bool]]:
    """英語版の :func:`_split_quotes`。閉じ引用符が無ければ地の文扱い（日本語版と同じ）。"""
    out: list[tuple[str, bool]] = []
    buf = ""
    i = 0
    while i < len(paragraph):
        ch = paragraph[i]
        close = _QUOTE_CLOSE_EN.get(ch)
        if close is None:
            buf += ch
            i += 1
            continue
        end = paragraph.find(close, i + 1)
        if end < 0 and ch != '"':
            end = paragraph.find('"', i + 1)  # 開きだけカーリー、閉じは直引用符
        if end < 0:
            buf += paragraph[i:]  # 閉じられなかった。地の文として読む
            break
        if buf:
            out.append((buf, False))
            buf = ""
        out.append((paragraph[i : end + 1], True))
        i = end + 1
    if buf:
        out.append((buf, False))
    return out


def _locate(text: str, name: str, *, nearest_to_end: bool) -> int:
    """``text`` 中の呼び名の位置。無ければ -1。

    英語は語境界を見ないと ``Ann`` が ``announced`` に、``Al`` が ``along`` に当たる。
    大小の揺れ（``HAKU:`` のような書き方）も拾うので大文字小文字は無視する。
    """
    if language.current() != "en":
        return text.rfind(name) if nearest_to_end else text.find(name)
    hits = [
        m.start()
        for m in re.finditer(rf"(?<!\w){re.escape(name)}(?!\w)", text, re.IGNORECASE)
    ]
    if not hits:
        return -1
    return hits[-1] if nearest_to_end else hits[0]


def _mentions(text: str, c: GeneratedDramaCharacter) -> bool:
    return any(_locate(text, n, nearest_to_end=False) >= 0 for n in c.names)


def _find_name(
    text: str, characters: list[GeneratedDramaCharacter], *, nearest_to_end: bool
) -> str | None:
    """text に現れる呼び名のうち、セリフに最も近いキャラの key を返す。

    セリフの手前を見るときは末尾に近い名前、後ろを見るときは先頭に近い名前が
    その発話の主である可能性が高い。
    """
    best_key: str | None = None
    best_pos: int | None = None
    for c in characters:
        for name in c.names:
            pos = _locate(text, name, nearest_to_end=nearest_to_end)
            if pos < 0:
                continue
            if best_pos is None or (pos > best_pos if nearest_to_end else pos < best_pos):
                best_pos, best_key = pos, c.key
    return best_key


# 英語は1文字あたりの情報量が低いので、日本語の 18 / 24 字では
# ``Haku:`` の手前にト書きが1つ挟まると窓から外れる。
_LOOKBEHIND_EN = 48
_LOOKAHEAD_EN = 64

# 伝達節の動詞。日本語の「…」と○○は言った の「と」に当たる目印が英語には無いので、
# 「引用のあと最初の1文にこの動詞があれば、そこに出てくる名前が話者」とみなす。
# ここを緩めて動詞を見ずに名前を拾うと、``"…" Haku turned away.`` のような
# 続きの文の主語を話者にしてしまう（日本語版が「新しい文の主語は見ない」と
# 決めているのと同じ理由）。
_SPEECH_VERBS_EN = frozenset(
    """
    said says say asked asks ask replied replies answered answers added adds
    called calls shouted shouts yelled cried muttered mutters whispered whispers
    murmured breathed snapped snaps hissed laughed chuckled grinned smiled sighed
    groaned growled barked offered ventured insisted admitted agreed observed
    remarked continued repeated began put told tells warned promised
    """.split()
)


def _attribute_en(
    before: str, after: str, speakers: list[GeneratedDramaCharacter]
) -> str | None:
    """英語版の「セリフ直前の話者名」「引用のあとの伝達節」判定。"""
    # 1) ``Haku: "…"`` / ``Haku turned. "…"``
    window = before[-_LOOKBEHIND_EN:]
    if len(before) > _LOOKBEHIND_EN:
        # 窓の先頭で名前を切らないよう、最初の空白まで捨てる。
        window = window.partition(" ")[2] or window
    key = _find_name(window, speakers, nearest_to_end=True)
    if key is not None:
        return key

    # 2) ``"…", said Haku.`` / ``"…" Haku said, turning away.``
    tail = after.lstrip(",，. 　\t")
    if not tail.strip():
        return None
    clause = (sentences.split_en(tail) or [tail])[0][:_LOOKAHEAD_EN]
    if not any(w in _SPEECH_VERBS_EN for w in _SFX_WORD_RE.findall(clause.lower())):
        return None
    return _find_name(clause, speakers, nearest_to_end=False)


def _attribute(
    quote: str,
    before: str,
    after: str,
    paragraph: str,
    characters: list[GeneratedDramaCharacter],
    present: list[GeneratedDramaCharacter],
) -> str | None:
    """セリフの話者を決める。分からなければ None（＝ナレーター読み）。

    0. セリフの中に出てくる名前は「呼びかけられている相手」なので話者から外す
       （``「おじいちゃん」`` を言っているのは、おじいちゃん本人ではない）
    1. セリフ直前の地の文（``健太は言った。「…」`` / ``健太「…」``）
    2. セリフ直後の**同じ文**（``「…」と健太が笑う``）。句点をまたぐと次の文の
       主語を拾ってしまうので、そこで打ち切る
    3. 段落全体にただ1人だけ名前が出ていれば、その人
    4. このシーンに2人しか出ておらず、片方が呼びかけられているなら、話者はもう片方
    """
    addressed = {c.key for c in characters if _mentions(quote, c)}
    speakers = [c for c in characters if c.key not in addressed]

    if language.current() == "en":
        key = _attribute_en(before, after, speakers)
        if key is not None:
            return key
    else:
        key = _find_name(before[-_LOOKBEHIND:], speakers, nearest_to_end=True)
        if key is not None:
            return key
        # 「…」と源太は告げた。 → 引用の「と」で続く節だけが話者を示す。
        # 「…」美穂子は子供の頃…  のように新しい文が始まっている場合、その主語は
        # 話者とは限らないので見ない（ここを緩めると別人のセリフになる）。
        tail = after.lstrip("、，　 ")
        if tail.startswith("と"):
            key = _find_name(tail[:_LOOKAHEAD].split("。")[0], speakers, nearest_to_end=False)
            if key is not None:
                return key

    hits = {c.key for c in speakers if _mentions(paragraph, c)}
    if len(hits) == 1:
        return next(iter(hits))

    if len(present) == 2 and addressed:
        rest = [c.key for c in present if c.key not in addressed]
        if len(rest) == 1:
            return rest[0]
    return None


def _strip_name_tag(narration: str, characters: list[GeneratedDramaCharacter]) -> str:
    """``健太「…」`` の「健太」のような、呼び名だけの地の文を落とす。

    ``健太は言った。`` のような普通の地の文は落とさない（読み上げる）。
    """
    stripped = narration.strip()
    if not stripped:
        return ""
    core = stripped.rstrip(_NAME_TAG_TAIL)
    en = language.current() == "en"
    for c in characters:
        # 英語は ``Haku:`` を ``HAKU:`` と書いてくることがあるので大小を無視する。
        if core in c.names or (en and core.casefold() in {n.casefold() for n in c.names}):
            return ""
    return narration


def _pack(
    segments: list[tuple[str, str | None, bool, bool]], target: int, maxc: int, minc: int
) -> list[tuple[str, str | None, bool]]:
    """地の文の連続セグメントを、目標文字数に収まる単位へ束ね直す。

    - 話者が変わるところでは必ず切る（1チャンク＝1話者）
    - セリフは1つで1チャンク。地の文とはくっつけない（話者が同じ＝ナレーター読みに
      なったセリフでも、地の文に埋めると掛け合いの間が消える）
    - 効果音の行もセリフと同じく単独チャンク（前後の地の文と混ぜない）
    """
    # 1) 長すぎるセグメントを文単位で割る。
    split: list[tuple[str, str | None, bool, bool]] = []
    for text, key, is_quote, is_sfx in segments:
        if len(text) <= maxc:
            split.append((text, key, is_quote, is_sfx))
            continue
        buf = ""
        for sent in _split_sentences(text):
            if buf and len(buf) + len(sent) > target:
                split.append((buf, key, is_quote, is_sfx))
                buf = sent
            else:
                buf += sent
        if buf:
            split.append((buf, key, is_quote, is_sfx))

    # 2) 同じ話者の短い地の文を、上限まで結合する。
    packed: list[tuple[str, str | None, bool, bool]] = []
    for text, key, is_quote, is_sfx in split:
        prev_atomic = packed and (packed[-1][2] or packed[-1][3])
        if (
            packed
            and not is_quote
            and not is_sfx
            and not prev_atomic
            and packed[-1][1] == key
        ):
            prev = packed[-1][0]
            # 英語は語の区切りが空白なので、結合するときに1つ挟む。日本語は
            # 元から空白なしでつながる書式なので、ここを変えると ja の出力が
            # 変わってしまう（区切りを持たない言語と持つ言語の違い）。
            sep = " " if language.current() == "en" else ""
            joined = prev + sep + text
            if len(joined) <= maxc and (
                len(prev) < minc or len(joined) <= target
            ):
                packed[-1] = (joined, key, False, False)
                continue
        packed.append((text, key, is_quote, is_sfx))
    return [(t, k, sfx) for t, k, _, sfx in packed]


def split_scene(
    body: str,
    characters: list[GeneratedDramaCharacter],
    *,
    dialogue_by_character: bool = True,
    chunk_target_chars: int = 400,
    chunk_max_chars: int = 500,
    chunk_min_chars: int = 120,
) -> list[GeneratedDramaChunk]:
    """シーン本文を朗読チャンクへ割る。``character_key`` が None なら地の文。"""
    text = body.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [p.strip() for p in text.split("\n")]
    # このシーンに実際に出ている人物（話者の絞り込みに使う）。
    present = [c for c in characters if _mentions(text, c)]

    segments: list[tuple[str, str | None, bool, bool]] = []
    for para in paragraphs:
        if not para:
            continue
        pieces = _split_quotes(para)
        for i, (piece, is_quote) in enumerate(pieces):
            if not is_quote:
                narration = _strip_name_tag(piece, characters) if characters else piece
                # 「…」。細い声が… の先頭の句読点はセリフ側のものなので落とす。
                narration = narration.strip().lstrip("。、，．,")
                if narration.strip():
                    seg_text, is_sfx = _narration_segment(narration)
                    if seg_text:
                        segments.append((seg_text, None, False, is_sfx))
                continue
            key = None
            if dialogue_by_character and characters:
                before = "".join(p for p, q in pieces[:i] if not q)
                after = "".join(p for p, q in pieces[i + 1 :] if not q)
                key = _attribute(piece, before, after, para, characters, present)
            segments.append((piece.strip(), key, True, False))

    packed = _pack(segments, chunk_target_chars, chunk_max_chars, chunk_min_chars)
    return [
        GeneratedDramaChunk(index=i, text=t, character_key=k, is_sfx=sfx)
        for i, (t, k, sfx) in enumerate(packed)
        if t.strip()
    ]


def _main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="ラジオドラマシーンのパーサー単体確認")
    ap.add_argument("scene", help="本文テキスト（data/generated_drama_data_<lang>/NNN/scenes/*.txt）")
    ap.add_argument("--characters", default="", help="characters.json のパス")
    ap.add_argument("--target", type=int, default=400)
    ap.add_argument("--lang", default="ja", help="判定を切り替える言語コード（ja / en）")
    args = ap.parse_args(argv)

    language.set_language(args.lang)

    characters: list[GeneratedDramaCharacter] = []
    if args.characters:
        raw = json.loads(Path(args.characters).read_text(encoding="utf-8"))
        entries = raw.get("characters", raw) if isinstance(raw, dict) else raw
        characters = [GeneratedDramaCharacter.from_dict(e) for e in entries]

    body = Path(args.scene).read_text(encoding="utf-8")
    chunks = split_scene(body, characters, chunk_target_chars=args.target)
    print(f"# {len(chunks)} chunks / characters {len(characters)} / lang={args.lang}\n")
    for c in chunks:
        who = "（効果音）" if c.is_sfx else (c.character_key or "（地の文）")
        print(f"--- {c.index:3d} [{who}] {len(c.text)} chars")
        print(f"    {c.text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
