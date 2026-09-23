"""Project Gutenberg テキストのパーサ（``aozora/parser.py`` の英語版）。

PG のヘッダ・フッタ・前付け（目次など）を落とし、TTS にそのまま流せる
``speech_text`` と画面字幕用の ``display_text`` に分けた :class:`Chunk` を並べる。

**朗読テキストはここから TTS へ直行する。LLM を通らない**（§10.0）。
``sanitize_text`` も通らないので、読み上げられない書式はここで全部落とすこと。

実測で分かった、落とさないといけないもの（Kokoro の G2P で確認済み）
------------------------------------------------------------------
- ``--``（古い版の em dash 代用。Pride and Prejudice に 498 個）。そのまま渡すと
  **前後の語が続けて読まれる**（``truth--universally`` → ``tɹˈuθjˌunəvˈɜɹsəli``）。
  本物の em dash ``—`` なら正しく切れるので、そこへ寄せる
- ``[Illustration: ...]`` ``[Footnote 1: ...]`` のような編集者の角括弧注記。
  そのまま渡すと「イラストレーション、ア シップ」と読み上げてしまう
- ``_italics_`` のアンダースコア。G2P は黙って落とすので読み上げは無事だが、
  字幕に ``_very_`` と出るので両方から落とす

ルビ処理（``aozora/parser.py:92-109``）に当たるものは英語には無い。ただし
``speech_text`` / ``display_text`` の二本立ては残してある（``--`` → ``—`` の
置換のように、読ませ方と見せ方が分かれる余地がまだある）。

単体確認（テキストは data/gutenberg/ 配下。作品IDだけでも可）:
    python -m llm_radio_daemon.gutenberg.parser data/gutenberg/1342.txt
    python -m llm_radio_daemon.gutenberg.parser 1342
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from .. import sentences
from . import Chunk

# --- PG の目印 --------------------------------------------------------------

# 本文の始まりと終わり。実測ではどの作品も同じ形で、独立した1行に来る。
# 古い版は "THIS PROJECT GUTENBERG EBOOK"、さらに古いものは "SMALL PRINT"。
_START_MARKER = re.compile(
    r"^\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*\s*$", re.MULTILINE
)
_END_MARKER = re.compile(
    r"^\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*\s*$", re.MULTILINE
)
# START マーカーが無い古い形式のための保険。
_OLD_START_MARKER = re.compile(r"^\*+\s*START OF THE PROJECT GUTENBERG.*$", re.MULTILINE)
# 旧い版は ``*** END OF ... ***`` の**手前**に、さらに古い形式の終端行
# （``End of Project Gutenberg's <題名>, by <著者>``）を残していることがある
# （実測: The Vicar's Daughter は本文の直後にこの行があり、新形式のマーカーは
# さらに数行あと）。本文の終わりは**どちらか早く出てくる方**で切る。
_OLD_END_MARKER = re.compile(
    r"^\s*End of (?:the )?Project Gutenberg'?s?\b.*$", re.IGNORECASE | re.MULTILINE
)

# 編集者の角括弧注記。読み上げないので本文から落とす。
# ``[Illustration:`` … ``]`` は**段落をまたぐ**ことがあるので（実測: Pride and
# Prejudice の扉絵で 68 か所）、段落へ割る前に本文まるごとへ当てる。
# 長さに上限を置いて、閉じ忘れの ``[`` が本文を食い進まないようにしてある。
_BRACKET_NOTE = re.compile(r"\[[^\[\]]{0,600}\]", re.DOTALL)
# 上で消し切れなかった孤立した角括弧。TTS が「かっこ」と読まないよう落とす。
_STRAY_BRACKET = re.compile(r"[\[\]]")

# 章見出しの行。段落の is_chapter_head 判定と、前付けの切り上げに使う。
# ACT / SCENE はキーワード単独だと弱い（"Scene in Szechuen" のような挿絵の
# キャプションや、ふつうの英単語としての "act"/"scene" にも当たる。実測:
# An Australian in China の挿絵一覧「SCENE IN SZECHUEN　58」が見出し扱いされ、
# 目次を抜けたと誤判定した）。数字（幕番号・場番号）が直後に続くときだけ見出しとみなす。
# LETTER は書簡体小説で「Letter to <人名>」（数字なし）が正規の見出し形なので
# 数字必須にはしない（Frankenstein の「Letter 1」のような数字つきも両方拾える）。
_CHAPTER_HEAD = re.compile(
    r"^\s*(?:CHAPTER|CHAP\.|BOOK|PART|VOLUME|LETTER|CANTO|STAVE|EPILOGUE"
    r"|PROLOGUE|INTRODUCTION)\b"
    r"|^\s*(?:ACT|SCENE)\s+(?:[IVXLCDM]+|\d+)\b"
    r"|^\s*[IVXLCDM]{1,7}\s*\.?\s*$",
    re.IGNORECASE,
)

# 目次の見出し行と、目次の項目に見える行。
_TOC_HEAD = re.compile(
    r"^\s*(?:CONTENTS?|TABLE OF CONTENTS|ILLUSTRATIONS?|LIST OF ILLUSTRATIONS)"
    r"\s*[.:]?\s*$",
    re.IGNORECASE,
)
# 底本を作った人の注記。PG では定番の前付けで、散文の段落が数本続く
# （実測: Moby Dick の "Original Transcriber's Notes:"）。段落の先頭でなく、
# 段落の頭のほう（罫線・box文字の飾りの後ろ）に出ることもあるので
# match ではなく search で見る（実測: An Australian in China は
# ``+---+\n| Transcriber's Note: |\n...`` という ASCII アートの箱の中）。
# 「transcriber's note」という言い回し自体が地の文にまず出ないので、
# 先頭付近に限らず拾ってよい（誤検知よりましな取りこぼしは覚悟する）。
_NOTE_HEAD = re.compile(
    r"(?:original\s+)?transcriber['’]?s?\s+note", re.IGNORECASE
)
# 旧い PG の自動挿入文。「HTML 版もある」という定型の案内で、URLを含むため
# 長く（実測 280 字前後）、_PROSE_CHARS を超えて「本文が始まった」と誤判定
# されやすい。実測で 20 冊中 3 冊にあった、かなり頻出の定型文。
_HTML_NOTE_HEAD = re.compile(
    r"^\s*Note:\s*Project Gutenberg (?:also\s+)?has an?\s+HTML version",
    re.IGNORECASE,
)
_TOC_ENTRY = re.compile(
    r"^\s*(?:CHAPTER|CHAP\.|BOOK|PART|LETTER|CANTO|STAVE)\b"
    r"|^\s*(?:ACT|SCENE)\s+(?:[IVXLCDM]+|\d+)\b"
    r"|^\s*[IVXLCDM]{1,7}\s*[.:]\s"
    r"|^\s*\d{1,3}\s*[.:]\s",
    re.IGNORECASE,
)

# 前付け（表題・著者・版元・目次）を探す範囲。目次は長い（Moby Dick は135項目）ので
# 段落数には余裕を持たせる。ここで決まらなければ何も落とさない。
_FRONT_MATTER_PARAS = 400
# 「これは前付けではなく本文だ」と判断する段落の長さ。
_PROSE_CHARS = 200


# --- ヘッダ・フッタ・前付けの除去 -------------------------------------------

def _strip_frame(raw: str) -> tuple[str, int]:
    """PG のヘッダ・ライセンスフッタを落として本文だけ返す。

    返り値は (本文, 元原文における本文開始オフセット)。``aozora`` 版と同じ契約。
    """
    text = raw.replace("\r\n", "\n").replace("\r", "\n")

    start = 0
    m = _START_MARKER.search(text) or _OLD_START_MARKER.search(text)
    if m is not None:
        start = m.end()
    else:
        # マーカーが見つからない（抜粋やミラーの加工版）。冒頭のライセンス文だけ
        # 落とせるところまで落とす。見つからなければ丸ごと本文として読む。
        head = text[:4000].lower()
        idx = head.find("project gutenberg")
        if idx >= 0:
            nl = text.find("\n\n", idx)
            start = nl + 2 if nl > 0 else 0

    end = len(text)
    for pattern in (_END_MARKER, _OLD_END_MARKER):
        m_end = pattern.search(text, start)
        if m_end is not None:
            end = min(end, m_end.start())

    return text[start:end], start


# 章見出しの段落として認める行数と長さ。``CHAPTER I.`` と副題が別の行に来る形が
# あるので1行に限れない（実測: Alice は ``CHAPTER I.`` + ``Down the Rabbit-Hole``）。
_HEADING_MAX_LINES = 3
_HEADING_MAX_CHARS = 120
# 目次まるごとの1段落と認める最小の項目数。ここを 2 にすると、上の
# 「見出し＋副題」を目次と読み違える。
_TOC_BLOCK_MIN_ENTRIES = 3


def _is_heading_line(para: str) -> bool:
    """章見出し（または目次の1項目）に見える短い段落か。

    **目次の項目と本文の章見出しは、書式がまったく同じ**（どちらも
    ``CHAPTER 1. Loomings.``）。ここでは見分けず、どちらなのかは
    :func:`_drop_front_matter` が出現回数で決める。
    """
    lines = [ln for ln in para.split("\n") if ln.strip()]
    if not lines or len(lines) > _HEADING_MAX_LINES:
        return False
    # 桁揃えの目次は各行の内側にページ番号ぶんの空白を大量に持つ
    # （実測: ``CHAPTER I.                    PAGES``）。単純に " ".join すると
    # その空白がそのまま残り、見た目の短い見出しが _HEADING_MAX_CHARS を超えて
    # 見出し扱いされない。読み上げる形（空白を1つに畳んだ形）で長さを見る。
    joined = _norm_heading(para)
    if len(joined) >= _HEADING_MAX_CHARS:
        return False
    head = lines[0].strip()
    return bool(_CHAPTER_HEAD.match(head) or _TOC_ENTRY.match(head))


def _is_toc_block(para: str) -> bool:
    """目次がまるごと1段落になっているか（実測: Sherlock Holmes / Alice）。

    項目が3つ以上あり、半分以上の行が目次の項目に見えるもの。本文の章見出しは
    たかだか副題つきの数行なので、この形は必ず目次であり、出現回数を見る必要がない。
    """
    lines = [ln for ln in para.split("\n") if ln.strip()]
    if len(lines) < _TOC_BLOCK_MIN_ENTRIES:
        return False
    hits = sum(1 for ln in lines if _TOC_ENTRY.match(ln))
    return hits >= _TOC_BLOCK_MIN_ENTRIES and hits * 2 >= len(lines)


def _norm_heading(para: str) -> str:
    return " ".join(para.split()).casefold()


# 見出しの「章番号だけ」を取り出す鍵。目次と本文とで、同じ章の見出しが
# **違う書式**で来ることがある（実測: The Vicar's Daughter は目次で
# ``CHAPTER I. INTRODUCTORY`` と1行にまとめ、本文では ``CHAPTER I.`` と
# ``INTRODUCTORY.`` を別の段落に分けて書く）。全文一致だけで見分けると、
# 本文側の見出しが「1回しか出てこない」ように見えてしまい、まだ目次の
# 途中なのに本文が始まったと誤判定する。番号だけを鍵にして、副題の書式差を吸収する。
_NUMBER_KEY_RE = re.compile(
    r"^\s*(chapter|chap\.|book|part|volume|letter|act|scene|canto|stave)"
    r"\s+([ivxlcdm]+|\d+)\b",
    re.IGNORECASE,
)


def _heading_number_key(para: str) -> str | None:
    m = _NUMBER_KEY_RE.match(para.strip())
    if not m:
        return None
    return f"{m.group(1).rstrip('.').lower()} {m.group(2).lower()}"


def _drop_front_matter(paragraphs: list[str]) -> list[str]:
    """表題・著者・版元・目次を落として、本文が始まる段落から返す。

    START マーカーの**後ろ**にも前付けが続く（実測: 表題／著者／版表示／
    ``CONTENTS`` のかたまり／底本注記／``[Illustration]``）。本文の頭から順に読む
    コーナーなので、ここを残すと最初の数分が目次の読み上げになる。

    目次の項目と本文の章見出しを分ける見方
    ------------------------------------
    どちらも書式が同じなので（``CHAPTER 1. Loomings.``）、**目次の項目は本文に
    もう一度出てくる**ことで見分ける。2回以上出てくる見出しの1回目は目次、
    1回しか出てこない見出しは本文の始まり。全文一致（``_norm_heading``）に加えて
    章番号だけの一致（``_heading_number_key``）も見る。副題の書式が目次と本文で
    ずれていても、章番号は動かないため。

    段落数で打ち切る方式は捨てた。Moby Dick は目次が135項目あり、上限40段落だと
    第34章の途中から朗読が始まってしまった。
    """
    # 文書全体での見出しの出現回数。2回以上なら目次にも載っている。
    seen: dict[str, int] = {}
    for para in paragraphs:
        s = para.strip()
        if s and _is_heading_line(s):
            seen[_norm_heading(s)] = seen.get(_norm_heading(s), 0) + 1
            num_key = _heading_number_key(s)
            if num_key is not None:
                seen[num_key] = seen.get(num_key, 0) + 1

    def _still_toc(s: str) -> bool:
        if seen.get(_norm_heading(s), 0) >= 2:
            return True
        num_key = _heading_number_key(s)
        return num_key is not None and seen.get(num_key, 0) >= 2

    skip_prose = False  # 底本注記の本文を読み飛ばしている最中
    for i, para in enumerate(paragraphs[:_FRONT_MATTER_PARAS]):
        stripped = para.strip()
        if not stripped:
            continue
        if _TOC_HEAD.match(stripped) or _is_toc_block(stripped):
            continue  # 「CONTENTS」の行そのものと、目次まるごとの段落
        if _NOTE_HEAD.search(stripped) or _HTML_NOTE_HEAD.match(stripped):
            skip_prose = True  # 次の見出しまで、散文は底本注記とみなす
            continue
        if _is_heading_line(stripped):
            skip_prose = False
            if _still_toc(stripped):
                continue  # 目次にも載っている＝いまは目次の側
            return paragraphs[i:]  # 1回しか出てこない見出し＝本文の始まり
        if skip_prose:
            continue
        # 実測の長さは折り返しの空白・桁揃えの空白を含んだ生の文字数で見ると、
        # 桁揃えされた目次の1行（ページ番号の位置を揃える詰め物入り）が
        # _PROSE_CHARS を超えて「本文が始まった」と誤判定することがある
        # （実測: An Australian in China の目次はページ番号列で右詰めされており、
        # 長い章題の1行が生の文字数で200字を超えていた）。実際に読み上げる形
        # （空白を1つに畳んだ長さ）で比べる。
        if len(_norm_heading(stripped)) >= _PROSE_CHARS:
            return paragraphs[i:]
        # 表題・著者・版元など。短い段落は前付けとして読み飛ばす。
    # 前付けの区間で決まらなかった。**何も落とさない**（勘で削るより、
    # 表題を数行読んでしまうほうが安い）。
    return paragraphs


# --- 記法の処理 ------------------------------------------------------------

def _clean(text: str) -> str:
    """PG の記法を落とす。

    ``--`` は本物の em dash へ寄せる。G2P は ``--`` を区切りとして扱わず、前後の語を
    つなげて読んでしまう（実測: ``truth--universally`` → ``tɹˈuθjˌunəvˈɜɹsəli``）。
    ``—`` なら正しく切れる。

    いまは ``speech_text`` と ``display_text`` で落とすものが同じなので、両方これ1本で
    作る（青空文庫版がルビで二本に分かれるのに当たるものが英語には無い）。
    ``Chunk`` の二本立て自体は、読ませ方と見せ方が分かれる余地を残すために保つ。
    """
    text = _BRACKET_NOTE.sub(" ", text)
    text = _STRAY_BRACKET.sub(" ", text)
    text = text.replace("_", "")
    text = re.sub(r"-{2,}", "—", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


# --- チャンク分割 ----------------------------------------------------------

def _pack_paragraph(text: str, target: int, maxc: int) -> list[str]:
    """1段落を、目標文字数に収まる原文セグメントのリストへ分割する。"""
    if len(text) <= maxc:
        return [text]
    segments: list[str] = []
    buf = ""
    for sent in sentences.split_en(text):
        if buf and len(buf) + len(sent) > target:
            segments.append(buf)
            buf = sent
        else:
            buf += sent
    if buf:
        segments.append(buf)
    return segments


def parse_work_text(
    raw: str,
    *,
    chunk_target_chars: int = 700,
    chunk_max_chars: int = 900,
    chunk_min_chars: int = 200,
) -> list[Chunk]:
    body, base_offset = _strip_frame(raw)
    # 段落をまたぐ ``[Illustration: … ]`` を先に落とす（段落へ割ってからでは
    # 開きと閉じが別の段落に入って一致しない）。
    body = _BRACKET_NOTE.sub(" ", body)

    raw_paragraphs = re.split(r"\n\s*\n", body)
    raw_paragraphs = _drop_front_matter(raw_paragraphs)

    pending: list[tuple[str, bool]] = []  # (原文セグメント, is_chapter_head)
    for para in raw_paragraphs:
        # PG は 70 字前後で折り返してあるだけなので、段落内の改行は**空白**で
        # つなぐ（日本語版が空文字でつなぐのと逆。詰めると語がくっつく）。
        para_1line = " ".join(ln.strip() for ln in para.split("\n") if ln.strip())
        if not para_1line:
            continue
        is_head = bool(_CHAPTER_HEAD.match(para_1line)) and len(para_1line) < 120
        cleaned = _clean(para_1line)
        if not cleaned:
            continue

        for seg in _pack_paragraph(cleaned, chunk_target_chars, chunk_max_chars):
            pending.append((seg, is_head))
            is_head = False  # 章題フラグは段落の先頭セグメントだけ

    # 短い段落を前のチャンクへ結合する（会話行・章題は結合しない）。§10.2-3
    merged: list[tuple[str, bool]] = []
    for seg, is_head in pending:
        if (
            merged
            and not is_head
            and not merged[-1][1]
            and not seg.lstrip().startswith(('"', "“"))
            and len(seg) < chunk_min_chars
            and len(merged[-1][0]) + len(seg) <= chunk_max_chars
        ):
            merged[-1] = (merged[-1][0] + " " + seg, False)
        else:
            merged.append((seg, is_head))

    chunks: list[Chunk] = []
    running = base_offset
    for seg, is_head in merged:
        if seg.strip():
            chunks.append(
                Chunk(
                    index=len(chunks),
                    display_text=seg,
                    speech_text=seg,
                    char_offset=running,
                    is_chapter_head=is_head,
                )
            )
        running += len(seg)
    return chunks


def parse_work_file(path: str | Path, **kwargs) -> list[Chunk]:
    p = Path(path)
    raw = p.read_text(encoding="utf-8", errors="replace")
    return parse_work_text(raw, **kwargs)


def _resolve_input(arg: str) -> Path:
    """パスをそのまま、または作品ID/ファイル名として data/gutenberg/ から解決する。"""
    p = Path(arg)
    if p.exists():
        return p
    for cand in (
        Path("data/gutenberg") / arg,
        Path("data/gutenberg") / f"{arg}.txt",
        Path("data/gutenberg") / Path(arg).name,
    ):
        if cand.exists():
            return cand
    raise SystemExit(
        f"見つかりません: {arg}\n"
        f"（テキストは data/gutenberg/ 配下です。"
        f"例: python -m llm_radio_daemon.gutenberg.parser data/gutenberg/1342.txt）"
    )


def _main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    chunks = parse_work_file(_resolve_input(argv[0]))
    print(f"# {len(chunks)} chunks\n")
    for c in chunks:
        head = "  [章題]" if c.is_chapter_head else ""
        print(f"--- chunk {c.index} (offset {c.char_offset}){head}")
        print(f"  display: {c.display_text}")
        print(f"  speech : {c.speech_text}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
