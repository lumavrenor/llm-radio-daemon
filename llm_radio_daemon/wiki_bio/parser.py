"""Wikipedia記事本文（explaintext, exsectionformat=wiki）のパーサ。

朗読ではなく「これを素材にMCが解説トークを作る」ための前処理。脚注・出典など
ノイズになる末尾セクションを切り落とし、見出しで節を追いながら段落を目標文字数へ
束ねて :class:`SourceChunk` 列にする（aozora/parser.py のチャンク分割ロジックを流用。
ルビ・底本除去は無関係なので持たない）。

単体確認:
    python -m llm_radio_daemon.wiki_bio.parser data/wiki_bio_<lang>/edison.txt
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from . import SourceChunk

# 見出し行 "== 節名 ==" / "=== 小節名 ===" 等（前後の "=" の数は一致する）。
_HEADING = re.compile(r"^(=+)\s*(.+?)\s*\1$")

# この見出しに達したら、そこから先（本文でない末尾）は丸ごと捨てる。
_TRAILING_SECTION_TITLES = {
    "脚注", "出典", "参考文献", "関連項目", "外部リンク", "注釈",
    "注釈・出典", "参照", "書誌情報", "参照リンク",
}

_SENTENCE_END = "。！？"


def _split_sentences(text: str) -> list[str]:
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


def _pack_paragraph(text: str, target: int, maxc: int) -> list[str]:
    """1段落を、目標文字数に収まる原文セグメントのリストへ分割する。"""
    if len(text) <= maxc:
        return [text]
    segments: list[str] = []
    buf = ""
    for sent in _split_sentences(text):
        if buf and len(buf) + len(sent) > target:
            segments.append(buf)
            buf = sent
        else:
            buf += sent
    if buf:
        segments.append(buf)
    return segments


def _iter_paragraphs(raw: str) -> list[tuple[str, str, bool]]:
    """行単位で見出しを検出し、(段落テキスト, 節名, その節で最初の段落か) の列にする。

    MediaWiki の explaintext は見出し行の直後に空行を入れないことがある
    （"=== 出生 ===\\n1847年..." のように本文と1行しか離れない）。段落を
    「空行区切り」でまとめる前に見出しを検出しないと、見出し記法が本文へ
    そのまま混入してしまうため、行単位の状態機械で処理する。
    """
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    current_section = ""
    pending_head = True  # 冒頭リード文もセクションの先頭として扱う
    para_lines: list[str] = []
    out: list[tuple[str, str, bool]] = []

    def flush() -> None:
        nonlocal para_lines, pending_head
        joined = "".join(ln.strip() for ln in para_lines)
        para_lines = []
        if not joined:
            return
        out.append((joined, current_section, pending_head))
        pending_head = False

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped == "":
            flush()
            continue
        m = _HEADING.match(stripped)
        if m:
            flush()
            title = m.group(2).strip()
            if title in _TRAILING_SECTION_TITLES:
                break  # 本文でない末尾セクションに入った。ここで打ち切る
            current_section = title
            pending_head = True
            continue
        para_lines.append(stripped)
    flush()
    return out


def parse_article_text(
    raw: str,
    *,
    chunk_target_chars: int = 700,
    chunk_max_chars: int = 1000,
    chunk_min_chars: int = 200,
) -> list[SourceChunk]:
    # (原文セグメント, 節名, そのセクションで最初のチャンクか)
    pending: list[tuple[str, str, bool]] = []
    for para, section, is_para_head in _iter_paragraphs(raw):
        is_head = is_para_head
        for seg in _pack_paragraph(para, chunk_target_chars, chunk_max_chars):
            pending.append((seg, section, is_head))
            is_head = False  # 見出し直後フラグは段落の先頭セグメントだけ

    # 短い段落は前のチャンクへ結合する。Wikipediaの節は1段落だけの短い小節が
    # 連続することが多く、節境界で結合を止めると数十字のチャンクが量産されて
    # LLM呼び出しが細切れになりすぎるため、節をまたいだ結合を許す
    # （節名は結合後＝より進んだ方を採用。見出し直後フラグは持ち越す）。
    merged: list[tuple[str, str, bool]] = []
    for seg, sec, is_head in pending:
        if (
            merged
            and len(seg) < chunk_min_chars
            and len(merged[-1][0]) + len(seg) <= chunk_max_chars
        ):
            merged[-1] = (merged[-1][0] + seg, sec, merged[-1][2] or is_head)
        else:
            merged.append((seg, sec, is_head))

    return [
        SourceChunk(index=i, text=seg, section=sec, is_section_head=is_head)
        for i, (seg, sec, is_head) in enumerate(merged)
    ]


def parse_article_file(path: str | Path, **kwargs) -> list[SourceChunk]:
    p = Path(path)
    raw = p.read_text(encoding="utf-8", errors="replace")
    return parse_article_text(raw, **kwargs)


def _main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    chunks = parse_article_file(argv[0])
    print(f"# {len(chunks)} chunks\n")
    for c in chunks:
        head = f"  [節頭: {c.section}]" if c.is_section_head else ""
        print(f"--- chunk {c.index} (section: {c.section or '(opening)'}){head}")
        print(f"  {c.text}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
