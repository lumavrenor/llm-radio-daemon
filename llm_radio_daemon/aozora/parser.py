"""青空文庫テキストのパーサ（v4 §10.2）。

青空文庫の独自注記記法を除去し、TTS にそのまま流せる ``speech_text`` と
画面字幕用の ``display_text`` に分けた :class:`Chunk` のリストを出力する。

**朗読テキストはここから TTS へ直行する。LLM を通さない**（§10.0）。

単体確認（テキストは data/aozora/ 配下。作品IDだけでも可）:
    python -m llm_radio_daemon.aozora.parser data/aozora/058050.txt
    python -m llm_radio_daemon.aozora.parser 058050
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from . import Chunk

# --- 記法パターン -----------------------------------------------------------

# 凡例ブロックの罫線（半角ハイフンが多数連続する行）。
_RULE_LINE = re.compile(r"^-{8,}\s*$")
# 底本情報。この行以降はすべて本文から除外する。
_COLOPHON_HEAD = re.compile(r"^(底本：|底本:|底本について)")

# 外字注記 ※［＃…、1-2-3］。対応字が拾えないので丸ごと除去する。
_GAIJI = re.compile(r"※［＃[^］]*］")
# 見出し注記（章題の目印）。除去はするが、段落の is_chapter_head 判定に使う。
_HEADING_ANNOT = re.compile(r"［＃[^］]*見出し[^］]*］")
# 入力者注・組版指定など ［＃…］ 全般。
_ANNOT = re.compile(r"［＃[^］]*］")

# ルビ。base《よみ》。base は ｜ で明示されるか、直前の連続した漢字・カタカナ等。
_RUBY_WITH_BAR = re.compile(r"｜([^｜《》]+)《([^》]+)》")
_RUBY_BARE = re.compile(
    r"([0-9A-Za-z々〆〇一-鿿豈-﫿"
    r"゠-ヿㇰ-ㇿｦ-ﾝ]+)《([^》]+)》"
)
# 取りこぼした《…》と孤立した ｜。
_RUBY_ORPHAN = re.compile(r"《[^》]*》")

_SENTENCE_END = "。！？"


# --- ヘッダ・底本の除去 ----------------------------------------------------

def _strip_frame(raw: str) -> tuple[str, int]:
    """冒頭ヘッダ（タイトル・著者・凡例）と底本情報を落として本文だけ返す。

    返り値は (本文, 元原文における本文開始オフセット)。
    """
    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    # 底本以降を切り捨てる。
    end = len(lines)
    for i, line in enumerate(lines):
        if _COLOPHON_HEAD.match(line.strip()):
            end = i
            break
    lines = lines[:end]

    # 凡例ブロック（罫線で挟まれた部分）があれば 2 本目の罫線まで捨てる。
    rule_idx = [i for i, ln in enumerate(lines[:40]) if _RULE_LINE.match(ln)]
    if len(rule_idx) >= 2:
        start = rule_idx[1] + 1
    else:
        # 罫線が無い場合はタイトル・著者行を最初の空行まで捨てる。
        start = 0
        for i, line in enumerate(lines[:12]):
            if line.strip() == "":
                start = i + 1
                break

    # 本文先頭の空行を詰める。
    while start < len(lines) and lines[start].strip() == "":
        start += 1

    offset = sum(len(ln) + 1 for ln in lines[:start])
    return "\n".join(lines[start:]), offset


# --- 注記・ルビの処理 ----------------------------------------------------

def _strip_annotations(text: str) -> str:
    text = _GAIJI.sub("", text)
    text = _ANNOT.sub("", text)
    return text


def _ruby_display(text: str) -> str:
    """字幕用。ルビ注記を外して漢字（base）だけ残す（§10.11）。"""
    text = _RUBY_WITH_BAR.sub(lambda m: m.group(1), text)
    text = _RUBY_BARE.sub(lambda m: m.group(1), text)
    text = _RUBY_ORPHAN.sub("", text)
    return text.replace("｜", "").strip()


def _ruby_speech(text: str, ruby_mode: str) -> str:
    """TTS 用。ruby_mode="kana" なら base を読み仮名へ置換する（§10.4 簡易版）。"""
    if ruby_mode == "kana":
        text = _RUBY_WITH_BAR.sub(lambda m: m.group(2), text)
        text = _RUBY_BARE.sub(lambda m: m.group(2), text)
    else:
        text = _RUBY_WITH_BAR.sub(lambda m: m.group(1), text)
        text = _RUBY_BARE.sub(lambda m: m.group(1), text)
    text = _RUBY_ORPHAN.sub("", text)
    return text.replace("｜", "").strip()


# --- チャンク分割 --------------------------------------------------------

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


def parse_work_text(
    raw: str,
    *,
    ruby_mode: str = "kana",
    chunk_target_chars: int = 400,
    chunk_max_chars: int = 500,
    chunk_min_chars: int = 150,
) -> list[Chunk]:
    body, base_offset = _strip_frame(raw)

    # 段落（空行区切り）へ。各段落について「章題か」「原文セグメント列」を持つ。
    raw_paragraphs = re.split(r"\n\s*\n", body)

    pending: list[tuple[str, bool]] = []  # (原文セグメント, is_chapter_head)
    for para in raw_paragraphs:
        para_1line = "".join(ln.strip("　 \t") for ln in para.split("\n"))
        is_head = bool(_HEADING_ANNOT.search(para_1line))
        cleaned = _strip_annotations(para_1line).strip()
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
            and not seg.startswith("「")
            and len(seg) < chunk_min_chars
            and len(merged[-1][0]) + len(seg) <= chunk_max_chars
        ):
            merged[-1] = (merged[-1][0] + seg, False)
        else:
            merged.append((seg, is_head))

    chunks: list[Chunk] = []
    running = base_offset
    for i, (seg, is_head) in enumerate(merged):
        chunks.append(
            Chunk(
                index=i,
                display_text=_ruby_display(seg),
                speech_text=_ruby_speech(seg, ruby_mode),
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
    """パスをそのまま、または作品ID/ファイル名として data/aozora/ から解決する。"""
    p = Path(arg)
    if p.exists():
        return p
    for cand in (
        Path("data/aozora") / arg,
        Path("data/aozora") / f"{arg}.txt",
        Path("data/aozora") / Path(arg).name,
    ):
        if cand.exists():
            return cand
    raise SystemExit(
        f"見つかりません: {arg}\n"
        f"（テキストは data/aozora/ 配下です。例: python -m llm_radio_daemon.aozora.parser data/aozora/058050.txt）"
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
