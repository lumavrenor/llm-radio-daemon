"""Project Gutenberg の取得プリミティブ（``aozora/fetch.py`` の英語版）。

``scripts/tools/fetch_gutenberg.py``（バルク取得）と ``gutenberg.corpus``（本体の実行時
lazy fetch）の両方がここを使う。**取得ロジックを二重に持たないための共有層。**

ネットワークへ出るのはこのモジュールだけ。

取得するもの
-----------
- カタログ: ``pg_catalog.csv.gz``（公式の機械可読カタログ。gz 5.5MB / 展開 21MB、
  約 79,000 作品。うち ``Language = en`` かつ ``Type = Text`` が約 61,000）。
  列は ``Text#, Type, Issued, Title, Language, Authors, Subjects, LoCC, Bookshelves``
- 本文: ``https://www.gutenberg.org/cache/epub/<id>/pg<id>.txt``（UTF-8）。
  カタログに本文 URL の列は無いので作品IDから組み立てる

**ミラーへ向けられるようにしてある**（``catalog_url`` / ``text_url_template``）。
gutenberg.org は自動取得をまとめてやられるのを嫌っており、本家もミラーの利用を
案内している。この番組の取得量は「1セッションに1冊、以降はローカルキャッシュ」
なので既定は本家のままにしてあるが、config から差し替えられる。

権利の二重チェック
-----------------
青空文庫の「作品著作権フラグ」に当たる列が ``pg_catalog.csv`` に無い。
gutenberg.org が配るのは原則として米国 PD のものだが、ごく一部
著作権が残る作品も混ざる。それらは本文ヘッダに ``Copyright (C) ...`` を持つので、
:func:`is_public_domain_text` で本文側を見て弾く（§10.2 の二重チェックと同じ役)。
"""

from __future__ import annotations

import csv
import gzip
import io
import logging
import re
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz"
TEXT_URL_TEMPLATE = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
USER_AGENT = (
    "llm-radio-daemon/0.1 (personal 24h local-LLM radio project; gutenberg fetch)"
)
TIMEOUT = (10, 120)  # カタログが 5.5MB あるので aozora より読み取りを長く取る

# 取得済み本文の索引。列は aozora 版と同じ形にしてある（運用の勘所を揃えるため）。
INDEX_NAME = "index.csv"
INDEX_COLUMNS = ("work_id", "title", "author", "translator", "copyright_flag", "filename")

# 全候補のカタログ（pg_catalog.csv を絞ってキャッシュしたもの）。
CATALOG_NAME = "catalog.csv"
CATALOG_COLUMNS = (
    "work_id", "title", "author", "translator", "copyright_flag",
    "subjects", "bookshelves", "text_url",
)

# pg_catalog.csv の列名
C_WORK_ID = "Text#"
C_TYPE = "Type"
C_TITLE = "Title"
C_LANGUAGE = "Language"
C_AUTHORS = "Authors"
C_SUBJECTS = "Subjects"
C_BOOKSHELVES = "Bookshelves"

# 著者名の末尾に付く生没年（"Austen, Jane, 1775-1817"）。読み上げるので落とす。
_LIFE_DATES = re.compile(r",\s*(?:ca\.\s*)?\d{3,4}\??\s*-\s*(?:ca\.\s*)?\d{0,4}\??\s*$")
# 役割の注記（"Homer; Butler, Samuel [Translator]" のような形）。
_ROLE_NOTE = re.compile(r"\s*\[[^\]]*\]\s*")

# 本文ヘッダに著作権が残ることを示す一文があれば PD ではない。
_COPYRIGHT_HEADER = re.compile(r"^\s*copyright\s*\(c\)", re.IGNORECASE | re.MULTILINE)
_HEADER_SCAN_CHARS = 6000  # ヘッダだけ見る（本文中の "Copyright (C)" で誤爆しない）


class WorkRow:
    """カタログ1行ぶんの作品。``aozora.fetch.WorkRow`` と同じ役割。"""

    __slots__ = (
        "work_id", "title", "author", "translator", "has_translator",
        "is_pd", "subjects", "bookshelves", "text_url",
    )

    def __init__(self, work_id: str) -> None:
        self.work_id = work_id
        self.title = ""
        self.author = ""
        self.translator = ""
        self.has_translator = False
        # カタログには権利の列が無い。gutenberg.org が配っている＝原則 PD として
        # 扱い、本文を取った時点で is_public_domain_text() が二重チェックする。
        self.is_pd = True
        self.subjects = ""
        self.bookshelves = ""
        self.text_url = ""

    @property
    def has_text_url(self) -> bool:
        return bool(self.text_url)

    @property
    def is_eligible(self) -> bool:
        return self.is_pd and self.has_text_url and bool(self.title)


def _clean_field(value: str | None) -> str:
    """カタログの値を1行へ正規化する。

    ``Title`` には改行が入っていることがある（作品ID 2 の副題など）。
    読み上げにも字幕にも使うので、ここで畳んでおかないと台本が2行に割れる。
    """
    return " ".join((value or "").split())


def _person_name(raw: str) -> str:
    """``"Austen, Jane, 1775-1817"`` → ``"Jane Austen"``。

    朗読コーナーの導入で**声に出して読む**ので、姓名の順と生没年をここで直す。
    """
    name = _ROLE_NOTE.sub(" ", raw or "").strip()
    name = _LIFE_DATES.sub("", name).strip().strip(",").strip()
    if not name:
        return ""
    # "Last, First Middle" を "First Middle Last" へ。カンマが2つ以上残っている
    # ものや団体名（"United States"）はそのまま返す。
    parts = [p.strip() for p in name.split(",")]
    if len(parts) == 2 and parts[0] and parts[1]:
        return f"{parts[1]} {parts[0]}"
    return name


def _authors_text(raw: str) -> str:
    names = [_person_name(p) for p in (raw or "").split(";")]
    names = [n for n in dict.fromkeys(names) if n]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return " and ".join([", ".join(names[:-1]), names[-1]])


def _download(url: str) -> bytes:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.content


def fetch_catalog_rows() -> list[dict]:
    """``pg_catalog.csv.gz`` を取得して行の list を返す（ネットワーク I/O）。"""
    logger.info("gutenberg: fetching catalog CSV %s", CATALOG_URL)
    blob = _download(CATALOG_URL)
    raw = gzip.decompress(blob).decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(raw)))


def aggregate_works(rows: list[dict], *, lang: str = "en") -> dict[str, WorkRow]:
    """カタログの行を作品IDの dict へ。言語と種別（Text）で絞る。

    ``Language`` は ``"en"`` のほか ``"en; fr"`` のような複数指定がありうるので、
    先頭の言語だけを見る（対訳本を英語の朗読へ混ぜない）。
    """
    want = (lang or "en").strip().lower()
    works: dict[str, WorkRow] = {}
    for row in rows:
        wid = _clean_field(row.get(C_WORK_ID))
        if not wid or not wid.isdigit():
            continue
        if _clean_field(row.get(C_TYPE)).lower() != "text":
            continue
        langs = [l.strip().lower() for l in _clean_field(row.get(C_LANGUAGE)).split(";")]
        if not langs or langs[0] != want:
            continue

        w = WorkRow(wid)
        w.title = _clean_field(row.get(C_TITLE))
        w.author = _authors_text(_clean_field(row.get(C_AUTHORS)))
        w.subjects = _clean_field(row.get(C_SUBJECTS))
        w.bookshelves = _clean_field(row.get(C_BOOKSHELVES))
        w.text_url = TEXT_URL_TEMPLATE.format(id=wid)
        works[wid] = w
    return works


def eligible_works(
    works: dict[str, WorkRow], *, allow_translations: bool = True
) -> list[WorkRow]:
    """候補になる作品。

    ``allow_translations`` は青空文庫版と引数を揃えるためだけに受ける。
    **Gutenberg のカタログには訳者の役割が入っていない**（RDF にしか無い）ので、
    ここで翻訳を判別することはできない。gutenberg.org が配っている訳文は訳者ぶんも
    含めて米国 PD なので、青空文庫版のような除外はそもそも要らない。
    """
    return [w for w in works.values() if w.is_eligible]


def is_public_domain_text(raw: str) -> bool:
    """本文ヘッダを見て、著作権が残る作品でないかを確かめる（二重チェック）。"""
    return not _COPYRIGHT_HEADER.search(raw[:_HEADER_SCAN_CHARS])


def fetch_work_text(w: WorkRow, *, url_template: str = "") -> str | None:
    """1作品の本文テキストを取得する（ネットワーク I/O）。失敗時は None。"""
    url = (
        url_template.format(id=w.work_id) if url_template else w.text_url
    ) or TEXT_URL_TEMPLATE.format(id=w.work_id)
    try:
        blob = _download(url)
    except requests.RequestException as e:
        logger.warning("gutenberg work_id=%s: failed to fetch text: %s", w.work_id, e)
        return None
    text = blob.decode("utf-8", errors="replace")
    # PG の配布ファイルは \r\n。ここで \n へ正規化しておかないと、Windows では
    # 後段の corpus.ensure_text() が write_text() で保存するときに \n が \r\n へ
    # 変換され、既存の \r と重なって \r\r\n（＝読み返すと空行が1本増える）になる。
    # 英語は行が70〜80字で折り返されているため、パラグラフの内側の改行が軒並み
    # 空行へ化け、パラグラフ分割・章見出し判定が総崩れになる（実測）。
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not is_public_domain_text(text):
        logger.warning(
            "gutenberg work_id=%s \"%s\": body header has a copyright notice. Not reading it aloud",
            w.work_id, w.title,
        )
        return None
    return text


# --- カタログ（全候補の索引）のキャッシュ入出力 -----------------------------

def write_catalog(path: str | Path, works: dict[str, WorkRow]) -> int:
    """候補になる全作品を catalog.csv へ書き出す。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CATALOG_COLUMNS))
        writer.writeheader()
        for w in works.values():
            if not w.is_eligible:
                continue
            writer.writerow(
                {
                    "work_id": w.work_id,
                    "title": w.title,
                    "author": w.author,
                    "translator": w.translator,
                    # aozora 版の WorkMeta.is_public_domain が "no" を PD と読む。
                    "copyright_flag": "no",
                    "subjects": w.subjects,
                    "bookshelves": w.bookshelves,
                    "text_url": w.text_url,
                }
            )
            n += 1
    return n


def read_catalog(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def row_from_catalog_entry(entry: dict) -> WorkRow:
    w = WorkRow((entry.get("work_id") or "").strip())
    w.title = (entry.get("title") or "").strip()
    w.author = (entry.get("author") or "").strip()
    w.translator = (entry.get("translator") or "").strip()
    w.has_translator = bool(w.translator)
    w.is_pd = (entry.get("copyright_flag") or "no").strip().lower() in ("no", "", "false", "0")
    w.subjects = (entry.get("subjects") or "").strip()
    w.bookshelves = (entry.get("bookshelves") or "").strip()
    w.text_url = (entry.get("text_url") or "").strip()
    return w
