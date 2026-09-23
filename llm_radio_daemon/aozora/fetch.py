"""青空文庫の取得プリミティブ（v4 §10.2）。

``scripts/tools/fetch_aozora.py``（バルク取得・完全オフライン運用向け）と
``llm_radio_daemon.aozora.corpus``（本体の実行時 lazy fetch）の両方がここを使う。
**取得ロジックを二重に持たないための共有層。**

ネットワークへ出るのはこのモジュールだけ。上位（corpus / corner）は
「無ければ取りに行く。以降はキャッシュから読む」という方針で薄く呼ぶ。
"""

from __future__ import annotations

import csv
import io
import logging
import zipfile
from collections import defaultdict
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

INDEX_URL = "https://www.aozora.gr.jp/index_pages/list_person_all_extended_utf8.zip"
USER_AGENT = "llm-radio-daemon/0.1 (personal 24h local-LLM radio project; aozora fetch)"
TIMEOUT = (10, 60)

# 取得済み本文の索引（従来からの互換ファイル）。
INDEX_NAME = "index.csv"
INDEX_COLUMNS = ("work_id", "title", "author", "translator", "copyright_flag", "filename")

# 全候補のカタログ（インデックス CSV を集計してキャッシュしたもの・§10.2）。
CATALOG_NAME = "catalog.csv"
CATALOG_COLUMNS = (
    "work_id", "title", "author", "translator", "copyright_flag", "ndc", "text_url", "charset",
)

# CSV の列名（list_person_all_extended_utf8.csv）
C_WORK_ID = "作品ID"
C_TITLE = "作品名"
C_SUBTITLE = "副題"
C_COPYRIGHT = "作品著作権フラグ"
C_ROLE = "役割フラグ"
C_LAST = "姓"
C_FIRST = "名"
C_TEXT_URL = "テキストファイルURL"
C_CHARSET = "テキストファイル文字集合"
C_NDC = "分類番号"

_PY_ENCODING = {"Shift_JIS": "cp932", "ShiftJIS": "cp932", "shift_jis": "cp932"}


class WorkRow:
    """インデックス CSV の複数行（著者・訳者など役割ごとに1行）を1作品へ集約したもの。"""

    __slots__ = (
        "work_id", "title", "author", "translator", "has_translator",
        "is_pd", "ndc", "text_url", "charset",
    )

    def __init__(self, work_id: str) -> None:
        self.work_id = work_id
        self.title = ""
        self.author = ""
        self.translator = ""
        self.has_translator = False
        self.is_pd = False
        self.ndc = ""
        self.text_url = ""
        self.charset = "Shift_JIS"

    @property
    def has_text_zip(self) -> bool:
        return self.text_url.lower().endswith(".zip")

    @property
    def is_eligible(self) -> bool:
        """翻訳を問わず、PD かつ本文 ZIP があるか（翻訳除外は読み手側の設定で行う）。"""
        return self.is_pd and self.has_text_zip


def _download(url: str) -> bytes:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.content


def fetch_index_rows() -> list[dict]:
    """青空文庫の拡張インデックス CSV を取得して行の list を返す（ネットワーク I/O）。"""
    logger.info("aozora: fetching index CSV %s", INDEX_URL)
    blob = _download(INDEX_URL)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        raw = zf.read(name).decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(raw)))


def aggregate_works(rows: list[dict]) -> dict[str, WorkRow]:
    """インデックス CSV の行を作品ID単位へ集約する。著作権フラグ・訳者の有無もここで判定。"""
    by_id: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        wid = (row.get(C_WORK_ID) or "").strip()
        if wid:
            by_id[wid].append(row)

    works: dict[str, WorkRow] = {}
    for wid, group in by_id.items():
        w = WorkRow(wid)
        first = group[0]
        w.title = (first.get(C_TITLE) or "").strip()
        subtitle = (first.get(C_SUBTITLE) or "").strip()
        if subtitle:
            w.title = f"{w.title}　{subtitle}"
        w.is_pd = (first.get(C_COPYRIGHT) or "").strip() == "なし"
        w.text_url = (first.get(C_TEXT_URL) or "").strip()
        w.charset = (first.get(C_CHARSET) or "Shift_JIS").strip() or "Shift_JIS"
        w.ndc = (first.get(C_NDC) or "").strip()

        fallback_name = ""
        translators: list[str] = []
        for row in group:
            role = (row.get(C_ROLE) or "").strip()
            name = f"{(row.get(C_LAST) or '').strip()}{(row.get(C_FIRST) or '').strip()}"
            if "翻訳" in role:
                w.has_translator = True
                if name:
                    translators.append(name)
            if "著者" in role and name:
                w.author = name
            elif name and not fallback_name:
                fallback_name = name
        w.translator = "、".join(dict.fromkeys(translators))
        if not w.author:
            w.author = fallback_name
        works[wid] = w
    return works


def eligible_works(
    works: dict[str, WorkRow], *, allow_translations: bool = False
) -> list[WorkRow]:
    return [
        w for w in works.values()
        if w.is_eligible and (allow_translations or not w.has_translator)
    ]


def fetch_work_text(w: WorkRow) -> str | None:
    """1作品の本文テキストを取得する（ネットワーク I/O）。失敗時は None。"""
    if not w.has_text_zip:
        logger.warning("aozora work_id=%s: no text ZIP URL (%s)", w.work_id, w.text_url or "empty")
        return None
    try:
        blob = _download(w.text_url)
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            txt_name = next(n for n in zf.namelist() if n.lower().endswith(".txt"))
            raw = zf.read(txt_name)
    except (requests.RequestException, zipfile.BadZipFile, StopIteration) as e:
        logger.warning("aozora work_id=%s: failed to fetch text: %s", w.work_id, e)
        return None
    enc = _PY_ENCODING.get(w.charset, "cp932")
    text = raw.decode(enc, errors="replace")
    # 青空文庫の配布ファイルは \r\n。ここで \n へ正規化しておかないと、Windows では
    # 後段の corpus.ensure_text() が write_text() で保存するときに \n が \r\n へ
    # 変換され、既存の \r と重なって \r\r\n（＝読み返すと空行が1本増える）になる
    # （gutenberg/fetch.py と同じ不具合。詳細は docs/handoff-en-tts.md §5.4）。
    return text.replace("\r\n", "\n").replace("\r", "\n")


# --- カタログ（全候補の索引）のキャッシュ入出力 -----------------------------

def write_catalog(path: str | Path, works: dict[str, WorkRow]) -> int:
    """PD かつ本文 ZIP のある全作品を catalog.csv へ書き出す（翻訳除外は読み手側）。"""
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
                    "copyright_flag": "なし",
                    "ndc": w.ndc,
                    "text_url": w.text_url,
                    "charset": w.charset,
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
    w.is_pd = (entry.get("copyright_flag") or "なし").strip() == "なし"
    w.ndc = (entry.get("ndc") or "").strip()
    w.text_url = (entry.get("text_url") or "").strip()
    w.charset = (entry.get("charset") or "Shift_JIS").strip() or "Shift_JIS"
    return w
