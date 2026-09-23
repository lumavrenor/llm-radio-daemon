"""Project Gutenberg テキストのローカルキャッシュと候補選定（``aozora/corpus.py`` の英語版）。

**遅延取得（lazy fetch）＋ローカルキャッシュ方式。** ``<corpus_dir>/`` に対象作品が
無ければ ``gutenberg.fetch`` でその場に取得し、以降はキャッシュから読む。

- ``index.csv``   : 取得済み本文の索引
- ``catalog.csv`` : 候補のカタログ（``pg_catalog.csv.gz`` を言語で絞ってキャッシュ。
  一定期間で再取得する）

``WorkMeta`` / ``Candidate`` は ``aozora.corpus`` のものをそのまま使う。
``literary_reading/corner.py`` はこの2つの型で書かれており、言語で corpus を
差し替えるだけで通るようにしてある（型を2つ作ると上の層が両方を知ることになる）。
"""

from __future__ import annotations

import csv
import logging
import random
import threading
import time
from pathlib import Path

from ..aozora.corpus import Candidate, WorkMeta
from . import fetch as _fetch
from .fetch import CATALOG_NAME, INDEX_COLUMNS, INDEX_NAME, WorkRow

logger = logging.getLogger(__name__)

# catalog.csv がこの日数より古ければ（auto_fetch 時に）再取得する。
CATALOG_MAX_AGE_DAYS = 30
# カタログ取得に失敗したあとの再試行間隔（秒）。オフライン時に candidates() の
# たびネットへ叩きに行かないためのスロットル。
_CATALOG_RETRY_COOLDOWN_SEC = 600

# 1文字あたりのバイト数。index.csv に文字数を持たないので、ファイルサイズから
# おおよその長さを出すのに使う（英語の UTF-8 はほぼ1バイト＝1文字）。
_BYTES_PER_CHAR = 1


class GutenbergCorpus:
    def __init__(
        self,
        corpus_dir: str | Path,
        *,
        auto_fetch: bool = True,
        allow_translations: bool = True,
        lang: str = "en",
        catalog_url: str = "",
        text_url_template: str = "",
    ):
        self._dir = Path(corpus_dir)
        self._auto_fetch = auto_fetch
        # Gutenberg のカタログには訳者の役割が入っていないので判別できない。
        # 引数は aozora 版と合わせるためだけに受ける（fetch.eligible_works 参照）。
        self._allow_translations = allow_translations
        self._lang = lang or "en"
        self._catalog_url = catalog_url or _fetch.CATALOG_URL
        self._text_url_template = text_url_template or _fetch.TEXT_URL_TEMPLATE
        self._lock = threading.RLock()
        self._catalog_retry_at = 0.0

    # --- パス ----------------------------------------------------------

    @property
    def index_path(self) -> Path:
        return self._dir / INDEX_NAME

    @property
    def catalog_path(self) -> Path:
        return self._dir / CATALOG_NAME

    # --- 取得済み本文の索引（index.csv）--------------------------------

    def _read_index(self) -> list[WorkMeta]:
        if not self.index_path.exists():
            return []
        out: list[WorkMeta] = []
        with self.index_path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                filename = (row.get("filename") or "").strip()
                if not filename:
                    continue
                out.append(
                    WorkMeta(
                        work_id=(row.get("work_id") or "").strip(),
                        title=(row.get("title") or "").strip(),
                        author=(row.get("author") or "").strip(),
                        translator=(row.get("translator") or "").strip(),
                        copyright_flag=(row.get("copyright_flag") or "no").strip(),
                        path=self._dir / filename,
                    )
                )
        return out

    def available_works(self, allow_translations: bool | None = None) -> list[WorkMeta]:
        """本文が存在し、PD 扱いのものだけ。"""
        works: list[WorkMeta] = []
        for w in self._read_index():
            if not w.is_public_domain:
                continue
            if not w.path.exists():
                logger.warning("gutenberg: text not found work_id=%s path=%s", w.work_id, w.path)
                continue
            works.append(w)
        return works

    def get_work(self, work_id: str) -> WorkMeta | None:
        """取得済み（index.csv 掲載）作品のメタ。未取得なら None。"""
        wid = str(work_id).strip()
        return next((w for w in self._read_index() if w.work_id == wid), None)

    def _append_index_row(self, w: WorkRow, filename: str) -> None:
        rows: dict[str, dict] = {}
        if self.index_path.exists():
            with self.index_path.open(encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    rows[row["work_id"]] = row
        rows[w.work_id] = {
            "work_id": w.work_id,
            "title": w.title,
            "author": w.author,
            "translator": w.translator,
            "copyright_flag": "no" if w.is_pd else "yes",
            "filename": filename,
        }
        self._dir.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(INDEX_COLUMNS))
            writer.writeheader()
            for row in rows.values():
                writer.writerow(row)

    # --- カタログ（catalog.csv）--------------------------------------

    def _catalog_stale(self) -> bool:
        p = self.catalog_path
        if not p.exists():
            return True
        age_days = (time.time() - p.stat().st_mtime) / 86400
        return age_days >= CATALOG_MAX_AGE_DAYS

    def refresh_catalog(self, *, force: bool = False) -> None:
        """catalog.csv を（無い・古い・force 時に）pg_catalog.csv から作り直す。"""
        if not force and not self._catalog_stale():
            return
        if not self._auto_fetch and not force:
            return
        if not force and time.time() < self._catalog_retry_at:
            return  # 直近に失敗している。クールダウン中は取りに行かない
        with self._lock:
            if not force and not self._catalog_stale():
                return
            try:
                rows = _fetch.fetch_catalog_rows()
                works = _fetch.aggregate_works(rows, lang=self._lang)
                n = _fetch.write_catalog(self.catalog_path, works)
                logger.info(
                    "gutenberg: catalog updated (%d %s works turned into candidates from %d rows)",
                    n, self._lang, len(rows),
                )
            except Exception:
                self._catalog_retry_at = time.time() + _CATALOG_RETRY_COOLDOWN_SEC
                logger.exception(
                    "gutenberg: failed to update catalog (continuing with the existing catalog / already-fetched works)"
                )

    def _read_catalog_rows(self) -> list[dict]:
        if self._catalog_stale():
            self.refresh_catalog()
        rows = _fetch.read_catalog(self.catalog_path)
        if rows:
            return rows
        # カタログが無い（オフライン初回など）。取得済みの index.csv を候補にする。
        return [
            {
                "work_id": w.work_id,
                "title": w.title,
                "author": w.author,
                "translator": w.translator,
                "copyright_flag": w.copyright_flag or "no",
                "subjects": "",
                "bookshelves": "",
                "text_url": "",
            }
            for w in self._read_index()
        ]

    # --- 候補選定 ---------------------------------------------------

    def candidates(
        self,
        *,
        exclude_ids: set[str] | None = None,
        allow_translations: bool | None = None,
        limit: int | None = None,
        rng: random.Random | None = None,
    ) -> list[Candidate]:
        """メタ付きの候補リスト（除外ID済み）。

        limit を渡すとシャッフルして先頭 limit 件へ絞る（LLM へ渡す件数制限用）。
        """
        exclude_ids = exclude_ids or set()
        downloaded = {w.work_id for w in self.available_works()}

        out: list[Candidate] = []
        for row in self._read_catalog_rows():
            wid = (row.get("work_id") or "").strip()
            if not wid or wid in exclude_ids:
                continue
            is_dl = wid in downloaded
            out.append(
                Candidate(
                    work_id=wid,
                    title=(row.get("title") or "").strip(),
                    author=(row.get("author") or "").strip(),
                    translator=(row.get("translator") or "").strip(),
                    ndc="",  # 青空文庫の日本十進分類。Gutenberg には無い
                    downloaded=is_dl,
                    approx_chars=self._approx_chars(wid) if is_dl else None,
                    subjects=_topic_label(row),
                )
            )

        if limit is not None and len(out) > limit:
            out = (rng or random).sample(out, limit)
        return out

    def _approx_chars(self, work_id: str) -> int | None:
        meta = self.get_work(work_id)
        if meta is None or not meta.path.exists():
            return None
        try:
            return meta.path.stat().st_size // _BYTES_PER_CHAR
        except OSError:
            return None

    def pick_work(
        self,
        exclude_ids: set[str] | None = None,
        allow_translations: bool | None = None,
        rng: random.Random | None = None,
    ) -> WorkMeta | None:
        """ランダム選定。候補から1つ選び、必要なら取得して WorkMeta を返す。"""
        cands = self.candidates(exclude_ids=exclude_ids, rng=rng)
        if not cands:
            cands = self.candidates(exclude_ids=set(), rng=rng)
        if not cands:
            return None
        pick = (rng or random).choice(cands)
        return self.ensure_text(pick.work_id)

    # --- 遅延取得 -------------------------------------------------------

    def ensure_text(self, work_id: str) -> WorkMeta | None:
        """取得済みなら WorkMeta を返す。未取得で auto_fetch なら取得してから返す。"""
        wid = str(work_id).strip()
        meta = self.get_work(wid)
        if meta is not None and meta.path.exists():
            return meta

        if not self._auto_fetch:
            logger.warning(
                "gutenberg: work_id=%s not fetched yet. Not fetching it because auto_fetch=false", wid
            )
            return None

        with self._lock:
            meta = self.get_work(wid)
            if meta is not None and meta.path.exists():
                return meta
            row = self._catalog_row(wid)
            if row is None:
                logger.warning("gutenberg: work_id=%s not in catalog. Cannot fetch", wid)
                return None
            w = _fetch.row_from_catalog_entry(row)
            text = _fetch.fetch_work_text(w, url_template=self._text_url_template)
            if text is None:
                return None
            self._dir.mkdir(parents=True, exist_ok=True)
            filename = f"{wid}.txt"
            (self._dir / filename).write_text(text, encoding="utf-8")
            self._append_index_row(w, filename)
            logger.info("gutenberg: fetched work_id=%s \"%s\" (%d chars)", wid, w.title, len(text))
            return self.get_work(wid)

    def _catalog_row(self, work_id: str) -> dict | None:
        return next(
            (r for r in self._read_catalog_rows() if (r.get("work_id") or "").strip() == work_id),
            None,
        )

    @staticmethod
    def load_text(meta: WorkMeta) -> str:
        return meta.path.read_text(encoding="utf-8", errors="replace")


# 選書プロンプトへ出す「分野」。Subjects は長いので Bookshelves の
# "Category: ..." を優先する（"Category: Fiction" のような短い見出しが付いている）。
def _topic_label(row: dict) -> str:
    shelves = [s.strip() for s in (row.get("bookshelves") or "").split(";") if s.strip()]
    cats = [s[len("Category:") :].strip() for s in shelves if s.startswith("Category:")]
    if cats:
        return ", ".join(cats[:3])
    if shelves:
        return ", ".join(shelves[:3])
    subjects = [s.strip() for s in (row.get("subjects") or "").split(";") if s.strip()]
    # Subjects は "England -- Fiction" のような形。先頭の語だけ拾う。
    return ", ".join(s.split(" -- ")[0] for s in subjects[:3])
