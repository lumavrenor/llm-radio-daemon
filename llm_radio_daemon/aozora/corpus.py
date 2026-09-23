"""青空文庫テキストのローカルキャッシュと候補選定（v4 §10.2）。

**遅延取得（lazy fetch）＋ローカルキャッシュ方式。** ``<corpus_dir>/`` に対象作品が
無ければ ``aozora.fetch`` でその場に取得し、以降はキャッシュから読む。毎回ネットへ
取りに行くことはしない（24時間モノでネットワーク依存を増やさない）。

- ``index.csv``   : 取得済み本文の索引（従来からの互換ファイル）
- ``catalog.csv`` : PD 全作品のカタログ（インデックス CSV を集計してキャッシュ。
  一定期間で再取得する）。作品の候補リストはここから作る。

取得ロジックそのものは ``aozora.fetch`` に一本化されており、
``scripts/tools/fetch_aozora.py``（バルク取得）も本体もそこを呼ぶ。
"""

from __future__ import annotations

import csv
import logging
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from . import fetch as _fetch
from .fetch import CATALOG_NAME, INDEX_COLUMNS, INDEX_NAME, WorkRow

logger = logging.getLogger(__name__)

# catalog.csv がこの日数より古ければ（auto_fetch 時に）再取得する。
CATALOG_MAX_AGE_DAYS = 30
# カタログ取得に失敗したあと、次に再試行するまでの最短間隔（秒）。
# オフライン時に candidates() のたびネットへ叩きに行かないためのスロットル。
_CATALOG_RETRY_COOLDOWN_SEC = 600


@dataclass(frozen=True)
class WorkMeta:
    work_id: str
    title: str
    author: str
    translator: str
    copyright_flag: str
    path: Path

    @property
    def is_public_domain(self) -> bool:
        return self.copyright_flag.strip() in ("なし", "no", "false", "0", "")

    @property
    def is_translation(self) -> bool:
        return bool(self.translator.strip())


@dataclass(frozen=True)
class Candidate:
    """作品選定に渡すメタ付き候補（§10.2「作品の選定」）。"""

    work_id: str
    title: str
    author: str
    translator: str
    ndc: str
    downloaded: bool
    approx_chars: int | None  # 取得済みのときだけ実文字数。未取得は None
    # 分野の見出し。青空文庫は ndc（日本十進分類）を使い、Project Gutenberg は
    # Bookshelves / Subjects を使う。分類体系が違うので同じ列に詰めず、
    # 使う側（literary_reading/curator.py）が言語で選ぶ。
    subjects: str = ""


class AozoraCorpus:
    def __init__(
        self,
        corpus_dir: str | Path,
        *,
        auto_fetch: bool = True,
        allow_translations: bool = False,
    ):
        self._dir = Path(corpus_dir)
        self._auto_fetch = auto_fetch
        self._allow_translations = allow_translations
        # index.csv / catalog.csv 書き込みの直列化。ensure_text がロック保持中に
        # カタログ再取得（refresh_catalog）へ再入するため RLock。
        self._lock = threading.RLock()
        self._catalog_retry_at = 0.0  # この時刻まではカタログ再取得を試みない

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
                        copyright_flag=(row.get("copyright_flag") or "").strip(),
                        path=self._dir / filename,
                    )
                )
        return out

    def available_works(self, allow_translations: bool | None = None) -> list[WorkMeta]:
        """著作権フラグ「なし」かつ（翻訳でない or 許可）で本文が存在するものだけ（§10.2 二重チェック）。"""
        allow = self._allow_translations if allow_translations is None else allow_translations
        works: list[WorkMeta] = []
        for w in self._read_index():
            if not w.is_public_domain:
                continue
            if w.is_translation and not allow:
                continue
            if not w.path.exists():
                logger.warning("aozora: body text not found work_id=%s path=%s", w.work_id, w.path)
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
            "translator": w.translator or ("訳あり" if w.has_translator else ""),
            "copyright_flag": "なし" if w.is_pd else "あり",
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
        """catalog.csv を（無い・古い・force 時に）インデックス CSV から作り直す。

        auto_fetch=False なら何もしない（手動取得ぶんだけで回す）。
        """
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
                works = _fetch.aggregate_works(_fetch.fetch_index_rows())
                n = _fetch.write_catalog(self.catalog_path, works)
                logger.info("aozora: updated catalog (%d works turned into candidates)", n)
            except Exception:
                self._catalog_retry_at = time.time() + _CATALOG_RETRY_COOLDOWN_SEC
                logger.exception("aozora: failed to update catalog (continuing with existing catalog / already-fetched works)")

    def _read_catalog_rows(self) -> list[dict]:
        if self._catalog_stale():
            self.refresh_catalog()
        rows = _fetch.read_catalog(self.catalog_path)
        if rows:
            return rows
        # カタログが無い（オフライン初回など）。取得済みの index.csv を候補として使う。
        return [
            {
                "work_id": w.work_id,
                "title": w.title,
                "author": w.author,
                "translator": w.translator,
                "copyright_flag": w.copyright_flag or "なし",
                "ndc": "",
                "text_url": "",
                "charset": "Shift_JIS",
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
        """メタ付きの候補リスト（PD・翻訳除外・除外ID済み）。

        limit を渡すとシャッフルして先頭 limit 件へ絞る（LLM へ渡す件数制限用・§10.2）。
        """
        allow = self._allow_translations if allow_translations is None else allow_translations
        exclude_ids = exclude_ids or set()
        downloaded = {w.work_id for w in self.available_works(allow_translations=allow)}

        out: list[Candidate] = []
        for row in self._read_catalog_rows():
            wid = (row.get("work_id") or "").strip()
            if not wid or wid in exclude_ids:
                continue
            translator = (row.get("translator") or "").strip()
            if translator and not allow:
                continue
            is_dl = wid in downloaded
            out.append(
                Candidate(
                    work_id=wid,
                    title=(row.get("title") or "").strip(),
                    author=(row.get("author") or "").strip(),
                    translator=translator,
                    ndc=(row.get("ndc") or "").strip(),
                    downloaded=is_dl,
                    approx_chars=self._approx_chars(wid) if is_dl else None,
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
            return meta.path.stat().st_size // 2  # UTF-8 日本語のおおよその文字数
        except OSError:
            return None

    def pick_work(
        self,
        exclude_ids: set[str] | None = None,
        allow_translations: bool | None = None,
        rng: random.Random | None = None,
    ) -> WorkMeta | None:
        """後方互換のランダム選定。候補から1つ選び、必要なら取得して WorkMeta を返す。"""
        cands = self.candidates(exclude_ids=exclude_ids, allow_translations=allow_translations, rng=rng)
        if not cands:
            cands = self.candidates(exclude_ids=set(), allow_translations=allow_translations, rng=rng)
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
            logger.warning("aozora: work_id=%s not yet fetched. Not fetching because auto_fetch=false", wid)
            return None

        with self._lock:
            meta = self.get_work(wid)
            if meta is not None and meta.path.exists():
                return meta
            row = self._catalog_row(wid)
            if row is None:
                logger.warning("aozora: work_id=%s not found in catalog. Cannot fetch", wid)
                return None
            w = _fetch.row_from_catalog_entry(row)
            text = _fetch.fetch_work_text(w)
            if text is None:
                return None
            self._dir.mkdir(parents=True, exist_ok=True)
            filename = f"{wid}.txt"
            (self._dir / filename).write_text(text, encoding="utf-8")
            self._append_index_row(w, filename)
            logger.info("aozora: fetched work_id=%s \"%s\" (%d chars)", wid, w.title, len(text))
            return self.get_work(wid)

    def _catalog_row(self, work_id: str) -> dict | None:
        return next(
            (r for r in self._read_catalog_rows() if (r.get("work_id") or "").strip() == work_id),
            None,
        )

    @staticmethod
    def load_text(meta: WorkMeta) -> str:
        return meta.path.read_text(encoding="utf-8", errors="replace")
