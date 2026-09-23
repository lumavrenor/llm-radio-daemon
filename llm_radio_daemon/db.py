"""SQLite による永続化と重複排除。

第1段階（v1）: external_id の UNIQUE 制約による完全一致排除。
第2段階（v3）: embedding 列に title+body のベクトルを保存し、直近トピックとの
コサイン類似度による類似トピック排除に使う（実際の類似度計算は dedup.py）。
"""

from __future__ import annotations

import datetime
import logging
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS topics (
    id           INTEGER PRIMARY KEY,
    source       TEXT NOT NULL,
    external_id  TEXT NOT NULL,
    title        TEXT NOT NULL,
    body         TEXT,
    url          TEXT,
    embedding    BLOB,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    used_at      TIMESTAMP,
    UNIQUE(source, external_id)
);
CREATE INDEX IF NOT EXISTS idx_topics_used ON topics(used_at);

CREATE TABLE IF NOT EXISTS broadcasts (
    id          INTEGER PRIMARY KEY,
    topic_id    INTEGER REFERENCES topics(id),
    script_json TEXT NOT NULL,
    aired_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS now_playing_log (
    id          INTEGER PRIMARY KEY,
    stream      TEXT,
    artist      TEXT,
    title       TEXT,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 読書コーナー（青空文庫）— v4 §10.6
CREATE TABLE IF NOT EXISTS literary_reading_sessions (
    work_id         TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    author          TEXT,
    cursor          INTEGER NOT NULL DEFAULT 0,
    total_chunks    INTEGER NOT NULL,
    rolling_summary TEXT,
    started_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP,
    finished_at     TIMESTAMP
);

CREATE TABLE IF NOT EXISTS literary_reading_log (
    id          INTEGER PRIMARY KEY,
    work_id     TEXT REFERENCES literary_reading_sessions(work_id),
    chunk_index INTEGER,
    kind        TEXT,      -- "read" | "comment"
    payload     TEXT,      -- 感想の場合は script JSON
    aired_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 翻訳朗読コーナー（Project Gutenberg → 翻訳ナレーション）。session/log の構造は
-- literary_reading と同じ考え方だが、literary_reading と時間帯が重ならないだけで
-- 同じ作品を並行して読むこともあり得るため状態を別テーブルに分けている。
CREATE TABLE IF NOT EXISTS translated_reading_sessions (
    work_id         TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    author          TEXT,
    cursor          INTEGER NOT NULL DEFAULT 0,
    total_chunks    INTEGER NOT NULL,
    rolling_summary TEXT,
    started_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP,
    finished_at     TIMESTAMP
);

CREATE TABLE IF NOT EXISTS translated_reading_log (
    id          INTEGER PRIMARY KEY,
    work_id     TEXT REFERENCES translated_reading_sessions(work_id),
    chunk_index INTEGER,
    kind        TEXT,      -- "translate" | "comment"
    payload     TEXT,      -- script JSON
    aired_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 偉人伝トーク（Wikipedia）。対象人物は figures_file（content/biography_figures.<lang>.txt）の
-- 明示指定のみ（自動選定はしない方針）。session/log の構造は literary_reading と同じ考え方。
CREATE TABLE IF NOT EXISTS biography_reading_sessions (
    figure_id       TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    cursor          INTEGER NOT NULL DEFAULT 0,
    total_chunks    INTEGER NOT NULL,
    rolling_summary TEXT,
    started_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP,
    finished_at     TIMESTAMP
);

CREATE TABLE IF NOT EXISTS biography_reading_log (
    id          INTEGER PRIMARY KEY,
    figure_id   TEXT REFERENCES biography_reading_sessions(figure_id),
    chunk_index INTEGER,
    payload     TEXT,      -- script JSON
    aired_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ラジオドラマ朗読（v6 §4.7）。執筆バッチ（generated_drama_writer）が generated_dramas /
-- generated_drama_scenes を書き、放送プロセスはそれを読むだけ。
-- generated_drama_progress だけは放送側が書く。
CREATE TABLE IF NOT EXISTS generated_dramas (
    id          INTEGER PRIMARY KEY,
    title       TEXT NOT NULL,
    premise     TEXT,
    priority    INTEGER NOT NULL DEFAULT 0,       -- run --auto が進める順番
    status      TEXT NOT NULL DEFAULT 'writing',  -- "writing" | "finished" | "paused"
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS generated_drama_scenes (
    id                  INTEGER PRIMARY KEY,
    generated_drama_id  INTEGER NOT NULL REFERENCES generated_dramas(id),
    chapter             INTEGER NOT NULL,
    scene               INTEGER NOT NULL,
    body                TEXT NOT NULL,                -- 確定済み本文（放送側はこれをそのまま読む）
    summary             TEXT,                         -- チェックフェーズ用のシーン要約
    ready               INTEGER NOT NULL DEFAULT 0,   -- チェック通過済み＝放送してよい
    attempts            INTEGER NOT NULL DEFAULT 0,   -- 書き直した回数（無限リトライ防止）
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(generated_drama_id, chapter, scene)
);

-- 放送済み位置。「同じ話題を避ける」topics/used_at とは別軸（順番に最後まで読み切る）。
-- chunk_cursor / finished_at は §4.7 のスキーマへの追加分。シーン単位だけで管理すると
-- 「シーンの途中で強制終了」したときに読み飛ばしか二重読みのどちらかが必ず起きるため、
-- シーン内のチャンク位置も持たせて再開できるようにしている（§9 のテスト項目）。
CREATE TABLE IF NOT EXISTS generated_drama_progress (
    id           INTEGER PRIMARY KEY,
    scene_id     INTEGER NOT NULL REFERENCES generated_drama_scenes(id),
    chunk_cursor INTEGER NOT NULL DEFAULT 0,   -- 次に読むチャンク番号（0始まり）
    aired_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- 最初に流し始めた時刻
    finished_at  TIMESTAMP,                    -- 最終チャンクを流し始めた時刻。NULL なら途中
    UNIQUE(scene_id)
);

-- リクエスト（番組表への一時オーバーライド）— §6.2
--
-- 時刻を他テーブルのような CURRENT_TIMESTAMP（UTC文字列）ではなく UNIX 秒で持つ。
-- 「あと30分」を request CLI（書く側）と放送プロセス（読む側）が別プロセスで
-- 突き合わせるので、タイムゾーンや文字列比較の解釈が入る余地を残さないため。
CREATE TABLE IF NOT EXISTS requests (
    id            INTEGER PRIMARY KEY,
    content_index INTEGER NOT NULL,  -- config.content の添字（同 type が複数あっても一意に指せる）
    content_type  TEXT NOT NULL,     -- 添字ズレ（発行後に config を編集した）の検出用
    label         TEXT,              -- 読み上げ用のコーナー名（発行時の config から複写）
    created_at    REAL NOT NULL,     -- UNIX 秒
    expires_at    REAL NOT NULL,     -- UNIX 秒。これを過ぎたら番組表へ戻る
    announced_at  REAL,              -- 番組進行がリクエストを受け付けた時刻。ここが埋まると
                                      -- 番組表より優先される（§6.2 参照）
    entered_at    REAL               -- 受けアナウンスの最後の行が実際に再生され始めた時刻（記録用）
);
CREATE INDEX IF NOT EXISTS idx_requests_expires ON requests(expires_at);
"""


class TopicStore:
    """SQLite への読み書きをまとめるクラス。

    sqlite3 のコネクションはスレッドをまたげないため、呼び出しごとに
    Lock で直列化しつつ、同一スレッド内で使い回す前提にはせず
    check_same_thread=False + Lock で複数ワーカースレッドからの利用を許す。
    """

    def __init__(self, path: str | Path, rebroadcast_after_days: int = 7):
        self._path = str(path)
        self._rebroadcast_after_days = rebroadcast_after_days
        self._lock = threading.Lock()
        parent = Path(self._path).parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
        # 放送プロセスと執筆バッチ（generated_drama_writer・§4.7.1）が同じ DB を同時に開く。
        # WAL なら「片方が書いている間ももう片方が読める」ので、放送中に
        # run --auto を叩いても database is locked で落ちない。
        self._conn = sqlite3.connect(self._path, check_same_thread=False, timeout=10.0)
        # 自動コミット。**Python 既定の legacy モードにしてはいけない。**
        #
        # legacy モードでは sqlite3 が INSERT/UPDATE/DELETE の前に勝手に BEGIN を張る。
        # その文が例外で落ちると、トランザクションは開いたまま残る。WAL の書き込み
        # ロックは COMMIT / ROLLBACK まで解放されないので、放送プロセスがロックを
        # 握りっぱなしになり、別プロセス（request CLI・執筆バッチ）からの書き込みが
        # busy_timeout を使い切って永久に "database is locked" になる。
        # 放送プロセス自身は同じコネクションなので書き続けられ、症状が出ない ――
        # 「1人で使っている間は誰も気づかない」たちの悪い壊れ方をする。
        #
        # 実際 try_insert_topic() の重複 INSERT（IntegrityError）は毎回の巡回で
        # 普通に起きるので、これを踏むのは事故ではなく既定動作だった。
        # 自動コミットなら「開きっぱなしのトランザクション」自体が作られない。
        # 各メソッドの commit() は無害な空振りになる（複数文をまとめたいところは
        # put_request() のように明示的に BEGIN する）。
        self._conn.isolation_level = None
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=10000")
        self._rename_legacy_tables()
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _rename_legacy_tables(self) -> None:
        """09-05 改名（novel → generated_drama / reading → literary_reading）の
        テーブル・列名移行。新名のテーブルが既に存在する場合は無視する（何度実行しても安全）。
        """
        existing = {
            r[0]
            for r in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        renames = (
            ("novels", "generated_dramas"),
            ("novel_scenes", "generated_drama_scenes"),
            ("novel_progress", "generated_drama_progress"),
            ("reading_sessions", "literary_reading_sessions"),
            ("reading_log", "literary_reading_log"),
        )
        for old, new in renames:
            if old in existing and new not in existing:
                try:
                    logger.info("db: renaming table %s to %s", old, new)
                    self._conn.execute(f"ALTER TABLE {old} RENAME TO {new}")
                except sqlite3.OperationalError:
                    logger.exception("db: failed to rename table %s -> %s", old, new)
                    continue
                existing.discard(old)
                existing.add(new)

        if "generated_drama_scenes" in existing:
            cols = {
                r[1]
                for r in self._conn.execute(
                    "PRAGMA table_info(generated_drama_scenes)"
                ).fetchall()
            }
            if "novel_id" in cols and "generated_drama_id" not in cols:
                try:
                    logger.info(
                        "db: renaming generated_drama_scenes.novel_id to generated_drama_id"
                    )
                    self._conn.execute(
                        "ALTER TABLE generated_drama_scenes "
                        "RENAME COLUMN novel_id TO generated_drama_id"
                    )
                except sqlite3.OperationalError:
                    logger.exception("db: failed to rename column novel_id -> generated_drama_id")
        self._conn.commit()

    def _migrate(self) -> None:
        """既存 DB に後から足した列を埋める（CREATE TABLE IF NOT EXISTS は列を足さない）。"""
        for table, column, ddl in (
            ("generated_drama_scenes", "attempts", "INTEGER NOT NULL DEFAULT 0"),
            ("generated_drama_progress", "chunk_cursor", "INTEGER NOT NULL DEFAULT 0"),
            ("generated_drama_progress", "finished_at", "TIMESTAMP"),
            ("requests", "entered_at", "REAL"),
        ):
            cols = {
                r[1] for r in self._conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if cols and column not in cols:
                logger.info("db: adding column %s to %s", table, column)
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def try_insert_topic(
        self,
        source: str,
        external_id: str,
        title: str,
        body: str | None,
        url: str | None,
    ) -> int | None:
        """新規トピックなら挿入して topic_id を返す。既知なら None。"""
        with self._lock:
            try:
                cur = self._conn.execute(
                    "INSERT INTO topics (source, external_id, title, body, url) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (source, external_id, title, body, url),
                )
                self._conn.commit()
                return cur.lastrowid
            except sqlite3.IntegrityError:
                # 既知トピック（UNIQUE 違反）。巡回のたびに普通に起きる。
                # 自動コミットなら開いたトランザクションは無いが、念のため戻す
                # （legacy モードに戻された場合にロックを握り続けないための保険）。
                self._conn.rollback()
                return None

    def delete_topics_by_source(self, source: str) -> int:
        """指定ソースのトピックを全削除し、消した件数を返す（デバッグ用）。

        ``external_id`` の既知判定は topics テーブルだけを見るので、行を消せば
        次の巡回で同じ記事がまた「新規」として取り込まれる。埋め込みは topics の
        列なので行ごと消える。broadcasts.topic_id は FK 未強制のため参照が
        宙に浮くだけで実害はない（放送履歴のログとして残る）。
        """
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM topics WHERE source = ?", (source,)
            )
            self._conn.commit()
            return cur.rowcount

    def reset_all_content(self) -> dict[str, int]:
        """「聴いた/読んだ」状態を全部消す（デバッグ用の全リセット）。

        トピック（全ソース）・放送履歴・読書/偉人伝コーナーの進捗・ラジオドラマの
        朗読進捗・リクエスト履歴を削除する。generated_dramas /
        generated_drama_scenes（執筆バッチが書いた小説の本文そのもの）は対象外
        —— 消すのは放送側が「読んだかどうか」の状態だけで、書いた小説は失わせない。
        戻り値はテーブルごとの削除件数。
        """
        tables = (
            "topics",
            "broadcasts",
            "now_playing_log",
            "literary_reading_sessions",
            "literary_reading_log",
            "translated_reading_sessions",
            "translated_reading_log",
            "biography_reading_sessions",
            "biography_reading_log",
            "generated_drama_progress",
            "requests",
        )
        counts: dict[str, int] = {}
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                for table in tables:
                    cur = self._conn.execute(f"DELETE FROM {table}")
                    counts[table] = cur.rowcount
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return counts

    def save_embedding(self, topic_id: int, embedding: np.ndarray) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE topics SET embedding = ? WHERE id = ?",
                (embedding.astype(np.float32).tobytes(), topic_id),
            )
            self._conn.commit()

    def recent_embeddings(self, exclude_id: int, limit: int) -> list[tuple[int, np.ndarray]]:
        """埋め込み済みの直近トピックを新しい順に最大 limit 件返す（比較対象自身は除く）。"""
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, embedding FROM topics WHERE embedding IS NOT NULL AND id != ? "
                "ORDER BY created_at DESC LIMIT ?",
                (exclude_id, limit),
            )
            rows = cur.fetchall()
        return [(row[0], np.frombuffer(row[1], dtype=np.float32)) for row in rows]

    def is_known(self, source: str, external_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "SELECT 1 FROM topics WHERE source = ? AND external_id = ?",
                (source, external_id),
            )
            return cur.fetchone() is not None

    def mark_used(self, topic_id: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE topics SET used_at = CURRENT_TIMESTAMP WHERE id = ?",
                (topic_id,),
            )
            self._conn.commit()

    def record_broadcast(self, topic_id: int | None, script_json: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO broadcasts (topic_id, script_json) VALUES (?, ?)",
                (topic_id, script_json),
            )
            self._conn.commit()

    def record_now_playing(self, stream: str, artist: str, title: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO now_playing_log (stream, artist, title) VALUES (?, ?, ?)",
                (stream, artist, title),
            )
            self._conn.commit()

    # --- 読書コーナー（v4 §10.6）---------------------------------------

    def load_literary_reading_session(self, work_id: str) -> dict | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT work_id, title, author, cursor, total_chunks, rolling_summary, finished_at "
                "FROM literary_reading_sessions WHERE work_id = ?",
                (work_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return {
            "work_id": row[0],
            "title": row[1],
            "author": row[2],
            "cursor": row[3],
            "total_chunks": row[4],
            "rolling_summary": row[5] or "",
            "finished_at": row[6],
        }

    def latest_unfinished_literary_reading_session(self) -> dict | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT work_id FROM literary_reading_sessions WHERE finished_at IS NULL "
                "ORDER BY COALESCE(updated_at, started_at) DESC LIMIT 1"
            )
            row = cur.fetchone()
        return self.load_literary_reading_session(row[0]) if row else None

    def open_literary_reading_session(
        self, work_id: str, title: str, author: str, total_chunks: int
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO literary_reading_sessions (work_id, title, author, total_chunks) "
                "VALUES (?, ?, ?, ?)",
                (work_id, title, author, total_chunks),
            )
            self._conn.commit()

    def advance_literary_reading_cursor(self, work_id: str, cursor: int) -> None:
        """朗読を再生開始した時点で呼ぶ（合成完了時ではない・§10.6）。後退はさせない。"""
        with self._lock:
            self._conn.execute(
                "UPDATE literary_reading_sessions SET cursor = MAX(cursor, ?), "
                "updated_at = CURRENT_TIMESTAMP WHERE work_id = ?",
                (cursor, work_id),
            )
            self._conn.commit()

    def save_literary_reading_summary(self, work_id: str, rolling_summary: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE literary_reading_sessions SET rolling_summary = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE work_id = ?",
                (rolling_summary, work_id),
            )
            self._conn.commit()

    def finish_literary_reading_session(self, work_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE literary_reading_sessions SET finished_at = CURRENT_TIMESTAMP WHERE work_id = ?",
                (work_id,),
            )
            self._conn.commit()

    def recently_read_work_ids(self, days: int) -> set[str]:
        """finished_at が days 日以内の作品ID（§10.6 の30日ルール）。"""
        with self._lock:
            cur = self._conn.execute(
                "SELECT work_id FROM literary_reading_sessions WHERE finished_at IS NOT NULL "
                "AND finished_at >= datetime('now', ?)",
                (f"-{int(days)} days",),
            )
            return {r[0] for r in cur.fetchall()}

    def recent_literary_reading_titles(self, limit: int = 5) -> list[str]:
        """直近に開始/進行した読書セッションの作品名（LLM 選書のヒント用・§10.2）。"""
        with self._lock:
            cur = self._conn.execute(
                "SELECT title FROM literary_reading_sessions "
                "ORDER BY COALESCE(updated_at, started_at) DESC LIMIT ?",
                (int(limit),),
            )
            return [r[0] for r in cur.fetchall() if r[0]]

    def record_literary_reading_log(
        self, work_id: str, chunk_index: int | None, kind: str, payload: str | None = None
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO literary_reading_log (work_id, chunk_index, kind, payload) VALUES (?, ?, ?, ?)",
                (work_id, chunk_index, kind, payload),
            )
            self._conn.commit()

    # --- 翻訳朗読コーナー（Project Gutenberg → 翻訳ナレーション）--------

    def load_translated_reading_session(self, work_id: str) -> dict | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT work_id, title, author, cursor, total_chunks, rolling_summary, finished_at "
                "FROM translated_reading_sessions WHERE work_id = ?",
                (work_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return {
            "work_id": row[0],
            "title": row[1],
            "author": row[2],
            "cursor": row[3],
            "total_chunks": row[4],
            "rolling_summary": row[5] or "",
            "finished_at": row[6],
        }

    def latest_unfinished_translated_reading_session(self) -> dict | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT work_id FROM translated_reading_sessions WHERE finished_at IS NULL "
                "ORDER BY COALESCE(updated_at, started_at) DESC LIMIT 1"
            )
            row = cur.fetchone()
        return self.load_translated_reading_session(row[0]) if row else None

    def open_translated_reading_session(
        self, work_id: str, title: str, author: str, total_chunks: int
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO translated_reading_sessions (work_id, title, author, total_chunks) "
                "VALUES (?, ?, ?, ?)",
                (work_id, title, author, total_chunks),
            )
            self._conn.commit()

    def advance_translated_reading_cursor(self, work_id: str, cursor: int) -> None:
        """翻訳ナレーションを再生開始した時点で呼ぶ（合成完了時ではない）。後退はさせない。"""
        with self._lock:
            self._conn.execute(
                "UPDATE translated_reading_sessions SET cursor = MAX(cursor, ?), "
                "updated_at = CURRENT_TIMESTAMP WHERE work_id = ?",
                (cursor, work_id),
            )
            self._conn.commit()

    def save_translated_reading_summary(self, work_id: str, rolling_summary: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE translated_reading_sessions SET rolling_summary = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE work_id = ?",
                (rolling_summary, work_id),
            )
            self._conn.commit()

    def finish_translated_reading_session(self, work_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE translated_reading_sessions SET finished_at = CURRENT_TIMESTAMP WHERE work_id = ?",
                (work_id,),
            )
            self._conn.commit()

    def recently_read_translated_reading_work_ids(self, days: int) -> set[str]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT work_id FROM translated_reading_sessions WHERE finished_at IS NOT NULL "
                "AND finished_at >= datetime('now', ?)",
                (f"-{int(days)} days",),
            )
            return {r[0] for r in cur.fetchall()}

    def recent_translated_reading_titles(self, limit: int = 5) -> list[str]:
        """直近に開始/進行した翻訳朗読セッションの作品名（LLM 選書のヒント用）。"""
        with self._lock:
            cur = self._conn.execute(
                "SELECT title FROM translated_reading_sessions "
                "ORDER BY COALESCE(updated_at, started_at) DESC LIMIT ?",
                (int(limit),),
            )
            return [r[0] for r in cur.fetchall() if r[0]]

    def record_translated_reading_log(
        self, work_id: str, chunk_index: int | None, kind: str, payload: str | None = None
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO translated_reading_log (work_id, chunk_index, kind, payload) VALUES (?, ?, ?, ?)",
                (work_id, chunk_index, kind, payload),
            )
            self._conn.commit()

    # --- 偉人伝トーク（Wikipedia）----------------------------------------

    def load_biography_reading_session(self, figure_id: str) -> dict | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT figure_id, title, cursor, total_chunks, rolling_summary, finished_at "
                "FROM biography_reading_sessions WHERE figure_id = ?",
                (figure_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return {
            "figure_id": row[0],
            "title": row[1],
            "cursor": row[2],
            "total_chunks": row[3],
            "rolling_summary": row[4] or "",
            "finished_at": row[5],
        }

    def latest_unfinished_biography_reading_session(self) -> dict | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT figure_id FROM biography_reading_sessions WHERE finished_at IS NULL "
                "ORDER BY COALESCE(updated_at, started_at) DESC LIMIT 1"
            )
            row = cur.fetchone()
        return self.load_biography_reading_session(row[0]) if row else None

    def open_biography_reading_session(self, figure_id: str, title: str, total_chunks: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO biography_reading_sessions (figure_id, title, total_chunks) "
                "VALUES (?, ?, ?)",
                (figure_id, title, total_chunks),
            )
            self._conn.commit()

    def advance_biography_reading_cursor(self, figure_id: str, cursor: int) -> None:
        """再生開始した時点で呼ぶ（合成完了時ではない）。後退はさせない。"""
        with self._lock:
            self._conn.execute(
                "UPDATE biography_reading_sessions SET cursor = MAX(cursor, ?), "
                "updated_at = CURRENT_TIMESTAMP WHERE figure_id = ?",
                (cursor, figure_id),
            )
            self._conn.commit()

    def save_biography_reading_summary(self, figure_id: str, rolling_summary: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE biography_reading_sessions SET rolling_summary = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE figure_id = ?",
                (rolling_summary, figure_id),
            )
            self._conn.commit()

    def finish_biography_reading_session(self, figure_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE biography_reading_sessions SET finished_at = CURRENT_TIMESTAMP WHERE figure_id = ?",
                (figure_id,),
            )
            self._conn.commit()

    def recently_read_biography_figure_ids(self, days: int) -> set[str]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT figure_id FROM biography_reading_sessions WHERE finished_at IS NOT NULL "
                "AND finished_at >= datetime('now', ?)",
                (f"-{int(days)} days",),
            )
            return {r[0] for r in cur.fetchall()}

    def record_biography_reading_log(
        self, figure_id: str, chunk_index: int | None, payload: str | None = None
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO biography_reading_log (figure_id, chunk_index, payload) VALUES (?, ?, ?)",
                (figure_id, chunk_index, payload),
            )
            self._conn.commit()

    # --- ラジオドラマ朗読（v6 §4.7）------------------------------------------
    #
    # 執筆側（generated_drama_writer）が generated_dramas / generated_drama_scenes を書き、
    # 放送側は読むだけ。放送側が書くのは generated_drama_progress のみ（放送済み位置の記録）。

    def create_generated_drama(
        self, title: str, premise: str = "", priority: int = 0
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO generated_dramas (title, premise, priority) VALUES (?, ?, ?)",
                (title, premise, int(priority)),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def get_generated_drama(self, generated_drama_id: int) -> dict | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, title, premise, priority, status FROM generated_dramas WHERE id = ?",
                (int(generated_drama_id),),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return {
            "id": row[0], "title": row[1], "premise": row[2] or "",
            "priority": row[3], "status": row[4],
        }

    def list_generated_dramas(self, status: str | None = None) -> list[dict]:
        """優先度の高い順に小説を並べて返す（run --auto の順番）。"""
        sql = "SELECT id, title, premise, priority, status FROM generated_dramas"
        params: tuple = ()
        if status is not None:
            sql += " WHERE status = ?"
            params = (status,)
        sql += " ORDER BY priority DESC, id"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [
            {"id": r[0], "title": r[1], "premise": r[2] or "", "priority": r[3], "status": r[4]}
            for r in rows
        ]

    def set_generated_drama_status(self, generated_drama_id: int, status: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE generated_dramas SET status = ? WHERE id = ?", (status, int(generated_drama_id))
            )
            self._conn.commit()

    def save_generated_drama_scene(
        self,
        generated_drama_id: int,
        chapter: int,
        scene: int,
        body: str,
        summary: str = "",
        ready: bool = False,
    ) -> int:
        """1シーンを書き込む（同じ (chapter, scene) は上書き）。scene_id を返す。

        書き直すたびに ``attempts`` が増える。呼び出し側はこれを見て
        「何度書き直しても通らないシーン」を打ち切る（§4.7.1）。
        """
        with self._lock:
            self._conn.execute(
                "INSERT INTO generated_drama_scenes "
                "(generated_drama_id, chapter, scene, body, summary, ready, attempts) "
                "VALUES (?, ?, ?, ?, ?, ?, 1) "
                "ON CONFLICT(generated_drama_id, chapter, scene) DO UPDATE SET "
                "body = excluded.body, summary = excluded.summary, ready = excluded.ready, "
                "attempts = generated_drama_scenes.attempts + 1",
                (int(generated_drama_id), int(chapter), int(scene), body, summary, 1 if ready else 0),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT id FROM generated_drama_scenes "
                "WHERE generated_drama_id = ? AND chapter = ? AND scene = ?",
                (int(generated_drama_id), int(chapter), int(scene)),
            ).fetchone()
        return int(row[0])

    def get_generated_drama_scene(self, generated_drama_id: int, chapter: int, scene: int) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, chapter, scene, body, summary, ready, attempts FROM generated_drama_scenes "
                "WHERE generated_drama_id = ? AND chapter = ? AND scene = ?",
                (int(generated_drama_id), int(chapter), int(scene)),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0], "generated_drama_id": generated_drama_id, "chapter": row[1], "scene": row[2],
            "body": row[3], "summary": row[4] or "", "ready": bool(row[5]),
            "attempts": row[6],
        }

    def generated_drama_scenes(self, generated_drama_id: int) -> list[dict]:
        """その小説の全シーンを (chapter, scene) 順に返す（本文は含めない）。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, chapter, scene, summary, ready, attempts FROM generated_drama_scenes "
                "WHERE generated_drama_id = ? ORDER BY chapter, scene",
                (int(generated_drama_id),),
            ).fetchall()
        return [
            {
                "id": r[0], "chapter": r[1], "scene": r[2], "summary": r[3] or "",
                "ready": bool(r[4]), "attempts": r[5],
            }
            for r in rows
        ]

    def recent_generated_drama_summaries(self, generated_drama_id: int, limit: int = 2) -> list[str]:
        """直前に書いたシーンの要約（新しい順→古い順に直して返す）。執筆時の文脈注入用。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT summary FROM generated_drama_scenes WHERE generated_drama_id = ? "
                "AND summary IS NOT NULL AND summary != '' ORDER BY chapter DESC, scene DESC LIMIT ?",
                (int(generated_drama_id), int(limit)),
            ).fetchall()
        return [r[0] for r in reversed(rows)]

    def get_next_ready_scene(self) -> dict | None:
        """放送してよい次のシーン（ready=1 で、まだ最後まで流していない最小のもの）。

        §4.7.2 の ``get_next_ready_scene()``。優先度の高い小説から、章・シーンの
        小さい順に1件だけ返す。途中まで流したシーンは ``chunk_cursor`` から再開する。
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT s.id, s.generated_drama_id, s.chapter, s.scene, s.body, n.title, "
                "       COALESCE(p.chunk_cursor, 0) "
                "FROM generated_drama_scenes s "
                "JOIN generated_dramas n ON n.id = s.generated_drama_id "
                "LEFT JOIN generated_drama_progress p ON p.scene_id = s.id "
                "WHERE s.ready = 1 AND n.status != 'paused' "
                "      AND (p.scene_id IS NULL OR p.finished_at IS NULL) "
                "ORDER BY n.priority DESC, s.generated_drama_id, s.chapter, s.scene LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0], "generated_drama_id": row[1], "chapter": row[2], "scene": row[3],
            "body": row[4], "generated_drama_title": row[5], "chunk_cursor": row[6],
        }

    def count_broadcastable_scenes(self) -> int:
        """まだ最後まで放送していない ready 済みシーンの本数（§4.7.6 の在庫指標）。

        ``GeneratedDramaSupervisorThread`` が「執筆をどれだけ急ぐか」を決めるのに使う。
        条件は ``get_next_ready_scene()`` と同じ（ready=1・小説が paused でない・未読了）。
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM generated_drama_scenes s "
                "JOIN generated_dramas n ON n.id = s.generated_drama_id "
                "LEFT JOIN generated_drama_progress p ON p.scene_id = s.id "
                "WHERE s.ready = 1 AND n.status != 'paused' "
                "      AND (p.scene_id IS NULL OR p.finished_at IS NULL)"
            ).fetchone()
        return int(row[0]) if row else 0

    def advance_generated_drama_progress(
        self, scene_id: int, chunk_cursor: int, finished: bool = False
    ) -> None:
        """朗読チャンクを**再生開始した時点**で呼ぶ（合成完了時ではない・§4.7.2）。

        ``chunk_cursor`` は「次に読むチャンク番号」。後退はさせない。
        ``finished`` はそのシーンの最終チャンクが流れ始めたとき。
        """
        finished_at = "CURRENT_TIMESTAMP" if finished else "NULL"
        with self._lock:
            self._conn.execute(
                "INSERT INTO generated_drama_progress (scene_id, chunk_cursor, finished_at) "
                f"VALUES (?, ?, {finished_at}) "
                "ON CONFLICT(scene_id) DO UPDATE SET "
                "chunk_cursor = MAX(generated_drama_progress.chunk_cursor, excluded.chunk_cursor), "
                "finished_at = COALESCE(generated_drama_progress.finished_at, excluded.finished_at)",
                (int(scene_id), int(chunk_cursor)),
            )
            self._conn.commit()

    def reset_generated_drama_progress(self, scene_id: int) -> None:
        """指定シーンの放送進捗を消す。書き直し後は必ず先頭から読ませる。"""
        with self._lock:
            self._conn.execute(
                "DELETE FROM generated_drama_progress WHERE scene_id = ?", (int(scene_id),)
            )
            self._conn.commit()

    def reset_generated_drama_progress_for_drama(self, generated_drama_id: int) -> int:
        """その小説の全シーンの放送進捗を消す（「頭から読み直す」リクエスト用）。

        放送プロセスは ``get_next_ready_scene()`` で毎回 DB を見に行くだけなので、
        ここを書き換えるだけで、次にシーンを選ぶ瞬間（今読んでいる分が終わり次第、
        かつ番組表が generated_drama の時間帯のとき）に自然と先頭から再選定される。
        戻り値はリセットした行数。
        """
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM generated_drama_progress WHERE scene_id IN "
                "(SELECT id FROM generated_drama_scenes WHERE generated_drama_id = ?)",
                (int(generated_drama_id),),
            )
            self._conn.commit()
            return cur.rowcount

    # --- リクエスト（番組表への一時オーバーライド）— §6.2 -------------------
    #
    # 放送プロセスは schedule.active_content() のたびにここを見に来る（1秒キャッシュ付き）。
    # request CLI 側は行を1本足すだけで、放送プロセスへの IPC は要らない
    # ―― generated-drama seek と同じ「DB を書き換えれば次の判断に効く」流儀。

    @staticmethod
    def _request_row(row: tuple) -> dict:
        return {
            "id": row[0],
            "content_index": row[1],
            "content_type": row[2],
            "label": row[3] or row[2],
            "created_at": row[4],
            "expires_at": row[5],
            "announced_at": row[6],
            "entered_at": row[7],
        }

    _REQUEST_COLS = (
        "id, content_index, content_type, label, created_at, expires_at, "
        "announced_at, entered_at"
    )

    def put_request(
        self, content_index: int, content_type: str, label: str, ttl_sec: float
    ) -> dict:
        """リクエストを1本立てる。既存の生きたリクエストはその場で失効させる。

        「今いちばん新しい我儘」だけが通る（重ねて2つ有効にはしない）。戻り値は
        立てたリクエストの行。
        """
        now = time.time()
        with self._lock:
            # 「失効させてから立てる」を1トランザクションにする。自動コミットなので
            # ここだけ明示的に囲む（囲まないと、その隙間に有効なリクエストが
            # 0本の瞬間ができる）。
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "UPDATE requests SET expires_at = ? WHERE expires_at > ?", (now, now)
                )
                cur = self._conn.execute(
                    "INSERT INTO requests "
                    "(content_index, content_type, label, created_at, expires_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (int(content_index), content_type, label, now, now + float(ttl_sec)),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            row = self._conn.execute(
                f"SELECT {self._REQUEST_COLS} FROM requests WHERE id = ?", (cur.lastrowid,)
            ).fetchone()
        return self._request_row(row)

    def active_request(self) -> dict | None:
        """今有効なリクエスト（期限内で最新の1本）。無ければ None。"""
        with self._lock:
            row = self._conn.execute(
                f"SELECT {self._REQUEST_COLS} FROM requests "
                "WHERE expires_at > ? ORDER BY id DESC LIMIT 1",
                (time.time(),),
            ).fetchone()
        return self._request_row(row) if row is not None else None

    def active_request_target(self) -> tuple[int, str] | None:
        """受け付け済みのリクエストの (content_index, content_type)。schedule へ渡す用。

        「受け付け済み」＝番組進行（director.py）が拾って announced_at を刻んだこと。
        まだなら None（番組表どおり）。番組進行は受け付けと同時にそのリクエストを
        理由とした切り替えを始めるので、受けアナウンス抜きで切り替わる隙はできない。
        実際の切り替え（暗転・局の差し替え・ネタ収集）は番組進行が段取りを踏んで行う。
        """
        req = self.active_request()
        if req is None or req["announced_at"] is None:
            return None
        return (req["content_index"], req["content_type"])

    def pending_request_announcement(self) -> dict | None:
        """まだ読み上げていない、有効なリクエスト。無ければ None。"""
        req = self.active_request()
        return req if req is not None and req["announced_at"] is None else None

    def mark_request_announced(self, request_id: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE requests SET announced_at = ? WHERE id = ?",
                (time.time(), int(request_id)),
            )
            self._conn.commit()

    def mark_request_entered(self, request_id: int) -> None:
        """受けアナウンスの最後の行が再生され始めた（＝コーナーへ入った）記録。"""
        with self._lock:
            self._conn.execute(
                "UPDATE requests SET entered_at = ? WHERE id = ?",
                (time.time(), int(request_id)),
            )
            self._conn.commit()

    def clear_requests(self) -> int:
        """生きているリクエストを全部失効させる。戻り値は失効させた本数。

        行は消さずに期限を今にする（いつ何をリクエストしたかの履歴は残す）。
        """
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "UPDATE requests SET expires_at = ? WHERE expires_at > ?", (now, now)
            )
            self._conn.commit()
            return cur.rowcount

    def is_rebroadcastable(self, source: str, external_id: str) -> bool:
        """7日以上前に放送済み、または未放送なら True。"""
        with self._lock:
            cur = self._conn.execute(
                "SELECT used_at FROM topics WHERE source = ? AND external_id = ?",
                (source, external_id),
            )
            row = cur.fetchone()
            if row is None or row[0] is None:
                return True
            used_at = datetime.datetime.fromisoformat(row[0])
            now_utc = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
            age = now_utc - used_at
            return age.days >= self._rebroadcast_after_days
