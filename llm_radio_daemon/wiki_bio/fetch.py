"""Wikipedia記事本文の取得とローカルキャッシュ（偉人伝トーク）。

対象人物は figures_file に明示指定されたものだけを使う（09-05相談。
「政治・宗教に無関係・1950年より前」という制約はLLM任せの自動選定では守り切れない
ため、選定そのものをconfigの明示リストに閉じる方針）。青空文庫のような候補カタログは
持たず、1人物 = 1回の GET で本文を取得して以降はキャッシュから読むだけの薄い層。
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import requests

from ..sources import USER_AGENT

logger = logging.getLogger(__name__)

API_URL_TMPL = "https://{lang}.wikipedia.org/w/api.php"
TIMEOUT = (10, 30)


@dataclass(frozen=True)
class FigureMeta:
    figure_id: str
    title: str    # 記事タイトル
    lang: str
    path: Path    # 本文キャッシュのパス


class WikiBioCorpus:
    """``data_dir`` 配下に人物ごとの本文（.txt）とメタ（.json）をキャッシュする。"""

    def __init__(self, data_dir: str | Path):
        self._dir = Path(data_dir)
        self._lock = threading.Lock()

    def _text_path(self, figure_id: str) -> Path:
        return self._dir / f"{figure_id}.txt"

    def _meta_path(self, figure_id: str) -> Path:
        return self._dir / f"{figure_id}.json"

    def get_cached(self, figure_id: str) -> FigureMeta | None:
        meta_path = self._meta_path(figure_id)
        text_path = self._text_path(figure_id)
        if not (meta_path.exists() and text_path.exists()):
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("wiki_bio: metadata for figure_id=%s is corrupted; re-fetching", figure_id)
            return None
        return FigureMeta(
            figure_id, meta.get("title", figure_id), meta.get("lang", "ja"), text_path
        )

    def ensure_text(self, figure_id: str, wiki_title: str, lang: str) -> FigureMeta | None:
        """取得済みならそのまま返す。未取得ならその場で取得してキャッシュする（遅延取得）。"""
        cached = self.get_cached(figure_id)
        if cached is not None:
            return cached
        with self._lock:
            cached = self.get_cached(figure_id)
            if cached is not None:
                return cached
            text = fetch_article_text(wiki_title, lang)
            if text is None:
                return None
            self._dir.mkdir(parents=True, exist_ok=True)
            self._text_path(figure_id).write_text(text, encoding="utf-8")
            self._meta_path(figure_id).write_text(
                json.dumps({"title": wiki_title, "lang": lang}, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("wiki_bio: fetched figure_id=%s \"%s\" (%d chars)", figure_id, wiki_title, len(text))
            return self.get_cached(figure_id)

    @staticmethod
    def load_text(meta: FigureMeta) -> str:
        return meta.path.read_text(encoding="utf-8", errors="replace")


def fetch_article_text(title: str, lang: str = "ja") -> str | None:
    """MediaWiki Action API から記事本文のプレーンテキストを取得する（ネットワークI/O）。

    ``explaintext=1`` で wikitext 記法を展開済みのプレーンテキストにし、
    ``exsectionformat=wiki`` で見出しを "== 節名 ==" のまま残す
    （parser.py がここから節境界を検出する）。失敗時は None。
    """
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "exsectionformat": "wiki",
        "redirects": 1,
        "titles": title,
        "format": "json",
        "formatversion": 2,
    }
    url = API_URL_TMPL.format(lang=lang)
    try:
        resp = requests.get(
            url, params=params, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        logger.warning("wiki_bio: failed to fetch article title=%r lang=%s: %s", title, lang, e)
        return None

    pages = data.get("query", {}).get("pages", [])
    if not pages:
        logger.warning("wiki_bio: article not found title=%r lang=%s", title, lang)
        return None
    page = pages[0]
    if page.get("missing"):
        logger.warning("wiki_bio: article does not exist title=%r lang=%s", title, lang)
        return None
    extract = (page.get("extract") or "").strip()
    if not extract:
        logger.warning("wiki_bio: body text is empty title=%r lang=%s", title, lang)
        return None
    return extract
