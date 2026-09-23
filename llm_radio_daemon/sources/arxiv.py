"""arXiv API ソース（v3）。

SPEC.md 4.1(c): cs.AI / cs.LG などの新着論文を拾う。arXiv APIはAtom形式で
返ってくるため feedparser でパースする。ポーリング間隔は最低60秒。
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Iterator

import feedparser
import requests

from ..source_status import SourceStatus
from . import USER_AGENT, Topic

logger = logging.getLogger(__name__)

QUERY_URL = "http://export.arxiv.org/api/query"
_WHITESPACE_RE = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text or "").strip()


class ArxivSource:
    name = "arxiv"

    def __init__(
        self,
        categories: tuple[str, ...] = ("cs.AI", "cs.LG"),
        max_results: int = 10,
        poll_interval_sec: float = 300.0,
        stop_event: threading.Event | None = None,
        timeout_sec: int = 10,
        status: SourceStatus | None = None,
    ):
        self._search_query = " OR ".join(f"cat:{c}" for c in categories)
        self._max_results = max_results
        self._base_poll_interval_sec = max(poll_interval_sec, 60.0)
        self._stop_event = stop_event or threading.Event()
        self._timeout_sec = timeout_sec
        self._status = status

    def fetch(self) -> Iterator[Topic]:
        headers = {"User-Agent": USER_AGENT}
        params = {
            "search_query": self._search_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": self._max_results,
        }
        while not self._stop_event.is_set():
            entries = 0
            try:
                resp = requests.get(
                    QUERY_URL, params=params, headers=headers, timeout=self._timeout_sec
                )
                resp.raise_for_status()
                feed = feedparser.parse(resp.content)
                for entry in feed.entries:
                    if self._stop_event.is_set():
                        return
                    topic = self._to_topic(entry)
                    if topic is not None:
                        entries += 1
                        yield topic
            except requests.RequestException as e:
                if self._status is not None:
                    self._status.fetch_failed(QUERY_URL, e)
                else:
                    logger.warning("arxiv poll failed: %s", e)

            if self._status is not None:
                self._status.cycle_end(entries, self._base_poll_interval_sec)
            if self._stop_event.wait(self._base_poll_interval_sec):
                return

    def _to_topic(self, entry) -> Topic | None:
        entry_id = entry.get("id")
        title = _clean(entry.get("title", ""))
        summary = _clean(entry.get("summary", ""))
        if not entry_id or not title:
            return None

        return Topic(
            source=self.name,
            external_id=entry_id,
            title=title,
            body=summary[:2000],
            url=entry.get("link", entry_id),
            hint="新しい研究論文",
        )
