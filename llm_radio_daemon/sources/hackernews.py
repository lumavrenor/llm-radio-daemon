"""Hacker News（Firebase API）ソース（v3）。

SPEC.md 4.1(c): newstories から新着記事を拾う。ポーリング間隔は
最低60秒を守る。個別記事の取得失敗は握りつぶして次の記事へ進む。
"""

from __future__ import annotations

import logging
import threading
from typing import Iterator

import requests

from ..sensitive import looks_sensitive
from ..source_status import SourceStatus
from . import USER_AGENT, Topic
from .rss import _decoded_html, extract_article_body

logger = logging.getLogger(__name__)

BASE_URL = "https://hacker-news.firebaseio.com/v0"
_MAX_ITEMS_PER_POLL = 10  # [[content]] max_items_per_poll 未指定時の既定
_MAX_BODY_CHARS = 2000


class HackerNewsSource:
    name = "hackernews"

    def __init__(
        self,
        poll_interval_sec: float = 90.0,
        stop_event: threading.Event | None = None,
        timeout_sec: int = 10,
        status: SourceStatus | None = None,
        max_items_per_poll: int = _MAX_ITEMS_PER_POLL,
        fetch_article_body: bool = False,
    ):
        self._base_poll_interval_sec = max(poll_interval_sec, 60.0)
        self._stop_event = stop_event or threading.Event()
        self._timeout_sec = timeout_sec
        self._status = status
        # 1周で newstories の先頭から何件見るか（[[content]] max_items_per_poll）。
        self._max_items_per_poll = max(1, int(max_items_per_poll))
        # リンク投稿（大半）は item に本文が無いので、リンク先ページから本文を抜く
        # かどうか（[[content]] fetch_article_body）。rss.py と同じ抽出器を使う。
        # 既定 off（重い・壊れやすい。rss.py と同じ理由）。
        self._fetch_article_body = fetch_article_body

    def fetch(self) -> Iterator[Topic]:
        headers = {"User-Agent": USER_AGENT}
        while not self._stop_event.is_set():
            items = 0
            try:
                resp = requests.get(
                    f"{BASE_URL}/newstories.json", headers=headers, timeout=self._timeout_sec
                )
                resp.raise_for_status()
                story_ids = resp.json()[: self._max_items_per_poll]
                for story_id in story_ids:
                    if self._stop_event.is_set():
                        return
                    topic = self._fetch_item(story_id, headers)
                    if topic is not None:
                        items += 1
                        yield topic
            except requests.RequestException as e:
                if self._status is not None:
                    self._status.fetch_failed(f"{BASE_URL}/newstories.json", e)
                else:
                    logger.warning("hackernews poll failed: %s", e)

            if self._status is not None:
                self._status.cycle_end(items, self._base_poll_interval_sec)
            if self._stop_event.wait(self._base_poll_interval_sec):
                return

    def _fetch_item(self, story_id: int, headers: dict) -> Topic | None:
        try:
            resp = requests.get(
                f"{BASE_URL}/item/{story_id}.json", headers=headers, timeout=self._timeout_sec
            )
            resp.raise_for_status()
            item = resp.json()
        except requests.RequestException as e:
            logger.warning("hackernews item %s fetch failed: %s", story_id, e)
            return None

        if not item or item.get("type") != "story" or not item.get("title"):
            return None

        title = item["title"]
        text = item.get("text", "")

        # 新着には規制・選挙・地政学・宗教まわりの記事も混ざる。話題自体がそちらへ
        # 寄っているものは、台本生成にかける前にここで落とす（供給は十分あるので、
        # 多少取りこぼしても番組は止まらない）。
        if looks_sensitive(title, text):
            logger.info("skipping sensitive hackernews story: %s", title)
            return None

        link_url = item.get("url")
        url = link_url or f"https://news.ycombinator.com/item?id={story_id}"
        body = text

        if not body and link_url and self._fetch_article_body:
            page_html = self._get_page(link_url, headers)
            body = extract_article_body(page_html, link_url, _MAX_BODY_CHARS)
            # リンク先の本文で初めて分かる分（タイトルだけでは無害に見えた記事）も
            # 同じフィルタにかける。
            if body and looks_sensitive(title, body):
                logger.info("skipping sensitive hackernews story (body): %s", title)
                return None

        if not body:
            body = f"(リンク記事。本文は取得していない: {url})"

        return Topic(
            source=self.name,
            external_id=str(story_id),
            title=title,
            body=body[:_MAX_BODY_CHARS],
            url=url,
            hint="技術系の新しい話題",
        )

    def _get_page(self, url: str, headers: dict) -> str:
        """リンク先の記事ページを1回だけ GET して HTML 文字列を返す（失敗は ""）。"""
        try:
            resp = requests.get(url, headers=headers, timeout=min(self._timeout_sec, 8))
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.debug("hackernews: failed to fetch linked page %s: %s", url, e)
            return ""
        return _decoded_html(resp)
