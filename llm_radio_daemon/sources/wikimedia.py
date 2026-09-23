"""Wikimedia EventStreams（recentchange）ソース。v1 の主力ネタ源。

SSE で毎秒数十件流れてくる編集イベントから、日本語/英語版・本文名前空間・
非botの編集だけを抜き出し、さらに30秒に1件だけサンプリングして
Wikipedia REST API の要約を添えて Topic 化する。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Iterator

import requests

from ..sensitive import looks_sensitive
from . import USER_AGENT, Topic

logger = logging.getLogger(__name__)

STREAM_URL = "https://stream.wikimedia.org/v2/stream/recentchange"
SUMMARY_URL_TMPL = "https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"

# 対応している Wikipedia の版（wiki 名 → 言語コード）。ここに無い版は捨てる。
ALLOWED_WIKIS = {"jawiki": "ja", "enwiki": "en"}


class WikimediaSource:
    name = "wikimedia"

    def __init__(
        self,
        sample_interval_sec: float = 30.0,
        timeout_sec: int = 10,
        langs: list[str] | None = None,
    ):
        """``langs`` は拾う Wikipedia の言語版（例: ``["en"]``）。

        以前は ALLOWED_WIKIS 全部（ja と en の両方）を無条件に拾っていたため、
        英語放送でも日本語の記事が流れてきていた。呼び出し側（main.py）は
        [[content]] の ``wikis`` か、無ければ ``[locale] lang`` を渡す。
        ``None`` なら従来どおり全部拾う。
        """
        self._sample_interval_sec = sample_interval_sec
        self._timeout_sec = timeout_sec
        if langs:
            wanted = {str(l).strip().lower() for l in langs}
            self._wikis = {w: l for w, l in ALLOWED_WIKIS.items() if l in wanted}
            if not self._wikis:
                logger.warning(
                    "wikimedia: unsupported language(s) specified (%s). Falling back to all languages "
                    "(supported: %s)", sorted(wanted), sorted(set(ALLOWED_WIKIS.values())),
                )
                self._wikis = dict(ALLOWED_WIKIS)
        else:
            self._wikis = dict(ALLOWED_WIKIS)

    def fetch(self) -> Iterator[Topic]:
        """SSEに接続してフィルタ条件に合う編集を30秒に1件だけyieldする。

        接続断はそのまま例外として送出する。再接続は呼び出し側
        （SourceThread）が try/except でループしながら行う。
        """
        last_emit = 0.0
        headers = {"User-Agent": USER_AGENT, "Accept": "text/event-stream"}
        with requests.get(
            STREAM_URL, headers=headers, stream=True, timeout=(self._timeout_sec, None)
        ) as resp:
            resp.raise_for_status()
            data_lines: list[str] = []
            for raw_line in resp.iter_lines(decode_unicode=True):
                if raw_line is None:
                    continue
                line = raw_line.rstrip("\n")
                if line == "":
                    if data_lines:
                        payload = "\n".join(data_lines)
                        data_lines = []
                        sample_interval = max(self._sample_interval_sec, 10.0)
                        if time.monotonic() - last_emit >= sample_interval:
                            topic = self._handle_event(payload)
                            if topic is not None:
                                last_emit = time.monotonic()
                                yield topic
                    continue
                if line.startswith(":"):
                    continue  # コメント行（keep-alive）
                if line.startswith("data:"):
                    data_lines.append(line[len("data:") :].strip())
                # event: / id: 等は無視

    def _handle_event(self, payload: str) -> Topic | None:
        try:
            change = json.loads(payload)
        except json.JSONDecodeError:
            return None

        if change.get("type") != "edit":
            return None
        if change.get("bot"):
            return None
        if change.get("namespace") != 0:
            return None
        wiki = change.get("wiki")
        lang = self._wikis.get(wiki)
        if lang is None:
            return None

        title = change.get("title")
        if not title:
            return None

        data = self._fetch_summary(lang, title)
        if data is None:
            return None
        summary = data.get("extract")
        if not summary:
            return None

        # ランダムサンプリングなので、政治家・政党・宗教・戦争などの記事を直に
        # 引くことがある。話題そのものがそちらへ寄っている記事は、台本生成に
        # かける前にここで丸ごと落とす（description は "American politician" の
        # ように短く分類的で、判定が効きやすい）。
        if looks_sensitive(title, data.get("description"), summary):
            logger.info("skipping sensitive wikipedia topic: %s", title)
            return None

        url = f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"
        return Topic(
            source=self.name,
            external_id=url,
            title=title,
            body=summary[:2000],
            url=url,
            # 「たった今誰かが編集した」はこの記事をサンプリングした“理由”であって、
            # 話す内容ではない。hintをそう書くとLLMが毎回「編集の仕組み」
            # 「リアルタイム書き換え」の話に収束してしまうため、hintには含めない
            # （記事の中身を話させる）。
            hint="雑学のネタ",
        )

    def _fetch_summary(self, lang: str, title: str) -> dict | None:
        """REST summary をそのまま返す（extract / description を呼び出し側で使う）。"""
        url = SUMMARY_URL_TMPL.format(lang=lang, title=title)
        headers = {"User-Agent": USER_AGENT}
        try:
            resp = requests.get(url, headers=headers, timeout=self._timeout_sec)
            if resp.status_code != 200:
                return None
            return resp.json()
        except requests.RequestException as e:
            logger.warning("wikipedia summary fetch failed for %s: %s", title, e)
            return None
