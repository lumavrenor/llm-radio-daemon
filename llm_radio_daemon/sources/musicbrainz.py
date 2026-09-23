"""ICYメタデータ（曲名）→ MusicBrainz によるネタ化（v3）。

SPEC.md 4.1(b): 曲が変わるたびにMusicBrainzへ問い合わせて情報を取得し、
Topic化する。これが曲替わりごとに自動でネタを供給するので、音楽とトーク
が構造的に噛み合う。呼び出し頻度は曲替わり時のみ（数分に1回程度）なので、
MusicBrainzのレート制限（匿名で1req/秒）は問題にならない。

このソースだけは Source プロトコル（fetch()のポーリングループ）ではなく、
IcyMetadataReader の曲変更コールバックから直接呼ばれる関数として実装する
（曲が変わった、というイベント自体がトリガーであり、ポーリング不要なため）。
"""

from __future__ import annotations

import datetime
import logging
import re
import time

import requests

from . import USER_AGENT, Topic

logger = logging.getLogger(__name__)

SEARCH_URL = "https://musicbrainz.org/ws/2/recording/"
_ARTIST_TITLE_RE = re.compile(r"\s*-\s*")

# MusicBrainz は混雑時・レート制限時に 503 を返す。1回だけ間を置いて試し直す。
# ここは IcyMetadataReader のスレッドから同期で呼ばれるので、待ちは短く抑える。
_RETRY_STATUS = 503
_RETRY_ATTEMPTS = 2
_RETRY_WAIT_SEC = 2.0


def _lookup_recording(artist: str, title: str, timeout_sec: int) -> dict | None:
    """MusicBrainz から録音情報を1件引く。ヒットなし・失敗はどちらも None。

    **戻り値が None でもネタ化は中止しない**（呼び出し側が曲名・アーティストだけで
    Topic を作る）。MusicBrainz は 503 もヒット0件も日常的に起きるため、ここを
    トークの有無のゲートにすると外部サービスの調子がそのまま無音になる。
    """
    query = f'artist:"{artist}" AND recording:"{title}"'
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            resp = requests.get(
                SEARCH_URL,
                params={"query": query, "fmt": "json", "limit": 1},
                headers=headers,
                timeout=timeout_sec,
            )
            if resp.status_code == _RETRY_STATUS and attempt < _RETRY_ATTEMPTS:
                logger.info(
                    "musicbrainz returned 503 (attempt %d/%d). Waiting %.0f seconds before retrying: %s - %s",
                    attempt, _RETRY_ATTEMPTS, _RETRY_WAIT_SEC, artist, title,
                )
                time.sleep(_RETRY_WAIT_SEC)
                continue
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.warning("musicbrainz lookup failed for %r - %r: %s", artist, title, e)
            return None
        except ValueError:
            logger.warning("musicbrainz returned non-JSON response for %r - %r", artist, title)
            return None

        recordings = data.get("recordings") or []
        return recordings[0] if recordings else None
    return None


def topic_for_now_playing(
    raw_title: str, timeout_sec: int = 10, *, back_announce: bool = False
) -> Topic | None:
    """'Artist - Title' 形式のICYメタデータからTopicを作る。

    None を返すのは「ICYメタデータが 'Artist - Title' 形式でない」ときだけ。
    MusicBrainz にヒットしなくても、曲名とアーティストは ICY から分かっているので
    それだけで Topic を作る（MusicBrainz のリリース年・アルバム・タグは
    **あれば付ける追加情報**という位置づけ。§4.1(b)）。

    back_announce=True のときは「直前にかかっていた曲」を振り返る切り口にする
    （曲の切り替わり後に、いま終わった曲について話すモード）。
    """
    parts = _ARTIST_TITLE_RE.split(raw_title, maxsplit=1)
    if len(parts) != 2:
        return None
    artist, title = (p.strip() for p in parts)
    if not artist or not title:
        return None

    rec = _lookup_recording(artist, title, timeout_sec)

    body_lines = [f"曲名: {title}", f"アーティスト: {artist}"]
    mbid = None
    if rec is None:
        logger.info(
            "musicbrainz has no info, building the topic from just the title/artist: %r", raw_title
        )
    else:
        mbid = rec.get("id")
        first_release = rec.get("first-release-date")
        if first_release:
            body_lines.append(f"初出リリース日: {first_release}")
        release_titles = [
            r.get("title") for r in (rec.get("releases") or [])[:3] if r.get("title")
        ]
        if release_titles:
            body_lines.append(f"収録アルバム: {', '.join(release_titles)}")
        tag_names = [t.get("name") for t in (rec.get("tags") or [])[:5] if t.get("name")]
        if tag_names:
            body_lines.append(f"タグ: {', '.join(tag_names)}")

    # external_id は「曲」ではなく「その再生」を指すキーにする。曲IDのままだと
    # topics の UNIQUE(source, external_id) に恒久的に弾かれ、一度紹介した曲は
    # 二度と話せない＝ローテーションが一巡すると無音になる。ラジオの選曲紹介は
    # 流れるたびに成立する催しなので、再生時刻まで含めて別イベントとして扱う。
    played_at = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    return Topic(
        source="musicbrainz",
        external_id=f"{mbid or raw_title}@{played_at}",
        title=f"{artist} - {title}",
        body="\n".join(body_lines)[:2000],
        url=f"https://musicbrainz.org/recording/{mbid}" if mbid else None,
        hint="直前にかかっていた曲について" if back_announce else "今かかっている曲について",
    )
