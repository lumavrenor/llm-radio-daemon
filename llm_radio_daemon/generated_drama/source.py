"""放送側の薄い Source（v6 §4.7.2）。

生成済み・未放送のシーンを DB から 1 件返すだけ。**生成は一切行わない**
（執筆は別プロセスの ``generated_drama_writer``）。

「小説が仕上がっていなければ代替コンテンツを流す」というフォールバック専用の
ロジックはここには持たせない。``fetch()`` が空を返せば、番組表（§6）と
フィラー階層（§8）がそのまま面倒を見る。

単体確認（§9 の実装順 4）:
    python -m llm_radio_daemon.generated_drama.source --config config/<lang>/config.toml
"""

from __future__ import annotations

import argparse
import logging
from typing import Iterator

from ..db import TopicStore
from ..sources import Topic
from . import GeneratedDramaScene

logger = logging.getLogger(__name__)


class GeneratedDramaSource:
    name = "generated_drama"

    def __init__(self, store: TopicStore):
        self._store = store

    def next_scene(self) -> GeneratedDramaScene | None:
        """生成済み・未放送の次シーン。無ければ None。"""
        row = self._store.get_next_ready_scene()
        if row is None:
            return None
        return GeneratedDramaScene(
            scene_id=row["id"],
            generated_drama_id=row["generated_drama_id"],
            generated_drama_title=row["generated_drama_title"],
            chapter=row["chapter"],
            scene=row["scene"],
            body=row["body"],
            chunk_cursor=row["chunk_cursor"],
        )

    def fetch(self) -> Iterator[Topic]:
        """§4.7.2 の Source インターフェース。

        返す ``Topic`` は **本文そのもの（確定済みテキスト）** を ``body`` に持ち、
        ``hint`` は使わない（後段の ScriptThread を通らないため）。
        """
        scene = self.next_scene()
        if scene is None:
            return
        yield self._scene_to_topic(scene)

    @staticmethod
    def _scene_to_topic(scene: GeneratedDramaScene) -> Topic:
        return Topic(
            source="generated_drama",
            external_id=f"{scene.generated_drama_id}:{scene.chapter}:{scene.scene}",
            title=f"{scene.generated_drama_title} {scene.label}",
            body=scene.body,
            url=None,
            hint="",
        )


def _main() -> int:
    from ..config import CONFIG_ARG_HELP, load_config

    ap = argparse.ArgumentParser(description="GeneratedDramaSource.fetch() の単体確認")
    ap.add_argument("--config", required=True, help=CONFIG_ARG_HELP)
    args = ap.parse_args()

    config = load_config(args.config)
    store = TopicStore(config.db.path, config.db.rebroadcast_after_days)
    try:
        topics = list(GeneratedDramaSource(store).fetch())
        if not topics:
            print("No broadcastable scenes available (fetch() returned empty).")
            print("-> Falling back to the next schedule candidate/filler is expected behavior.")
            print("  To write one: python -m llm_radio_daemon.generated_drama_writer run --auto")
            return 0
        t = topics[0]
        print(f"external_id: {t.external_id}")
        print(f"title      : {t.title}")
        print(f"body ({len(t.body)} chars):\n{t.body[:600]}")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
