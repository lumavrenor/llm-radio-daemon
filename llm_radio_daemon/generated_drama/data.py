"""``data/generated_drama_data_<lang>/`` の読み書き（v6 §4.7.1）。

執筆バッチが書き、放送プロセスは読むだけ。放送プロセスが動いている最中でも
安全に書けるよう、**一時ファイルへ書いてから rename する**（読み手が半端な原稿を
掴まない）。SQLite 側の WAL 化は :class:`db.TopicStore` が担当。

    data/generated_drama_data_<lang>/
      001/
        novel.json        全体設計（ログライン・テーマ・章立て）
        characters.json   登場人物（放送側の話者振り分けにも使う）
        world.json        世界観・設定
        summaries.json    章要約・全体要約（チェックフェーズの積み上げ）
        beats/ch01.json   中間展開（ハコ書き／ビートシート）
        scenes/ch01_sc01.txt  本文（DB が正、こちらは人が読む用のミラー）
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

from . import GeneratedDramaCharacter

logger = logging.getLogger(__name__)


def _atomic_write(path: Path, text: str) -> None:
    """同じディレクトリへ一時ファイルを書いてから rename する（§4.7.1）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=path.suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)  # 同一ボリューム内なので原子的に差し替わる
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


class GeneratedDramaData:
    """1本の小説ぶんのファイル置き場。"""

    def __init__(self, data_dir: str | Path, generated_drama_id: int):
        self.root = Path(data_dir) / f"{int(generated_drama_id):03d}"
        self.generated_drama_id = int(generated_drama_id)

    # --- パス -----------------------------------------------------------

    @property
    def novel_path(self) -> Path:
        return self.root / "novel.json"

    @property
    def characters_path(self) -> Path:
        return self.root / "characters.json"

    @property
    def world_path(self) -> Path:
        return self.root / "world.json"

    @property
    def summaries_path(self) -> Path:
        return self.root / "summaries.json"

    def beats_path(self, chapter: int) -> Path:
        return self.root / "beats" / f"ch{int(chapter):02d}.json"

    def scene_path(self, chapter: int, scene: int) -> Path:
        return self.root / "scenes" / f"ch{int(chapter):02d}_sc{int(scene):02d}.txt"

    # --- 読み ------------------------------------------------------------

    def _read_json(self, path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.exception("generated_drama: %s could not be read / is corrupted", path)
            return {}

    def load_novel(self) -> dict:
        return self._read_json(self.novel_path)

    def load_world(self) -> dict:
        return self._read_json(self.world_path)

    def load_summaries(self) -> dict:
        d = self._read_json(self.summaries_path)
        d.setdefault("chapters", {})
        d.setdefault("overall", "")
        return d

    def load_beats(self, chapter: int) -> dict:
        return self._read_json(self.beats_path(chapter))

    def load_characters(self) -> list[GeneratedDramaCharacter]:
        raw = self._read_json(self.characters_path)
        entries = raw.get("characters", []) if isinstance(raw, dict) else raw
        out: list[GeneratedDramaCharacter] = []
        for e in entries or []:
            try:
                c = GeneratedDramaCharacter.from_dict(e)
            except (TypeError, ValueError):
                logger.warning("generated_drama: skipping characters.json entry: %r", e)
                continue
            if c.name:
                out.append(c)
        return out

    # --- 書き ------------------------------------------------------------

    def save_novel(self, data: dict) -> None:
        _atomic_write(self.novel_path, json.dumps(data, ensure_ascii=False, indent=2))

    def save_world(self, data: dict) -> None:
        _atomic_write(self.world_path, json.dumps(data, ensure_ascii=False, indent=2))

    def save_summaries(self, data: dict) -> None:
        _atomic_write(self.summaries_path, json.dumps(data, ensure_ascii=False, indent=2))

    def save_beats(self, chapter: int, data: dict) -> None:
        _atomic_write(
            self.beats_path(chapter), json.dumps(data, ensure_ascii=False, indent=2)
        )

    def save_characters(self, characters: list[GeneratedDramaCharacter]) -> None:
        _atomic_write(
            self.characters_path,
            json.dumps(
                {"characters": [c.to_dict() for c in characters]},
                ensure_ascii=False,
                indent=2,
            ),
        )

    def save_scene_text(self, chapter: int, scene: int, body: str) -> None:
        _atomic_write(self.scene_path(chapter, scene), body.rstrip() + "\n")
