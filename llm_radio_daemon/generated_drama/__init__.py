"""ラジオドラマ朗読（v6 §4.7）。

**執筆（generated_drama_writer）と放送（GeneratedDramaSource / GeneratedDramaCorner）は別プロセス。**
執筆ハーネスが伏線・口調・時系列の一貫性を担保した「完成原稿」を書き溜め、
放送側はそれを ``ScriptThread``（Ollama 生成）を丸ごと迂回して読み上げるだけ。
放送中の LLM 負荷は実質ゼロになる（§4.7.0）。

    [generated_drama_writer]  別プロセス。手動 / タスクスケジューラ
        │ 本文・進行状態を書く
        ↓
    data/generated_drama_data_<lang>/ + SQLite ──→ GeneratedDramaSource ──→ GeneratedDramaCorner ──→ TTSThread
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .. import language

# LLM は aliases に「愛称、呼び捨て（家族のみ）」のような1文字列を詰め込みがちなので、
# 読み込み時に区切って括弧の注釈を落とす。ここを通しておかないと、本文中の呼び名と
# 一致せずセリフの話者判定（§4.7.3）がまるごと空振りする。
_ALIAS_SPLIT = re.compile(r"[、,，/／・|｜]")
_ALIAS_NOTE = re.compile(r"[（(][^）)]*[）)]")
_ALIAS_MAX_CHARS = 12


@dataclass
class GeneratedDramaCharacter:
    """``characters.json`` の1キャラ。執筆側が書き、放送側は声の割り当てに使う。"""

    key: str                       # 参照キー（ASCII 短縮名）
    name: str                      # 本文中に出てくる呼び名
    aliases: list[str] = field(default_factory=list)  # 別称・愛称（セリフの話者判定に使う）
    role: str = ""                 # 主人公 / 相棒 / 敵役 など
    persona: str = ""              # 人物像（執筆時のプロンプトへ注入）
    speech_style: str = ""         # 口調（執筆時のプロンプトへ注入）
    cast_id: str = ""              # [[cast]] の id。放送側の話者振り分けに最優先で使う
    voicevox_speaker: int | None = None   # VOICEVOX 話者ID（§4.7.1）
    voicevox_style: str = ""       # スタイル名（cast 側に同名があれば使う）

    @property
    def names(self) -> list[str]:
        """本文から話者を拾うときに探す呼び名の一覧（長い順）。"""
        out = [n for n in [self.name, *self.aliases] if n]
        return sorted(dict.fromkeys(out), key=len, reverse=True)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "aliases": list(self.aliases),
            "role": self.role,
            "persona": self.persona,
            "speech_style": self.speech_style,
            "cast_id": self.cast_id,
            "voicevox_speaker": self.voicevox_speaker,
            "voicevox_style": self.voicevox_style,
        }

    @staticmethod
    def _clean_aliases(raw) -> list[str]:
        if isinstance(raw, str):
            raw = [raw]
        out: list[str] = []
        for entry in raw or []:
            for piece in _ALIAS_SPLIT.split(str(entry)):
                name = _ALIAS_NOTE.sub("", piece).strip()
                if name and len(name) <= _ALIAS_MAX_CHARS:
                    out.append(name)
        return list(dict.fromkeys(out))

    @classmethod
    def from_dict(cls, d: dict) -> "GeneratedDramaCharacter":
        return cls(
            key=str(d.get("key") or d.get("name") or "").strip(),
            name=str(d.get("name") or "").strip(),
            aliases=cls._clean_aliases(d.get("aliases", [])),
            role=str(d.get("role", "")),
            persona=str(d.get("persona", "")),
            speech_style=str(d.get("speech_style", "")),
            cast_id=str(d.get("cast_id", "")),
            voicevox_speaker=(
                int(d["voicevox_speaker"]) if d.get("voicevox_speaker") is not None else None
            ),
            voicevox_style=str(d.get("voicevox_style", "")),
        )


@dataclass
class GeneratedDramaChunk:
    """朗読の1単位。地の文かセリフかで話者が変わる（§4.7.3）。"""

    index: int                       # シーン内の通し番号（0始まり）
    text: str                        # 読み上げるテキスト
    character_key: str | None = None  # セリフなら話者のキー。地の文は None
    is_sfx: bool = False             # 擬音・効果音だけの行（紙芝居的なメリハリ用に間を厚くする）


@dataclass
class GeneratedDramaScene:
    """放送対象の1シーン（DB の generated_drama_scenes 1行ぶん）。"""

    scene_id: int
    generated_drama_id: int
    generated_drama_title: str
    chapter: int
    scene: int
    body: str
    chunk_cursor: int = 0  # 途中まで流していれば、その続きから再開する

    @property
    def label(self) -> str:
        """画面の NOW PLAYING とログに出す位置表示（読み上げはしない）。"""
        return language.pick(
            ja=f"第{self.chapter}章 第{self.scene}場",
            en=f"chapter {self.chapter}, scene {self.scene}",
        )
