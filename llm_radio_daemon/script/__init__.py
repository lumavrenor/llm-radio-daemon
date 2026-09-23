"""原稿データ構造。"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

# 台本の通し番号。ウォッチドッグが TTSThread 等を作り直しても重複しないよう
# モジュールに1本だけ持つ（itertools.count の next() は CPython では原子的）。
_script_seq = itertools.count(1)


@dataclass
class ScriptLine:
    speaker: str  # cast の id（"host" / "tsubasa" など）。フィラーは司会/アシスタントの id
    text: str
    # TTS用の読み上げテキスト（誤読対策のLLM後処理・script/reading.py）。
    # None なら text をそのまま読む（従来どおり）。字幕は常に text を使う。
    speech_text: str | None = None
    style: str | None = None  # VOICEVOX スタイル名。None / 未知の名前なら cast の既定スタイル
    speaker_label: str | None = None  # 字幕・ログに出す名前の上書き（ラジオドラマの役名など）。None なら cast 名
    # --- ひな壇（通常トーク）---
    # この行の再生が始まった時にひな壇へ並べる出演者。台本の切り替わりで顔ぶれを
    # 変えるのを「合成時」ではなく「再生開始時」に遅らせるためのタグ（AnnounceThread が適用）。
    talk_cast_ids: tuple[str, ...] | None = None
    # --- 朗読（読書コーナー v4 §10.4 / ラジオドラマ v6 §4.7.3）---
    pad_ms: int | None = None            # >0 なら合成 PCM 末尾に無音を付す（朗読の「間」）
    duck_db: float | None = None         # この行の再生中だけ BGM をこの深さまで絞る
    reading_chunk_index: int | None = None  # 朗読チャンクの通し番号。再生開始時に cursor を進める
    # --- 翻訳朗読コーナー（Project Gutenberg）。再生開始時に cursor を進める ---
    translated_chunk_index: int | None = None
    # --- 偉人伝トーク（Wikipedia）。再生開始時に cursor を進める ---
    biography_chunk_index: int | None = None
    # --- ラジオドラマ朗読（v6 §4.7.2）。再生開始時に generated_drama_progress を進める ---
    generated_drama_scene_id: int | None = None    # generated_drama_scenes.id
    generated_drama_chunk_index: int | None = None  # シーン内のチャンク番号（0始まり）
    generated_drama_scene_end: bool = False        # そのシーンの最終チャンクか（流し始めたら読了扱い）
    # --- リクエスト受けアナウンス（§6.2）---
    # 受けアナウンス台本の最後の行にだけ立てる。requests.id。再生開始時に
    # AnnounceThread が mark_request_entered() で記録する。
    request_id: int | None = None
    # --- コーナー切り替え（director.py）。TTSThread が合成時に台本から写す ---
    # ミキサーは切り替え待ち（DRAINING）の間、「いま流れている台本」と締めの台本の
    # 行だけを通し、それ以外（＝次のトーク）を捨てる。その判定に使う。
    script_seq: int | None = None
    closing: bool = False


@dataclass
class Script:
    topic_id: int | None
    topic_title: str
    lines: list[ScriptLine]
    # 抽選された出演者の id（min/max_speakers で決まる人数ぶん）。台本に実際のセリフが
    # 無い人も含む。ひな壇（active_cast_ids）はこちらを使い、抽選人数どおりに並べる。
    # 空なら TTSThread が実際にセリフのある話者へフォールバックする。
    appearer_ids: tuple[str, ...] = ()
    is_filler: bool = False  # True ならフィラートーク（放送記録・重複排除の対象外）
    is_literary_reading: bool = False  # True なら読書コーナー（topics 経路の記録対象外）
    is_translated_reading: bool = False  # True なら翻訳朗読コーナー（topics 経路の記録対象外）
    is_biography_reading: bool = False  # True なら偉人伝トーク（topics 経路の記録対象外）
    is_generated_drama: bool = False   # True ならラジオドラマ朗読（ひな壇はシーン単位で GeneratedDramaCorner が出す）
    # 前のコーナーの締め（読書コーナーの「続きはまた明日」など）。コーナー切り替え待ちの
    # 間も捨てずに流す台本の印。
    is_closing: bool = False
    seq: int = field(default_factory=lambda: next(_script_seq))
