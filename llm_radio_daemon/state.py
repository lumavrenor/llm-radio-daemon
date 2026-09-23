"""スレッド間で共有する状態。

MainThread(Ursina。v1ではCLI表示スレッド) は読むだけ。
書き込みは各ワーカースレッドから行う。単純なプリミティブ型のみなので
GILに任せるが、"喋っている側/喋っていない側"のような複数フィールドを
同時に整合させたい更新だけ Lock で保護する。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class SharedState:
    current_rms: float = 0.0
    current_speaker: str | None = None  # cast の id（"host" 等）| None
    subtitle: str = ""
    now_playing: str = ""
    current_station: str = ""    # 今つないでいるネットラジオ局（StreamConfig.label）
    is_speaking: bool = False
    on_air: bool = True          # ON AIR ランプの状態（v2で画面表示に使う）
    filler_active: bool = False  # フィラー再生中かどうか（ランプ点滅の判定用）

    # Ollama 等がモデルをメモリへロードし終えたか。起動直後は数十秒〜1分ほど
    # かかることがあるため、完了するまで画面左上の LLM 名の横に "Loading..." を出す
    # （main.py の起動時ウォームアップが完了時に True へ立てる）。
    llm_ready: bool = False

    # 現在アクティブな台本に出演している cast の id。3D表示はこの人数だけ
    # ひな壇に並べる（空なら全員表示＝起動直後のフォールバック）。フィラー・
    # 読書コーナー中は更新しない（読書は reading_cast_ids が別途担当）。
    # ラジオドラマ朗読（§4.7.3）はシーン単位でここへ「ナレーター＋登場人物」を入れる。
    active_cast_ids: tuple[str, ...] = ()

    # ネタ源の取得状況。{コンテンツtype: 短い英語のステータス} で、画面左上の
    # コンテンツ名の横に出す。「スレッドが死んだのか、単にネタが無いのか」を
    # ログを掘らずに見分けるための表示なので、値は SourceStatus が作る。
    source_status: dict[str, str] = field(default_factory=dict)

    # --- 朗読（読書コーナー v4 §10.11 / ラジオドラマ v6 §4.7.3）---
    # NOW PLAYING の表示欄と「朗読中はフィラーを出さない」判定は両コーナーで共用する。
    reading_work_id: str | None = None  # 朗読中の青空文庫 作品ID（cursor 前進の宛先）
    reading_now_playing: str = ""       # 「作品名 / 著者名 / 青空文庫」「作品名 / 章 / ラジオドラマ」
    reading_progress: str = ""          # 「第3章 42%」など
    reading_cast_ids: tuple[str, ...] = ()  # 読書コーナー中に表示する出演者（朗読・つっこみの2名）。空なら全員表示
    duck_db_override: float | None = None  # 朗読中だけ深いダッキング。None なら audio.duck_db

    # ラジオドラマの自動執筆（§4.7.6）。GeneratedDramaSupervisorThread が generated_drama_writer
    # 子プロセスを実際に走らせている間だけ True。今アクティブなコンテンツが generated_drama
    # かどうかとは無関係（他コーナー放送中でも裏で執筆は進む）なので、画面左上には別行で出す。
    generated_drama_writing: bool = False

    # 今読み上げ中のラジオドラマの作品ID。GeneratedDramaCorner がシーンを積むたびに
    # 更新する（シーン間・時間帯をまたいでも同じ作品なら値は変わらない）。
    # display 側はこれが変わった瞬間を「1つのドラマの終わり→次のドラマの始まり」と
    # 見なし、コンテンツ切り替え時と同じ暗転を挟む。
    generated_drama_id: int | None = None

    # コーナー内ローカルな暗転（フィラー⇄朗読・シーンまたぎ）。director.py 本体の
    # fade_command()/ack_fade() とは別チャンネルで、director.is_steady() の間だけ
    # 使われる（本物のコーナー切り替えの暗転と衝突しないように）。
    local_fade: tuple[str, int] | None = None       # corner が書く: ("out"|"in", seq)
    local_fade_ack: tuple[str, int] | None = None   # display が書く: 描き終えた (kind, seq)
    generated_drama_filler_hold: bool = False  # True の間 FillerThread は新規フィラーを起こさない

    # 起動音量フェードインの開始時刻（time.monotonic()）。ストリーミング接続直後は
    # ノイズが乗りやすいので、AudioMixer は明転が始まるこの時刻まで無音のまま待ち、
    # そこから画面の明転と同じ長さでゆっくり音量を上げる（display/app.py が設定）。
    # display 無効時は main.py がミキサー起動と同時に即値を入れ、その場でフェードする。
    startup_fade_started_at: float | None = None

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def set_speaking(self, speaker: str | None, rms: float | None = None) -> None:
        with self._lock:
            self.current_speaker = speaker
            self.is_speaking = speaker is not None
            if rms is not None:
                self.current_rms = rms

    def set_source_status(self, content_type: str, text: str) -> None:
        with self._lock:
            self.source_status[content_type] = text

    def update_rms(self, rms: float) -> None:
        # 平滑化は呼び出し側（AudioCallback）で行う。ここは単純代入で十分。
        self.current_rms = rms

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "current_rms": self.current_rms,
                "current_speaker": self.current_speaker,
                "subtitle": self.subtitle,
                "now_playing": self.now_playing,
                "current_station": self.current_station,
                "is_speaking": self.is_speaking,
                "on_air": self.on_air,
                "filler_active": self.filler_active,
                "llm_ready": self.llm_ready,
                "active_cast_ids": self.active_cast_ids,
                "source_status": dict(self.source_status),
                "reading_work_id": self.reading_work_id,
                "reading_now_playing": self.reading_now_playing,
                "reading_progress": self.reading_progress,
                "reading_cast_ids": self.reading_cast_ids,
                "duck_db_override": self.duck_db_override,
                "generated_drama_writing": self.generated_drama_writing,
                "generated_drama_id": self.generated_drama_id,
            }
