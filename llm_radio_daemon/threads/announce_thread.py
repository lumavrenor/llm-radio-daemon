"""字幕・発話ログ担当スレッド。announce_queue → SharedState.subtitle + ログ。

字幕とログを TTS 合成時（TTSThread）に更新すると、speech_queue に積まれた
数行ぶんだけ音声より先行してしまう（「字幕が4〜5人ぶん先に出る」状態）。
そこで AudioMixer が実際に再生を開始した行だけをここへ流し、
このスレッドが字幕とログを更新する。ミキサーのコールバックからは
put_nowait だけ行い、文字列整形・ログ出力（I/O）はこのスレッドに逃がす。

朗読（読書コーナー v4 §10.6 / ラジオドラマ v6 §4.7.2）: 朗読チャンクが**再生開始した
時点**で進行位置を進める（合成完了時ではない）。停電・強制終了時に「合成したが
流していない分」を読み飛ばさないため。同じ層で朗読中の深いダッキングも出し入れする。

リクエスト受付（§6.2）: 受けアナウンス台本の最後の行が**再生開始した時点**で
DB へ entered を刻む（記録用。コーナーの切り替え自体は director.py が段取りを踏んで済ませている）。
"""

from __future__ import annotations

import logging
import queue
import threading

from .. import language
from ..config import CastMember
from ..db import TopicStore
from ..script import ScriptLine
from ..state import SharedState

logger = logging.getLogger(__name__)


class AnnounceThread(threading.Thread):
    def __init__(
        self,
        announce_queue: "queue.Queue[tuple[ScriptLine, bool]]",
        state: SharedState,
        cast: list[CastMember],
        stop_event: threading.Event | None = None,
        store: TopicStore | None = None,
    ):
        super().__init__(name="AnnounceThread", daemon=True)
        self._announce_queue = announce_queue
        self._state = state
        self._names = {m.id: m.name for m in cast}
        self._stop_event = stop_event or threading.Event()
        self._store = store

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                line, _is_filler = self._announce_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            name = line.speaker_label or self._names.get(line.speaker, line.speaker)
            # 話者名と本文の区切り。全角コロンは日本語の字幕でしか見栄えがしないので
            # 言語で出し分ける（英語では半角コロン＋スペース）。
            sep = ": " if language.current() == "en" else "："
            self._state.subtitle = f"{name}{sep}{line.text}"
            logger.info("[%s] %s", name, line.text)
            self._on_air_talk_cast(line)
            # 朗読行だけ BGM を深く絞る（通常トーク・感想パートでは None に戻す）。
            self._state.duck_db_override = line.duck_db
            self._on_air_literary_reading(line)
            self._on_air_translated_reading(line)
            self._on_air_biography_reading(line)
            self._on_air_generated_drama(line)
            self._on_air_request_entered(line)

    def _on_air_talk_cast(self, line: ScriptLine) -> None:
        """通常トークのひな壇の顔ぶれを、その行の再生開始時に切り替える。

        TTSThread は合成が再生より数行先行するため、台本の切り替わりで
        ``active_cast_ids`` を直接書くと前の話題のトークがまだ流れているうちに
        次の話題のキャストへ入れ替わる。そこで顔ぶれは行に貼っておき（``talk_cast_ids``）、
        ここで実際に再生が始まった時に適用する。変化時のみ反映（毎行の書き込みを避ける）。
        """
        ids = line.talk_cast_ids
        if ids and self._state.active_cast_ids != ids:
            self._state.active_cast_ids = ids

    def _on_air_literary_reading(self, line: ScriptLine) -> None:
        """読書コーナー（青空文庫）の cursor を再生開始時に進める（§10.6）。"""
        if line.reading_chunk_index is None:
            return
        work_id = self._state.reading_work_id
        if self._store is not None and work_id:
            try:
                self._store.advance_literary_reading_cursor(work_id, line.reading_chunk_index + 1)
            except Exception:
                logger.exception("failed to advance literary reading cursor for %s", work_id)

    def _on_air_translated_reading(self, line: ScriptLine) -> None:
        """翻訳朗読コーナーの cursor を再生開始時に進める（読書コーナーと同じ理由）。"""
        if line.translated_chunk_index is None:
            return
        work_id = self._state.reading_work_id
        if self._store is not None and work_id:
            try:
                self._store.advance_translated_reading_cursor(work_id, line.translated_chunk_index + 1)
            except Exception:
                logger.exception("failed to advance translated reading cursor for %s", work_id)

    def _on_air_biography_reading(self, line: ScriptLine) -> None:
        """偉人伝トークの cursor を再生開始時に進める（読書コーナーと同じ理由）。"""
        if line.biography_chunk_index is None:
            return
        figure_id = self._state.reading_work_id
        if self._store is not None and figure_id:
            try:
                self._store.advance_biography_reading_cursor(figure_id, line.biography_chunk_index + 1)
            except Exception:
                logger.exception("failed to advance biography reading cursor for %s", figure_id)

    def _on_air_request_entered(self, line: ScriptLine) -> None:
        """受けアナウンスの最後の行が再生され始めた瞬間に、DB へ entered を刻む（§6.2・記録用）。"""
        if line.request_id is None or self._store is None:
            return
        try:
            self._store.mark_request_entered(line.request_id)
        except Exception:
            logger.exception("Failed to update request entered")

    def _on_air_generated_drama(self, line: ScriptLine) -> None:
        """ラジオドラマ朗読の generated_drama_progress を再生開始時に進める（§4.7.2）。"""
        if line.generated_drama_scene_id is None or self._store is None:
            return
        try:
            self._store.advance_generated_drama_progress(
                line.generated_drama_scene_id,
                (line.generated_drama_chunk_index or 0) + 1,
                finished=line.generated_drama_scene_end,
            )
        except Exception:
            logger.exception(
                "failed to advance generated drama progress for scene %s", line.generated_drama_scene_id
            )
