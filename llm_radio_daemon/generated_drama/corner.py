"""ラジオドラマ朗読のライフサイクル管理（v6 §4.7.2 / §4.7.3 / §4.7.4）。

``ScriptThread`` は時間帯内でこのクラスの ``produce_batch()`` を繰り返し呼ぶだけ。
**Ollama は一切呼ばない。** 執筆バッチが確定させた本文を、地の文とセリフに割って
``ScriptLine`` に組み直し、そのまま TTSThread へ渡す。

「小説が用意できていないことは異常ではない。放送が止まることだけが異常。」（§4.7.4）
"""

from __future__ import annotations

import logging
import time

from .. import language
from ..config import CastMember, ContentConfig, GeneratedDramaParams
from ..db import TopicStore
from ..director import ProgramDirector
from ..script import Script, ScriptLine
from ..script.ollama_client import pick_speakers
from ..state import SharedState
from . import GeneratedDramaCharacter, GeneratedDramaChunk, GeneratedDramaScene
from .data import GeneratedDramaData
from .parser import split_scene
from .source import GeneratedDramaSource

logger = logging.getLogger(__name__)

_MAX_SCENE_FAILURES = 3  # 本文が読めないシーンがこれだけ続いたらコーナー中止（§4.7.4）

# 積み終わったシードの再生が、これだけ進まなかったら見切って次へ進む。
# 進行位置は「再生開始時」にしか進まないので（§4.7.2）、TTS が死んで1行も
# 流れなくなるとここで待ち続けてしまう。無音を作らないための保険。
_STALL_SEC = 180.0

# フィラー⇄朗読・シーンまたぎのローカル暗転。ack が来ない場合の見切り時間
# （director.py の _FADE_ACK_TIMEOUT_SEC と同じ考え方）。
_TRANSITION_FADE_ACK_TIMEOUT_SEC = 5.0


class GeneratedDramaCorner:
    def __init__(
        self,
        content: ContentConfig,
        store: TopicStore,
        state: SharedState,
        cast: list[CastMember],
        director: ProgramDirector | None = None,
    ):
        self._content = content
        self._np: GeneratedDramaParams = content.generated_drama or GeneratedDramaParams()
        self._store = store
        self._state = state
        self._cast = cast
        self._cast_by_id = {m.id: m for m in cast}
        self._source = GeneratedDramaSource(store)
        # フェード演出には番組進行（director.py）の is_steady()/audio_idle()/
        # display_attached を参照する。None（オフライン検証スクリプト等）なら
        # 演出は挟まず、常に今まで通り同期的に切り替える。
        self._director = director

        self._narrator: CastMember | None = None
        self._scene: GeneratedDramaScene | None = None
        self._chunks: list[GeneratedDramaChunk] = []
        self._cursor = 0
        self._voices: dict[str, tuple[CastMember, str | None]] = {}  # key -> (cast, style)
        self._characters: dict[str, GeneratedDramaCharacter] = {}
        self._scene_failures = 0
        self._disabled = False
        # 「script_queue へ積み終わったが、まだ流し終わっていないシーン」。
        # 進行位置（generated_drama_progress）は再生開始時にしか進まないので、これを覚えて
        # おかないと、同じシーンを DB から何度も拾い直して二重・三重に積んでしまう。
        self._awaiting_scene_id: int | None = None
        self._awaiting_cursor = -1
        self._awaiting_since = 0.0

        # --- フィラー⇄朗読・シーンまたぎのローカル暗転（フェード）用 ---
        # None: 通常運転。それ以外: quiet(音が止まるのを待つ) → pre_pause(間) →
        # fade_out(暗転指示・ack待ち) → fade_in(明転指示・ack待ち) → post_pause(間)
        self._transition_phase: str | None = None
        self._transition_target: GeneratedDramaScene | None = None  # None ならフィラーへ復帰
        self._transition_chunks: list[GeneratedDramaChunk] = []
        self._transition_since = 0.0
        self._transition_seq = 0

    # --- 公開 API ---------------------------------------------------------

    @property
    def disabled(self) -> bool:
        return self._disabled

    @property
    def transition_pending(self) -> bool:
        """フェード演出の途中か（ScriptThread がポーリング間隔を詰めるために読む）。"""
        return self._transition_phase is not None

    def produce_batch(self) -> list[Script]:
        """次に script_queue へ積む Script を返す。読むものが無ければ空を返す。"""
        if self._disabled:
            return []

        if self._transition_phase is not None:
            if not self._tick_transition():
                return []

        if self._scene is None and not self._load_next_scene():
            return []

        scene = self._scene
        assert scene is not None

        lines: list[ScriptLine] = []
        if self._cursor == 0:
            lines.extend(self._opening_lines(scene))

        take = self._chunks[self._cursor : self._cursor + self._np.chunks_per_batch]
        if not take:
            self._finish_scene()
            return []

        for ch in take:
            member, style = self._voice_for(ch.character_key)
            # 字幕・ログは声の cast 名ではなく役名を出す（「原野 静：」ではなく「カレン：」）。
            # 地の文（character_key=None）は None のままにして cast 名（ナレーター）を使う。
            char = self._characters.get(ch.character_key) if ch.character_key else None
            speaker_label = char.name if char else None
            is_last = ch.index == len(self._chunks) - 1
            if is_last:
                pad_ms = self._np.pause_scene_ms
            elif ch.is_sfx:
                pad_ms = self._np.pause_se_ms
            else:
                pad_ms = self._np.pause_ms
            # 効果音の直前でも一拍おく（「その時、大きな音がして」→ 間 → 「ドオオン！」）。
            if ch.is_sfx and lines:
                lines[-1].pad_ms = max(lines[-1].pad_ms or 0, self._np.pause_se_ms)
            lines.append(
                ScriptLine(
                    speaker=member.id,
                    text=ch.text,
                    style=style,
                    speaker_label=speaker_label,
                    pad_ms=pad_ms,
                    duck_db=self._np.duck_db,
                    generated_drama_scene_id=scene.scene_id,
                    generated_drama_chunk_index=ch.index,
                    generated_drama_scene_end=is_last,
                )
            )
        self._cursor += len(take)

        self._publish_state(scene)
        script = Script(
            topic_id=None,
            topic_title=f"generated_drama:{scene.generated_drama_id}:{scene.chapter}:{scene.scene}:{self._cursor}",
            lines=lines,
            is_generated_drama=True,
        )
        if self._cursor >= len(self._chunks):
            # このシーンは積み終わり。実際の読了記録は再生開始時（AnnounceThread）
            # なので、流し終わるまでは次のシーンを取りに行かない（積み直し防止）。
            self._scene = None
            self._awaiting_scene_id = scene.scene_id
            self._awaiting_cursor = -1
            self._awaiting_since = time.monotonic()
        return [script]

    def end_corner(self) -> list[Script]:
        """時間帯を抜けたときに1回だけ呼ぶ。途中のシーンは次回そこから再開する。"""
        self._scene = None
        self._chunks = []
        self._cursor = 0
        self._awaiting_scene_id = None
        # 本物のコーナー切り替えがローカル遷移の途中に割り込んだ場合、ここで
        # 片付けないと generated_drama_filler_hold が立ちっぱなしになり、
        # 他のどのコンテンツに切り替わってもフィラーが永久に出なくなる。
        self._end_transition()
        self._clear_state()
        return []

    # --- シーンの読み込み -------------------------------------------------

    def _load_next_scene(self) -> bool:
        for _ in range(_MAX_SCENE_FAILURES):
            try:
                scene = self._source.next_scene()
            except Exception:
                logger.exception("generated_drama: failed to get next scene. Skipping this time")
                return False
            if scene is None:
                # 直前まで何かのシーンの読了を待っていたなら、それは「シーンが終わった
                # のに次が用意できていない」＝フィラーへ戻るタイミング。フェードを挟む。
                # 単に「まだ何も無い」を毎回検知しているだけ（既にフィラー中）なら、
                # 都度フェードを起こさず今まで通り即座に反映する。
                was_awaiting = self._awaiting_scene_id is not None
                self._awaiting_scene_id = None
                # 執筆が追いついていないだけで異常ではない（§4.7.4）。ただ画面上は
                # 「フィラーで場をつないでいる理由」が分かった方がよいので出す。
                self._state.set_source_status("generated_drama", "waiting for next scene")
                if was_awaiting and self._director is not None and self._director.is_steady():
                    self._begin_transition(None, [])
                else:
                    self._clear_state()
                return False  # 生成済み・未放送のシーンが無い

            if scene.scene_id == self._awaiting_scene_id:
                # 積み終わったシーンがまだ流れている。読み終わるのを待つ
                # （ここで積み直すと同じ本文を何度も朗読することになる）。
                if not self._wait_for_playback(scene):
                    return False
                continue  # 見切って次のシーンへ
            self._awaiting_scene_id = None

            chunks = self._parse(scene)
            if not chunks:
                self._scene_failures += 1
                logger.warning(
                    "generated_drama: scene %s (scene_id=%d) is unreadable/empty. Skipping it (attempt %d)",
                    scene.label, scene.scene_id, self._scene_failures,
                )
                # スキップしないと同じシーンを永久に掴み続けるので、放送済み扱いにする。
                self._store.advance_generated_drama_progress(scene.scene_id, 0, finished=True)
                if self._scene_failures >= _MAX_SCENE_FAILURES:
                    logger.error(
                        "generated_drama: body failures reached %d. Aborting the drama corner and returning to normal broadcast (§4.7.4)",
                        self._scene_failures,
                    )
                    self._disabled = True
                    self._clear_state()
                    return False
                continue

            self._scene_failures = 0
            cursor = max(0, min(scene.chunk_cursor, len(chunks)))
            if cursor >= len(chunks):
                # 全チャンク再生済みなのに finished_at が入っていない（強制終了直後など）。
                self._store.advance_generated_drama_progress(scene.scene_id, cursor, finished=True)
                continue

            if self._director is not None and self._director.is_steady():
                # 今流れている音が止まる → 間 → 暗転 → キャスト交換 → 明転 → 間、
                # を挟んでから読み始める（フィラーの途中でキャストが飛ぶのを防ぐ）。
                self._begin_transition(scene, chunks)
                return False

            self._scene = scene
            self._chunks = chunks
            self._cursor = cursor
            self._publish_state(scene)
            logger.info(
                "generated_drama: starting to read \"%s\" %s chunk=%d/%d (characters: %s)",
                scene.generated_drama_title, scene.label, self._cursor, len(chunks),
                ", ".join(self._scene_character_names()) or "-",
            )
            return True
        return False

    def _wait_for_playback(self, scene: GeneratedDramaScene) -> bool:
        """積み終わったシーンの再生を待つ。True なら見切って次へ進んでよい。

        待っている間も画面の進捗は「実際に流れた位置」で更新する。
        """
        if scene.chunk_cursor != self._awaiting_cursor:
            self._awaiting_cursor = scene.chunk_cursor
            self._awaiting_since = time.monotonic()
            total = len(self._chunks) or 1
            self._state.reading_progress = f"{int(100 * scene.chunk_cursor / total)}%"
            return False
        if time.monotonic() - self._awaiting_since < _STALL_SEC:
            return False
        logger.warning(
            "generated_drama: playback of %s hasn't advanced in %.0f seconds (TTS stopped?). Treating it as finished and moving on",
            scene.label, _STALL_SEC,
        )
        self._store.advance_generated_drama_progress(scene.scene_id, scene.chunk_cursor, finished=True)
        self._awaiting_scene_id = None
        return True

    # --- フィラー⇄朗読・シーンまたぎのローカル暗転 -------------------------

    def _begin_transition(self, target: GeneratedDramaScene | None, chunks: list[GeneratedDramaChunk]) -> None:
        """演出を開始する。target=None ならフィラーへの復帰。"""
        self._transition_target = target
        self._transition_chunks = chunks
        self._transition_phase = "quiet"
        self._transition_since = time.monotonic()
        if target is not None:
            # フィラーの続きが新しく起きないよう止める（今流れているぶんは最後まで流す）。
            self._state.generated_drama_filler_hold = True

    def _tick_transition(self) -> bool:
        """演出を1段階進める。まだ途中なら False。完了したら True（同じ tick で通常処理へ続けてよい）。"""
        director = self._director
        if director is None:
            # 遷移開始は director がある時にしか起きない（念のための保険）。
            self._end_transition()
            return True
        if not director.is_steady():
            # 本物のコーナー切り替えが割り込んだ。end_corner() の片付けに任せる。
            return False

        phase = self._transition_phase
        if phase == "quiet":
            if not director.audio_idle():
                return False
            self._transition_phase = "pre_pause"
            self._transition_since = time.monotonic()
            return False

        if phase == "pre_pause":
            if not self._elapsed_ms(self._np.transition_pause_before_ms):
                return False
            self._transition_seq += 1
            self._state.local_fade = ("out", self._transition_seq)
            self._transition_phase = "fade_out"
            self._transition_since = time.monotonic()
            return False

        if phase == "fade_out":
            if not self._fade_acked(director, "out"):
                return False
            self._apply_swap()
            self._state.local_fade = ("in", self._transition_seq)
            self._transition_phase = "fade_in"
            self._transition_since = time.monotonic()
            return False

        if phase == "fade_in":
            if not self._fade_acked(director, "in"):
                return False
            self._transition_phase = "post_pause"
            self._transition_since = time.monotonic()
            return False

        if phase == "post_pause":
            if not self._elapsed_ms(self._np.transition_pause_after_ms):
                return False
            self._end_transition()
            return True

        # 来ないはずの状態。安全側に倒して終了する（暗いまま固まらせない）。
        self._end_transition()
        return True

    def _elapsed_ms(self, ms: int) -> bool:
        return (time.monotonic() - self._transition_since) * 1000 >= ms

    def _fade_acked(self, director: ProgramDirector, kind: str) -> bool:
        if not director.display_attached:
            return True  # 画面が無ければ描画されないので即完了扱い
        if self._state.local_fade_ack == (kind, self._transition_seq):
            return True
        return (time.monotonic() - self._transition_since) >= _TRANSITION_FADE_ACK_TIMEOUT_SEC

    def _apply_swap(self) -> None:
        """暗転しきったところでキャスト・表示状態を入れ替える。"""
        # director.py の _swap() と同様、暗転しきって画面が真っ黒な間に字幕を消す。
        # ここで消さないと、フィラー⇄朗読の切り替わりで前の文章が明転後も
        # 一瞬（次の発話が subtitle を上書きするまで）残って見えてしまう。
        self._state.subtitle = ""
        scene = self._transition_target
        chunks = self._transition_chunks
        if scene is not None and chunks:
            self._scene = scene
            self._chunks = chunks
            self._cursor = max(0, min(scene.chunk_cursor, len(chunks)))
            self._publish_state(scene)
            logger.info(
                "generated_drama: starting to read \"%s\" %s chunk=%d/%d (characters: %s)",
                scene.generated_drama_title, scene.label, self._cursor, len(chunks),
                ", ".join(self._scene_character_names()) or "-",
            )
        else:
            # フィラーへ復帰（またはチャンクが空だった場合の保険）。
            self._clear_state()

    def _end_transition(self) -> None:
        self._transition_phase = None
        self._transition_target = None
        self._transition_chunks = []
        self._transition_since = 0.0
        self._state.generated_drama_filler_hold = False
        self._state.local_fade = None

    def _parse(self, scene: GeneratedDramaScene) -> list[GeneratedDramaChunk]:
        try:
            characters = GeneratedDramaData(self._np.data_dir, scene.generated_drama_id).load_characters()
        except Exception:
            logger.exception("generated_drama: cannot read characters.json. Reading everything as narrator")
            characters = []
        self._characters = {c.key: c for c in characters}
        self._assign_voices(characters)
        try:
            return split_scene(
                scene.body,
                characters,
                dialogue_by_character=self._np.dialogue_by_character,
                chunk_target_chars=self._np.chunk_target_chars,
                chunk_max_chars=self._np.chunk_max_chars,
                chunk_min_chars=self._np.chunk_min_chars,
            )
        except Exception:
            logger.exception("generated_drama: failed to parse text scene_id=%d", scene.scene_id)
            return []

    def _finish_scene(self) -> None:
        self._scene = None
        self._chunks = []
        self._cursor = 0

    # --- 声の割り当て -----------------------------------------------------

    def _narrator_member(self) -> CastMember:
        """地の文を読む出演者。未指定・不在なら抽選1人目で代行する（§4.7.3）。"""
        if self._narrator is not None:
            return self._narrator
        wanted = self._np.narrator_cast_id
        member = next((m for m in self._cast if m.role == wanted), None) if wanted else None
        if member is None:
            if wanted:
                logger.warning(
                    "generated_drama: no [[cast]] has role narrator_cast_id=%r. The first randomly picked speaker will read the narration",
                    wanted,
                )
            member = pick_speakers(self._content, self._cast)[0]
        self._narrator = member
        return member

    def _assign_voices(self, characters: list[GeneratedDramaCharacter]) -> None:
        """characters.json の各キャラを [[cast]] の1人へ割り当てる（§4.7.3）。

        1. ``cast_id`` が [[cast]] にあればそれ
        2. ``voicevox_speaker`` が一致する出演者がいればその人（スタイルまで合わせる）
        3. 見つからなければナレーターへフォールバック（放送は止めない）
        """
        narrator = self._narrator_member()
        self._voices = {}
        for c in characters:
            member = self._cast_by_id.get(c.cast_id) if c.cast_id else None
            style: str | None = None
            if member is None and c.voicevox_speaker is not None:
                for m in self._cast:
                    hit = next((s for s in m.styles if s.id == c.voicevox_speaker), None)
                    if hit is not None:
                        member, style = m, hit.name
                        break
            if member is None:
                logger.info(
                    "generated_drama: no [[cast]] matches \"%s\". The narrator (%s) will read it",
                    c.name, narrator.name,
                )
                self._voices[c.key] = (narrator, None)
                continue
            if c.voicevox_style and any(s.name == c.voicevox_style for s in member.styles):
                style = c.voicevox_style
            self._voices[c.key] = (member, style)

    def _voice_for(self, character_key: str | None) -> tuple[CastMember, str | None]:
        if character_key is None:
            return self._narrator_member(), None
        return self._voices.get(character_key, (self._narrator_member(), None))

    def _scene_cast_ids(self) -> tuple[str, ...]:
        """ひな壇に出す顔ぶれ＝ナレーター＋そのシーンに出るキャラ（§4.7.3）。"""
        ids = [self._narrator_member().id]
        for ch in self._chunks:
            if ch.character_key is None:
                continue
            member, _ = self._voice_for(ch.character_key)
            ids.append(member.id)
        return tuple(dict.fromkeys(ids))

    def _scene_character_names(self) -> list[str]:
        keys = {ch.character_key for ch in self._chunks if ch.character_key}
        return [self._characters[k].name for k in keys if k in self._characters]

    # --- 冒頭アナウンス・状態 ---------------------------------------------

    def _opening_lines(self, scene: GeneratedDramaScene) -> list[ScriptLine]:
        """章の頭だけ、章題を一言だけ挟む（LLM は通さない・固定文）。

        **LLM を通らないので language.LANGUAGE_GUIDANCE が効かない。** 言語ごとの
        定型文をここに持つ（filler.py のテンプレートと同じ扱い）。素の数字は
        misaki が num2words で読むので（``Chapter 3.`` → chapter three）、
        ここで語に開く必要はない。読めないのは ``2:30`` のような記号混じりだけ。
        """
        if scene.scene != 1:
            return []
        narrator = self._narrator_member()
        if language.current() == "en":
            text = (
                f"{scene.generated_drama_title}. Chapter {scene.chapter}."
                if scene.chapter == 1
                else f"Chapter {scene.chapter}."
            )
        elif scene.chapter == 1:
            text = f"{scene.generated_drama_title}。第{scene.chapter}章。"
        else:
            text = f"第{scene.chapter}章。"
        return [ScriptLine(narrator.id, text, pad_ms=self._np.pause_scene_ms)]

    def _publish_state(self, scene: GeneratedDramaScene) -> None:
        self._state.set_source_status("generated_drama", "")
        # NOW PLAYING は読書コーナーの表示欄を流用する（§4.7.3）。
        source_label = language.pick(ja="ラジオドラマ", en="radio drama")
        self._state.reading_now_playing = (
            f"{scene.generated_drama_title} / {scene.label} / {source_label}"
        )
        # シーンをまたいで積むたびに書くだけ（＝同じ作品の間は値が変わらない）。
        # 表示側はこれの変化だけを見るので、ここで毎回書いても暗転の頻発にはならない。
        self._state.generated_drama_id = scene.generated_drama_id
        total = len(self._chunks) or 1
        self._state.reading_progress = f"{int(100 * self._cursor / total)}%"
        # ひな壇は「ナレーター＋そのシーンに出るキャラ」だけ。空だと10人全員並ぶ。
        self._state.active_cast_ids = self._scene_cast_ids()

    def _clear_state(self) -> None:
        self._state.reading_now_playing = ""
        self._state.reading_progress = ""
        self._state.duck_db_override = None
        # active_cast_ids を空のままにすると、表示側の「空＝全員表示」フォールバック
        # （起動直後用）が働いて、データ待ちの間ずっと出演者全員が並んでしまう。
        # ナレーター1人だけにしておく。
        self._state.active_cast_ids = (self._narrator_member().id,)
