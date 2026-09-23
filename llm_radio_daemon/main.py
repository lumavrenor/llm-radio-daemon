"""LLM Radio Daemon v1 エントリポイント（3Dなし・CLI）。

実装順(SPEC.md 9章)のステップ7に対応：
config → SharedState → ミキサー → ネットラジオ → VOICEVOX → Wikimedia →
Ollama をスレッド＋キューで接続する。
"""

from __future__ import annotations

import argparse
import logging
import queue
import random
import sys
import threading
import time

from .audio.icy import IcyMetadataReader
from .audio.mixer import AudioMixer
from .audio.music import MusicThread
from .audio.ringbuffer import AudioRingBuffer
from .config import CONFIG_ARG_HELP, ConfigError, load_config
from .db import TopicStore
from .dedup import EmbeddingDeduplicator, check_embedding
from .director import ProgramDirector
from .logging_setup import setup_logging
from . import language, llm_http, schedule, sensitive
from .biography_reading.corner import BiographyReadingCorner
from .generated_drama.corner import GeneratedDramaCorner
from .literary_reading.corner import LiteraryReadingCorner
from .translated_reading.corner import TranslatedReadingCorner
from .script.ollama_client import (
    set_pin_angle_index,
    set_pinned_cast_ids,
    warm_up,
)
from .source_status import SourceStatus
from .sources import Source
from .sources.arxiv import ArxivSource
from .sources.hackernews import HackerNewsSource
from .sources.musicbrainz import topic_for_now_playing
from .sources.rss import RssSource
from .sources.sports import SportsSource
from .sources.weather import WeatherSource
from .sources.wikimedia import WikimediaSource
from .sources.worry_consultation import WorryConsultationSource
from .mood import CastMoodProvider
from .state import SharedState
from .weather import WeatherProvider
from .threads.announce_thread import AnnounceThread
from .threads.director_thread import DirectorThread
from .threads.filler_thread import FillerThread
from .threads.generated_drama_supervisor import GeneratedDramaSupervisorThread
from .threads.script_thread import ScriptThread
from .threads.source_thread import SourceThread, enqueue_topic
from .threads.station_thread import StationHolder, StationThread
from .threads.tts_thread import TTSThread
from .threads.watchdog import WatchdogThread
from .tts import create_backend

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM Radio Daemon (v1: 3Dなし・CLI) / LLM Radio Daemon (v1: no 3D display, CLI)"
    )
    parser.add_argument("--config", required=True, help=CONFIG_ARG_HELP)
    args = parser.parse_args()

    # ログの設定より前なので、ここだけは stderr に直接出す。言語の指定間違い
    # （go.bat en のタイプミスなど）はまずここに出るので、traceback は見せない。
    try:
        config = load_config(args.config)
    except ConfigError as e:
        print(f"Failed to load config: {e}", file=sys.stderr)
        raise SystemExit(1)

    setup_logging(config.log.dir, config.log.level)
    logger.info("starting LLM Radio Daemon v1")

    # 台本を書かせる言語をプロンプトに反映する（[locale] lang）。
    # 未知の言語コードなら日本語のまま据え置くので、ここで気付けるよう警告しておく。
    if not language.is_supported(config.locale.lang):
        logger.warning(
            "[locale] lang=%r is not supported. Scripts will be generated in Japanese "
            "(supported: ja / en).",
            config.locale.lang,
        )
    language.set_language(config.locale.lang)
    sensitive.set_language(config.locale.lang)

    # 「このパソコンの中の生成AIが喋っている」を楽屋ネタにする番組なので、実際に
    # どこで動いているかを一度だけ確かめて config へ書き戻す（[llm] placement = "auto"）。
    # ollama の `<model>-cloud` はローカルと同じ host から使えてしまうため、モデル名を
    # 差し替えただけで台詞が嘘になる。判定できなければ "unknown" ＝ 実行場所には触れない。
    if config.llm.placement == "auto":
        config.llm.placement = llm_http.detect_placement(config.llm)
    logger.info(
        "LLM placement=%s (model=%s engine=%s host=%s)",
        config.llm.placement, config.llm.model, config.llm.engine, config.llm.host,
    )
    if config.llm.placement == "unknown":
        logger.warning(
            "Could not determine where the model is running. Talk segments won't mention "
            "\"local/cloud\". To force it, set [llm] placement to local / cloud."
        )

    # Ollama はモデル未ロード状態からの初回リクエストで数十秒〜1分ほどかかることが
    # あるため、TTS 初期化など他の起動処理と並行してバックグラウンドで一度ウォーム
    # アップする。ここで早めに投げておかないと、後段の TTS 初期化（kokoro のロード等）
    # が終わるまでモデルロードが始まらず、待ち時間がただ直列に積み上がってしまう。
    # 完了（成否問わず）で state.llm_ready を立て、画面左上の "Loading..." を消す。
    state = SharedState()

    def _warm_up_llm() -> None:
        warm_up(config.llm)
        state.llm_ready = True

    threading.Thread(target=_warm_up_llm, name="LLMWarmup", daemon=True).start()

    # 天気（[weather]）。フィラー雑談とお天気コーナーが同じ値を見るよう、取得口は
    # プロセスにひとつだけ持つ。無効・未設定なら None ＝ どちらも天気に触れない。
    weather_provider: WeatherProvider | None = None
    if config.weather.enabled:
        weather_provider = WeatherProvider(
            latitude=float(config.weather.latitude),
            longitude=float(config.weather.longitude),
            location_name=config.weather.location_name,
            lang=config.locale.lang,
            refresh_sec=config.weather.refresh_sec,
            timeout_sec=config.weather.timeout_sec,
            url=config.weather.forecast_url,
        )
        logger.info(
            "weather enabled: %s (%.4f, %.4f) refresh=%.0fs",
            config.weather.location_name,
            config.weather.latitude, config.weather.longitude,
            config.weather.refresh_sec,
        )

    if config.debug.pinned_cast_ids:
        set_pinned_cast_ids(config.debug.pinned_cast_ids)
        logger.warning(
            "DEBUG: pinning cast ([debug] pinned_cast_ids): %s",
            ", ".join(config.debug.pinned_cast_ids),
        )

    if config.debug.pin_angle_index is not None:
        set_pin_angle_index(config.debug.pin_angle_index)
        logger.warning(
            "DEBUG: pinning every cast member's \"current angle\" to angle_variants[%d] ([debug] pin_angle_index)",
            config.debug.pin_angle_index,
        )

    # TTS が使えないまま走ると、生成した台本が TTSThread で1行ずつ捨てられ、
    # 音楽とフィラーだけが流れる（しかも原稿は消費済み）。ここで止めてしまう。
    try:
        tts_backend = create_backend(config)
    except Exception as e:
        logger.error("[tts] Failed to initialize backend: %s", e)
        raise SystemExit(1)
    try:
        tts_version = tts_backend.health_check()
    except Exception as e:
        logger.error(
            "TTS (%s) を使用できません: %s。設定を確認して起動し直してください。",
            config.tts.backend,
            e,
        )
        raise SystemExit(1)
    logger.info("Using TTS backend=%s (%s)", config.tts.backend, tts_version)

    # [[cast]] は声の「名前」だけを持つ（数値IDや配合済みベクトルは書かない）。ここで
    # バックエンドの実データと突き合わせて解決する。1つでも解決できなければ、聞こえない
    # 声のまま放送を始めるより先に、ここで即座に落とす。
    try:
        tts_backend.resolve_cast_voices(config.cast)
    except ConfigError as e:
        logger.error("[[cast]] voice settings do not match TTS backend=%s: %s", config.tts.backend, e)
        raise SystemExit(1)
    except Exception as e:
        logger.error("Failed to get the voice list from TTS backend=%s: %s", config.tts.backend, e)
        raise SystemExit(1)

    stop_event = threading.Event()

    store = TopicStore(config.db.path, config.db.rebroadcast_after_days)

    # リクエスト（§6.2）を番組表より優先させる。番組進行（director.py）が作られるより
    # 先に登録するので、放送を落として上げ直しても生きているリクエストはそのまま効く。
    schedule.set_request_provider(store.active_request_target)
    pending = store.active_request()
    if pending is not None:
        logger.info(
            "request in progress: %s (%.0f minutes remaining)",
            pending["label"],
            max(0.0, pending["expires_at"] - time.time()) / 60.0,
        )

    deduper = EmbeddingDeduplicator(store, config.embedding) if config.embedding.enabled else None
    if deduper is not None:
        # 起動時に埋め込みエンドポイントへ1回だけ実リクエストして疎通をログに出す。
        # 失敗しても止めない（重複排除が無効になるだけ）。並列でモデルロードが
        # 走ることもあるのでバックグラウンドで。
        threading.Thread(
            target=check_embedding, args=(config.embedding,),
            name="EmbeddingCheck", daemon=True,
        ).start()

    topic_queue: queue.Queue = queue.Queue(maxsize=20)
    script_queue: queue.Queue = queue.Queue(maxsize=5)
    # speech_queue を深くすると、その行数ぶん字幕・キャラ演出より音声が遅れて聞こえる。
    # 浅めにして「合成済みだが未再生」の作り置きを最小限にする。
    speech_queue: queue.Queue = queue.Queue(maxsize=2)
    announce_queue: queue.Queue = queue.Queue(maxsize=64)

    ring = AudioRingBuffer(capacity_frames=config.audio.samplerate * 10, channels=2)

    mixer = AudioMixer(
        state=state,
        music_ring=ring,
        speech_queue=speech_queue,
        samplerate=config.audio.samplerate,
        blocksize=config.audio.blocksize,
        device=config.audio.device,
        duck_db=config.audio.duck_db,
        duck_attack_ms=config.audio.duck_attack_ms,
        duck_release_ms=config.audio.duck_release_ms,
        duck_hold_ms=config.audio.duck_hold_ms,
        announce_queue=announce_queue,
    )
    mixer.start()
    if not config.display.enabled:
        # 画面フェードインが無い（display無効）ときは、ここで
        # 即座にフェード開始時刻を立てて自前でゆっくり音量を上げる。
        state.startup_fade_started_at = time.monotonic()

    # 番組進行。「今どのコーナーを放送しているか」はここが決め、各スレッド・画面は
    # schedule.active_content() 経由でそれを読む。番組表・リクエストが変わったら、
    # 今のトークを流し切る → 暗転 → 入れ替え → 次のコーナーの準備 → 明転 の順に切り替える。
    # 起動時のひな壇（トーク系コーナーの出演者抽選）もここで済ませる。
    director = ProgramDirector(
        config.content,
        config.cast,
        config.llm,
        state,
        store,
        mixer,
        script_queue,
        topic_queue,
        speech_queue,
        tts_backend,
        config.audio.samplerate,
        has_filler=config.filler.enabled,
    )
    schedule.set_on_air_provider(director.on_air_content)
    initial_content = director.on_air_content()

    # 起動時の局は、いまアクティブなコーナーの stream 候補から抽選する。
    # 以降の切り替え（コーナーが変わったとき／局が黙ったとき）は StationThread。
    station = StationHolder(
        random.choice(config.streams_for(initial_content)), tuned_content=initial_content
    )
    director.bind_station(station)
    state.current_station = station.stream.label

    # back_announce モードで「直前にかかっていた曲」を参照するための1件バッファ。
    prev_title: list[str] = [""]

    def on_station_change(stream) -> None:
        # 局が変われば「直前の曲」は別の局の曲。back_announce が前局の曲を
        # 振り返ってしまわないよう捨てる。
        prev_title[0] = ""
        state.now_playing = ""
        state.current_station = stream.label

    def on_title_change(title: str) -> None:
        played_out = prev_title[0]
        prev_title[0] = title
        state.now_playing = title
        logger.info("now playing: %s", title)
        # 曲をネタ化するのは、いまアクティブなコーナーが song_talk を有効にした
        # radio のときだけ。どのエントリがアクティブかで判断する（同じ type の
        # エントリが複数あっても、その時間帯の設定が使われる）。
        ac = schedule.active_content(config.content)
        if ac is None or ac.type != "radio" or ac.song_talk == "off":
            return
        back_announce = ac.song_talk == "back"
        # song_talk = "back": いま終わった曲（played_out）について、切り替わり後に話す。
        # song_talk = "intro": いまかかり始めた曲（title）について話す。
        target = played_out if back_announce else title
        if not target:
            return  # song_talk = "back" の初回は「直前の曲」がまだ無い
        try:
            topic = topic_for_now_playing(target, back_announce=back_announce)
        except Exception:
            logger.exception("musicbrainz enrichment crashed for %r", target)
            return
        if topic is None:
            # 'Artist - Title' 形式でないICYメタデータ（局のジングル等）。話しようがない
            logger.info("ICY metadata is not in 'song title' format; skipping topic generation: %r", target)
            return
        logger.info("musicbrainz topic: %s（%s）", topic.title, topic.hint)
        # 選曲紹介は同じ曲がまた流れれば改めて成立するので、意味的重複排除は掛けない
        # （掛けると2度目以降が「前と同じ内容」として消え、その曲間が無音になる）。
        enqueue_topic(
            topic, store, topic_queue, stop_event, deduper, semantic_dedup=False
        )

    # ウォッチドッグがこの2つを作り直すことがあるので、繋ぎ先は station（StationHolder）
    # から取り、作ったインスタンスを station に登録し直す。そうしないと再起動のたびに
    # 起動時の局へ戻ってしまう。
    def make_music_thread() -> MusicThread:
        t = MusicThread(
            station.stream.url, ring, samplerate=config.audio.samplerate, stop_event=stop_event
        )
        station.music = t
        return t

    def make_icy_reader() -> IcyMetadataReader:
        t = IcyMetadataReader(station.stream.url, on_title_change, stop_event=stop_event)
        station.icy = t
        return t

    def make_station_thread() -> StationThread:
        return StationThread(
            config,
            station,
            # ウォッチドッグに作り直されたときは「最後に局を選んだ」コーナーから再開する
            # （起動時のコーナーを渡すと、その場で局を抽選し直してしまう。放送中のコーナーを
            # 渡すと、切り替えの途中で作り直されたときに局の切り替えを取りこぼす）。
            initial_content=station.tuned_content,
            on_change=on_station_change,
            stop_event=stop_event,
        )

    def build_source_for(c, status: SourceStatus) -> Source | None:
        """コンテンツ設定からネタ源を1つ作る。ネタ源を持たない type は None。"""
        t = c.type
        if t == "wikimedia":
            # wikis 省略時は [locale] lang の版だけを拾う。既定が両言語だと
            # 英語放送に ja の記事が混ざる（逆も同じ）。
            return WikimediaSource(
                sample_interval_sec=c.poll_interval_sec,
                langs=c.params.get("wikis") or [config.locale.lang],
            )
        if t == "hackernews":
            return HackerNewsSource(
                poll_interval_sec=c.poll_interval_sec,
                stop_event=stop_event,
                status=status,
                max_items_per_poll=int(c.params.get("max_items_per_poll", 10)),
                fetch_article_body=bool(c.params.get("fetch_article_body", False)),
            )
        if t == "arxiv":
            return ArxivSource(
                poll_interval_sec=c.poll_interval_sec,
                stop_event=stop_event,
                status=status,
                categories=tuple(c.params.get("categories") or ("cs.AI", "cs.LG")),
                max_results=int(c.params.get("max_results", 10)),
            )
        if t == "rss":
            urls = c.params.get("rss_urls", [])
            if not urls:
                logger.warning("[[content]] type=rss has no rss_urls. Skipping")
                return None
            return RssSource(
                urls,
                poll_interval_sec=c.poll_interval_sec,
                stop_event=stop_event,
                max_entries_per_feed=int(c.params.get("max_entries_per_feed", 10)),
                max_body_chars=int(c.params.get("max_body_chars", 2000)),
                enrich_thin_bodies=bool(c.params.get("enrich_thin_bodies", True)),
                fetch_article_body=bool(c.params.get("fetch_article_body", False)),
                title_exclude=c.params.get("title_exclude"),
                status=status,
            )
        if t == "sports":
            return SportsSource(
                watch=c.params.get("sports_watch", []),
                poll_interval_sec=c.poll_interval_sec,
                stop_event=stop_event,
                status=status,
            )
        if t == "worry_consultation":
            return WorryConsultationSource(
                llm_config=config.llm,
                poll_interval_sec=c.poll_interval_sec,
                stop_event=stop_event,
                status=status,
                # 相談者の属性候補。省略時はソース側が言語ごとの既定を使う
                age_groups=c.params.get("age_groups"),
                roles=c.params.get("roles"),
                themes=c.params.get("themes"),
            )
        if t == "weather":
            if weather_provider is None:
                # [weather] が無効。config 読み込み時に警告済みなので、ここは黙って
                # スレッドを作らない（作っても毎周 None を返すだけになる）。
                return None
            return WeatherSource(
                weather_provider,
                poll_interval_sec=c.poll_interval_sec,
                stop_event=stop_event,
                status=status,
            )
        return None  # literary_reading / translated_reading / biography_reading / generated_drama / radio は SourceThread を作らない
        #              （radio の曲ネタは IcyMetadataReader のコールバックから直接入り、
        #                generated_drama は GeneratedDramaCorner が DB から直接読む）

    def build_sources() -> list[tuple[str, Source, SourceStatus]]:
        out: list[tuple[str, Source, SourceStatus]] = []
        seen: set[str] = set()
        for c in config.content:
            if not c.enabled or c.type in seen:
                continue
            # 取得状況はソースと SourceThread の両方が書くので、先に作って両方へ渡す。
            # ウォッチドッグが SourceThread を作り直しても状態が続くよう、ここで1つだけ持つ。
            # pending_check: topic_queue/script_queue に読み上げ待ちが残っているか
            # （番組表は排他型なので、残っていればほぼ確実に今アクティブなコンテンツの分）。
            status = SourceStatus(
                c.type, state,
                pending_check=lambda: not topic_queue.empty() or not script_queue.empty(),
            )
            src = build_source_for(c, status)
            if src is not None:
                out.append((c.type, src, status))
                seen.add(c.type)
        return out

    def make_source_thread_factory(content_type: str, source: Source, status: SourceStatus):
        def factory() -> SourceThread:
            return SourceThread(
                source,
                topic_queue,
                store,
                stop_event=stop_event,
                deduper=deduper,
                is_active=lambda: director.collecting(content_type),
                status=status,
            )

        return factory

    # 読書コーナー（v4 §10）。corner はセッション状態を持つので、ウォッチドッグで
    # ScriptThread が作り直されても使い回せるよう、ここで1つだけ生成する。
    literary_reading_content = config.content_by_type("literary_reading")
    literary_reading_corner = (
        LiteraryReadingCorner(literary_reading_content, store, state, config.cast, config.llm)
        if literary_reading_content is not None
        else None
    )

    # 翻訳朗読コーナー（Project Gutenberg → 翻訳ナレーション）。読書コーナーと同じ理由で1つだけ生成する。
    translated_reading_content = config.content_by_type("translated_reading")
    translated_reading_corner = (
        TranslatedReadingCorner(translated_reading_content, store, state, config.cast, config.llm)
        if translated_reading_content is not None
        else None
    )

    # 偉人伝トーク（Wikipedia）。読書コーナーと同じ理由で1つだけ生成する。
    biography_reading_content = config.content_by_type("biography_reading")
    biography_reading_corner = (
        BiographyReadingCorner(biography_reading_content, store, state, config.cast, config.llm)
        if biography_reading_content is not None
        else None
    )

    # ラジオドラマ朗読（v6 §4.7）。執筆は別プロセス（generated_drama_writer）が済ませているので、
    # ここは確定済み本文を読むだけ。読書コーナーと同じくシーンの進行状態を持つため、
    # ウォッチドッグで ScriptThread が作り直されても使い回せるよう1つだけ生成する。
    generated_drama_content = config.content_by_type("generated_drama")
    generated_drama_corner = (
        GeneratedDramaCorner(generated_drama_content, store, state, config.cast, director)
        if generated_drama_content is not None
        else None
    )

    # 日替わりの「今日のキャラごとの気分」（持ちネタキャラバリエーション §2）。
    # weather.py 型の遅延キャッシュで、ScriptThread が作り直されても使い回せるよう
    # ここで1つだけ生成する。既定 off。[llm] daily_mood = true のときだけ作る
    # （作らなければ mood は台本に一切影響しない）。
    mood_provider: CastMoodProvider | None = None
    if config.llm.daily_mood:
        logger.info("Enabling daily mood ([llm] daily_mood = true)")
        mood_provider = CastMoodProvider(
            config.llm,
            config.cast,
            lang=config.locale.lang,
            weather=weather_provider,
        )

    def make_script_thread() -> ScriptThread:
        return ScriptThread(
            topic_queue,
            script_queue,
            store,
            config.llm,
            config.cast,
            director,
            stop_event=stop_event,
            literary_reading_corner=literary_reading_corner,
            translated_reading_corner=translated_reading_corner,
            biography_reading_corner=biography_reading_corner,
            generated_drama_corner=generated_drama_corner,
            mood_provider=mood_provider,
        )

    def make_tts_thread() -> TTSThread:
        return TTSThread(
            script_queue,
            speech_queue,
            tts_backend,
            config.cast_by_id,
            config.audio.samplerate,
            director,
            stop_event=stop_event,
            state=state,
        )

    def make_announce_thread() -> AnnounceThread:
        return AnnounceThread(
            announce_queue,
            state,
            config.cast,
            stop_event=stop_event,
            store=store,
        )

    def make_director_thread() -> DirectorThread:
        return DirectorThread(director, stop_event=stop_event)

    factories = {
        "DirectorThread": make_director_thread,
        # StationThread はコーナー切り替えの局選定（tuned_content の更新）を担うので、
        # music_enabled = false でも動かし続ける（director の切り替え待ちが詰まらないよう）。
        # 実際に音を鳴らす MusicThread / ICYメタデータ取得（IcyMetadataReader）だけ止める。
        "StationThread": make_station_thread,
        "ScriptThread": make_script_thread,
        "TTSThread": make_tts_thread,
        "AnnounceThread": make_announce_thread,
    }
    if config.audio.music_enabled:
        factories["MusicThread"] = make_music_thread
        factories["IcyMetadataReader"] = make_icy_reader
    else:
        logger.info("audio.music_enabled = false: not playing net radio BGM (talk audio only)")
    for content_type, source, status in build_sources():
        factories[f"SourceThread:{source.name}"] = make_source_thread_factory(
            content_type, source, status
        )

    if config.filler.enabled:

        def make_filler_thread() -> FillerThread:
            return FillerThread(
                script_queue,
                state,
                config.content,
                config.cast,
                config.llm,
                director,
                idle_threshold_sec=config.filler.idle_threshold_sec,
                check_interval_sec=config.filler.check_interval_sec,
                filler_config=config.filler,
                weather=weather_provider,
                stop_event=stop_event,
            )

        factories["FillerThread"] = make_filler_thread

    # ラジオドラマの自動執筆（§4.7.6）。type = "generated_drama" の auto_write = true のときだけ、
    # 放送が暇な隙に generated_drama_writer を子プロセスで走らせる。放送プロセスと執筆プロセスは
    # 分けたまま（GPU 取り合いの回避）だが、起動はユーザーの手を離れる。
    if (
        generated_drama_content is not None
        and generated_drama_content.generated_drama is not None
        and generated_drama_content.generated_drama.auto_write
    ):

        def make_generated_drama_supervisor() -> GeneratedDramaSupervisorThread:
            return GeneratedDramaSupervisorThread(
                args.config,
                config.content,
                state,
                script_queue,
                store,
                director,
                stop_event=stop_event,
            )

        factories["GeneratedDramaSupervisorThread"] = make_generated_drama_supervisor
        logger.info(
            "generated_drama auto-write enabled; "
            "GeneratedDramaSupervisorThread will run generated_drama_writer when idle"
        )

    watchdog = WatchdogThread(factories, check_interval_sec=5.0, stop_event=stop_event)
    for name, factory in factories.items():
        t = factory()
        t.start()
        watchdog.register(name, t)
    watchdog.start()

    logger.info(
        "all threads started. stream=%s model=%s tts=%s",
        station.stream.label,
        config.llm.model,
        config.tts.host,
    )

    try:
        if config.display.enabled:
            from .display.app import run_display

            run_display(
                state,
                config.display,
                config.cast,
                config.llm.model,
                config.llm.engine_label,
                config.content,
                director,
                music_enabled=config.audio.music_enabled,
            )
        else:
            while not stop_event.is_set():
                time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("shutting down (Ctrl+C)")
    finally:
        stop_event.set()
        watchdog.stop_all()
        watchdog.join(timeout=5.0)
        mixer.stop()
        store.close()
        logger.info("stopped")


if __name__ == "__main__":
    main()
