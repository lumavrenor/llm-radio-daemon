"""config.toml の読み込み。ハードコード禁止 — すべての可変値はここを経由する。

設定は言語ごとに ``config/<lang>/`` へ3点セットで置く::

    config/ja/config.toml   config/ja/config_cast.toml   config/ja/config_content.toml
    config/en/config.toml   config/en/config_cast.toml   config/en/config_content.toml

``load_config()`` に渡すのは config.toml のパスだけでよい。cast と content は
同じディレクトリから引く（``Path.with_name()``）ので、言語の切り替えは
``--config`` のパス1本で済む。ファイル内の相対パス（``models/`` ``content/``
``data/``）はカレントディレクトリ基準なので、リポジトリルートから実行すること。
"""

from __future__ import annotations

import logging
import os
import platform
import re
import sys
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from . import language, sensitive
from .display.models import (
    DEFAULT_MODEL,
    known_accessories,
    known_accessory_sides,
    known_hair_styles,
    known_models,
)
from .schedule import parse_window
from .weather import DEFAULT_FORECAST_URL

_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")

# --config には既定値を置かない。ja を既定にすると「どの言語で動いているか」が
# 引数から読めなくなり、英語版のつもりで日本語版の DB を叩く事故につながる。
# 使い方の文言はここに集約する（各エントリポイントの argparse が参照する）。
CONFIG_ARG_HELP = (
    "設定ファイル config/<lang>/config.toml へのパス（必須）。"
    "config_cast.toml / config_content.toml は同じディレクトリから読むので、"
    "言語の切り替えはこのパス1本で済む。例: config/ja/config.toml"
    " / Path to config/<lang>/config.toml (required). config_cast.toml and"
    " config_content.toml are read from the same directory, so switching"
    " language is just this one path. Example: config/en/config.toml"
)


class ConfigError(RuntimeError):
    pass


# --- 推論エンジン -----------------------------------------------------------
#
# [llm] / [embedding] の engine で選ぶ。API の叩き方は 4 系統:
#   "ollama"   … ネイティブ API（/api/chat, /api/embeddings）
#   "openai"   … OpenAI 互換 API（/v1/chat/completions, /v1/embeddings）。LM Studio・OpenAI。
#   "llamacpp" … llama.cpp サーバ（llama-server）。エンドポイントは openai と同じだが
#                structured output は response_format を {"type":"json_object","schema":…}
#                で渡し（llama-server で一番安定して効く形）、thinking 抑止に
#                chat_template_kwargs={"enable_thinking": false} も併せて送る。
#   "unsloth"  … Unsloth Studio。openai と同じエンドポイント・同じ json_schema 形だが、
#                中身は llama.cpp なので thinking 対応モデル（gemma-4 等）では
#                reasoning_effort だけでは思考が止まらず、本文が空／途中切れになる。
#                llamacpp と同じく chat_template_kwargs={"enable_thinking": false} を送る。
#                埋め込み（/v1/embeddings）もあるが、モデルは UI 側設定で固定・request の
#                model は無視され、既定は英語専用の bge-small。日本語放送では [embedding] を
#                ollama にするのを推奨（EmbeddingConfig.__post_init__ が warning を出す）。
# 実際の HTTP は llm_http.py が api_style を見て振り分ける。認証（Authorization: Bearer）は
# api_style とは独立で、[llm] / [embedding] の api_key が空でなければ openai/llamacpp/unsloth
# 経路で付く（Unsloth は Keyless 設定なら鍵不要。LAN 公開・Colab のときだけ要る）。

_OLLAMA_ENGINES = frozenset({"ollama"})
_OPENAI_ENGINES = frozenset({"lmstudio", "lm-studio", "openai"})
_LLAMACPP_ENGINES = frozenset({"llamacpp", "llama.cpp", "llama_cpp", "llama-cpp"})
_UNSLOTH_ENGINES = frozenset({"unsloth", "unsloth-studio", "unsloth_studio"})

_ENGINE_LABELS = {
    "ollama": "Ollama",
    "lmstudio": "LM Studio",
    "lm-studio": "LM Studio",
    "llamacpp": "llama.cpp",
    "llama.cpp": "llama.cpp",
    "llama_cpp": "llama.cpp",
    "llama-cpp": "llama.cpp",
    "openai": "OpenAI",
    "unsloth": "Unsloth",
    "unsloth-studio": "Unsloth",
    "unsloth_studio": "Unsloth",
}


def engine_api_style(engine: str) -> str:
    """engine 名 → "ollama" / "openai" / "llamacpp" / "unsloth"。未知なら ConfigError。"""
    e = (engine or "").lower()
    if e in _OLLAMA_ENGINES:
        return "ollama"
    if e in _OPENAI_ENGINES:
        return "openai"
    if e in _LLAMACPP_ENGINES:
        return "llamacpp"
    if e in _UNSLOTH_ENGINES:
        return "unsloth"
    raise ConfigError(
        f"engine = {engine!r} は未知の推論エンジンです"
        f"（対応: ollama / lmstudio / llamacpp / openai / unsloth） / "
        f"unknown inference engine {engine!r} (supported: ollama / lmstudio / llamacpp / openai / unsloth)"
    )


def _resolve_api_key(value: str, *, section: str) -> str:
    """api_key の値を解決する。``env:NAME`` なら環境変数 NAME を引く。

    config.toml は git 管理下に置かれることが多いので、鍵を直書きせず
    ``api_key = "env:UNSLOTH_API_KEY"`` と書けるようにしておく。参照先の環境変数が
    無ければ起動時に落とす（他の設定ミスと同じく、走らせてから気づくより安い）。
    """
    value = (value or "").strip()
    if not value.startswith("env:"):
        return value
    name = value[len("env:"):].strip()
    if not name:
        raise ConfigError(
            f'{section} api_key = "env:" に環境変数名がありません / '
            f'{section} api_key = "env:" is missing an environment variable name'
        )
    try:
        return os.environ[name]
    except KeyError:
        raise ConfigError(
            f'{section} api_key = "env:{name}" ですが環境変数 {name} が未設定です / '
            f'{section} api_key = "env:{name}" but the environment variable {name} is not set'
        ) from None


# モデルの実行場所。トークで「このパソコンの中で動いている」と名乗るかどうかがこれで変わる。
# config に書けるのは auto / local / cloud の3つ。"auto" は起動時に
# llm_http.detect_placement() の結果（local / cloud / unknown）へ潰される。
_PLACEMENTS = frozenset({"auto", "local", "cloud", "unknown"})
_PLACEMENT_LABELS = {"local": "ローカル", "cloud": "クラウド"}


@dataclass
class LLMConfig:
    host: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5:14b-instruct-q4_K_M"
    keep_alive: int = -1
    temperature: float = 0.9
    timeout_sec: int = 60
    # 推論エンジン。ollama（ネイティブ）/ lmstudio / llamacpp / openai / unsloth（OpenAI 互換）。
    # トーク内で名乗る表示ラベルにもなる（engine_label）。
    engine: str = "ollama"
    # OpenAI 互換エンジンの API キー（Authorization: Bearer ヘッダ）。空なら付けない。
    # LM Studio / llama.cpp は不要。Unsloth Studio は localhost でも必須
    # （Settings → API で sk-unsloth-… を発行）。OpenAI 本体や OpenRouter 等でも使える。
    # 鍵を直書きしたくなければ api_key = "env:UNSLOTH_API_KEY" のように環境変数名を書ける
    # （参照先が未設定なら起動時にエラー）。__post_init__ で実値へ解決する。
    api_key: str = ""
    # モデルの実行場所。"auto"（起動時に判定）/ "local" / "cloud"。
    # 番組は「このパソコンの中の生成AIが喋っている」ことを楽屋ネタにするが、ollama の
    # `<model>-cloud` はローカルと同じ host から使えてしまうため、名乗りが嘘になり得る。
    # 起動時に main.py が llm_http.detect_placement() で実値（local / cloud / unknown）へ
    # 潰し、filler.py がその値に合わせて台詞とプロンプトを切り替える。
    placement: str = "auto"
    # 日替わり mood（mood.py / CastMoodProvider・持ちネタキャラバリエーション §2）を有効にするか。
    # 天気・季節・曜日から「今日のキャラごとの気分」を1日1回作り、全コーナーの台本へ薄く効かせる。
    # 既定 off。roster の注記を「読み上げるセリフ」として扱う小型モデル（gemma3n e2b/e4b 級）では
    # 戯画化・話題の脱線を招くため、20〜30b 級モデルで使うとき明示的に true にする。
    daily_mood: bool = False
    # 字幕とは別に、TTS用の読み上げテキストをLLMでもう1回生成するか（誤読対策・
    # script/reading.py）。字幕は常に生成された原文のまま、TTSだけ読み間違えやすい
    # 漢字を書き換えた版を読む。既定 off。現状メイン雑談コーナー（script/ollama_client.py）
    # のみ対応。本編と同じ model へもう1往復LLMを呼ぶため、台本生成そのものが
    # 遅くなる（実測: 20行の台本で+15秒前後）。大きめのモデルを遅いGPUで動かしていると
    # 台本1本あたりの生成時間が伸びて体感に響くので、そのときだけ true にする想定。
    # 実測: gemma3n/gemma4 の e2b（2B）級は「一部の語だけ直す」選択的編集ができず、
    # ほぼ全文をかな化した上に誤変換もする（1行ずつ投げても同じ）。
    #
    # 【2026-09-20】台本生成と同じ1回の呼び出しへ統合する案（低温固定を諦める代わりに
    # 追加往復を無くす）を試したが、qwen3.8:27b でも本編と同じ温度（既定0.9）で
    # speech_text を生成すると、この指示を無視して行全体をかな化し、誤変換も発生した
    # （例: 「大胆な彩色」→「だれんなさいしき」）。低温固定の別呼び出し方式に戻し、
    # 統合版は不採用。
    speech_reading_pass: bool = False
    # コンテキスト長（num_ctx ＝ プロンプト＋生成トークンの上限）。ollama のみ有効
    # （OpenAI 互換・llama.cpp・Unsloth Studio ではサーバ起動時に決まるのでここでは無視される。
    #  Unsloth は起動時 -c を小さくしすぎると台本が途中で切れる。8192〜16384 を目安に）。
    # 0 なら ollama の既定（環境変数 OLLAMA_CONTEXT_LENGTH。未設定なら 4096）に従う。
    # この番組はネタ本文（最大2000字）＋ロースター＋定型文で入力が数千トークンになり、
    # さらに 10〜20 行の JSON 台本を書かせるため、4096 だと生成が途中で切れて台本が
    # 没になり（→フィラー送り）やすい。8192〜16384 を明示するのを推奨。KV キャッシュの
    # VRAM 増分は 9〜20B 級で 1 万トークンあたり 0.5〜1GB 程度。
    num_ctx: int = 0
    # 1 回の生成のトークン上限（ollama: options.num_predict / OpenAI 互換: max_tokens）。
    # 0 ならエンジン既定（ollama は -1 ＝ コンテキストの残り全部まで書ける）。
    # 呼び出し側が明示指定したとき（執筆ハーネス等）はそちらが優先される。
    # モデルが JSON を閉じ損ねて延々書き続けるのを止める保険。num_ctx を絞ったときは
    # 「入力 ＋ num_predict ≦ num_ctx」になるよう、こちらも合わせて指定するとよい。
    num_predict: int = 0
    # 反復抑制（ollama: options.repeat_penalty / frequency_penalty）。0 でエンジン既定。
    # ⚠ この番組は全コーナーが JSON Schema 出力なので、実質使えない。ペナルティが JSON の
    # 構造トークン（" : , { } speaker text）にも効いて出力が壊れる（実測: 1行だけの台本・
    # 断片混入）。structured output を使わない別用途向けに残してあるだけ。
    # 台本のループは script/ollama_client.py 側のプロンプト文言で抑えること。
    repeat_penalty: float = 0.0
    frequency_penalty: float = 0.0
    # LLM へ送る最終プロンプトをログ（[log] level = INFO 以上）へ丸ごと吐くデバッグ用スイッチ。
    # 既定 off（コーナーによっては記事本文込みで数千字になり、24時間稼働のログを圧迫するため）。
    # 中身を覗きたいときだけ一時的に true にする。llm_http.chat() 経由の全呼び出し（台本・
    # フィラー・ネタだし等）が対象で、structured output のスキーマをプロンプトへ書き足す
    # フォールバック経路（llm_http._schema_in_prompt）を使う場合はその分も含めて出る。
    log_prompts: bool = False
    # LLM からの生の応答をログ（[log] level = INFO 以上）へ丸ごと吐くデバッグ用スイッチ。
    # log_prompts と対になるが別スイッチ（プロンプトだけ・応答だけ・両方、を選べるように）。
    # 既定 off。structured output の JSON をそのまま出す（コードフェンス剥がし等の後処理前）。
    log_responses: bool = False
    # structured output で使う JSON Schema をログ（[log] level = INFO 以上）へ吐くデバッグ用
    # スイッチ。schema はプロンプト文字列ではなく別パラメータ（ollama: format /
    # openai・unsloth: response_format.json_schema / llamacpp: response_format.schema）として
    # 送っているため、log_prompts を true にしてもログには出てこない。中身を確認したいときだけ
    # 一時的に true にする。既定 off（コーナーごとに固定のスキーマで、頻繁には変わらないため）。
    log_schema: bool = False

    def __post_init__(self) -> None:
        engine_api_style(self.engine)  # 未知エンジンならここで ConfigError
        self.api_key = _resolve_api_key(self.api_key, section="[llm]")
        if self.placement not in _PLACEMENTS:
            raise ConfigError(
                f"[llm] placement = {self.placement!r} は不正です"
                f"（対応: auto / local / cloud） / "
                f"invalid; must be one of: auto / local / cloud"
            )
        if self.num_ctx < 0:
            raise ConfigError(
                f"[llm] num_ctx = {self.num_ctx!r} は 0 以上で指定してください"
                f"（0 でエンジン既定に従う） / "
                f"must be 0 or greater (0 = follow the engine's default)"
            )
        if self.num_predict < -2:
            raise ConfigError(
                f"[llm] num_predict = {self.num_predict!r} は -2 以上で指定してください"
                f"（0 でエンジン既定、-1 でコンテキスト上限まで） / "
                f"must be -2 or greater (0 = engine default, -1 = up to the context limit)"
            )
        if self.repeat_penalty < 0:
            raise ConfigError(
                f"[llm] repeat_penalty = {self.repeat_penalty!r} は 0 以上で指定してください"
                f"（0 でエンジン既定、1.0 で無効、1.2〜1.3 が目安） / "
                f"must be 0 or greater (0 = engine default, 1.0 = disabled, 1.2-1.3 is a typical value)"
            )

    @property
    def api_style(self) -> str:
        return engine_api_style(self.engine)

    @property
    def engine_label(self) -> str:
        """トークで名乗る用の推論エンジン名。未知の値はそのまま返す。"""
        return _ENGINE_LABELS.get(self.engine.lower(), self.engine)

    @property
    def placement_label(self) -> str:
        """トークで名乗る用の実行場所（ローカル / クラウド）。不明なら空文字。

        空文字は「実行場所には触れさせない」の意味。判定に失敗したまま
        どちらかを断定して喋らせるより、黙らせる方が事故が小さい。
        """
        return _PLACEMENT_LABELS.get(self.placement, "")


@dataclass
class LocaleConfig:
    """番組全体の言語・地域設定。[[content]] 側の個別設定（Wikipedia版など）は
    ここから注入される。将来 en 版などを作るときの切り替えスイッチもここに集約する。
    """

    lang: str = "ja"  # ISO 639-1。biography_reading の Wikipedia 言語版などに使う


@dataclass
class TTSConfig:
    backend: str = "voicevox"  # "voicevox"（日本語）/ "kokoro"（英語）
    # --- backend = "voicevox" のとき使う ---
    host: str = "http://127.0.0.1:50021"
    timeout_sec: int = 10
    # --- backend = "kokoro" のとき使う ---
    # モデル（kokoro-v1.0.onnx / voices-v1.0.bin）の置き場。初回起動時に自動取得する。
    model_dir: str = "kokoro_models"
    # onnxruntime のスレッド数。0 で onnxruntime 任せ。
    # 実測（i7-7700K 4コア）で 2 スレッド 2.0x / 4 スレッド 2.5x リアルタイム。
    # 全コアを与えると LLM のサンプリングと 3D 表示を食うので、既定は控えめの 2。
    threads: int = 2
    # 読みを明示したい語の辞書（1行 "word<TAB>IPA"）。省略・不在なら何もしない。
    # フォールバック G2P は「破綻しない読み」を保証するが「正しい読み」ではないので、
    # 番組の看板になる語（モデル名・常連アーティスト名）だけここで押さえる。
    lexicon_file: str = ""


@dataclass
class AudioConfig:
    samplerate: int = 48000
    device: str = ""
    blocksize: int = 1024
    duck_db: float = -12.0
    duck_attack_ms: int = 150
    duck_release_ms: int = 500
    duck_hold_ms: int = 300
    # false にするとネットラジオのBGM（MusicThread / ICYメタデータ取得）を一切起動しない。
    # トーク音声だけが流れる（YouTube 収録などで曲を著作権的に含めたくない場合向け）。
    # song_talk（曲紹介トーク）は曲名が来なくなるので自然に発生しなくなる。
    music_enabled: bool = True


# --- 背景演出 -----------------------------------------------------------------
#
# 3D 表示（ursina）の背景に流す暗めのアニメーション。実装は display/background.py。
# `+` で重ねられる（例: "grid+dust"）。"none" / 空 で背景なし。
# [display] background が既定値で、[[content]] の background がコーナーごとに上書きする。

BACKGROUND_ATOMS = (
    "grid",      # 奥から手前へ流れるワイヤーフレームの格子床
    "dust",      # ゆっくり周回する微粒子
    "logs",      # daemon 自身のログを薄く流す
    "coderain",  # Matrix 風の落下文字
)

DEFAULT_BACKGROUND = "grid+dust"


def parse_background(spec: str) -> tuple[str, ...]:
    """"grid+dust" → ("grid", "dust")。空／"none" は ()。未知の名前は ValueError。"""
    spec = (spec or "").strip().lower()
    if spec in ("", "none"):
        return ()
    atoms = tuple(a.strip() for a in spec.split("+") if a.strip())
    unknown = [a for a in atoms if a not in BACKGROUND_ATOMS]
    if unknown:
        raise ValueError(
            f"未知の背景 {', '.join(repr(u) for u in unknown)}"
            f"（使えるのは {', '.join(BACKGROUND_ATOMS)}、"
            f'"+" で重ねられる。無効化は "none"） / '
            f"unknown background {', '.join(repr(u) for u in unknown)} "
            f"(supported: {', '.join(BACKGROUND_ATOMS)}; combine with \"+\"; \"none\" disables it)"
        )
    return atoms


@dataclass
class DisplayConfig:
    enabled: bool = False
    fps_limit: int = 30
    # 背景演出の既定値。[[content]] の background で上書きできる。
    background: str = DEFAULT_BACKGROUND
    # 既定のフォントは日本語グリフを含まないため、CJK対応フォントを明示指定する。
    font: str = "C:/Windows/Fonts/meiryo.ttc"
    # ウィンドウの位置・サイズを終了時に覚え、次回起動時に復元する。
    # 最大化状態は Panda3D の API に無いので復元されない（通常サイズのみ）。
    remember_window: bool = True
    window_state_path: str = "data/window_state.json"
    # 字幕。日本語は空白で折り返せないため文字数で手動改行する。
    subtitle_wrap: int = 49       # 1行あたりの全角文字数のめやす。画面幅の 9 割ほどを狙う
    subtitle_max_lines: int = 10  # 表示する最大行数。あふれた分は末尾を … で丸める
    #   朗読チャンクは 1 つ約 400〜500 字（[literary_reading] chunk_target_chars）なので、
    #   全文を出すには 49 字 × 10 行 ≒ 490 字ぶんの枠が要る。通常トークでは
    #   必要な行数しか描かないので、この上限を上げても字幕は縦に伸びない。
    # トークが途切れて誰も喋っていない状態がこの秒数だけ続いたら、最後の字幕を消す
    # （次の発話が始まるまでの間、直前の台詞が居座って見た目が悪いのを防ぐ）。
    subtitle_idle_clear_sec: float = 10.0


# --- 天気 ---------------------------------------------------------------------
#
# 場所は config に明示する（IP ジオロケーションも OS の位置情報も使わない）。
# この番組にとって天気は「リスナーの現在地」ではなく「スタジオの所在地」という
# 番組設定であり、勝手に推測して当てにいくものではないため。取得と鮮度管理は
# weather.py（WeatherProvider）が持ち、ここは設定の入り口だけ。

@dataclass
class WeatherConfig:
    """``[weather]``。フィラーの雑談とお天気コーナー（type = "weather"）が共有する。

    ``enabled = false``、あるいは緯度経度・地名が欠けていれば天気には一切触れない
    （``[llm] placement = "unknown"`` と同じ扱い）。
    """

    enabled: bool = False
    location_name: str = ""          # トークで呼ぶ地名。座標は読み上げられないのでこれが必須
    latitude: float | None = None
    longitude: float | None = None
    refresh_sec: float = 1800.0      # 取得間隔。天気は変化が遅いので取りに行きすぎない
    timeout_sec: float = 5.0         # 放送スレッドを止めないよう短く。失敗は無言でフォールバック
    forecast_url: str = DEFAULT_FORECAST_URL  # 差し替え用（通常は書かない）


@dataclass
class FillerConfig:
    enabled: bool = True
    idle_threshold_sec: float = 20.0  # script_queueがこの秒数空のままならフィラーを投入
    check_interval_sec: float = 2.0


@dataclass
class EmbeddingConfig:
    enabled: bool = True
    host: str = "http://127.0.0.1:11434"
    model: str = "nomic-embed-text"
    keep_alive: int = -1  # 常駐。外すとアイドル時にアンロードされ、次回呼び出しがロード待ちになる
    similarity_threshold: float = 0.85  # これを超えたら「もう話した内容とほぼ同じ」として破棄
    max_recent: int = 1000  # 直近何件のトピックと比較するか
    timeout_sec: int = 10
    engine: str = "ollama"  # ollama / lmstudio / llamacpp / openai。[llm] とは独立に選べる
    api_key: str = ""  # OpenAI 互換エンジンの API キー。[llm] api_key と同じ書式（env: 参照可）

    def __post_init__(self) -> None:
        engine_api_style(self.engine)  # 未知エンジンならここで ConfigError
        if self.enabled and self.engine.lower() in _UNSLOTH_ENGINES:
            # Unsloth Studio は /v1/embeddings を持つ（keyless の "Chat and inference" で叩ける）が、
            # 使う埋め込みモデルは UI の Settings → API 側で固定されていて、リクエストの model
            # フィールドは無視される（[embedding] model の指定が効かない）。既定は
            # bge-small-en-v1.5 で英語専用 —— 日本語放送では類似度がほぼ無意味になる（実測:
            # 無関係な文どうしが 0.8 台）。落としはしないが気づけるようにする。
            logging.getLogger(__name__).warning(
                "[embedding] engine = %r: 埋め込みモデルは Unsloth Studio 側の設定で決まり、"
                "[embedding] model = %r は無視されます。既定は英語専用の bge-small なので、"
                "日本語放送では [embedding] engine = \"ollama\"（nomic-embed-text 等）を推奨します。"
                "重複判定を使わないなら enabled = false に。",
                self.engine, self.model,
            )
        self.api_key = _resolve_api_key(self.api_key, section="[embedding]")

    @property
    def api_style(self) -> str:
        return engine_api_style(self.engine)


# --- コンテンツ（番組編成） ----------------------------------------------------
#
# 番組表型（排他）。config.toml の [[content]] を記述順に評価し、enabled かつ
# schedule の時間帯に入っている最初のエントリだけが「今アクティブなコンテンツ」に
# なる（schedule 省略 = 常時対象）。時間帯が重なったら先に書いた方が勝つ。

CONTENT_TYPES = (
    "literary_reading",
    "translated_reading",
    "biography_reading",
    "generated_drama",
    "worry_consultation",
    "hackernews",
    "arxiv",
    "wikimedia",
    "rss",
    "sports",
    "weather",
    "radio",
)

# type = "radio" の「曲についてどれだけ喋るか」。ネットラジオが鳴っている区間は
# これ1つで決まる（かつては喋らない "radio" と喋る "musicbrainz" の2種別に分かれていた）。
#   off   … 喋らない。音楽だけ流す
#   intro … 曲が始まったら、その曲をネタに通常のひな壇トーク（フィラーも出る）
#   back  … 曲が終わってから短く振り返るだけ（back-announce）。フィラーは出さず、
#           代わりに mid_song_chat で曲中にひとこと挟める
SONG_TALK_MODES = ("off", "intro", "back")

# [[content]] の共通キー。これ以外は params（type 固有設定）へ回す。
_CONTENT_COMMON_KEYS = frozenset(
    {
        "type",
        "schedule",
        "enabled",
        "label",
        "tone_hint",
        "host_id",
        "min_speakers",
        "max_speakers",
        "poll_interval_sec",
        "background",
        "stream",
        "song_talk",
    }
)


@dataclass
class LiteraryReadingParams:
    """読書コーナー — v4 §10.9。[[content]] type = "literary_reading" の固有設定。

    朗読キャラ・つっこみキャラは他コンテンツと同じく [[cast]] からランダム抽選される
    （抽選 1 人目＝朗読、2 人目＝つっこみ）。

    原典は [locale] lang で決まる。ja = 青空文庫、en = Project Gutenberg
    （llm_radio_daemon/gutenberg/__init__.py 参照）。**corpus_dir の既定は
    青空文庫向けなので、英語版では config で必ず別ディレクトリを指すこと**
    （既定のままだと英語の本が data/aozora へ溜まる）。
    """

    corpus_dir: str = "data/aozora"      # テキスト／インデックスのキャッシュ置き場
    auto_fetch: bool = True              # corpus_dir に無い作品を実行時に自動取得（§10.2）
    prefetch_next: bool = True           # 現在作品の朗読中に次作品を先読み取得
    work_selector: str = "random"        # "random" | "llm"（候補リストから LLM に選ばせる・§10.2）
    llm_candidate_count: int = 20        # work_selector="llm" のとき候補として渡す最大件数
    beat_interval: int = 3               # 何チャンクごとに感想（Beat）を挟むか
    chunk_target_chars: int = 400
    chunk_max_chars: int = 500
    chunk_min_chars: int = 150
    summary_max_chars: int = 800
    comment_lines: tuple[int, int] = (4, 8)  # 感想の行数レンジ
    duck_db: float = -18.0               # 朗読中の BGM 減衰（通常トークは audio.duck_db）
    pause_ms: int = 400                  # チャンク間の無音
    pause_chapter_ms: int = 800          # 章題の後
    pause_beat_ms: int = 600             # 感想パートの前後
    reread_after_days: int = 30          # 同じ作品を再読しない期間
    allow_translations: bool = False     # 訳者の著作権が残りうるため既定 false（青空文庫のみ。
                                         # Gutenberg のカタログには訳者の役割が無く、配っている
                                         # 訳文は訳者ぶんも米国 PD なので効かない）
    ruby_mode: str = "kana"              # "kana"（かな置換）| "as-is"（漢字のまま）。青空文庫のみ
    # 以下2つは Project Gutenberg（en）のみ。省略すると gutenberg/fetch.py の既定
    # （本家 gutenberg.org）を使う。本家は自動取得をまとめてやられるのを嫌っており
    # ミラーの利用も案内しているので、差し替えられるようにしてある
    catalog_url: str = ""                # pg_catalog.csv.gz の URL
    text_url_template: str = ""          # 本文 URL。"{id}" が作品IDに置き換わる
    works: list[dict] = field(default_factory=list)  # [[content.works]] 明示指定（省略時ランダム）


@dataclass
class TranslatedReadingParams:
    """翻訳朗読コーナー — [[content]] type = "translated_reading" の固有設定。

    docs/idea-translated-reading.md の案1。literary_reading と同じ Project Gutenberg
    （英語）を素材にするが、原文をそのまま読まず**チャンクごとに翻訳しながら
    ナレーションする**（biography_reading と同じ「全チャンクがLLM入力」の構造）。
    ja 放送で英語圏の作品を紹介するためのコーナーなので、corpus_dir の既定は
    literary_reading と違って言語に依らず常に ``data/gutenberg``（英語原文）。

    セッション管理（DB永続化・再起動時の再開・要約の持ち越し）は literary_reading と
    同じ考え方だが、テーブルは ``translated_reading_sessions`` / ``_log`` として別に持つ
    （literary_reading と時間帯が重ならないだけで、同じ本を並行して2コーナーが
    読むこともあり得るため状態を混ぜない）。
    """

    corpus_dir: str = "data/gutenberg"   # テキスト／インデックスのキャッシュ置き場（常に英語原文）
    auto_fetch: bool = True              # corpus_dir に無い作品を実行時に自動取得
    prefetch_next: bool = True           # 現在作品の翻訳中に次作品を先読み取得
    work_selector: str = "random"        # "random" | "llm"（候補リストから LLM に選ばせる）
    llm_candidate_count: int = 20        # work_selector="llm" のとき候補として渡す最大件数
    chunk_target_chars: int = 600        # 1回のLLM呼び出しに渡す原文（英語）の目標文字数
    chunk_max_chars: int = 800
    chunk_min_chars: int = 200
    translate_lines: tuple[int, int] = (1, 4)  # 1チャンクの翻訳ナレーションの行数レンジ
    beat_interval: int = 4               # 何チャンク訳すごとに感想（Beat）を挟むか
    comment_lines: tuple[int, int] = (4, 8)  # 感想の行数レンジ
    summary_max_chars: int = 800
    pause_ms: int = 400                  # チャンク間の無音
    pause_chapter_ms: int = 800          # 章題の後
    pause_beat_ms: int = 600             # 感想パートの前後
    reread_after_days: int = 30          # 同じ作品を再読しない期間
    catalog_url: str = ""                # pg_catalog.csv.gz の URL（省略時 gutenberg/fetch.py の既定）
    text_url_template: str = ""          # 本文 URL。"{id}" が作品IDに置き換わる
    works: list[dict] = field(default_factory=list)  # [[content.works]] 明示指定（省略時ランダム）


@dataclass
class BiographyReadingParams:
    """偉人伝トーク — [[content]] type = "biography_reading" の固有設定。

    Wikipedia記事を先頭から順にMCが解説していく。literary_reading と違い朗読は
    せず、全チャンクがLLM入力になる（MCの解説自体がその場で生成される）。
    対象人物は自動選定せず figures_file に明示指定する（09-05相談。
    「政治・宗教に無関係・1950年より前」をLLM任せの選定では守り切れないため）。
    """

    lang: str = "ja"                      # 取得する Wikipedia の言語版。[locale] lang から注入される
                                           # （config_content.toml には書かない。内部用）
    # 以下2つは省略時 [locale] lang から data/wiki_bio_<lang> /
    # content/biography_figures.<lang>.txt を組み立てる。既定値を ja に固定すると
    # 英語放送が日本語の人物リストとキャッシュを掴む事故になる。
    data_dir: str = ""                    # 取得済み本文のキャッシュ置き場
    figures_file: str = ""                # 対象人物リスト（1行1人＝Wikipedia記事タイトル、# で始まる行と空行は無視）
    chunk_target_chars: int = 700
    chunk_max_chars: int = 1000
    chunk_min_chars: int = 200
    summary_max_chars: int = 800
    segment_lines: tuple[int, int] = (5, 10)  # 1チャンクぶんの解説トークの行数レンジ
    reread_after_days: int = 60           # 同じ人物を再放送しない期間
    figures: list[dict] = field(default_factory=list)  # figures_file から読み込んだ結果（内部用。configには書かない）


@dataclass
class GeneratedDramaParams:
    """ラジオドラマ朗読 — v6 §4.7。[[content]] type = "generated_drama" の固有設定。

    朗読するのは執筆バッチ（``python -m llm_radio_daemon.generated_drama_writer``）が
    書き溜めた確定済みテキスト。放送中は LLM を一切通さない（§4.7.0）。
    """

    # 省略時は [locale] lang から data/generated_drama_data_<lang> を組み立てる
    # （原稿は言語ごとに別在庫。既定値を ja に固定しない）。
    data_dir: str = ""                    # characters.json / world.json / 本文の置き場
    narrator_cast_id: str | None = None   # 地の文を読む [[cast]] の role（"host" / "assistant" 等）。未指定・不在なら抽選1人目
    dialogue_by_character: bool = True    # 「」を characters.json の話者へ振り分ける
    chunk_target_chars: int = 400
    chunk_max_chars: int = 500
    chunk_min_chars: int = 120
    chunks_per_batch: int = 4             # 1回の produce_batch で積むチャンク数
    pause_ms: int = 400                   # チャンク間の無音
    pause_scene_ms: int = 900             # シーンの切れ目
    pause_se_ms: int = 700               # 効果音（擬音だけの行）の前後に取る「間」。紙芝居的なメリハリ
    duck_db: float = -18.0                # 朗読中の BGM 減衰（通常トークは audio.duck_db）
    transition_pause_before_ms: int = 2500  # 場面転換前：音が止まってから暗転するまでの間
    transition_pause_after_ms: int = 2500   # 場面転換後：明転してから読み始めるまでの間
    # --- 執筆バッチ（generated_drama_writer）側の既定値。放送プロセスは使わない ---
    chapters: int = 8                     # new のとき組み立てる章数
    scenes_per_chapter: int = 4           # 1章あたりのシーン数
    scene_target_chars: int = 1600        # 1シーンの目標文字数
    writer_timeout_sec: int = 600         # 執筆時の Ollama タイムアウト（放送側より長い）
    # --- 放送本体からの自動執筆（§4.7.6）。GeneratedDramaSupervisorThread が隙を見て generated_drama_writer を起動 ---
    auto_write: bool = False              # true なら放送プロセスが暇な時に run --auto を子プロセスで走らせる
    # --- ステージ0「企画立案」（§4.7.1）。new でタイトル省略時／run --auto の自動補充で使う ---
    auto_concept: bool = False            # run --auto 実行時に執筆中が閾値未満なら企画を自動生成
    concept_candidates: int = 3           # 1回の企画立案で LLM に出させる候補数
    concept_min_writing: int = 1          # status=writing がこれ未満なら run --auto が新企画を立てる
    title_style: str = "narou_long"       # "narou_long"（説明的な長文）| "short"（短いキャッチー）
    # トロープのプール。乱数で組み合わせて企画のシードにする。空でも動く
    genres: list[str] = field(default_factory=lambda: [
        "異世界転生", "悪役令嬢", "追放ざまぁ", "学園ラブコメ", "スローライフ異世界",
        "ダンジョン配信", "現代ダンジョン", "チート能力もの", "ざまぁ系婚約破棄",
    ])
    trope_pool: list[str] = field(default_factory=lambda: [
        "実は最強", "英雄の生まれ変わり", "二週目・やり直し", "記憶喪失",
        "ステータス偽装", "地味だが規格外", "面倒事を避けたいだけ", "追放されて自由",
        "スキルが外れ扱いだが最強", "前世は社畜", "内政チート", "従魔・使い魔",
    ])
    relationship_pool: list[str] = field(default_factory=lambda: [
        "幼なじみ", "許嫁・政略結婚", "契約結婚", "隣の席", "主従", "ライバル",
        "正体を隠している", "年下の懐かれ役", "クールな先輩", "護衛対象",
    ])
    twist_pool: list[str] = field(default_factory=lambda: [
        "実は相手も転生者", "ヒロインが黒幕側", "世界がゲームの続編だった",
        "入れ替わっている", "モブのはずが主要人物", "死に戻りのループ", "予言がずれている",
    ])


@dataclass
class ContentConfig:
    """番組表の 1 コーナー。時間帯が重なったら記述順が先のものが優先される。"""

    type: str
    schedule: list[str] = field(default_factory=list)   # "HH:MM-HH:MM"。空なら常時対象
    enabled: bool = True
    label: str = ""                                     # 声に出すコーナー名（「お天気コーナー」等）。
                                                        # 空なら type をそのまま使う。リクエスト
                                                        # （§6.2）の読み上げと request --list の表示用
    tone_hint: str = ""                                 # プロンプトへ注入するトーンのヒント
    host_id: str | None = None                          # 進行役の role（"host" / "assistant" 等）。未指定なら抽選 1 人目
    min_speakers: int = 3
    max_speakers: int = 4
    poll_interval_sec: float = 300.0
    background: str = ""                                # 背景演出。空なら [display] background
    stream: list[str] = field(default_factory=list)     # BGM の候補局（[[streams]] の id）。
                                                        # 複数なら入るたびにランダムで1局。空なら既定局
    song_talk: str = "off"                              # type == "radio" の曲トーク（SONG_TALK_MODES）
    params: dict = field(default_factory=dict)          # type 固有キー（rss_urls など）
    literary_reading: LiteraryReadingParams | None = None  # type == "literary_reading" のときだけ
    translated_reading: TranslatedReadingParams | None = None  # type == "translated_reading" のときだけ
    biography_reading: BiographyReadingParams | None = None  # type == "biography_reading" のときだけ
    generated_drama: GeneratedDramaParams | None = None    # type == "generated_drama" のときだけ

    @property
    def display_label(self) -> str:
        """人・声に向けて出すコーナー名。label 未設定なら type で代用する。"""
        return self.label or self.type

    @property
    def is_talk(self) -> bool:
        """出演者の掛け合いが発生するコーナーか（radio は song_talk 次第）。"""
        return self.type != "radio" or self.song_talk != "off"

    @property
    def is_back_announce(self) -> bool:
        return self.type == "radio" and self.song_talk == "back"


@dataclass
class DBConfig:
    # 省略時は [locale] lang から data/llm_radio_daemon.<lang>.db を組み立てる。
    # 放送履歴・重複判定・朗読の進行状況が入るので、言語をまたいで共有すると
    # 「英語版で読んだ記事を日本語版が既読扱い」といった取り違えが起きる。
    path: str = ""
    rebroadcast_after_days: int = 7


@dataclass
class LogConfig:
    dir: str = "logs"
    level: str = "INFO"


@dataclass
class DebugConfig:
    """開発・実験用のスイッチ。通常運用では空にしておく。"""

    # 出演者をこの id 並びに固定する（[debug] pinned_cast_ids）。
    # 空なら通常どおり [[content]] の min/max_speakers から毎回ランダム抽選する。
    # 設定すると全トーク（ひな壇・フィラー・読書コーナーの感想役・ラジオドラマなど）が
    # 毎回この面子・この順で出る。host_id や進行役ガードより優先される。
    # 先頭が進行役／朗読役になるので、朗読コーナー用に試すなら2人以上並べること。
    pinned_cast_ids: list[str] = field(default_factory=list)

    # 全 cast の「今回の角度」を angle_variants のこのインデックスに固定する
    # （[debug] pin_angle_index）。null＝通常どおりトピック毎にランダム選択。
    # 範囲外のインデックス（負・件数以上）は「角度なし」扱い。再現性のあるデバッグ用。
    pin_angle_index: int | None = None


@dataclass
class StreamConfig:
    """ネットラジオ局1つ。``id`` は [[content]] の ``stream`` から参照する短い名前。"""

    id: str
    url: str
    name: str = ""   # 画面表示・ログ用。省略時は id をそのまま使う

    @property
    def label(self) -> str:
        return self.name or self.id


@dataclass
class VoiceStyle:
    """声色1つ（解決済み）。name は LLM に提示する呼び名、id は合成に渡すハンドル。

    config.toml には name しか書かない。id は起動時に TTS バックエンドの
    resolve_cast_voices() が解決して埋める（CastMember.resolved_voices）。

    id の中身はバックエンド依存で、それを作ったバックエンドだけが解釈できる
    （VOICEVOX なら話者ID＝int、Kokoro ならスタイルベクトルと速度の組）。
    ただし generated_drama は VOICEVOX の話者IDを int のまま DB に永続化して
    いるため（generated_drama/__init__.py の voicevox_speaker）、
    VOICEVOX バックエンドに限っては今までどおり int が入る。
    """

    name: str
    id: Any


@dataclass
class VoiceBlend:
    """Kokoro のボイス配合レシピ1要素。スタイルベクトルを weight で加重平均する。"""

    voice: str
    weight: float = 1.0


@dataclass
class KokoroStyle:
    """[[cast.kokoro_styles]] の1エントリ。Kokoro には VOICEVOX のような感情スタイルが
    無いので、「配合レシピ＋速度」の名前つき組をスタイルとして扱う。

    こうしておくと LLM 側のスタイル選択機構（ScriptLine.style）を無改造で使える。
    blend が空なら CastMember.kokoro_voice を単体で使う。
    """

    name: str
    blend: list[VoiceBlend] = field(default_factory=list)
    speed: float = 1.0


@dataclass
class Accessory:
    """[[cast]] の accessory 配列の1要素。頭部の装飾品（髪飾り・イヤリング等）。

    type は display/models.py の ACCESSORY_TYPES のいずれか。color は "#rrggbb"
    （省略時は poly_character 側の既定色）。side は "left" / "right" / "both"
    （省略時は type ごとの自然な既定）。妥当性は _validate_cast でまとめてチェックする。
    """

    type: str
    color: str | None = None
    side: str | None = None


@dataclass
class CastMember:
    """ひな壇トークの出演者1人。id は台本の speaker キー（短い ASCII）。

    role は 3D 表示（display/app.py）のひな壇配置・色分けに使うタグで、
    host / assistant / other の3種のみ。host・assistant は前列（床）、
    other は後列（台の上）へ並ぶ。さらに generate_script の進行役ガードが
    「抽選メンバーに host / assistant がいれば先頭（進行役）へ寄せる」ために参照する。
    """

    id: str
    name: str
    role: str  # "host" / "assistant" / "other" のいずれか。省略時は "other"
    desc: str
    # 持ちネタの「切り口」候補（持ちネタキャラバリエーション §1）。トピック毎に1つ選んで
    # プロンプトへ短い1行として上乗せする。desc（根本人格）は壊さず「今回はこの角度で」を
    # 足すだけ。省略時 []＝角度なし（現状動作）。各エントリの1文を消せば元に戻る可逆性を
    # 保つこと。ja / en で言語別に config_cast.toml へ書く。
    angle_variants: list[str] = field(default_factory=list)
    # --- 声の設定。どのキーが必須かは [tts] backend による ---------------------
    # backend = "voicevox"（日本語）のとき使う。キャラ名＋スタイル名だけを書き、
    # id は書かない。先頭が既定スタイル。
    voicevox_speaker_name: str | None = None  # VOICEVOX キャラ名（例: "春日部つむぎ"）
    voicevox_speaker_styles: list[str] = field(default_factory=lambda: ["ノーマル"])
    # backend = "kokoro"（英語）のとき使う。単一ボイス名か、スタイルごとの配合レシピ。
    kokoro_voice: str | None = None  # 例: "af_heart"
    kokoro_styles: list[KokoroStyle] = field(default_factory=list)

    # 起動時に TTS バックエンドの resolve_cast_voices() が埋める（インプレース）。
    # generated_drama_writer など TTS に接続しないプロセスでは未解決のままになる。
    resolved_voices: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    # 解決済みスタイル名（順序つき・先頭が既定）。どのキーを見るかは backend 依存なので
    # バックエンドに決めさせる。未解決のときは voicevox 側の一覧へフォールバックする。
    resolved_style_names: list[str] = field(default_factory=list, repr=False, compare=False)
    model: str = DEFAULT_MODEL  # 3D 表示のモデル種別キーワード（display/models.py の MODEL_PRESETS）
    # 以下は3体だけの見た目の個別上書き。省略時は model プリセットや役割ベースの既定値を使う
    # （display/app.py 側で解決する。ここでは形式チェックのみ）。
    # 肌色は個別指定不可（クレイ風の統一トーンを保つ方針。09-04 相談）。
    hair_style: str | None = None    # display/models.py の HAIR_STYLES。model プリセットの既定髪型を上書き
    hair_color: str | None = None    # "#rrggbb"。未指定なら既定のグレー系
    eye_color: str | None = None     # "#rrggbb"。未指定なら黒
    clothing_color: str | None = None  # "#rrggbb"。未指定なら role ごとの既定パレット（_ROLE_PALETTE）
    accessory: list[Accessory] = field(default_factory=list)  # 頭部の装飾品（headphones / cat_ears / pin）

    @property
    def style_names(self) -> list[str]:
        """この出演者の声色の呼び名（順序つき・先頭が既定）。

        解決済みならバックエンドが書いた一覧を、未解決なら voicevox 側の設定を返す
        （generated_drama_writer は TTS に繋がずに default_style_name を参照するため、
        未解決でも必ず1つ以上返す必要がある）。
        """
        if self.resolved_style_names:
            return self.resolved_style_names
        return self.voicevox_speaker_styles or ["ノーマル"]

    @property
    def styles(self) -> list[VoiceStyle]:
        """台本生成時に LLM へ提示する声のスタイル一覧（id は解決済みなら実値、未解決なら0）。"""
        return [VoiceStyle(n, self.resolved_voices.get(n, 0)) for n in self.style_names]

    @property
    def default_style_name(self) -> str:
        """既定スタイル（一覧の先頭）の名前。解決前でも参照できる。"""
        return self.style_names[0]

    @property
    def voicevox_speaker(self) -> int | None:
        """既定スタイルの解決済み VOICEVOX 話者ID。未解決なら None。

        generated_drama はこの int を台本と一緒に DB へ永続化するため、
        int 以外のハンドル（Kokoro のスタイルベクトル等）のときは None を返す。
        """
        handle = self.resolved_voices.get(self.default_style_name)
        return handle if isinstance(handle, int) else None

    def resolve_style(self, style_name: str | None) -> Any:
        """スタイル名 → 合成に渡すハンドル。未指定・一覧にない名前は既定へフォールバックする。

        TTS バックエンドの resolve_cast_voices() 実行後（放送プロセス）でのみ呼ぶこと。
        """
        name = style_name if style_name in self.style_names else self.default_style_name
        try:
            return self.resolved_voices[name]
        except KeyError:
            raise ConfigError(
                f"[[cast]] {self.id!r}: 声が未解決です"
                "（TTSBackend.resolve_cast_voices() を実行してから呼んでください） / "
                "voice not resolved yet (call TTSBackend.resolve_cast_voices() first)"
            ) from None


@dataclass
class Config:
    locale: LocaleConfig = field(default_factory=LocaleConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    filler: FillerConfig = field(default_factory=FillerConfig)
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    db: DBConfig = field(default_factory=DBConfig)
    log: LogConfig = field(default_factory=LogConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    content: list[ContentConfig] = field(default_factory=list)
    cast: list[CastMember] = field(default_factory=list)
    streams: list[StreamConfig] = field(default_factory=list)

    @property
    def primary_stream(self) -> StreamConfig:
        """既定局。[[content]] が stream を指定していない時間帯はこれが流れる。"""
        if not self.streams:
            raise ConfigError(
                "config.toml に [[streams]] が1つも定義されていません。"
                " config.toml.example を参考に1局以上設定してください。 / "
                "no [[streams]] defined in config.toml. Define at least one, "
                "using config.toml.example as a reference."
            )
        return self.streams[0]

    @property
    def stream_by_id(self) -> dict[str, StreamConfig]:
        return {s.id: s for s in self.streams}

    def streams_for(self, content: "ContentConfig | None") -> list[StreamConfig]:
        """そのコーナーで使う候補局。stream 未指定なら既定局1つだけ。"""
        if content is not None and content.stream:
            by_id = self.stream_by_id
            picked = [by_id[sid] for sid in content.stream if sid in by_id]
            if picked:
                return picked
        return [self.primary_stream]

    @property
    def cast_by_id(self) -> dict[str, CastMember]:
        return {m.id: m for m in self.cast}

    def content_by_type(self, content_type: str) -> ContentConfig | None:
        """指定 type の（有効な）コンテンツ設定。複数あれば最初のもの。無ければ None。"""
        return next(
            (c for c in self.content if c.type == content_type and c.enabled), None
        )


def load_config(path: str | Path) -> Config:
    p = Path(path)
    if not p.exists():
        raise ConfigError(
            f"{p} が見つかりません。同じ場所の config.toml.example をコピーして作成してください。"
            f"  copy {p.with_name('config.toml.example')} {p}"
            f" / {p} not found. Copy config.toml.example in the same directory to create it:"
            f"  copy {p.with_name('config.toml.example')} {p}"
        )
    # PowerShell の Out-File などで BOM 付きになると tomllib が 1 行 1 列目で
    # 失敗するため、読み込み時に取り除いておく。
    text = p.read_text(encoding="utf-8-sig")
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{p} の TOML 構文エラー: {e} / TOML syntax error in {p}: {e}") from e

    streams = _parse_streams(raw.get("streams", []))

    display = DisplayConfig(**raw.get("display", {}))
    try:
        parse_background(display.background)
    except ValueError as e:
        raise ConfigError(f"[display] background: {e}") from e

    locale = LocaleConfig(**raw.get("locale", {}))
    # 台本・定型文の言語をここで一度だけ差し替える。
    #
    # **入口ごとに呼ぶ形にしてはいけない。** 以前は main.py と generated_drama_writer.py
    # だけが set_language() を呼んでおり、その2つを直したあとも scripts/ の
    # 切り分けスクリプトが呼び忘れていて、`--config config/en/config.toml` を渡しても
    # 日本語版として動いていた（英語の朗読コーナーが青空文庫を読みに行った）。
    # config を読まない入口は無いので、ここに置けば忘れようがない。
    language.set_language(locale.lang)
    sensitive.set_language(locale.lang)

    llm = LLMConfig(**raw.get("llm", {}))

    if "cast" in raw:
        raise ConfigError(
            "[[cast]] は config.toml から廃止しました。同じ内容を config_cast.toml に"
            "移してください（config_cast.toml.example を参照） / "
            "[[cast]] was removed from config.toml. Move the same content to "
            "config_cast.toml (see config_cast.toml.example)"
        )

    # cast / content は config.toml と同じディレクトリから引く。言語ごとに
    # config/<lang>/ を丸ごと切り替えられるようにするための約束（モジュール docstring 参照）。
    cast_path = p.with_name("config_cast.toml")
    if not cast_path.exists():
        raise ConfigError(
            f"{cast_path} が見つかりません。同じ場所の config_cast.toml.example をコピーして"
            "[[cast]] を1人以上定義してください。"
            f"  copy {cast_path.with_name('config_cast.toml.example')} {cast_path}"
            f" / {cast_path} not found. Copy config_cast.toml.example in the same directory"
            " and define at least one [[cast]] entry:"
            f"  copy {cast_path.with_name('config_cast.toml.example')} {cast_path}"
        )
    cast_text = cast_path.read_text(encoding="utf-8-sig")
    try:
        cast_raw = tomllib.loads(cast_text).get("cast", [])
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{cast_path} の TOML 構文エラー: {e} / TOML syntax error in {cast_path}: {e}") from e

    if not cast_raw:
        raise ConfigError(
            f"{cast_path} に [[cast]] が1つも定義されていません。"
            "config_cast.toml.example を参考に1人以上定義してください。 / "
            f"no [[cast]] entries defined in {cast_path}. Define at least one, "
            "using config_cast.toml.example as a reference."
        )
    try:
        cast = [_parse_cast_member(m) for m in cast_raw]
    except TypeError as e:
        raise ConfigError(f"[[cast]] の項目が不正です: {e} / invalid [[cast]] entry: {e}") from e
    # どの声設定が必須かは backend で変わるので、[tts] を先に覗いておく
    # （TTSConfig の組み立て自体は下の方でまとめて行う）。
    _validate_cast(cast, str(raw.get("tts", {}).get("backend", "voicevox")).strip().lower())
    _expand_cast_placeholders(cast, llm)

    if "content" in raw:
        raise ConfigError(
            "[[content]] は config.toml から廃止しました。同じ内容を config_content.toml に"
            "移してください（config_content.toml.example を参照） / "
            "[[content]] was removed from config.toml. Move the same content to "
            "config_content.toml (see config_content.toml.example)"
        )

    content_path = p.with_name("config_content.toml")
    if not content_path.exists():
        raise ConfigError(
            f"{content_path} が見つかりません。同じ場所の config_content.toml.example を"
            "コピーして作成してください。"
            f"  copy {content_path.with_name('config_content.toml.example')} {content_path}"
            f" / {content_path} not found. Copy config_content.toml.example in the same"
            " directory to create it:"
            f"  copy {content_path.with_name('config_content.toml.example')} {content_path}"
        )
    content_text = content_path.read_text(encoding="utf-8-sig")
    try:
        content_raw = tomllib.loads(content_text).get("content", [])
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{content_path} の TOML 構文エラー: {e} / TOML syntax error in {content_path}: {e}") from e

    content = _parse_content(
        content_raw, {m.role for m in cast}, len(cast), {s.id for s in streams}, locale.lang
    )

    weather = _parse_weather(raw.get("weather", {}))
    if not weather.enabled and any(c.type == "weather" and c.enabled for c in content):
        # 番組表にお天気コーナーがあるのに天気を取りに行かない設定。コーナーの
        # 時間帯がまるごとフィラーになるだけなので落としはしないが、気づけるようにする。
        logging.getLogger(__name__).warning(
            '[[content]] type = "weather" があるのに [weather] enabled = false です。'
            "お天気コーナーはネタを1件も出しません（[weather] に緯度経度を設定してください）"
        )

    debug_raw = dict(raw.get("debug", {}))
    if "disable_daily_mood" in debug_raw:
        debug_raw.pop("disable_daily_mood")
        logging.getLogger(__name__).warning(
            "[debug] disable_daily_mood は廃止しました。日替わり mood は既定 off です。"
            "有効化するなら [llm] daily_mood = true を書いてください。"
        )
    try:
        debug = DebugConfig(**debug_raw)
    except TypeError as e:
        raise ConfigError(f"[debug] の項目が不正です: {e} / invalid [debug] entry: {e}") from e
    if debug.pinned_cast_ids:
        known = {m.id for m in cast}
        unknown = [cid for cid in debug.pinned_cast_ids if cid not in known]
        if unknown:
            raise ConfigError(
                f"[debug] pinned_cast_ids に config_cast.toml に無い id があります: "
                f"{', '.join(unknown)} / "
                f"[debug] pinned_cast_ids has id(s) not found in config_cast.toml: {', '.join(unknown)}"
            )
    if debug.pin_angle_index is not None and (
        isinstance(debug.pin_angle_index, bool)
        or not isinstance(debug.pin_angle_index, int)
    ):
        raise ConfigError(
            f"[debug] pin_angle_index = {debug.pin_angle_index!r} は整数で指定してください"
            "（範囲外のインデックスは「角度なし」扱いになります） / "
            "must be an integer (an out-of-range index is treated as \"no angle\")"
        )

    try:
        db = DBConfig(**raw.get("db", {}))
    except TypeError as e:
        raise ConfigError(f"[db] の項目が不正です: {e} / invalid [db] entry: {e}") from e
    # 言語ごとに別 DB。省略時に ja へ倒すと英語放送が日本語の履歴を掴むので、
    # 既定値はハードコードせず [locale] lang から組み立てる。
    if not db.path:
        db.path = f"data/llm_radio_daemon.{locale.lang}.db"

    return Config(
        locale=locale,
        llm=llm,
        tts=TTSConfig(**raw.get("tts", {})),
        audio=AudioConfig(**raw.get("audio", {})),
        display=display,
        filler=_parse_filler(raw.get("filler", {})),
        weather=weather,
        embedding=EmbeddingConfig(**raw.get("embedding", {})),
        db=db,
        log=LogConfig(**raw.get("log", {})),
        debug=debug,
        content=content,
        cast=cast,
        streams=streams,
    )


def _parse_filler(raw: dict) -> FillerConfig:
    try:
        return FillerConfig(**raw)
    except TypeError as e:
        raise ConfigError(f"[filler] の項目が不正です: {e} / invalid [filler] entry: {e}") from e


def _parse_weather(raw: dict) -> WeatherConfig:
    """``[weather]``。有効にしたのに場所が書かれていなければ起動時に落とす。

    「有効なのに黙って無効」は、放送を1本流してから「天気の話が出ない」と気づく形の
    事故になる。設定の書き間違いは起動時に見せるほうが安い。
    """
    try:
        wc = WeatherConfig(**dict(raw))
    except TypeError as e:
        raise ConfigError(f"[weather] の項目が不正です: {e} / invalid [weather] entry: {e}") from e
    if not wc.enabled:
        return wc

    missing = [
        k for k, v in (
            ("location_name", wc.location_name),
            ("latitude", wc.latitude),
            ("longitude", wc.longitude),
        ) if v in (None, "")
    ]
    if missing:
        raise ConfigError(
            f"[weather] enabled = true には {', '.join(missing)} が必要です"
            "（天気は「スタジオの所在地」の設定なので自動判定しません。"
            ' 例: location_name = "東京" / latitude = 35.6895 / longitude = 139.6917） / '
            f"[weather] enabled = true requires {', '.join(missing)}"
            " (the location is never auto-detected; it's a studio-location setting."
            ' Example: location_name = "Tokyo" / latitude = 35.6895 / longitude = 139.6917)'
        )
    if not -90.0 <= float(wc.latitude) <= 90.0:
        raise ConfigError(
            f"[weather] latitude = {wc.latitude!r} は -90〜90 の範囲で指定してください / "
            f"[weather] latitude = {wc.latitude!r} must be between -90 and 90"
        )
    if not -180.0 <= float(wc.longitude) <= 180.0:
        raise ConfigError(
            f"[weather] longitude = {wc.longitude!r} は -180〜180 の範囲で指定してください / "
            f"[weather] longitude = {wc.longitude!r} must be between -180 and 180"
        )
    return wc


def _parse_streams(raw_list: list) -> list[StreamConfig]:
    out: list[StreamConfig] = []
    seen: set[str] = set()
    for i, entry in enumerate(raw_list, start=1):
        entry = dict(entry)
        name = str(entry.get("name", ""))
        # id が無い旧書式（name と url だけ）との互換。name をそのまま id として扱う。
        sid = str(entry.get("id") or name)
        url = str(entry.get("url", ""))
        if not sid or not url:
            raise ConfigError(f"[[streams]] #{i}: id と url は必須です / [[streams]] #{i}: id and url are required")
        if sid in seen:
            raise ConfigError(
                f"[[streams]] #{i}: id = {sid!r} が重複しています / "
                f"[[streams]] #{i}: id = {sid!r} is duplicated"
            )
        seen.add(sid)
        out.append(StreamConfig(id=sid, url=url, name=name))
    return out


def _parse_content(
    raw_list: list, cast_roles: set[str], cast_count: int, stream_ids: set[str], lang: str
) -> list[ContentConfig]:
    if not raw_list:
        raise ConfigError(
            "config_content.toml に [[content]] が1つも定義されていません。"
            " config_content.toml.example を参考にコンテンツ（時間帯・出演人数）を設定してください。 / "
            "no [[content]] entries defined in config_content.toml. Define at least one"
            " (schedule, speaker count), using config_content.toml.example as a reference."
        )

    out: list[ContentConfig] = []
    for i, entry in enumerate(raw_list, start=1):
        entry = dict(entry)
        ctype = entry.get("type")
        if ctype == "musicbrainz":
            # 旧 type。曲トークは type = "radio" の song_talk に統合した。
            raise ConfigError(
                f'[[content]] #{i}: type = "musicbrainz" は廃止しました。'
                ' type = "radio" に song_talk を付けてください'
                '（back_announce = true だったものは song_talk = "back"、'
                ' そうでなければ song_talk = "intro"） / '
                f'[[content]] #{i}: type = "musicbrainz" was removed. Use type = "radio" with'
                ' song_talk instead (back_announce = true becomes song_talk = "back",'
                ' otherwise song_talk = "intro")'
            )
        if ctype not in CONTENT_TYPES:
            raise ConfigError(
                f"[[content]] #{i}: type = {ctype!r} は無効です"
                f"（使えるのは {', '.join(CONTENT_TYPES)}） / "
                f"[[content]] #{i}: type = {ctype!r} is invalid (supported: {', '.join(CONTENT_TYPES)})"
            )

        song_talk = str(entry.get("song_talk", "off"))
        if song_talk not in SONG_TALK_MODES:
            raise ConfigError(
                f"[[content]] #{i} ({ctype}): song_talk = {song_talk!r} は無効です"
                f"（使えるのは {', '.join(SONG_TALK_MODES)}） / "
                f"[[content]] #{i} ({ctype}): song_talk = {song_talk!r} is invalid"
                f" (supported: {', '.join(SONG_TALK_MODES)})"
            )
        if song_talk != "off" and ctype != "radio":
            raise ConfigError(
                f'[[content]] #{i} ({ctype}): song_talk は type = "radio" のときだけ使えます / '
                f'[[content]] #{i} ({ctype}): song_talk only applies when type = "radio"'
            )
        if "back_announce" in entry:
            raise ConfigError(
                f"[[content]] #{i} ({ctype}): back_announce は廃止しました。"
                ' song_talk = "back"（曲間で振り返る）/ "intro"（曲頭で紹介する）'
                " に置き換えてください / "
                f'[[content]] #{i} ({ctype}): back_announce was removed. Replace it with'
                ' song_talk = "back" (a remark after the song) / "intro" (introduced as it starts)'
            )

        sched_raw = entry.get("schedule", [])
        schedule = [sched_raw] if isinstance(sched_raw, str) else list(sched_raw)
        for spec in schedule:
            try:
                parse_window(spec)
            except (ValueError, AttributeError) as e:
                raise ConfigError(
                    f"[[content]] #{i} ({ctype}): schedule {spec!r} の書式が不正です"
                    f"（HH:MM-HH:MM、日跨ぎ可） / "
                    f"[[content]] #{i} ({ctype}): schedule {spec!r} has an invalid format"
                    f" (HH:MM-HH:MM, may cross midnight)"
                ) from e

        host_id = entry.get("host_id")
        if host_id is not None and host_id not in cast_roles:
            raise ConfigError(
                f"[[content]] #{i} ({ctype}): host_id = {host_id!r} という role を持つ"
                f" [[cast]] がありません（host_id は role の値を指定します。"
                f' "host" / "assistant" のいずれかで、その role の [[cast]] が'
                f"必要です） / "
                f"[[content]] #{i} ({ctype}): host_id = {host_id!r} does not match any"
                f" [[cast]] role (host_id names a role value — \"host\" or \"assistant\" —"
                f" and a [[cast]] entry with that role must exist)"
            )

        default_min, default_max = (2, 2) if ctype in ("literary_reading", "translated_reading") else (3, 4)
        mn = int(entry.get("min_speakers", default_min))
        mx = int(entry.get("max_speakers", default_max))
        # 出演人数を検証しないケース:
        #   - 喋らない区間（song_talk = "off" の radio）
        #   - ラジオドラマ朗読（出演者は原稿側で決まる・§4.7.5）
        if ctype != "generated_drama" and (ctype != "radio" or song_talk != "off"):
            if not (1 <= mn <= mx):
                raise ConfigError(
                    f"[[content]] #{i} ({ctype}): min_speakers/max_speakers は "
                    f"1 <= min <= max である必要があります（min={mn}, max={mx}） / "
                    f"[[content]] #{i} ({ctype}): min_speakers/max_speakers must satisfy"
                    f" 1 <= min <= max (min={mn}, max={mx})"
                )
            if mx > cast_count:
                raise ConfigError(
                    f"[[content]] #{i} ({ctype}): max_speakers={mx} が "
                    f"[[cast]] の人数 {cast_count} を超えています / "
                    f"[[content]] #{i} ({ctype}): max_speakers={mx} exceeds the [[cast]] count ({cast_count})"
                )

        background = entry.get("background", "")
        try:
            parse_background(background)
        except ValueError as e:
            raise ConfigError(
                f"[[content]] #{i} ({ctype}): background: {e}"
            ) from e

        stream_raw = entry.get("stream", [])
        stream = [stream_raw] if isinstance(stream_raw, str) else list(stream_raw)
        for sid in stream:
            if sid not in stream_ids:
                raise ConfigError(
                    f"[[content]] #{i} ({ctype}): stream = {sid!r} は [[streams]] に"
                    f"存在しません（定義済み: {', '.join(sorted(stream_ids)) or 'なし'}） / "
                    f"[[content]] #{i} ({ctype}): stream = {sid!r} is not defined in [[streams]]"
                    f" (defined: {', '.join(sorted(stream_ids)) or 'none'})"
                )

        params = {k: v for k, v in entry.items() if k not in _CONTENT_COMMON_KEYS}
        literary_reading = _parse_literary_reading_params(params, i) if ctype == "literary_reading" else None
        translated_reading = _parse_translated_reading_params(params, i) if ctype == "translated_reading" else None
        biography_reading = _parse_biography_reading_params(params, i, lang) if ctype == "biography_reading" else None
        generated_drama = _parse_generated_drama_params(params, i, cast_roles, lang) if ctype == "generated_drama" else None

        out.append(
            ContentConfig(
                type=ctype,
                schedule=schedule,
                enabled=bool(entry.get("enabled", True)),
                label=str(entry.get("label", "")),
                tone_hint=entry.get("tone_hint", ""),
                host_id=host_id,
                min_speakers=mn,
                max_speakers=mx,
                poll_interval_sec=float(entry.get("poll_interval_sec", 300.0)),
                background=background,
                stream=stream,
                song_talk=song_talk,
                params=params,
                literary_reading=literary_reading,
                translated_reading=translated_reading,
                biography_reading=biography_reading,
                generated_drama=generated_drama,
            )
        )
    return out


def _parse_literary_reading_params(params: dict, idx: int) -> LiteraryReadingParams:
    known = {f.name for f in fields(LiteraryReadingParams)}
    p = {k: v for k, v in params.items() if k in known}
    works = p.pop("works", [])
    if "comment_lines" in p:
        p["comment_lines"] = tuple(p["comment_lines"])
    try:
        rp = LiteraryReadingParams(**p)
    except TypeError as e:
        raise ConfigError(
            f"[[content]] #{idx} (literary_reading): 項目が不正です: {e} / "
            f"invalid entry: {e}"
        ) from e
    if rp.work_selector not in ("random", "llm"):
        raise ConfigError(
            f"[[content]] #{idx} (literary_reading): work_selector = {rp.work_selector!r} は "
            f'"random" / "llm" のいずれかです / '
            f'[[content]] #{idx} (literary_reading): work_selector = {rp.work_selector!r} must be'
            f' "random" or "llm"'
        )
    rp.works = [dict(w) for w in works]
    return rp


def _parse_translated_reading_params(params: dict, idx: int) -> TranslatedReadingParams:
    known = {f.name for f in fields(TranslatedReadingParams)}
    p = {k: v for k, v in params.items() if k in known}
    works = p.pop("works", [])
    if "comment_lines" in p:
        p["comment_lines"] = tuple(p["comment_lines"])
    if "translate_lines" in p:
        p["translate_lines"] = tuple(p["translate_lines"])
    try:
        tp = TranslatedReadingParams(**p)
    except TypeError as e:
        raise ConfigError(
            f"[[content]] #{idx} (translated_reading): 項目が不正です: {e} / "
            f"invalid entry: {e}"
        ) from e
    if tp.work_selector not in ("random", "llm"):
        raise ConfigError(
            f"[[content]] #{idx} (translated_reading): work_selector = {tp.work_selector!r} は "
            f'"random" / "llm" のいずれかです / '
            f'[[content]] #{idx} (translated_reading): work_selector = {tp.work_selector!r} must be'
            f' "random" or "llm"'
        )
    tp.works = [dict(w) for w in works]
    return tp


def _parse_biography_reading_params(params: dict, idx: int, lang: str) -> BiographyReadingParams:
    if "lang" in params:
        raise ConfigError(
            f"[[content]] #{idx} (biography_reading): lang はここでは指定できません。"
            " 番組全体の言語設定なので config.toml の [locale] lang に移してください / "
            f"[[content]] #{idx} (biography_reading): lang cannot be set here."
            " It's a broadcast-wide setting — use [locale] lang in config.toml instead"
        )
    known = {f.name for f in fields(BiographyReadingParams)}
    p = {k: v for k, v in params.items() if k in known}
    if "figures" in p:
        raise ConfigError(
            f"[[content]] #{idx} (biography_reading): [[content.figures]] は廃止しました。"
            f" figures_file（既定: content/biography_figures.{lang}.txt）に1行1人物で書いてください / "
            f"[[content]] #{idx} (biography_reading): [[content.figures]] was removed."
            f" List one figure per line in figures_file (default: content/biography_figures.{lang}.txt)"
        )
    if "segment_lines" in p:
        p["segment_lines"] = tuple(p["segment_lines"])
    try:
        bp = BiographyReadingParams(**p)
    except TypeError as e:
        raise ConfigError(
            f"[[content]] #{idx} (biography_reading): 項目が不正です: {e} / invalid entry: {e}"
        ) from e
    bp.lang = lang
    # 人物リストと本文キャッシュは言語ごと。既定値を ja に固定すると英語放送が
    # 日本語の人物リストを読み、en.wikipedia に無い記事名を引きにいく。
    if not bp.figures_file:
        bp.figures_file = f"content/biography_figures.{lang}.txt"
    if not bp.data_dir:
        bp.data_dir = f"data/wiki_bio_{lang}"

    figures_path = Path(bp.figures_file)
    if not figures_path.is_file():
        raise ConfigError(
            f"[[content]] #{idx} (biography_reading): figures_file={bp.figures_file!r} が見つかりません。"
            " 1行1人物（Wikipediaの記事タイトル）のテキストファイルを用意してください / "
            f"[[content]] #{idx} (biography_reading): figures_file={bp.figures_file!r} not found."
            " Provide a text file with one figure (a Wikipedia article title) per line"
        )

    figures: list[dict] = []
    seen_ids: set[str] = set()
    for line in figures_path.read_text(encoding="utf-8").splitlines():
        wiki_title = line.strip()
        if not wiki_title or wiki_title.startswith("#"):
            continue
        if wiki_title in seen_ids:
            raise ConfigError(
                f"[[content]] #{idx} (biography_reading): {bp.figures_file} に重複した行があります: {wiki_title!r} / "
                f"[[content]] #{idx} (biography_reading): {bp.figures_file} has a duplicate line: {wiki_title!r}"
            )
        seen_ids.add(wiki_title)
        figures.append({"id": wiki_title, "wiki_title": wiki_title})

    if not figures:
        raise ConfigError(
            f"[[content]] #{idx} (biography_reading): {bp.figures_file} に人物が1人も書かれていません。"
            " 対象人物は自動選定せず明示指定する方針です / "
            f"[[content]] #{idx} (biography_reading): {bp.figures_file} lists no figures."
            " Figures are never auto-selected; they must be listed explicitly"
        )
    bp.figures = figures
    return bp


def _parse_generated_drama_params(
    params: dict, idx: int, cast_roles: set[str], lang: str
) -> GeneratedDramaParams:
    known = {f.name for f in fields(GeneratedDramaParams)}
    p = {k: v for k, v in params.items() if k in known}
    try:
        np_ = GeneratedDramaParams(**p)
    except TypeError as e:
        raise ConfigError(
            f"[[content]] #{idx} (generated_drama): 項目が不正です: {e} / invalid entry: {e}"
        ) from e
    # 原稿は言語ごとに別在庫。既定値を ja に固定すると英語放送が日本語の原稿を読む。
    if not np_.data_dir:
        np_.data_dir = f"data/generated_drama_data_{lang}"
    if np_.title_style not in ("narou_long", "short"):
        raise ConfigError(
            f"[[content]] #{idx} (generated_drama): title_style = {np_.title_style!r} は "
            f'"narou_long" / "short" のいずれかです / '
            f'[[content]] #{idx} (generated_drama): title_style = {np_.title_style!r} must be'
            f' "narou_long" or "short"'
        )
    # narrator_cast_id は role の値（"host" / "assistant" 等）。「不在でも落とさない」
    # （§4.7.3）。設定ミスに気づけるよう警告だけ出し、抽選 1 人目での代行は
    # GeneratedDramaCorner 側で行う。
    if np_.narrator_cast_id and np_.narrator_cast_id not in cast_roles:
        logging.getLogger(__name__).warning(
            "[[content]] #%d (generated_drama): narrator_cast_id = %r という role を持つ"
            " [[cast]] が無い。地の文は抽選 1 人目が読む",
            idx, np_.narrator_cast_id,
        )
    return np_


_OS_LABELS = {
    "Windows": "Windows",
    "Darwin": "macOS",
    "Linux": "Linux",
}


def _os_label() -> str:
    """トークで名乗る用の OS 名。未知の値（BSD 等）は platform.system() の生値を返す。"""
    return _OS_LABELS.get(platform.system(), platform.system())


def _expand_cast_placeholders(cast: list[CastMember], llm: LLMConfig) -> None:
    """[[cast]] の name / desc に書ける差し込み記法を実値へ置換する（インプレース）。

    - ``{model}``                     → ``[llm].model``（例: qwen3.8:27b）
    - ``{engine}`` / ``{engine_label}`` → ``[llm].engine_label``（例: Ollama）
    - ``{os}``                        → 実行 OS 名（例: Windows / macOS / Linux）

    番組を動かしている当のモデル自身をキャラとして出演させる（`name = "{model}"`）用途向け。
    [[cast]] の ``model`` キーは 3D モデル種別なので別物。
    """
    subs = {
        "{model}": llm.model,
        "{engine}": llm.engine_label,
        "{engine_label}": llm.engine_label,
        "{os}": _os_label(),
    }
    for m in cast:
        for token, value in subs.items():
            m.name = m.name.replace(token, value)
            m.desc = m.desc.replace(token, value)


def _parse_cast_member(m: dict) -> CastMember:
    """[[cast]] の1エントリを CastMember に変換する。

    voicevox_speaker_styles はスタイル名の文字列配列で書く（例: ["ノーマル", "ツンツン"]）。
    話者IDは書かない——起動時に TTS バックエンドの resolve_cast_voices() が解決する。
    ここは形式チェックだけを行い、「どのキーが必須か」は backend 依存なので
    _validate_cast と各バックエンドに任せる。
    """
    m = dict(m)
    m.setdefault("role", "other")  # "host" / "assistant" / "other" のいずれか。省略時は "other"
    if "voicevox_speaker" in m:
        raise ConfigError(
            f"[[cast]] {m.get('id', '?')!r}: voicevox_speaker（数値ID）は廃止しました。"
            " voicevox_speaker_name（VOICEVOX のキャラ名）だけを書いてください。"
            " 話者IDは起動時に VOICEVOX ENGINE から自動解決します。 / "
            f"[[cast]] {m.get('id', '?')!r}: voicevox_speaker (numeric id) was removed."
            " Write only voicevox_speaker_name (the VOICEVOX character name);"
            " the speaker id is resolved automatically from VOICEVOX ENGINE at startup."
        )
    kokoro_styles = _parse_kokoro_styles(m.pop("kokoro_styles", []), m.get("id", "?"))
    styles_raw = m.pop("voicevox_speaker_styles", ["ノーマル"])
    if not isinstance(styles_raw, list) or not all(isinstance(s, str) for s in styles_raw):
        raise ConfigError(
            f"[[cast]] {m.get('id', '?')!r} の voicevox_speaker_styles はスタイル名の"
            f' 文字列配列で書いてください（例: ["ノーマル", "ツンツン"]）。'
            " id 付きのインラインテーブル形式は廃止しました。 / "
            f"[[cast]] {m.get('id', '?')!r}: voicevox_speaker_styles must be an array of style"
            f' name strings (e.g. ["ノーマル", "ツンツン"]). The old inline-table-with-id form was removed.'
        )
    angle_variants_raw = m.pop("angle_variants", [])
    if not isinstance(angle_variants_raw, list) or not all(
        isinstance(a, str) for a in angle_variants_raw
    ):
        raise ConfigError(
            f"[[cast]] {m.get('id', '?')!r} の angle_variants は文字列配列で"
            f' 書いてください（例: ["市場規模の角度から入りたがる", "競合との差別化にしたがる"]）。 / '
            f"[[cast]] {m.get('id', '?')!r}: angle_variants must be an array of strings"
            f' (e.g. ["goes straight to market size", "wants to talk about competitors"]).'
        )
    angle_variants = [a.strip() for a in angle_variants_raw if a.strip()]
    accessory_raw = m.pop("accessory", [])
    if not isinstance(accessory_raw, list) or not all(isinstance(a, dict) for a in accessory_raw):
        raise ConfigError(
            f"[[cast]] {m.get('id', '?')!r} の accessory はインラインテーブルの配列で"
            f' 書いてください（例: accessory = [{{ type = "cat_ears", color = "#7ab0d9" }}]）。 / '
            f"[[cast]] {m.get('id', '?')!r}: accessory must be an array of inline tables"
            f' (e.g. accessory = [{{ type = "cat_ears", color = "#7ab0d9" }}]).'
        )
    accessories: list[Accessory] = []
    for a in accessory_raw:
        try:
            accessories.append(Accessory(**a))
        except TypeError as e:
            raise ConfigError(
                f"[[cast]] {m.get('id', '?')!r}: accessory の要素 {a!r} が不正です"
                ' （キーは type / color / side のみ、type は必須）。 / '
                f"[[cast]] {m.get('id', '?')!r}: invalid accessory entry {a!r}"
                " (allowed keys are type / color / side; type is required)."
            ) from e
    return CastMember(
        voicevox_speaker_styles=styles_raw,
        kokoro_styles=kokoro_styles,
        accessory=accessories,
        angle_variants=angle_variants,
        **m,
    )


def _parse_kokoro_styles(raw: object, cast_id: str) -> list[KokoroStyle]:
    """[[cast.kokoro_styles]] を KokoroStyle の一覧にする。

    書式（blend を省くと CastMember.kokoro_voice を単体で使う）::

        [[cast.kokoro_styles]]
        name = "excited"
        speed = 1.12
        blend = [ { voice = "af_bella", weight = 0.7 }, { voice = "af_sky", weight = 0.3 } ]
    """
    if not raw:
        return []
    if not isinstance(raw, list) or not all(isinstance(s, dict) for s in raw):
        raise ConfigError(
            f"[[cast]] {cast_id!r}: kokoro_styles は [[cast.kokoro_styles]] の"
            " テーブル配列で書いてください。 / "
            f"[[cast]] {cast_id!r}: kokoro_styles must be written as an array of"
            " [[cast.kokoro_styles]] tables."
        )
    out: list[KokoroStyle] = []
    for s in raw:
        name = s.get("name")
        if not isinstance(name, str) or not name:
            raise ConfigError(
                f"[[cast]] {cast_id!r}: kokoro_styles の各要素には name が必須です。 / "
                f"[[cast]] {cast_id!r}: every kokoro_styles entry requires a name."
            )
        blend_raw = s.get("blend", [])
        if not isinstance(blend_raw, list) or not all(isinstance(b, dict) for b in blend_raw):
            raise ConfigError(
                f"[[cast]] {cast_id!r} の kokoro_styles {name!r}: blend は"
                ' インラインテーブルの配列で書いてください'
                '（例: blend = [{ voice = "af_bella", weight = 0.7 }]）。 / '
                f"[[cast]] {cast_id!r} kokoro_styles {name!r}: blend must be an array of"
                ' inline tables (e.g. blend = [{ voice = "af_bella", weight = 0.7 }]).'
            )
        blend: list[VoiceBlend] = []
        for b in blend_raw:
            try:
                blend.append(VoiceBlend(**b))
            except TypeError as e:
                raise ConfigError(
                    f"[[cast]] {cast_id!r} の kokoro_styles {name!r}: blend の要素 {b!r}"
                    " が不正です（キーは voice / weight のみ、voice は必須）。 / "
                    f"[[cast]] {cast_id!r} kokoro_styles {name!r}: invalid blend entry {b!r}"
                    " (allowed keys are voice / weight; voice is required)."
                ) from e
        speed = s.get("speed", 1.0)
        if not isinstance(speed, (int, float)) or isinstance(speed, bool) or not 0.1 <= speed <= 3.0:
            raise ConfigError(
                f"[[cast]] {cast_id!r} の kokoro_styles {name!r}: speed={speed!r} は"
                " 0.1〜3.0 の数値で指定してください（等速は 1.0）。 / "
                f"[[cast]] {cast_id!r} kokoro_styles {name!r}: speed={speed!r} must be a"
                " number between 0.1 and 3.0 (1.0 = normal speed)."
            )
        unknown = set(s) - {"name", "blend", "speed"}
        if unknown:
            raise ConfigError(
                f"[[cast]] {cast_id!r} の kokoro_styles {name!r}: 未知のキー"
                f" {', '.join(sorted(unknown))}（使えるのは name / blend / speed）。 / "
                f"[[cast]] {cast_id!r} kokoro_styles {name!r}: unknown key(s)"
                f" {', '.join(sorted(unknown))} (allowed: name / blend / speed)."
            )
        out.append(KokoroStyle(name=name, blend=blend, speed=float(speed)))
    return out


def _validate_cast(cast: list[CastMember], backend: str = "voicevox") -> None:
    if not cast:
        raise ConfigError("[[cast]] が空です。1人以上定義してください。 / [[cast]] is empty. Define at least one.")

    for m in cast:
        if m.role not in ("host", "assistant", "other"):
            raise ConfigError(
                f"[[cast]] {m.id!r}: role={m.role!r} は無効です。"
                f'"host" / "assistant" / "other" のいずれかを指定してください（省略時 "other"）。 / '
                f'[[cast]] {m.id!r}: role={m.role!r} is invalid. Must be one of'
                f' "host" / "assistant" / "other" (defaults to "other" if omitted).'
            )

        if m.model not in known_models():
            raise ConfigError(
                f"[[cast]] {m.id!r}: model={m.model!r} は未知のモデル種別です。"
                f"次のいずれかを指定してください: {', '.join(known_models())} / "
                f"[[cast]] {m.id!r}: model={m.model!r} is an unknown model kind."
                f" Must be one of: {', '.join(known_models())}"
            )

        if m.hair_style is not None and m.hair_style not in known_hair_styles():
            raise ConfigError(
                f"[[cast]] {m.id!r}: hair_style={m.hair_style!r} は未知の髪型です。"
                f"次のいずれかを指定してください: {', '.join(known_hair_styles())} / "
                f"[[cast]] {m.id!r}: hair_style={m.hair_style!r} is an unknown hairstyle."
                f" Must be one of: {', '.join(known_hair_styles())}"
            )

        for field_name in ("hair_color", "eye_color", "clothing_color"):
            value = getattr(m, field_name)
            if value is not None and not _HEX_COLOR_RE.match(value):
                raise ConfigError(
                    f"[[cast]] {m.id!r}: {field_name}={value!r} は不正です。"
                    f'"#rrggbb" 形式（例: "#3a2a1a"）で指定してください。 / '
                    f'[[cast]] {m.id!r}: {field_name}={value!r} is invalid.'
                    f' Use "#rrggbb" format (e.g. "#3a2a1a").'
                )

        for a in m.accessory:
            if a.type not in known_accessories():
                raise ConfigError(
                    f"[[cast]] {m.id!r}: accessory type={a.type!r} は未知の装飾品です。"
                    f"次のいずれかを指定してください: {', '.join(known_accessories())} / "
                    f"[[cast]] {m.id!r}: accessory type={a.type!r} is an unknown accessory."
                    f" Must be one of: {', '.join(known_accessories())}"
                )
            if a.color is not None and not _HEX_COLOR_RE.match(a.color):
                raise ConfigError(
                    f"[[cast]] {m.id!r}: accessory color={a.color!r} は不正です。"
                    f'"#rrggbb" 形式（例: "#7ab0d9"）で指定してください。 / '
                    f'[[cast]] {m.id!r}: accessory color={a.color!r} is invalid.'
                    f' Use "#rrggbb" format (e.g. "#7ab0d9").'
                )
            if a.side is not None and a.side not in known_accessory_sides():
                raise ConfigError(
                    f"[[cast]] {m.id!r}: accessory side={a.side!r} は無効です。"
                    f"次のいずれかを指定してください: {', '.join(known_accessory_sides())} / "
                    f"[[cast]] {m.id!r}: accessory side={a.side!r} is invalid."
                    f" Must be one of: {', '.join(known_accessory_sides())}"
                )

        # どの声設定が必須かは [tts] backend による。実在チェック（名前がエンジンに
        # あるか、ボイス名が正しいか）は起動時に各バックエンドの resolve_cast_voices()
        # が行う。ここでは「そもそも書かれているか」だけを見る。
        if backend == "kokoro":
            if not m.kokoro_voice and not m.kokoro_styles:
                raise ConfigError(
                    f"[[cast]] {m.id!r}: [tts] backend=\"kokoro\" では kokoro_voice"
                    "（例: \"af_heart\"）か [[cast.kokoro_styles]] のどちらかが必須です。 / "
                    f'[[cast]] {m.id!r}: [tts] backend="kokoro" requires either kokoro_voice'
                    ' (e.g. "af_heart") or [[cast.kokoro_styles]].'
                )
            names = [s.name for s in m.kokoro_styles]
            label = "kokoro_styles の name"
        else:
            if not m.voicevox_speaker_name:
                raise ConfigError(
                    f"[[cast]] {m.id!r}: voicevox_speaker_name は必須です"
                    "（VOICEVOX ENGINE の /speakers に実在するキャラ名）。 / "
                    f"[[cast]] {m.id!r}: voicevox_speaker_name is required"
                    " (a character name that exists in VOICEVOX ENGINE's /speakers)."
                )
            names = list(m.voicevox_speaker_styles)
            label = "voicevox_speaker_styles"
        dup_names = sorted({n for n in names if names.count(n) > 1})
        if dup_names:
            raise ConfigError(
                f"[[cast]] {m.id!r}: {label} が重複しています: {', '.join(dup_names)} / "
                f"[[cast]] {m.id!r}: {label} has duplicate(s): {', '.join(dup_names)}"
            )

    ids = [m.id for m in cast]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ConfigError(
            f"[[cast]] の id が重複しています: {', '.join(dupes)} / "
            f"[[cast]] has duplicate id(s): {', '.join(dupes)}"
        )


def resolve_voicevox_voices(cast: list[CastMember], speakers: dict[str, dict[str, int]]) -> None:
    """後方互換のための薄いラッパー。実装は VoicevoxBackend.resolve_cast_voices() に移した
    （声の解決はバックエンド固有の知識なので、config ではなく tts 層に置くのが筋）。"""
    from .tts.voicevox import resolve_voicevox_voices as _impl

    _impl(cast, speakers)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: python -m llm_radio_daemon.config <config/<lang>/config.toml>\n"
            "  例: python -m llm_radio_daemon.config config/ja/config.toml\n"
            "  example: python -m llm_radio_daemon.config config/en/config.toml"
        )
    print(load_config(sys.argv[1]))
