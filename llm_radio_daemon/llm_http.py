"""推論エンジンごとの HTTP 差異を吸収する薄いクライアント。

放送側・執筆側のどのモジュールも requests で ``/api/chat`` を直叩きせず、ここを通す。

対応エンジン（[llm] engine / [embedding] engine、config.py の engine_api_style 参照）:
  ollama            … ネイティブ API（/api/chat, /api/embeddings）
  lmstudio / openai … OpenAI 互換 API（/v1/chat/completions, /v1/embeddings）
  unsloth           … Unsloth Studio。openai と同じ /v1/chat/completions・同じ json_schema 形
                       だが、中身は llama.cpp なので thinking 抑止に
                       ``chat_template_kwargs={"enable_thinking": False}`` を併せて送る
                       （llamacpp と同じ理由）。埋め込みは非対応（[embedding] では弾く）
  llamacpp          … llama.cpp サーバ（llama-server）。エンドポイントは openai と同じ

認証:
  [llm] / [embedding] の api_key が空でなければ、openai / llamacpp / unsloth 経路のリクエストに
  ``Authorization: Bearer <api_key>`` を付ける（ollama ネイティブ経路には付けない）。
  Unsloth Studio は Keyless 設定なら鍵不要 —— その場合 api_key は空のままでよい。

structured output（JSON Schema）:
  ollama          … ``format`` にスキーマをそのまま渡す
  openai / unsloth … ``response_format = {"type": "json_schema", "json_schema": {"name", "schema"}}``
                     （LM Studio・OpenAI・Unsloth Studio）
  llamacpp        … ``response_format = {"type": "json_object", "schema": {...}}``。
                     llama-server で最も安定して schema 制約が効く形（``json_schema`` の平たい形は
                     ビルドによって無視され本文が自由文で返る実測あり。ネスト形は効くが json_object
                     の方が対応バージョンが広い）。
  llamacpp / unsloth … 加えて thinking 抑止に ``chat_template_kwargs={"enable_thinking": False}``
                     を送る（reasoning_effort="none" だけでは効かないテンプレートがあるため両方。
                     gemma-4 を Unsloth で回すと、これが無いと content が空／途中切れになる実測）

structured output が効かない/壊れるエンジンへのフォールバック:
  ollama のクラウドモデル（``<model>-cloud``）は ``format`` を無視して Markdown の
  自由文を返す。ollama ネイティブの一部モデル（実測: gpt-oss:20b）は逆に ``format``
  を付けた途端に生成そのものが壊れて空応答になる。``chat(schema=...)`` はどちらの
  場合もスキーマをプロンプトへ書き足し、以後ネイティブの format 制約は付けずに
  本文から JSON を取り出して返す（詳細は下の「structured output が効かない/壊れる〜」節）。

``chat()`` は message の content 文字列を返す（JSON のパースは呼び出し側の責務）。
ただし ``schema`` を渡したときの戻り値は ``json.loads`` できる文字列であることを保証する。
HTTP エラーは ``requests.RequestException``、応答の形が壊れていれば ``ValueError`` を送出する
—— どちらも既存の呼び出し側が捕捉しているのでシグネチャは変えていない。
"""

from __future__ import annotations

import json
import logging
import threading
from urllib.parse import urlparse

import requests

from .config import EmbeddingConfig, LLMConfig

logger = logging.getLogger(__name__)

# 直近の ollama ネイティブ応答のメタ情報（token 数・所要時間）。スレッドごとに持つ。
# 放送側は読まない —— bench_corners.py が1呼び出しごとの実測値を拾うための窓口。
_last_stats = threading.local()

_OLLAMA_STAT_KEYS = (
    "total_duration", "load_duration",
    "prompt_eval_count", "prompt_eval_duration",
    "eval_count", "eval_duration",
)


def last_ollama_stats() -> dict | None:
    """このスレッドで直近に完了した ollama ネイティブ ``/api/chat`` のメタ情報。

    ``prompt_eval_count`` / ``eval_count`` はトークン数、``*_duration`` はナノ秒。
    OpenAI 互換エンジンや、まだ ``chat()`` を呼んでいないスレッドでは ``None``。
    """
    return getattr(_last_stats, "value", None)


def _auth_headers(config: LLMConfig | EmbeddingConfig) -> dict[str, str]:
    """api_key があれば ``Authorization: Bearer …`` を返す。無ければ空 dict。

    LM Studio / llama.cpp は鍵不要なので通常は空。Unsloth Studio や OpenAI 本体・
    OpenRouter 等に向けるときだけ [llm] / [embedding] api_key を設定する。
    """
    key = getattr(config, "api_key", "") or ""
    return {"Authorization": f"Bearer {key}"} if key else {}


def _raise_for_status(resp: requests.Response) -> None:
    """4xx/5xx なら本文つきで送出する。

    ``resp.raise_for_status()`` の文言は「400 Client Error: Bad Request for url: …」
    止まりで、サーバが返した理由（LM Studio や Unsloth Studio は JSON の error.message で
    「No model loaded」等を返す）が消える。エンジンを取っ替え引っ替えする用途では
    その1行が一番効くので、本文の先頭を足してから投げ直す。型は変えない
    （呼び出し側は requests.RequestException を捕捉している）。
    """
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        body = (resp.text or "").strip().replace("\n", " ")
        if body:
            raise requests.HTTPError(f"{e} — {body[:500]}", response=resp) from None
        raise


def _base(host: str) -> str:
    """host のベース URL を正規化する。末尾の "/" と "/v1" を落とす。"""
    base = host.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return base


def _chat_raw(
    llm_config: LLMConfig,
    prompt: str,
    *,
    schema: dict | None = None,
    temperature: float | None = None,
    num_predict: int | None = None,
    timeout_sec: int | None = None,
) -> str:
    """1 往復のチャット補完。message の content 文字列をそのまま返す。

    structured output の要求はここでは「エンジンへ投げるだけ」。効いたかどうかは
    見ない（無視するエンジンがある）。後始末は ``chat()`` が行う。
    """
    temp = llm_config.temperature if temperature is None else temperature
    timeout = llm_config.timeout_sec if timeout_sec is None else timeout_sec
    # 呼び出し側が num_predict を明示していなければ [llm] num_predict（0 = 未指定）を使う。
    if num_predict is None and llm_config.num_predict:
        num_predict = llm_config.num_predict
    if llm_config.log_prompts:
        logger.info(
            "LLM prompt -> %s/%s:\n%s", llm_config.engine, llm_config.model, prompt
        )
    messages = [{"role": "user", "content": prompt}]

    if llm_config.api_style in ("openai", "llamacpp", "unsloth"):
        payload: dict = {
            "model": llm_config.model,
            "messages": messages,
            "temperature": temp,
            "stream": False,
            # ollama 側の "think": False と同じ意図。推論(thinking)対応モデルだと
            # 思考にトークン予算・時間を使い切って本文生成が遅く/空になるため止める。
            # LM Studio・llama.cpp サーバは未知フィールドを無視するので害はない。
            "reasoning_effort": "none",
        }
        if llm_config.api_style in ("llamacpp", "unsloth"):
            # reasoning_effort だけでは thinking を止めないテンプレート向け。
            # Qwen3 系（llama.cpp）、gemma-4 系（Unsloth Studio、実測: これが無いと
            # json_schema 制約下で思考にトークンを使い切り content が空／途中切れになる）。
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if num_predict is not None:
            payload["max_tokens"] = num_predict
        if schema is not None:
            if llm_config.api_style == "llamacpp":
                # llama-server は古くから ``{"type":"json_object","schema":{…}}`` が
                # 一番安定して schema 制約を効かせる形（``{"type":"json_schema","schema":{…}}``
                # の平たい形はビルドによって無視され、本文が自由文で返る実測あり）。
                payload["response_format"] = {"type": "json_object", "schema": schema}
            else:
                # openai / unsloth。Unsloth Studio も json_schema 形を尊重する（実測）。
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "response", "schema": schema},
                }
            if llm_config.log_schema:
                logger.info(
                    "LLM schema -> %s/%s (response_format):\n%s",
                    llm_config.engine, llm_config.model,
                    json.dumps(payload["response_format"], ensure_ascii=False, indent=2),
                )
        resp = requests.post(
            f"{_base(llm_config.host)}/v1/chat/completions",
            json=payload,
            headers=_auth_headers(llm_config),
            timeout=timeout,
        )
        _raise_for_status(resp)
        _last_stats.value = None  # ollama ネイティブ以外はメタ情報を持たない
        try:
            content = resp.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"unexpected OpenAI-style chat response: {e}") from e
        if llm_config.log_responses:
            logger.info(
                "LLM response <- %s/%s:\n%s", llm_config.engine, llm_config.model, content
            )
        return content

    # ollama ネイティブ
    options: dict = {"temperature": temp}
    if llm_config.num_ctx:
        options["num_ctx"] = llm_config.num_ctx
    if llm_config.repeat_penalty:
        options["repeat_penalty"] = llm_config.repeat_penalty
    if llm_config.frequency_penalty:
        options["frequency_penalty"] = llm_config.frequency_penalty
    if num_predict is not None:
        options["num_predict"] = num_predict
    payload = {
        "model": llm_config.model,
        "messages": messages,
        "options": options,
        "keep_alive": llm_config.keep_alive,
        "stream": False,
        # 推論(thinking)対応モデルだと思考にトークン予算を使い切って本文が空になる。
        "think": False,
    }
    if schema is not None:
        payload["format"] = schema
        if llm_config.log_schema:
            logger.info(
                "LLM schema -> %s/%s (format):\n%s",
                llm_config.engine, llm_config.model,
                json.dumps(schema, ensure_ascii=False, indent=2),
            )
    resp = requests.post(
        f"{_base(llm_config.host)}/api/chat", json=payload, timeout=timeout
    )
    _raise_for_status(resp)
    body = resp.json()
    _last_stats.value = {
        k: body[k] for k in _OLLAMA_STAT_KEYS if isinstance(body.get(k), int)
    }
    try:
        content = body["message"]["content"]
    except (KeyError, TypeError) as e:
        raise ValueError(f"unexpected Ollama chat response: {e}") from e
    if llm_config.log_responses:
        logger.info(
            "LLM response <- %s/%s:\n%s", llm_config.engine, llm_config.model, content
        )
    return content


# --- structured output が効かない/壊れるエンジンへのフォールバック ----------
#
# 「効かない」相手が2種類いる。
#   (a) 無視するだけ（例: ollama のクラウドモデル gpt-oss:120b-cloud 等）。
#       ``format`` を完全に無視し、Markdown の自由文を返す（実測）
#   (b) 制約そのもので生成が壊れる（例: ollama ネイティブの gpt-oss:20b）。
#       ``format`` を付けた途端、数トークンで生成が止まり本文が空になる（実測。
#       スキーマ無しなら普通に応答する）
# この番組は台本・フィラー・各コーナーのすべてを JSON Schema で受けているので、
# 素通しだと原稿が1本も通らず、無限にフィラー送りになる。そこで3段構え：
#
#   1. 効かない/壊れると分かっている相手には、最初からスキーマをプロンプトへ書き足し、
#      ネイティブの format / response_format は付けない（(b) には必須。付けたままだと
#      プロンプトへ書き足しても壊れたまま）
#   2. 返ってきた本文からは、コードフェンスや前置きを剥がして JSON を取り出す
#   3. それでも JSON にならなかった相手は「効かない」と覚え、次回から 1 を適用する
#
# ``chat(schema=...)`` の契約（JSON としてパースできる文字列を返すか、さもなくば
# ValueError）は変わらないので、呼び出し側（json.loads する側）に変更は要らない。

# structured output が効かないと分かった (host, model)。プロセス内でのみ保持する。
_schema_unreliable: set[tuple[str, str]] = set()

# スキーマをプロンプトへ書くときの指示文。英語で書くのは JSON の出力形式に関する
# 指示だからで、台本の言語（language.LANGUAGE_GUIDANCE）とは干渉しない。
_SCHEMA_PROMPT = """

## Output format (strict)
Reply with a single JSON object and nothing else: no Markdown, no code fences, no
commentary before or after. The object must validate against this JSON Schema:

{schema}
"""


def _schema_key(config: LLMConfig) -> tuple[str, str]:
    return (_base(config.host), config.model)


def _schema_in_prompt(config: LLMConfig) -> bool:
    """スキーマをプロンプトへ書き足すべき相手か。"""
    if _schema_key(config) in _schema_unreliable:
        return True
    # ollama のクラウド実行は format を無視する。placement は起動時に確定している。
    return config.api_style == "ollama" and config.placement == "cloud"


def _extract_json(text: str) -> str | None:
    """本文から JSON オブジェクトを1つ取り出す。見つからなければ None。

    そのまま JSON → コードフェンス（```json …```）→ 最初の ``{`` から対応する
    ``}`` まで、の順に試す。文字列リテラル中の波括弧とエスケープは飛ばす。
    """
    text = (text or "").strip()
    if not text:
        return None
    if text.startswith("{"):
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass  # 途中で切れている等。下の走査でもう一度拾いにいく

    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        break  # この ``{`` は当たりではない。次の ``{`` から
        start = text.find("{", start + 1)
    return None


def chat(
    llm_config: LLMConfig,
    prompt: str,
    *,
    schema: dict | None = None,
    temperature: float | None = None,
    num_predict: int | None = None,
    timeout_sec: int | None = None,
) -> str:
    """1 往復のチャット補完。message の content 文字列を返す。

    ``schema`` を渡すと structured output（JSON Schema 制約）を要求する。その場合の
    戻り値は「``json.loads`` できる文字列」で、エンジンが ``format`` を無視して
    自由文を返してきた場合もここで JSON を取り出して返す（取り出せなければ
    ``ValueError``）。``num_predict`` は生成トークン上限
    （ollama: options.num_predict / openai: max_tokens）。
    """
    if schema is None:
        return _chat_raw(
            llm_config, prompt, temperature=temperature,
            num_predict=num_predict, timeout_sec=timeout_sec,
        )

    inlined = _schema_in_prompt(llm_config)
    if inlined:
        prompt += _SCHEMA_PROMPT.format(schema=json.dumps(schema, ensure_ascii=False, indent=2))

    # 一度「効かない」と分かった相手には、ネイティブの format / response_format 自体を
    # 二度と付けない。素通しで無視するだけの相手には元々害が無いが、gpt-oss のように
    # format を付けた瞬間に生成そのものが壊れる（空応答・数トークンで停止）相手もいる
    # 実測があり、そちらには「無視されるはずだから送っておく」が致命傷になるため。
    content = _chat_raw(
        llm_config, prompt, schema=None if inlined else schema, temperature=temperature,
        num_predict=num_predict, timeout_sec=timeout_sec,
    )
    extracted = _extract_json(content)
    if extracted is not None:
        return extracted

    if not inlined:
        # このエンジンは structured output で失敗する（無視して自由文を返す／
        # 制約自体で生成が壊れる、のどちらか）。次回からはプロンプトへ書き足し、
        # ネイティブの制約は付けない（ここでは投げ直さない。呼び出し側のリトライが
        # 次の1本で拾う）。
        _schema_unreliable.add(_schema_key(llm_config))
        logger.warning(
            "structured output failed with %s (%s); "
            "スキーマをプロンプトへ書き足し、format制約は今後付けない方式へ切り替えます",
            llm_config.engine, llm_config.model,
        )
    raise ValueError(f"no JSON object in response: {content[:200]!r}")


# --- 実行場所（ローカル / クラウド）の判定 ---------------------------------
#
# 番組は「このパソコンの中の生成AIが喋っている」ことを楽屋ネタにする。ところが
# ollama は `<model>-cloud`（例: gpt-oss:120b-cloud）という名前で、クラウド実行の
# モデルを同じ localhost:11434 から使わせる。設定を書き換えただけで台詞が嘘に
# なるので、実際にどこで動いているかをここで判定して filler.py へ渡す。

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def detect_placement(config: LLMConfig, *, timeout_sec: int = 10) -> str:
    """モデルの実行場所を返す。``"local"`` / ``"cloud"`` / ``"unknown"``。

    ollama ネイティブなら ``/api/show`` を1回だけ叩き、モデルの実体が手元にあるか
    （``modelfile`` / ``tensors`` が返るか）で見分ける。クラウドのモデルは ``details``
    と ``capabilities`` しか返さず ``details.format`` が空文字になる。名前の
    ``-cloud`` サフィックスを見るより堅い —— ``/api/chat`` の応答に入っている
    ``model`` はサフィックスを落として返ってくるので、そちらは判定に使えない。

    OpenAI 互換エンジン（lmstudio / llamacpp / openai）には相当する情報が無いので、
    host がループバックかどうかだけで判断する。LAN 上の別マシンに向いている場合は
    「このパソコンの中」ではないが「クラウド」でもないので ``"unknown"`` に倒す
    （``"unknown"`` のときトークは実行場所に触れない）。
    """
    host = urlparse(_base(config.host)).hostname or ""
    if config.api_style != "ollama":
        return "local" if host in _LOOPBACK_HOSTS else "unknown"
    if host.endswith("ollama.com"):
        return "cloud"

    try:
        resp = requests.post(
            f"{_base(config.host)}/api/show",
            json={"model": config.model},
            timeout=timeout_sec,
        )
        _raise_for_status(resp)
        info = resp.json()
    except (requests.RequestException, ValueError) as e:
        logger.warning("placement detection failed (%s): %s", config.model, e)
        return "unknown"

    if info.get("modelfile") or info.get("tensors"):
        # 実体は手元にある。ただし ollama 自体が別マシンなら「このパソコン」ではない。
        return "local" if host in _LOOPBACK_HOSTS else "unknown"
    if (info.get("details") or {}).get("format") == "":
        return "cloud"
    return "unknown"


def embed(config: EmbeddingConfig, text: str) -> list[float] | None:
    """埋め込みベクトルを返す。失敗時は None（呼び出し側は「重複ではない＝通す」と扱う）。"""
    try:
        # unsloth は埋め込み非対応で config 側が弾くのでここには来ないが、
        # 形としては openai と同じ /v1/embeddings なので同じ枝に入れておく。
        if config.api_style in ("openai", "llamacpp", "unsloth"):
            resp = requests.post(
                f"{_base(config.host)}/v1/embeddings",
                json={"model": config.model, "input": text},
                headers=_auth_headers(config),
                timeout=config.timeout_sec,
            )
            _raise_for_status(resp)
            return resp.json()["data"][0]["embedding"] or None

        resp = requests.post(
            f"{_base(config.host)}/api/embeddings",
            json={"model": config.model, "prompt": text, "keep_alive": config.keep_alive},
            timeout=config.timeout_sec,
        )
        _raise_for_status(resp)
        return resp.json().get("embedding") or None
    except (requests.RequestException, ValueError, KeyError, IndexError) as e:
        logger.warning("embedding request failed (%s): %s", config.engine, e)
        return None
