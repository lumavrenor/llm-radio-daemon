[English](README.md) | **日本語**

# LLM Radio Daemon（エルエルエム・ラジオ・デーモン）

ローカルLLMがネット上の情報源からネタを拾い、十数人のキャラクター
（司会・アシスタント＋切り口で立つひな壇キャラ）からコーナーごとに2〜10人が
出演して掛け合いを続ける、24時間動きっぱなしのラジオ番組です。BGMにはインターネットラジオの
ストリームを流し、喋っている間は自動で音量を下げます（ダッキング）。

これはテレビではなく「映像付きラジオ」です。トークが主、BGMが従。
画面を消しても番組として成立する構造を目指しています。

個人の趣味プロジェクトです。ご質問・ご要望への対応はしておりません。
ただし、権利・規約上の問題や不具合のご指摘は、GitHub の Issue で受け付けています。

実装はすべて Python です。3D表示は [Ursina](https://www.ursinaengine.org/)
エンジンを使っており、キャラクターの動き（待機時の揺れ・身振り・喋っている間の
口の動き）はモーションファイルを焼き込むのではなく、コードで直接表現しています。
出演者の設定（役割・人数・切り口）や見た目（髪型・髪色・瞳の色・服の色など）は
config でキャラごとにカスタマイズ可能です。

2B〜4B程度の小型ローカルLLMでも十分動くので、`[llm].model` を差し替えて
「同じ番組をどのモデルに喋らせるとどう変わるか」を聴き比べる、ローカルLLMの
性能チェック的な使い方もできます。GPUが無くても2B〜4Bクラスなら十分動きますし、
Ollama のクラウドモデル（後述）も動作確認済みなので、GPUを積んでいないノートPCでも
気軽に試せます。

複数ネタ源・フィラートーク・時間帯編成・embedding重複排除を含め、
**主要機能は実装・動作確認済みです**（24時間soakテストのみ未実施）。

## デモ動画

[![デモ動画](https://img.youtube.com/vi/o028pIn0Y_U/0.jpg)](https://www.youtube.com/watch?v=o028pIn0Y_U&t=11s)

YouTube: https://www.youtube.com/watch?v=o028pIn0Y_U&t=11s

（音声トラックは含んでいません）

## 本プロジェクトの位置づけ

これは実用ソフトウェアではありません。プログラマーが手元で遊ぶための実験的なソフトウェアです。
本気で実用的な24時間ラジオ番組を作るなら、素直に大規模なクラウドAPIを使うべきです。
このプロジェクトの主眼はそこではなく、「2B〜4B程度の小型モデル・自分のPCだけで、
ここまで遊べる」ということを見せる点にあります。課金ゼロで動かし続けられる分、
出力の質はクラウドの大規模モデルには及びません。

## LLM が生成する内容について（重要）

出演者のセリフ・雑談・深夜の悩み相談への回答は、すべてローカルLLMがその場で自動生成した
フィクションです。実在の専門家によるチェックは一切入っていません。

生成内容を穏当な方向へ抑え込む工夫はしています。プロンプト側で「医療・健康は断定しない」
「投資・金融商品を勧めない」「政治・宗教に肩入れしない」「誹謗中傷やR-18表現をしない」と
常に指示し、悩み相談のような自由記述コーナーでは希死念慮・自傷・虐待といった重い話題を
明示的に禁止するなど、複数の層でガードをかけています。

さらに出力側にも、生成済みのセリフを読み上げ直前に照合する NG ワードフィルタを入れています。
性的表現・差別語・直接的な罵倒など明白に不適切な語を含むセリフは、キャラクターの声で読み上げず
無音に差し替え、ログに記録します（青空文庫・Project Gutenberg の原文をそのまま読む朗読本文は対象外）。
語の一覧は `llm_radio_daemon/sensitive.py` にあり、必要に応じて追加できます。

ただし、プロンプトによる指示は**LLMの出力を確実に制御する手段ではありません**。
NG ワードフィルタも決まった語句との照合にすぎず、言い換えや文脈で不適切になる表現までは
捉えられません。LLMが指示を無視したり、事実と異なる内容や、常識的に見て不適切・危険な内容を、
もっともらしい口調で生成してしまう可能性を、現状の生成AI技術でゼロにすることはできません。
これは実装の作り込みが甘いということではなく、LLMという技術の原理的な限界です。

- **医療・法律・金融・安全に関わる内容を、実際の助言として信じて行動しないでください。**
- 本ソフトの出力内容によって生じたいかなる損害についても、作者は責任を負いません。
  出力を鵜呑みにして行動した結果は、すべて利用者自身の判断によるものとします。

## 特徴

- **完全ローカル完結・APIの課金ゼロ** — LLM推論（Ollama / LM Studio / llama.cpp / Unsloth Studio のいずれか）・
  音声合成(VOICEVOX)・音声デコード(ffmpeg)をすべて自分のPC上で実行します。24時間動かしっぱなしに
  しても外部APIの従量課金は発生しません。これが本プロジェクトの一番の存在意義です。
- **ひな壇トーク** — 出演者を `config_cast.toml` の `[[cast]]` で定義（必須。
  config/ja/config_cast.toml.example は18人）。トピックごとに
  `min_speakers`〜`max_speakers` 人が抽選され、VOICEVOXの別々の話者でボケ・ツッコミ・脱線・
  素朴な疑問を交わします（単独のモノローグ読み上げではありません）。
- **外部ネットラジオをBGMに、喋る時だけダッキング** — インターネットラジオのストリームを常時再生し、
  トークが入る間だけ自動で音量を下げます。
- **複数のネタ源から自動でネタを拾い続ける** — Wikipedia・Hacker News・arXiv・任意のRSSフィード・
  再生中の楽曲（MusicBrainz経由）などから継続的にトピックを収集し、LLMが台本を生成し続けます。
  深夜帯だけは架空の悩み相談（LLMが実在人物を介さず生成）もネタ源に加わります。
- **番組表型の編成** — `config_content.toml` の `[[content]]` に「どのコンテンツを・いつ・何人で」流すかを
  書きます。時間帯ごとにアクティブなコンテンツが1つだけ選ばれ（記述順で先勝ち）、出演者は
  毎回 `[[cast]]` からランダムに抽選されます。
- **embeddingによるネタの重複排除** — 直近の発言と意味的に似すぎているネタは自動で捨て、
  同じ話の繰り返しを避けます。
- **お天気コーナー** — `config.toml` の `[weather]` に地点（緯度経度）を書いておくと、
  現在の天気を [Open-Meteo](https://open-meteo.com/)（APIキー不要）から取得し、
  お天気コーナーとして1本のトークにします。同じ値はフィラー雑談の「状況」としても使われ、
  お題リストの `@rain` / `@cold` といったタグで「今の天気に合うお題」だけを引かせられます。
  ここで書く地点はリスナーの現在地ではなく「スタジオの所在地」という番組設定なので、
  IPや位置情報からの自動判定は一切していません。取得できなかったときは天気に触れません。
- **読書コーナー（青空文庫 / Project Gutenberg）** — 深夜の時間帯だけ、パブリックドメイン作品
  （日本語版は青空文庫、英語版は Project Gutenberg）を朗読キャラが読み上げ、区切りごとに
  つっこみキャラが感想を挟みます。原文はLLMを一切通さずそのまま読み上げ、感想だけをLLMが
  生成します。日をまたいで連続ドラマ的に少しずつ進みます。
- **翻訳読書コーナー** — 日本語版のコーナーです。Project Gutenberg の英語の作品を、
  LLM が区切りごとに日本語へ訳しながら朗読し、合間に感想のトークを挟みます。
  読書コーナーと違い、読み上げる本文そのものが LLM の出力です。
- **ラジオドラマコーナー** — 別プロセスの執筆バッチ（`generated_drama_writer`）がプロット・章立て・登場人物を
  設計し、シーン単位で本文を書き溜めます。放送側はその完成原稿を読み上げるだけで、
  地の文はナレーター、セリフは登場人物ごとの声に振り分けられます。放送中のLLM負荷はゼロです。
- **偉人トーク** — Wikipediaの偉人の情報を元に、トークします。

## 設計の前提と、公開時の注意

本ソフトは、自宅で自分が非商用で楽しむ個人利用を想定した設計です。
そのため、以下は意図的に実装していません。

- 配信・ストリーミング出力機能
- 音声・音楽の録音／ファイル保存機能
- 音楽のローカル生成、YouTube からの音源取得
- LLM出力を使った外部書き込み系アクション（ファイル操作・メール送信・コード実行等）
  — 本システムの LLM 出力は**音声になって消えるだけ**です

### 音声つきの動画・配信を公開する場合

組み込んで使う素材（VOICEVOX の各キャラクター、ネットラジオ局、ネタ元のAPI など）には、
それぞれ別の利用規約があります。画面録画などで音声つきの動画を公開・配信する場合の権利処理は、
利用者自身の責任で行ってください。とくにキャラクターによっては、生成AIで作った台詞との組み合わせ、
公開・収益化、扱ってよい表現に条件があります。クレジットは、実際に使ったキャラクターだけを、
各規約が指定する表記で書いてください。作者は各規約を保証・代弁しません。

また、前述のとおり LLM のセリフは完全には制御できないため、**キャラクターの利用規約で禁じられた
内容（政治・宗教・性的表現・誹謗中傷など）を、そのキャラクターの声で話してしまう可能性があります。**
自宅で聴くだけなら音声は外に出ませんが、録画・配信などで公開する場合は、公開前に内容を確認し、
各キャラクターの規約に沿っているかを利用者自身の責任で判断してください。

## 必要なもの

| 項目 | 用途 |
|---|---|
| ローカルLLM推論エンジン（下記のいずれか1つ） | 台本・フィラー等の生成。`config.toml` の `[llm].engine` で選択します |
| &nbsp;&nbsp;・[Ollama](https://ollama.com/) | 既定。事前に使用するモデルを `ollama pull` しておくこと（`engine = "ollama"`）。Ollama のクラウドモデル（`gpt-oss:120b-cloud` のように名前が `-cloud` で終わるもの。手元のGPUでは載らない大きさのモデルを試せます）もそのまま指定できます |
| &nbsp;&nbsp;・[LM Studio](https://lmstudio.ai/) | ローカルサーバ（OpenAI互換 API）を起動しておくこと。`engine = "lmstudio"` / `host = "http://127.0.0.1:1234"` |
| &nbsp;&nbsp;・[llama.cpp](https://github.com/ggml-org/llama.cpp)（`llama-server`） | `llama-server` を起動しておくこと。`engine = "llamacpp"` / `host = "http://127.0.0.1:8080"` |
| &nbsp;&nbsp;・[Unsloth Studio](https://unsloth.ai/docs/new/studio) | `unsloth studio` でサーバを起動し、UI でモデルをロード（context length は 8192 以上に）。`engine = "unsloth"` / `host = "http://127.0.0.1:8888"`。localhost からの利用は Settings → API → Keyless API access の "Chat and inference" を ON にすれば鍵不要（ON にしないなら `api_key` にトークンを設定）。`[embedding]` に `unsloth` も使えるが埋め込みモデルは Unsloth 側設定で固定（既定は英語専用の bge-small）なので、日本語では `ollama` 推奨 |
| [VOICEVOX ENGINE](https://voicevox.hiroshiba.jp/) | 音声合成（HTTP API、既定 `http://127.0.0.1:50021`） |
| [ffmpeg](https://ffmpeg.org/) | ネットラジオのストリーム（MP3/AAC等）を音声データにデコードするために使用。`requirements.txt` の `imageio-ffmpeg` の wheel に同梱されており `pip install` 時点で入手済みになるので、手動インストール・PATH設定は不要（取得に失敗した場合のみ、フォールバックとして PATH 上の `ffmpeg` を探しに行く） |
| Python 3.11+ | 本体の実行環境 |
| [Visual Studio Code](https://code.visualstudio.com/)（推奨・任意） | 導入すると Python 拡張機能が仮想環境を自動検出し、統合ターミナルを開くたびに自動で有効化してくれます。下記セットアップ手順の「仮想環境の有効化」の面倒（PowerShellの実行ポリシー等）を丸ごと回避できます |

各ソフトウェアのライセンス・利用規約は**各自で確認してください**。本READMEはそれらを
代弁するものではありません。

> **対応プラットフォームについて**：開発・動作確認は Windows（PowerShell）でのみ行っています。
> 以下に記載の macOS / Linux 向け手順は「動くはず」という参考情報であり、実機での動作確認は
> できていません。macOS / Linux とも、動作確認を行う予定は未定です。

## セットアップ（Windows / PowerShell）

つまずきやすい手順なので、各ステップで「何が表示されれば成功か」を明記しています。
1つずつ確認しながら進めてください。

### 0. Python がインストールされているか確認

```powershell
python --version
```

`Python 3.11.x` のようにバージョンが表示されればOKです。

もし**何も表示されずMicrosoft Storeが開く**場合、`python` コマンドが実体を持たない
「アプリ実行エイリアス」を指しています。[python.org](https://www.python.org/downloads/) から
Python 3.11以降をインストールし直してください（インストーラーの「Add python.exe to PATH」に
チェックを入れること）。

### 1. 仮想環境の作成

> `go.bat` / `go.sh` から起動する場合は、この手順は省略できます。`.venv` が
> 無ければ起動時に自動で作成し、依存パッケージ（`requirements.txt`）も
> 自動でインストールしてから起動します（詳しくは「実行」章）。ここでは
> `python` を直接使う場合の手順を説明します。

```powershell
python -m venv .venv
```

何も表示されないまま数秒〜十数秒かかって終わります（失敗ではありません）。
プロジェクト直下に `.venv` フォルダができていれば成功です。

> `venv` ではなく **`.venv`**（先頭にドット）にしてください。同梱の
> `go.bat` / `go.sh` / `request.bat` はいずれも `.venv` を直接呼び出す
> 決め打ちなので、名前が違うと「見つからない」系のエラーになります。

<details>
<summary>macOS / Linux の場合</summary>

```bash
python3 -m venv .venv
```

</details>

### 2. 仮想環境の有効化

```powershell
.\.venv\Scripts\Activate.ps1
```

プロンプトの先頭に `(.venv)` と表示されれば成功です。**このあとの手順は、
毎回この `(.venv)` が付いていることを確認してから実行してください**
（付いていない状態で `pip install` すると、無関係な別のPython環境に
インストールされてしまいます）。

次のようなエラーが出た場合：

```
このシステムではスクリプトの実行が無効になっているため、ファイル
...\Activate.ps1 を読み込むことができません。
```

PowerShellの実行ポリシーが厳しく設定されています。一度だけ次を実行してから
（管理者権限は不要）、上の有効化コマンドをやり直してください。

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

> コマンドプロンプト（cmd.exe）を使う場合は `.venv\Scripts\activate.bat`
> （`.bat` は省略可）で同じことができ、実行ポリシーの制限は受けません。

<details>
<summary>macOS / Linux の場合</summary>

```bash
source .venv/bin/activate
```

</details>

### 3. 依存パッケージのインストール

> `go.bat` / `go.sh` から起動する場合、`.venv` を新規作成したときに限り
> `requirements.txt` の内容もこのスクリプトが自動でインストールします
> （英語版の `requirements-en.txt` は対象外なので、英語版を使う場合は
> 下記コマンドで別途インストールしてください）。

プロンプトに `(.venv)` が付いていることを確認してから：

```powershell
pip install -r requirements.txt
```

<details>
<summary>macOS / Linux の場合</summary>

```bash
pip install -r requirements.txt
```

</details>

### 4. 設定ファイルの用意

> `go.bat` / `go.sh` から起動する場合は、この手順は省略できます。3点セットの
> どれかが無ければ、対応する `.example` から自動でコピーして作ってから起動します
> （既存のファイルは上書きしません）。ここでは `python` を直接使う場合の手順を
> 説明します。

設定は**言語ごとに `config/<lang>/` へ3点セット**で置きます。

```
config/
  ja/  config.toml  config_cast.toml  config_content.toml   # 日本語版（VOICEVOX）
  en/  config.toml  config_cast.toml  config_content.toml   # 英語版（Kokoro）
```

`config_cast.toml` と `config_content.toml` は `config.toml` と**同じディレクトリ**から
読まれるので、言語の切り替えは起動時に渡すパス1つだけで済みます。

日本語版を使う場合（英語版なら `ja` を `en` に読み替えてください）:

```powershell
Copy-Item config\ja\config.toml.example         config\ja\config.toml
Copy-Item config\ja\config_cast.toml.example    config\ja\config_cast.toml
Copy-Item config\ja\config_content.toml.example config\ja\config_content.toml
```

<details>
<summary>macOS / Linux の場合</summary>

```bash
cp config/ja/config.toml.example         config/ja/config.toml
cp config/ja/config_cast.toml.example    config/ja/config_cast.toml
cp config/ja/config_content.toml.example config/ja/config_content.toml
```

</details>

Ollama を使う場合（既定の `engine`）は、`config.toml.example` の既定モデル
`gemma4:e4b`（8GB級のGPUでも動く軽量モデル）を事前に取得しておいてください。
これをしないと初回起動時にモデルが見つからずエラーになります。

```powershell
ollama pull gemma4:e4b
```

16GB以上のVRAMがある場合は、`[llm].model` を `gemma4:31b` 等の大きいモデルに
差し替えると台本の質が上がります（その場合は `ollama pull gemma4:31b` を先に）。

以降、`python -m llm_radio_daemon.main --config config\ja\config.toml` などを実行する際は、新しいターミナルを
開くたびに手順2（`.\.venv\Scripts\Activate.ps1`）で仮想環境を有効化してから
実行してください（VSCodeの統合ターミナルなら、後述の設定で自動化できます）。

> **VSCodeを使う場合**：`Ctrl+Shift+P` →「Python: Select Interpreter」→
> `.\.venv\Scripts\python.exe` を選択しておくと、以後VSCodeで開く統合ターミナルは
> 自動で `(.venv)` が有効化された状態で開きます。

`config/ja/config.toml` を編集し、以下を自分の環境に合わせて設定してください
（以下、ファイル名だけを書いているものは同じ `config/<lang>/` の中のファイルです）。

- `[display].enabled` — 既定 `true`（3D表示ウィンドウを出す）。音声のみで使いたい場合は
  `false` に変更してください
- `[llm].engine` — 推論エンジン。`ollama`（既定）/ `lmstudio` / `llamacpp` / `unsloth` から選びます。
  `lmstudio` / `llamacpp` / `unsloth` を選ぶ場合は `host`（例 `http://127.0.0.1:1234`、`/v1` は付けない）も
  併せて設定してください（詳しい例は `config/ja/config.toml.example` の `[llm]` 節にあります）。
  `unsloth`（Unsloth Studio）は Settings → API → Keyless API access の "Chat and inference" を ON に
  すれば localhost からは鍵なしで使えます。ON にしない場合は Settings → API でトークンを発行し
  `[llm].api_key` に設定します（鍵の直書きを避けるなら `api_key = "env:UNSLOTH_API_KEY"` と環境変数名を書けます）。
  `[embedding].engine` も `[llm]` とは独立に選べます（`unsloth` も可ですが、埋め込みモデルは
  Unsloth 側の設定で固定され `[embedding].model` は無視されます。既定は英語専用の `bge-small` なので
  日本語放送では `ollama` を推奨。起動時に warning が出ます）
- `[llm].model` — 使用するモデル名。Ollama なら `ollama pull` 済みの名前、LM Studio /
  llama.cpp / Unsloth Studio ならサーバ上のモデル識別子（`GET /v1/models` や UI で確認）
- `[embedding].model` — 話題の重複検出（任意）に使う埋め込みモデル。既定 `nomic-embed-text`。
  使う場合は事前に `ollama pull nomic-embed-text` が必要（`ollama pull` した直後の
  `ollama list` にはタグ付きで `nomic-embed-text:latest` と出るが、config 側はタグ無しの
  `nomic-embed-text` のままでよい）
- `config_cast.toml` の `[[cast]]` — 出演者（`id` / `name` / `role` / `desc` / `voicevox_speaker_name`）。
  最低1人必須で、多いほどトピックごとの顔ぶれの振れ幅が出ます（config/ja/config_cast.toml.example は18人）。
  `role` は `host` / `assistant` / `other` の3種（省略時 `other`）。`host` / `assistant` は
  前列に並び、抽選で入っていれば進行役に寄せられます。
  `config_cast.toml` が無い、または `[[cast]]` が空だと起動時に即エラーで終了します
- `[[streams]]` — 流したいネットラジオの局（`id` / `url`、表示名は任意の `name`）。
  何局でも並べられ、先頭が既定局になります。**公式にAPIと利用条件を公開している局のみ**を
  指定してください（例として [SomaFM](https://somafm.com/) を config/ja/config.toml.example に記載）。
  radiko 等、地域認証を要するものは対象外です。
  コーナーごとに局を変えたいときは `config_content.toml` の `[[content]]` に
  `stream = ["id", ...]` を書きます（複数書くとそのコーナーに入るたびにランダムで1局。
  省略すると既定局）
- `config_content.toml` の `[[content]]` — 番組表（「どのコンテンツを・いつ・何人で」流すか）。
  必須の設定ファイルで、無いと起動時にエラーになります（`config/ja/config_content.toml.example` を
  コピーして使ってください）

`config.toml` / `config_cast.toml` / `config_content.toml` はどの階層のものも `.gitignore` 済みで、
リポジトリにはコミットされません（入るのは `*.toml.example` だけです）。

## 実行

LLM推論エンジン（Ollama / LM Studio / llama.cpp / Unsloth Studio のうち `[llm].engine` で選んだもの）と
VOICEVOX ENGINE を起動し、仮想環境を有効化した状態で
（プロンプトに `(.venv)` が付いていることを確認）：

**PowerShell（Windows）**

```powershell
python -m llm_radio_daemon.main --config config\ja\config.toml
```

<details>
<summary>macOS / Linux の場合</summary>

```bash
python -m llm_radio_daemon.main --config config/ja/config.toml
```

</details>

`--config` は必須です。コマンドラインを見れば何語で放送中か分かるよう、既定の言語はあえて設けていません。
英語版で放送したいときは `--config` に `config/en/config.toml` を指定するだけで、出演者（`config_cast.toml`）も
番組表（`config_content.toml`）も同じディレクトリのものへ丸ごと切り替わります。

同梱のランチャを使うと第1引数で言語を選べます（仮想環境の有効化も不要）。言語は必須で、省略すると使い方を表示して終了します。
`.venv` が無ければ、起動時にランチャが自動で作成し `requirements.txt` もインストールしてから起動します
（初回だけ数十秒〜数分よけいにかかります）：

```powershell
.\go.bat ja     # 日本語版
.\go.bat en     # 英語版
```

```bash
./go.sh ja      # 日本語版
./go.sh en      # 英語版
```

> **`go.bat` の中身は PowerShell（`go.ps1`）の薄いラッパーです。** cmd.exe のバッチ解析は
> 日本語混じりのファイルに弱く、`chcp` で回避しようとしても文字化けやコマンドの誤認識が
> 起きるため、実処理と日英併記メッセージは `go.ps1` 側に置き、`go.bat` は ASCII のみの
> シムにしています（`.\go.bat` を叩けばそのまま動くので、普段は `go.ps1` を意識する必要は
> ありません）。`request.bat` / `request.ps1` も同じ構成です。

放送中にリクエストを送る `request.bat` も同じく第1引数で言語を指定します
（`.\request.bat ja` / `.\request.bat en`）。言語を間違えると別言語の履歴DBを叩きます。

### 今これが聴きたい（リクエスト）

つけっぱなしで聞いていると「今この番組が聴きたい」が出てきます。放送を止めずに、
別ウィンドウから番組表へ割り込めます。

```powershell
.\request.bat ja now --list          # リクエストできるコーナーの一覧
.\request.bat ja now 論文コーナー      # 今すぐそのコーナーへ（既定30分）
.\request.bat ja now arxiv --for 2h  # type でも指せる。長さは --for で
.\request.bat ja now --clear         # 期限を待たずに番組表へ戻す
```

<details>
<summary>macOS / Linux の場合</summary>

```bash
./request.sh ja now --list
./request.sh ja now 論文コーナー
./request.sh ja now arxiv --for 2h
./request.sh ja now --clear
```

</details>

コーナー名は `config_content.toml` の `label`（未設定なら `type`）です。
指定した時間が過ぎると自動的に番組表へ戻るので、消し忘れても編成は壊れません。

切り替わりぎわに、DJ が一度「リクエストをいただきました」と受けてからコーナーへ
入ります。前のコーナーの作り置き（生成済みの台本・収集済みのネタ）はそこで捨てるので、
数分待たされることはありません。放送プロセスへの通知は要りません（DB を1行書くだけで、
放送側が次に編成を判断する瞬間＝1秒以内に反映されます）。停止中に打っておけば、
次に `go.bat` した時点から効きます。

### 最初からやり直したい（全リセット）

「聴いた/読んだ」状態——ネタの既読判定・放送履歴・読書コーナーやラジオドラマの朗読進捗・
リクエスト履歴——を全部消して、まっさらな状態から放送を再開したいときに使います
（デバッグ用）。`generated_drama_writer` が書き溜めたラジオドラマの本文そのものは消えません。
消えるのは放送側の「どこまで読んだか」だけです。

```powershell
.\request.bat ja reset all           # 確認プロンプトが出ます
.\request.bat ja reset all --yes     # スクリプトから叩くとき用に確認をスキップ
```

<details>
<summary>macOS / Linux の場合</summary>

```bash
./request.sh ja reset all
./request.sh ja reset all --yes
```

</details>

> 仮想環境を有効化せずに直接呼びたい場合は、PowerShellなら
> `.\.venv\Scripts\python.exe -m llm_radio_daemon.main --config config\ja\config.toml`、
> macOS/Linuxなら `./.venv/bin/python -m llm_radio_daemon.main --config config/ja/config.toml`
> でも同じ実行ファイルを直接呼び出せます。

コンソールに現在流れているセリフと、ネットラジオの now playing が表示されます。
`Ctrl+C` で終了します。

## 読書コーナー（青空文庫 / Project Gutenberg）

深夜の時間帯（`config_content.toml` の `[[content]]` で `type = "literary_reading"` の `schedule`。既定 02:00–04:00）に、
[青空文庫](https://www.aozora.gr.jp/) のパブリックドメイン作品を朗読するコーナーです。

朗読テキストは**リポジトリに含めません**。既定（`[reading].auto_fetch = true`）では
起動時に未取得の作品を自動でダウンロードするので、事前準備は不要です。

- 取得先は `data/aozora/`（`.gitignore` 済み）。取得済み本文は `index.csv`、
  候補選定に使う PD 全作品のカタログは `catalog.csv` にキャッシュされます
- 著作権フラグが「なし」の作品のみを対象とし、**翻訳作品は既定で除外**します
  （原著者の権利が切れていても訳者の著作権が別に存続しうるため）
- 読む作品は候補リストからランダムに選ばれます。`[reading].work_selector = "llm"` に
  すると、時刻・季節・直近に読んだ作品を添えて Ollama に候補から1本選ばせます
  （深夜の枠に合わせた「短め・静かめ」等のキュレーション。失敗時はランダムに戻ります）
- 朗読キャラ・つっこみキャラは、他コーナーと同じくその回に `[[cast]]` から
  ランダム抽選された2名です（抽選1人目＝朗読、2人目＝つっこみ）
- パーサの動作確認: `python -m llm_radio_daemon.aozora.parser data/aozora/<作品ID>.txt`

読書コーナーで読み上げるテキストは青空文庫のものです。青空文庫の収録作品および
利用に関する案内は [青空文庫](https://www.aozora.gr.jp/) を参照してください。

### 英語版（Project Gutenberg）

英語版の設定（`config/en/`）では、同じコーナーが [Project Gutenberg](https://www.gutenberg.org/) の
英語の作品を朗読します。流れは日本語版と同じで、原文は LLM を通さずそのまま読み上げ、
つっこみキャラの感想だけを LLM が生成します。

- `literary_reading` のエントリに `corpus_dir = "data/gutenberg"` を書きます（既定は `data/aozora`。
  `config/en/config_content.toml.example` には記入済みです）。取得先は `data/gutenberg/`（`.gitignore` 済み）
- Project Gutenberg が配布しているのは**米国で**パブリックドメインの作品です。
  国によって保護期間が異なる場合があり、この点からも本ソフトは個人が自宅で聴く用途を前提にしています
- パーサの動作確認: `python -m llm_radio_daemon.gutenberg.parser data/gutenberg/<作品ID>.txt`

英語版で読み上げるテキストは Project Gutenberg のものです。収録作品および利用に関する案内は
[Project Gutenberg](https://www.gutenberg.org/) を参照してください。

## 翻訳読書コーナー

日本語版のコーナーです（`type = "translated_reading"`）。Project Gutenberg の英語の作品
（`data/gutenberg/` にキャッシュ）を、朗読しながら LLM が区切りごとに日本語へ訳し、
合間につっこみキャラの感想を挟みます。読書コーナーと違って読み上げる本文そのものが
LLM の出力なので、ほかの生成セリフと同じく NG ワードフィルタの対象になり、訳に誤りが
含まれることもあります。

## ラジオドラマの朗読

LLM に書かせたラジオドラマを、シーン単位で少しずつ朗読するコーナーです
（`[[content]]` の `type = "generated_drama"`）。

**執筆と放送は別プロセスです。** 執筆は重いので、放送中のGPUを取り合わないよう
別プロセスで走らせます。放送側は完成した原稿を読むだけで、LLM を一切呼びません。

原稿は台本形式で、見せ場には効果音（擬音だけの行。例「ゴゴゴゴゴーーーッ！」）が入ります。
放送側は擬音の行を単独チャンクにして前後に厚めの「間」を取り、紙芝居のようなメリハリを付けます
（`pause_se_ms` で調整）。

**おすすめは自動執筆（`auto_write`）です。** `[[content]]` の `type = "generated_drama"` に
`auto_write = true` を書いておくと、放送プロセスが「自分が LLM を使っていない隙」
（ラジオドラマの朗読中・音楽コーナー中）を見つけて `generated_drama_writer` を子プロセスで1シーンずつ
走らせます。話芸コーナーが始まるなど放送側が LLM を使い出したら執筆を中断します。
書き溜まっている在庫が少ないほど短い間隔で、十分先まで書けていれば最大60分あけて
実行します。`auto_concept = true` も併せると、執筆中のラジオドラマが尽きたときに企画立案から
次の1本を自動で立ち上げます。**この2つを有効にすれば、あとは VOICEVOX と放送本体を
起動するだけでラジオドラマコーナーが回り続けます**（下のコマンドは手動で先に書き溜めたい／
特定のラジオドラマを進めたいときだけ使います）。

> 自動執筆を使うときは、Ollama の環境変数 `OLLAMA_NUM_PARALLEL=2` を設定しておくと、
> 万一 執筆と放送の生成が重なっても台本生成がブロックされず「少し遅くなるだけ」で
> 済みます（Windows なら `setx OLLAMA_NUM_PARALLEL 2` 後に Ollama を再起動）。

執筆バッチも `--config` が必須です。原稿の在庫は言語ごとに分かれる
（`data/generated_drama_data_<lang>/` と各言語の DB）ので、書きたい言語の設定を指すこと。

```bash
# 1本立ち上げる（プロット・章立て・登場人物・世界観をまとめて設計する）
python -m llm_radio_daemon.generated_drama_writer --config config/ja/config.toml new \
    --title "灯台守の最後の夜" --premise "廃止が決まった灯台の一夜"

# 1バッチ（＝1シーン）進める。手動でも、タスクスケジューラからでも
python -m llm_radio_daemon.generated_drama_writer --config config/ja/config.toml run --generated-drama-id 1
python -m llm_radio_daemon.generated_drama_writer --config config/ja/config.toml run --auto --scenes 3

# 進行状況（書けたシーン数・次に放送されるシーン）
python -m llm_radio_daemon.generated_drama_writer --config config/ja/config.toml list
```

- 生成物は `data/generated_drama_data_<lang>/`（`.gitignore` 済み）と SQLite に入ります。**放送プロセスを
  動かしたまま実行して構いません**（SQLite は WAL、本文は一時ファイル→rename で書きます）
- 執筆は「全体設計 → ハコ書き → 本文（シーン単位）→ チェック」の4段階。チェックを
  通らなかったシーンは `ready = 0` のまま放送されません（3回書き直しても通らなければ、
  放送を止めないために最後の原稿を採用します）
- 地の文は `narrator_cast_id` の出演者、セリフは `characters.json` の
  `cast_id` / `voicevox_speaker` に対応する出演者が読みます。話者が判別できない
  セリフはナレーターが読みます（放送は止めません）
- **まだ1本も書いていない状態で `[[content]]` を有効にしても放送は止まりません。**
  読むシーンが無ければフィラー／次のコーナーへ自然に落ちます
- 切り分け用: `python -m llm_radio_daemon.generated_drama.parser <本文.txt> --characters <characters.json>`
  （地の文・セリフの割り振りを目視）

## 3Dキャラクター

キャラクターは Ursina のプリミティブ＋手続きメッシュ（`display/poly_character.py`）で
すべてコードから組み立てます。外部のモデルファイルは不要です。`[[cast]]` の `model`
（`girl` / `boy`。`display/models.py`）で種別を選び、`hair_style` / `hair_color` /
`eye_color` / `clothing_color` / `accessory`（ヘッドホン・猫耳・ヘアピン等）で
見た目を個別調整できます。

## 背景演出

画面の背景には、暗く低速なアニメーションを重ねられます。動画ファイルは使わず
すべて手続き生成なので、裏で回っている LLM 推論や音声合成と CPU を取り合いません。

| 指定 | 内容 |
|---|---|
| `grid` | 奥から手前へ流れるワイヤーフレームの格子床 |
| `dust` | ゆっくり周回する微粒子（星屑） |
| `logs` | daemon 自身のログを薄く流す |
| `coderain` | Matrix 風の落下文字 |

`[display].background` が既定値（`"grid+dust"`）で、`"+"` で重ねられます。`"none"` で
背景なし。`[[content]]` 側に `background` を書くと、そのコーナーの間だけ切り替わります。

```toml
[[content]]
type = "hackernews"
background = "grid+coderain"      # 技術ネタの時間帯だけコードレインを重ねる
```

## クレジット

音声合成には [VOICEVOX](https://voicevox.hiroshiba.jp/) を使用しています。
使用するキャラクターごとに利用規約が異なるため、必ず各キャラクターの利用規約を
確認してください。

本ソフトは VOICEVOX および各キャラクターの公式・運営とは無関係の個人制作です。
出演者（`[[cast]]`）は本ソフトのために作ったオリジナルの人格で、VOICEVOX の各キャラクター
本人ではありません（声だけをお借りしています）。各キャラクターの規約は更新されることが
あるため、利用時点の最新の規約を各自で確認してください。

`config/ja/config_cast.toml.example` のロースターが使う話者は以下のとおりです
（`[[cast]].voicevox_speaker_name` を変えた場合はそれに合わせて書き換えること）：

```
VOICEVOX:ナースロボ＿タイプＴ
VOICEVOX:春日部つむぎ
VOICEVOX:ずんだもん
VOICEVOX:冥鳴ひまり
VOICEVOX:猫使ビィ
VOICEVOX:四国めたん
VOICEVOX:あんこもん
VOICEVOX:東北ずん子
VOICEVOX:春歌ナナ
VOICEVOX:中部つるぎ
VOICEVOX:WhiteCUL
VOICEVOX:中国うさぎ
VOICEVOX:東北イタコ
VOICEVOX:櫻歌ミコ
VOICEVOX:東北きりたん
VOICEVOX:小夜/SAYO
VOICEVOX:暁記ミタマ
VOICEVOX:雨晴はう
```

英語版の音声合成には [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)（Apache-2.0）と
[misaki](https://github.com/hexgrad/misaki)（Apache-2.0）を使用しています。misaki の英語 G2P が
数字の読み上げに [num2words](https://github.com/savoirfairelinux/num2words)（LGPL-2.1）を
利用しており、本プロジェクトは非改変のライブラリとして import しているだけですが、
念のためここに明記します。

Kokoro のモデル本体（`kokoro-v1.0.onnx`）とボイスファイル（`voices-v1.0.bin`）はリポジトリに
含めていません。初回起動時に `kokoro_models/` へ自動で取得されます。取得元は、Kokoro-82M を
ONNX 形式に変換して配布している [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx)
（コードは MIT）の GitHub Releases です。モデルとボイスの中身は Kokoro-82M のもの（Apache-2.0）です。
`config.json` だけは Hugging Face の Kokoro-82M リポジトリから
取得します。学習データの内訳と CC BY 音声のクレジットは、公式のモデルカード
（https://huggingface.co/hexgrad/Kokoro-82M）を参照してください。

ネットラジオのデコードには [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg)（BSD-2-Clause）
を使用しており、`pip install -r requirements.txt` の時点でこのパッケージの wheel に
同梱された ffmpeg バイナリが取得されます（本プロジェクト自身がリポジトリや配布物に
バイナリを同梱しているわけではありません）。Windows 版（0.6.0時点）は gyan.dev の
ffmpeg 7.1 "essentials" build で、`--enable-gpl --enable-version3` を含む **GPLv3**
ビルドです。`requirements.txt` ではこのパッケージのバージョンを固定していないため、
将来のインストールでビルド元やビルドオプションが変わる可能性があります。実際に
インストールされているバイナリを確認したい場合は、
`python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"`
で得られるパスに対して `ffmpeg -version` を実行してください。本プロジェクトはこの
バイナリを subprocess 経由で外部プロセスとして呼び出しているだけで、静的・動的
リンクは行っていませんが、GPLv3ビルドである点は認識しておいてください。

お天気コーナーの気象データは [Open-Meteo](https://open-meteo.com/) の API から取得しています
（データは [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。取得した値をもとに LLM がトークにしています）。
Open-Meteo の無料 API は非商用利用が対象です。

## ライセンス

本ソフトの著作権は作者に帰属します。個人が自宅で非商用の目的で動かして楽しむことは自由です。
複製物・改変版の公開・配布と、商用利用は認めていません。詳しくは LICENSE を参照してください。
本ソフトは無保証で提供され、利用によって生じた損害やトラブルについて、作者は責任を負いません。

依存する外部サービス（Ollama, VOICEVOX, 各種ネタ源API, ネットラジオ局）や
上記クレジット欄に挙げたライブラリのライセンス・利用規約はそれぞれ別に確認してください。

本 README の記述と LICENSE の内容が食い違う場合は、LICENSE が優先します。
