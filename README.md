**English** | [日本語](README.ja.md)

# LLM Radio Daemon

A 24/7 "radio show" run by a local LLM. It picks up topics from various sources
around the net and turns them into back-and-forth banter among a rotating panel drawn
from a cast of a dozen-plus characters (host, assistant, plus several "angle"
characters) — anywhere from 2 to 10 on air at once, depending on the segment.
An internet radio stream plays underneath as BGM, and its volume automatically
ducks whenever someone is talking.

This isn't TV — it's "radio with a picture." Talk is the main act, BGM is
background. The goal is a show that still works as a show even with the screen off.

This is a personal hobby project. I don't respond to questions or requests
via Issues, DMs, etc.

Written entirely in Python. The 3D display runs on the
[Ursina](https://www.ursinaengine.org/) engine, and character motion (idle
sway, gestures, mouth movement while talking) is driven directly by code
rather than baked animation clips. Each character's setup (role, cast size,
angle) and appearance (hairstyle, hair/eye/clothing color, etc.) can be
customized per-character via config.

Because it runs comfortably on small local models too (2B–4B parameters),
it also works as an informal way to compare local LLMs against each other —
point `[llm].model` at a different model and listen to how its jokes,
reactions, and narration style differ. No GPU? 2B–4B models run fine on CPU
alone, and Ollama's cloud models (see below) are verified working too, so a
GPU-less laptop is enough to give it a try.

**All main features are implemented and verified working**, including
multiple topic sources, filler talk, time-of-day scheduling, and
embedding-based dedup (only the 24-hour soak test has not been run).

## Demo Video

[![Demo video](https://img.youtube.com/vi/o028pIn0Y_U/0.jpg)](https://www.youtube.com/watch?v=o028pIn0Y_U&t=11s)

YouTube: https://www.youtube.com/watch?v=o028pIn0Y_U&t=11s

(This video has no audio track.)

## What this project is (and isn't)

This is not practical software. It's an experimental project for a programmer to
play with on their own machine. If you actually wanted a practical 24/7 radio show,
the sensible choice would be a large cloud API. That's not the point here — the
point is to show how far you can push things using nothing but a small local model
(2B–4B parameters) and your own PC. Running for free, with no usage charges, comes
at a cost: the output quality doesn't match a large cloud model's.

## About what the LLM generates (important)

Everything the cast says — dialogue, chit-chat, the late-night advice-column
answers — is fiction, generated on the fly by a local LLM. None of it is checked by
a real expert.

The project does make an effort to keep generated content on the safe side. The
prompts always include instructions like "don't state medical or health claims as
fact," "don't recommend investments or financial products," "don't take a side on
politics or religion," and "no personal attacks or R-18 content." Free-form segments
like the advice-column corner explicitly forbid heavy subject matter — suicidal
ideation, self-harm, abuse — as an added layer on top of that.

On the output side, there is also an NG-word filter that checks every generated
line just before it is spoken. A line containing clearly inappropriate words
(sexual content, slurs, direct abuse) is not read out in the character's voice;
it is replaced with a short silence and logged instead. Literary-reading passages
read verbatim from Aozora Bunko or Project Gutenberg are exempt. The word list
lives in `llm_radio_daemon/sensitive.py`, and you can add to it.

That said, prompt-level instructions are not a mechanism that reliably
controls what the LLM outputs, and the NG-word filter only matches fixed words
and phrases; it cannot catch paraphrases or lines that are inappropriate only in
context. An LLM can ignore its instructions, or generate
something factually wrong or genuinely inappropriate/dangerous in a perfectly
confident tone, and current generative AI technology cannot bring that risk to
zero. This isn't a gap in the implementation — it's an inherent limit of the
technology itself.

- **Don't treat anything related to medical, legal, financial, or safety matters
  as real advice and act on it.**
- The author is not liable for any damage arising from this software's output.
  Acting on what it says, and any consequences of that, is entirely on you.

## Features

- **Fully local, zero API cost** — LLM inference (Ollama / LM Studio / llama.cpp / Unsloth Studio —
  pick one), text-to-speech (VOICEVOX for Japanese / Kokoro for English), and
  audio decoding (ffmpeg) all run on
  your own PC. You can leave it running 24/7 without incurring any external
  API usage charges. This is the whole point of the project.
- **Panel talk** — Cast members are defined in `[[cast]]` in `config_cast.toml`
  (required; `config/en/config_cast.toml.example` ships with 18). For each topic, `min_speakers`–`max_speakers` members are drawn by lottery and
  banter it out through separate voices (VOICEVOX for Japanese / Kokoro for
  English) — jokes, straight-man reactions,
  tangents, naive questions (this is not a single narrator reading a monologue).
- **An external internet radio station as BGM, ducked only while someone talks** —
  An internet radio stream plays continuously; its volume automatically dips
  only while talk is happening.
- **Continuously pulls topics from multiple sources** — Wikipedia, Hacker News,
  arXiv, arbitrary RSS feeds, and the currently-playing track (via MusicBrainz)
  are all continuously scanned for topics, which the LLM keeps turning into
  scripts. Late at night, fictional advice-column letters (generated by the LLM,
  with no real person involved) join the rotation too.
- **Timetable-style scheduling** — `[[content]]` in `config_content.toml` says
  which content plays when and with how many cast members. Exactly one content
  entry is active for any given time slot (first match in file order wins), and
  the cast for each round is drawn at random from `[[cast]]`.
- **Dedup by embedding** — Topics that are semantically too similar to
  recent talk are automatically dropped so the show doesn't repeat itself.
- **Weather segment** — Set a location (lat/lon) under `[weather]` in
  `config.toml` and the current weather is fetched from
  [Open-Meteo](https://open-meteo.com/) (no API key needed) and turned into a
  dedicated weather segment. The same value also feeds filler chit-chat as a
  "current situation," and topic-list tags like `@rain` / `@cold` let you draw
  only prompts that fit the current weather. The location you set here isn't
  the listener's location — it's studio-location worldbuilding for the show, so
  there's no automatic detection from IP or geolocation. If the fetch fails, the
  show simply doesn't mention weather.
- **Reading corner (Project Gutenberg / Aozora Bunko)** — Late at night only, a
  "reader" cast member reads aloud from a public-domain work (Project Gutenberg
  for the English broadcast, Aozora Bunko for the Japanese one), with a
  "commentator" cast member chiming in with reactions between breaks. The
  original text is read verbatim with no LLM involved; only the reactions are
  LLM-generated. It advances a little at a time across days, like a serial
  drama.
- **Translated reading corner** — For the Japanese broadcast: English-language
  works from Project Gutenberg are translated into Japanese by the LLM chunk by
  chunk and narrated, with a commentator's reactions in between. Unlike the
  reading corner, the narrated text itself is LLM output.
- **Radio drama corner** — A separate writing batch process
  (`generated_drama_writer`) designs the plot, chapter breakdown, and characters,
  and writes the manuscript scene by scene. The broadcast side just reads the
  finished manuscript aloud — narration goes to a narrator voice, dialogue is
  routed to each character's own voice. LLM load during the broadcast itself is
  zero.
- **Great-figures talk** — Talks based on Wikipedia information about
  historical figures.

## Design assumptions and notes if you publish anything

This software is designed for personal, non-commercial, at-home use.
Accordingly, the following are deliberately not implemented:

- Streaming/broadcast output functionality
- Recording or saving audio/music to files
- Local music generation, or pulling audio from YouTube
- Outbound write actions driven by LLM output (file operations, sending email,
  code execution, etc.) — in this system, LLM output **only ever turns into
  speech and disappears**

### If you publish or stream a video that includes the audio

Material this project pulls in at runtime — each VOICEVOX character, the
internet radio stations, the various topic-source APIs — carries its own,
separate terms of use. If you publish or stream a screen recording that
includes the generated audio, clearing the rights for that is your own
responsibility. Some VOICEVOX characters in particular put conditions on
combining their voice with AI-generated lines, on publishing or
monetizing the result, or on what kind of content it can be used for.
When crediting, list only the characters you actually used, worded exactly
as each character's own terms require. The author does not vouch for or
speak on behalf of any of these terms.

And since, as described above, the LLM's lines can't be fully controlled,
**a character may end up saying something its terms of use forbid (politics,
religion, sexual content, personal attacks, and so on) in its own voice.**
If you only listen at home, the audio never leaves your room. But if you
record, stream or otherwise publish it, check the content before publishing
and judge for yourself whether it stays within each character's terms.

## What you need

| Item | Purpose |
|---|---|
| A local LLM inference engine (pick one of the below) | Generates scripts, filler, etc. Chosen via `[llm].engine` in `config.toml` |
| &nbsp;&nbsp;· [Ollama](https://ollama.com/) | The default. Run `ollama pull` for the model you'll use beforehand (`engine = "ollama"`). Ollama's cloud models (names ending in `-cloud`, e.g. `gpt-oss:120b-cloud` — lets you try models too big for your own GPU) can be specified directly too |
| &nbsp;&nbsp;· [LM Studio](https://lmstudio.ai/) | Start its local server (OpenAI-compatible API) beforehand. `engine = "lmstudio"` / `host = "http://127.0.0.1:1234"` |
| &nbsp;&nbsp;· [llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama-server`) | Start `llama-server` beforehand. `engine = "llamacpp"` / `host = "http://127.0.0.1:8080"` |
| &nbsp;&nbsp;· [Unsloth Studio](https://unsloth.ai/docs/new/studio) | Start the server with `unsloth studio` and load a model in the UI (set context length to 8192 or more). `engine = "unsloth"` / `host = "http://127.0.0.1:8888"`. Turning on Settings → API → Keyless API access → "Chat and inference" lets localhost use it without a key (otherwise set a token in `api_key`). `[embedding]` can also use `unsloth`, but the embedding model is fixed by Unsloth Studio's own settings (default is the English-only bge-small), so `ollama` is recommended for Japanese |
| [VOICEVOX ENGINE](https://voicevox.hiroshiba.jp/) | Text-to-speech (HTTP API, default `http://127.0.0.1:50021`) |
| [ffmpeg](https://ffmpeg.org/) | Used to decode the internet radio stream (MP3/AAC/etc.) into audio data. It's bundled inside the wheel of `imageio-ffmpeg` in `requirements.txt`, so it's already available once `pip install` completes — no manual install or PATH setup needed (it only falls back to searching PATH for `ffmpeg` if that fails) |
| Python 3.11+ | Runtime for the main program |
| [Visual Studio Code](https://code.visualstudio.com/) (recommended, optional) | Installing it lets the Python extension auto-detect the virtual environment and activate it automatically every time you open an integrated terminal — sidestepping the "activating the virtual environment" hassle (PowerShell execution policy, etc.) in the setup steps below entirely |

**Check each dependency's own license and terms of use yourself.** This README
does not speak for them.

> **A note on platform support**: development and testing are done on Windows
> (PowerShell) only. The macOS / Linux steps below are provided as a
> best-effort reference and have not been verified on real hardware.
> There's no timeline for verifying either macOS or Linux on real hardware.

## Setup (Windows / PowerShell)

These steps are easy to trip up on, so each one spells out what success looks
like. Go through them one at a time.

### 0. Confirm Python is installed

```powershell
python --version
```

Success looks like a version number, e.g. `Python 3.11.x`.

If instead **nothing is printed and the Microsoft Store opens**, the `python`
command is pointing at an "app execution alias" with no real binary behind it.
Reinstall Python 3.11+ from [python.org](https://www.python.org/downloads/)
(check "Add python.exe to PATH" in the installer).

### 1. Create a virtual environment

> If you'll be launching via `go.bat` / `go.sh`, you can skip this step — it
> creates `.venv` automatically if missing and installs dependencies
> (`requirements.txt`) before launching (see the "Running" section below).
> This step is for using `python` directly.

```powershell
python -m venv .venv
```

This prints nothing and takes a few to a dozen-odd seconds (that's not a
failure). Success looks like a `.venv` folder appearing at the project root.

> Use **`.venv`** (leading dot), not `venv`. The bundled `go.bat` / `go.sh` /
> `request.bat` all hard-code `.venv`, so a different name will produce
> "not found"-style errors.

<details>
<summary>macOS / Linux</summary>

```bash
python3 -m venv .venv
```

</details>

### 2. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

Success looks like `(.venv)` appearing at the start of the prompt. **For every
step after this, confirm `(.venv)` is present before running it** — running
`pip install` without it will install into some unrelated Python environment.

If you see an error like this:

```
File ...\Activate.ps1 cannot be loaded because running scripts is disabled
on this system.
```

PowerShell's execution policy is set too strictly. Run the following once (no
admin rights needed), then retry the activation command above.

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

> If you're using Command Prompt (cmd.exe) instead, `.venv\Scripts\activate.bat`
> (the `.bat` can be omitted) does the same thing without hitting the execution
> policy restriction.

<details>
<summary>macOS / Linux</summary>

```bash
source .venv/bin/activate
```

</details>

### 3. Install dependencies

> If you launch via `go.bat` / `go.sh`, the script installs the contents of
> `requirements.txt` automatically, but only the first time it creates
> `.venv` (this does not cover the English-language `requirements-en.txt` —
> install that separately with the command below if you're using the English
> version).

Confirm `(.venv)` is present in the prompt, then:

```powershell
pip install -r requirements.txt
```

<details>
<summary>macOS / Linux</summary>

```bash
pip install -r requirements.txt
```

</details>

### 4. Prepare config files

> If you'll be launching via `go.bat` / `go.sh`, you can skip this step — any
> missing file in the 3-file set is auto-copied from its matching `.example`
> before launching (existing files are never overwritten). This step is for
> using `python` directly.

Config lives as **a set of 3 files per language**, under `config/<lang>/`.

```
config/
  ja/  config.toml  config_cast.toml  config_content.toml   # Japanese (VOICEVOX)
  en/  config.toml  config_cast.toml  config_content.toml   # English (Kokoro)
```

`config_cast.toml` and `config_content.toml` are read from the **same
directory** as `config.toml`, so switching languages is just a matter of which
path you pass at launch.

To use the English version (swap `en` for `ja` for Japanese):

```powershell
Copy-Item config\en\config.toml.example         config\en\config.toml
Copy-Item config\en\config_cast.toml.example    config\en\config_cast.toml
Copy-Item config\en\config_content.toml.example config\en\config_content.toml
```

<details>
<summary>macOS / Linux</summary>

```bash
cp config/en/config.toml.example         config/en/config.toml
cp config/en/config_cast.toml.example    config/en/config_cast.toml
cp config/en/config_content.toml.example config/en/config_content.toml
```

</details>

If you use Ollama (the default `engine`), pull the default model from
`config.toml.example` — `gemma4:e4b`, a lightweight model that runs on
8GB-class GPUs — before your first run:

```powershell
ollama pull gemma4:e4b
```

Skipping this makes the first launch fail because the model isn't available
yet. If you have 16GB+ of VRAM, swap `[llm].model` for a bigger model such as
`gemma4:31b` for better scripts (pull that instead: `ollama pull gemma4:31b`).

From here on, whenever you run something like
`python -m llm_radio_daemon.main --config config\en\config.toml`, activate the
virtual environment first (step 2, `.\.venv\Scripts\Activate.ps1`) every time
you open a new terminal (with VS Code's integrated terminal, this can be
automated — see below).

> **If you use VS Code**: `Ctrl+Shift+P` → "Python: Select Interpreter" →
> pick `.\.venv\Scripts\python.exe`, and every integrated terminal you open in
> VS Code afterward will automatically have `(.venv)` active.

Edit `config/en/config.toml` and set the following for your environment
(below, any bare filename refers to the file of that name in the same
`config/<lang>/` directory):

- `[display].enabled` — defaults to `true` (shows the 3D display window). Set
  to `false` if you want audio only
- `[llm].engine` — the inference engine: `ollama` (default) / `lmstudio` /
  `llamacpp` / `unsloth`. If you pick `lmstudio` / `llamacpp` / `unsloth`, also
  set `host` (e.g. `http://127.0.0.1:1234`, no trailing `/v1`) — see the
  `[llm]` section of `config/en/config.toml.example` for worked examples.
  For `unsloth` (Unsloth Studio), turning on Settings → API → Keyless API
  access → "Chat and inference" lets you use it from localhost without a key.
  Otherwise, issue a token under Settings → API and set it as `[llm].api_key`
  (to avoid hardcoding the key, you can write `api_key = "env:UNSLOTH_API_KEY"`
  and name an environment variable instead). `[embedding].engine` can be chosen
  independently of `[llm]` (it can also be `unsloth`, but then the embedding
  model is fixed by Unsloth Studio's own settings and `[embedding].model` is
  ignored — the default is the English-only `bge-small`, so for Japanese
  broadcasts `ollama` is recommended; a warning is printed at startup if you
  don't)
- `[llm].model` — the model name to use. For Ollama, the name you've already
  `ollama pull`ed; for LM Studio / llama.cpp / Unsloth Studio, the model
  identifier on the server (check `GET /v1/models` or the UI)
- `[embedding].model` — the embedding model used for topic dedup (optional).
  Default `nomic-embed-text`. If you use it, run
  `ollama pull nomic-embed-text` first (right after pulling, `ollama list`
  shows it tagged as `nomic-embed-text:latest`, but the config value should
  stay untagged as `nomic-embed-text`)
- `[[cast]]` in `config_cast.toml` — cast members (`id` / `name` / `role` /
  `desc` / `voicevox_speaker_name` — or, for the English/Kokoro config, the
  equivalent TTS voice field). At least 1 is required; more cast members means
  more variety in who shows up per topic (`config/ja/config_cast.toml.example`
  ships 18). `role` is one of `host` / `assistant` / `other` (defaults to
  `other`); `host` / `assistant` sit in the front row and are favored for the
  moderator role when drawn.
  If `config_cast.toml` is missing, or `[[cast]]` is empty, startup fails
  immediately with an error
- `[[streams]]` — internet radio stations to play (`id` / `url`, with an
  optional display `name`). You can list as many as you like; the first one
  is the default. **Only use stations that officially publish an API and
  terms of use** (e.g. [SomaFM](https://somafm.com/), listed as an example in
  `config/en/config.toml.example`). Stations requiring regional
  authentication (e.g. radiko) are out of scope.
  To use a different station for a specific segment, set
  `stream = ["id", ...]` on that entry in `config_content.toml` (list more than
  one and a station is picked at random each time that segment starts; omit it
  to use the default)
- `[[content]]` in `config_content.toml` — the programming schedule (which
  content plays when, with how many people). This file is required; startup
  fails without it (copy `config/en/config_content.toml.example` to get
  started)

`config.toml` / `config_cast.toml` / `config_content.toml`, at every level, are
all `.gitignore`d and never committed to the repo (only the `*.toml.example`
files are).

## Running

With the LLM inference engine (whichever of Ollama / LM Studio / llama.cpp /
Unsloth Studio you chose via `[llm].engine`) and VOICEVOX ENGINE running, and
the virtual environment active (confirm `(.venv)` is in the prompt):

**PowerShell (Windows)**

```powershell
python -m llm_radio_daemon.main --config config\en\config.toml
```

<details>
<summary>macOS / Linux</summary>

```bash
python -m llm_radio_daemon.main --config config/en/config.toml
```

</details>

`--config` is required (there's deliberately no default language, so the
command line itself always shows which language is broadcasting). To
broadcast in Japanese, just point at `config/ja/config.toml` instead — the
cast (`config_cast.toml`) and schedule (`config_content.toml`) both switch
over wholesale, since they live in the same directory.

The bundled launcher lets you pick the language as the first argument (and
skips activating the virtual environment yourself). The language is required;
omitting it prints usage and exits. If `.venv` doesn't exist yet, the launcher
creates it and installs `requirements.txt` automatically before launching
(this adds a few tens of seconds to a few minutes, but only the first time):

```powershell
.\go.bat ja     # Japanese
.\go.bat en     # English
```

```bash
./go.sh ja      # Japanese
./go.sh en      # English
```

> **`go.bat` is a thin wrapper around PowerShell (`go.ps1`).** cmd.exe's batch
> parsing handles files with Japanese text poorly, and working around it with
> `chcp` just trades one set of mojibake/misparsed-command issues for another —
> so the actual logic and the bilingual messages live in `go.ps1`, and
> `go.bat` is an ASCII-only shim (running `.\go.bat` just works, so you don't
> normally need to think about `go.ps1` at all). `request.bat` / `request.ps1`
> follow the same pattern.

`request.bat`, used to send requests to a running broadcast, also takes the
language as its first argument (`.\request.bat ja` / `.\request.bat en`).
Getting the language wrong means you're poking the wrong language's history
database.

### "I want to hear this right now" (requests)

If you leave it running and listening, you'll eventually want "this specific
segment, right now." You can interrupt the schedule from a separate window
without stopping the broadcast.

```powershell
.\request.bat en now --list          # list segments you can request
.\request.bat en now arxiv           # switch to that segment now (30 min by default)
.\request.bat en now arxiv --for 2h  # you can also name it by type; length via --for
.\request.bat en now --clear         # drop the request and go back to the schedule
```

<details>
<summary>macOS / Linux</summary>

```bash
./request.sh en now --list
./request.sh en now arxiv
./request.sh en now arxiv --for 2h
./request.sh en now --clear
```

</details>

The segment name is the `label` in `config_content.toml` (or `type` if
unset). Once the requested duration elapses, it automatically reverts to the
schedule — forgetting to clear it won't break the programming.

Right at the switch, the DJ acknowledges it once ("Got a request...") before
moving into the segment. Whatever the previous segment had queued up
(generated scripts, collected topics) is discarded at that point, so you're
never left waiting more than a couple of minutes. No need to notify the
broadcast process — it's just one row written to the DB, and the broadcast
side picks it up the next time it decides what to play (within a second).
Set it while the broadcast is stopped, and it takes effect the moment you
next run `go.bat`.

### Starting over from scratch (full reset)

Use this when you want to wipe all "listened/read" state — topic dedup
history, broadcast history, reading-corner and radio-drama reading
progress, request history — and restart the broadcast completely fresh
(debugging use). The manuscript text `generated_drama_writer` has written up
is not deleted; only the broadcast side's "how far have I read" state is.

```powershell
.\request.bat en reset all           # prompts for confirmation
.\request.bat en reset all --yes     # skip the prompt, e.g. when scripted
```

<details>
<summary>macOS / Linux</summary>

```bash
./request.sh en reset all
./request.sh en reset all --yes
```

</details>

> If you want to call the executable directly without activating the virtual
> environment, PowerShell:
> `.\.venv\Scripts\python.exe -m llm_radio_daemon.main --config config\en\config.toml`,
> macOS/Linux: `./.venv/bin/python -m llm_radio_daemon.main --config config/en/config.toml`
> both invoke the same entry point directly.

The console shows the line currently being spoken and the internet radio's
now-playing info. `Ctrl+C` to quit.

## Reading corner (Aozora Bunko / Project Gutenberg)

Late at night (the `schedule` of the `[[content]]` entry with
`type = "literary_reading"` in `config_content.toml`; defaults to 02:00–04:00),
a "reader" cast member reads aloud from a public-domain work from
[Aozora Bunko](https://www.aozora.gr.jp/) (the Japanese public-domain text
archive), with a "commentator" cast member chiming in between breaks.

Reading text is **not included in the repo**. By default
(`[reading].auto_fetch = true`), any not-yet-downloaded work is fetched
automatically at startup, so no prep is needed.

- Downloads go to `data/aozora/` (`.gitignore`d). Fetched text is indexed in
  `index.csv`; the catalog of all public-domain works used for candidate
  selection is cached in `catalog.csv`
- Only works flagged with no copyright restriction are used, and
  **translations are excluded by default** (a translator's own copyright can
  still be in effect even once the original author's has lapsed)
- The work read is picked at random from the candidate list. Setting
  `[reading].work_selector = "llm"` has Ollama pick one from the candidates
  instead, given the time of day, season, and recently-read works (curating
  e.g. "shorter, quieter" picks for the late-night slot; falls back to random
  on failure)
- As with other segments, the reader and commentator are 2 cast members drawn
  at random from `[[cast]]` each time (1st drawn = reader, 2nd = commentator)
- To sanity-check the parser:
  `python -m llm_radio_daemon.aozora.parser data/aozora/<work-id>.txt`

The text read in this segment comes from Aozora Bunko. See
[Aozora Bunko](https://www.aozora.gr.jp/) for what it hosts and its terms of
use.

### English broadcast (Project Gutenberg)

With the English config (`config/en/`), the same corner reads English-language
works from [Project Gutenberg](https://www.gutenberg.org/) instead. The flow is
identical — the original text is read verbatim with no LLM involved, and only
the commentator's reactions are generated.

- Set `corpus_dir = "data/gutenberg"` on the `literary_reading` entry (the
  default is `data/aozora`; `config/en/config_content.toml.example` already
  sets it). Downloads go to `data/gutenberg/` (`.gitignore`d)
- Project Gutenberg distributes works that are in the public domain **in the
  United States**. Copyright terms elsewhere can differ, which is one more
  reason this software assumes personal, at-home listening
- To sanity-check the parser:
  `python -m llm_radio_daemon.gutenberg.parser data/gutenberg/<ebook-id>.txt`

The text read in the English broadcast comes from Project Gutenberg. See
[Project Gutenberg](https://www.gutenberg.org/) for what it hosts and its
terms of use.

## Translated reading corner

A Japanese-broadcast corner (`type = "translated_reading"`). It takes
English-language works from Project Gutenberg (cached in `data/gutenberg/`) and
has the LLM translate them into Japanese chunk by chunk as the reader narrates,
with a commentator's reactions in between. Unlike the reading corner, the
narrated text itself is LLM output, so it goes through the same NG-word filter
as any other generated line, and the translation may contain mistakes.

## Radio drama reading

A segment that reads a radio drama, written by an LLM, aloud a little at a
time, scene by scene (`[[content]]` entry with `type = "generated_drama"`).

**Writing and broadcasting are separate processes.** Writing is heavy, so it
runs in a separate process to avoid competing for the GPU during broadcast.
The broadcast side just reads the finished manuscript — it never calls an
LLM.

The manuscript is script-formatted; big moments get sound-effect lines
(onomatopoeia-only lines, e.g. "RUMMMBLE!!"). The broadcast side treats each
sound-effect line as its own chunk with extra pauses before and after, giving
it a comic-panel-like beat (tune with `pause_se_ms`).

**Auto-write (`auto_write`) is recommended.** Set `auto_write = true` on the
`[[content]]` entry with `type = "generated_drama"`, and the broadcast process
will find moments when it isn't itself using the LLM (during a radio drama
reading, during a music segment) and run `generated_drama_writer` as a child
process, one scene at a time. It pauses writing as soon as the broadcast side
starts using the LLM again (e.g. a talk segment starting up). The less stock
is banked, the shorter the interval between writing runs; once there's enough
stock ahead, it backs off to as long as 60 minutes between runs. Combine it
with `auto_concept = true` and, once the radio drama being written runs out,
it automatically kicks off planning for the next one too. **With both
enabled, all you need to do is start VOICEVOX and the broadcast itself, and
the radio drama segment keeps itself fed** (the commands below are only for
manually banking ahead, or advancing a specific radio drama).

> When using auto-write, setting Ollama's `OLLAMA_NUM_PARALLEL=2` environment
> variable means that even if writing and broadcast generation happen to
> overlap, script generation just slows down slightly instead of blocking
> (on Windows: `setx OLLAMA_NUM_PARALLEL 2`, then restart Ollama).

The writer batch also requires `--config`. Manuscript stock is kept separate
per language (`data/generated_drama_data_<lang>/` and a per-language DB), so
point it at whichever language's config you want to write for.

```bash
# Start a new one (designs plot, chapter breakdown, characters, and world all at once)
python -m llm_radio_daemon.generated_drama_writer --config config/en/config.toml new \
    --title "The Lighthouse Keeper's Last Night" --premise "One last night at a lighthouse slated for decommission"

# Advance one batch (= one scene). Works fine manually or from a task scheduler
python -m llm_radio_daemon.generated_drama_writer --config config/en/config.toml run --generated-drama-id 1
python -m llm_radio_daemon.generated_drama_writer --config config/en/config.toml run --auto --scenes 3

# Progress (scenes written so far, which scene broadcasts next)
python -m llm_radio_daemon.generated_drama_writer --config config/en/config.toml list
```

- Output goes to `data/generated_drama_data_<lang>/` (`.gitignore`d) and
  SQLite. **It's fine to run this while the broadcast process is also
  running** (SQLite uses WAL; manuscript text is written to a temp file then
  renamed into place)
- Writing happens in 4 stages: overall design → chapter/scene outline → prose
  (scene by scene) → check. A scene that fails the check stays `ready = 0` and
  isn't broadcast (if it still fails after 3 rewrites, the last draft is used
  anyway, so the broadcast never stalls on it)
- Narration is read by the `narrator_cast_id` cast member; dialogue is routed
  to whichever cast member matches the `cast_id` / `voicevox_speaker` in
  `characters.json`. A line whose speaker can't be resolved falls back to the
  narrator (the broadcast is never blocked by this)
- **Enabling the `[[content]]` entry before any radio drama has been written
  yet does not stall the broadcast.** With no scene to read, it falls through
  naturally to filler / the next segment
- For debugging:
  `python -m llm_radio_daemon.generated_drama.parser <manuscript.txt> --characters <characters.json>`
  (visually inspect the narration/dialogue split)

## 3D characters

Characters are built entirely from Ursina primitives and procedural meshes
(`display/poly_character.py`) — no external model files needed. Pick a kind
per `[[cast]]` entry with `model` (`girl` / `boy`, see `display/models.py`),
and tweak looks with `hair_style`, `hair_color`, `eye_color`,
`clothing_color`, and `accessory` (headphones / cat ears / hairpin / etc.).

## Background effects

The screen background can layer dark, slow-moving animations. No video files
are used — everything is procedurally generated, so it doesn't compete for
CPU with the LLM inference or TTS running behind the scenes.

| Value | Effect |
|---|---|
| `grid` | A wireframe floor grid scrolling from back to front |
| `dust` | Slowly orbiting particles (stardust) |
| `logs` | The daemon's own log lines, faintly scrolling |
| `coderain` | Matrix-style falling characters |

`[display].background` defaults to `"grid+dust"`; combine layers with `"+"`.
`"none"` disables the background entirely. Setting `background` on a
`[[content]]` entry switches it just for that segment.

```toml
[[content]]
type = "hackernews"
background = "grid+coderain"      # layer in the code-rain only during tech-news segments
```

## Credits

Text-to-speech uses [VOICEVOX](https://voicevox.hiroshiba.jp/). Terms of use
differ per character, so be sure to check the terms for each character you
use.

This software is an independent, personal project with no affiliation to
VOICEVOX or to any character's official rights holders. The cast (`[[cast]]`)
are original personas created for this show, not the VOICEVOX characters
themselves — only their voices are borrowed. Each character's terms of use
can change, so check the latest terms yourself at the time you use it.

The voices used by the roster in `config/ja/config_cast.toml.example` are
as follows (if you change `[[cast]].voicevox_speaker_name`, update this to
match):

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

The English TTS backend uses [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
(Apache-2.0) and [misaki](https://github.com/hexgrad/misaki) (Apache-2.0).
misaki's English G2P uses [num2words](https://github.com/savoirfairelinux/num2words)
(LGPL-2.1) to read numbers aloud; this project only imports it as an
unmodified library, but it's noted here for completeness.

The Kokoro model (`kokoro-v1.0.onnx`) and voice file (`voices-v1.0.bin`) are
not included in this repository. They are downloaded into `kokoro_models/` on
first run from the GitHub Releases of
[kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) (code under MIT),
which distributes an ONNX conversion of Kokoro-82M; the model and voice data
themselves are Kokoro-82M's (Apache-2.0). Only `config.json` is fetched
from the Kokoro-82M repository on Hugging Face. For training data details and
CC BY attributions, see the official model card:
https://huggingface.co/hexgrad/Kokoro-82M

Internet-radio decoding uses [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg)
(BSD-2-Clause); running `pip install -r requirements.txt` fetches the ffmpeg
binary bundled inside that package's wheel (this project's own repo and
distribution do not bundle the binary themselves). The Windows build (as of
0.6.0) is gyan.dev's ffmpeg 7.1 "essentials" build, which is a **GPLv3** build
(`--enable-gpl --enable-version3`). `requirements.txt` doesn't pin this
package's version, so a future install could fetch a newer build from a
different source or with different build options; to check what's actually
installed, run `python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"`
and pass the printed path to `ffmpeg -version`. This project only invokes the
binary as an external subprocess — it performs no static or dynamic linking —
but it's worth being aware it's a GPLv3 build.

Weather data by [Open-Meteo.com](https://open-meteo.com/)
([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)); the LLM turns the
fetched values into talk. Open-Meteo's free API is for non-commercial use.

## License

Copyright in this software belongs to the author. Running it for personal,
non-commercial use at home is fine.
Publishing or distributing copies or modified versions, and any commercial use,
are not permitted. See LICENSE for details.
This software is provided with no warranty; the author is not liable for any
damage or trouble arising from its use.

The external services this project depends on (Ollama, VOICEVOX, the various
topic-source APIs, internet radio stations) and the libraries listed in the
Credits section above each have their own separate licenses and terms of use
— check those individually.

If anything in this README conflicts with LICENSE, LICENSE takes precedence.
