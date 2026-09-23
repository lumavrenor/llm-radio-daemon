#!/bin/sh
# Start LLM Radio Daemon
#
# The <lang> argument is required and names the config/<lang>/ directory to
# load. There is deliberately no default, so the command line always shows
# which language is on air.
#
#   ./go.sh ja
#   ./go.sh en
#   ./go.sh en --log-level DEBUG   remaining args pass through to main.py
#
# Relative paths inside the config (kokoro_models/ content/ data/) resolve
# from the repository root (this script cds there itself).
#
# If .venv doesn't exist yet, this script creates it, installs the
# dependencies from requirements.txt, and then launches.
#
# If config/<lang>/config.toml / config_cast.toml / config_content.toml don't
# exist yet, they are auto-copied from the matching .example file (existing
# files are never overwritten).
set -e
cd "$(dirname "$0")"

case "$1" in
  ""|-*)
    echo "usage: ./go.sh <lang> [args...]" >&2
    echo >&2
    echo "  <lang> is a directory name under config/. Currently available:" >&2
    for d in config/*/; do [ -d "$d" ] && echo "    $(basename "$d")" >&2; done
    echo >&2
    echo "  example: ./go.sh ja" >&2
    echo "           ./go.sh en --log-level DEBUG" >&2
    exit 2
    ;;
esac
lang="$1"
shift

config_dir="config/$lang"
for f in config.toml config_cast.toml config_content.toml; do
  if [ ! -f "$config_dir/$f" ] && [ -f "$config_dir/$f.example" ]; then
    cp "$config_dir/$f.example" "$config_dir/$f"
    echo "[first-time setup] Created $config_dir/$f from $config_dir/$f.example." >&2
  fi
done

if [ -x .venv/Scripts/python.exe ]; then
  py=.venv/Scripts/python.exe   # Windows (Git Bash)
elif [ -x .venv/bin/python ]; then
  py=.venv/bin/python
else
  echo "[first-time setup] Virtual environment (.venv) not found. Creating it..." >&2

  bootstrap_py=""
  for cand in python3 python; do
    if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "" >/dev/null 2>&1; then
      bootstrap_py="$cand"
      break
    fi
  done
  if [ -z "$bootstrap_py" ]; then
    echo "[error] No working python / python3 command found." >&2
    exit 1
  fi
  "$bootstrap_py" -m venv .venv

  if [ -x .venv/Scripts/python.exe ]; then
    py=.venv/Scripts/python.exe   # Windows (Git Bash)
  else
    py=.venv/bin/python
  fi

  echo "[first-time setup] Installing dependencies (pip install -r requirements.txt)..." >&2
  "$py" -m pip install -q --upgrade pip
  "$py" -m pip install -q -r requirements.txt
  echo "[first-time setup] Done." >&2

  if [ "$lang" = "en" ]; then
    echo "[note] The English edition (en) also needs:" >&2
    echo "         pip install -r requirements-en.txt && python -m spacy download en_core_web_sm" >&2
  fi
fi

exec "$py" -m llm_radio_daemon.main --config "config/$lang/config.toml" "$@"
