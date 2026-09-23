#!/bin/sh
# Send a request to the running broadcast
#
# The language argument is REQUIRED: it names the config/<lang>/ directory
# to load. There is deliberately no default -- picking the wrong language
# writes to the wrong database (ja: data/llm_radio_daemon.ja.db / en: data/llm_radio_daemon.en.db).
#
#   ./request.sh ja
#   ./request.sh en
#
# Common uses (see llm_radio_daemon/request.py for the full list):
#
#   ./request.sh ja now --list           list the corners you can request
#   ./request.sh ja now arxiv            switch to that corner now (30 min by default)
#   ./request.sh ja now arxiv --for 2h   ... for a different length
#   ./request.sh ja now --clear          drop the request, back to the timetable
#   ./request.sh ja reset all            wipe read/broadcast state, back to "nothing seen yet"
#   ./request.sh ja reset data           relocate & recreate all of data/ (last resort; stop the broadcast first)
#
# Relative paths inside the config resolve from the repository root
# (this script cds there itself).
set -e
cd "$(dirname "$0")"

case "$1" in
  ""|-*)
    echo "usage: ./request.sh <lang> [args...]" >&2
    echo >&2
    echo "  <lang> is a directory name under config/. Currently available:" >&2
    for d in config/*/; do [ -d "$d" ] && echo "    $(basename "$d")" >&2; done
    echo >&2
    echo "  example: ./request.sh ja now --list" >&2
    echo "           ./request.sh ja now arxiv --for 30m" >&2
    exit 2
    ;;
esac
lang="$1"
shift

if [ -x .venv/Scripts/python.exe ]; then
  py=.venv/Scripts/python.exe   # Windows (Git Bash)
elif [ -x .venv/bin/python ]; then
  py=.venv/bin/python
else
  echo "[error] Virtual environment (.venv) not found. Run ./go.sh $lang once first." >&2
  exit 1
fi

exec "$py" -m llm_radio_daemon.request --config "config/$lang/config.toml" "$@"
