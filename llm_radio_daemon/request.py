"""Lightweight CLI for sending requests to the broadcast process.

The broadcast process (``python -m llm_radio_daemon.main``) always re-picks
the next topic and the current corner from SQLite (``TopicStore``), so
writing directly to the DB from here reflects a request without touching the
broadcast process at all. This can be run whether the broadcast process is
running or not (if running, it's picked up at the next decision point; if
stopped, at the next startup).

Usage
-----

now <corner> [--for 30m]
    Switch to that corner right now (a temporary schedule override, see
    §6.2). Default is 30 minutes, after which it automatically returns to
    the timetable. The current talk segment finishes, there's a scene
    transition to the new corner, and the DJ acknowledges "got your
    request" once before starting the corner.

    <corner> is a [[content]] type or label, or the number shown by --list.

        python -m llm_radio_daemon.request now arxiv
        python -m llm_radio_daemon.request now arxiv --for 90m
        python -m llm_radio_daemon.request now 3 --for 2h

now --list
    Show the list of requestable corners (number, type, label, time slot)
    and the remaining time on the currently active request.

now --clear
    Cancel the active request and return to the timetable (without waiting
    for it to expire).

generated-drama seek <generated_drama_id>
    Restart the novel with the given generated_drama_id from the beginning.
    This just clears ``generated_drama_progress`` for every scene of that
    novel, so the broadcast side treats it as "not yet read" and restarts
    from the first scene. Takes effect once the scene currently playing
    finishes and the timetable enters the generated_drama slot.
    generated_drama_id can be checked with ``generated_drama_writer list``.

        python -m llm_radio_daemon.request generated-drama seek 1

rss reset
    Delete all topics with ``source = "rss"`` from the DB (for debugging).
    Since "already seen" is determined only by the topics table, the same
    articles get picked up again as "new" on the next crawl, letting you
    re-run the whole pipeline (dedup -> queue -> script generation ->
    broadcast). Embeddings are deleted along with them. Broadcast history
    (broadcasts) is kept.

        python -m llm_radio_daemon.request rss reset

reset all
    Clear all "listened/read" state and go back to a fresh start (for
    debugging). Deletes topics (all sources), broadcast history, reading/
    biography corner progress, radio drama reading progress, and request
    history. generated_dramas / generated_drama_scenes (the novel bodies
    written by the writer batch) are NOT affected -- only the broadcast
    side's read state is cleared. Prompts for confirmation (use --yes to
    skip when scripting).

        python -m llm_radio_daemon.request reset all
        python -m llm_radio_daemon.request reset all --yes

reset data
    Relocate the whole data/ directory to a timestamped backup and recreate
    an empty data/ (for debugging; last resort). Broader than ``reset
    all``: wipes both the ja and en DB files entirely (including novel
    bodies and all broadcast history), the Aozora Bunko/Gutenberg/Wikipedia
    caches, and the radio drama novel.json / characters.json. Does not
    touch config.toml-family config files or logs/. This only moves things
    to a backup dir (data.bak_YYYYMMDD_HHMMSS/), so you can undo it by hand
    by renaming it back. **If the broadcast process is still running, it
    will be holding the DB files open and this will fail -- stop it first.**

        python -m llm_radio_daemon.request reset data
        python -m llm_radio_daemon.request reset data --yes
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import time
from pathlib import Path

from .config import CONFIG_ARG_HELP, Config, ContentConfig, load_config
from .db import TopicStore

# Default request length. Wanting "this, right now" is usually satisfied by
# one corner's worth, and anything longer risks leaving the timetable
# clobbered and forgotten. Use --for to ask for longer explicitly.
DEFAULT_REQUEST_SEC = 30 * 60

# Upper bound. If you need more than this, edit the [[content]] schedule
# instead.
MAX_REQUEST_SEC = 24 * 60 * 60

_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([hms]?)", re.IGNORECASE)
_UNIT_SEC = {"h": 3600.0, "m": 60.0, "s": 1.0, "": 60.0}  # no unit means minutes


class RequestError(RuntimeError):
    """A usage mistake. main() prints just the message and returns exit code 1."""


def parse_duration(spec: str) -> float:
    """"30m" / "90" / "2h" / "1h30m" -> seconds. No unit is read as minutes."""
    text = (spec or "").strip()
    if not text:
        raise RequestError("--for is empty (example: 30m / 90 / 2h / 1h30m)")
    total = 0.0
    pos = 0
    for m in _DURATION_RE.finditer(text):
        if m.start() != pos:
            break
        total += float(m.group(1)) * _UNIT_SEC[m.group(2).lower()]
        pos = m.end()
    if pos != len(text) or total <= 0:
        raise RequestError(
            f"can't read --for {spec!r} as a duration (example: 30m / 90 / 2h / 1h30m)"
        )
    if total > MAX_REQUEST_SEC:
        raise RequestError(
            f"--for is too long (max {MAX_REQUEST_SEC // 3600} hours). "
            "If you want that corner to run longer than that, edit the schedule "
            "in config_content.toml instead"
        )
    return total


def _describe(index: int, c: ContentConfig) -> str:
    when = ", ".join(c.schedule) if c.schedule else "always (fallback)"
    label = f" [{c.label}]" if c.label else ""
    off = "" if c.enabled else "  * enabled = false"
    return f"  {index:>2}  {c.type}{label}  —  {when}{off}"


def resolve_content(spec: str, content: list[ContentConfig]) -> tuple[int, ContentConfig]:
    """Resolve a corner spec (number / type / label) to (index, entry).

    If multiple entries share the same type (e.g. rss split by time slot),
    the first one is used and the other candidates are listed with their
    numbers. Specifying by number removes the ambiguity.
    """
    text = (spec or "").strip()
    if not text:
        raise RequestError("specify a corner (see --list for the list)")

    if text.isdigit():
        idx = int(text)
        if not (0 <= idx < len(content)):
            raise RequestError(
                f"number {idx} is out of range (0-{len(content) - 1}; see --list for the list)"
            )
        matches = [idx]
    else:
        key = text.casefold()
        matches = [
            i
            for i, c in enumerate(content)
            if c.type.casefold() == key or (c.label and c.label.casefold() == key)
        ]
        if not matches:
            raise RequestError(
                f"no corner matches {text!r} (see --list for the list)"
            )

    idx = matches[0]
    c = content[idx]
    if not c.enabled:
        raise RequestError(
            f"{c.type} (#{idx}) has enabled = false. Enable it in "
            "config_content.toml and restart the broadcast process first "
            "(a disabled corner never spins up its topic-gathering thread, "
            "so switching to it would only play filler)"
        )
    if len(matches) > 1:
        print(
            f"note: {len(matches)} entries have type = {c.type!r}. "
            f"Requested the first one, #{idx}. To pick another, use its number:",
            file=sys.stderr,
        )
        for i in matches:
            print(_describe(i, content[i]), file=sys.stderr)
    return idx, c


def _print_active(store: TopicStore) -> None:
    req = store.active_request()
    if req is None:
        print("active request: none (following the timetable)")
        return
    remain_min = max(0.0, req["expires_at"] - time.time()) / 60.0
    print(
        f"active request: {req['label']}"
        f" (#{req['content_index']} / {remain_min:.0f} min left)"
    )


def _cmd_now_list(store: TopicStore, config: Config) -> int:
    print("requestable corners (number / type / time slot):")
    for i, c in enumerate(config.content):
        print(_describe(i, c))
    print()
    _print_active(store)
    return 0


def _cmd_now_clear(store: TopicStore) -> int:
    n = store.clear_requests()
    if n == 0:
        print("no active request (already following the timetable)")
    else:
        print("cleared the request. Back to the timetable")
    return 0


def _cmd_now(spec: str, for_spec: str, store: TopicStore, config: Config) -> int:
    ttl = parse_duration(for_spec)
    idx, content = resolve_content(spec, config.content)
    req = store.put_request(idx, content.type, content.display_label, ttl)
    until = time.strftime("%H:%M", time.localtime(req["expires_at"]))
    print(
        f"request accepted: {content.display_label}"
        f" (#{idx} {content.type})"
    )
    print(f"  until {until} ({ttl / 60:.0f} min). Returns to the timetable after that")
    print("  to cancel: request ... now --clear")
    return 0


def _cmd_generated_drama_seek(generated_drama_id: int, store: TopicStore) -> int:
    generated_drama = store.get_generated_drama(generated_drama_id)
    if generated_drama is None:
        print(
            f"generated_drama_id={generated_drama_id} not found "
            "(check with generated_drama_writer list)",
            file=sys.stderr,
        )
        return 1
    n = store.reset_generated_drama_progress_for_drama(generated_drama_id)
    print(
        f"\"{generated_drama['title']}\" (generated_drama_id={generated_drama_id}) "
        f"will restart from the beginning ({n} scene(s) of progress reset)"
    )
    return 0


def _cmd_rss_reset(store: TopicStore) -> int:
    n = store.delete_topics_by_source("rss")
    print(
        f"deleted {n} rss topic(s). "
        "They'll be picked up again as new on the next crawl."
    )
    return 0


def _cmd_reset_all(store: TopicStore, assume_yes: bool) -> int:
    if not assume_yes:
        answer = input(
            "This will reset the whole DB "
            "(deletes all topics, broadcast history, reading/biography "
            "progress, novel reading progress, and request history; novel "
            "bodies themselves are kept). Are you sure? [y/N]: "
        )
        if answer.strip().lower() not in ("y", "yes"):
            print("aborted. Nothing was deleted")
            return 1
    counts = store.reset_all_content()
    total = sum(counts.values())
    print(f"DB reset complete ({total} row(s) deleted):")
    for table, n in counts.items():
        print(f"  {table}: {n}")
    print("Starting from the next crawl/startup, everything is treated as unread/unbroadcast")
    return 0


def _cmd_reset_data(assume_yes: bool) -> int:
    """Relocate and recreate the whole data/ directory (for debugging; last resort).

    Unlike ``reset all``, this targets the DB files themselves and all data
    for both ja and en. Runs without ever opening TopicStore (keeping it
    open would make this script itself the cause of the lock).
    """
    data_dir = Path("data")
    if not assume_yes:
        answer = input(
            "This will reset the whole data/ directory "
            "(deletes both the ja and en DB files, broadcast history, radio "
            "drama bodies, and the Aozora Bunko/Gutenberg/Wikipedia caches; "
            "config.toml-family config and logs/ are kept). Have you already "
            "stopped the broadcast process? [y/N]: "
        )
        if answer.strip().lower() not in ("y", "yes"):
            print("aborted. Nothing was deleted")
            return 1
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        print("data/ didn't exist, so it was created empty")
        return 0
    backup = Path(f"data.bak_{time.strftime('%Y%m%d_%H%M%S')}")
    try:
        data_dir.rename(backup)
    except OSError as e:
        print(f"error: couldn't relocate data/ ({e})", file=sys.stderr)
        print(
            "The broadcast process may still have the DB or cache files open. "
            "Stop it with Ctrl+C and try again.",
            file=sys.stderr,
        )
        return 1
    data_dir.mkdir(parents=True)
    print(f"Relocated data/ to {backup}/ and recreated an empty data/.")
    print("On the next startup, each corner will automatically recreate its DB/cache/radio drama data.")
    print(f"To undo this, delete data/ and rename {backup} back to data.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="python -m llm_radio_daemon.request",
        description=(
            "Lightweight requests to the broadcast process "
            "(works whether or not it's currently running)"
        ),
    )
    ap.add_argument("--config", required=True, help=CONFIG_ARG_HELP)
    sub = ap.add_subparsers(dest="command", required=True)

    p_now = sub.add_parser(
        "now",
        help="switch to the given corner right now (a temporary schedule override)",
    )
    p_now.add_argument(
        "content", nargs="?",
        help="a [[content]] type or label, or the number shown by --list",
    )
    p_now.add_argument(
        "--for", dest="duration", default="30m",
        help="how long to stay (default 30m). 30m / 90 / 2h / 1h30m; no unit means minutes",
    )
    p_now.add_argument(
        "--list", action="store_true",
        help="list requestable corners and the currently active request",
    )
    p_now.add_argument(
        "--clear", action="store_true",
        help="cancel the active request and return to the schedule",
    )

    p_drama = sub.add_parser(
        "generated-drama",
        help="operations on the radio drama reading corner",
    )
    drama_sub = p_drama.add_subparsers(dest="generated_drama_command", required=True)
    p_seek = drama_sub.add_parser(
        "seek",
        help="restart the given novel from the beginning",
    )
    p_seek.add_argument(
        "generated_drama_id", type=int,
        help="the generated_drama_id shown by generated_drama_writer list",
    )

    p_rss = sub.add_parser(
        "rss", help="operations on the RSS source (for debugging)"
    )
    rss_sub = p_rss.add_subparsers(dest="rss_command", required=True)
    rss_sub.add_parser(
        "reset",
        help="delete all RSS topics so they get re-fetched",
    )

    p_reset = sub.add_parser(
        "reset", help="reset the whole DB (for debugging)"
    )
    reset_sub = p_reset.add_subparsers(dest="reset_command", required=True)
    p_reset_all = reset_sub.add_parser(
        "all",
        help="delete all topics, broadcast history and per-corner progress, resetting everything to unread",
    )
    p_reset_all.add_argument(
        "--yes", "-y", action="store_true",
        help="skip the confirmation prompt",
    )
    p_reset_data = reset_sub.add_parser(
        "data",
        help=(
            "relocate and recreate the whole data/ directory "
            "(wipes DB files, novel bodies and caches too; last resort)"
        ),
    )
    p_reset_data.add_argument(
        "--yes", "-y", action="store_true",
        help="skip the confirmation prompt",
    )

    args = ap.parse_args()

    # Wiping data targets the DB files themselves, so to avoid opening a
    # TopicStore and becoming the cause of the lock ourselves, this branch
    # only loads the config (--config is still required, same as every
    # other command; the whole data/ dir is targeted regardless of ja/en).
    if args.command == "reset" and args.reset_command == "data":
        load_config(args.config)
        return _cmd_reset_data(args.yes)

    config = load_config(args.config)
    store = TopicStore(config.db.path, config.db.rebroadcast_after_days)
    try:
        if args.command == "now":
            if args.list:
                return _cmd_now_list(store, config)
            if args.clear:
                return _cmd_now_clear(store)
            if not args.content:
                p_now.error("specify a corner (see --list for the list)")
            return _cmd_now(args.content, args.duration, store, config)
        if args.command == "generated-drama" and args.generated_drama_command == "seek":
            return _cmd_generated_drama_seek(args.generated_drama_id, store)
        if args.command == "rss" and args.rss_command == "reset":
            return _cmd_rss_reset(store)
        if args.command == "reset" and args.reset_command == "all":
            return _cmd_reset_all(store, args.yes)
        ap.print_help()
        return 2
    except RequestError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except sqlite3.OperationalError as e:
        # Effectively always "database is locked". The broadcast process is
        # holding the write lock and hasn't released it -- almost always
        # because that broadcast process is stale (builds older than db.py's
        # move to auto-commit could leave a transaction open after a failed
        # INSERT on a duplicate topic). A traceback wouldn't help here, so
        # just say what to do about it.
        print(f"error: can't write to the DB ({e})", file=sys.stderr)
        print(
            "The broadcast process may still be holding the DB lock. "
            "Stop it with Ctrl+C and restart it via go.bat to fix this.",
            file=sys.stderr,
        )
        return 1
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
