"""ラジオドラマの執筆バッチ CLI（v6 §4.7.1）。

**放送プロセス（``python -m llm_radio_daemon.main``）とは別プロセスで動かすこと。**
同じ GPU・同じ Ollama 常駐モデルを放送中の生成と取り合わないよう、放送と執筆は
必ず別プロセスに分ける。手動実行と Windows タスクスケジューラのほか、放送本体の
``GeneratedDramaSupervisorThread``（``[[content]] type="generated_drama"`` の ``auto_write = true``）が
「放送が LLM を使っていない隙」にこの CLI を子プロセスとして起動する。いずれの
経路でも入口はこの1つ。

    # 新しい小説を1本立ち上げる（全体設計・章立て・登場人物・世界観）
    python -m llm_radio_daemon.generated_drama_writer new --title "夜間飛行" --premise "..."

    # 指定の小説を1バッチ（＝1シーン）進める
    python -m llm_radio_daemon.generated_drama_writer run --generated-drama-id 1

    # 定期実行用。進行中の小説を優先度順に1バッチ進める
    python -m llm_radio_daemon.generated_drama_writer run --auto

    # 進行状況を見る
    python -m llm_radio_daemon.generated_drama_writer list

放送プロセスを動かしたまま実行してよい（SQLite は WAL、本文ファイルは一時ファイル→
rename で書くため、読み手が半端な原稿を掴まない）。
"""

from __future__ import annotations

import argparse
import logging
import sys

from . import language, sensitive
from .config import CONFIG_ARG_HELP, Config, GeneratedDramaParams, load_config
from .db import TopicStore
from .logging_setup import setup_logging
from .generated_drama.data import GeneratedDramaData
from .generated_drama.writer import GeneratedDramaWriterService

logger = logging.getLogger(__name__)


def _generated_drama_params(config: Config, data_dir: str | None) -> GeneratedDramaParams:
    """[[content]] type = "generated_drama" の設定を使う。未設定なら既定値で動かす。"""
    content = config.content_by_type("generated_drama")
    params = (content.generated_drama if content is not None else None) or GeneratedDramaParams()
    if data_dir:
        params.data_dir = data_dir
    return params


def _cmd_new(args, service: GeneratedDramaWriterService) -> int:
    # --title 省略時はステージ0（企画立案）でタイトル・狙いを自動生成する。
    if not args.title:
        concept = service.build_concept(steer=args.premise)
        if concept is None:
            print("Failed to generate the concept", file=sys.stderr)
            return 1
        title, premise = concept
        print(f"Title: {title}")
        print(f"Premise: {premise}")
        if args.dry_run:
            print("\n--dry-run: nothing will be written to the DB.")
            return 0
    else:
        title, premise = args.title, args.premise
        if args.dry_run:
            print("--dry-run is only for when --title is omitted (automatic concept generation)", file=sys.stderr)
            return 2

    generated_drama_id = service.create(
        title,
        premise,
        chapters=args.chapters,
        scenes_per_chapter=args.scenes_per_chapter,
        priority=args.priority,
    )
    if generated_drama_id is None:
        return 1
    print(f"generated_drama_id = {generated_drama_id}")
    print(f"Next: python -m llm_radio_daemon.generated_drama_writer run --generated-drama-id {generated_drama_id}")
    return 0


def _cmd_run(args, service: GeneratedDramaWriterService) -> int:
    if not args.auto and args.generated_drama_id is None:
        print("Specify either --generated-drama-id or --auto", file=sys.stderr)
        return 2
    if args.auto:
        new_id = service.replenish()
        if new_id is not None:
            print(f"Automatically replenished a new concept: generated_drama_id = {new_id}")
    wrote = False
    for _ in range(max(1, args.scenes)):
        step = service.run_auto() if args.auto else service.run_one(args.generated_drama_id)
        wrote = wrote or step
        if not step:
            break
    return 0 if wrote else 1


def _cmd_rewrite(args, service: GeneratedDramaWriterService, store: TopicStore) -> int:
    check = not args.no_check

    if args.all:
        if args.generated_drama_id is not None or args.chapter is not None or args.scene is not None:
            print("--all cannot be combined with --generated-drama-id / --chapter / --scene", file=sys.stderr)
            return 2
        generated_drama_ids = [n["id"] for n in store.list_generated_dramas()]
        if not args.yes:
            print(f"This will rewrite all scenes of {len(generated_drama_ids)} novel(s). Re-run with --yes to proceed.")
            return 0
        total_ok = total_ng = 0
        for nid in generated_drama_ids:
            ok, ng = service.rewrite_generated_drama(nid, check=check)
            print(f"generated_drama_id={nid}: succeeded {ok} / failed (kept old text) {ng}")
            total_ok += ok
            total_ng += ng
        print(f"\nTotal: succeeded {total_ok} / failed (kept old text) {total_ng}")
        return 0 if total_ng == 0 else 1

    if args.generated_drama_id is None:
        print("Specify either --generated-drama-id or --all", file=sys.stderr)
        return 2

    if args.chapter is not None or args.scene is not None:
        if args.chapter is None or args.scene is None:
            print("--chapter and --scene must be specified together", file=sys.stderr)
            return 2
        ok = service.rewrite_scene(args.generated_drama_id, args.chapter, args.scene, check=check)
        print("Rewrite complete" if ok else "Rewrite failed (kept old text)")
        return 0 if ok else 1

    if not args.yes:
        n = len(store.generated_drama_scenes(args.generated_drama_id))
        print(f"This will rewrite all {n} scenes of generated_drama_id={args.generated_drama_id}. Re-run with --yes to proceed.")
        return 0
    ok, ng = service.rewrite_generated_drama(args.generated_drama_id, check=check)
    print(f"Rewrite complete: succeeded {ok} / failed (kept old text) {ng}")
    return 0 if ng == 0 else 1


def _cmd_list(args, service: GeneratedDramaWriterService, store: TopicStore, params: GeneratedDramaParams) -> int:
    dramas = store.list_generated_dramas()
    if not dramas:
        print("No novels yet. Use 'new' to start one.")
        return 0
    for n in dramas:
        scenes = store.generated_drama_scenes(n["id"])
        ready = sum(1 for s in scenes if s["ready"])
        plan = GeneratedDramaData(params.data_dir, n["id"]).load_novel()
        planned = sum(
            int(c.get("scene_count", params.scenes_per_chapter))
            for c in plan.get("chapters", [])
        )
        print(
            f"[{n['id']:3d}] {n['title']}  status={n['status']} priority={n['priority']}  "
            f"ready {ready}/{planned or '?'} scenes"
        )
    nxt = store.get_next_ready_scene()
    if nxt is None:
        print("\nNext scene ready for broadcast: none (the broadcast side falls back to filler / the next segment)")
    else:
        print(
            f"\nNext scene ready for broadcast: \"{nxt['generated_drama_title']}\" "
            f"Chapter {nxt['chapter']} Scene {nxt['scene']} (chunk_cursor={nxt['chunk_cursor']})"
        )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="python -m llm_radio_daemon.generated_drama_writer",
        description=(
            "ラジオドラマの執筆バッチ（放送プロセスとは別プロセスで動かすこと） / "
            "Radio drama writer batch job (run this as a separate process from the broadcast)"
        ),
    )
    ap.add_argument("--config", required=True, help=CONFIG_ARG_HELP)
    ap.add_argument(
        "--data-dir", default="",
        help=(
            "data/generated_drama_data_<lang>/ の場所（既定は [[content]] の data_dir） / "
            "path to data/generated_drama_data_<lang>/ (default: [[content]]'s data_dir)"
        ),
    )
    ap.add_argument(
        "--log-file",
        default="llm_radio_daemon.log",
        help=(
            "ログファイル名（[log].dir 内）。放送本体から起動されるときは別名が渡る / "
            "log file name (inside [log].dir); a different name is passed when launched by the broadcast itself"
        ),
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p_new = sub.add_parser(
        "new", help="新しい小説を1本立ち上げる / start a new novel"
    )
    p_new.add_argument(
        "--title", default="",
        help=(
            "省略するとトロープから企画（タイトル・狙い）を自動生成する / "
            "if omitted, a concept (title and premise) is generated automatically from the trope pools"
        ),
    )
    p_new.add_argument(
        "--premise", default="",
        help=(
            "題材・狙い（自由記述）。--title 省略時は企画生成への追加指定として渡る / "
            "free-form premise; when --title is omitted, passed as extra guidance to concept generation"
        ),
    )
    p_new.add_argument(
        "--dry-run", action="store_true",
        help=(
            "--title 省略時、企画を生成して表示するだけで DB に書かない / "
            "with --title omitted, only generate and print the concept without writing it to the DB"
        ),
    )
    p_new.add_argument("--chapters", type=int, default=None)
    p_new.add_argument("--scenes-per-chapter", type=int, default=None)
    p_new.add_argument(
        "--priority", type=int, default=0,
        help="run --auto が進める順番（大きいほど先） / order run --auto picks in (higher goes first)",
    )

    p_run = sub.add_parser(
        "run", help="1バッチ（＝1シーン）進める / advance one batch (= one scene)"
    )
    p_run.add_argument("--generated-drama-id", type=int, default=None)
    p_run.add_argument(
        "--auto", action="store_true",
        help="進行中の小説を優先度順に選ぶ / pick among in-progress novels by priority",
    )
    p_run.add_argument(
        "--scenes", type=int, default=1,
        help="続けて進めるシーン数（既定 1） / number of scenes to advance in a row (default 1)",
    )

    p_rewrite = sub.add_parser(
        "rewrite",
        help=(
            "既存シーンを新書式（台本形式）で書き直す / "
            "rewrite an existing scene in the new (script) format"
        ),
    )
    p_rewrite.add_argument(
        "--generated-drama-id", type=int, default=None, help="対象の小説 / the target novel"
    )
    p_rewrite.add_argument(
        "--chapter", type=int, default=None,
        help="対象の章（--scene とセットで1件だけ指定） / target chapter (pair with --scene to target exactly one)",
    )
    p_rewrite.add_argument(
        "--scene", type=int, default=None,
        help="対象のシーン（--chapter とセットで1件だけ指定） / target scene (pair with --chapter to target exactly one)",
    )
    p_rewrite.add_argument(
        "--all", action="store_true",
        help="全小説の全シーンを対象にする（--generated-drama-id と併用不可） / target every scene of every novel (mutually exclusive with --generated-drama-id)",
    )
    p_rewrite.add_argument(
        "--no-check", action="store_true",
        help="チェック段階を省略して即採用する（速いが品質未確認） / skip the check step and accept immediately (faster, unchecked quality)",
    )
    p_rewrite.add_argument(
        "--yes", action="store_true",
        help="複数シーンの書き直しを確認なしで実行する / rewrite multiple scenes without a confirmation prompt",
    )

    sub.add_parser("list", help="小説と進行状況の一覧 / list novels and their progress")

    args = ap.parse_args()

    config = load_config(args.config)
    setup_logging(config.log.dir, config.log.level, args.log_file)
    # 放送プロセス（main.py）と同じく、台本の言語をプロセス全体へ反映しておく。
    # 執筆は別プロセスなので main.py の set_language は効かず、ここで呼ばないと
    # config/en/config.toml を渡しても日本語で書かれてしまう。
    language.set_language(config.locale.lang)
    sensitive.set_language(config.locale.lang)
    params = _generated_drama_params(config, args.data_dir)

    store = TopicStore(config.db.path, config.db.rebroadcast_after_days)
    service = GeneratedDramaWriterService(store, config.llm, params, config.cast)
    try:
        if args.command == "new":
            return _cmd_new(args, service)
        if args.command == "run":
            return _cmd_run(args, service)
        if args.command == "rewrite":
            return _cmd_rewrite(args, service, store)
        return _cmd_list(args, service, store, params)
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
