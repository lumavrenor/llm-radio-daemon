"""GeneratedDramaWriterService — オフラインの執筆バッチ（v6 §4.7.1）。

**放送プロセスとは完全に別プロセス**で動かす。放送スレッド群と同じ Ollama 常駐モデル・
同じ GPU を取り合わないよう、CLI（``llm_radio_daemon.generated_drama_writer``）から手動 /
Windows タスクスケジューラで叩く。放送プロセス側からは決してここを呼ばない。

ハーネスは4段階:

    1. 全体設計   … プロット・章立て・characters.json / world.json（``create()``）
    2. 中間展開   … ハコ書き／ビートシート（章ごと・``_ensure_beats()``）
    3. 本文生成   … シーン単位（``_write_scene()``）
    4. チェック   … 部分要約→章要約→全体要約の積み上げ照合（``_check_scene()``）

**1回の起動＝1シーン生成が基本単位**（章ごとに一気に進めると、チェックで NG が出た
ときのロールバック範囲が大きくなりすぎるため）。生成物は SQLite と ``data/generated_drama_data_<lang>/`` へ
書き溜め、放送プロセスは読むだけ。
"""


from __future__ import annotations

import datetime
import logging

from .. import language
from ..config import CastMember, LLMConfig, GeneratedDramaParams
from ..db import TopicStore
from ..script.ollama_client import sanitize_text
from . import GeneratedDramaCharacter
from .concept import generate_concept
from .data import GeneratedDramaData
from .llm import chat_json

logger = logging.getLogger(__name__)

_MIN_SANE_BODY = 200  # これ未満の本文は生成失敗とみなす

# 同じシーンを何回書き直してもチェックが通らないときに諦める回数。
# 校閲側が細かい指摘を出し続けると同じシーンで永久に足踏みし、放送する原稿が
# 1本も増えないため、この回数を超えたら最後の原稿をそのまま採用する
# （「放送が止まることだけが異常」— §4.7.4 と同じ考え方）。
_MAX_SCENE_ATTEMPTS = 3

# 校閲の指摘は、書き直しプロンプトとログの両方でここまでに絞る。
# 全部渡すと文脈が指摘で埋まり、肝心の設定・段取りが薄まる。
_MAX_ISSUES = 5
_ISSUE_CHARS = 200

# 既存シーンの書き直し（rewrite）専用の「指摘」。通常の校閲指摘と同じ経路
# （issues/previous）に乗せることで、_scene_prompt() 側の変更なしに
# 「内容は保ったまま書式だけ直させる」を実現する。
_REWRITE_FORMAT_ISSUE_JA = (
    "この原稿は旧い自然文体で書かれている。出来事・セリフの内容・展開・結末は"
    "変えないこと。書式だけを台本形式（「名前：「セリフ」」＋最小限のト書き）に"
    "書き直すこと。新しい出来事を足したり、あった出来事を削ったりしないこと。"
)

_REWRITE_FORMAT_ISSUE_EN = (
    "This draft is written as old-style continuous prose. Do not change what happens, "
    "what is said, how it develops or how it ends. Rewrite the format only, into script "
    'form (Name: "line of dialogue", plus the bare minimum of stage directions). '
    "Do not add events that were not there and do not drop events that were."
)


def _rewrite_format_issue() -> str:
    return language.pick(ja=_REWRITE_FORMAT_ISSUE_JA, en=_REWRITE_FORMAT_ISSUE_EN)


def _trim_issues(issues: list[str]) -> list[str]:
    return [i.strip()[:_ISSUE_CHARS] for i in issues[:_MAX_ISSUES]]


# --- JSON スキーマ ---------------------------------------------------------

_DESIGN_SCHEMA = {
    "type": "object",
    "properties": {
        "logline": {"type": "string"},
        "theme": {"type": "string"},
        "world": {
            "type": "object",
            "properties": {
                "setting": {"type": "string"},
                "era": {"type": "string"},
                "tone": {"type": "string"},
                "rules": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["setting", "tone"],
        },
        "characters": {
            "type": "array",
            "minItems": 2,
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "name": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "role": {"type": "string"},
                    "persona": {"type": "string"},
                    "speech_style": {"type": "string"},
                },
                "required": ["key", "name", "role", "persona", "speech_style"],
            },
        },
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chapter": {"type": "integer"},
                    "title": {"type": "string"},
                    "synopsis": {"type": "string"},
                },
                "required": ["chapter", "title", "synopsis"],
            },
        },
    },
    "required": ["logline", "world", "characters", "chapters"],
}


def _beats_schema(scene_count: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "scenes": {
                "type": "array",
                "minItems": scene_count,
                "maxItems": scene_count,
                "items": {
                    "type": "object",
                    "properties": {
                        "scene": {"type": "integer"},
                        "purpose": {"type": "string"},
                        "setting": {"type": "string"},
                        "characters": {"type": "array", "items": {"type": "string"}},
                        "beats": {"type": "array", "items": {"type": "string"}},
                        "ends_with": {"type": "string"},
                    },
                    "required": ["scene", "purpose", "setting", "characters", "beats"],
                },
            }
        },
        "required": ["scenes"],
    }


_SCENE_SCHEMA = {
    "type": "object",
    "properties": {"body": {"type": "string"}, "summary": {"type": "string"}},
    "required": ["body", "summary"],
}

_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "ok": {"type": "boolean"},
        "issues": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["ok", "issues", "summary"],
}

_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}


# --- 本文の後始末 ----------------------------------------------------------

def _clean_body(raw: str) -> str:
    """TTS に流せる形へ整える。段落は1行1段落に正規化する。"""
    text = sanitize_text(
        raw.replace("\r\n", "\n").replace("\r", "\n"), keep_stage_directions=True
    )
    paragraphs = [p.strip() for p in text.split("\n")]
    return "\n".join(p for p in paragraphs if p)


class GeneratedDramaWriterService:
    def __init__(
        self,
        store: TopicStore,
        llm_config: LLMConfig,
        params: GeneratedDramaParams,
        cast: list[CastMember],
    ):
        self._store = store
        self._llm = llm_config
        self._p = params
        self._cast = cast
        self._timeout = params.writer_timeout_sec

    # --- 1. 全体設計 -----------------------------------------------------

    def create(
        self,
        title: str,
        premise: str,
        *,
        chapters: int | None = None,
        scenes_per_chapter: int | None = None,
        priority: int = 0,
    ) -> int | None:
        """新しいラジオドラマを1本立ち上げる。generated_drama_id を返す（失敗なら None）。"""
        n_chapters = chapters or self._p.chapters
        n_scenes = scenes_per_chapter or self._p.scenes_per_chapter

        design = chat_json(
            self._llm,
            self._design_prompt(title, premise, n_chapters, n_scenes),
            _DESIGN_SCHEMA,
            timeout_sec=self._timeout,
        )
        if design is None:
            logger.error("generated_drama_writer: failed to generate the overall design")
            return None

        characters = self._assign_voices(
            [GeneratedDramaCharacter.from_dict(c) for c in design.get("characters", [])]
        )
        if not characters:
            logger.error("generated_drama_writer: not a single character was generated")
            return None

        chapters_raw = design.get("chapters", [])
        plan = [
            {
                "chapter": int(c.get("chapter", i + 1)),
                "title": str(
                    c.get("title")
                    or language.pick(ja=f"第{i + 1}章", en=f"Chapter {i + 1}")
                ),
                "synopsis": str(c.get("synopsis", "")),
                "scene_count": n_scenes,
            }
            for i, c in enumerate(chapters_raw)
        ][:n_chapters]
        if not plan:
            logger.error("generated_drama_writer: the chapter outline was empty")
            return None

        generated_drama_id = self._store.create_generated_drama(title, premise, priority)
        data = GeneratedDramaData(self._p.data_dir, generated_drama_id)
        data.save_novel(
            {
                "novel_id": generated_drama_id,
                "title": title,
                "premise": premise,
                "logline": design.get("logline", ""),
                "theme": design.get("theme", ""),
                "chapters": plan,
                "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            }
        )
        data.save_world(design.get("world", {}))
        data.save_characters(characters)
        data.save_summaries({"chapters": {}, "overall": ""})

        logger.info(
            "generated_drama_writer: launched \"%s\" (generated_drama_id=%d). %d chapters x %d scenes / %d characters",
            title, generated_drama_id, len(plan), n_scenes, len(characters),
        )
        return generated_drama_id

    # --- 0. 企画立案（§4.7.1 ステージ0） -------------------------------

    def build_concept(self, *, steer: str = "") -> tuple[str, str] | None:
        """トロープから企画を1本立て ``(title, premise)`` を返す（DB には書かない）。"""
        existing = self._store.list_generated_dramas()
        return generate_concept(self._llm, self._p, existing, steer=steer)

    def create_auto(self, *, priority: int = 0, steer: str = "") -> int | None:
        """企画を1本立て、そのまま ``create()`` へ渡す。generated_drama_id を返す。"""
        concept = self.build_concept(steer=steer)
        if concept is None:
            return None
        title, premise = concept
        return self.create(title, premise, priority=priority)

    def replenish(self, *, priority: int = 0) -> int | None:
        """``run --auto`` 用。執筆中のラジオドラマが閾値未満なら企画を1本補充する。"""
        if not self._p.auto_concept:
            return None
        writing = self._store.list_generated_dramas(status="writing")
        if len(writing) >= self._p.concept_min_writing:
            return None
        logger.info(
            "generated_drama_writer: %d dramas in progress (threshold %d). Starting a new concept",
            len(writing), self._p.concept_min_writing,
        )
        return self.create_auto(priority=priority)

    def _assign_voices(self, characters: list[GeneratedDramaCharacter]) -> list[GeneratedDramaCharacter]:
        """各キャラへ [[cast]] の出演者を1人ずつ割り当てる（§4.7.1）。

        放送側はこの ``cast_id`` / ``voicevox_speaker`` をそのまま話者振り分けに使う。
        ナレーター役は地の文で使うので、キャラ用の割り当てからは後回しにする。
        """
        pool = [m for m in self._cast if m.id != (self._p.narrator_cast_id or "")]
        pool = pool or list(self._cast)
        out: list[GeneratedDramaCharacter] = []
        for i, c in enumerate(characters):
            if not c.name:
                continue
            if not c.key:
                c.key = f"c{i + 1}"
            if pool:
                m = pool[i % len(pool)]
                c.cast_id = m.id
                c.voicevox_speaker = m.voicevox_speaker
                c.voicevox_style = m.default_style_name
            out.append(c)
        return out

    # --- 2〜4. 1バッチ ＝ 1シーン ----------------------------------------

    def run_auto(self) -> bool:
        """進行中のラジオドラマを優先度順に1本選び、1バッチ進める。"""
        for drama in self._store.list_generated_dramas(status="writing"):
            if self.run_one(drama["id"]):
                return True
        logger.info("generated_drama_writer: no radio drama to advance (all finished or paused)")
        return False

    def run_one(self, generated_drama_id: int) -> bool:
        """指定のラジオドラマを1シーンぶん進める。何か書けたら True。"""
        drama = self._store.get_generated_drama(generated_drama_id)
        if drama is None:
            logger.error("generated_drama_writer: generated_drama_id=%d not found", generated_drama_id)
            return False

        data = GeneratedDramaData(self._p.data_dir, generated_drama_id)
        plan = data.load_novel()
        if not plan.get("chapters"):
            logger.error(
                "generated_drama_writer: novel.json for generated_drama_id=%d is missing or broken. "
                "Start over with new", generated_drama_id,
            )
            return False

        target = self._next_target(generated_drama_id, plan)
        if target is None:
            self._store.set_generated_drama_status(generated_drama_id, "finished")
            logger.info("generated_drama_writer: \"%s\" has finished writing all scenes (finished)", drama["title"])
            return False
        chapter, scene = target

        beats = self._ensure_beats(data, plan, chapter)
        if beats is None:
            return False
        beat = next(
            (s for s in beats.get("scenes", []) if int(s.get("scene", 0)) == scene), None
        )
        if beat is None:
            logger.error(
                "generated_drama_writer: chapter %d's beat sheet has no scene %d. "
                "Delete beats/ch%02d.json and regenerate it", chapter, scene, chapter,
            )
            return False

        logger.info(
            "generated_drama_writer: writing \"%s\" chapter %d scene %d...", drama["title"], chapter, scene
        )
        result = self._produce_and_check(data, plan, beat, generated_drama_id, chapter, scene)
        if result is None:
            return False
        body, summary, ok, issues = result

        if not ok:
            prev = self._store.get_generated_drama_scene(generated_drama_id, chapter, scene)
            if (prev["attempts"] if prev else 0) + 1 >= _MAX_SCENE_ATTEMPTS:
                logger.warning(
                    "generated_drama_writer: chapter %d scene %d still fails the check after %d rewrites. "
                    "Adopting the last draft and moving on (issues: %s)",
                    chapter, scene, _MAX_SCENE_ATTEMPTS, " / ".join(issues) or "reason unknown",
                )
                ok = True

        scene_id = self._store.save_generated_drama_scene(
            generated_drama_id, chapter, scene, body, summary, ready=ok
        )
        data.save_scene_text(chapter, scene, body)
        if ok:
            logger.info(
                "generated_drama_writer: chapter %d scene %d complete (scene_id=%d, %d chars) ready=1",
                chapter, scene, scene_id, len(body),
            )
        else:
            logger.error(
                "generated_drama_writer: chapter %d scene %d failed the check (ready=0, will not air): %s",
                chapter, scene, " / ".join(issues) or "reason unknown",
            )

        self._roll_up_summaries(data, plan, generated_drama_id, chapter)
        return True

    def _produce_and_check(
        self,
        data: GeneratedDramaData,
        plan: dict,
        beat: dict,
        generated_drama_id: int,
        chapter: int,
        scene: int,
        *,
        previous: str = "",
        extra_issue: str = "",
        full_previous: bool = False,
    ) -> tuple[str, str, bool, list[str]] | None:
        """本文生成→チェック→（NGなら1回だけ）書き直し、の共通処理。

        新規執筆（``run_one``）と既存シーンの書き直し（``rewrite_scene``）の
        両方から使う。
        """
        seed_issues = [extra_issue] if extra_issue else []
        written = self._write_scene(
            data, plan, beat, generated_drama_id, chapter, scene,
            issues=seed_issues, previous=previous, full_previous=full_previous,
        )
        if written is None:
            return None
        body, summary = written

        ok, issues, summary = self._check_scene(data, plan, beat, body, summary, chapter)
        if not ok:
            logger.warning(
                "generated_drama_writer: the check raised issues (%s). Rewriting", " / ".join(issues) or "reason unknown"
            )
            revised = self._write_scene(
                data, plan, beat, generated_drama_id, chapter, scene, issues=issues, previous=body
            )
            if revised is not None:
                body, summary = revised
                ok, issues, summary = self._check_scene(
                    data, plan, beat, body, summary, chapter
                )
        return body, summary, ok, issues

    # --- 対象シーンの決定 -------------------------------------------------

    def _next_target(self, generated_drama_id: int, plan: dict) -> tuple[int, int] | None:
        """次に書くべき (chapter, scene)。未着手か ready=0 の最小のものを返す。"""
        done = {
            (s["chapter"], s["scene"]): s["ready"]
            for s in self._store.generated_drama_scenes(generated_drama_id)
        }
        for ch in plan.get("chapters", []):
            chapter = int(ch.get("chapter", 0))
            count = int(ch.get("scene_count", self._p.scenes_per_chapter))
            for scene in range(1, count + 1):
                if not done.get((chapter, scene), False):
                    return chapter, scene
        return None

    # --- 2. 中間展開（ハコ書き） -----------------------------------------

    def _ensure_beats(self, data: GeneratedDramaData, plan: dict, chapter: int) -> dict | None:
        beats = data.load_beats(chapter)
        if beats.get("scenes"):
            return beats

        chapter_plan = next(
            (c for c in plan.get("chapters", []) if int(c.get("chapter", 0)) == chapter), {}
        )
        count = int(chapter_plan.get("scene_count", self._p.scenes_per_chapter))
        logger.info("generated_drama_writer: building the outline for chapter %d (%d scenes)...", chapter, count)
        beats = chat_json(
            self._llm,
            self._beats_prompt(data, plan, chapter_plan, count),
            _beats_schema(count),
            timeout_sec=self._timeout,
        )
        if beats is None or not beats.get("scenes"):
            logger.error("generated_drama_writer: failed to generate the outline for chapter %d", chapter)
            return None
        beats["chapter"] = chapter
        data.save_beats(chapter, beats)
        return beats

    # --- 3. 本文生成 ------------------------------------------------------

    def _write_scene(
        self,
        data: GeneratedDramaData,
        plan: dict,
        beat: dict,
        generated_drama_id: int,
        chapter: int,
        scene: int,
        *,
        issues: list[str] | None = None,
        previous: str = "",
        full_previous: bool = False,
    ) -> tuple[str, str] | None:
        result = chat_json(
            self._llm,
            self._scene_prompt(
                data, plan, beat, generated_drama_id, chapter, scene, issues or [], previous, full_previous
            ),
            _SCENE_SCHEMA,
            timeout_sec=self._timeout,
            # 目標文字数の2倍を上限に。日本語1文字がおよそ1トークンなので目標＋summary は
            # これで収まる。3倍だと小型モデルが同じセリフ・効果音のループで枠を使い切り、
            # JSON を閉じられず没になる（bench iter03）。ここで早めに止める。
            max_tokens=max(2048, int(self._p.scene_target_chars * 2)),
        )
        if result is None:
            logger.error("generated_drama_writer: failed to generate the text of chapter %d scene %d", chapter, scene)
            return None
        body = _clean_body(str(result.get("body", "")))
        if len(body) < _MIN_SANE_BODY:
            logger.error(
                "generated_drama_writer: the generated text is too short (%d chars). Not writing it this time", len(body)
            )
            return None
        return body, str(result.get("summary", "")).strip()

    # --- 4. チェック ------------------------------------------------------

    def _check_scene(
        self,
        data: GeneratedDramaData,
        plan: dict,
        beat: dict,
        body: str,
        summary: str,
        chapter: int,
    ) -> tuple[bool, list[str], str]:
        result = chat_json(
            self._llm,
            self._check_prompt(data, plan, beat, body, chapter),
            _CHECK_SCHEMA,
            timeout_sec=self._timeout,
            temperature=0.2,
            max_tokens=2048,
        )
        if result is None:
            # チェックできなかっただけで本文が壊れているとは限らない。ready=0 で保留する。
            return (
                False,
                [
                    language.pick(
                        ja="チェックフェーズが応答しなかった",
                        en="the check phase did not respond",
                    )
                ],
                summary,
            )
        ok = bool(result.get("ok"))
        issues = _trim_issues([str(i) for i in result.get("issues", []) if str(i).strip()])
        new_summary = str(result.get("summary", "")).strip() or summary
        return ok, issues, new_summary

    def _roll_up_summaries(
        self, data: GeneratedDramaData, plan: dict, generated_drama_id: int, chapter: int
    ) -> None:
        """章のシーンが揃ったら、章要約→全体要約を積み上げ直す（§4.7.1 のチェック段）。"""
        chapter_plan = next(
            (c for c in plan.get("chapters", []) if int(c.get("chapter", 0)) == chapter), {}
        )
        count = int(chapter_plan.get("scene_count", self._p.scenes_per_chapter))
        scenes = [s for s in self._store.generated_drama_scenes(generated_drama_id) if s["chapter"] == chapter]
        if len([s for s in scenes if s["ready"]]) < count:
            return  # まだ章の途中

        summaries = data.load_summaries()
        joined = "\n".join(f"- {s['summary']}" for s in scenes if s["summary"])
        result = chat_json(
            self._llm,
            language.pick(
                ja=f"""次はラジオドラマ「{plan.get('title', '')}」第{chapter}章の各シーンの要約です。
この章で何が起きたのかを、後続の章を書くときの参照用に 300 字以内でまとめてください。
伏線・人物関係の変化・未解決のまま残ったことを落とさないこと。

{joined}
""",
                en=f"""Below are the summaries of each scene in chapter {chapter} of
"{plan.get('title', '')}". Write what happened in this chapter in eighty words or fewer,
as a reference for writing the chapters that follow. Do not drop anything that was set up
for later, any change in how the characters stand with each other, or anything left
unresolved.

{joined}
""",
            ),
            _SUMMARY_SCHEMA,
            timeout_sec=self._timeout,
            temperature=0.3,
        )
        if result is None:
            return
        summaries["chapters"][str(chapter)] = str(result.get("summary", "")).strip()
        overall = chat_json(
            self._llm,
            language.pick(
                ja="次はラジオドラマの各章の要約です。全体で何が起きているかを 400 字以内でまとめてください。\n\n"
                + "\n".join(
                    f"第{k}章: {v}"
                    for k, v in sorted(summaries["chapters"].items(), key=lambda x: int(x[0]))
                ),
                en="Below are the chapter summaries of a serial. Write what is happening in the "
                "story as a whole in a hundred words or fewer.\n\n"
                + "\n".join(
                    f"Chapter {k}: {v}"
                    for k, v in sorted(summaries["chapters"].items(), key=lambda x: int(x[0]))
                ),
            ),
            _SUMMARY_SCHEMA,
            timeout_sec=self._timeout,
            temperature=0.3,
        )
        if overall is not None:
            summaries["overall"] = str(overall.get("summary", "")).strip()
        data.save_summaries(summaries)
        logger.info("generated_drama_writer: updated the summary for chapter %d", chapter)

    # --- 既存シーンの書き直し（台本形式への移行など） --------------------

    def rewrite_scene(self, generated_drama_id: int, chapter: int, scene: int, *, check: bool = True) -> bool:
        """既存の1シーンを新書式で書き直す（DBとテキストファイルの両方を上書き）。

        実行中は対象のラジオドラマを一時的に status='paused' にし、放送プロセスが
        書き直し中の本文を掴まないようにする（終了後は元の status に戻す）。
        """
        drama = self._store.get_generated_drama(generated_drama_id)
        if drama is None:
            logger.error("generated_drama_writer: generated_drama_id=%d not found", generated_drama_id)
            return False
        original_status = drama["status"]
        self._store.set_generated_drama_status(generated_drama_id, "paused")
        try:
            return self._rewrite_scene_inner(generated_drama_id, chapter, scene, check=check)
        finally:
            self._store.set_generated_drama_status(generated_drama_id, original_status)

    def rewrite_generated_drama(self, generated_drama_id: int, *, check: bool = True) -> tuple[int, int]:
        """ラジオドラマまるごとの既存シーンを新書式で書き直す。``(成功数, 失敗数)`` を返す。"""
        drama = self._store.get_generated_drama(generated_drama_id)
        if drama is None:
            logger.error("generated_drama_writer: generated_drama_id=%d not found", generated_drama_id)
            return 0, 0
        original_status = drama["status"]
        self._store.set_generated_drama_status(generated_drama_id, "paused")
        try:
            ok_count = ng_count = 0
            for s in self._store.generated_drama_scenes(generated_drama_id):
                if self._rewrite_scene_inner(generated_drama_id, s["chapter"], s["scene"], check=check):
                    ok_count += 1
                else:
                    ng_count += 1
            return ok_count, ng_count
        finally:
            self._store.set_generated_drama_status(generated_drama_id, original_status)

    def _rewrite_scene_inner(self, generated_drama_id: int, chapter: int, scene: int, *, check: bool) -> bool:
        data = GeneratedDramaData(self._p.data_dir, generated_drama_id)
        plan = data.load_novel()
        existing = self._store.get_generated_drama_scene(generated_drama_id, chapter, scene)
        if existing is None:
            logger.error(
                "generated_drama_writer: generated_drama_id=%d chapter %d scene %d does not exist yet (cannot rewrite)",
                generated_drama_id, chapter, scene,
            )
            return False

        beats = self._ensure_beats(data, plan, chapter)
        if beats is None:
            return False
        beat = next(
            (s for s in beats.get("scenes", []) if int(s.get("scene", 0)) == scene), None
        )
        if beat is None:
            logger.error("generated_drama_writer: chapter %d's beat sheet has no scene %d", chapter, scene)
            return False

        if check:
            result = self._produce_and_check(
                data, plan, beat, generated_drama_id, chapter, scene,
                previous=existing["body"], extra_issue=_rewrite_format_issue(), full_previous=True,
            )
            if result is None:
                return False
            body, summary, ok, issues = result
            if not ok:
                if existing["attempts"] + 1 >= _MAX_SCENE_ATTEMPTS:
                    logger.warning(
                        "generated_drama_writer: chapter %d scene %d still fails the check after rewriting, "
                        "but the retry limit was reached so it is being adopted (issues: %s)",
                        chapter, scene, " / ".join(issues) or "reason unknown",
                    )
                    ok = True
                else:
                    logger.error(
                        "generated_drama_writer: chapter %d scene %d still fails the check after rewriting. "
                        "Keeping the old text (issues: %s)",
                        chapter, scene, " / ".join(issues) or "reason unknown",
                    )
                    return False
        else:
            written = self._write_scene(
                data, plan, beat, generated_drama_id, chapter, scene,
                issues=[_rewrite_format_issue()], previous=existing["body"], full_previous=True,
            )
            if written is None:
                return False
            body, summary = written
            ok = True  # --no-check：チェックを省略。放送前に人手で確認すること

        scene_id = self._store.save_generated_drama_scene(generated_drama_id, chapter, scene, body, summary, ready=ok)
        data.save_scene_text(chapter, scene, body)
        self._store.reset_generated_drama_progress(scene_id)
        logger.info(
            "generated_drama_writer: rewrote chapter %d scene %d in the new format (scene_id=%d, %d chars, ready=%d)",
            chapter, scene, scene_id, len(body), int(ok),
        )
        self._roll_up_summaries(data, plan, generated_drama_id, chapter)
        return True

    # --- プロンプト -------------------------------------------------------

    def _design_prompt(
        self, title: str, premise: str, chapters: int, scenes: int
    ) -> str:
        target = self._p.scene_target_chars
        return language.pick(
            ja=f"""あなたは連載ラジオドラマの構成作家です。ラジオ番組の深夜枠で、1シーンずつ朗読される
オリジナルのラジオドラマを1本立ち上げます。全体設計（プロット・章立て・登場人物・世界観）を作ってください。

## 作品
題名: {title}
狙い・題材: {premise or "（指定なし。題名から自由に発想してよい）"}

## 条件
- 全{chapters}章。各章は{scenes}シーンで構成する前提で、章ごとの筋書きを書くこと
- 1シーンは声で聴いて 5 分程度（1500〜2000字）。派手な場面転換より、会話と情景で運ぶこと
- 登場人物は2〜6人。**声で聴き分けられるよう、口調（speech_style）をはっきり書き分けること**
- key は登場人物ごとの半角英字の短い識別子（例: taro, rin）。name は本文に出てくる呼び名
- aliases には本文で使う別の呼び方（愛称・呼び捨て・肩書き）を、**1つにつき1要素**で
  入れること（"愛称、呼び捨て" のように1つの文字列へ詰め込まない。注釈も書かない）
- 章立ては起承転結が付き、最終章で話が終わること（続きものにしない）
- 音声で読み上げるため、記号・絵文字・アルファベットを使わないこと
""",
            en=f"""You are the story editor for a serial. You are starting an original story that
will be read out one scene at a time on a radio show's late-night slot. Produce the
overall design: the plot, the chapter breakdown, the cast and the world.

## The work
Title: {title}
What it is about: {premise or "(nothing specified; invent freely from the title)"}

## Requirements
- {chapters} chapters in total. Write the outline for each chapter on the basis that
  each one is built from {scenes} scenes
- One scene runs about five minutes aloud (roughly {target} characters of prose). Carry it
  on dialogue and on the feel of the place, not on constant changes of location
- Two to six characters. **Write speech_style so that each one is distinguishable by
  ear alone** — rhythm, vocabulary, how much they say, what they never say
- key is a short lowercase ASCII identifier per character (for example: mara, finn).
  name is the name they are actually called in the prose
- aliases holds the other ways the prose refers to them (nickname, surname, title),
  **one per array element**. Do not pack several into one string and do not annotate them
- The chapters must build and then finish: the last chapter ends the story.
  This is not an open-ended series
- Every word will be read aloud by a speech synthesiser. Use no symbols, no emoji and
  no digits — write any number out as words
""",
        )

    def _character_sheet(self, data: GeneratedDramaData) -> str:
        if language.current() == "en":
            return "\n".join(
                f"- {c.name} (key: {c.key}"
                f"{'; also called: ' + ', '.join(c.aliases) if c.aliases else ''})"
                f" {c.role}: {c.persona}\n  How they talk: {c.speech_style}"
                for c in data.load_characters()
            )
        return "\n".join(
            f"- {c.name}（key: {c.key}{'／別称: ' + '、'.join(c.aliases) if c.aliases else ''}）"
            f" {c.role}: {c.persona}\n  口調: {c.speech_style}"
            for c in data.load_characters()
        )

    def _world_sheet(self, data: GeneratedDramaData) -> str:
        w = data.load_world()
        if language.current() == "en":
            rules = "\n".join(f"  - {r}" for r in w.get("rules", []))
            return (
                f"Where: {w.get('setting', '')}\n"
                f"When: {w.get('era', '')}\n"
                f"Tone: {w.get('tone', '')}\n"
                + (f"Rules this world runs on:\n{rules}\n" if rules else "")
            )
        rules = "\n".join(f"  - {r}" for r in w.get("rules", []))
        return (
            f"舞台: {w.get('setting', '')}\n"
            f"時代: {w.get('era', '')}\n"
            f"トーン: {w.get('tone', '')}\n"
            + (f"設定上のルール:\n{rules}\n" if rules else "")
        )

    def _beats_prompt(
        self, data: GeneratedDramaData, plan: dict, chapter_plan: dict, count: int
    ) -> str:
        summaries = data.load_summaries()
        target = self._p.scene_target_chars
        return language.pick(
            ja=f"""あなたは連載ラジオドラマの構成作家です。次の章のハコ書き（シーンごとの段取り）を作ってください。

## 作品全体
題名: {plan.get('title', '')}
ログライン: {plan.get('logline', '')}
テーマ: {plan.get('theme', '')}

## 世界観
{self._world_sheet(data)}

## 登場人物
{self._character_sheet(data)}

## ここまでのあらすじ
{summaries.get('overall') or '（まだ何も起きていない。ここが物語の始まり）'}

## 今回の章
第{chapter_plan.get('chapter')}章「{chapter_plan.get('title', '')}」
筋書き: {chapter_plan.get('synopsis', '')}

## 条件
- ちょうど{count}シーンに割ること。scene は 1 から始まる連番
- 各シーンについて、狙い（purpose）・場所（setting）・登場する人物（characters には
  上の key を使う）・起きること（beats を3〜6個）・シーンの終わり方（ends_with）を書く
- 1シーンは 1500〜2000 字で書ける分量に収めること。詰め込みすぎないこと
- 章の最後のシーンは、次章へ引く「引き」で終えること（最終章なら締めること）
- 音で聴かせる番組なので、章のどこかに**音が派手に動く見せ場**（衝突・急変・
  大きな音・立ち回り）を最低1つ入れること。静かな会話シーンとの緩急をつけること
""",
            en=f"""You are the story editor for a serial. Break the next chapter down into scenes.

## The work as a whole
Title: {plan.get('title', '')}
Logline: {plan.get('logline', '')}
Theme: {plan.get('theme', '')}

## The world
{self._world_sheet(data)}

## The cast
{self._character_sheet(data)}

## The story so far
{summaries.get('overall') or '(nothing has happened yet; this is where the story starts)'}

## This chapter
Chapter {chapter_plan.get('chapter')}: "{chapter_plan.get('title', '')}"
Outline: {chapter_plan.get('synopsis', '')}

## Requirements
- Break it into exactly {count} scenes. scene is a one-based running number
- For each scene write: what it is for (purpose), where it happens (setting), who is in
  it (characters — use the keys above), what happens (three to six beats) and how the
  scene ends (ends_with)
- Each scene must fit in about {target} characters of prose. Do not overload one scene
- The last scene of the chapter ends on a pull into the next one
  (or brings things to a close, if this is the final chapter)
- This is a show heard rather than read, so somewhere in the chapter put at least one
  **set piece where the sound moves hard** — a collision, a sudden turn, something loud,
  a scuffle. Play it against the quiet conversation scenes
""",
        )

    def _scene_prompt(
        self,
        data: GeneratedDramaData,
        plan: dict,
        beat: dict,
        generated_drama_id: int,
        chapter: int,
        scene: int,
        issues: list[str],
        previous: str,
        full_previous: bool = False,
    ) -> str:
        summaries = data.load_summaries()
        recent = self._store.recent_generated_drama_summaries(generated_drama_id, 2)
        recent_text = "\n".join(f"- {s}" for s in recent) or language.pick(
            ja="（今回が最初のシーン）", en="(this is the first scene)"
        )
        characters = {c.key: c for c in data.load_characters()}
        appearing = [characters[k] for k in beat.get("characters", []) if k in characters]
        style_label = language.pick(ja="口調", en="How they talk")
        sheet = "\n".join(
            f"- {c.name}: {c.persona}\n  {style_label}: {c.speech_style}" for c in appearing
        ) or self._character_sheet(data)

        rewrite = ""
        if previous:
            prev_text = previous if full_previous else previous[:3000]
            issue_head, draft_head = (
                ("## Notes on the last draft (you must fix these)", "## The draft to fix")
                if language.current() == "en"
                else ("## 前回の原稿への指摘（必ず直すこと）", "## 直す対象の原稿")
            )
            # チェックが ok=false なのに issues が空のことがある（指摘なしの不合格）。
            # その場合も前回の原稿は見せないと、指摘なしのまま白紙から書き直すことになり
            # 同じ理由不明の不合格を繰り返してしまう。
            issue_lines = issues or [
                language.pick(
                    ja="具体的な指摘は得られなかったが、前回の原稿は不合格だった。"
                    "書式（セリフ・ト書き・効果音の形式）と設定・あらすじとの整合性を"
                    "自分で見直して書き直すこと。",
                    en="No specific reason was given, but the last draft failed the check. "
                    "Re-examine the format (dialogue, stage directions, sound effects) and "
                    "consistency with the setting and story so far, and rewrite it yourself.",
                )
            ]
            rewrite = (
                f"\n{issue_head}\n"
                + "\n".join(f"- {i}" for i in issue_lines)
                + f"\n\n{draft_head}\n"
                + prev_text
                + "\n"
            )

        if language.current() == "en":
            return self._scene_prompt_en(
                data, plan, beat, chapter, scene, sheet, summaries, recent_text, rewrite
            )

        return f"""あなたは連載ラジオドラマの書き手です。指定されたシーンの**本文だけ**を書いてください。
この原稿はラジオ番組でそのまま朗読されます（編集も要約もされません）。

## 作品
題名: {plan.get('title', '')}
ログライン: {plan.get('logline', '')}

## 世界観
{self._world_sheet(data)}

## このシーンに出る人物
{sheet}

## ここまでのあらすじ
{summaries.get('overall') or '（まだ何も起きていない）'}

## 直前のシーンの要約
{recent_text}

## 今回書くシーン（第{chapter}章 第{scene}場）
狙い: {beat.get('purpose', '')}
場所: {beat.get('setting', '')}
起きること:
{chr(10).join(f'- {b}' for b in beat.get('beats', []))}
終わり方: {beat.get('ends_with', '')}
{rewrite}
## 書き方の制約（朗読されるため厳守）
この本文は「ラジオドラマの台本」です。地の文で情景をつづる小説ではなく、
**セリフと短いト書きを積み重ねて場面を運ぶ**ことを最優先にしてください。
音だけで聴かせるので、紙芝居のように**メリハリ**をつけること。見せ場では
効果音（擬音）と短い叫びで場面を派手に動かし、静かな場面との落差を作ること。

- 分量は {self._p.scene_target_chars} 字前後を目安に。{self._p.scene_target_chars // 2} 字を
  下回ると場面が薄くなるので、その場合はやり取りごとに反応・間（ま）・短い仕草をもう一手ずつ
  足して深めること（同じ話の往復や効果音の連打で字数を埋めるのは不可）。逆に
  {int(self._p.scene_target_chars * 1.5)} 字を超えそうなら、引き延ばさず「終わり方」へ着地させること
- **同じセリフ・同じ効果音・同じト書きを繰り返さないこと。** 場面は毎行進める。
  堂々巡り（同じやり取りの往復、効果音だけが続く展開）になったら、その時点で
  「終わり方」へ進めて締めること
- **1行につき1発言、または1つの短いト書きだけを書くこと。** 空行を入れず、改行だけで区切ること
- **セリフは必ず次の形式で書くこと：**
  `名前：「セリフ本文」`
  例）ハク：「もう戻れないところまで来てしまいましたね」
  - 名前は「このシーンに出る人物」に書かれている呼び名をそのまま使うこと（略さない・言い換えない）
  - 名前とセリフの間は**全角コロン「：」を1つだけ**使うこと。半角コロン・読点「、」・
    スペース・ダッシュなど、コロン以外の記号は使わないこと
  - セリフの本文は必ず「」で囲むこと。地の文にセリフを埋め込まないこと（
    「〜と○○は言った」のような書き方はしないこと）
  - 同じ人物が続けて話すときも、行ごとに毎回「名前：」を書くこと（2行目以降を省略しない）
  - 一行に2人以上のセリフを混ぜないこと
- **ト書き（地の文）は、動作・表情・間（ま）を一言添える程度に最小限へ絞ること。**
  1行は短く（目安30字以内）。情景描写や心理描写を何行も続けないこと
  例）ハクは目を伏せた。
  例）（沈黙。遠くで風の音だけが響く。）
  - ト書きの行頭に「人物名＋コロン」や「ト書き：」のような見出しラベルを書かないこと
    （丸括弧（）で囲むこと自体が地の文の印。セリフの行専用の書式や見出しは付けない）
  - 心の中の声は、かぎ括弧を使わずト書きと同じ短い形で書くこと
- **効果音（擬音）は独立した1行で書くこと。** カタカナと長音・促音・感嘆符だけで組み、
  他の言葉を混ぜないこと。ナレーターが1行まるごと読み上げるので短く強く
  例）ゴゴゴゴゴーーーッ！
  例）（ドオオオン！）
  例）バタン
  - 見せ場・急変・大きな音のたびに1つ入れてよい。ただし1シーンに3〜4個までを目安に
  - 驚きや悲鳴はセリフとして書くこと（例）健太：「うわあああっ！」）。効果音の行に人の声を入れない
- 記号・絵文字・Markdown記法（*、#、` など）を使わないこと。ただし次は使ってよい：
  ト書きを全角の丸括弧（）で囲む／効果音やセリフで「！」「？」「ー」「っ」を重ねて勢いを出す
- アルファベットや中国語の漢字を使わず、すべて日本語（かな・常用漢字・カタカナ）で書くこと
  （「谁」「哈」「那」などの中国語字を混ぜない）
- 章題・シーン番号・見出しを本文に書かないこと。いきなり本文から始めること
- summary には、このシーンで起きたことを 150 字以内で（次のシーンを書くための引き継ぎ）
"""

    def _scene_prompt_en(
        self,
        data: GeneratedDramaData,
        plan: dict,
        beat: dict,
        chapter: int,
        scene: int,
        sheet: str,
        summaries: dict,
        recent_text: str,
        rewrite: str,
    ) -> str:
        """英語版の :meth:`_scene_prompt`。段落構成は日本語版と1対1で対応させてある。

        **書式の指定は ``parser.py`` の判定とセットで動く。** 話者区切りは半角コロン、
        セリフは直引用符、効果音は全部大文字の独立行。片方だけ変えると、
        セリフが全部ナレーター読みになったり、効果音の「間」が付かなくなる。
        """
        target = self._p.scene_target_chars
        return f"""You are the writer on a serial. Write **only the prose of the scene** you are given.
This draft is read out on the radio exactly as it stands — nobody edits or trims it.

## The work
Title: {plan.get('title', '')}
Logline: {plan.get('logline', '')}

## The world
{self._world_sheet(data)}

## Who is in this scene
{sheet}

## The story so far
{summaries.get('overall') or '(nothing has happened yet)'}

## Summary of the scenes just before this one
{recent_text}

## The scene to write now (chapter {chapter}, scene {scene})
What it is for: {beat.get('purpose', '')}
Where: {beat.get('setting', '')}
What happens:
{chr(10).join(f'- {b}' for b in beat.get('beats', []))}
How it ends: {beat.get('ends_with', '')}
{rewrite}
## How to write it (this is read aloud, so these are not optional)
What you are writing is **a radio play script**, not a novel that describes things in
continuous prose. Above all else, carry the scene on **dialogue with short stage
directions between the lines.** It is heard and not seen, so give it **hard contrast**,
the way a picture-story show does. At the set pieces, move the scene with sound effects
and short shouts, and let the quiet stretches sit against them.

- Aim for about {target} characters. Below {target // 2} the scene goes thin — if that
  happens, deepen it by giving each exchange one more beat of reaction, one more pause,
  one more small piece of business (do not pad the length with repeated exchanges or
  strings of sound effects). If you are heading past {int(target * 1.5)}, stop extending
  and land on "How it ends"
- **Do not repeat a line, a sound effect or a stage direction.** Every line moves the
  scene on. If it starts going in circles — the same exchange back and forth, sound
  effects with nothing between them — go to "How it ends" and close it there
- **One line of text is one utterance, or one short stage direction. Nothing else.**
  No blank lines: separate everything with single line breaks
- **Write every line of dialogue in exactly this form:**
  `Name: "what they say"`
  For example) Mara: "We are past the point where any of us can go back."
  - Use the name exactly as it appears under "Who is in this scene" (do not shorten it,
    do not substitute another word for it)
  - Between the name and the line put **one half-width colon and a space**, nothing else.
    No dash, no comma, no full-width colon
  - The spoken words always go inside double quotation marks. Do not bury dialogue in
    narration (never write it as `"..." she said` or `she said that ...`)
  - When the same person speaks twice in a row, write `Name: ` again on every line.
    Do not drop it from the second line onwards
  - Never put two people's lines on one line of text
- **Stage directions are one gesture, one look or one pause — the bare minimum.**
  Keep the line short, about fifteen words. Do not run several lines of scenery or of
  what someone is feeling
  For example) Mara looked away.
  For example) (A long silence. Somewhere below, a door.)
  - Put stage directions in round brackets, or write them as one plain short sentence.
    Never head one with a name and a colon, and never label one "Stage direction:"
  - Thoughts go in the same short plain form, without quotation marks
- **Sound effects go on a line of their own, IN CAPITAL LETTERS, with nothing else on
  the line.** The narrator reads the whole line out, so keep it short and hard
  For example) CRASH!
  For example) A DOOR SLAMS BELOW.
  For example) BOOOOM!
  - Capital letters are what marks the line as a sound rather than narration. A sound
    effect line has no colon in it and is five words at most
  - One at each set piece, each sudden turn, each loud moment is fine. Three or four in
    a scene is the sensible ceiling
  - A gasp or a scream is dialogue, not a sound effect
    (for example) Finn: "Look out!"). Never put a human voice on a sound effect line
- Use no symbols, no emoji and no Markdown (*, #, ` and so on). These two are allowed:
  round brackets around a stage direction, and repeated letters or exclamation marks in
  dialogue and sound effects to give them force
- Write no digits. Every number, time and date is spelled out in words
  ("half past two", not "2:30"). The speech synthesiser cannot read a colon in a number
- Do not put a chapter title, a scene number or any heading in the prose.
  Start straight into the scene
- In summary, put what happened in this scene in forty words or fewer
  (it is the handover for writing the next scene)
"""

    def _check_prompt(
        self, data: GeneratedDramaData, plan: dict, beat: dict, body: str, chapter: int
    ) -> str:
        summaries = data.load_summaries()
        if language.current() == "en":
            return f"""You are the continuity editor on a serial. Check whether the draft below
contradicts the setting, what has happened so far, or the plan for this scene, and
whether it is written **in radio play script form**. You are not judging how good the
writing is. You look for **breakages and format violations only**.

## The world
{self._world_sheet(data)}

## The cast
{self._character_sheet(data)}

## The story so far
{summaries.get('overall') or '(nothing has happened yet)'}
Chapter {chapter} up to now: {summaries.get('chapters', {}).get(str(chapter), '(the start of the chapter)')}

## The plan for this scene
What it is for: {beat.get('purpose', '')}
What happens:
{chr(10).join(f'- {b}' for b in beat.get('beats', []))}

## The format you expect (script form)
- Dialogue is `Name: "what they say"`. Between the name and the line, one half-width colon
- Stage directions are short lines: one gesture, one look, one pause
- Sound effects are a line of their own, in capital letters, with nothing else on the
  line (for example "CRASH!", "A DOOR SLAMS BELOW."). **That is the correct format —
  never report a capitalised sound effect line as a problem**

## The draft
{body}

## What counts as a problem (if any of these apply, ok = false and say which, specifically)
- A character's name, what they are called, or how they talk contradicts the setting
- The order of events or a stated fact contradicts the story so far or an earlier scene
- What was planned to happen in this scene does not happen in the draft at all
- A line of dialogue is not inside quotation marks
- A line of dialogue has no speaker name on it, or the name is separated from the line by
  something other than a half-width colon (a comma, a dash, a full-width colon)
- Two people's names or lines are mixed onto one line of text
- A stage direction runs to two sentences or more, or one line is plainly too long
  (well past fifteen words, writing scenery or feelings out like a novel). A sound
  effect line is a short line in capitals and is not this
- There are symbols, emoji or digits in the prose (it cannot be read aloud). Repeated
  letters, repeated exclamation marks, ellipses and capitalised sound effect lines are
  all fine and must not be reported
- A chapter title, a heading or a scene number has got into the prose

## How to write the notes (this matters)
- **Only the things that are genuinely broken, and at most five.** Never repeat a note
- Taste in style, stiffness of phrasing, whether a metaphor works, how thickly a thing
  is described, anything at the level of "this might read as unnatural" — **do not
  report these**. That is not the continuity editor's job
- If nothing matches the criteria above, set ok = true even if the draft is clumsy

In summary, put what happened in this scene in forty words or fewer.
"""
        return f"""あなたは連載ラジオドラマの校閲担当です。次の原稿が、設定・これまでの流れ・今回の段取りと
矛盾していないか、また本文が「ラジオドラマの台本形式」で書かれているかを確認してください。
文章の巧拙は評価しません。**破綻と書式違反だけ**を見ます。

## 世界観
{self._world_sheet(data)}

## 登場人物
{self._character_sheet(data)}

## ここまでのあらすじ
{summaries.get('overall') or '（まだ何も起きていない）'}
第{chapter}章のここまで: {summaries.get('chapters', {}).get(str(chapter), '（章の冒頭）')}

## 今回の段取り
狙い: {beat.get('purpose', '')}
起きること:
{chr(10).join(f'- {b}' for b in beat.get('beats', []))}

## 期待する書式（台本形式）
- セリフは `名前：「セリフ本文」` の形。名前とセリフの間は全角コロン「：」
- 地の文（ト書き）は動作・表情・間を一言添える程度の短い行のみ
- 効果音（擬音）は独立した1行。カタカナ＋「！」「？」「ー」「っ」だけで組まれた
  短い行（例「ゴゴゴゴーーーッ！」「（ドオオン！）」）は正しい書式なので指摘しない

## 原稿
{body}

## 判定の基準（どれかに当てはまれば ok = false、issues に具体的に書く）
- 人物の名前・呼び方・口調が設定と食い違っている
- あらすじや前のシーンと時系列・事実が矛盾している
- 今回の段取り（起きること）が本文でまったく起きていない
- セリフがかぎ括弧で囲まれていない
- セリフの行に話者名（名前：）が付いていない、または名前とセリフの区切りが
  全角コロン「：」以外（読点・ダッシュ・半角コロンなど）になっている
- 一行に2人以上の名前・セリフが混ざっている
- ト書き（地の文）が2文以上続く、または1行が明らかに長い（30字を大きく超え、
  情景描写や心理描写を小説のように書き連ねている）。効果音の行はカタカナだけの
  短い1行なので、これには当たらない
- 記号・絵文字・アルファベットが混ざっている（朗読できない）。ただし「！」「？」「ー」「っ」
  「〜」「…」の連打や、効果音の行（カタカナだけの短い行）は問題ないので指摘しない
- 章題・見出し・シーン番号が本文に混ざっている

## 指摘の書き方（重要）
- **本当に破綻している箇所だけを、多くても5件**。同じ指摘を繰り返さないこと
- 文体の好み・表現の硬さ・比喩の是非・描写の厚み・「不自然かもしれない」程度のことは
  **指摘しない**（それは校閲の仕事ではない）
- 上の基準に当てはまらないなら、多少ぎこちなくても ok = true にすること

summary には、このシーンで起きたことを 150 字以内でまとめること。
"""
