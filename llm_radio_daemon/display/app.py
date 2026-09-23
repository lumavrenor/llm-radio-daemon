"""Ursina による3D表示（v2）。

SPEC.md 2.1節「メインループを絶対にブロックしない」に従い、この関数は
MainThread から一度だけ呼ばれ、Ursina の app.run() に制御を渡す。
update() 内ではネットワークI/O・LLM呼び出しは一切行わず、SharedState を
読むだけに徹する（get_nowait 相当。SharedState は素朴な属性アクセスな
ので、ここでは snapshot() でロックを一瞬だけ取るだけで即座に返る）。

ウィンドウを閉じると Panda3D 側が sys.exit() を呼ぶため、この関数は
戻らずに SystemExit を送出する。呼び出し側(main.py)の finally節で
ワーカースレッドの後始末を行うこと。
"""

from __future__ import annotations

import atexit
import hashlib
import time
from pathlib import Path

import __main__
from panda3d.core import ClockObject, Point2, Point3
from ursina import (
    AmbientLight,
    DirectionalLight,
    Entity,
    Text,
    Ursina,
    application,
    camera,
    color,
    curve,
    invoke,
    scene,
    window,
)
from ursina.shaders import lit_with_shadows_shader

from .. import language
from ..config import CastMember, ContentConfig, DisplayConfig
from ..director import ProgramDirector
from ..schedule import active_content
from ..state import SharedState
from .background import BG_COLOR, Background
from .poly_character import Character
from .switching_log import SwitchingLog
from .window_geometry import load_geometry, start_tracking

_BLINK_HZ = 2.0  # フィラー再生中のONAIRランプの点滅速度

# 初回起動時（保存された位置・サイズが無いとき）のウィンドウサイズ。
_DEFAULT_SIZE = (1000, 700)

# 頭上の名札。3D空間の Text は極小になって見えないので camera.ui(2D) に描画し、
# 各キャラの頭上アンカーを毎フレーム画面投影して追従させる。数値一発で微調整可。
_NAMEPLATE_SCALE = 0.8    # 文字サイズ（他の UI Text と同じ基準）
_NAMEPLATE_UI_LIFT = 0.009  # 投影位置からさらに上へ持ち上げる量（camera.ui 座標）

# UI パネルの下地。素の黒べったりだと安っぽく見えるので、GitHub Dark の
# オーバーレイ色（#161b22）を半透明で敷く。alpha は 0.40→0.62 に上げてあり、
# 背景演出（logs / coderain 等）が動いていても文字が沈まないようにしている。
_TEXT_BG = color.rgba(0.086, 0.106, 0.133, 0.62)

# 文字色は2階調。字幕・ON AIR など「読ませたい」ものは明るいオフホワイト、
# 左上ステータスや名札のような副次情報は少し沈んだ色にして優先度の差を付ける
# （全部同じ白グレーだとデバッグ表示も台詞も同格に見えて素っ気なくなるため）。
_TEXT_PRIMARY = color.rgb(0.90, 0.93, 0.96)
_TEXT_SECONDARY = color.rgb(0.68, 0.75, 0.82)

# 字幕の基準文字サイズ。1行が画面幅を超えるときは、この値から必要なだけ縮小する。
_SUBTITLE_SCALE = 1.15

# --- 画面フェード -------------------------------------------------------------
# camera.overlay は ursina 標準の全画面フェード用エンティティ（UI最前面の黒帯）。
# 起動直後の黒からのフェードインと、コンテンツ切り替え時の暗転に使い回す。
# コンテンツ切り替えの暗転・明転は番組進行（director.py）の指示で行い、描き終えたら
# 報告を返す。番組進行は「暗くなりきった」報告を受けてから裏で入れ替えを行い、
# 次のコーナーの準備が整ってから明転を指示する（ここで時間を見込んで待つことはしない）。
_STARTUP_FADE_DELAY = 0.2   # ウィンドウが出てから始まるまでの間（開いた瞬間の白飛び防止）
_STARTUP_FADE_SEC = 1.4
_CONTENT_FADE_OUT_SEC = 0.35
_CONTENT_FADE_IN_SEC = 0.45
# アニメーション完了から報告までの小さな余裕（最後のフレームが描かれてから報告する）。
_FADE_ACK_MARGIN_SEC = 0.05

# --- コーナー切り替え中（暗転しきってから明転が始まるまで）の待たせ表示 ------------
# 何十秒か真っ黒のままのことがあるので、暗転の上に奥へ傾いたログの面を浮かべる
# （switching_log.py。camera.overlay より手前の専用リージョンに描く）。「切り替え中」
# だと分かる合図は右上の STANDBY ランプ（これも overlay より手前）で足りるので、
# ログは背景の logs レイヤーと同じ色相で、あくまで背景の続きに見える程度に留める。
# director.is_switching() の間だけ出す（他の表示は隠す）。
_OVERLAY_TOP_Z = -99.4  # camera.overlay（z=-99）より手前
_STANDBY_LABEL = "STANDBY"
_STANDBY_BLINK_HZ = 0.8  # フィラーの点滅(_BLINK_HZ)よりゆっくりにして意味を区別する

# --- ひな壇レイアウト --------------------------------------------------------
# 前列（床）に司会・アシスタント、後列（台の上）にその他の出演者を一段高く。
# 1段目と2段目の上下位置・奥行きはここの定数だけで調整する。

_FRONT_ROLES = ("host", "assistant")

_FRONT_Y, _FRONT_Z, _FRONT_SCALE, _FRONT_STEP = 0.0, -1.6, 0.9, 3.0
_BACK_Y, _BACK_Z, _BACK_SCALE, _BACK_STEP = 2.2, 4.0, 0.82, 2.6

# 等間隔グリッドのままだと綺麗に整列しすぎて不自然なので、id ごとに安定した
# 小さなオフセットを左右・前後へ加えて「なんとなく自然にバラけた」立ち位置にする。
# random だと _stage_poses() が呼ばれるたび（毎フレーム）に値が変わってガタつくため、
# 乱数ではなく id のハッシュから決める（同じ人物は毎回同じ位置に立つ）。
_STAGE_JITTER_X = 0.15  # 左右のずれ幅(m)
_STAGE_JITTER_Z = 0.12  # 前後のずれ幅(m)


def _jitter(cast_id: str, salt: str, amplitude: float) -> float:
    """cast_id ごとに安定な -amplitude 〜 +amplitude の疑似乱数値。"""
    digest = hashlib.sha1(f"{cast_id}:{salt}".encode()).digest()
    frac = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF  # 0.0-1.0
    return (frac * 2 - 1) * amplitude


def _edge_jitter_x(i: int, n: int, cast_id: str, salt: str, amplitude: float) -> float:
    """段の左右端（i==0 / i==n-1）は横方向のジッターを与えない。

    段の横幅は「両端の人＋半ステップ」ぴったりで画面内に収まるよう調整済み
    （_stage_poses の riser 幅コメント参照）なので、端の人まで外側へジッター
    させると満員の段で画面端から見切れてしまう（実際に en 版で発生した不具合）。
    内側の人だけ左右にバラけさせれば、両端の実効幅は変わらないまま自然さは保てる。
    """
    if n <= 1 or i == 0 or i == n - 1:
        return 0.0
    return _jitter(cast_id, salt, amplitude)


# 台（riser）の色。動的ライトだけに任せると、現在のライト角度（rx=40/ry=-28、
# キャラの顔まわり用にチューニング済み＝台のために変えると影響が出る）では
# 軸並行な直方体の「上面」と「正面」の入射角がほぼ同じになり、色を変えても
# 陰影が付かず単色の板に見えてしまう（実測で確認済み）。そのため天面だけ
# 厚さ0の板（model="plane"。側面ポリゴンが無いので box 版で出た二重の
# エッジ線が出ない）を本体のわずか上に重ね、_RISER_CAP_TINT で明るくして
# 「上面が明るい・側面が地の色」という陰影を動的ライトに依存せず焼き込む。
_RISER_COLOR = color.rgba(0.1, 0.1, 0.1, 0.65)
_RISER_CAP_TINT = 0.18
_RISER_CAP_LIFT = 0.01  # 本体の天面とのZファイティングを避けるための微小なかさ上げ

# 後列が多いとき（10人ひな壇など）は 1 段に詰め込まず、奥へ段を足して雛壇にする。
# 1 段の最大人数を超えたら段数 = ceil(人数 / _BACK_MAX_PER_ROW) に分け、
# 奥の段ほど高く・遠く・少し小さく置く（前の段の頭の上からのぞく形）。
_BACK_MAX_PER_ROW = 5
_BACK_ROW_DY = 1.6      # 1 段奥へ行くごとに上げる高さ
_BACK_ROW_DZ = 2.8      # 1 段奥へ行くごとに足す奥行き
_BACK_ROW_SHRINK = 0.92  # 1 段奥へ行くごとの縮小率

_ROLE_PALETTE = {
    "host": color.hsv(45, 0.15, 0.9),
    "assistant": color.hsv(340, 0.35, 0.9),
    "other": color.hsv(205, 0.55, 0.85),
}

# [[cast]] の未指定時に使う既定色（poly_character.Character のデフォルト引数と揃える）。
_DEFAULT_HAIR_COLOR = color.gray
_DEFAULT_EYE_COLOR = color.black


def _hex_color(value: str | None, default: color.Color) -> color.Color:
    """[[cast]] の "#rrggbb" 文字列を ursina の color.Color へ。未指定なら default。

    形式チェックは config._validate_cast 側で済んでいる前提。
    """
    if not value:
        return default
    r, g, b = (int(value[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return color.rgb(r, g, b)

# 字幕の折り返し。ursina の Text.wordwrap は空白でしか改行できず日本語では
# 効かない（一度セットした時にしか適用されない）ため、こちらで手動改行する。
# 1行の文字数・行数は [display] subtitle_wrap / subtitle_max_lines で調整する。
#
# 日本語はどこで切っても読めるので単純に文字数で割るが、英語で同じことをすると
# 語の途中で改行される（"inference" が "infere / nce" になる）。[locale] lang = "en"
# のときだけ語境界で折る。全角・半角の幅の違いは、1行の文字数そのものを
# [display] subtitle_wrap で言語ごとに設定して吸収する（en は 80 前後）。


def _wrap_subtitle(text: str, wrap: int = 49, max_lines: int = 10) -> str:
    """字幕を wrap 文字ごとに改行し、max_lines 行に収める。"""
    text = text.replace("\n", " ").strip()
    if not text:
        return " "
    if language.current() == "en":
        lines = _wrap_words(text, wrap)
    else:
        lines = [text[i : i + wrap] for i in range(0, len(text), wrap)]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][: wrap - 1] + "…"
    return "\n".join(lines)


def _wrap_words(text: str, wrap: int) -> list[str]:
    """空白で折り返す（英語用）。``wrap`` より長い単語だけは途中で切る。"""
    lines: list[str] = []
    current = ""
    for word in text.split():
        while len(word) > wrap:  # URL やモデル名など、1語で1行を超えるもの
            if current:
                lines.append(current)
                current = ""
            lines.append(word[:wrap])
            word = word[wrap:]
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= wrap:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


_READING_ROW_Y, _READING_ROW_Z, _READING_ROW_SCALE, _READING_ROW_STEP = 0.0, -1.4, 1.0, 3.0
# この深さ(z=-1.4)でカメラに映る横幅は実測で片側 5.3m 程度しかなく、step=3.0 だと
# 5人目（x=±7.5）以降は画面外に出て消えてしまう（実際に en 版の biography_reading
# で発生した不具合）。4人（x=±4.5）までは収まるのでそこで段を折り返す。
_READING_MAX_PER_ROW = 4


def _reading_poses(
    ids: list[str],
) -> tuple[dict[str, tuple[float, float, float, float]], list[dict]]:
    """読書コーナー中の出演者を前列中央へ寄せて並べる（§10.11）。

    literary_reading/translated_reading は常に朗読・つっこみの2名だが、
    biography_reading は MC＋その他（min/max_speakers 通り、最大10人）になりうる
    （docs/spec-biography-reading.md）。全員を1段の横一列に並べる単純な実装だと、
    人数が _READING_MAX_PER_ROW を超えたときに画面外へはみ出て消えてしまうため、
    _stage_poses の後列と同じ考え方（人数超過分は奥・高い段へ折り返し、台を敷く）
    で続きを並べる。2名だけのときは従来と全く同じ位置になる。
    """
    if not ids:
        return {}, []
    rows = _split_rows(list(ids), _READING_MAX_PER_ROW)
    poses: dict[str, tuple[float, float, float, float]] = {}
    risers: list[dict] = []
    for r, row_ids in enumerate(rows):
        if r == 0:
            y, z, sc, step = _READING_ROW_Y, _READING_ROW_Z, _READING_ROW_SCALE, _READING_ROW_STEP
        else:
            y = _BACK_Y + (r - 1) * _BACK_ROW_DY
            z = _BACK_Z + (r - 1) * _BACK_ROW_DZ
            sc = _BACK_SCALE * (_BACK_ROW_SHRINK ** (r - 1))
            step = _BACK_STEP
        x0 = -step * (len(row_ids) - 1) / 2
        row_xs = [x0 + i * step for i in range(len(row_ids))]
        for cid, x in zip(row_ids, row_xs):
            poses[cid] = (x, y, z, sc)
        if r > 0:
            half_w = max(abs(x) for x in row_xs) + step / 2
            risers.append({"scale": (2 * half_w, y, 2.8), "y": y / 2, "z": z, "x": 0.0})
    return poses, risers


def _split_rows(members: list, max_per_row: int) -> list[list]:
    """出演者を「なるべく均等な」段へ分ける（例: 9 人 → 5 + 4）。"""
    if len(members) <= max_per_row:
        return [members]
    rows = -(-len(members) // max_per_row)  # ceil
    per_row = -(-len(members) // rows)
    return [members[i : i + per_row] for i in range(0, len(members), per_row)]


_MAX_BACK_ROWS = 4  # _build_stage が用意する riser の数（後列 20 人ぶん）


def _stage_poses(
    members: list[CastMember],
) -> tuple[dict[str, tuple[float, float, float, float]], list[dict]]:
    """アクティブな出演者だけをひな壇へ並べる。

    戻り値は (cast_id -> (x, y, z, scale)) と、後列の各段の台(riser)の寸法リスト
    （手前の段から順。後列が空なら空リスト）。段ごとに別の台を出すのは、1 枚の
    高い台にすると手前の段のキャラを隠してしまうため（各台の高さ＝その段の足元）。
    人数に合わせて毎回センタリングし直すので「今日の番組は3人」なら3人ぶんの幅で
    綺麗に並ぶ。後列が多いときは _BACK_MAX_PER_ROW ごとに段を足す。
    前列ロールの出演者が1人も居ないときは、後段ブロックを丸ごと床まで下げて
    先頭段を接地させる（全員が後段に浮いて画面下が空くのを防ぐ）。
    """
    front = [m for m in members if m.role in _FRONT_ROLES]
    back = [m for m in members if m.role not in _FRONT_ROLES]  # other / 未定義はすべて後列

    poses: dict[str, tuple[float, float, float, float]] = {}

    if front:
        x0 = -_FRONT_STEP * (len(front) - 1) / 2
        for i, m in enumerate(front):
            jitter_x = _edge_jitter_x(i, len(front), m.id, "front-x", _STAGE_JITTER_X)
            x = x0 + i * _FRONT_STEP + jitter_x
            z = _FRONT_Z + _jitter(m.id, "front-z", _STAGE_JITTER_Z)
            poses[m.id] = (x, _FRONT_Y, z, _FRONT_SCALE)

    # 前列（司会・アシスタント）が1人も居ない編成だと、後段が _BACK_Y 以上に
    # 浮いたまま並んで画面の上半分にキャストが固まり、下が間延びする。その場合は
    # 後段ブロックを丸ごと _BACK_Y ぶん下へ平行移動し、先頭段を床に接地させる。
    # 横間隔・奥行き・段数・センタリングは通常の後段とまったく同じなので、後列で
    # 成立している並び（両端が見切れない幅）がそのまま下がるだけ。前列へ「繰り上げ」
    # ないのは、床近くまで前に出すと _BACK_MAX_PER_ROW(5) 人が画面端で見切れるため。
    # 床に降りた先頭段には台を出さない（前列と同じく床置き）。
    ground_drop = _BACK_Y if (not front and back) else 0.0

    back_rows = _split_rows(back, _BACK_MAX_PER_ROW) if back else []
    # 段ごとに素の座標を組み、最後に後列ブロック全体を横センタリングし直す。
    # 前段と同じ人数の段だけ半ステップずらして「煉瓦積み」にする（人数が違う段は
    # そのままで自然に互い違いになる）。ずらしで後列が片側へ寄らないよう、最後に
    # 全キャラの x の平均を 0 に合わせる。ただし段が最大人数（_BACK_MAX_PER_ROW）
    # まで埋まっているときはずらさない＝半ステップ分だけ横に広がって両端が画面外へ
    # 出てしまうため（満員の段はそのまま真上に重ねる）。
    rows_raw: list[tuple[list, float, float, float, float]] = []  # (members, x_center, y, z, sc)
    prev_len: int | None = None
    offset = 0.0
    for r, row_members in enumerate(back_rows):
        y = _BACK_Y + r * _BACK_ROW_DY - ground_drop
        z = _BACK_Z + r * _BACK_ROW_DZ
        sc = _BACK_SCALE * (_BACK_ROW_SHRINK ** r)
        if (
            prev_len is not None
            and len(row_members) == prev_len
            and len(row_members) < _BACK_MAX_PER_ROW
        ):
            offset = _BACK_STEP / 2 - offset  # 0 ↔ 半ステップ を交互に
        else:
            offset = 0.0
        prev_len = len(row_members)
        rows_raw.append((row_members, offset, y, z, sc))

    if rows_raw:
        xs = [
            xc + (i - (len(mem) - 1) / 2) * _BACK_STEP
            for mem, xc, *_ in rows_raw
            for i in range(len(mem))
        ]
        recenter = -sum(xs) / len(xs)
    else:
        recenter = 0.0

    risers: list[dict] = []
    for row_members, x_center, y, z, sc in rows_raw:
        x0 = x_center + recenter - _BACK_STEP * (len(row_members) - 1) / 2
        row_xs = [x0 + i * _BACK_STEP for i in range(len(row_members))]
        for i, (m, x) in enumerate(zip(row_members, row_xs)):
            jx = x + _edge_jitter_x(i, len(row_members), m.id, "back-x", _STAGE_JITTER_X)
            jz = z + _jitter(m.id, "back-z", _STAGE_JITTER_Z)
            poses[m.id] = (jx, y, jz, sc)
        # 床に接地した段（前列なしで先頭段が下がったケース）は前列と同じ床置きなので台なし。
        if y <= 0.05:
            continue
        # その段だけの台。高さ＝その段の足元まで（手前の段のキャラは隠さない）。
        # 台はキャラの「半ステップずらし」にも段ごとのセンタリング補正にも追従させず、
        # 常に画面中央（x=0）へ据える（段ごとに台が左右に振れて見えるのを防ぐ）。
        # 幅は「いちばん外側のキャラ＋左右 0.5 人分」でぴったり包む。満員の段でも
        # 画面端には届かず、煉瓦ずらしで片寄った段は寄った側も 0.5 人分は残る。
        half_w = max(abs(x) for x in row_xs) + _BACK_STEP / 2
        risers.append(
            {
                "scale": (2 * half_w, y, 2.8),
                "y": y / 2,
                "z": z,
                "x": 0.0,
            }
        )
    return poses, risers


def _build_stage(cast: list[CastMember]) -> tuple[dict[str, Character], list[dict[str, Entity]]]:
    """出演者全員のキャラ Entity と、後列の各段の台を1回だけ生成する。

    立ち位置（誰を並べるか・どこに並べるか）と台の寸法は毎フレーム update() 側で
    _stage_poses() を使って決め直す。ここでは生成だけ（台は _MAX_BACK_ROWS 枚）。
    """
    characters = {
        m.id: Character(
            m.id,
            name=m.name,
            model=m.model,
            hair_style=m.hair_style,  # None なら model プリセットの既定髪型（Character側で解決）
            hair_color=_hex_color(m.hair_color, _DEFAULT_HAIR_COLOR),
            eye_color=_hex_color(m.eye_color, _DEFAULT_EYE_COLOR),
            dress_color=_hex_color(m.clothing_color, _ROLE_PALETTE.get(m.role, color.gray)),
            # 装飾品は (type, 色 or None, side or None) のタプルで渡す。色の既定値は
            # 髪色に依存するので解決は poly_character 側に任せる（None のまま渡す）。
            accessories=[
                (a.type, _hex_color(a.color, None) if a.color else None, a.side)
                for a in m.accessory
            ],
        )
        for m in cast
    }
    risers = [
        {
            # 本体（側面・底面）。地の色のまま。
            "body": Entity(
                model="cube",
                shader=lit_with_shadows_shader,
                color=_RISER_COLOR,
                scale=(_BACK_STEP + 2.0, _BACK_Y, 2.8),
                y=_BACK_Y / 2,
                z=_BACK_Z,
                enabled=False,
            ),
            # 天面だけの厚さ0の板。_RISER_CAP_TINT で明るくして、動的ライトの
            # 角度に依存せず「上面が明るい」陰影を焼き込む（上のコメント参照）。
            # 片面描画のみ（上から見たときだけ明るい面を出す）。奥の段はカメラより
            # 高い位置に来るため、両面描画にすると下から裏側（表と同じ明るい色）が
            # 見えてしまい不自然だった。片面にすればカリングされ、下から覗いた際は
            # 素通しで奥が見える自然な見え方になる。
            "cap": Entity(
                model="plane",
                shader=lit_with_shadows_shader,
                color=_RISER_COLOR.tint(_RISER_CAP_TINT),
                scale=(_BACK_STEP + 2.0, 1, 2.8),
                y=_BACK_Y + _RISER_CAP_LIFT,
                z=_BACK_Z,
                enabled=False,
            ),
        }
        for _ in range(_MAX_BACK_ROWS)
    ]
    return characters, risers


def run_display(
    state: SharedState,
    display_config: DisplayConfig,
    cast: list[CastMember],
    llm_model: str = "",
    llm_engine_label: str = "",
    content: list[ContentConfig] | None = None,
    director: ProgramDirector | None = None,
    music_enabled: bool = True,
) -> None:
    # 前回終了時のウィンドウ位置・サイズを復元する（無ければ既定サイズで中央）。
    restored = (
        load_geometry(display_config.window_state_path)
        if display_config.remember_window
        else None
    )
    app = Ursina(
        title="LLM Radio Daemon",
        fullscreen=False,
        size=(restored.width, restored.height) if restored else _DEFAULT_SIZE,
        vsync=True,
        development_mode=False,
    )

    # 位置の復元は Ursina() の後で行う。ursina 8.3 は Ursina(position=...) を
    # 実際には使っておらず、初期化の最後に必ず中央配置するため。
    tracker = None
    if display_config.remember_window:
        tracker = start_tracking(display_config.window_state_path, restored)
        # Panda3D はウィンドウを閉じると sys.exit() するので、保存は atexit に置く。
        atexit.register(tracker.save)

    clock = ClockObject.get_global_clock()
    clock.set_mode(ClockObject.M_limited)
    clock.set_frame_rate(display_config.fps_limit)

    window.color = BG_COLOR

    # 起動時フェードイン。camera.overlay は ursina 標準の全画面フェード用エンティティ
    # （UI最前面に敷かれる黒帯）。ここで不透明にしておき、シーン構築が終わって
    # app.run() に入る直前にフェードを仕込む（本体は末尾）ことで、黒画面から
    # 静かに絵が浮かび上がる形にする。
    camera.overlay.color = color.black

    # ライト。ursina の DirectionalLight は Light.default_values で rotation_x=90
    # （真上向き）が既定になっており、rotation=(x, y, z) というタプル引数だと
    # このデフォルトの方が後勝ちして rotation_x が常に90に上書きされてしまう
    # （実測で確認済みのライブラリ側の挙動）。rotation_x= / rotation_y= を個別の
    # キーワード引数で渡さないと値が反映されない。写真のループライティング
    # （水平より40〜50°上・正面よりやや斜め）に寄せて rotation_x=40 とし、
    # フラットシェードの顔に濃い片影が出過ぎないよう環境光も少し上げる。
    # test_character.py 側は床つき単体プレビュー用で別調整。
    DirectionalLight(rotation_x=40, rotation_y=-28, color=color.rgb(1.0, 0.98, 0.95))
    AmbientLight(color=color.rgba(1, 1, 1, 0.38))

    if display_config.font:
        # 既定フォントは日本語グリフを含まないため、CJK対応フォントに差し替える。
        # ursina のフォント探索は既知フォルダからのファイル名検索なので、
        # 任意パスのフォントを使うにはそのフォルダを検索対象に加える。
        font_path = Path(display_config.font)
        application.fonts_folder = font_path.parent
        Text.default_font = font_path.name

    # 前列(床)と後列(台の上)が段として読めるよう、やや高く・見下ろし気味に。
    # rotation_x は見下ろし角度。7だと後列（台の上）の頭が画面上端に寄りすぎ、
    # 初期ウィンドウサイズで左上のステータスパネル（LLM/CONTENT/NOW PLAYING）と
    # 名札が重なっていたため 4 へ弱めて全体を少し下げてある（実測で確認済み）。
    camera.position = (0, 2.9, -12.8)
    camera.rotation_x = 6.2
    camera.fov = 50

    cast_by_id = {m.id: m for m in cast}
    characters, risers = _build_stage(cast)

    # --- 頭上の名札（camera.ui へ 2D 描画し、update() で画面追従させる） ---
    _lens = application.base.camLens

    def _project_to_ui(world_pos) -> tuple[float, float] | None:
        """ワールド座標を camera.ui 座標系 (x:±aspect/2, y:±0.5) へ投影する。
        カメラ背面・視錐台外なら None。"""
        cam_space = camera.getRelativePoint(scene, world_pos)
        out = Point2()
        if not _lens.project(Point3(cam_space[0], cam_space[1], cam_space[2]), out):
            return None
        return (out[0] * 0.5 * camera.aspect_ratio, out[1] * 0.5)

    nameplates = {}
    for cid, ch in characters.items():
        if not ch.display_name:
            continue
        plate = Text(
            text=ch.display_name,
            parent=camera.ui,
            origin=(0, 0),
            scale=_NAMEPLATE_SCALE,
            color=_TEXT_SECONDARY,
        )
        plate.create_background(color=_TEXT_BG, padding=(0.01, 0.01))
        nameplates[cid] = plate

    subtitle = Text(
        text=" ",  # 空文字だと一部の内部処理で落ちるため空白で初期化する
        parent=camera.ui,
        # 各行は左そろえ（origin x=-0.5）。ブロック全体の左右中央そろえは
        # テキスト確定後に x を「幅の半分ぶん左」へ寄せて実現する（下の update 内）。
        # origin y=-0.5 で下端を基準に上方向へ行が伸びる。
        position=window.bottom + (0, 0.05),
        origin=(-0.5, -0.5),
        scale=_SUBTITLE_SCALE,
        color=_TEXT_PRIMARY,
    )
    _SUBTITLE_ANCHOR_Y = subtitle.y  # x は毎回ブロック幅で決め直すので y だけ保持
    _subtitle_shown = [""]  # 直近の字幕。変化時だけ背景を貼り直す（毎フレームは重い）
    # 無音（current_speaker が None）が続いている開始時刻。喋っている間は None に戻す。
    # 1行ごとの再生の合間にも current_speaker は一瞬 None になる（mixer._pull_speech）ため、
    # 即座に消すのではなく subtitle_idle_clear_sec だけ連続で無音が続いた場合だけ消す。
    _subtitle_silence_since: list[float | None] = [None]

    content = content or []

    def _model_line(ready: bool) -> str:
        if not llm_model:
            return "LLM: (unknown)"
        # モデル名の後ろに推論エンジン名を添える（例: "LLM: qwen3:8b (Ollama)"）。
        label = f"{llm_model} ({llm_engine_label})" if llm_engine_label else llm_model
        # 起動直後は Ollama 側のモデルロードに数十秒〜1分ほどかかることがあるため、
        # ロード完了（state.llm_ready）まで「Loading...」を添えて待機中だと分かるようにする。
        return f"LLM: {label}" if ready else f"LLM: {label} Loading..."

    # 放送中のコーナー（番組進行が暗転中に切り替える）は毎フレーム要らないのでごく
    # 短くキャッシュする。左上のコンテンツ名と背景の切り替えが同じ結果を共有する。
    _content_cache: dict = {"t": 0.0, "active": None}

    def _active_content() -> ContentConfig | None:
        now = time.time()
        if now - _content_cache["t"] >= 0.2:
            _content_cache["active"] = active_content(content) if content else None
            _content_cache["t"] = now
        return _content_cache["active"]

    if director is not None:
        director.attach_display()
    _fade_state: dict = {"handled": None}
    _local_fade_state: dict = {"handled": None}
    _stage_cache: dict = {"poses": None, "risers": None}

    def _drive_fades() -> None:
        """番組進行のフェード指示を描き、描き終えたら報告する。

        ラジオドラマの作品の変わり目（同じコーナー内でのシーンまたぎ）も含め、
        コーナー内の暗転は _drive_local_fade() 側（state.local_fade）が受け持つ。
        """
        cmd = director.fade_command() if director is not None else None
        if cmd is not None and cmd != _fade_state["handled"]:
            _fade_state["handled"] = cmd
            kind, tid = cmd
            if kind == "out":
                camera.overlay.animate_color(
                    color.black, duration=_CONTENT_FADE_OUT_SEC, curve=curve.in_quad
                )
                duration = _CONTENT_FADE_OUT_SEC
            else:
                camera.overlay.animate_color(
                    color.clear, duration=_CONTENT_FADE_IN_SEC, curve=curve.out_quad
                )
                duration = _CONTENT_FADE_IN_SEC
            invoke(director.ack_fade, kind, tid, delay=duration + _FADE_ACK_MARGIN_SEC)

    def _drive_local_fade() -> None:
        """コーナー内ローカルな暗転（フィラー⇄朗読・シーンまたぎ）を描き、描き終えたら報告する。

        番組進行（director.py）の fade_command()/ack_fade() とは別チャンネル。本物の
        コーナー切り替えの暗転と重ねて動かないよう is_steady() のときだけ見る。
        """
        if director is not None and not director.is_steady():
            return
        cmd = state.local_fade
        if cmd is None or cmd == _local_fade_state["handled"]:
            return
        _local_fade_state["handled"] = cmd
        kind, seq = cmd
        if kind == "out":
            camera.overlay.animate_color(
                color.black, duration=_CONTENT_FADE_OUT_SEC, curve=curve.in_quad
            )
            duration = _CONTENT_FADE_OUT_SEC
        else:
            camera.overlay.animate_color(
                color.clear, duration=_CONTENT_FADE_IN_SEC, curve=curve.out_quad
            )
            duration = _CONTENT_FADE_IN_SEC
        invoke(setattr, state, "local_fade_ack", (kind, seq), delay=duration + _FADE_ACK_MARGIN_SEC)

    def _content_line(source_status: dict | None = None) -> str:
        """今アクティブなコンテンツ名と、そのネタ源の状態（デバッグ用）。

        ``CONTENT: rss - no new articles since 07:31`` のように、番組表の解決結果に
        SourceStatus が書いた短い英語のステータスを添える。ログを掘らずに
        「ネタが来ていないだけ」なのかを画面で判断するためのもの。
        """
        if not content:
            return ""
        ac = _active_content()
        if ac is None:
            return "CONTENT: (none)"
        status = (source_status or {}).get(ac.type, "")
        return f"CONTENT: {ac.type} - {status}" if status else f"CONTENT: {ac.type}"

    def _generated_drama_writing_line(writing: bool) -> str:
        """ラジオドラマの自動執筆（§4.7.6）が今まさに子プロセスを走らせているか。

        今アクティブなコンテンツが generated_drama かどうかとは無関係に（他コーナー放送中でも
        裏で執筆は進むため）、常に最下行として独立に出す。"""
        return "DRAMA: Making..." if writing else ""

    def _background_spec() -> str:
        """今の背景。コンテンツ側の指定が優先で、無指定なら [display] の既定値。"""
        ac = _active_content()
        if ac is not None and ac.background:
            return ac.background
        return display_config.background

    # 背景演出。レイヤーは使い回すので、コーナーが変わっても作り直しは起きない。
    background = Background(_background_spec())

    # 左上の情報。1行目は常時 LLM モデル名、2行目にコンテンツ名、3行目に NOW PLAYING（あれば）、
    # 最下行にラジオドラマの裏執筆中フラグ（あれば）。
    _initial_snap = state.snapshot()
    info_text = Text(
        text="\n".join(filter(None, [
            _model_line(_initial_snap["llm_ready"]),
            _content_line(),
            _generated_drama_writing_line(_initial_snap["generated_drama_writing"]),
        ])),
        parent=camera.ui,
        position=window.top_left + (0.02, -0.03),
        origin=(-0.5, 0.5),
        scale=0.82,
        color=_TEXT_SECONDARY,
    )
    info_text.create_background(color=_TEXT_BG)
    _info_shown = [info_text.text]

    # ON AIR / STANDBY ランプは暗転中も見えている必要があるので camera.overlay より手前に置く。
    lamp = Entity(
        model="circle",
        parent=camera.ui,
        position=window.top_right + (-0.07, -0.05),
        z=_OVERLAY_TOP_Z,
        scale=0.028,
        color=color.dark_gray,
    )
    onair_label = Text(
        text="ON AIR",
        parent=camera.ui,
        position=window.top_right + (-0.11, -0.05),
        z=_OVERLAY_TOP_Z,
        origin=(0.5, 0),
        scale=1.0,
        color=_TEXT_PRIMARY,
    )
    _onair_label_shown = ["ON AIR"]

    switching_log = SwitchingLog()

    def update() -> None:
        t = time.time()
        if tracker is not None:
            tracker.tick()  # 位置・サイズを控えるだけ（I/Oは終了時の save() のみ）
        snap = state.snapshot()
        speaker = snap["current_speaker"]
        rms = snap["current_rms"]

        _drive_fades()
        _drive_local_fade()
        switching = director is not None and director.is_switching()
        background.set_spec(_background_spec())
        background.update(t)

        # 前のコーナーを畳んでいる間（締めの朗読〜暗転）は、ひな壇と左上の表示を
        # 据え置く。コーナーは締めを積んだ時点で自分の出演者表示を片付けるので、
        # そのまま描くと締めの最中に全員表示へ崩れて見える。入れ替えは暗転中に行う。
        frozen = (
            director is not None
            and director.in_closing_window()
            and _stage_cache["poses"] is not None
        )

        # 読書コーナー中は表示を出演者（通常2名。biography_reading は最大10名）に
        # 絞って中央へ寄せる。それ以外は「今アクティブな台本の出演者」だけをひな壇へ
        # 並べる（active_cast_ids が空＝起動直後は全員表示のフォールバック）。
        if frozen:
            poses, riser_geoms = _stage_cache["poses"], _stage_cache["risers"]
        else:
            reading_ids = [cid for cid in snap["reading_cast_ids"] if cid in characters]
            if reading_ids:
                poses, riser_geoms = _reading_poses(reading_ids)
            else:
                active_ids = [cid for cid in snap["active_cast_ids"] if cid in characters]
                members = (
                    [cast_by_id[cid] for cid in active_ids] if active_ids else list(cast)
                )
                poses, riser_geoms = _stage_poses(members)
            _stage_cache["poses"], _stage_cache["risers"] = poses, riser_geoms

        for speaker_id, ch in characters.items():
            pose = poses.get(speaker_id)
            ch.set_visible(pose is not None)
            if pose is not None:
                ch.set_pose(*pose)
            ch.animate(t, is_speaking=(speaker == speaker_id), rms=rms)

        for i, riser in enumerate(risers):
            geom = riser_geoms[i] if i < len(riser_geoms) else None
            body, cap = riser["body"], riser["cap"]
            body.enabled = geom is not None
            cap.enabled = geom is not None
            if geom is not None:
                w, h, d = geom["scale"]
                body.scale = (w, h, d)
                body.x = geom["x"]
                body.y = geom["y"]
                body.z = geom["z"]
                cap.scale = (w, 1, d)
                cap.x = geom["x"]
                cap.y = h + _RISER_CAP_LIFT
                cap.z = geom["z"]

        # 名札を各キャラの頭上へ追従（キャラの位置・大きさは上のループで確定済み）。
        half_w = camera.aspect_ratio / 2
        for cid, plate in nameplates.items():
            ch = characters[cid]
            ui = _project_to_ui(ch.label_anchor()) if ch.enabled else None
            if ui is None or abs(ui[0]) > half_w or abs(ui[1]) > 0.5:
                plate.enabled = False
            else:
                plate.enabled = True
                plate.position = (ui[0], ui[1] + _NAMEPLATE_UI_LIFT)
        if switching:
            # 切り替え中（暗転しきって中身を入れ替えている間）は次コーナーの名札が
            # 一瞬映り込まないよう、まとめて消す。
            for plate in nameplates.values():
                plate.enabled = False

        if speaker is not None:
            _subtitle_silence_since[0] = None
        elif _subtitle_silence_since[0] is None:
            _subtitle_silence_since[0] = t
        silence_since = _subtitle_silence_since[0]
        subtitle_timed_out = (
            silence_since is not None
            and t - silence_since >= display_config.subtitle_idle_clear_sec
        )
        wrapped = (
            ""
            if subtitle_timed_out
            else _wrap_subtitle(
                snap["subtitle"],
                display_config.subtitle_wrap,
                display_config.subtitle_max_lines,
            )
        )
        if wrapped != _subtitle_shown[0]:
            if not wrapped.strip():
                # 字幕なし（発話の合間など）。空テキストでも create_background が
                # 小さな黒帯を残すので、字幕ごと非表示にして画面から消す。
                subtitle.enabled = False
                if getattr(subtitle, "background", None):
                    subtitle.background.enabled = False
            else:
                subtitle.enabled = True
                subtitle.text = wrapped
                # 小説の地の文など、1行が subtitle_wrap いっぱいの字幕は画面幅を
                # はみ出すことがある。画面に収まるよう、必要なぶんだけ縮小する
                # （拡大はしない）。日本語は wordwrap が効かないのでここで担保する。
                subtitle.scale = _SUBTITLE_SCALE
                max_w = camera.aspect_ratio * 0.96
                block_w = subtitle.width * subtitle.scale_x
                if block_w > max_w:
                    subtitle.scale *= max_w / block_w
                subtitle.create_background(color=_TEXT_BG)
                # origin x=-0.5 なのでブロックはアンカーから右へ伸びる。
                # アンカーを「最長行の幅の半分」ぶん左へ置くと、行は左そろえのまま
                # ブロック全体が画面中央にそろう。
                subtitle.x = -subtitle.width * subtitle.scale_x / 2
                subtitle.y = _SUBTITLE_ANCHOR_Y
            _subtitle_shown[0] = wrapped
        if switching:
            subtitle.enabled = False
            if getattr(subtitle, "background", None):
                subtitle.background.enabled = False

        # 読書コーナー中は「作品名 / 著者名 / 青空文庫（進捗）」に切り替える（§10.11）。
        if snap["reading_now_playing"]:
            prog = f"　（{snap['reading_progress']}）" if snap["reading_progress"] else ""
            now_playing = f"{snap['reading_now_playing']}{prog}"
        elif music_enabled and snap["now_playing"]:
            station = f"　［{snap['current_station']}］" if snap["current_station"] else ""
            now_playing = f"NOW PLAYING: {snap['now_playing']}{station}"
        elif music_enabled and snap["current_station"]:
            # ICY メタデータを出さない局。せめて今どの局に繋いでいるかは出す。
            now_playing = f"NOW PLAYING: ［{snap['current_station']}］"
        else:
            now_playing = ""
        info = "\n".join(
            filter(None, [
                _model_line(snap["llm_ready"]),
                _content_line(snap["source_status"]),
                now_playing,
                _generated_drama_writing_line(snap["generated_drama_writing"]),
            ])
        )
        if info != _info_shown[0] and not frozen:
            info_text.text = info
            info_text.create_background(color=_TEXT_BG)
            _info_shown[0] = info
        info_text.enabled = not switching
        if getattr(info_text, "background", None):
            info_text.background.enabled = not switching

        if switching:
            lamp.color = color.orange if int(t * _STANDBY_BLINK_HZ) % 2 == 0 else color.dark_gray
            label = _STANDBY_LABEL
        elif not snap["on_air"]:
            lamp.color = color.dark_gray
            label = "ON AIR"
        elif snap["filler_active"]:
            lamp.color = color.orange if int(t * _BLINK_HZ) % 2 == 0 else color.dark_gray
            label = "ON AIR"
        else:
            lamp.color = color.rgba(0.5, 0.1, 0.1, 1.0)  # 赤色
            label = "ON AIR"
        if label != _onair_label_shown[0]:
            onair_label.text = label
            _onair_label_shown[0] = label

        switching_log.set_active(switching)
        switching_log.update(t)

    __main__.update = update

    # 起動フェードイン本体。シーン一式を組み終えた今ここで仕込む（黒画面のまま
    # 数フレーム待ってから始まるので、初回描画のもたつきとフェード開始が被らない）。
    invoke(
        camera.overlay.animate_color,
        color.clear,
        duration=_STARTUP_FADE_SEC,
        curve=curve.in_out_quad,
        delay=_STARTUP_FADE_DELAY,
    )
    # AudioMixer は接続直後のストリーミングノイズを避けるため、この時刻が立つまで
    # 無音で待っている（state.startup_fade_started_at）。画面の明転と同じタイミング・
    # 長さ（_STARTUP_FADE_SEC）で音量も上げてもらう。invoke() は呼び出し時点で
    # 引数を評価してしまうので、実際に発火する時刻を取るにはラムダで包む。
    invoke(
        lambda: setattr(state, "startup_fade_started_at", time.monotonic()),
        delay=_STARTUP_FADE_DELAY,
    )

    app.run()
