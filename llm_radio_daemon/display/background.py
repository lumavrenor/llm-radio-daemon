"""背景演出（ursina）。番組表の [[content]] ごとに切り替わる、暗めの動く背景。

SPEC.md 2.1節「メインループを絶対にブロックしない」に従い、毎フレームの
ジオメトリ再生成は行わない。各レイヤーは初回表示のときに一度だけメッシュを
組み、update() では Entity の position / rotation を数個いじるだけにしてある
（logs / coderain の文字更新も秒単位に間引く）。レイヤーは使い回しで、
切り替えは enabled の付け外しだけ ＝ コーナーが変わっても作り直さない。

種類の語彙は config.BACKGROUND_ATOMS が唯一の出どころ（config.py は ursina を
import できないため、名前だけあちらに置いてある）。増やすときは向こうに1語、
こちらの _LAYER_FACTORIES に1行足す。

いずれのレイヤーも「暗い・低コントラスト・遅い」を守る。長時間つけっぱなしに
する番組なので、目を引く動きや高い彩度はここでは作らない。
"""

from __future__ import annotations

import logging
import math
import random
import time
from collections import deque

from ursina import Entity, Mesh, Text, Vec3, color, scene

from ..config import BACKGROUND_ATOMS, parse_background

# ウィンドウのベース色。GitHub Dark 系の青寄りグレーで、白字幕とのコントラストを
# 確保しつつ黒より目が疲れにくい明度に置いている。
BG_COLOR = color.hex("#0d1117")

# 背景を一括で明るく／暗くしたいときはここだけ触る（各レイヤーの alpha に掛かる）。
_INTENSITY = 1.0


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


# --- grid: 奥から手前へ流れるワイヤーフレームの格子床 --------------------------
#
# 線は「細長い三角形の帯」で描く。ursina の Mesh(mode="line") は線分1本ごとに
# プリミティブが分かれてドローコールが増えるうえ、太さが画面ピクセル固定で
# 遠近が出ない。帯にすればメッシュ1枚・1ドローコールで、奥ほど細くなる。
# 端が唐突に切れないよう、濃さ（頂点アルファ）を奥・手前・左右で落としてある。

_GRID_CELL = 2.0          # 格子の間隔
_GRID_HALF_W = 30.0       # 左右の広がり（±）
_GRID_Z_NEAR = -18.0      # 手前端（カメラは z=-12.8 付近）
_GRID_Z_FAR = 48.0        # 奥端
_GRID_Y = -0.02           # キャラの足元(y=0)のわずかに下
_GRID_LINE_W = 0.035      # 線の太さ（ワールド単位）
_GRID_SPEED = 0.55        # 手前へ流れる速さ（units/sec）。1マス通過に約3.6秒
_GRID_RGB = (0.36, 0.58, 0.85)
_GRID_ALPHA = 0.22
_GRID_X_STEPS = 12        # 横線をx方向に何分割するか（左右フェードの滑らかさ）
_GRID_Z_STEPS = 16        # 縦線をz方向に何分割するか（奥行きフェードの滑らかさ）


def _grid_alpha(x: float, z: float) -> float:
    """格子の濃さ。奥ほど淡く、手前と左右の端はフェードアウトさせる。"""
    far = 1.0 - _clamp01((z - 2.0) / (_GRID_Z_FAR - 2.0))
    near = _clamp01((z - _GRID_Z_NEAR) / 8.0)
    side = 1.0 - _clamp01((abs(x) - 7.0) / (_GRID_HALF_W - 7.0))
    return _GRID_ALPHA * _INTENSITY * far * far * near * side


def _grid_mesh() -> Mesh:
    verts: list[Vec3] = []
    tris: list[tuple[int, int, int]] = []
    cols: list[tuple[float, float, float, float]] = []
    half = _GRID_LINE_W / 2

    def ribbon(points: list[tuple[float, float]], along: str) -> None:
        """中心線 points=[(x, z), ...] を幅 _GRID_LINE_W の帯にして積む。"""
        base = len(verts)
        dx, dz = (0.0, half) if along == "x" else (half, 0.0)
        for x, z in points:
            a = _grid_alpha(x, z)
            verts.append(Vec3(x - dx, _GRID_Y, z - dz))
            verts.append(Vec3(x + dx, _GRID_Y, z + dz))
            cols.append((*_GRID_RGB, a))
            cols.append((*_GRID_RGB, a))
        for i in range(len(points) - 1):
            v0, v1 = base + i * 2, base + i * 2 + 1
            tris.append((v0, v1, v1 + 2))
            tris.append((v0, v1 + 2, v0 + 2))

    n_z = int((_GRID_Z_FAR - _GRID_Z_NEAR) / _GRID_CELL) + 1
    for i in range(n_z):  # 横線（x方向に走る）
        z = _GRID_Z_NEAR + i * _GRID_CELL
        step = (_GRID_HALF_W * 2) / _GRID_X_STEPS
        ribbon([(-_GRID_HALF_W + j * step, z) for j in range(_GRID_X_STEPS + 1)], "x")

    n_x = int((_GRID_HALF_W * 2) / _GRID_CELL) + 1
    for i in range(n_x):  # 縦線（z方向に走る＝消失点へ向かう）
        x = -_GRID_HALF_W + i * _GRID_CELL
        step = (_GRID_Z_FAR - _GRID_Z_NEAR) / _GRID_Z_STEPS
        ribbon([(x, _GRID_Z_NEAR + j * step) for j in range(_GRID_Z_STEPS + 1)], "z")

    return Mesh(vertices=verts, triangles=tris, colors=cols, mode="triangle")


# --- dust: ゆっくり周回する微粒子 ----------------------------------------------
#
# 「上へ流して端で折り返す」方式は折り返しの継ぎ目が必ず見えるので、
# カメラの真下を軸にした極ゆっくりの周回にしている。円は閉じているので
# 継ぎ目が生まれず、シェルごとに速度を変えるだけで視差も出る。
# 点(GeomPoints)なので常にカメラを向き、メッシュ1枚＝1ドローコール。

_DUST_PIVOT_Z = -12.0     # 周回の中心（カメラの足元あたり）
_DUST_RGB = (0.62, 0.72, 0.92)

#            粒数, 内半径, 外半径, 上限y,  濃さ, 点の太さ(px), 角速度(deg/sec)
_DUST_SHELLS = (
    (180, 16.0, 26.0, 14.0, 0.38, 3, 0.90),
    (240, 26.0, 38.0, 18.0, 0.26, 2, 0.55),
    (300, 38.0, 55.0, 22.0, 0.16, 2, 0.32),
)
_DUST_BOB = 0.25          # ふわつきの振幅（ワールド単位）


def _dust_mesh(
    rng: random.Random, count: int, r_in: float, r_out: float, y_max: float,
    alpha: float, thickness: int,
) -> Mesh:
    verts: list[Vec3] = []
    cols: list[tuple[float, float, float, float]] = []
    for _ in range(count):
        theta = rng.uniform(0.0, math.tau)
        # 面積が均等になるよう半径は sqrt 分布。内側に粒が密集するのを防ぐ。
        r = math.sqrt(rng.uniform(r_in**2, r_out**2))
        verts.append(Vec3(math.sin(theta) * r, rng.uniform(0.5, y_max), math.cos(theta) * r))
        cols.append((*_DUST_RGB, alpha * _INTENSITY * rng.uniform(0.45, 1.0)))
    return Mesh(
        vertices=verts,
        colors=cols,
        mode="point",
        thickness=thickness,
        # ピクセル指定にする。透視モード(=ワールド単位)はシェーダ側で
        # gl_PointSize を書かないと効かないため、ここでは使わない。
        render_points_in_3d=False,
    )


# --- logs: daemon 自身のログを薄く流す -----------------------------------------

_LOG_Z = 26.0
_LOG_SCALE = 30           # 3D空間に置くテキストの拡大率（1行 ≒ 0.025 * scale ワールド）
_LOG_LINES = 26
_LOG_COLS = 78
_LOG_REFRESH_SEC = 0.7    # 文字の作り直しは重いので、この間隔でしか更新しない
# switching_log.py がコーナー切り替え中に「2000行前から現在へ」ゆっくりスクロール
# させるための巻き戻し分（_SCROLL_BACK_LINES）を含めた保持行数。
# _LogRing.capacity のデフォルトに使う。
_LOG_RING_CAPACITY = 2500
# 元は (0.50, 0.66, 0.86, 0.15) と青みが強く目立ちすぎたため、彩度・明度・alpha を
# 落として BG_COLOR (#0d1117) に馴染む沈んだ色に調整した。
_LOG_COLOR = color.rgba(0.32, 0.42, 0.55, 0.10)

# コーナー切り替え中の待たせ表示（switching_log.py）が、いつもの背景と同じ沈んだ色味を使うための公開名。
LOG_COLOR = _LOG_COLOR

_LOG_Y = -1.0             # 左上の情報表示と重ならないよう、少しだけ下げた位置に置く
_LOG_IDLE = "…"           # ログが1行も溜まっていないとき


class _LogRing(logging.Handler):
    """直近のログ行をメモリに溜めるだけのハンドラ。logs レイヤーの唯一のデータ源。

    ファイルを tail すると update() の中でディスクI/Oが走ってしまうので、
    root logger にぶら下げて在プロセスで拾う。emit はどのワーカースレッドからも
    呼ばれるが、deque.append / list() は GIL 下で atomic なのでロックは要らない。
    """

    def __init__(self, capacity: int = _LOG_RING_CAPACITY) -> None:
        super().__init__(level=logging.INFO)
        self.lines: deque[str] = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            name = record.name.rsplit(".", 1)[-1][:14]
            # LLM prompt/response ログなどは埋め込みの改行を含む1レコードになりうるが、
            # ここに溜める各行は「1行=1エントリ」前提（switching_log.py が行ごとに
            # 固定位置の Text を割り当てる）。改行を残すと Text が内部で2行に
            # なって次の行の位置へはみ出し、半透明の文字同士が重なって白く見える。
            message = record.getMessage().replace("\n", " ")
            line = f"{time.strftime('%H:%M:%S')} {name:<14} {message}"
            self.lines.append(line[:_LOG_COLS])
        except Exception:  # noqa: BLE001 - 背景演出のために本編を落とさない
            pass


_log_ring: _LogRing | None = None


def _log_source() -> _LogRing:
    """ログ収集ハンドラ。プロセスに1本だけぶら下げる（多重登録＝多重表示を防ぐ）。

    レイヤーの生成時ではなく Background の生成時から回しておく。そうしないと
    logs の時間帯に入った瞬間だけ画面が空っぽになる。
    """
    global _log_ring
    if _log_ring is None:
        _log_ring = _LogRing()
        logging.getLogger().addHandler(_log_ring)
    return _log_ring


def recent_log_lines(n: int) -> list[str]:
    """直近 n 行（古い→新しい順）。3D の logs レイヤーとは別に、UI 側（コーナー
    切り替え中の待たせ表示など）が直接使うための軽量な取り出し口。"""
    return list(_log_source().lines)[-n:]


# --- coderain: Matrix 風の落下文字 ---------------------------------------------
#
# 列ごとに独立した「筋」が落ちて、画面下へ抜けたら上へ戻して文字を引き直す。
# 戻す瞬間は画面外なので継ぎ目は見えない。毎フレーム動かすのは y だけで、
# テキストの再生成は筋が1周したときだけ（14列なら数秒に1回）。

_RAIN_Z = 24.0
_RAIN_SCALE = 26
_RAIN_COLUMNS = 20
_RAIN_HALF_W = 24.0       # 列を配置する左右の幅（±）
_RAIN_LEN = (8, 18)       # 1本の筋の行数レンジ
_RAIN_TOP = 18.0          # ここより上から降りはじめる（画面外）
_RAIN_BOTTOM = -20.0      # 筋の頭がここを切ったら上へ戻す（画面外）
_RAIN_SPEED = (2.2, 4.8)  # 落下速度レンジ（units/sec）
_RAIN_COLOR = color.rgba(0.22, 0.78, 0.45, 0.14)
_RAIN_HEAD_COLOR = color.rgba(0.62, 0.98, 0.74, 0.30)
_RAIN_GLYPHS = "01アイウエオカキクケコサシスセソタチツテトナニヌネノ<>[]{}/\\|=+-*#$&%@_"


# --- レイヤー ------------------------------------------------------------------


class _Layer:
    """1種類の背景。生成は初回表示のときだけで、以後は enabled で出し入れする。"""

    def __init__(self) -> None:
        self.root = Entity(parent=scene, enabled=False)

    def set_enabled(self, value: bool) -> None:
        self.root.enabled = value

    def update(self, t: float) -> None:
        pass


class _GridLayer(_Layer):
    def __init__(self) -> None:
        super().__init__()
        # 帯を上からも下からも見えるようにして、巻き方向を気にしなくて済ませる。
        Entity(parent=self.root, model=_grid_mesh(), double_sided=True)

    def update(self, t: float) -> None:
        # 1マスぶんだけ手前へずらしては戻す。格子は周期的なので継ぎ目は出ない。
        self.root.z = -((t * _GRID_SPEED) % _GRID_CELL)


class _DustLayer(_Layer):
    def __init__(self) -> None:
        super().__init__()
        self.root.z = _DUST_PIVOT_Z
        rng = random.Random(0xD057)  # 見た目を毎回同じにするため種は固定
        self.shells: list[tuple[Entity, float, float]] = []
        for i, (n, r_in, r_out, y_max, a, px, deg) in enumerate(_DUST_SHELLS):
            shell = Entity(
                parent=self.root,
                model=_dust_mesh(rng, n, r_in, r_out, y_max, a, px),
            )
            self.shells.append((shell, deg, 0.13 + i * 0.05))  # (実体, 角速度, ふわつき周波数)

    def update(self, t: float) -> None:
        for shell, deg_per_sec, bob_w in self.shells:
            shell.rotation_y = (t * deg_per_sec) % 360.0
            shell.y = math.sin(t * bob_w) * _DUST_BOB


class _LogsLayer(_Layer):
    def __init__(self) -> None:
        super().__init__()
        self.root.z = _LOG_Z
        self.root.y = _LOG_Y
        self.ring = _log_source()
        self.text = Text(
            parent=self.root,
            text=_LOG_IDLE,
            origin=(0, 0),
            scale=_LOG_SCALE,
            color=_LOG_COLOR,
            # ログ本文の "<" ">" を ursina のカラータグとして解釈されると壊れる。
            use_tags=False,
        )
        self._next_refresh = 0.0
        self._shown = ""

    def update(self, t: float) -> None:
        if t < self._next_refresh:
            return
        self._next_refresh = t + _LOG_REFRESH_SEC
        lines = list(self.ring.lines)[-_LOG_LINES:]
        body = "\n".join(lines) if lines else _LOG_IDLE
        if body != self._shown:
            self.text.text = body
            self._shown = body


class _CodeRainLayer(_Layer):
    def __init__(self) -> None:
        super().__init__()
        self.root.z = _RAIN_Z
        self.rng = random.Random(0xC0DE)
        self.columns: list[dict] = []
        step = (_RAIN_HALF_W * 2) / max(_RAIN_COLUMNS - 1, 1)
        for i in range(_RAIN_COLUMNS):
            x = -_RAIN_HALF_W + i * step
            tail = Text(parent=self.root, text="", x=x, origin=(0, 0.5),
                        scale=_RAIN_SCALE, color=_RAIN_COLOR, use_tags=False)
            head = Text(parent=self.root, text="", x=x, origin=(0, 0.5),
                        scale=_RAIN_SCALE, color=_RAIN_HEAD_COLOR, use_tags=False)
            col = {"tail": tail, "head": head, "y": 0.0, "speed": 0.0, "len": 0}
            self._respawn(col, spread=True)
            self.columns.append(col)

    def _respawn(self, col: dict, spread: bool = False) -> None:
        """筋を画面上へ戻し、文字と長さと速度を引き直す。"""
        rng = self.rng
        col["len"] = rng.randint(*_RAIN_LEN)
        col["speed"] = rng.uniform(*_RAIN_SPEED)
        # 起動直後だけ全部が横一線に並ばないよう、初回は縦位置をばらけさせる。
        col["y"] = rng.uniform(_RAIN_BOTTOM, _RAIN_TOP) if spread else _RAIN_TOP
        col["tail"].text = "\n".join(rng.choice(_RAIN_GLYPHS) for _ in range(col["len"]))
        col["head"].text = rng.choice(_RAIN_GLYPHS)

    def update(self, t: float) -> None:
        dt = min(t - getattr(self, "_last_t", t), 0.25)  # 復帰直後の大ジャンプを抑える
        self._last_t = t
        line_h = 0.025 * _RAIN_SCALE
        for col in self.columns:
            col["y"] -= col["speed"] * dt
            col["tail"].y = col["y"]
            col["head"].y = col["y"] - (col["len"] - 1) * line_h  # 筋の先頭＝いちばん下
            if col["head"].y < _RAIN_BOTTOM:
                self._respawn(col)


# 語彙は config.BACKGROUND_ATOMS 側が持ち主。ここはその名前 -> 実装の対応表。
_LAYER_FACTORIES = {
    "grid": _GridLayer,
    "dust": _DustLayer,
    "logs": _LogsLayer,
    "coderain": _CodeRainLayer,
}

assert set(_LAYER_FACTORIES) == set(BACKGROUND_ATOMS), "背景の語彙と実装がずれている"


class Background:
    """背景レイヤーの束。コンテンツが変わったら set_spec() で切り替える。

    レイヤーは一度作ったら捨てない（切り替えのたびにメッシュを組み直すと
    番組の切り替わりでフレームが飛ぶため）。使う分だけ遅延生成して以後は
    enabled の付け外しだけにしている。
    """

    def __init__(self, spec: str) -> None:
        _log_source()  # logs を使う時間帯に入る前からログを溜めておく
        self._layers: dict[str, _Layer] = {}
        self._active: tuple[str, ...] = ()
        self._spec = ""
        self.set_spec(spec)

    def set_spec(self, spec: str) -> None:
        """"grid+dust" 等で表示するレイヤーを決める。同じ指定なら何もしない。"""
        if spec == self._spec:
            return
        self._spec = spec
        try:
            atoms = parse_background(spec)
        except ValueError:
            # config 読み込み時に検証済みなのでここへは来ない想定。来たら背景なし。
            atoms = ()
        for name in atoms:
            if name not in self._layers:
                self._layers[name] = _LAYER_FACTORIES[name]()
        for name, layer in self._layers.items():
            layer.set_enabled(name in atoms)
        self._active = atoms

    def update(self, t: float) -> None:
        for name in self._active:
            self._layers[name].update(t)
