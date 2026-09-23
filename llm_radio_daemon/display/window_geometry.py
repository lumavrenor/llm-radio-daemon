"""ウィンドウの位置・サイズを終了時に保存し、次回起動時に復元する。

Panda3D の WindowProperties は3OS共通の抽象化なので、この仕組み自体は
Windows / macOS / Linux(X11) でそのまま動く。OS別の分岐は持たない。
ただし「効き方」には差があり、いずれも復元が効かないだけで動作は壊れない:

  - Linux(Wayland) … アプリからの位置指定は仕様上無視される（サイズだけ復元）
  - macOS(Retina)  … points/pixels の換算差でサイズがずれる可能性がある
  - 最大化状態     … WindowProperties に該当プロパティが無く復元できない
                     （復元されるのは通常サイズのみ。minimized はあるが maximized は無い）

Windows + Panda3D 1.10 + ursina 8.3 の実測では、`window.position` は
「クライアント領域の左上を要求した位置ちょうどに置く」動作で、getter も
同じ座標を返す。set→get の往復はピクセル単位で一致し、ズレは無い。
ユーザーがウィンドウを動かした結果も getter に反映される。
したがってここでは値をそのまま控えて、そのまま書き戻すだけにする。

（以前は「get_origin がクライアント基準・set_origin が外枠基準」という
座標系のズレを起動直後に一度だけ実測して差し引いていたが、実際にはズレが
無いうえ、起動直後にユーザーがウィンドウを少し動かすとその移動量を
「ズレ」と誤認してセッション中ずっと差し引き続け、保存位置が固定される／
毎回ずれるという不具合になっていたので廃止した。）

保存値が今のモニタ構成から外れている（モニタを外した・解像度を変えた）場合や、
タイトルバーが全モニタの上端より上にあって掴めない場合は、黙って既定の
中央配置に戻す。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from ursina import window

logger = logging.getLogger(__name__)

# 復元を受け付けるサイズの範囲。壊れた値や異常値を弾く。
_MIN_SIZE = (320, 240)
_MAX_SIZE = (16384, 16384)

# 位置の許容範囲。Windows は最小化中のウィンドウ位置に -32000 を返すので、
# 最小化したまま終了したときの値をここで落とす。
_MIN_ORIGIN, _MAX_ORIGIN = -30000, 30000

# 復元位置が「どこかのモニタに乗っている」と認めるのに必要な、タイトルバー
# 付近の水平方向の重なり(px)。タイトルバーを掴める程度は見えていること。
_VISIBLE_MARGIN_X = 120

# タイトルバーの想定高さ(px)。この帯がモニタの縦範囲に掛かっていること。
_TITLEBAR_H = 24

# タイトルバー上端がモニタ上端より少しだけ上でも許容する量(px)。
# 最大化状態のウィンドウ（外枠が画面外に -8px ほどはみ出す）を弾かないため。
_TOP_TOLERANCE = 16


@dataclass(frozen=True)
class WindowGeometry:
    x: int
    y: int
    width: int
    height: int


def _is_sane(geom: WindowGeometry) -> bool:
    return (
        _MIN_SIZE[0] <= geom.width <= _MAX_SIZE[0]
        and _MIN_SIZE[1] <= geom.height <= _MAX_SIZE[1]
        and _MIN_ORIGIN <= geom.x <= _MAX_ORIGIN
        and _MIN_ORIGIN <= geom.y <= _MAX_ORIGIN
    )


def _monitor_rects() -> list[tuple[int, int, int, int]]:
    """接続中のモニタ矩形 (x, y, w, h) を返す。取れなければ空リスト。

    ursina が起動時に screeninfo で拾ったものをそのまま借りる（Ursina() の後でのみ有効）。
    """
    rects: list[tuple[int, int, int, int]] = []
    for mon in getattr(window, "monitors", None) or []:
        try:
            rects.append((int(mon.x), int(mon.y), int(mon.width), int(mon.height)))
        except (AttributeError, TypeError, ValueError):
            continue
    return rects


def _is_on_screen(geom: WindowGeometry, rects: list[tuple[int, int, int, int]]) -> bool:
    """タイトルバーを掴める程度にどこかのモニタへ掛かっているか。

    「ウィンドウ全体の重なり」ではなく「タイトルバー帯（上端から _TITLEBAR_H）」で
    判定する。本文が画面に大きく掛かっていても、タイトルバーが全モニタの上端より
    上にあると掴めず詰むため（今回の不具合の再発防止）。
    """
    if not rects:
        return True  # モニタ情報が取れない環境では判定できないので復元を試す
    bar_top, bar_bottom = geom.y, geom.y + _TITLEBAR_H
    for mx, my, mw, mh in rects:
        overlap_w = min(geom.x + geom.width, mx + mw) - max(geom.x, mx)
        if overlap_w < _VISIBLE_MARGIN_X:
            continue
        # タイトルバー帯がモニタの縦範囲に掛かり、かつ上端が画面外へ大きく
        # はみ出していないこと。
        if bar_bottom > my and bar_top >= my - _TOP_TOLERANCE and bar_top < my + mh:
            return True
    return False


def load_geometry(path: str | Path) -> WindowGeometry | None:
    """保存済みのジオメトリを読む。無い・壊れている・値が異常なら None。"""
    file = Path(path)
    try:
        # utf-8-sig: 手で編集したファイルに BOM が付いていても読めるように。
        raw = json.loads(file.read_text(encoding="utf-8-sig"))
        geom = WindowGeometry(
            x=int(raw["x"]), y=int(raw["y"]), width=int(raw["width"]), height=int(raw["height"])
        )
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError, KeyError) as exc:
        logger.warning("Failed to load window position (starting with defaults): %s", exc)
        return None
    if not _is_sane(geom):
        logger.info("Saved window position is out of range, starting with defaults: %s", geom)
        return None
    return geom


def save_geometry(path: str | Path, geom: WindowGeometry) -> None:
    """ジオメトリを書き出す。異常値は書かない（次回の復元を壊さないため）。"""
    if not _is_sane(geom):
        logger.debug("Skipping save because window position is invalid: %s", geom)
        return
    file = Path(path)
    try:
        file.parent.mkdir(parents=True, exist_ok=True)
        tmp = file.with_name(file.name + ".tmp")
        tmp.write_text(
            json.dumps(
                {"x": geom.x, "y": geom.y, "width": geom.width, "height": geom.height},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        tmp.replace(file)
    except OSError as exc:
        logger.warning("Failed to save window position: %s", exc)


class GeometryTracker:
    """現在の位置・サイズを毎フレーム控え、終了時に一度だけ書き出す。

    tick() は update() から呼ばれるのでI/Oを一切しない（SPEC 2.1: メインループを
    ブロックしない）。実際の書き込みは save() で、atexit から呼ぶ。
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._latest: WindowGeometry | None = None
        self._saved = False

    def tick(self) -> None:
        try:
            pos, size = window.position, window.size
            x, y = int(pos[0]), int(pos[1])
            w, h = int(size[0]), int(size[1])
        except (AttributeError, TypeError, ValueError, IndexError):
            return  # ウィンドウが閉じかけている等。次のフレームで拾い直す

        geom = WindowGeometry(x, y, w, h)
        if not _is_sane(geom):
            # 最小化中（Windows は位置に -32000 を返す）など。直前の正常値を保持する。
            return
        self._latest = geom

    def save(self) -> None:
        if self._saved or self._latest is None:
            return
        self._saved = True
        save_geometry(self._path, self._latest)


def start_tracking(path: str | Path, restored: WindowGeometry | None) -> GeometryTracker:
    """復元位置をウィンドウへ適用し、以後の追跡を始める。Ursina() の直後に呼ぶ。

    サイズは Ursina(size=...) 側で復元済み。ここで扱うのは位置だけで、
    復元値がどのモニタにも掛からない（またはタイトルバーが掴めない位置の）
    ときは ursina が既に行った中央配置のままにする。
    """
    if restored is not None:
        if _is_on_screen(restored, _monitor_rects()):
            window.position = (restored.x, restored.y)
        else:
            logger.info(
                "Saved window position is off-screen or not grabbable, centering instead: %s", restored
            )
    return GeometryTracker(path)
