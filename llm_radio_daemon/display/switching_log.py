"""コーナー切り替え中（暗転しきってから明転が始まるまで）の待たせ表示。

暗転は camera.overlay（UI レイヤーの不透明な黒）なので、3D シーンに置いたものは
全部その下に隠れる。かといって UI レイヤーは正射影で遠近が付かない。そこで専用の
ディスプレイリージョンを UI の1つ上に重ね、メインカメラと同じ透視レンズで
「ログだけの小さなシーン」を描く。黒の上に、奥へ傾いたログの面が浮かぶ形になる。

リージョンは切り替え中だけ有効にする（それ以外は描画コストもゼロ）。
行ごとに別の Text にして、上（古い行）ほど薄く・下（新しい行）ほど濃くする。
"""

from __future__ import annotations

from panda3d.core import Camera as PandaCamera
from panda3d.core import NodePath
from ursina import Entity, Text, application, camera, color

from .background import LOG_COLOR, recent_log_lines

# ログの面の置き場所（カメラ基準・ursina 座標）と傾き。
# rotation_y を正にすると左端が、rotation_x を正にすると上端が奥へ下がる
# ＝ 左上の角がいちばん遠くなる。
_PLANE_POS = (0.8, 2.6, 23.0)
_PLANE_ROT_X = 20.0
_PLANE_ROT_Y = -11.0

_LINES = 34
_SCALE = 13            # 1行の高さ ≒ 0.025 * _SCALE ワールド単位
_LINE_SPACING = 1.35   # 行送り（文字の高さに対する倍率）
_REFRESH_SEC = 0.7     # background._LogsLayer の更新間隔と揃える

# 切り替え中は実ログがほとんど動かず画面が止まって見えるため、表示開始時だけ
# 過去 _SCROLL_BACK_LINES 行まで巻き戻し、以後は現在に向かって1行ずつゆっくり
# 送り出す（追いついたら普段どおり最新行を表示する状態に自然に合流する）。
# コーナー切り替えの時間帯にはログは十分溜まっている想定なので、多めに
# 2000行遡っておけば、この速さ（0.75秒/行 ＝ 追いつくまで最大25分）では
# 切り替え中に本当に追いついてしまうことはまず無い。
_SCROLL_BACK_LINES = 2000
_SCROLL_STEP_SEC = 0.75

# 上端（古い行）→ 下端（新しい行）の濃さ。色相は背景の logs レイヤーと同じ。
_ALPHA_TOP = 0.03
_ALPHA_BOTTOM = 0.34
_ALPHA_CURVE = 1.6     # 1 より大きいほど、下の方だけが濃く残る

# このリージョンは ui_display_region より手前（sort が上）なので、フルスクリーンで
# 取ると右上の ON AIR/STANDBY ランプ（app.py 側で ui レイヤーに描画済み）を
# 毎フレームの黒クリアで塗りつぶしてしまう。ランプは window.top_right +
# (-0.11, -0.05) 付近（画面上端から見て高々7%程度）にしか無いので、この帯を
# リージョンの範囲から外し、ui レイヤーが描いたランプをそのまま透けさせる。
_TOP_RESERVED_FOR_LAMP = 0.10


class SwitchingLog:
    def __init__(self) -> None:
        win = application.base.win
        self._region = win.make_display_region(0, 1, 0, 1 - _TOP_RESERVED_FOR_LAMP)
        self._region.set_sort(camera.ui_display_region.get_sort() + 1)
        # 色クリアをしないと、更新のたびに変わる行の文字が前フレーム分だけ
        # 消えずに残り、薄い文字同士が重なって白っぽく蓄積してしまう。
        # camera.overlay と同じ黒で毎フレーム塗り直してから描く。
        self._region.set_clear_color_active(True)
        self._region.set_clear_color(color.rgba(0, 0, 0, 1))
        self._region.set_clear_depth_active(True)
        self._region.set_active(False)

        self._render = NodePath("switching_log_render")
        self._render.set_depth_test(False)
        self._render.set_depth_write(False)
        cam = NodePath(PandaCamera("switching_log_cam"))
        # メインカメラのレンズを共有する（ウィンドウのリサイズで縦横比も一緒に追従する）。
        cam.node().set_lens(camera.perspective_lens)
        cam.reparent_to(self._render)
        self._region.set_camera(cam)

        self._plane = Entity(
            parent=self._render,
            position=_PLANE_POS,
            rotation_x=_PLANE_ROT_X,
            rotation_y=_PLANE_ROT_Y,
        )
        step = Text.size * _SCALE * _LINE_SPACING
        top = step * (_LINES - 1) / 2
        self._rows: list[Text] = []
        for i in range(_LINES):
            k = (i / (_LINES - 1)) ** _ALPHA_CURVE
            alpha = _ALPHA_TOP + (_ALPHA_BOTTOM - _ALPHA_TOP) * k
            self._rows.append(Text(
                parent=self._plane,
                text=" ",  # 空文字だと一部の内部処理で落ちるため空白で初期化する
                # 左そろえ。面の中心が行の途中に来るよう、左端を少し左へ寄せる。
                position=(-step * 22, top - step * i),
                origin=(-0.5, 0),
                scale=_SCALE,
                color=color.rgba(LOG_COLOR[0], LOG_COLOR[1], LOG_COLOR[2], alpha),
                use_tags=False,  # ログ本文の "<" ">" をカラータグとして解釈させない
            ))
        self._shown: list[str] = [" "] * _LINES
        self._next_refresh = 0.0
        self._active = False
        # 現在の最新行から何行遡った窓を見せているか。0 になったら最新行に追いつく。
        self._back = 0
        # スクロール開始時点の back とその時刻。以後は「経過時間 ÷ 1行あたりの
        # 秒数」で back を直接計算する。_next_refresh ゲート（約 _REFRESH_SEC
        # 間隔でしか中身が実行されない）を挟んで「次はいつ進めるか」を
        # 積み上げ式で予約すると、_SCROLL_STEP_SEC が _REFRESH_SEC に近いときに
        # 2つの独立タイマーの位相がじわじわずれて（うなり）、進みが波打つように
        # 遅くなる。経過時間から毎回計算し直せば、間引きの影響を受けない。
        self._scroll_start_back = 0
        self._scroll_start_t: float | None = None

    def set_active(self, value: bool) -> None:
        if value != self._active:
            self._active = value
            self._region.set_active(value)
            if value:
                self._next_refresh = 0.0  # 出した瞬間に最新のログへ
                # 溜まっている行数が _SCROLL_BACK_LINES に満たないと、後段の
                # クランプで実質ずっと同じ行に張り付いてスクロールが起きない
                # （溜まるまで何分も待つ羽目になる）。開始時点で届く範囲に
                # 収めておき、初手から動き出すようにする。
                available = len(recent_log_lines(_LINES + _SCROLL_BACK_LINES))
                self._scroll_start_back = min(_SCROLL_BACK_LINES, max(0, available - 1))
                self._back = self._scroll_start_back
                self._scroll_start_t = None  # 実際の時刻は次の update() で確定させる

    def update(self, t: float) -> None:
        if not self._active or t < self._next_refresh:
            return
        self._next_refresh = t + _REFRESH_SEC
        if self._scroll_start_t is None:
            self._scroll_start_t = t
        elapsed = t - self._scroll_start_t
        steps = int(elapsed / _SCROLL_STEP_SEC)
        self._back = max(0, self._scroll_start_back - steps)
        # 過去 _back 行ぶん巻き戻した窓を取り、下詰め：窓の最新行がいちばん下
        # （濃い側）に来るようにする。_back が 0 まで下がれば通常どおり最新行。
        # 溜まっている行数が _back に満たない（起動直後など）ときに空の窓を
        # 引いてしまわないよう、実際に取れた行数を上限にして back を頭打ちにする。
        history = recent_log_lines(_LINES + _SCROLL_BACK_LINES)
        back = min(self._back, max(0, len(history) - 1))
        end = len(history) - back
        lines = history[max(0, end - _LINES):end]
        lines = [" "] * (_LINES - len(lines)) + [ln or " " for ln in lines]
        for row, shown, line in zip(self._rows, self._shown, lines):
            if line != shown:
                row.text = line
        self._shown = lines
