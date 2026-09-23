"""poly_character.Character の単体表示確認（ポリゴン表示だけ）。

    python -m llm_radio_daemon.display.test_character

- 中央にグレー単色のキャラを1体、その左右に hair_color / dress_color を
  変えた2体を並べて、パラメータ化が効いているか見せる。
- マウスドラッグ（左＝回転 / 右＝パン / ホイール＝ズーム）で視点を回せる。
- 環境変数 LRD_TEST_SECONDS を設定すると、その秒数で自動終了する
  （CI / スモークテスト用。未設定なら手動で閉じるまで表示し続ける）。
- 環境変数 LRD_TEST_FOCUS=<speaker_id> で、その1体の顔まわりへ最初からカメラを
  寄せる（例: hairband の位置確認なら "hairband"。候補は hair_style 名 / accessory
  種別名。見つからないと候補一覧が標準出力に出る）。
- 環境変数 LRD_TEST_SHOT=<path> と LRD_TEST_SECONDS を併用すると、終了直前に
  スクリーンショットを保存する（3D描画を直接見られない環境からの確認用）。
  例（PowerShell、hairband に寄って1枚保存）:
    $env:LRD_TEST_FOCUS="hairband"; $env:LRD_TEST_SECONDS="2"; $env:LRD_TEST_SHOT="hairband.png"; \
    python -m llm_radio_daemon.display.test_character
"""

from __future__ import annotations

import os
import time

from ursina import (
    AmbientLight,
    DirectionalLight,
    EditorCamera,
    Entity,
    Text,
    Ursina,
    application,
    camera,
    color,
    scene,
)

from .poly_character import Character


def main() -> None:
    app = Ursina(title="poly-character test", borderless=False, size=(1100, 720), vsync=True)

    window_bg = color.rgb(0.16, 0.16, 0.18)
    from ursina import window

    window.color = window_bg

    # 床
    Entity(model="plane", scale=30, color=color.rgb(0.28, 0.28, 0.3), y=0)

    # ライト（フラットシェーディングの陰影を出すため。unlit にしない）
    # 参考画像のクレイ風に寄せて、やや強めの斜め光＋控えめの環境光で
    # パーツ間の陰影差（＝立体感）を出す。
    # 注：ursina の DirectionalLight は default_values の rotation_x=90（真上向き）が
    # rotation=(x,y,z) タプル指定より後勝ちして常に90を上書きしてしまうため、
    # rotation_x= / rotation_y= を個別に渡す必要がある（実測で確認済み）。
    DirectionalLight(rotation_x=50, rotation_y=-35, color=color.rgb(1.0, 0.98, 0.95))
    AmbientLight(color=color.rgba(1, 1, 1, 0.22))

    # 中央2体：参考画像どおりグレー系（クレイ風）で model="girl" と "boy" を並べ、
    # モデル種別キーワードの切り替えを見せる。
    Character(
        model="girl",
        hair_color=color.rgb(0.66, 0.66, 0.67),
        skin_color=color.rgb(0.8, 0.8, 0.81),
        dress_color=color.rgb(0.86, 0.86, 0.87),
        x=-1.1,
    )
    Character(
        model="boy",
        hair_color=color.rgb(0.5, 0.5, 0.52),
        skin_color=color.rgb(0.8, 0.8, 0.81),
        dress_color=color.rgb(0.82, 0.82, 0.84),
        x=1.1,
    )

    # 左右：色違いでパラメータ化の確認（左=girl / 右=boy）
    Character(
        model="girl",
        hair_color=color.rgb(0.35, 0.22, 0.16),
        skin_color=color.rgb(0.95, 0.82, 0.72),
        dress_color=color.rgb(0.3, 0.5, 0.75),
        x=-3.4,
    )
    Character(
        model="boy",
        hair_color=color.rgb(0.1, 0.1, 0.12),
        skin_color=color.rgb(0.98, 0.88, 0.8),
        dress_color=color.rgb(0.35, 0.4, 0.45),
        x=3.4,
        arm_raise=55,
    )

    for label, x in (
        ("gray / bob+dress", -1.1),
        ("gray / cap+box", 1.1),
        ("brown / blue (f)", -3.4),
        ("black / gray (m) + arm", 3.4),
    ):
        Text(text=label, position=(x * 0.07 - 0.18, -0.42), origin=(0, 0), scale=0.6, color=color.light_gray)

    # 奥の列：hair_style 6種の見た目確認（09-04 髪型バリエーション検討）。
    # 全員 model="girl"（dress 胴体）に揃え、hair_style だけを変えて比較する
    # （cap も dress 胴体に強制できることの確認を兼ねる。素の boy は手前の列にある）。
    # 個別ラベルは camera.ui 投影の計算が面倒なので付けない。並び順は下の1行キャプション
    # （左から順）で示す。
    hair_styles = (
        "bob", "bear_buns", "blunt_bangs",
        "cap", "medium", "medium_twintail",
    )
    # 実際の app.py 後列と同じ y/z（台の上、_BACK_Y=1.5 / _BACK_Z=3.2）に載せて、
    # 手前の4体（床）と画面上でも高さが分かれるようにする。
    hs_step = 2.6
    hs_x0 = -hs_step * (len(hair_styles) - 1) / 2
    for i, hs in enumerate(hair_styles):
        Character(
            speaker_id=hs,  # LRD_TEST_FOCUS で個別にズームするための識別子
            model="girl",
            hair_style=hs,
            hair_color=color.rgb(0.3, 0.18, 0.14),
            skin_color=color.rgb(0.92, 0.8, 0.7),
            dress_color=color.rgb(0.5, 0.35, 0.65),
            eye_color=color.rgb(0.15, 0.1, 0.3),
            x=hs_x0 + i * hs_step,
            y=1.5,
            z=3.2,
            scale=0.82,
        )

    Text(
        text="back row (left to right): " + " / ".join(hair_styles),
        position=(-0.85, 0.46),
        origin=(-0.5, 0),
        scale=0.65,
        color=color.light_gray,
    )

    # さらに奥の列：accessory 4種の見た目確認（headphones / cat_ears / pin / hairband）。
    acc_specs = (
        ("headphones", [("headphones", color.rgb(0.48, 0.69, 0.85), None)]),
        ("cat_ears", [("cat_ears", color.rgb(0.48, 0.69, 0.85), None)]),
        ("pin (left)", [("pin", color.white, "left")]),
        ("hairband", [("hairband", color.rgb(0.75, 0.86, 0.93), None)]),
    )
    acc_step = 2.6
    acc_x0 = -acc_step * (len(acc_specs) - 1) / 2
    for i, (_, specs) in enumerate(acc_specs):
        Character(
            speaker_id=specs[0][0],  # LRD_TEST_FOCUS で個別にズームするための識別子（例: "hairband"）
            model="girl",
            hair_style="bob",
            hair_color=color.rgb(0.3, 0.18, 0.14),
            skin_color=color.rgb(0.92, 0.8, 0.7),
            dress_color=color.rgb(0.5, 0.35, 0.65),
            eye_color=color.rgb(0.15, 0.1, 0.3),
            accessories=specs,
            x=acc_x0 + i * acc_step,
            y=3.0,
            z=6.0,
            scale=0.82,
        )
    Text(
        text="far row (left to right): " + " / ".join(n for n, _ in acc_specs),
        position=(-0.85, 0.40),
        origin=(-0.5, 0),
        scale=0.65,
        color=color.light_gray,
    )

    camera.position = (0, 3.0, -14.5)
    camera.rotation_x = 16
    camera.fov = 58
    editor_camera = EditorCamera(rotation_speed=200, panning_speed=6)

    chars = [e for e in scene.entities if isinstance(e, Character)]

    # LRD_TEST_FOCUS=<speaker_id> で、その1体の顔まわりへカメラを寄せる
    # （hairband の位置合わせなど、引きの全景だけでは細部が見えない確認用）。
    # 見つからなければ候補一覧を出して手動の EditorCamera 操作に任せる。
    focus = os.environ.get("LRD_TEST_FOCUS", "")
    if focus:
        target = next((c for c in chars if c.speaker_id == focus), None)
        if target is None:
            ids = [c.speaker_id for c in chars if c.speaker_id]
            print(f"LRD_TEST_FOCUS={focus!r}: not found. candidates: {ids}")
        else:
            head_y = target.y + 1.62 * target.scale_y  # Character.__init__ の head_y と同じ
            camera.position = (target.x, head_y + 0.45 * target.scale_y, target.z - 3.4)
            camera.rotation_x = 5
            camera.fov = 32
            # EditorCamera.update() は毎フレーム camera.z を「作成時点の引きの距離」
            # (target_z) へ戻そうと lerp し続ける。target_z も合わせて書き換えないと
            # 数フレームでズームが解けて元の引きの画角に戻ってしまう（実測で確認済み）。
            editor_camera.target_z = camera.z

    def update() -> None:
        t = time.time()
        for i, c in enumerate(chars):
            speaking = int(t) % (len(chars) + 1) == i
            c.animate(t, is_speaking=speaking, rms=0.6 if speaking else 0.0)

    import __main__

    __main__.update = update

    # LRD_TEST_SECONDS 秒後に自動終了。LRD_TEST_SHOT を指定するとその直前に
    # スクリーンショットを保存する（3D描画を目視できない環境での確認用）。
    limit = float(os.environ.get("LRD_TEST_SECONDS", "0") or "0")
    shot_path = os.environ.get("LRD_TEST_SHOT", "")
    if limit > 0:
        start = time.time()
        state = {"shot": False}

        def _auto_quit() -> None:
            elapsed = time.time() - start
            if shot_path and not state["shot"] and elapsed > limit * 0.6:
                from panda3d.core import Filename

                application.base.win.save_screenshot(Filename.from_os_specific(shot_path))
                state["shot"] = True
                print(f"screenshot saved: {shot_path}")
            if elapsed > limit:
                application.quit()

        Entity(update=_auto_quit)

    app.run()


if __name__ == "__main__":
    main()
