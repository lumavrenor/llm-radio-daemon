"""poly_character.Character を1体だけ表示する、アイコン撮影用のテストコード。

    python -m llm_radio_daemon.display.test_icon

- GitHub 等で使うアイコン画像を作るために、キャストを1人だけ画面中央に大きく
  表示する（test_character.py はパラメータ比較のため複数体を並べる別物）。
- 既定は司会の "rin"（葛西りん）の見た目。環境変数 LRD_ICON_CAST=<id> で
  他のキャストに差し替えられる（下の _PRESETS 参照。config/ja/config_cast.toml
  から見た目パラメータだけ抜き出したもの。フルの config.py 読み込み〈TTS解決等〉は
  重いのでここでは行わない）。
- 向き（方向）は左右矢印キーで変えられる。poly_character 側に facing の概念は
  無いので、このテストコードで Character.rotation_y を直接回している。
    ←/→        … 回転（長押しで連続回転）
    Q/E        … 微調整（押すたびに5度）
    Space      … スクリーンショット保存（LRD_ICON_OUT で保存先を指定、既定 icon_shot.png）
  マウスドラッグでカメラも動かせる（EditorCamera。左＝回転 / 右＝パン / ホイール＝ズーム）。
- 環境変数 LRD_TEST_SECONDS / LRD_TEST_SHOT を設定すると、その秒数で自動的に
  スクリーンショットを撮って終了する（test_character.py と同じ規約。CI /
  スクリプトからの自動撮影用）。
  例（PowerShell、rin を30度回して1枚保存して終了）:
    $env:LRD_ICON_ROTATION="30"; $env:LRD_TEST_SECONDS="1"; $env:LRD_TEST_SHOT="icon.png"; \
    python -m llm_radio_daemon.display.test_icon
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
    held_keys,
    invoke,
    window,
)

from .poly_character import Character

# config/ja/config_cast.toml から見た目パラメータだけ抜き出したプリセット。
_PRESETS: dict[str, dict] = {
    "rin": dict(
        model="girl", hair_style="bob",
        hair_color="#562f12", eye_color="#3f2717", dress_color="#c89d53",
        accessories=[("pin", "#c89d53", "right")],
    ),
    "aoi": dict(
        model="girl", hair_style="medium_twintail",
        hair_color="#a57262", eye_color="#41261d", dress_color="#7cb0ac",
    ),
    "tsubasa": dict(
        model="boy", hair_style="cap",
        hair_color="#c8ccd4", eye_color="#010101", dress_color="#c0c0c0",
        accessories=[("headphones", "#e8eaee", "")],
    ),
    "shizuka": dict(
        model="girl", hair_style="blunt_bangs",
        hair_color="#181a23", eye_color="#2c2d34", dress_color="#5e5b97",
        accessories=[("pin", "#5e5b97", "left")],
    ),
}


def _hex(value: str) -> color.Color:
    r, g, b = (int(value[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return color.rgb(r, g, b)


def _save_screenshot(path: str) -> None:
    from panda3d.core import Filename

    application.base.win.save_screenshot(Filename.from_os_specific(path))
    print(f"screenshot saved: {path}")


def main() -> None:
    # development_mode=False で右上の FPS/entities デバッグ表示と閉じるボタンの
    # 赤いXを消す（アイコン用スクリーンショットに写り込ませないため。app.py と同じ設定）。
    app = Ursina(
        title="cast icon shot", borderless=False, size=(900, 900), vsync=True,
        development_mode=False,
    )
    window.color = color.rgb(0.0, 0.0, 0.0)

    DirectionalLight(rotation_x=30, rotation_y=-45, color=color.rgb(0.5, 0.48, 0.45))
    AmbientLight(color=color.rgba(1, 1, 1, 0.22))

    cast_id = os.environ.get("LRD_ICON_CAST", "rin")
    preset = _PRESETS.get(cast_id)
    if preset is None:
        print(f"LRD_ICON_CAST={cast_id!r}: not found. candidates: {list(_PRESETS)}")
        preset = _PRESETS["rin"]

    accessories = [
        (t, _hex(c) if c else None, side) for t, c, side in preset.get("accessories", [])
    ]
    char = Character(
        cast_id,
        model=preset["model"],
        hair_style=preset["hair_style"],
        hair_color=_hex(preset["hair_color"]),
        eye_color=_hex(preset["eye_color"]),
        dress_color=_hex(preset["dress_color"]),
        accessories=accessories,
    )
    char.rotation_y = float(os.environ.get("LRD_ICON_ROTATION", "0") or "0")

    camera.position = (0, 1.3, -3.6)
    camera.rotation_x = 6
    camera.fov = 40
    EditorCamera(rotation_speed=200, panning_speed=6)

    angle_label = Text(
        text="", position=(-0.85, -0.46), origin=(-0.5, 0), scale=0.7, color=color.light_gray
    )

    def capture(path: str) -> None:
        """操作ヒントのテキストを1フレーム消してから撮影する（アイコンに写り込ませないため）。
        save_screenshot() は直前に描画済みのバッファを保存するため、非表示にした直後に
        同期で呼んでも間に合わない＝ invoke で1フレーム後にずらす必要がある（実測で確認済み）。
        """
        angle_label.enabled = False

        def _shoot() -> None:
            _save_screenshot(path)
            angle_label.enabled = True

        invoke(_shoot, delay=0.05)

    def update() -> None:
        t = time.time()
        char.animate(t, is_speaking=False, rms=0.0)

        rot_speed = 90.0  # 度/秒（矢印キー長押し）
        if held_keys["left arrow"]:
            char.rotation_y -= rot_speed * time.dt
        if held_keys["right arrow"]:
            char.rotation_y += rot_speed * time.dt

        angle_label.text = (
            f"{cast_id}  facing: {char.rotation_y % 360:.0f} deg"
            "  (arrows/Q,E to rotate, space to save)"
        )

    def input(key: str) -> None:  # noqa: A001 - ursina が __main__.input をこの名前で探す
        if key == "q":
            char.rotation_y -= 5
        elif key == "e":
            char.rotation_y += 5
        elif key == "space":
            capture(os.environ.get("LRD_ICON_OUT", "icon_shot.png"))

    import __main__

    __main__.update = update
    __main__.input = input

    # LRD_TEST_SECONDS / LRD_TEST_SHOT: 自動終了・自動撮影（test_character.py と同じ規約）。
    limit = float(os.environ.get("LRD_TEST_SECONDS", "0") or "0")
    shot_path = os.environ.get("LRD_TEST_SHOT", "")
    if limit > 0:
        start = time.time()
        state = {"shot": False}

        def _auto_quit() -> None:
            elapsed = time.time() - start
            if shot_path and not state["shot"] and elapsed > limit * 0.6:
                capture(shot_path)
                state["shot"] = True
            if elapsed > limit:
                application.quit()

        Entity(update=_auto_quit)

    app.run()


if __name__ == "__main__":
    main()
