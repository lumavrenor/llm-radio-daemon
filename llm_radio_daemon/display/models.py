"""キャラクターモデル種別の登録表（ursina 非依存の純データ）。

`[[cast]]` の `model = "..."` がここのキーを指す。`poly_character.Character` が
このプリセットを読んでメッシュ構成（髪型・胴体）を決める。config.py も
キーワードの妥当性チェックにこの表を使う（＝モデル一覧の唯一の出どころ）。

今後キャラの種類を増やすときは、この表に1行足すだけでよい
（config.py も app.py も poly_character 側の分岐追加以外は変更不要）。
このモジュールは ursina を import しない — config.py から安全に読めるようにするため。
"""

from __future__ import annotations

DEFAULT_MODEL = "girl"

# model キーワード -> poly_character.Character のメッシュ構成パラメータ（既定値）。
#   hair_style: HAIR_STYLES のいずれか（既定は "bob"）
#   body_style: "dress"（ワンピース円錐台） / "box"（箱型胴体）
# [[cast]] の hair_style を明示すれば、ここの既定値を個別に上書きできる
# （poly_character.Character が hair_style 引数をそのまま優先するため）。
MODEL_PRESETS: dict[str, dict[str, str]] = {
    "girl": {"hair_style": "bob", "body_style": "dress"},  # 女子・ボブ・ワンピース
    "boy": {"hair_style": "cap", "body_style": "box"},     # 男子・短髪・箱型
}

# [[cast]] の hair_style で選べる髪型。poly_character.py 側の生成関数と対応。
#   ショート: bob（ボブ） / bear_buns（熊ダンゴ） /
#            blunt_bangs（前髪パッツン）/ cap（男子の浅い短髪。既存）
#   ミディアム: medium（ノーマル） / medium_twintail（ツインテール）
# ※ side_part / medium_side_part（サイド分け）は分け目メッシュの生成が難しく 09-04 に廃止。
HAIR_STYLES: tuple[str, ...] = (
    "bob",
    "bear_buns",
    "blunt_bangs",
    "cap",
    "medium",
    "medium_twintail",
)


# [[cast]] の accessory = [{ type, color, side }] で頭に足せる装飾品。
# 髪飾りに限らない想定（イヤリング等も将来ここに足す）。poly_character.py 側に
# 対応する _make_<type>() があり、self.head の子として生成する（首かしげに追従）。
#   headphones … 頭頂をまたぐバンド＋左右のイヤーカップ
#   cat_ears   … 頭頂の左右に立つ三角の猫耳
#   pin        … 前髪脇に留める X 字のヘアピン（side で左右／両方を選ぶ）
#   hairband   … 頭を前から耳の上を通ってまたぐ、イヤーカップ無しの細いカチューシャ
# color 省略時は poly_character 側の既定（cat_ears/pin/hairband は髪色、headphones は暗色）。
# side 省略時は type ごとの自然な既定（headphones/cat_ears は "both"、pin は "left"）。
# hairband は side を取らない（常に両耳の上を通る一本）。
ACCESSORY_TYPES: tuple[str, ...] = (
    "headphones",
    "cat_ears",
    "pin",
    "hairband",
)

# side は「画面から見た」左右（[[cast]] の腕 l/r と同じ約束＝ l は -X 側）。
ACCESSORY_SIDES: tuple[str, ...] = ("left", "right", "both")


def known_models() -> tuple[str, ...]:
    return tuple(MODEL_PRESETS)


def known_hair_styles() -> tuple[str, ...]:
    return HAIR_STYLES


def known_accessories() -> tuple[str, ...]:
    return ACCESSORY_TYPES


def known_accessory_sides() -> tuple[str, ...]:
    return ACCESSORY_SIDES


def resolve_model(model: str) -> dict[str, str]:
    """model キーワード -> メッシュ構成。未知キーは既定モデルへフォールバックする
    （設定ミスで 3D 表示が落ちないように。妥当性は config 側で先に弾く）。"""
    return MODEL_PRESETS.get(model, MODEL_PRESETS[DEFAULT_MODEL])
