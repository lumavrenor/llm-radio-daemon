"""プロシージャル・キャラクターメッシュ（Ursina プリミティブ + カスタムMesh）。

docs/lrd-poly-cast.md の指示に沿った実装。外部モデルファイル（.glb/.vrm）に
依存せず、コードだけでクレイ風のキャラクター1体を組み立てる。

- 低ポリ・フラットシェーディング（法線を活かした陰影は出す。unlit ではない）
- 色は単色塗り分け（テクスチャ画像は使わない）
- 髪ドームだけ「前面をカットした球」をカスタム Mesh で生成し、他はすべて
  Ursina 標準プリミティブ / 手続きモデル（sphere / cube / Cylinder / Pipe）

SPEC.md 4.6 節との整合:
- 全パーツは __init__ で1回だけ生成し、update 相当では既存 Entity の
  scale / rotation / position を書き換えるだけ（メモリリーク対策）。
- 口・目は独立 Entity（`self.mouth` / `self.eye_l` / `self.eye_r`）。
  リップシンクは `self.mouth.scale_y`、まばたきは `self.eye_*.scale_y` で駆動。

display/app.py のひな壇レイアウトから直接生成する（旧・簡易 character.py を置換）。
app.py 側が使う API（`speaker_id` / 名札 / `home` / `set_pose` / `set_visible` /
`update`）を旧 Character と同じシグネチャで提供する。単体確認は
display/test_character.py（`animate()` を直接叩く）。
"""

from __future__ import annotations

import math
import random
import time

from ursina import Entity, Mesh, Vec3, color
from ursina.shader import Shader
from ursina.shaders import lit_with_shadows_shader

from .models import DEFAULT_MODEL, resolve_model

# --- 髪の裏面（前面カットで開いた球の内側）を、表面よりやや暗い単色で塗る -------
# 髪メッシュは前を大きく切り欠いた「開いた」形状なので、標準のバックフェイス
# カリングのままだと裏側が完全に透けて背景が見えてしまう。lit_with_shadows_shader
# のフラグメント側だけ fork し、gl_FrontFacing で裏面を検出して暗くする。
# double_sided=True と組み合わせて使う（そうしないと裏面自体が描画されない）。
_hair_frag = lit_with_shadows_shader.fragment.replace(
    "uniform vec4 shadow_color;",
    "uniform vec4 shadow_color;\nuniform float backface_darken;",
).replace(
    "fragment_color = cast_shadows(fragment_color);",
    "fragment_color = cast_shadows(fragment_color);\n\n"
    "    if (!gl_FrontFacing) {\n"
    "        fragment_color.rgb *= backface_darken;\n"
    "    }",
)
hair_shader = Shader(
    language=Shader.GLSL,
    name="hair_backface_darken_shader",
    vertex=lit_with_shadows_shader.vertex,
    fragment=_hair_frag,
    default_input={
        **lit_with_shadows_shader.default_input,
        "backface_darken": 0.7,
    },
)
hair_shader.continuous_input["camera_world_position"] = (
    lit_with_shadows_shader.continuous_input["camera_world_position"]
)

# --- 前面をカットした球（ヘルメット状の髪）のメッシュ生成 -----------------------

# キャラクターの「正面」は -Z 側（既定カメラが -Z から原点を見るため。
# 目・口もこちら側 z<0 に置く）。


def _uv_sphere(radius: float, rings: int, sectors: int) -> tuple[list[Vec3], list[tuple[int, int, int]]]:
    verts: list[Vec3] = []
    for r in range(rings + 1):
        phi = math.pi * r / rings  # 0=天頂, pi=真下
        for s in range(sectors + 1):
            theta = 2 * math.pi * s / sectors
            verts.append(
                Vec3(
                    radius * math.sin(phi) * math.cos(theta),
                    radius * math.cos(phi),
                    radius * math.sin(phi) * math.sin(theta),
                )
            )

    def vid(r: int, s: int) -> int:
        return r * (sectors + 1) + s

    tris: list[tuple[int, int, int]] = []
    for r in range(rings):
        for s in range(sectors):
            a, b, c, d = vid(r, s), vid(r + 1, s), vid(r + 1, s + 1), vid(r, s + 1)
            # 各クワッドを2三角形に割るとき、対角線の向きを x=0 面について
            # 鏡像対称にする。すべて同じ向き (a-c) で割ると、フラットシェーディング下で
            # 対角エッジが片側だけ「斜めの面」として浮き（頭頂の右上だけ斜めに切れて
            # 見える原因）、左右で見た目が食い違う。クワッド中心の経度で +x / -x を
            # 判定し、-x 側だけ逆対角 (b-d) にすると左右が揃う。
            theta_mid = 2 * math.pi * (s + 0.5) / sectors
            if math.cos(theta_mid) >= 0:
                tris.append((a, b, c))
                tris.append((a, c, d))
            else:
                tris.append((a, b, d))
                tris.append((b, c, d))
    return verts, tris


def _tube_mesh(bottom_r: float, top_r: float, height: float, segments: int = 14) -> Mesh:
    """縦軸(+Y)に沿った円錐台/円柱。y=0 が底面、y=height が上面。

    側面はフラットシェーディング用に四角形ごとの独立頂点で張り、上下に蓋をつける。
    縮退三角形は作らないので generate_normals(smooth=False) が安全に通る。
    """
    verts: list[Vec3] = []
    tris: list[tuple[int, int, int]] = []

    def ring(radius: float, y: float) -> list[Vec3]:
        pts = []
        for s in range(segments):
            a = 2 * math.pi * s / segments
            pts.append(Vec3(radius * math.cos(a), y, radius * math.sin(a)))
        return pts

    lo = ring(bottom_r, 0.0)
    hi = ring(top_r, height)

    # 側面（四角形を2三角形で。頂点は共有せず face 単位）
    for s in range(segments):
        n = (s + 1) % segments
        base = len(verts)
        verts.extend([lo[s], lo[n], hi[n], hi[s]])
        tris.append((base, base + 1, base + 2))
        tris.append((base, base + 2, base + 3))

    # 底面（-Y を向く扇）と上面（+Y を向く扇）
    cb = len(verts)
    verts.append(Vec3(0, 0.0, 0))
    for s in range(segments):
        n = (s + 1) % segments
        verts.extend([lo[s], lo[n]])
        b = cb + 1 + s * 2
        tris.append((cb, b + 1, b))
    ct = len(verts)
    verts.append(Vec3(0, height, 0))
    for s in range(segments):
        n = (s + 1) % segments
        verts.extend([hi[s], hi[n]])
        b = ct + 1 + s * 2
        tris.append((ct, b, b + 1))

    m = Mesh(vertices=verts, triangles=tris, mode="triangle")
    m.generate_normals(smooth=False)
    return m


def _clip_sphere_mesh(radius: float, rings: int, sectors: int, drop, lower_stretch: float = 1.0) -> Mesh:
    """UV球を張り、drop(v) が True になる頂点を含む三角形を落とした Mesh。

    ursina の generate_normals は smooth=False でも「頂点が属する全三角形の法線を
    平均する」処理を必ず先に行う（smooth はその後の"座標が同じ頂点をさらに束ねて
    平均するか"の追加処理でしかない）。そのため UV球のように頂点を隣接三角形と
    共有したままだと、rings/sectors をいくら減らしても常に滑らかな陰影になって
    しまう。_tube_mesh 同様、面ごとに頂点を複製して共有をなくし、
    generate_normals(smooth=False) が本当にフラットシェーディングになるようにする
    （複製するので未参照頂点や 0除算の心配もそもそも無くなる）。

    極付近の縮退三角形（面積ゼロ→法線が nan）は除外する。

    lower_stretch: 1.0 より大きいと中心(y=0)より下側だけを縦に伸ばす（上半分＝頭頂は
    そのまま、下半分＝後頭部・サイドだけ肩まで伸びる）。ミディアム系の髪の長さ調整用。
    drop() にはこの伸ばし後の座標が渡る。
    """
    verts, tris = _uv_sphere(radius, rings, sectors)
    if lower_stretch != 1.0:
        verts = [Vec3(v.x, v.y if v.y >= 0 else v.y * lower_stretch, v.z) for v in verts]

    packed_verts: list[Vec3] = []
    packed_tris: list[tuple[int, int, int]] = []
    for tri in tris:
        p, q, r = (verts[i] for i in tri)
        ux, uy, uz = q.x - p.x, q.y - p.y, q.z - p.z
        vx, vy, vz = r.x - p.x, r.y - p.y, r.z - p.z
        nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        if nx * nx + ny * ny + nz * nz < 1e-18:
            continue
        if drop(p) or drop(q) or drop(r):
            continue
        base = len(packed_verts)
        packed_verts += [p, q, r]
        packed_tris.append((base, base + 1, base + 2))

    m = Mesh(vertices=packed_verts, triangles=packed_tris, mode="triangle")
    m.generate_normals(smooth=False)  # フラットシェーディング
    return m


def _low_poly_sphere_mesh(rings: int = 8, sectors: int = 10) -> Mesh:
    """低ポリの単位球（半径1・フラットシェーディング）。Entity 側の scale で
    好きな大きさ・扁平率に伸ばして使う。

    ursina 標準の "sphere" プリミティブは分割数が多くなめらかすぎ、お団子や
    イヤーカップのような低ポリパーツの上に乗せると質感だけ浮いて見える
    （そこだけ丸くツルッとして見える）ため、代わりにこちらを使う。
    """
    return _clip_sphere_mesh(1.0, rings=rings, sectors=sectors, drop=lambda v: False)


def _helmet_hair_mesh(
    radius: float,
    front_cut: float,
    side_x: float = 0.44,
    lower_stretch: float = 1.0,
    rings: int = 14,
    sectors: int = 16,
) -> Mesh:
    """上・後ろ・左右を覆い、前は生え際から下を大きく開けたヘルメット/ボブ髪（女子）。

    小さな窓を1つ開けるのではなく、球の「前面かつ生え際より下」の頂点を
    まるごと落とす。残るのは 天面ドーム＋後頭部＋左右のサイド髪＋
    おでこ上に垂れる前髪バンドだけ。これで宇宙ヘルメット状の"前面ガラス"が
    無くなり、顔がはまる大きな開口になる。

    front_cut が大きいほど生え際(hairline_y)が上がり、前髪が短くなる。
    side_x は radius に対する比率（0〜1）。大きいほど前髪の幅が広がり
    "パッツン"寄りに、小さいほどサイドだけの細い前髪になる。
    """
    hairline_y = 0.02 * radius + front_cut   # この高さより下の前面には髪を残さない
    front_z = -0.12 * radius                 # これより手前(-Z)を「前面」とみなす
    side_limit = side_x * radius             # これより外側は前面でもサイド髪として残す（小さいほどサイド髪が太い）

    def drop(v: Vec3) -> bool:
        return v.z < front_z and v.y < hairline_y and abs(v.x) < side_limit

    # rings/sectors はスカート等（segments=16前後）に合わせた低ポリ数。高くする
    # ほど前髪の段差（フラットシェーディングの帯）が目立たなくなって丸く見える
    # （40/44 だとほぼ球体に見えて他パーツと質感が合わなかった）。低すぎると
    # 前髪の切れ端がギザギザに尖る（10前後で確認済み）ので、その中間を取る。
    # 既定14/16は bob/medium 用。side_x が広い blunt_bangs は前髪の開口が
    # 大きく、球面の丸みがそのまま広い面積で見えて他より「なめらか」に浮くため、
    # 呼び出し側でさらに rings/sectors を落として使う。
    return _clip_sphere_mesh(radius, rings=rings, sectors=sectors, drop=drop, lower_stretch=lower_stretch)


def _cap_hair_mesh(radius: float) -> Mesh:
    """頭に浅くかぶさる短髪キャップ（男子）。半球よりも短く、上部だけ残す。

    球を水平にカットするが、切り口を赤道より上に取るので「半球より短い」。
    前面(-Z)はさらに一段上でカットしておでこを広く見せる。サイド・後頭部は
    耳のあたりまで（＝女子のヘルメット髪ほど下りない）。
    """
    base_y = 0.12 * radius                 # 全周でこの高さより下は落とす（>0 なので半球より短い）
    front_y = base_y + 0.14 * radius        # 前面はさらに一段上でカットしておでこを見せる

    def drop(v: Vec3) -> bool:
        limit = front_y if v.z < -0.05 * radius else base_y
        return v.y < limit

    return _clip_sphere_mesh(radius, rings=22, sectors=28, drop=drop)


# --- 追加パーツ（髪の一部を独立 Entity で足すスタイル用） -----------------------
# ヘルメット/キャップ状のベース髪だけでは表現できない突起（お団子・ツインテール）を
# 頭部に追加の子 Entity として生やす。ベース髪と同じ hair_color で塗る。


def _make_bear_buns(head: Entity, radius: float, hair_color, shd) -> list[Entity]:
    """頭頂の左右にお団子を2つ（熊ダンゴ）。土台のキャップ髪の球面よりだいぶ高い位置に
    置いて、埋もれずにちゃんと飛び出て見えるようにする（お団子は地肌から生えるというより
    頭の輪郭の外側に乗る飾りなので、これくらい高くて自然）。"""
    bun_r = 0.14 * radius
    bx, by, bz = 0.48 * radius, 0.92 * radius, 0.0
    buns = []
    for sx in (-1, 1):
        buns.append(
            Entity(
                parent=head,
                # 標準の "sphere" は滑らかすぎて浮くため低ポリ球を使う。
                model=_low_poly_sphere_mesh(),
                color=hair_color,
                shader=shd,
                scale=bun_r * 2,
                x=sx * bx,
                y=by,
                z=bz,
            )
        )
    return buns


def _make_twintails(head: Entity, radius: float, hair_color, shd) -> list[Entity]:
    """耳の高さ左右から垂れる、先端がすぼまるツインテール（円錐台×2）。

    _tube_mesh は自分のローカル y=0（bottom_r側）〜y=height（top_r側）に張られる。
    「結び目を支点に毛先が振れる」動きにしたいので、まず結び目(attach_y)の位置に
    ピボット用の空 Entity を置き、その回転で房の角度を決める。房本体はピボットの
    子として y=-height にぶら下げ、天面(top_r=太い)が結び目、底面(bottom_r=細い)が
    毛先に来るようにする。返り値はピボット Entity のリスト（房を揺らすときはこれを回す）。

    調整ポイント:
      attach_y        … 結び目の高さ（頭中心基準）
      pivot.x / pivot.z  … 結び目の左右・前後位置
      rotation_z      … 結び目支点の開き角（係数を上げるほど毛先が外へ開く。
                        房は下垂なので符号は -sx）。rotation_x を足せば前後の振り。
                        揺らすときも pivot.rotation_* を時間で振ればよい
      height          … 房の長さ
      bottom_r / top_r … 毛先 / 結び目の太さ
      knot_bun_r      … 結び目に乗せる熊ダンゴの半径（_make_bear_buns の bun_r=0.32R より
                        やや小さめ。房の付け根を覆って「結んでまとめた」感を出す）
    """
    height = 3.10 * radius
    attach_y = 0.80 * radius  # 結び目（耳の高さ）
    knot_bun_r = 0.11 * radius
    pivots = []
    for sx in (-1, 1):
        pivot = Entity(
            parent=head,
            x=sx * 0.70 * radius,
            y=attach_y,
            z=0.1 * radius,
            rotation_z=-sx * 12,  # 毛先を外側へ開いて土台の髪から分離して見せる（房は下垂なので符号は -sx）
        )
        Entity(
            parent=pivot,
            # segments は腕・脚の円錐台（8）に合わせ、房だけ丸く滑らかに見えないようにする。
            model=_tube_mesh(bottom_r=0.1 * radius, top_r=0.22 * radius, height=height, segments=8),
            color=hair_color,
            shader=shd,
            y=-height,  # 結び目(ピボット)から下へ房を垂らす
        )
        # 結び目の熊ダンゴ。房の付け根に置く（球なので pivot 回転の影響なし）。
        # わずかに外・上・手前へずらして土台の髪に埋もれず「結んでまとめた玉」に見せる。
        # 標準の "sphere" は滑らかすぎて浮くため低ポリ球を使う。
        Entity(
            parent=pivot,
            model=_low_poly_sphere_mesh(),
            color=hair_color,
            shader=shd,
            scale=knot_bun_r * 2.5,
            x=sx * 0.08 * radius,
            y=0.04 * radius,
            z=-0.04 * radius,
        )
        pivots.append(pivot)
    return pivots


# --- 装飾品（accessory）。ベース髪の上に足す頭部パーツ ------------------------
# [[cast]] の accessory = [{ type, color, side }] から生成する。すべて head の子に
# して首かしげ演出へ追従させ、bear_buns / twintails と同じく単色プリミティブで組む。
# radius は _make_bear_buns 等と同じく hair_radius（= face_radius * 0.65）を渡す。


def _acc_sides(side: str | None, default: str) -> tuple[int, ...]:
    """accessory の side（"left" / "right" / "both" / None）→ 生成する x 符号のタプル。

    左右の約束は [[cast]] の腕 l/r と同じ＝ "left" が画面左（-X）。
    """
    s = side or default
    if s == "left":
        return (-1,)
    if s == "right":
        return (1,)
    return (-1, 1)


def _arc_band_mesh(
    ring_r: float, thick: float, width: float, deg0: float, deg1: float, segments: int = 24
) -> Mesh:
    """X-Y 平面内で Z 軸まわりに deg0→deg1 の弧を描く、断面が長方形の細い帯。

    ヘッドフォンのバンド用（頭の左右＝±X と頭頂＝+Y を通す）。thick は半径方向の
    厚み、width は Z 方向（顔の前後）の幅。フラットシェーディング前提で面ごとに
    独立頂点を張る。
    """
    verts: list[Vec3] = []
    tris: list[tuple[int, int, int]] = []

    def corners(a: float) -> list[Vec3]:
        rad = Vec3(math.cos(a), math.sin(a), 0)
        outer, inner = rad * (ring_r + thick), rad * (ring_r - thick)
        zf = Vec3(0, 0, width / 2)
        return [outer + zf, outer - zf, inner - zf, inner + zf]

    rings = [
        corners(math.radians(deg0 + (deg1 - deg0) * i / segments)) for i in range(segments + 1)
    ]
    for i in range(segments):
        a_ring, b_ring = rings[i], rings[i + 1]
        for p, q in ((0, 1), (1, 2), (2, 3), (3, 0)):  # 外・後・内・前の4面
            base = len(verts)
            verts += [a_ring[p], a_ring[q], b_ring[q], b_ring[p]]
            # ursina の generate_normals は素朴な外積の符号を反転するため（理由不明、
            # ursina/scripts/generate_normals.py の "inverse it, dunno why" 参照）、
            # _tube_mesh と同じ外向き法線を得るには巻き順を逆にする必要がある
            # （09-18 実測：反転前は外・後・内・前の4面すべての法線が裏返っていた）。
            tris += [(base, base + 2, base + 1), (base, base + 3, base + 2)]

    m = Mesh(vertices=verts, triangles=tris, mode="triangle")
    m.generate_normals(smooth=False)
    return m


def _make_cat_ears(head: Entity, radius: float, ear_color, shd, sides: tuple[int, ...]) -> list[Entity]:
    """頭頂の左右に立つ三角の猫耳（前後を潰した円錐×2）。根元は髪に埋め、先端を外側へ倒す。

    rotation_z（= sx * 角度）が外への開き具合。bottom_r と height の比が三角形の鋭さ
    （bottom_r を大きく・height を小さくするほど、角ばらず横に広い猫耳になる）。
    """
    ears: list[Entity] = []
    for sx in sides:
        pivot = Entity(parent=head, x=sx * 0.52 * radius, y=0.60 * radius, z=0.0, rotation_z=sx * 21)
        Entity(
            parent=pivot,
            model=_tube_mesh(bottom_r=0.46 * radius, top_r=0.035 * radius, height=0.56 * radius, segments=14),
            color=ear_color,
            shader=shd,
            scale=(1.0, 1.0, 0.4),  # 前後を潰して耳らしく
        )
        ears.append(pivot)
    return ears


def _make_headphones(head: Entity, radius: float, band_color, shd) -> list[Entity]:
    """頭頂をまたぐ1本の帯＋左右のイヤーカップ（潰した球）。side は無視して常に両耳。"""
    parts: list[Entity] = [
        Entity(
            parent=head,
            # segments は腕・脚等（8）に合わせる。128°の弧なので8でも十分滑らかに見える。
            model=_arc_band_mesh(1.02 * radius, 0.05 * radius, 0.34 * radius, 26, 154, segments=8),
            color=band_color,
            shader=shd,
        )
    ]
    for sx in (-1, 1):
        parts.append(
            Entity(
                parent=head,
                # 標準の "sphere" は滑らかすぎて浮くため低ポリ球を使う。
                model=_low_poly_sphere_mesh(),
                color=band_color,
                shader=shd,
                x=sx * 1.0 * radius,
                y=0.02 * radius,
                z=0.0,
                scale=(0.22 * radius, 0.5 * radius, 0.5 * radius),
            )
        )
    return parts


def _make_hairband(head: Entity, radius: float, band_color, shd) -> list[Entity]:
    """頭を前から左右の耳の上を通ってまたぐ、細いカチューシャ（イヤーカップ無し）。

    _arc_band_mesh は headphones と共用。headphones の分厚いパッド帯と違い、
    width（前後の幅）を大きく絞って薄い一本のバンドにし、弧の範囲もやや広めに
    とって耳の近くまで下ろす。side は取らず常に両耳の上を通る。
    ring_r は髪表面の半径（= radius）ちょうどだと髪メッシュに埋もれて見えなくなる
    （headphones が 1.02 で外側へ逃がしているのと同じ理由）。薄いバンドは埋もれ
    やすいので、headphones よりさらに外側を使う（09-18 報告：等倍だと完全に
    不可視、1.05 でもまだ少しめり込みがあったため 1.12 まで上げた）。

    土台の輪はヘッドホンと同じ「頭頂(+Y)を通る平面の輪」のまま、頭の中心を通る
    X 軸（耳から耳の軸）でまるごと前に傾ける（09-18 報告：平面のままだと
    headphones と同じ高さに見えてしまうため）。09-18 実測: ursina は内部で
    coordinate-system を y-up-left にしており、Entity.rotation_x は Panda3D の
    ピッチ(P)と符号が逆（entity.rotation_x = -P）。P=+90 で頭頂方向(+Y)が
    ちょうど顔正面(-Z、目の高さ)まで倒れることを NodePath.setP() で確認済みなので、
    「頭頂(0°)から目の高さ(90°)まで」のうち何度分だけ前に傾けるかを _TILT_DEG に
    度数でそのまま指定し、rotation_x にはその符号反転（-_TILT_DEG）を渡す。
    """
    _TILT_DEG = 48.0  # 0=headphonesと同じ真上 / 90=目を覆う正面（ユーザー指定の基準）
    return [
        Entity(
            parent=head,
            # _arc_band_mesh(ring_r, thick, width, deg0, deg1, segments):
            #   ring_r … 頭の中心からの距離（外に出すほどめり込みが減る。09-18 に 1.05→1.12 へ）
            #   width  … 帯の横幅＝前後方向の幅（09-18 に半分の 0.07 へ。狭めたいときはここ）
            #   thick  … 帯の半径方向の厚み（太さ）
            model=_arc_band_mesh(1.12 * radius, 0.035 * radius, 0.15 * radius, 18+50, 162-50, segments=10),
            color=band_color,
            shader=shd,
            rotation_x=-_TILT_DEG,
        )
    ]


def _make_pin(head: Entity, radius: float, pin_color, shd, sides: tuple[int, ...]) -> list[Entity]:
    """前髪の脇に留める一文字のヘアピン（細いキューブ1本）×side。

    rotation_z が寝かせ具合、scale の1つ目が長さ・2つ目が太さ。
    """
    pins: list[Entity] = []
    for sx in sides:
        pivot = Entity(parent=head, x=sx * 0.65 * radius, y=0.52 * radius, z=-0.78 * radius)
        Entity(
            parent=pivot,
            model="cube",
            color=pin_color,
            shader=shd,
            scale=(0.40 * radius, 0.085 * radius, 0.05 * radius),
            rotation_z=-sx * 30,  # 外側の端を上げる（外上がり）
        )
        pins.append(pivot)
    return pins


class Character(Entity):
    """クレイ風キャラクター1体。self を親に、全パーツを子 Entity で持つ。

    色・サイズはコンストラクタ引数。後で10人分をループ生成できる設計。
    """

    def __init__(
        self,
        speaker_id: str = "",
        name: str = "",
        hair_color: color.Color = color.gray,
        skin_color: color.Color = color.rgb(0.82, 0.82, 0.82),
        dress_color: color.Color = color.rgb(0.9, 0.9, 0.9),
        eye_color: color.Color = color.black,
        face_radius: float = 1.0,
        hair_front_cut: float = 0.12,
        model: str = DEFAULT_MODEL,  # [[cast]] の model キーワード（models.MODEL_PRESETS のキー）
        hair_style: str | None = None,   # 明示指定で model プリセットを上書き。"bob" / "cap"
        body_style: str | None = None,   # 明示指定で model プリセットを上書き。"dress" / "box"
        arm_raise: float = 0.0,
        accessories: list | None = None,  # (type, color or None, side or None) のタプル列。app.py が [[cast]] から渡す
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # model キーワードからメッシュ構成を引き、明示引数があればそちらを優先。
        preset = resolve_model(model)
        hair_style = hair_style or preset["hair_style"]
        body_style = body_style or preset["body_style"]

        # app.py 互換：ひな壇↔読書コーナーの立ち位置切り替え用に、生成時の
        # 位置・大きさを「定位置(home)」として控えておく。scale はスカラー前提
        # （app.py は常にスカラーを渡す）。
        self.speaker_id = speaker_id
        base_scale = kwargs.get("scale", 1.0) or 1.0
        self.home = (self.x, self.y, self.z, base_scale)

        shd = lit_with_shadows_shader
        self.hair_color = hair_color
        self.skin_color = skin_color
        self.dress_color = dress_color
        self.eye_color = eye_color

        head_y = 1.62

        # --- 頭グループ（首かしげ演出はこの Entity を回す） ---
        self.head = Entity(parent=self, y=head_y)

        # 顔（大きめの肌色の球。髪の開口部にはまり込み、正面から顔として見える）
        self.face = Entity(
            parent=self.head,
            model="sphere",
            color=skin_color,
            shader=shd,
            scale=(face_radius * 1.0, face_radius * 1.05, face_radius * 1.0),
            z=-0.12,
            y=0.01,
        )

        # 髪。ベース形状は3系統:
        #   cap    = 頭の球より一回り大きいだけの浅い短髪キャップ（男子。既存）
        #   helmet = 天面・後頭部・サイドを覆うヘルメット/ボブ。ショート4種＋ミディアム3種の
        #            土台はすべてこれで、side_x（前髪の幅）/ lower_stretch（後ろの長さ）
        #            のパラメータだけで書き分ける。
        #   bear_buns だけ土台に cap を使い、頭頂にお団子2つを別 Entity で足す。
        # hair_style ごとの数値はここでしか出てこないので、見た目を詰めるときは
        # このブロックの定数だけ触ればよい。
        hair_radius = face_radius * 0.65
        self.hair_extra: list[Entity] = []  # お団子・ツインテールなど、ベース髪に追加する子パーツ

        if hair_style == "cap":
            # ベースの球は他スタイルと同じ hair_radius に統一する（pin / cat_ears /
            # headphones はどのスタイルでも hair_radius 基準で置くため、ここだけ
            # 別半径だとアクセサリが髪の表面とずれて埋まる/浮く原因になる）。
            hair_mesh = _cap_hair_mesh(hair_radius)
        elif hair_style == "bear_buns":
            # おでこを広く見せるやや浅めのキャップ＋頭頂の左右にお団子。
            # cap 同様、半径は hair_radius に統一（理由は cap 側コメント参照）。
            hair_mesh = _cap_hair_mesh(hair_radius)
        elif hair_style == "blunt_bangs":
            # 前髪パッツン: side_x を広げてサイドに流れず全幅まっすぐ切りそろえた
            # 前髪にする＝おでこが広く見える。
            # side_x が広い分、球面の丸みが前面に広く残って他の髪型より滑らかに
            # 見えてしまうため、rings/sectors を落として低ポリ感を揃える。ただし
            # rings=7 は生え際の高さが離散的（リング1段単位）にしか動かせないほど粗く、
            # +0.10 上乗せだとリング境界が高い側に寄って前髪がほぼ消える（おでこ全開）
            # 状態になっていた。-0.12 でちょうど1段下のリング境界（眉あたり）に落ち着く。
            # sectors は奇数だと前面中央(x=0)を通る経線が無く、生え際カットの
            # 左右対称が崩れて額の中央に三角の隙間（顔が覗く「めり込み」）が出る
            # ため偶数にする必要がある（9→10）が、それだけでは不十分だった：
            # 偶数なだけ（4の倍数でない）だと x=0 をまたぐ経度の頂点列が無いまま
            # 前面中央の1クワッドだけが左右対称の位置に来る形になり、そのクワッドを
            # 2枚の三角形に割る対角線が必ず非対称になる（頂点列自体は左右対称でも
            # 対角線の向きは片側に寄る）。この1枚だけ法線の向きが隣と食い違い、
            # 額の中央にライトを受けた白い点が浮いて見えていた（09-16 報告）。
            # sectors を4の倍数にすると x=0 上に頂点列そのものができ、前面中央の
            # クワッドが左右の2枚に分かれて対角線も左右対称になり、この点が消える
            # （10→12で確認）。rings=7 はコメントの通り維持。
            hair_mesh = _helmet_hair_mesh(
                hair_radius, hair_front_cut - 0.12, side_x=0.85, lower_stretch=1.45, rings=7, sectors=12
            )
        elif hair_style == "medium":
            hair_mesh = _helmet_hair_mesh(hair_radius, hair_front_cut, lower_stretch=1.55)
        elif hair_style == "medium_twintail":
            # 土台は bob と同じ既定カットの helmet 髪。長さは _make_twintails の
            # 房だけで見せる（土台まで伸ばすと房と一体化してただの長い髪にしか
            # 見えなくなるため、房を土台から分離させる）。
            # 以前は側面を広めに刈っていた(side_x=0.75)が、耳の高さで side_x が
            # 効く範囲がサイド〜襟足まで及び、髪の裾が肩に届かず結び目の下に
            # 背景が素通しで見える隙間ができていた（09-15 報告）。既定(0.44)に
            # 戻すと bob と同じ裾になり隙間が塞がる。
            hair_mesh = _helmet_hair_mesh(hair_radius, hair_front_cut)
        else:  # "bob"（既定）
            hair_mesh = _helmet_hair_mesh(hair_radius, hair_front_cut)

        self.hair = Entity(
            parent=self.head,
            model=hair_mesh,
            color=hair_color,
            shader=hair_shader,
            double_sided=True,  # 前面を切り欠いた開いた形状なので、裏面もこのシェーダーで暗く塗って描画する
            scale=1.0,
            z=0.0,
            y=0.03 if hair_style in ("cap", "bear_buns") else 0.02,
        )

        if hair_style == "bear_buns":
            self.hair_extra = _make_bear_buns(self.head, hair_radius, hair_color, shd)
        elif hair_style == "medium_twintail":
            self.hair_extra = _make_twintails(self.head, hair_radius, hair_color, shd)

        # 装飾品（accessory）。ベース髪・お団子の上に足す頭部パーツ。self.head の子
        # なので首かしげに追従し、set_visible(=self.enabled) でまとめて出し入れされる。
        self.accessories: list[Entity] = []
        for spec in accessories or []:
            acc_type, acc_color, acc_side = spec
            tinted = acc_color if acc_color is not None else hair_color
            if acc_type == "cat_ears":
                self.accessories += _make_cat_ears(
                    self.head, hair_radius, tinted, shd, _acc_sides(acc_side, "both")
                )
            elif acc_type == "headphones":
                band = acc_color if acc_color is not None else color.rgb(0.24, 0.24, 0.27)
                self.accessories += _make_headphones(self.head, hair_radius, band, shd)
            elif acc_type == "pin":
                self.accessories += _make_pin(
                    self.head, hair_radius, tinted, shd, _acc_sides(acc_side, "left")
                )
            elif acc_type == "hairband":
                self.accessories += _make_hairband(self.head, hair_radius, tinted, shd)

        # 顔の前面（球）はおよそ z=-0.645。目・口はそれより十分手前に置いて、
        # 首かしげ・見下ろしカメラでも顔にめり込まないようにする。
        face_front_z = -0.72

        # 目：楕円の quad 2枚（まばたきで scale_y を動かす）。遠目でも見えるよう気持ち大きめ。
        eye = dict(parent=self.head, model="quad", color=eye_color, shader=shd, double_sided=True)
        self.eye_l = Entity(**eye, scale=(0.12, 0.15), x=-0.17, y=0.03, z=face_front_z)
        self.eye_r = Entity(**eye, scale=(0.12, 0.15), x=0.17, y=0.03, z=face_front_z)
        self._eye_open = 0.15  # 開いているときの scale_y

        # 口：黒い quad（リップシンクで scale_y を動かす。独立 Entity 必須）。
        # 閉口時はほぼ線、発話時ははっきり開くよう開閉幅を大きめに取る。
        self.mouth = Entity(
            parent=self.head,
            model="quad",
            color=color.black,
            shader=shd,
            double_sided=True,
            scale=(0.26, 0.04),
            y=-0.21,
            z=face_front_z,
        )
        self._mouth_closed = 0.04

        # --- 胴体。dress=上が細く下が広がる円錐台（ワンピース、女子）。
        #     box=肩から腰までまっすぐな箱型（男子）。 ---
        if body_style == "box":
            self.dress = Entity(
                parent=self,
                model="cube",
                color=dress_color,
                shader=shd,
                scale=(0.82, 0.86, 0.5),
                y=0.85,
            )
        else:
            self.dress = Entity(
                parent=self,
                model=_tube_mesh(bottom_r=0.58, top_r=0.22, height=0.82, segments=16),
                color=dress_color,
                shader=shd,
                y=0.42,
            )

        # 襟：首元の薄い白リング（簡略化して1パーツ）
        self.collar = Entity(
            parent=self,
            model="sphere",
            color=color.rgb(0.97, 0.97, 0.97),
            shader=shd,
            scale=(0.42, 0.1, 0.42),
            y=1.3,
        )

        # --- 腕（円柱）＋手（立方体）。左右独立、回転角を変数化 ---
        arm_len = 0.72
        self.arm_l = Entity(parent=self, x=-0.46, y=1.24)
        self.arm_r = Entity(parent=self, x=0.46, y=1.24)
        for side, pivot in (("l", self.arm_l), ("r", self.arm_r)):
            Entity(
                parent=pivot,
                model=_tube_mesh(0.075, 0.085, arm_len, segments=8),
                color=skin_color,
                shader=shd,
                y=-arm_len,
            )
            setattr(
                self,
                f"hand_{side}",
                Entity(
                    parent=pivot,
                    model="cube",
                    color=skin_color,
                    shader=shd,
                    scale=0.17,
                    y=-arm_len,
                ),
            )
        self.arm_raise = arm_raise  # setter で両腕へ反映

        # --- 脚（円柱）＋足（潰した立方体） ---
        leg_len = 0.46
        for side, sx in (("l", -0.17), ("r", 0.17)):
            Entity(
                parent=self,
                model=_tube_mesh(0.1, 0.11, leg_len, segments=8),
                color=dress_color.tint(-0.15),
                shader=shd,
                x=sx,
                y=0.0,
            )
            setattr(
                self,
                f"foot_{side}",
                Entity(
                    parent=self,
                    model="cube",
                    color=dress_color.tint(-0.25),
                    shader=shd,
                    scale=(0.2, 0.12, 0.34),
                    x=sx,
                    y=0.06,
                    z=-0.06,
                ),
            )

        # 名札（頭上のキャラ名）。
        # 3D空間に Text を子として置くと glyph が size(0.025)×scale で極小になり、
        # このバージョンの ursina には billboard 指定も無いため実質見えない。
        # そこで app.py 側で camera.ui（2D）に Text を描画し、毎フレーム
        # label_anchor() を画面へ投影して頭上に追従させる。ここでは表示名と
        # アンカーの高さ（キャラ座標系）だけを保持する。
        self.display_name = name
        self.label_local_y = head_y + 0.72

        # 待機モーション / まばたき用の内部状態
        self._phase = random.uniform(0, math.tau)
        self._base_head_y = head_y
        self._next_blink = time.time() + random.uniform(3.0, 6.0)
        self._blink_until = 0.0

    # --- パラメータ ---------------------------------------------------------

    @property
    def arm_raise(self) -> float:
        return self._arm_raise

    @arm_raise.setter
    def arm_raise(self, deg: float) -> None:
        """0=体側に下ろす。正の値で前方に振り上げる（将来「発話中に挙げる」演出用）。"""
        self._arm_raise = deg
        self.arm_l.rotation_x = -deg
        self.arm_r.rotation_x = -deg

    # --- app.py 互換 API（旧 character.Character と同じ使い勝手） ----------

    def set_pose(self, x: float, y: float, z: float, scale: float) -> None:
        """ひな壇↔読書コーナーのレイアウト切り替え。Entity なので位置・大きさを直接書き換える。"""
        self.position = (x, y, z)
        self.scale = scale

    def set_visible(self, visible: bool) -> None:
        if self.enabled != visible:
            self.enabled = visible  # 子パーツ（顔・髪…）にも波及する

    def label_anchor(self) -> Vec3:
        """頭上の名札を出すワールド座標。app.py が camera.ui へ画面投影する。"""
        return self.world_position + Vec3(0, self.label_local_y * self.world_scale_y, 0)

    # --- 毎フレーム（既存 Entity の属性書き換えのみ。新規生成しない） -------
    #
    # メソッド名は animate。Entity サブクラスで update() を定義すると Ursina が
    # 毎フレーム引数なしで呼んでしまう（t を渡せない）ため使わない。app.py の
    # update ループから明示的に animate(t, is_speaking, rms) を呼ぶ。

    def animate(self, t: float, is_speaking: bool = False, rms: float = 0.0) -> None:
        # 待機：数mmの上下＋首かしげ
        self.head.y = self._base_head_y + math.sin(t * 1.6 + self._phase) * 0.02
        self.head.rotation_z = math.sin(t * 0.7 + self._phase) * 2.0

        # まばたき：3〜6秒間隔で0.1秒閉じる
        if t >= self._next_blink:
            self._blink_until = t + 0.1
            self._next_blink = t + random.uniform(3.0, 6.0)
        sy = 0.02 if t < self._blink_until else self._eye_open
        self.eye_l.scale_y = sy
        self.eye_r.scale_y = sy

        # リップシンク：喋っている側だけ current_rms で口を開く。
        # current_rms は 0.05〜0.2 程度なので係数を大きめに取り、開閉をはっきり見せる。
        self.mouth.scale_y = (
            self._mouth_closed + min(rms * 4.0, 1.0) * 0.34 if is_speaking else self._mouth_closed
        )
