# bead3d — 3D 插接拼豆 (interlocking 3D perler/fuse-bead kit generator)

Turn a 2D character picture (or a 3D mesh) into a **glue-free 3D 拼豆 kit**: flat
pieces you iron separately, then slide/slot together in a fixed order.

This directory is `bead3d/` at the repo root. From the repo root:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
python bead3d/bead3d.py --help
```

## Setup / dependencies
* Python packages (see repo-root `requirements.txt`): `numpy scipy pillow trimesh gradio_client networkx`.
  `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
* Palettes + colour maths are reused from the 2D pattern tool checked out beside this directory:
  `../bead-pattern/scripts/generate.py` (palettes in `../bead-pattern/palettes`).
  Override that scripts directory with env `BEAD_PATTERN_SCRIPTS` or `--bead-pattern-scripts`.
  Palettes: `mard` (default), `coco`, `manman`, `mixiaowo`, `panpan`.
* Fonts: Noto CJK (`/usr/share/fonts/opentype/noto/NotoSansCJK-*.ttc`, `apt install fonts-noto-cjk`).
* `from-image` needs internet: HuggingFace Space `tencent/Hunyuan3D-2` (free; optional `HF_TOKEN` env
  for quota). Cold start can take 1–4 min; the script retries the connection 3×.

## Usage
```bash
# from the repo root, after pip install -r requirements.txt
# 2D picture -> Hunyuan3D-2 shape -> kit (colours come from the picture)
python bead3d/bead3d.py from-image char.png --out out_dir --title 猪猪侠 --replace P22=E20
# existing mesh (glb/obj/stl/ply), painted from a front image
python bead3d/bead3d.py from-model shape.glb --image char.png --height 32 --out out_dir
# re-run quickly with the mesh cached by a previous from-image run
python bead3d/bead3d.py from-image char.png --shape out_dir/shape.glb --height 28 --back-color F08 --out out2
# STL lying on Z-up, front facing +x
python bead3d/bead3d.py from-model toy.stl --up z --yaw 90 --base-color H02 --out out3
```
Options (both modes):
| flag | meaning |
|---|---|
| `--out DIR` | output folder (required) |
| `--palette mard` | bead brand palette |
| `--target-beads 3000` | size by total beads (binary search on height, ~10 trial designs ≈ 1–2 min) |
| `--height N` | size by beads tall (fast, overrides target) |
| `--max-colors 8` | colour limit (support colour not counted) |
| `--support-color 任意色` | hidden inner core/pegs: `任意色` (drawn green, any leftover beads) or a code e.g. `H02` |
| `--replace P22=E20` | colour substitution, repeatable |
| `--back-color F15` | force back/side colour (default: auto, per-row outline colour) — use if face colour leaks onto the back |
| `--face-colors R20,H07` | optional: make cells of these codes a separate front face panel/relief (codes must appear in the output) |
| `--boost 1.1` | brighten source colours (shaded/dark renders) |
| `--base-color H02` | colour for an uncoloured mesh with no `--image` |
| `--up y/z`, `--yaw deg` | orient the mesh (GLB = y-up) |
| `--title` | title on the sheets |
| from-image only: `--shape mesh`, `--space`, `--seed 1234` | skip Hunyuan / other Space / shape seed |

## Outputs (in `--out`, plain PNG/CSV, no zip)
* `sheet.png` — main sheet: 3 previews (front/side/back), colour+count table, all pieces, exploded view
* `parts_1.png …` — every piece as a bead grid with piece number and colour codes, split over several pages
* `assembly-steps.png` — numbered step-by-step assembly order with direction of insertion
* `exploded.png`, `preview_front/side/back.png`
* `shopping-list.csv` (色号, hex, 数量, 任意色 row, 总计 — no spare added), `pieces.csv` (件号 = assembly order, 类型, 颗数, 安装方向, 插接 neighbours, 装好后可动方向 / 锁死)
* `summary.json` (also printed to stdout): bead counts, piece kinds, locked/one-way/loose counts,
  `valid_assembly_order`, file paths, `warnings`.
* `shape.glb` (from-image only) — reuse with `--shape`.

## How the design works
Voxelise → colour surface from the image (k-means in Lab → CIEDE2000 snap to palette) →
hollow kit: horizontal ring piece per layer + hidden cross cores (two plates joined by a half-lap)
→ rings are trimmed to slide over the cores → vertical pegs through ring holes →
2–3-bead key bars across adjacent layers → cap pinning → validation by greedy
disassembly (reverse = assembly order) and per-piece lock test (locked / one-way / loose).

## Known limitations
* Front colours come from ONE front image; back/sides are guessed (outline colour per row). Use `--back-color`.
* Small details (nose, mouth lines) can merge into the skin at small heights; go taller (`--height 40`) for detail.
* Real total is usually ~5–10 % below `--target-beads` (core trimming after the estimate).
* Often 1 loose small piece and long 1-bead-wide pegs remain → reported in `warnings` (glue that one / iron pegs doubled).
* Only one core pair unless a clear neck (waist) is found; thin limbs/ears become ring bumps, not separate parts.
* Hunyuan3D-2 Space can be slow/asleep/out of quota; fall back to `from-model` with any mesh.
* Physical fit untested: assumes 5 mm beads stay on grid after ironing (iron lightly, keep flat).

## Box style（盒子式插接，MARD 风格）: `--style box`

Builds a MARD-style 2.5D "box" kit from flat ironed panels instead of auto-sliced voxel layers:

- **正面板 FRONT** = the character art (pixel-art PNGs with ≤32 colours and ≤100 px are used 1 bead per pixel;
  other images are resized to `--height` and colour-quantised).
- **背面板 BACK** = same silhouette; default = front art with small face details (eyes, nose, mouth; components
  ≤60 beads not touching the outline) painted over with the surrounding colour. `--back-color CODE` = one colour.
  Drawn as seen from behind (mirrored).
- **侧条 SIDE strips** (竖侧条 in the YZ plane, 横侧条 in the XZ plane) stand between the panels, `--depth` beads
  deep. They are straight runs chosen greedily inside a 2-bead band just inside the outline (silhouette inset by
  1 so each panel keeps a solid rim), so the wall follows the silhouette; crossing strips are trimmed (butt joints,
  ≥1 bead apart).
- **立体小件 ACCESSORIES** (raised layers): regions of `--accessory-colors` codes (holes filled, adjacent dark
  outline added) and/or `--accessory-box y0:y1,x0:x1`, or — if the 3D model has real relief — cells ≥2 voxels in
  front of the body plane. Each sits 1 bead in front of the front panel on 1–3 horizontal **垫条 spacer** strips.
  Front-panel beads hidden under them become 任意色 (`--support-color` to force a code).
- **Tab/slot joints**: on every strip, every other bead (never the end beads) has a 1-bead **凸起 tab** on both
  long edges. The panel gets a 1-bead **孔/凹口 hole** (empty peg) at exactly that (row, col). The tab fills the
  hole and is coloured like the panel cell it replaces, so the picture stays complete. Spacers work the same way
  (one edge into the front panel, the other into the accessory). Validator checks: no overlaps, each piece flat
  and 4-connected, every hole filled by exactly one tab.
- **3D model**: `from-image --style box` still runs Hunyuan3D (or `--shape mesh.glb`, or `from-model`); the model
  sets the depth (`--depth` overrides) and relief accessories, and `model_preview.png` renders it. If the model is
  flat (pixel art often gives a flat card) or the Space is down, depth falls back to ~12% of width (4–6);
  `--no-model` skips it.

```
python bead3d/bead3d.py from-image bear.png --style box --palette mard --depth 5 \
    --accessory-colors F18,A11,G10 --out out/bear_box --title 西兰花小熊 [--shape out/bear/shape.glb]
```
Outputs: `sheet.png` (title, MARD legend+counts, all flat parts, assembled/back/exploded/3D-model views),
`parts_N.png` (one per piece, codes on every bead, holes = white with red ring, tabs = red border),
`assembly-steps.png` (front down → insert strips → back panel → spacers+accessories, plus exploded view),
`exploded.png`, `preview_{front,iso,side,back}.png`, `model_preview.png`, `shopping-list.csv`, `pieces.csv`, `summary.json`.
Friction fit only (like MARD kits): nothing is mechanically locked; physical fit is untested.
