---
name: bead3d 3D插接拼豆
description: >-
  Use when the user wants a 3D (立体) 拼豆/perler bead kit that snaps together
  without glue from a 2D picture or a 3D model file (glb/obj/stl). Default
  output is a MARD-style box kit (front/back panels, side strips, plug-in raised
  pieces). For a flat 2D pattern use bead-pattern instead.
---
# 3D 插接拼豆 / Interlocking 3D perler kit (bead3d)

Tool: `bead3d/bead3d.py` (from the repo root: `python bead3d/bead3d.py`; install with `pip install -r requirements.txt`; see README.md, incl. the "Box style" section). Palettes reused from `bead-pattern/` (`../bead-pattern/scripts` relative to this directory; override with `BEAD_PATTERN_SCRIPTS` or `--bead-pattern-scripts`).

## 2D vs 3D
- 2D request: do NOT run this; make a flat pattern with `bead-pattern` (`bead-pattern/scripts/generate.py`).
- 3D request: image -> 3D model (Hunyuan3D) -> MARD-style box kit (`--style box`). If unclear which, ask.

## Steps / 步骤
1. Get a clean, front-facing, single-character image (white/transparent background). For pixel art, first recover one pixel per cell so the front panel matches the art exactly.
2. Box style (MARD-like: front + back panels following the silhouette, side strips with tab/slot edges, raised accessory pieces on spacer strips):
```
python bead3d/bead3d.py from-image /path/char.png --style box --out out3d/<name> --title <名字> --palette mard [--depth 5] [--height 36] [--accessory-colors F18,A11] [--accessory-box y0:y1,x0:x1] [--back-color <code>] [--no-model | --shape <glb>]
```
   - Runs Hunyuan3D by default to take depth and raised parts from the model; if the model is flat or fails, depth falls back to ~12% of width (4-6 beads). Pixel-art/flat drawings usually come out flat, so pick raised parts with `--accessory-colors` / `--accessory-box`.
   - Smaller/easier kit: `--height 36`, `--depth 4`.
3. Legacy auto-sliced mode (no --style box): `from-image <img> --out ... [--target-beads 3000 | --height 32]`, or `from-model model.glb --image char.png`. Only when the user wants a rounded sculpted figure.
4. Read the printed JSON (total_beads, colors, pieces, joints, warnings). Look at `preview_iso.png`, `preview_back.png`, `sheet.png` yourself before sending.
5. Deliver images directly in chat (no zip): `preview_iso.png`, `sheet.png` (legend + all parts), `assembly-steps.png`; `parts_1..N.png` on request; `model_preview.png` to show the 3D model. Summarise in the user's language: size (颗高×宽×深, cm), piece count, beads per code, 任意色 count, warnings.

## Caveats / 注意
- Hunyuan Space can be asleep or out of free quota (hours); retry later or use `--no-model`. Mention when depth was set by hand.
- Friction fit only; short strips with few tabs are the weakest joints; round outlines leave small side gaps.
- Physical fit not tested; tell the user it is a design to build and adjust. Iron flat and lightly so tabs still fit holes.
- Big pieces (>29 beads) need multiple boards; say so.
- If dependencies are missing: from the repo root, `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`.
