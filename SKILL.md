---
name: pindou
description: >-
  把图片做成拼豆（拼豆 / perler / hama / fuse beads）。平面图纸用 bead-pattern：像素化后按咪小窝、MARD、COCO、漫漫、盼盼色卡做 CIEDE2000 匹配，产出带色号网格、备料清单和 PDF。立体免胶插接套件用 bead3d，传 --style box 得到 MARD 风格盒子（正面板、背面板、侧条凸起/卡槽、凸起小件）；也可从 glb/obj/stl 出发。触发词：拼豆、拼豆图纸、拼豆模版、立体拼豆、3D拼豆、插接、色号、备料清单、perler、hama。
---

# 拼豆（平面图纸 + 立体插接套件）

仓库根目录运行。依赖见根目录 `requirements.txt`（立体）和 `bead-pattern/requirements.txt`（平面）。中文图需要系统里的 Noto CJK（`fonts-noto-cjk`）。

先分清用户要哪一种。含糊就问一句。

- **平面**：一张图 → 照着拼的网格图纸。用 `bead-pattern/scripts/generate.py`。
- **立体**：免胶、能卡在一起的套件。用 `bead3d/bead3d.py`，并加上 `--style box`。

## 平面图纸

1. 确认图（动漫角色、logo、照片、像素画）和品牌色卡。顾客要好拼用 `--mode simple`；要高还原用 `--mode fine`。
2. 在仓库根目录：

```bash
python bead-pattern/scripts/generate.py <图片> --palette mard --mode simple --out output/<名字>
```

3. 按需加选项：
   - 色卡：`--palette manman|mard|coco|mixiaowo|panpan`（命令行默认 manman）
   - 板数：`--size 29x29`（单板）/ `58x58`（四拼）/ `87x87`（九拼），或 `宽x高`
   - 照片渐变：`--dither on`；抠图：`--remove-bg flood|ai|none`（`ai` 需 `pip install rembg onnxruntime`）
   - 主体铺满格子：`--crop-subject`；照片高还原常用 `--remove-bg ai --crop-subject --max-side 80`
4. 打开输出目录里的 `pattern.png`（色号网格）、`shopping-list.csv` / `shopping-list.png`、`pattern.pdf`。想看成品长什么样：

```bash
python bead-pattern/scripts/preview.py output/<名字>
```

5. 把图纸和清单直接发给用户，用用户的语言说明格子数、色数、总颗数。屏幕色是实物近似值，采购以实物色卡为准。

更细的档位说明、白点校正（`scripts/wb.py`）和去碎块（`scripts/keep_main.py`）见 `bead-pattern/SKILL.md`。

## 立体盒子套件

1. 要一张干净、正面、单角色的图（白底或透明底）。像素画尽量一像素一格，正面板才能和原图对齐。
2. 盒子式（正面板 + 背面板 + 侧条凸起/卡槽 + 垫条上的凸起小件）：

```bash
python bead3d/bead3d.py from-image /path/char.png --style box \
  --out output/<名字> --title <名字> --palette mard \
  [--depth 5] [--height 36] [--accessory-colors F18,A11] \
  [--accessory-box y0:y1,x0:x1] [--back-color <色号>] [--no-model | --shape <glb>]
```

   - 默认会调 Hunyuan3D-2 取深度和起伏。Space 睡着、没额度，或模型是一块平板时，深度退回宽度的大约 12%（4–6 颗）。像素画通常是平的，凸起部位用 `--accessory-colors` 或 `--accessory-box` 指定。
   - 更小、更好拼：`--height 36 --depth 4`。不想联网：`--no-model`。
   - 已有网格：`python bead3d/bead3d.py from-model model.glb --image char.png --style box --out output/<名字> --palette mard`
   - 色卡脚本默认在 `bead-pattern/scripts`（相对 `bead3d/` 是 `../bead-pattern/scripts`）。别处放色卡时设 `BEAD_PATTERN_SCRIPTS` 或传 `--bead-pattern-scripts`。
3. 只要圆雕、不要盒子时，去掉 `--style box`，用 `--target-beads 3000` 或 `--height 32`。这是旧的分层切片，用户明确要圆雕再走这条。
4. 读终端打印的 JSON（`total_beads`、`colors`、件数、`warnings`）。自己先看 `preview_iso.png`、`sheet.png`，再发给用户。
5. 直接发图，不要打包：`preview_iso.png`、`sheet.png`、`assembly-steps.png`。用户再要单件图时给 `parts_1..N.png`。用用户的语言概括：高×宽×深（颗，以及大约厘米：每颗 5mm）、件数、各色号颗数、任意色颗数、警告。

## 注意

- 立体套件的实物配合还没有试过。告诉用户这是设计稿，熨烫时要轻、要平，凸起才塞得进孔。短侧条、凸起少的接口最松。超过 29 颗的大片要分多块底板。
- Hunyuan Space 可能要几分钟才醒，或免费额度用完。失败时说明深度是手填的，或改用 `--no-model` / `from-model`。
- 依赖没装时，在仓库根目录：`python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`。
