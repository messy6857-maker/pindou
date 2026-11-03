"""bead3d box style — MARD-style 2.5D "box" interlocking kit (盒子式插接拼豆).

Flat ironed panels only:  FRONT panel (character art) + BACK panel (back art) held apart by perpendicular
SIDE strips (left/right = YZ plane, top/bottom = XZ plane) that follow the silhouette, plus optional raised
ACCESSORY layers mounted on short spacer strips.  Joints = tab/slot: a strip edge has a 1-bead tab every other
bead; the panel has a 1-bead hole (empty peg) at exactly that spot; the tab bead fills the hole and shows the
panel's colour there, so the front still reads as the full picture.  Glue-free friction fit.
Grid: col[y,x,z], y top->bottom, x left->right, z=0 front.  -1 empty, -2 hidden support (任意色).
"""
import os, csv, json
from collections import Counter
import numpy as np
from scipy import ndimage as ND
import render_vox as RV

SUP = -2
C4 = np.array([[0,1,0],[1,1,1],[0,1,0]], bool)

# ------------------------------------------------------------------ art
def pixel_art(path, pal, height=None, replace=None, maxc=None):
    """Exact pixel-art image (one pixel per bead, few colours) -> codes grid. Returns None if not pixel art."""
    from PIL import Image
    import generate as G
    im = Image.open(path).convert('RGBA'); a = np.array(im)
    m = a[..., 3] > 128
    if not m.any(): return None
    ys, xs = np.where(m); a = a[ys.min():ys.max()+1, xs.min():xs.max()+1]
    if max(a.shape[:2]) > 100: return None
    if height and height != a.shape[0]:
        w = max(1, round(a.shape[1] * height / a.shape[0]))
        a = np.array(Image.fromarray(a).resize((w, height), Image.NEAREST))
    m = a[..., 3] > 128
    cols, inv = np.unique(a[..., :3][m].reshape(-1, 3), axis=0, return_inverse=True)
    if len(cols) > 32: return None
    code = np.argmin(G.ciede2000(G.srgb_to_lab(cols.astype(float)), pal.lab), 1)
    g = np.full(m.shape, -1, int); g[m] = code[inv.ravel()]
    return apply_replace(g, pal, replace)

def apply_replace(g, pal, replace):
    for r in replace or []:
        s, t = r.split('='); 
        if s in pal.codes and t in pal.codes: g[g == pal.codes.index(s)] = pal.codes.index(t)
    return g

def back_art(front, sil, pal, back_code=None, mode='clean', max_detail=60):
    """Back panel colours (as positioned in the grid, i.e. NOT mirrored): outline ring copied from the front,
    inside = per-row dominant body colour taken from just inside the outline (dark line colours skipped),
    so faces/eyes are not copied to the back."""
    Y, X = sil.shape
    ring = sil & ~ND.binary_erosion(sil, C4)
    if back_code is not None:
        return np.where(ring, front, np.where(sil, back_code, -1))
    if mode == 'clean':      # same art as the front, minus small face details (eyes, nose, mouth, highlights)
        b = front.copy(); inner = sil & ~ring
        for _ in range(3):
            small = np.zeros_like(sil)
            for c in np.unique(b[inner]):
                lab, n = ND.label((b == c) & inner, C4)
                sz = ND.sum(np.ones_like(lab), lab, range(1, n+1))
                for k, z in enumerate(sz, 1):
                    if z <= max_detail and not (ND.binary_dilation(lab == k, C4) & ring).any(): small |= lab == k
            if not small.any(): break
            _, (iy, ix) = ND.distance_transform_edt(small | ~sil, return_indices=True)
            b = np.where(small, b[iy, ix], b)
        return b
    band = ND.binary_erosion(sil, C4) & ~ND.binary_erosion(sil, C4, iterations=5)
    L = pal.lab[:, 0]; rowc = [-1]*Y
    for y in range(Y):
        v = front[max(0, y-2):y+3][band[max(0, y-2):y+3]]; v = v[v >= 0]
        vv = v[L[v] > 30] if (L[v] > 30).any() else v
        if len(vv): rowc[y] = Counter(vv.tolist()).most_common(1)[0][0]
    ks = [k for k in range(Y) if rowc[k] >= 0]
    for y in range(Y):
        if rowc[y] < 0 and ks: rowc[y] = rowc[min(ks, key=lambda k: abs(k-y))]
    back = np.full((Y, X), -1, int)
    for y in range(Y): back[y] = np.where(sil[y], rowc[y], -1)
    return np.where(ring, front, back)

# ------------------------------------------------------------------ strips
def runs_1d(v):
    out = []; i = 0; n = len(v)
    while i < n:
        if v[i]:
            j = i
            while j < n and v[j]: j += 1
            out.append((i, j)); i = j
        else: i += 1
    return out

def choose_strips(sil, min_len=5, max_strips=12, band_w=2):
    """Perimeter wall made of straight strips.  Candidates = straight vertical/horizontal runs inside a band
    `band_w` beads wide just inside the outline (silhouette inset by 1 so the panels keep a solid rim).
    Greedy picks the run covering most still-uncovered band cells; crossing strips are trimmed (butt joints).
    Returns list of (orient, fixed, a, b): 'v' -> column=fixed, rows a..b-1; 'h' -> row=fixed, cols a..b-1."""
    E = ND.binary_erosion(sil, C4)
    band = E & ~ND.binary_erosion(E, C4, iterations=band_w)
    occ = np.zeros_like(E); covered = np.zeros_like(E); strips = []
    total = band.sum()
    for _ in range(max_strips):
        best = None
        for orient in 'vh':
            M = band if orient == 'v' else band.T; O = occ if orient == 'v' else occ.T
            Cv = (band & ~covered) if orient == 'v' else (band & ~covered).T
            for f in range(M.shape[1]):
                for a, b in runs_1d(M[:, f] & ~O[:, f]):
                    if b - a < min_len: continue
                    gain = int(Cv[a:b, f].sum())
                    if gain >= min_len - 1 and (best is None or gain > best[0]): best = (gain, orient, f, a, b)
        if best is None: break
        _, o, f, a, b = best; strips.append((o, f, a, b))
        cells = np.zeros_like(E)
        if o == 'v': cells[a:b, f] = True
        else: cells[f, a:b] = True
        occ |= ND.binary_dilation(cells, C4) & ~cells | cells   # keep 1 bead between strips (no double walls)
        covered |= ND.binary_dilation(cells, np.ones((3, 3), bool), iterations=band_w)
        if (band & ~covered).sum() < 0.08 * total: break
    return strips

# ------------------------------------------------------------------ kit
class Kit:
    def __init__(s): s.pieces = []
    def add(s, name, kind, plane, cells, cols, holes=()):
        s.pieces.append(dict(name=name, kind=kind, plane=plane, cells=np.array(cells, int), cols=np.array(cols, int),
                             holes=np.array(list(holes), int).reshape(-1, 3), tabs=[]))
        return len(s.pieces) - 1

def design_box(front, sil, back, D, accessories, pal, side_mode='back', spacer_every=6):
    Y, X = sil.shape; has_acc = len(accessories) > 0
    zF = 2 if has_acc else 0; zB = zF + D + 1
    K = Kit(); hole_owner = {}            # (y,x,z) -> piece index providing the tab
    strips = choose_strips(sil)
    sidecol = back if side_mode == 'back' else front
    tabsF, tabsB = set(), set()
    strip_cells = set()
    sp = []
    for o, f, a, b in strips:
        L = b - a; cells, cols = [], []; tabs = []
        for i in range(L):
            y, x = (a+i, f) if o == 'v' else (f, a+i)
            strip_cells.add((y, x))
            for z in range(zF+1, zB):
                cells.append((y, x, z)); cols.append(sidecol[y, x])
            if i % 2 == 1 and i < L-1:
                cells.append((y, x, zF)); cols.append(front[y, x]); tabsF.add((y, x))
                cells.append((y, x, zB)); cols.append(back[y, x]); tabsB.add((y, x))
                tabs.append((y, x))
        sp.append((o, f, a, b, cells, cols, tabs))
    # panels
    fc, fcol, bc, bcol = [], [], [], []
    acc_union = np.zeros_like(sil)
    for A in accessories: acc_union |= A['mask']
    hid = ND.binary_erosion(acc_union, C4, iterations=2)     # panel cells hidden under raised pieces -> 任意色
    for y, x in zip(*np.where(sil)):
        if (y, x) not in tabsF:
            fc.append((y, x, zF)); fcol.append(SUP if hid[y, x] else front[y, x])
        if (y, x) not in tabsB:
            bc.append((y, x, zB)); bcol.append(back[y, x])
    iF = K.add('正面板', 'front', 'XY', fc, fcol, [(y, x, zF) for y, x in sorted(tabsF)])
    for n, (o, f, a, b, cells, cols, tabs) in enumerate(sp):
        nm = ('竖侧条' if o == 'v' else '横侧条')
        K.add(nm, 'side_v' if o == 'v' else 'side_h', 'YZ' if o == 'v' else 'XZ', cells, cols)
        K.pieces[-1]['tabs'] = tabs
    iB = K.add('背面板', 'back', 'XY', bc, bcol, [(y, x, zB) for y, x in sorted(tabsB)])
    # accessories + spacers
    usedF = set(tabsF)
    for A in accessories:
        M = A['mask']; inner = ND.binary_erosion(M, C4)
        if not inner.any(): inner = M
        ys = np.where(inner.any(1))[0]; nsp = max(1, round(len(ys) / spacer_every))
        rows = []
        for k in range(nsp):
            seg = ys[int(k*len(ys)/nsp):int((k+1)*len(ys)/nsp)]
            r = max(seg, key=lambda y: max((b-a for a, b in runs_1d(inner[y])), default=0)); rows.append(r)
        sp_specs = []
        for r in rows:
            a, b = max(runs_1d(inner[r]), key=lambda t: t[1]-t[0]); L = b - a
            pos = [i for i in range(L) if (i % 2 == 1 and i < L-1)] or [L//2]
            pos = [i for i in pos if (r, a+i) not in usedF and (r, a+i) not in strip_cells] or \
                  [i for i in range(L) if (r, a+i) not in usedF and (r, a+i) not in strip_cells][:1]
            if not pos: continue
            sp_specs.append((r, a, b, pos))
        acc_holes = {(r, a+i) for r, a, b, pos in sp_specs for i in pos}
        ac, acol = [], []
        for y, x in zip(*np.where(M)):
            if (y, x) not in acc_holes: ac.append((y, x, 0)); acol.append(front[y, x])
        ia = K.add(A['name'], 'acc', 'XY', ac, acol, [(y, x, 0) for y, x in sorted(acc_holes)])
        for r, a, b, pos in sp_specs:
            cells, cols = [], []
            for i in range(a, b): cells.append((r, i, 1)); cols.append(front[r, i])
            for i in pos:
                cells.append((r, a+i, 0)); cols.append(front[r, a+i])
                cells.append((r, a+i, zF)); cols.append(SUP if hid[r, a+i] else front[r, a+i])
                usedF.add((r, a+i))
            K.add(A['name'] + '垫条', 'spacer', 'XZ', cells, cols); K.pieces[-1]['tabs'] = [(r, a+i) for i in pos]
            K.pieces[-1]['acc'] = ia
        # the front panel needs holes for the spacer tabs
    # punch spacer holes into the front panel
    fr = K.pieces[iF]; keep = [k for k, c in enumerate(fr['cells']) if (c[0], c[1]) not in usedF]
    newh = [(y, x, zF) for (y, x) in sorted(usedF)]
    fr['cells'] = fr['cells'][keep]; fr['cols'] = fr['cols'][keep]; fr['holes'] = np.array(newh, int).reshape(-1, 3)
    # order: front, sides, back, spacers, accessories (spacers right before their accessory)
    order = [iF] + [i for i, p in enumerate(K.pieces) if p['kind'].startswith('side')] + [iB]
    for i, p in enumerate(K.pieces):
        if p['kind'] == 'acc':
            order += [j for j, q in enumerate(K.pieces) if q.get('acc') == i] + [i]
    K.order = order; K.shape = (Y, X, zB + 1); K.zF, K.zB, K.D = zF, zB, D
    return K

def validate_box(K):
    errs = []; occ = Counter()
    for p in K.pieces:
        for c in map(tuple, p['cells']): occ[c] += 1
    if any(v > 1 for v in occ.values()): errs.append('overlapping beads: %d' % sum(1 for v in occ.values() if v > 1))
    ax = {'XY': 2, 'YZ': 1, 'XZ': 0}
    for n, i in enumerate(K.order, 1):
        p = K.pieces[i]; a = ax[p['plane']]; c = p['cells']
        if not (c[:, a] == c[0, a]).all(): errs.append(f'#{n} not flat'); continue
        o = [t for t in range(3) if t != a]; m = np.zeros(tuple(c[:, t].max()+1 for t in o), bool); m[c[:, o[0]], c[:, o[1]]] = True
        if ND.label(m, C4)[1] != 1: errs.append(f'#{n} {p["name"]} not connected')
    # every hole filled by exactly one tab
    for p in K.pieces:
        for h in map(tuple, p['holes']):
            if occ.get(h, 0) != 1: errs.append(f'hole {h} in {p["name"]} not filled')
    return errs

def grid_of(K, ids=None):
    col = np.full(K.shape, -1, int)
    for i in (range(len(K.pieces)) if ids is None else ids):
        p = K.pieces[i]; col[tuple(p['cells'].T)] = p['cols']
    return col

# ------------------------------------------------------------------ mesh preview
def mesh_preview(path, out_png, size=900):
    import trimesh
    from PIL import Image, ImageDraw
    m = trimesh.load(path, force='mesh')
    if len(m.faces) > 60000: 
        try: m = m.simplify_quadric_decimation(face_count=60000)
        except Exception: pass
    v = m.vertices - m.bounds.mean(0); v /= np.abs(v).max()
    imgs = []
    for yaw in (-30, 0, 60, 180):
        t = np.radians(yaw); R = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
        p = np.radians(-15); P = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
        w = v @ R.T @ P.T; tri = w[m.faces]; n = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        vis = n[:, 2] > 0; L = np.array([0.4, 0.6, 0.7]); L /= np.linalg.norm(L)
        sh = 0.35 + 0.65*np.clip(n @ L, 0, 1)
        S = size//2; img = Image.new('RGB', (S, S), (245, 245, 250)); d = ImageDraw.Draw(img)
        idx = np.where(vis)[0]; idx = idx[np.argsort(tri[idx, :, 2].mean(1))]
        for k in idx:
            pts = [(S/2 + q[0]*S*0.45, S/2 - q[1]*S*0.45) for q in tri[k]]
            c = int(70 + 170*sh[k]); d.polygon(pts, fill=(c, int(c*0.97), int(c*0.9)))
        imgs.append(img)
    big = Image.new('RGB', (size*2, size//2), 'white')
    for i, im in enumerate(imgs): big.paste(im, (i*size//2, 0))
    big.save(out_png); return out_png

# ------------------------------------------------------------------ drawing
def F(s, b=False):
    from PIL import ImageFont
    return ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-%s.ttc' % ('Bold' if b else 'Regular'), s)

def piece_img(K, i, num, pal, cs=16, label=True):
    """Flat drawing of piece i as you lay it on the pegboard. Holes = empty peg with red ring; tabs = red border."""
    from PIL import Image, ImageDraw
    p = K.pieces[i]; c = p['cells']; h = p['holes']
    allc = np.vstack([c, h]) if len(h) else c
    if p['plane'] == 'XY': r, q, flip = 0, 1, p['kind'] == 'back'   # back panel drawn as seen from behind
    elif p['plane'] == 'YZ': r, q, flip = 0, 2, False
    else: r, q, flip = 2, 1, False
    r0, q0 = allc[:, r].min(), allc[:, q].min(); H = allc[:, r].max()-r0+1; W = allc[:, q].max()-q0+1
    top = int(cs*1.7) if label else 0
    img = Image.new('RGB', (W*cs+8, H*cs+8+top), 'white'); d = ImageDraw.Draw(img)
    tabset = set(map(tuple, (p['tabs'] or [])))
    def xy(cc):
        cq = cc[q]-q0; cq = W-1-cq if flip else cq
        return 4 + cq*cs, 4 + top + (cc[r]-r0)*cs
    lum = lambda rgb: 0.299*rgb[0]+0.587*rgb[1]+0.114*rgb[2]
    for cc, v in zip(c, p['cols']):
        x0, y0 = xy(cc); rgb = RV.rgb_of(int(v))
        istab = (p['kind'] in ('side_v', 'side_h') and cc[2] in (K.zF, K.zB)) or (p['kind'] == 'spacer' and cc[2] in (0, K.zF))
        d.ellipse([x0+1, y0+1, x0+cs-1, y0+cs-1], fill=rgb, outline=(220, 20, 20) if istab else (90, 90, 90), width=3 if istab else 1)
        if cs >= 22 and v != SUP:
            d.text((x0+cs/2, y0+cs/2), pal.codes[int(v)], font=F(int(cs*0.34), True), fill=(0, 0, 0) if lum(rgb) > 140 else (255, 255, 255), anchor='mm')
    for cc in h:
        x0, y0 = xy(cc); d.ellipse([x0+3, y0+3, x0+cs-3, y0+cs-3], fill='white', outline=(230, 20, 20), width=2)
    if label: d.text((4, 2), f'#{num[i]} {p["name"]}', font=F(int(cs*1.0), True), fill=(30, 30, 30))
    return img

def legend(cnt, pal, sup_label, W):
    from PIL import Image, ImageDraw
    rows = sorted([(k, v) for k, v in cnt.items() if k != SUP], key=lambda t: -t[1])
    if cnt.get(SUP): rows.append((SUP, cnt[SUP]))
    per = 5; nr = (len(rows)+per-1)//per
    img = Image.new('RGB', (W, 70 + nr*56), 'white'); d = ImageDraw.Draw(img)
    d.text((30, 10), 'MARD小豆色号 / 用量', font=F(30, True), fill=(60, 60, 60))
    for k, (code, v) in enumerate(rows):
        x = 30 + (k % per)*(W-60)//per; y = 60 + (k//per)*56
        d.rectangle([x, y, x+44, y+44], fill=RV.rgb_of(code), outline=(80, 80, 80), width=2)
        d.text((x+56, y+6), f'{sup_label if code == SUP else pal.codes[code]}  ×{v}', font=F(26, True), fill=(40, 40, 40))
    return img

def flow(items, W, gap=24):
    rows, cur, x, h = [], [], 0, 0
    for im in items:
        if cur and x + im.width > W: rows.append((cur, h)); cur, x, h = [], 0, 0
        cur.append((x, im)); x += im.width + gap; h = max(h, im.height)
    if cur: rows.append((cur, h))
    return rows

def exploded(K, gap=10):
    """Pieces pulled apart along their insertion directions (accessories forward, back panel backward,
    side strips outward)."""
    Y, X, Z = K.shape; cy, cx = Y/2, X/2
    dz = {'acc': -3*gap, 'spacer': -int(1.6*gap), 'front': 0, 'side_v': gap, 'side_h': gap, 'back': int(2.6*gap)}
    PAD = 3*gap + 4; big = np.full((Y + 2*PAD, X + 2*PAD, Z + 6*gap + 2*PAD), -1, int)
    for i in K.order:
        p = K.pieces[i]; c = p['cells'].copy(); k = p['kind']; off = np.array([PAD, PAD, PAD + 3*gap + dz[k]])
        if k == 'side_v': off[1] += int(np.sign(c[0, 1] - cx + 0.1) * gap * 0.6)
        if k == 'side_h': off[0] += int(np.sign(c[0, 0] - cy + 0.1) * gap * 0.6)
        cc = np.clip(c + off, 0, np.array(big.shape)-1); big[tuple(cc.T)] = p['cols']
    nz = [np.where((big != -1).any(axis=t))[0] for t in ((1, 2), (0, 2), (0, 1))]
    return big[nz[0].min():nz[0].max()+1, nz[1].min():nz[1].max()+1, nz[2].min():nz[2].max()+1]

def outputs_box(K, pal, out, title, sup_label, model_png=None):
    from PIL import Image, ImageDraw
    RV.PAL_RGB = pal.rgb; num = {i: n+1 for n, i in enumerate(K.order)}
    col = grid_of(K); Y, X, Z = K.shape
    cnt = Counter(int(v) for p in K.pieces for v in p['cols']); total = sum(cnt.values())
    rows_c = sorted([(k, v) for k, v in cnt.items() if k != SUP], key=lambda t: -t[1])
    with open(f'{out}/shopping-list.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f); w.writerow(['色号', 'hex', '数量'])
        for k, v in rows_c: w.writerow([pal.codes[k], pal.hexes[k], v])
        if cnt.get(SUP): w.writerow([f'{sup_label}(被挡住,任意色)', '', cnt[SUP]])
        w.writerow(['总计', '', total])
    with open(f'{out}/pieces.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f); w.writerow(['件号', '名称', '颗数', '孔(凹口)', '凸起', '尺寸(颗)'])
        for i in K.order:
            p = K.pieces[i]; c = p['cells']; ext = c.max(0) - c.min(0) + 1
            w.writerow([num[i], p['name'], len(c), len(p['holes']), len(p['tabs']) * (2 if p['kind'] != 'acc' else 0), '×'.join(map(str, ext))])
    pv = {'front': RV.render(col, 0, 8, 12), 'iso': RV.render(col, -35, 22, 12), 'side': RV.render(col, -90, 10, 12), 'back': RV.render(col, 180, 8, 12)}
    for n, im in pv.items(): im.save(f'{out}/preview_{n}.png')
    ex = RV.render(exploded(K), -52, 22, 8); ex.save(f'{out}/exploded.png')
    # per-part images
    for f in os.listdir(out):
        if f.startswith('parts_') and f.endswith('.png'): os.remove(f'{out}/{f}')
    files = []
    for i in K.order:
        p = K.pieces[i]; im = piece_img(K, i, num, pal, cs=30)
        n_ = len(p['cells']); cc = Counter(int(v) for v in p['cols'])
        info = Image.new('RGB', (max(im.width, 900), im.height + 110), 'white'); info.paste(im, (0, 0)); d = ImageDraw.Draw(info)
        d.text((8, im.height + 4), f'{n_}颗 · 白圈红边 = 留空的孔(凹口)，别放豆；红边豆 = 凸起(插进对应的孔)', font=F(22), fill=(200, 30, 30))
        d.text((8, im.height + 40), '  '.join(f'{sup_label if k == SUP else pal.codes[k]}×{v}' for k, v in cc.most_common()), font=F(22, True), fill=(40, 40, 40))
        if p['kind'] == 'back': d.text((8, im.height + 74), '背面板按"从背后看"的样子画（已左右镜像）', font=F(20), fill=(90, 90, 90))
        if p['kind'] in ('side_v', 'side_h'): d.text((8, im.height + 74), '左边一列/上边一行凸起插正面板，另一边插背面板', font=F(20), fill=(90, 90, 90))
        if p['kind'] == 'spacer': d.text((8, im.height + 74), '一边凸起插正面板（绿色=任意色，被挡住），另一边插立体小件', font=F(20), fill=(90, 90, 90))
        fn = f'{out}/parts_{num[i]}.png'; info.save(fn); files.append(fn)
    # sheet (like the MARD card: legend, all flat parts, assembled + exploded)
    W = 2000; blocks = []
    head = Image.new('RGB', (W, 140), 'white'); d = ImageDraw.Draw(head)
    d.rounded_rectangle([20, 12, W-20, 128], radius=28, fill=(232, 245, 222), outline=(80, 140, 60), width=4)
    d.text((W//2, 52), f'{title} · 盒子式插接拼豆', font=F(56, True), fill=(40, 100, 40), anchor='mm')
    d.text((W//2, 104), f'{len(K.pieces)}件 · 宽{X}×高{Y}×厚{Z}颗 · 共{total}颗 · {len(rows_c)}色' + (f'+{sup_label}{cnt[SUP]}' if cnt.get(SUP) else '') + f' · 免胶插接 · {pal.name.upper()}', font=F(28), fill=(60, 90, 60), anchor='mm')
    blocks.append(head); blocks.append(legend(cnt, pal, sup_label, W))
    groups = [('① 正面板 + 背面板', ['front', 'back']), ('② 侧条（立在两块面板之间）', ['side_v', 'side_h']), ('③ 立体小件 + 垫条', ['spacer', 'acc'])]
    for gt, ks in groups:
        ids = [i for i in K.order if K.pieces[i]['kind'] in ks]
        if not ids: continue
        cs = 14 if ks[0] == 'front' else 16
        rows = flow([piece_img(K, i, num, pal, cs) for i in ids], W - 60)
        b = Image.new('RGB', (W, 50 + sum(h + 20 for _, h in rows)), (238, 238, 238)); dd = ImageDraw.Draw(b)
        dd.text((30, 8), gt, font=F(28, True), fill=(60, 60, 60)); y = 50
        for line, h in rows:
            for x, im in line: b.paste(im, (30 + x, y))
            y += h + 20
        blocks.append(b)
    vb = Image.new('RGB', (W, 760), 'white'); d = ImageDraw.Draw(vb)
    tiles = [(pv['iso'], '拼好后'), (pv['back'], '背面'), (ex, '分解图')] + ([(Image.open(model_png), '3D模型')] if model_png else [])
    tw = (W - 40) // len(tiles)
    for k, (im, lab) in enumerate(tiles):
        t = im.copy(); t.thumbnail((tw - 20, 680)); vb.paste(t, (20 + k*tw + (tw - t.width)//2, 60), t if t.mode == 'RGBA' else None)
        d.text((20 + k*tw + 10, 14), lab, font=F(30, True), fill=(60, 60, 60))
    blocks.append(vb)
    note = Image.new('RGB', (W, 110), 'white'); d = ImageDraw.Draw(note)
    d.text((30, 10), '白圈红边 = 面板上留空的孔（凹口）；红边豆 = 侧条/垫条的凸起，熨好后插进对应的孔，凸起那颗豆的颜色就是面板那一格的颜色。', font=F(24), fill=(180, 30, 30))
    d.text((30, 52), '熨烫：两面都轻熨、保持平整，孔不要熨变形；先试插，太紧就把凸起豆稍微修一下。背面板按从背后看的样子画。', font=F(24), fill=(90, 90, 90))
    blocks.append(note)
    sheet = Image.new('RGB', (W, sum(b.height for b in blocks)), 'white'); y = 0
    for b in blocks: sheet.paste(b, (0, y)); y += b.height
    sheet.save(f'{out}/sheet.png')
    # assembly steps
    kinds = lambda ks: [i for i in K.order if K.pieces[i]['kind'] in ks]
    sides = kinds(['side_v', 'side_h'])
    steps = [('正面板朝下平放', kinds(['front']), '#1 图案朝下放平，孔朝上'),
             ('插上侧条', kinds(['front']) + sides, f'#{num[sides[0]]}–#{num[sides[-1]]} 凸起对准正面板的孔竖着插' if sides else ''),
             ('盖上背面板', kinds(['front', 'back']) + sides, f'#{num[kinds(["back"])[0]]} 对准侧条上端凸起压下去')]
    if kinds(['acc']):
        steps.append(('装立体小件', list(range(len(K.pieces))), '翻到正面：垫条凸起插进正面板的孔，再把小件按到垫条上'))
    tiles = []
    for k, (t, ids, sub) in enumerate(steps):
        g = grid_of(K, ids); im = RV.render(g, -35 if k == len(steps)-1 and kinds(['acc']) else 215, 30, 9)
        tile = Image.new('RGB', (480, 560), 'white'); d = ImageDraw.Draw(tile)
        tt = im.copy(); tt.thumbnail((460, 440)); tile.paste(tt, ((480 - tt.width)//2, 100), tt)
        d.text((12, 8), f'步骤{k+1}：{t}', font=F(30, True), fill=(40, 100, 40)); d.text((12, 54), sub, font=F(17), fill=(90, 90, 90))
        tiles.append(tile)
    exw = 1000; t = ex.copy(); t.thumbnail((exw, 1100))
    S = Image.new('RGB', (max(20 + 490*len(tiles), exw + 40), 650 + t.height + 80), 'white'); d = ImageDraw.Draw(S)
    d.text((20, 12), f'{title} 组装步骤（免胶插接）', font=F(40, True), fill=(40, 100, 40))
    for k, tile in enumerate(tiles): S.paste(tile, (20 + k*490, 70))
    d.text((20, 650), '分解图：各件沿插入方向拉开', font=F(28, True), fill=(60, 60, 60))
    S.paste(t, ((S.width - t.width)//2, 700), t)
    S.save(f'{out}/assembly-steps.png')
    return dict(total=total, colors={pal.codes[k]: v for k, v in rows_c}, support=cnt.get(SUP, 0), parts_files=files, num=num)

# ------------------------------------------------------------------ driver
def run_box(args, pal, out, model=None):
    import bead3d as B
    log = B.log; warns = []
    front = pixel_art(args.image, pal, args.height, args.replace) if args.image else None
    if front is None:
        im = B.image_rgba(args.image); H = args.height or 40
        Wd = max(1, round(im.width * H / im.height)); a = np.array(im.resize((Wd, H)))
        sil = ND.binary_fill_holes(a[..., 3] > 128)
        front = B.front_map_from_image(args.image, sil, pal, args.max_colors, args.boost)
        front = apply_replace(front, pal, args.replace)
    sil = ND.binary_fill_holes(front >= 0)
    if (sil & (front < 0)).any():
        _, (iy, ix) = ND.distance_transform_edt(front < 0, return_indices=True); front = np.where(sil, front[iy, ix], -1)
    Y, X = sil.shape
    # depth & protrusions from the 3D model
    D = args.depth; relief = None; model_png = None
    if model:
        try:
            mesh = B.load_mesh(model, args.up, args.yaw)
            model_png = mesh_preview(model, f'{out}/model_preview.png')
            ex = mesh.extents; ratio = ex[2] / ex[1]
            if ratio < 0.08: warns.append(f'3D model is almost flat (depth/height={ratio:.2f}); box depth uses --depth / default instead')
            else:
                g, _ = B.voxelize(mesh, Y); t = g.sum(2); ms = g.any(2)
                if D is None: D = int(np.clip(round(np.median(t[ms]) * 0.6) - 2, 3, 8))
                zf = np.where(ms, np.argmax(g, 2), 10**6); base = np.percentile(zf[ms], 60)
                rel = ms & (zf <= base - 2)
                if rel.any():   # resample to front-art grid
                    from PIL import Image
                    relief = np.array(Image.fromarray(rel.astype(np.uint8)*255).resize((X, Y), Image.NEAREST)) > 0
        except Exception as e:
            warns.append(f'3D model not used: {e!r}')
    if D is None: D = int(np.clip(round(X * 0.12), 4, 6))
    back = back_art(front, sil, pal, pal.codes.index(args.back_color) if args.back_color else None)
    # accessories
    accs = []
    cand = []
    if args.accessory_colors:
        codes = [pal.codes.index(c) for c in args.accessory_colors.split(',') if c in pal.codes]
        cand.append(np.isin(front, codes) & sil)
    elif relief is not None: cand.append(relief & sil)
    for spec in (args.accessory_box or []):        # y0:y1,x0:x1 rectangle -> silhouette region inside it
        y0, y1, x0, x1 = map(int, spec.replace(':', ',').split(','))
        m = np.zeros_like(sil); m[y0:y1, x0:x1] = True; cand.append(m & sil)
    dark = pal.lab[:, 0] < 30
    for cm in cand:
        lab, n = ND.label(cm, C4)
        for k in range(1, n+1):
            m = ND.binary_fill_holes(lab == k) & sil
            if m.sum() < args.min_accessory: continue
            # add the adjacent dark outline so the raised piece keeps its black border
            ring = ND.binary_dilation(m, C4) & sil & ~m & dark[np.maximum(front, 0)]
            m2 = m | ring
            if ND.label(m2, C4)[1] != 1: m2 = m
            yy, xx = np.where(m2); cy_, cx_ = yy.mean()/Y, xx.mean()/X
            pos = ('上' if cy_ < 0.34 else '中' if cy_ < 0.67 else '下') + ('左' if cx_ < 0.4 else '右' if cx_ > 0.6 else '')
            accs.append(dict(mask=m2, name=f'立体小件{len(accs)+1}({pos})'))
    # overlapping accessories -> merge-free: drop later overlaps
    used = np.zeros_like(sil); acc2 = []
    for A in accs:
        if (A['mask'] & used).any(): A['mask'] &= ~used
        if A['mask'].sum() >= args.min_accessory and ND.label(A['mask'], C4)[1] == 1: acc2.append(A); used |= A['mask']
    K = design_box(front, sil, back, D, acc2, pal)
    errs = validate_box(K)
    sup_label = '任意色'
    if args.support_color not in ('任意色', 'any') and args.support_color in pal.codes:
        si = pal.codes.index(args.support_color); sup_label = args.support_color
        for p in K.pieces: p['cols'][p['cols'] == SUP] = si
    res = outputs_box(K, pal, out, args.title, sup_label, model_png)
    num = res['num']
    for i in K.order:
        p = K.pieces[i]
        if p['kind'].startswith('side') and len(p['tabs']) < 2: warns.append(f'#{num[i]} {p["name"]} has only {len(p["tabs"])} tab pair: weak, iron well')
        if p['kind'] == 'spacer' and len(p['tabs']) < 2: warns.append(f'#{num[i]} {p["name"]} holds its accessory with 1 tab only: may rotate, press firmly')
    warns.append('friction fit: no piece is mechanically locked; panels are held only by the tabs pressed into holes (same as MARD box kits)')
    summary = dict(style='box', out=out, model=model, size_WHD=[X, Y, K.shape[2]], depth_gap=D, pieces=len(K.pieces),
                   kinds=dict(Counter(p['kind'] for p in K.pieces)), total_beads=res['total'], colors=res['colors'],
                   support_beads=res['support'], support_label=sup_label, tab_pairs=int(sum(len(p['tabs']) for p in K.pieces)),
                   errors=errs, warnings=warns,
                   files=dict(sheet=f'{out}/sheet.png', parts=res['parts_files'], steps=f'{out}/assembly-steps.png',
                              exploded=f'{out}/exploded.png', model_preview=model_png,
                              previews=[f'{out}/preview_{n}.png' for n in ('front', 'iso', 'side', 'back')],
                              shopping=f'{out}/shopping-list.csv', pieces=f'{out}/pieces.csv'))
    json.dump(summary, open(f'{out}/summary.json', 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return summary
