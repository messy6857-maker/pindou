#!/usr/bin/env python3
"""bead3d — turn an image or a 3D model into a glue-free, interlocking 3D 拼豆 (fuse-bead) kit.

Pipeline: shape (Hunyuan3D-2 HF Space | your mesh) -> voxelize -> palette colours (CIEDE2000, bead-pattern palettes)
 -> hollow design: horizontal layer rings + hidden cross "core" plates (half-lap) -> pegs / keys / caps to lock loose
 parts -> validation (overlap, flat+connected, disassembly-order search, per-piece lock state) -> PNG/CSV outputs.
See README.md.  Coordinates: grid[y,x,z] with y top->bottom, x left->right, z front(0)->back.
"""
import argparse, json, os, sys, shutil, csv
from collections import Counter
import numpy as np
from scipy import ndimage as ND

HERE = os.path.dirname(os.path.abspath(__file__))
# bead-pattern lives next to this package in the repo (bead3d/../bead-pattern/scripts).
# Override with BEAD_PATTERN_SCRIPTS or --bead-pattern-scripts (parsed before import).
_DEFAULT_BP_SCRIPTS = os.path.normpath(os.path.join(HERE, '..', 'bead-pattern', 'scripts'))

def _resolve_bp_scripts(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    for i, a in enumerate(argv):
        if a == '--bead-pattern-scripts' and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith('--bead-pattern-scripts='):
            return a.split('=', 1)[1]
    return os.environ.get('BEAD_PATTERN_SCRIPTS', _DEFAULT_BP_SCRIPTS)

BP_SCRIPTS = _resolve_bp_scripts()
sys.path.insert(0, BP_SCRIPTS); sys.path.insert(0, HERE)
try:
    import generate as G      # bead-pattern: load_palette, srgb_to_lab, ciede2000, limit_colors
except ModuleNotFoundError as exc:
    sys.exit(f'[bead3d] cannot import generate.py from {BP_SCRIPTS} ({exc}). '
             'Expected ../bead-pattern/scripts next to bead3d, or set BEAD_PATTERN_SCRIPTS / --bead-pattern-scripts.')
import render_vox as RV

SUPPORT = -2                  # hidden interior / structural cell ("任意色")
DIRS = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
NAMES = {(1,0,0):'下',(-1,0,0):'上',(0,1,0):'右',(0,-1,0):'左',(0,0,1):'后',(0,0,-1):'前'}
log = lambda *a: print('[bead3d]', *a, file=sys.stderr, flush=True)

# ------------------------------------------------------------------ shape
def hunyuan_shape(image, out_glb, space='tencent/Hunyuan3D-2', seed=1234):
    from gradio_client import Client, handle_file
    import time
    tok = os.environ.get('HF_TOKEN'); c = None; last = None
    for attempt in range(3):                      # Spaces often need a minute to wake up
        try:
            c = Client(space, verbose=False, token=tok, httpx_kwargs={'timeout': 300}); break
        except Exception as e:
            last = e; log('connect attempt', attempt + 1, 'failed:', repr(e)[:150]); time.sleep(20)
    if c is None: raise RuntimeError(f'Hunyuan3D-2 Space unreachable ({space}): {last!r}. Use from-model with your own mesh.')
    for api in ('/shape_generation', '/generation_all'):
        try:
            r = c.predict(caption=None, image=handle_file(image), steps=30, guidance_scale=5.0, seed=seed,
                          octree_resolution=256, check_box_rembg=True, num_chunks=8000, randomize_seed=False, api_name=api)
            v = r[0]['value'] if isinstance(r[0], dict) else r[0]
            shutil.copy(v, out_glb); log('Hunyuan3D-2 ok via', api); return out_glb
        except Exception as e:
            last = e; log('Hunyuan api', api, 'failed:', repr(e)[:200])
    raise RuntimeError(f'Hunyuan3D-2 Space unavailable: {last!r}')

def load_mesh(path, up='y', yaw=0.0):
    import trimesh
    m = trimesh.load(path, force='mesh')
    if up == 'z': m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [1,0,0]))
    if yaw: m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(yaw), [0,1,0]))
    return m

def voxelize(m, H):
    v = m.voxelized(m.extents[1] / H)
    occ = ND.binary_fill_holes(v.matrix)
    for ax in range(3):
        for i in range(occ.shape[ax]):
            s = [slice(None)]*3; s[ax] = i; occ[tuple(s)] = ND.binary_fill_holes(occ[tuple(s)])
    g = np.transpose(occ, (1,0,2))[::-1, :, ::-1].copy()          # mesh +y up, +z front -> grid
    lab, n = ND.label(g)
    if n > 1: g = lab == 1 + int(np.argmax(ND.sum(g, lab, range(1, n+1))))
    zz, yy, xx = (np.where(g.any(a))[0] for a in ((0,1),(1,2),(0,2)))
    g = g[yy[0]:yy[-1]+1, xx[0]:xx[-1]+1, zz[0]:zz[-1]+1]
    # world coords of grid cell centres (for mesh-colour sampling)
    pitch = m.extents[1] / H
    return g, (v.transform, pitch, (yy[0], xx[0], zz[0]), v.matrix.shape)

# ------------------------------------------------------------------ colour
def image_rgba(path):
    from PIL import Image
    im = Image.open(path).convert('RGBA'); a = np.array(im)
    if (a[..., 3] < 128).mean() < 0.01:                       # no alpha: flood-remove background from corners
        rgb = a[..., :3].astype(float); lab = G.srgb_to_lab(rgb.reshape(-1,3)).reshape(rgb.shape)
        corners = [lab[0,0], lab[0,-1], lab[-1,0], lab[-1,-1]]
        bg = np.zeros(a.shape[:2], bool)
        for c in corners: bg |= G.ciede2000(lab.reshape(-1,3), c[None])[:, 0].reshape(bg.shape) < 8
        L, _ = ND.label(bg); edge_l = set(np.unique(np.r_[L[0], L[-1], L[:, 0], L[:, -1]])) - {0}
        a[np.isin(L, list(edge_l)), 3] = 0
    ys, xs = np.where(a[..., 3] > 128)
    return Image.fromarray(a).crop((xs.min(), ys.min(), xs.max()+1, ys.max()+1))

def quantize_to_palette(rgb, pal, maxc, boost=1.0, seed=0, flatten=True):
    """k-means in Lab on the image itself, then snap each cluster centre to the nearest palette bead (CIEDE2000).
    Gives cleaner, more faithful colours than mapping every pixel to the full palette."""
    from scipy.cluster.vq import kmeans2
    rgb = np.clip(rgb * boost + (boost - 1) * 40, 0, 255)
    lab = G.srgb_to_lab(rgb)
    rng = np.random.default_rng(seed); samp = lab[rng.choice(len(lab), min(len(lab), 20000), replace=False)]
    cent, _ = kmeans2(samp, min(maxc + 4, len(samp)), minit='++', seed=seed)
    lab_c = np.argmin(((lab[:, None, :] - cent[None]) ** 2).sum(-1), 1)
    w = np.bincount(lab_c, minlength=len(cent)).astype(float)
    if flatten:   # merge shading variants (same hue, different lightness) into the dominant cluster
        C = np.hypot(cent[:, 1], cent[:, 2]); hue = np.degrees(np.arctan2(cent[:, 2], cent[:, 1]))
        tgt = np.arange(len(cent))
        for i in np.argsort(-w):
            for j in range(len(cent)):
                if j == i or tgt[j] != j or w[j] > w[i] or tgt[i] != i: continue
                dh = abs((hue[i] - hue[j] + 180) % 360 - 180)
                if C[i] > 15 and C[j] > 12 and dh < 15 and abs(C[i] - C[j]) < max(12, 0.3 * max(C[i], C[j])) and cent[j, 0] > 22: tgt[j] = i
        lab_c = tgt[lab_c]
    code = np.argmin(G.ciede2000(cent, pal.lab), 1)
    idx = code[lab_c]
    return G.limit_colors(idx[None, :], pal, maxc)[0]

def front_map_from_image(img_path, sil, pal, maxc, boost=1.0, S=8):
    from PIL import Image
    Y, X = sil.shape; im = image_rgba(img_path)
    sy, sx = np.where(sil); y0, y1, x0, x1 = sy.min(), sy.max()+1, sx.min(), sx.max()+1
    small = np.array(im.resize(((x1-x0)*S, (y1-y0)*S), Image.LANCZOS)).astype(float)
    alpha = small[..., 3] > 100
    pix = np.full(alpha.shape, -1, int)
    pix[alpha] = quantize_to_palette(small[..., :3][alpha], pal, maxc, boost)
    lum = pal.lab[:, 0]
    front = np.full((Y, X), -1, int)
    for yy in range(y0, y1):
        for xx in range(x0, x1):
            blk = pix[(yy-y0)*S:(yy-y0+1)*S, (xx-x0)*S:(xx-x0+1)*S].ravel(); blk = blk[blk >= 0]
            if len(blk) < S*S*0.2: continue
            c = Counter(blk.tolist())    # dark details (eyes, outlines) get extra weight so they survive
            front[yy, xx] = max(c, key=lambda k: c[k] * (2.2 if lum[k] < 25 else 1.0))
    if (sil & (front < 0)).any() and (front >= 0).any():
        _, (iy, ix) = ND.distance_transform_edt(front < 0, return_indices=True)
        front = np.where(sil & (front < 0), front[iy, ix], front)
    return front

def back_map(front, sil, back_code=None):
    """Colours for sides/back. Default: per row, the dominant colour along the silhouette outline (±2 rows),
    so eyes/logos/faces are not copied to the back. --back-color forces one colour."""
    Y, X = sil.shape
    if back_code is not None: return np.where(sil, back_code, -1)
    border = sil & ~ND.binary_erosion(sil, iterations=2)
    back = np.full((Y, X), -1, int); rowc = []
    for y in range(Y):
        vals = front[max(0, y-2):y+3][border[max(0, y-2):y+3]]
        vals = vals[vals >= 0]
        rowc.append(Counter(vals.tolist()).most_common(1)[0][0] if len(vals) else -1)
    for y in range(Y):
        if rowc[y] < 0:
            ks = [k for k in range(Y) if rowc[k] >= 0]; rowc[y] = rowc[min(ks, key=lambda k: abs(k-y))] if ks else front[sil].max()
        back[y] = np.where(sil[y], rowc[y], -1)
    # keep outline-touching side details (e.g. ears, shoes) from the front where they reach the outline
    return np.where(border, front, back)

def colour_voxels(g, pal, args, meshinfo=None, mesh=None):
    Y, X, Z = g.shape; sil = g.any(2)
    er = ND.binary_erosion(np.pad(g, 1))[1:-1, 1:-1, 1:-1]; surf = g & ~er
    col = np.full(g.shape, -1, int); col[g] = SUPPORT
    default = pal.codes.index(args.base_color) if args.base_color else None
    if args.image:
        front = front_map_from_image(args.image, sil, pal, args.max_colors, args.boost)
    elif mesh is not None and _mesh_has_colour(mesh):
        front = None
    else:
        front = np.where(sil, default if default is not None else pal.codes.index('H02'), -1)
    if front is not None:
        back = back_map(front, sil, pal.codes.index(args.back_color) if args.back_color else None)
        gs = ND.gaussian_filter(g.astype(float), 1.0); gy, gx, gz = np.gradient(gs)
        firstz = np.argmax(g, axis=2)
        for y, x, z in zip(*np.where(surf)):
            n = np.array([-gy[y,x,z], -gx[y,x,z], -gz[y,x,z]]); facing = -n[2] / (np.linalg.norm(n) + 1e-9)
            col[y, x, z] = front[y, x] if (z <= firstz[y, x] + 1 or facing > 0.35) else back[y, x]
    else:
        col = _mesh_colours(g, surf, col, mesh, meshinfo, pal, args.max_colors)
    for rep in args.replace or []:
        a, b = rep.split('='); col[col == pal.codes.index(a)] = pal.codes.index(b)
    return col

def _mesh_has_colour(mesh):
    try:
        vc = mesh.visual.to_color().vertex_colors[:, :3]
        return vc.std(0).max() > 3
    except Exception: return False

def _mesh_colours(g, surf, col, mesh, info, pal, maxc):
    from scipy.spatial import cKDTree
    T, pitch, (oy, ox, oz), shp = info
    vc = mesh.visual.to_color().vertex_colors[:, :3].astype(float); tree = cKDTree(mesh.vertices)
    ys, xs, zs = np.where(surf)
    # invert grid transform: g = transpose(occ,(1,0,2))[::-1,:,::-1] then cropped
    my = shp[1] - 1 - (ys + oy); mx = xs + ox; mz = shp[2] - 1 - (zs + oz)
    pts = (T @ np.c_[mx, my, mz, np.ones(len(mx))].T).T[:, :3]
    _, idx = tree.query(pts)
    k = np.argmin(G.ciede2000(G.srgb_to_lab(vc[idx]), pal.lab), 1)
    k2 = G.limit_colors(k[None, :], pal, maxc)[0]
    col[ys, xs, zs] = k2; return col

# ------------------------------------------------------------------ design
class Kit:
    def __init__(s, solid, colsrc):
        s.solid = solid; s.colsrc = colsrc; s.S = solid.shape
        s.pid = np.full(solid.shape, -1, int); s.pieces = []
    def add(s, kind, axis, cells, **kw):
        cells = np.array(cells); s.pieces.append(dict(kind=kind, axis=axis, cells=cells, **kw))
        s.pid[tuple(cells.T)] = len(s.pieces) - 1
    def comps2d(s, mask2, ax, k, kind):
        L, n = ND.label(mask2)
        for i in range(1, n+1): s.add(kind, ax, np.insert(np.argwhere(L == i), ax, k, axis=1), k=k)
    def free_dirs(s, i, present=None):
        pid = s.pid; c = s.pieces[i]['cells']; out = []
        for d in DIRS:
            ax = int(np.nonzero(d)[0][0]); ok = True
            for cc in c:
                sl = tuple((slice(cc[a]+1, None) if d[ax] > 0 else slice(0, cc[a])) if a == ax else cc[a] for a in range(3))
                line = pid[sl]; hit = line[(line >= 0) & (line != i)]
                if len(hit) and (present is None or present[hit].any()): ok = False; break
            if ok: out.append(d)
        return out
    def nf(s, i): return len(s.free_dirs(i))
    def plane_conn(s, cells, ax):
        if len(cells) == 0: return False
        m = np.zeros(s.S, bool); m[tuple(np.asarray(cells).T)] = True
        sl = [slice(None)]*3; sl[ax] = cells[0][ax]; return ND.label(m[tuple(sl)])[1] == 1

def split_rows(solid, face_rows=None):
    Y = solid.shape[0]
    if face_rows: return [(0, face_rows[1] + 1), (face_rows[1] + 1, Y)]
    a = solid.sum((1, 2)).astype(float); best = None
    for y in range(int(Y*0.2), int(Y*0.8)):
        lo = a[max(0, y-1):y+2].min()
        if a[y] > lo: continue
        r = a[y] / max(1, min(a[:y].max(), a[y+1:].max()))
        if r < 0.8 and (best is None or r < best[0]): best = (r, y)
    if best: return [(0, best[1]), (best[1], Y)]
    return [(0, Y)]

def design(col0, face_codes, pal):
    solid = ND.binary_fill_holes(col0 != -1); Y, X, Z = solid.shape
    K = Kit(solid, col0); pid = K.pid
    firstz = np.where(solid.any(2), np.argmax(solid, 2), 10**6)
    face_rows = None
    if face_codes:                                   # optional separate front face panel
        fc = [pal.codes.index(c) for c in face_codes]
        fcol = np.array([[col0[y, x, firstz[y, x]] if firstz[y, x] < Z else -1 for x in range(X)] for y in range(Y)])
        f2 = np.isin(fcol, fc); L, n = ND.label(f2)
        if n:
            f2 = ND.binary_fill_holes(L == 1 + np.argmax(ND.sum(f2, L, range(1, n+1))))
            fy = np.where(f2.any(1))[0]; face_rows = (fy.min(), fy.max())
            zf = int(np.median(firstz[f2]))
            fm = f2 & solid[:, :, zf]; L, n = ND.label(fm)
            if n:
                fm = L == 1 + np.argmax(ND.sum(fm, L, range(1, n+1)))
                K.add('face', 2, [(y, x, zf) for y, x in np.argwhere(fm)], k=zf)
                sn = solid & (np.arange(Z)[None, None, :] < zf) & f2[:, :, None]
                for y in range(Y):
                    if sn[y].any(): K.comps2d(sn[y], 0, y, 'relief')
    full = np.ones((3, 3), bool); shell2d = np.zeros_like(solid)
    for y in range(Y):
        s = solid[y]; shell2d[y] = s & ~ND.binary_erosion(np.pad(s, 1), full)[1:-1, 1:-1]
    capup = solid & ~np.vstack([np.zeros((1, X, Z), bool), solid[:-1]])
    capdn = solid & ~np.vstack([solid[1:], np.zeros((1, X, Z), bool)])
    ring = (shell2d | capup | capdn) & (pid < 0); interior = solid & ~ring & (pid < 0)
    for y0, y1 in split_rows(solid, face_rows):
        seg = interior[y0:y1]
        if not seg.any(): continue
        xs = np.where(seg.any((0, 2)))[0]; zs = np.where(seg.any((0, 1)))[0]
        xc = int(round(np.average(np.arange(X), weights=seg.sum((0, 2))))); zc = int(round(np.average(np.arange(Z), weights=seg.sum((0, 1)))))
        a = np.zeros((Y, X), bool); b = np.zeros((Y, Z), bool)
        a[y0:y1] = interior[y0:y1, :, zc]; b[y0:y1] = interior[y0:y1, xc, :]
        rows = np.where(a[:, xc] & b[:, zc])[0]
        if len(rows):
            mid = (rows.min() + rows.max()) // 2
            for y in rows:
                if y <= mid: b[y, zc] = False        # half-lap: XY plate owns upper half of the crossing line
                else: a[y, xc] = False
        for m, ax, k, kind in ((a, 2, zc, 'core_fb'), (b, 1, xc, 'core_lr')):
            L, n = ND.label(m)
            if n == 0: continue
            main = L == 1 + np.argmax(ND.sum(m, L, range(1, n+1)))
            if main.sum() >= 4: K.add(kind, ax, np.insert(np.argwhere(main), ax, k, axis=1), k=k, seg=(y0, y1))
    ring &= (pid < 0)
    for y in range(Y): K.comps2d(ring[y], 0, y, 'ring')
    return K

def trim_cores(K):
    """Remove core cells that would block a ring sliding down over the core from above."""
    pid, P = K.pid, K.pieces; rings = [i for i, p in enumerate(P) if p['kind'] == 'ring']
    for i, p in enumerate(P):
        if not p['kind'].startswith('core'): continue
        c = p['cells']; y0, y1 = c[:, 0].min(), c[:, 0].max()
        acc = np.zeros(pid.shape[1:], bool); foot = {}
        for y in range(y1, y0 - 1, -1): foot[y] = acc.copy(); acc |= np.isin(pid[y], rings)
        keep = np.array([not foot[y][x, z] for y, x, z in c])
        m = np.zeros(pid.shape, bool); m[tuple(c[keep].T)] = True
        sl = [slice(None)]*3; sl[p['axis']] = p['k']; lab, n = ND.label(m[tuple(sl)])
        pid[tuple(c.T)] = -1
        if n == 0: p['cells'] = np.zeros((0, 3), int); continue
        m2 = np.zeros_like(m); m2[tuple(sl)] = lab == np.argmax(np.bincount(lab.ravel())[1:]) + 1
        p['cells'] = np.argwhere(m2); pid[m2] = i
    _compact(K)

def _compact(K):
    keep = [i for i, p in enumerate(K.pieces) if len(p['cells'])]
    K.pieces = [K.pieces[i] for i in keep]; K.pid[:] = -1
    for j, p in enumerate(K.pieces): K.pid[tuple(p['cells'].T)] = j

def add_pegs(K, warn):
    pid, P, solid = K.pid, K.pieces, K.solid
    for _ in range(2):
        for b in [i for i, p in enumerate(P) if p['kind'] == 'ring' and K.nf(i) > 1]:
            if K.nf(b) <= 1: continue
            y = P[b]['k']; m = np.zeros(K.S[1:], bool); m[tuple(P[b]['cells'][:, 1:].T)] = True
            hole = ND.binary_fill_holes(m) & ~m & (pid[y] < 0)
            cand = []
            for x, z in np.argwhere(hole):
                y0 = y
                while y0 - 1 >= 0 and pid[y0-1, x, z] < 0 and solid[y0-1, x, z]: y0 -= 1
                y1 = y
                while y1 + 1 < K.S[0] and pid[y1+1, x, z] < 0 and solid[y1+1, x, z]: y1 += 1
                if y1 + 1 >= K.S[0] or pid[y1+1, x, z] < 0: continue
                cand.append((x, z, y0, y1))
            if not cand: continue
            cx, cz = np.argwhere(hole).mean(0)
            x, z, y0, y1 = min(cand, key=lambda t: (abs(t[0]-cx) + abs(t[1]-cz), t[3]-t[2]))
            K.add('peg', 1, [(yy, x, z) for yy in range(y0, y1+1)], k=x)

def add_keys(K, rounds=3):
    pid, P, S = K.pid, K.pieces, np.array(K.S)
    def ok_owner(o): return P[o]['kind'] in ('ring', 'relief', 'face') and len(P[o]['cells']) > 1
    for _ in range(rounds):
        B = [i for i, p in enumerate(P) if p['kind'] in ('ring', 'relief') and K.nf(i) > 1]
        if not B: break
        for b in B:
            if K.nf(b) <= 1: continue
            p = P[b]; ax = p['axis']; k = p['k']; best = None
            for cc in p['cells']:
                for ks in ([k-1, k, k+1], [k, k+1, k+2], [k-2, k-1, k], [k-1, k], [k, k+1]):
                    if min(ks) < 0 or max(ks) >= S[ax]: continue
                    cells = []
                    for kk in ks: q = cc.copy(); q[ax] = kk; cells.append(q)
                    cells = np.array(cells); own = [int(pid[tuple(q)]) for q in cells]
                    if min(own) < 0 or len(set(own)) < len(ks) or not all(ok_owner(o) for o in own): continue
                    newc = {o: np.array([c for c in P[o]['cells'] if not any((c == q).all() for q in cells)]) for o in own}
                    if not all(K.plane_conn(newc[o], P[o]['axis']) for o in own): continue
                    for nax in [a for a in range(3) if a != ax]:
                        old = {o: P[o]['cells'] for o in own}
                        for o in own: P[o]['cells'] = newc[o]
                        P.append(dict(kind='key', axis=nax, cells=cells, k=int(cells[0, nax]))); kid = len(P) - 1; pid[tuple(cells.T)] = kid
                        sc = (K.nf(b) > 1)*10 + sum(K.nf(o) > 1 for o in own)*5 + (K.nf(kid) != 1)*3
                        if best is None or sc < best[0]: best = (sc, cells, nax)
                        P.pop()
                        for o in own: P[o]['cells'] = old[o]; pid[tuple(old[o].T)] = o
                        if best[0] == 0: break
                    if best and best[0] == 0: break
                if best and best[0] == 0: break
            if best and best[0] < 10:
                _, cells, nax = best; own = [int(pid[tuple(q)]) for q in cells]
                for o in own: P[o]['cells'] = np.array([c for c in P[o]['cells'] if not any((c == q).all() for q in cells)])
                K.add('key', nax, cells, k=int(cells[0, nax]))

def pin_caps(K):
    pid, P = K.pid, K.pieces
    for b, p in enumerate(P):
        if p['kind'] != 'ring' or K.nf(b) <= 1: continue
        for (yy, x, z) in p['cells']:
            if yy + 1 >= K.S[0]: continue
            j = pid[yy+1, x, z]
            if j >= 0 and P[j]['kind'] == 'peg':
                rest = np.array([c for c in p['cells'] if not (c[1] == x and c[2] == z)])
                if len(rest) and K.plane_conn(rest, 0):
                    p['cells'] = rest; P[j]['cells'] = np.vstack([[(yy, x, z)], P[j]['cells']]); pid[yy, x, z] = j; break

def absorb_fragments(K, maxsize=2):
    """Merge tiny loose ring fragments into an in-plane-adjacent... (none exist in-plane), so try vertical key."""
    # tiny fragments are handled by add_keys; anything still loose is reported.
    return

def validate(K):
    occ = Counter()
    for p in K.pieces:
        for c in map(tuple, p['cells']): occ[c] += 1
    errs = []
    if any(v > 1 for v in occ.values()): errs.append('overlapping beads')
    for i, p in enumerate(K.pieces):
        if not (p['cells'][:, p['axis']] == p['cells'][0, p['axis']]).all(): errs.append(f'piece {i} not flat')
        elif not K.plane_conn(p['cells'], p['axis']): errs.append(f'piece {i} not connected')
    N = len(K.pieces); present = np.ones(N, bool); order = []; rdir = {}
    GO = {'peg': 0, 'key': 0, 'relief': 1, 'face': 2, 'ring': 3, 'core_fb': 4, 'core_lr': 4}
    while present.any():
        for _, i in sorted(((GO[K.pieces[i]['kind']], K.pieces[i]['cells'][:, 0].min()), i) for i in np.where(present)[0]):
            f = K.free_dirs(i, present)
            if f:
                rdir[i] = sorted(f, key=lambda d: (0 if d[K.pieces[i]['axis']] != 0 else 1, 0 if d == (-1, 0, 0) else 1))[0]
                present[i] = False; order.append(i); break
        else:
            errs.append('no assembly order for pieces ' + str(list(np.where(present)[0]))); break
    return errs, order[::-1], rdir

# ------------------------------------------------------------------ outputs
def outputs(K, col, pal, out, title, support_label, order, rdir):
    from PIL import Image, ImageDraw, ImageFont
    RV.PAL_RGB = pal.rgb
    P = K.pieces; pid = K.pid; N = len(P); Y, X, Z = K.S
    CJK = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'; CJKB = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
    F = lambda s, b=False: ImageFont.truetype(CJKB if b else CJK, s)
    rgb_of = RV.rgb_of; tc = lambda rgb: (0, 0, 0) if (0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]) > 140 else (255, 255, 255)
    BG = (230, 238, 246); num = {p: n+1 for n, p in enumerate(order)}
    FREE = {i: K.free_dirs(i) for i in range(N)}
    INS = {(1,0,0):'从下往上装', (-1,0,0):'从上往下装', (0,1,0):'从右往左插', (0,-1,0):'从左往右插', (0,0,1):'从后往前插', (0,0,-1):'从前往后插'}
    KIND = {'ring':'横向层片', 'face':'正面面板', 'relief':'正面浮雕小片', 'core_fb':'前后向骨架', 'core_lr':'左右向骨架', 'peg':'竖插销', 'key':'小锁扣'}
    DISP = {0: (2, 1), 1: (0, 2), 2: (0, 1)}
    mates = {i: set() for i in range(N)}; slot = {i: [] for i in range(N)}
    for i, p in enumerate(P):
        a = p['axis']; r, c = DISP[a]
        for cc in p['cells']:
            for dr, dc in ((1,0), (-1,0), (0,1), (0,-1)):
                q = list(cc); q[r] += dr; q[c] += dc
                if not all(0 <= q[t] < K.S[t] for t in range(3)): continue
                j = pid[tuple(q)]
                if j >= 0 and j != i and P[j]['axis'] != a: mates[i].add(j); mates[j].add(i); slot[i].append((cc, dr, dc))
    ml = lambda i: ('接' + ','.join(str(num[j]) for j in sorted(mates[i], key=lambda j: num[j]))) if mates[i] else ''
    def piece_img(i, cs):
        p = P[i]; r, c = DISP[p['axis']]; cells = p['cells']
        r0, c0 = cells[:, r].min(), cells[:, c].min(); h = cells[:, r].max() - r0 + 1; w = cells[:, c].max() - c0 + 1
        lab = ml(i); tw = max(w*cs, int(len(lab)*cs*0.33) + 10, 30)
        img = Image.new('RGB', (tw + 8, h*cs + int(cs*1.6) + 8), BG); d = ImageDraw.Draw(img)
        for cc in cells:
            x0 = 4 + (cc[c]-c0)*cs; y0 = 4 + (cc[r]-r0)*cs; v = col[tuple(cc)]; rgb = rgb_of(v)
            d.rectangle([x0, y0, x0+cs, y0+cs], fill=rgb, outline=(60, 60, 60))
            if v != SUPPORT and cs >= 14: d.text((x0+cs/2, y0+cs/2), pal.codes[v], font=F(max(7, int(cs*0.36)), True), fill=tc(rgb), anchor='mm')
        for cc, dr, dc in slot[i]:
            x0 = 4 + (cc[c]-c0)*cs; y0 = 4 + (cc[r]-r0)*cs
            seg = {(1,0): [x0, y0+cs, x0+cs, y0+cs], (-1,0): [x0, y0, x0+cs, y0], (0,1): [x0+cs, y0, x0+cs, y0+cs], (0,-1): [x0, y0, x0, y0+cs]}[(dr, dc)]
            d.line(seg, fill=(235, 20, 20), width=max(2, cs//7))
        d.text((4, 4 + h*cs + 2), str(num[i]), font=F(int(cs*0.8), True), fill=(20, 20, 20))
        d.text((4 + int(cs*1.4), 4 + h*cs + int(cs*0.25)), lab, font=F(max(9, int(cs*0.42))), fill=(200, 30, 30))
        return img
    GROUPS = [('正面面板 / 浮雕', ['face', 'relief']), ('横向层片（从上到下）', ['ring']), (f'骨架 / 插销 / 锁扣（{support_label}）', ['core_fb', 'core_lr', 'peg', 'key'])]
    def grouped():
        res = []
        for gt, ks in GROUPS:
            ids = [i for i in order if P[i]['kind'] in ks]
            if ks == ['ring']: ids.sort(key=lambda i: (P[i]['k'], num[i]))
            if ids: res.append((gt, ids))
        return res
    def flow(items, W, gap=14):
        rows = []; cur = []; x = 0; h = 0
        for im in items:
            if cur and x + im.width > W: rows.append((cur, h)); cur = []; x = 0; h = 0
            cur.append((x, im)); x += im.width + gap; h = max(h, im.height)
        if cur: rows.append((cur, h))
        return rows
    cnt = Counter(int(v) for v in col[col != -1].ravel()); total = sum(cnt.values())
    rows_c = sorted([(k, c) for k, c in cnt.items() if k != SUPPORT], key=lambda t: -t[1])
    with open(f'{out}/shopping-list.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f); w.writerow(['色号', 'hex', '数量'])
        for k, c in rows_c: w.writerow([pal.codes[k], pal.hexes[k], c])
        if cnt.get(SUPPORT): w.writerow([f'{support_label}(内部支撑,图中绿色)', '', cnt[SUPPORT]])
        w.writerow(['总计', '', total])
    with open(f'{out}/pieces.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f); w.writerow(['件号', '类型', '颗数', '安装方向', '插接', '装好后可动方向'])
        for i in order: w.writerow([num[i], KIND[P[i]['kind']], len(P[i]['cells']), INS[tuple(-np.array(rdir[i]))], ml(i), ''.join(NAMES[d] for d in FREE[i]) or '锁死'])
    pv = [RV.render(col, 0, 12, 14), RV.render(col, -90, 12, 14), RV.render(col, 180, 12, 14)]
    for im, n in zip(pv, ('front', 'side', 'back')): im.save(f'{out}/preview_{n}.png')
    # exploded view
    PAD = 60; big = np.full((Y*4 + 2*PAD, X*3 + 2*PAD, Z*3 + 2*PAD), -1, int)
    for i in order:
        p = P[i]; k = p['kind']; cc = p['cells'].copy(); off = np.zeros(3, int)
        off[0] = int(cc[:, 0].min()) + (p['k']*2 if k == 'ring' else 0)
        off[2] += {'face': -12, 'relief': -20, 'core_fb': Z + 6, 'key': -26}.get(k, 0)
        off[1] += {'core_lr': X + 4, 'peg': -(X//2 + 6)}.get(k, 0)
        cc = np.clip(cc + off + PAD, 0, np.array(big.shape) - 1); big[tuple(cc.T)] = col[tuple(p['cells'].T)]
    nz = [np.where((big != -1).any(axis=t))[0] for t in ((1, 2), (0, 2), (0, 1))]
    big = big[nz[0].min():nz[0].max()+1, nz[1].min():nz[1].max()+1, nz[2].min():nz[2].max()+1]
    ex = RV.render(big, -30, 18, 8); ex.save(f'{out}/exploded.png')
    # poster
    W = 1800; parts = []
    head = Image.new('RGB', (W, 150), 'white'); d = ImageDraw.Draw(head)
    d.rounded_rectangle([20, 15, W-20, 135], radius=30, fill=(255, 225, 225), outline=(220, 40, 60), width=5)
    d.text((W//2, 58), f'{title} · 3D插接拼豆', font=F(60, True), fill=(210, 20, 40), anchor='mm')
    d.text((W//2, 112), f'{N}件 · {X}×{Y}×{Z}豆(宽×高×深) · 共{total}颗 · {len(rows_c)}色+{support_label} · 免胶插接 · {pal.name.upper()}', font=F(26), fill=(120, 40, 50), anchor='mm')
    parts.append(head)
    pvb = Image.new('RGB', (W, 520), 'white'); x = 40
    for im, lab in zip(pv, ('正面', '侧面', '背面')):
        t = im.copy(); t.thumbnail((540, 470)); pvb.paste(t, (x + (540 - t.width)//2, 40), t)
        ImageDraw.Draw(pvb).text((x + 250, 10), lab, font=F(24, True), fill=(90, 90, 90)); x += 580
    parts.append(pvb)
    LEG = 330; blocks = []
    for gt, ids in grouped():
        rows = flow([piece_img(i, 13) for i in ids], W - 60 - LEG)
        b = Image.new('RGB', (W, 40 + sum(h + 14 for _, h in rows)), BG); dd = ImageDraw.Draw(b); dd.text((30, 6), gt, font=F(24, True), fill=(50, 70, 100)); yy = 40
        for line, h in rows:
            for xx, im in line: b.paste(im, (30 + xx, yy))
            yy += h + 14
        blocks.append(b)
    tb = Image.new('RGB', (W, 140 + 40*(len(rows_c) + 2)), BG); d = ImageDraw.Draw(tb); tx = W - LEG - 10
    d.text((tx, 20), '色号', font=F(26, True), fill=(30, 30, 30)); d.text((tx + 190, 20), '数量', font=F(26, True), fill=(30, 30, 30)); ty = 64
    for k, c in rows_c + ([(SUPPORT, cnt[SUPPORT])] if cnt.get(SUPPORT) else []):
        d.rectangle([tx, ty, tx+32, ty+30], fill=rgb_of(k), outline=(90, 90, 90))
        d.text((tx+44, ty), support_label if k == SUPPORT else pal.codes[k], font=F(24, True), fill=(30, 30, 30)); d.text((tx+190, ty), str(c), font=F(24, True), fill=(30, 30, 30)); ty += 40
    d.text((tx, ty+4), '总计', font=F(26, True), fill=(30, 30, 30)); d.text((tx+190, ty+4), str(total), font=F(26, True), fill=(30, 30, 30))
    d.text((30, 20), '零件图（红线 = 卡槽/插接边；数字 = 件号 = 组装顺序；“接n” = 与第n件插接）', font=F(22, True), fill=(50, 70, 100))
    d.text((30, 56), f'绿色 = {support_label}（内部支撑，外面看不见）', font=F(22), fill=(50, 70, 100))
    first = blocks[0] if blocks else Image.new('RGB', (W, 10), BG)
    top = Image.new('RGB', (W, max(tb.height, first.height + 90)), BG); top.paste(tb, (0, 0)); top.paste(first.crop((0, 0, W - LEG - 20, first.height)), (0, 90))
    parts.append(top); parts += blocks[1:]
    exb = Image.new('RGB', (W, 1300), BG); t = ex.copy(); t.thumbnail((W - 80, 1220)); exb.paste(t, ((W - t.width)//2, 60), t)
    ImageDraw.Draw(exb).text((30, 14), '爆炸图：层片沿竖轴从上到下排列；正面件向前、骨架/插销向侧面拉出', font=F(26, True), fill=(50, 70, 100))
    parts.append(exb)
    sheet = Image.new('RGB', (W, sum(p.height for p in parts) + 20), BG); y = 0
    for p in parts: sheet.paste(p, (0, y)); y += p.height
    sheet.save(f'{out}/sheet.png')
    # phone-size parts pages
    for f in os.listdir(out):
        if f.startswith('parts_') and f.endswith('.png'): os.remove(f'{out}/{f}')
    allb = []
    for gt, ids in grouped():
        for line, h in flow([piece_img(i, 24) for i in ids], 1540): allb.append((gt, line, h))
    pages = []; cur = []; hc = 0
    for b in allb:
        if cur and hc + b[2] + 20 > 3400: pages.append(cur); cur = []; hc = 0
        cur.append(b); hc += b[2] + 60
    if cur: pages.append(cur)
    files = []
    for pi, pg in enumerate(pages, 1):
        H = sum(h + 60 for _, _, h in pg) + 90; im = Image.new('RGB', (1600, H), BG); d = ImageDraw.Draw(im)
        d.text((30, 20), f'{title} 零件图 {pi}/{len(pages)}（红线=卡槽边 · 绿色={support_label}）', font=F(34, True), fill=(200, 30, 40)); y = 90; last = None
        for gt, line, h in pg:
            if gt != last: d.text((30, y), gt, font=F(26, True), fill=(50, 70, 100)); y += 40; last = gt
            for xx, t in line: im.paste(t, (30 + xx, y))
            y += h + 20
        fn = f'{out}/parts_{pi}.png'; im.crop((0, 0, 1600, y + 20)).save(fn); files.append(fn)
    # steps
    chunks = [c for c in np.array_split(np.array(order), min(12, N)) if len(c)]; tiles = []; done = []
    for gi, g in enumerate(chunks):
        done += list(g); im = RV.render(np.where(np.isin(pid, done), col, -1), -35, 22, 8)
        tile = Image.new('RGB', (420, 470), 'white'); d = ImageDraw.Draw(tile)
        t = im.copy(); t.thumbnail((400, 380)); tile.paste(t, ((420 - t.width)//2, 80), t)
        d.text((12, 6), f'步骤{gi+1}：#{num[g[0]]}–#{num[g[-1]]}', font=F(26, True), fill=(30, 30, 30))
        d.text((12, 44), ' '.join(f'{KIND[k]}×{v}' for k, v in Counter(P[i]['kind'] for i in g).items()), font=F(16), fill=(90, 90, 90))
        tiles.append(tile)
    rowsn = (len(tiles) + 3)//4
    steps = Image.new('RGB', (1720, 140 + 470*rowsn + 20), 'white'); d = ImageDraw.Draw(steps)
    d.text((20, 15), '组装步骤（按件号顺序，免胶）', font=F(40, True), fill=(200, 30, 40))
    d.text((20, 75), '1 立起骨架 → 2 层片从下往上逐层套入 → 3 装正面件/锁扣 → 4 插入竖插销并盖上顶片', font=F(20), fill=(80, 80, 80))
    for t, tile in enumerate(tiles): steps.paste(tile, (20 + (t % 4)*425, 130 + (t//4)*470))
    steps.save(f'{out}/assembly-steps.png')
    return dict(total=total, colors={pal.codes[k]: c for k, c in rows_c}, support=cnt.get(SUPPORT, 0), parts_files=files,
                free=FREE, num=num)

# ------------------------------------------------------------------ main
def build(col, pal, args):
    K = design(col, args.face_colors, pal)
    trim_cores(K)
    warn = []
    add_pegs(K, warn); add_keys(K); pin_caps(K)
    return K

def run(args):
    pal = G.load_palette(args.palette)
    out = os.path.abspath(args.out); os.makedirs(out, exist_ok=True)
    if getattr(args, 'style', 'slice') == 'box':
        import box
        model = None
        if args.cmd == 'from-model': model = args.model
        elif args.shape: model = args.shape
        elif not args.no_model:
            try: model = hunyuan_shape(args.image, f'{out}/shape.glb', args.space, args.seed)
            except Exception as e: log('image-to-3D failed, continuing with --depth/default depth:', repr(e)[:200])
        return box.run_box(args, pal, out, model)
    if args.cmd == 'from-image':
        model = args.shape or hunyuan_shape(args.image, f'{out}/shape.glb', args.space, args.seed)
    else:
        model = args.model
    mesh = load_mesh(model, args.up, args.yaw)
    def make(H):
        g, info = voxelize(mesh, H); col = colour_voxels(g, pal, args, info, mesh)
        return g, col
    if args.height: H = args.height; g, col = make(H)
    else:                                   # search height so the designed kit hits --target-beads
        lo, hi, best = 10, 90, None
        while lo <= hi:
            mid = (lo + hi)//2; g, col = make(mid); K = design(col, args.face_colors, pal)
            n = int((K.pid >= 0).sum()); log(f'height {mid}: ~{n} beads')
            if best is None or abs(n - args.target_beads) < abs(best[1] - args.target_beads): best = (mid, n)
            if n < args.target_beads: lo = mid + 1
            else: hi = mid - 1
        H = best[0]; g, col = make(H)
    log('grid (H,W,D)=', g.shape)
    K = build(col, pal, args)
    errs, order, rdir = validate(K)
    if any(e.startswith('no assembly') for e in errs): log('ERROR', errs); 
    # final colours: visible cells keep nearest colour, hidden -> support (or a real code)
    solid = K.solid; vis = solid & ~ND.binary_erosion(np.pad(solid, 1))[1:-1, 1:-1, 1:-1]
    has = col >= 0; ix = ND.distance_transform_edt(~has, return_distances=False, return_indices=True); colv = col[tuple(ix)]
    fin = np.full(solid.shape, -1, int); occ = K.pid >= 0
    fin[occ & vis] = colv[occ & vis]; fin[occ & ~vis] = SUPPORT
    support_label = '任意色'
    if args.support_color not in ('任意色', 'any'):
        fin[fin == SUPPORT] = pal.codes.index(args.support_color); support_label = args.support_color
    unc = int((vis & ~occ).sum())
    if len(order) < len(K.pieces):     # outputs need a full order
        order = order + [i for i in range(len(K.pieces)) if i not in order]
        for i in range(len(K.pieces)): rdir.setdefault(i, (-1, 0, 0))
    res = outputs(K, fin, pal, out, args.title, support_label, order, rdir)
    warns = list(errs)
    num = res['num']
    loose = [(num[i], K.pieces[i]['kind'], len(K.pieces[i]['cells']), ''.join(NAMES[d] for d in f)) for i, f in res['free'].items() if len(f) > 1]
    for n_, k, s, f in loose: warns.append(f'piece #{n_} ({k}, {s} beads) is not locked: can move {f}')
    for i, p in enumerate(K.pieces):
        L = len(p['cells'])
        if p['kind'] == 'peg' and L > 10: warns.append(f'piece #{num[i]} peg is a 1-bead-wide stick of {L} beads: fragile, iron well / use 2 layers')
        if L == 1: warns.append(f'piece #{num[i]} ({p["kind"]}) is a single bead')
    if unc: warns.append(f'{unc} visible cells not covered by any piece')
    summary = dict(out=out, model=model, grid_HWD=list(K.S), height_beads=K.S[0], pieces=len(K.pieces),
                   kinds=dict(Counter(p['kind'] for p in K.pieces)), total_beads=res['total'], colors=res['colors'],
                   support_beads=res['support'], support_label=support_label,
                   locked=sum(1 for f in res['free'].values() if not f), one_way=sum(1 for f in res['free'].values() if len(f) == 1),
                   loose=len(loose), valid_assembly_order=not any(e.startswith('no assembly') for e in errs),
                   files=dict(sheet=f'{out}/sheet.png', parts=res['parts_files'], steps=f'{out}/assembly-steps.png',
                              exploded=f'{out}/exploded.png', previews=[f'{out}/preview_{n}.png' for n in ('front', 'side', 'back')],
                              shopping=f'{out}/shopping-list.csv', pieces=f'{out}/pieces.csv'),
                   warnings=warns)
    json.dump(summary, open(f'{out}/summary.json', 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))

def main():
    ap = argparse.ArgumentParser(description='Image / 3D model -> interlocking 3D fuse-bead (拼豆) kit')
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('from-image', 'from-model'):
        p = sub.add_parser(name)
        if name == 'from-image':
            p.add_argument('image', help='character image (transparent PNG best); also used for colours')
            p.add_argument('--shape', help='skip Hunyuan and use this mesh (e.g. a cached shape.glb)')
            p.add_argument('--space', default='tencent/Hunyuan3D-2'); p.add_argument('--seed', type=int, default=1234)
        else:
            p.add_argument('model', help='.glb/.gltf/.obj/.stl/.ply mesh')
            p.add_argument('--image', help='front-view image to paint colours from (recommended for untextured meshes)')
        p.add_argument('--out', required=True)
        p.add_argument('--palette', default='mard', help='bead-pattern palette: mard, coco, manman, mixiaowo, panpan')
        p.add_argument('--bead-pattern-scripts', help='path to bead-pattern scripts/ (default: ../bead-pattern/scripts next to this file, or env BEAD_PATTERN_SCRIPTS)')
        p.add_argument('--target-beads', type=int, default=3000); p.add_argument('--height', type=int, help='beads tall (overrides --target-beads)')
        p.add_argument('--max-colors', type=int, default=8)
        p.add_argument('--support-color', default='任意色', help="hidden support colour: 任意色 (any) or a palette code e.g. H02")
        p.add_argument('--replace', action='append', help='colour substitution, e.g. P22=E20 (repeatable)')
        p.add_argument('--face-colors', type=lambda s: s.split(','), help='optional: codes forming a front face (e.g. E14,H16,H02) -> separate front panel')
        p.add_argument('--back-color', help='force one colour for the back/sides (default: auto from outline)')
        p.add_argument('--boost', type=float, default=1.0, help='brighten image colours before matching (e.g. 1.1 for dark/shaded art)')
        p.add_argument('--base-color', help='colour for uncoloured meshes without --image (default H02)')
        p.add_argument('--up', choices=['y', 'z'], default='y', help='model up axis (GLB=y, many STL=z)')
        p.add_argument('--yaw', type=float, default=0.0, help='rotate model about up axis so its front faces the viewer')
        p.add_argument('--title', default='3D拼豆')
        p.add_argument('--style', choices=['slice', 'box'], default='slice', help='slice = auto-sliced voxel kit; box = MARD-style panel box (front/back panels + side strips + raised accessories, tab/slot)')
        p.add_argument('--depth', type=int, help='box: gap between front and back panel in beads (default: from 3D model, else ~12%% of width, 4-6)')
        p.add_argument('--accessory-colors', help='box: codes whose regions become raised accessory pieces, e.g. B29,G10')
        p.add_argument('--accessory-box', action='append', help='box: y0:y1,x0:x1 region (bead coords) -> raised accessory piece (repeatable)')
        p.add_argument('--min-accessory', type=int, default=8, help='box: smallest accessory in beads')
        p.add_argument('--no-model', action='store_true', help='box: skip image-to-3D entirely')
    a = ap.parse_args()
    if a.cmd == 'from-image': a.model = None
    else: a.shape = None
    run(a)

if __name__ == '__main__':
    main()
