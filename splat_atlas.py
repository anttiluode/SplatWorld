#!/usr/bin/env python3
# splat_atlas.py — automatic latent-space surveyor for splat_decoder.onnx
#
# Burns a few GB of CIFAR-size (32x32) thumbnails to disk, systematically,
# and keeps the z of EVERY thumbnail (float16) — so any dot in the atlas can
# be re-rendered at full model resolution, or used as a surf starting point.
#
#   python splat_atlas.py --dump --gb 2          # survey -> ./atlas/
#   python splat_atlas.py --browse               # click a tile -> full res + z
#   python splat_atlas.py --analyze              # departure curves, csv + png
#   python splat_atlas.py --selftest
#
# Survey strategies (fixed seed, resumable by shard):
#   prior   z ~ N(0, r) at radii r in RADII — the on-manifold -> splat-soup
#           transition you found by hand, sampled densely at every shell
#   ray     straight-line walks outward along fixed directions, r = 0..MAX —
#           "going the same direction, the splats get stronger", measured
#   walk    long random walks of small steps (your surf+TAB accumulation,
#           automated): step ~ N(0, 0.15), thousands of steps, save each
#   slerp   great-circle interpolations between random prior points —
#           does the manifold stay face-like BETWEEN faces?
#
# Storage: atlas/shard_XXXX_img.npy  uint8 (N,32,32,3)
#          atlas/shard_XXXX_z.npy    float16 (N,128)
#          atlas/shard_XXXX_meta.csv strategy,param,low,mid,fine per row
#          atlas/sheet_XXXX.png      32x32 contact sheet per shard (eyeball)
#
# Analysis written by --analyze: band energies vs radius per strategy — the
# fine-band curve is the quantitative version of "splats appear and grow".

import argparse, csv, glob, math, os, sys, time
import numpy as np

LATENT   = 128
RSEED    = 7
THUMB    = 32
SHARD_N  = 4096                       # images per shard (12.6 MB img + 1 MB z)
RADII    = [0.3, 0.6, 1.0, 1.5, 2.2, 3.2, 4.6, 6.8, 10.0, 15.0]
RAY_MAX  = 20.0
RAY_STEPS = 64
WALK_STEP = 0.15
BATCH    = 64
OUTDIR   = "atlas"

# ------------------------------------------------------------- decoders
class OnnxDecoder:
    def __init__(self, path="splat_decoder.onnx"):
        self.backend = None
        # onnxruntime first: OpenCV 5's dnn importer can't parse this graph's
        # ConstantOfShape (dynamic batch) node, which BATCH=64 dumps rely on.
        try:
            import onnxruntime as ort
            self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self.iname = self.sess.get_inputs()[0].name
            self.oname = self.sess.get_outputs()[0].name
            self.backend = "onnxruntime"
        except Exception as e:
            import cv2 as cv
            self.net = None
            for eng in (getattr(cv.dnn, "ENGINE_CLASSIC", None),
                        getattr(cv.dnn, "ENGINE_AUTO", None)):
                if eng is None:
                    continue
                try:
                    self.net = cv.dnn.readNetFromONNX(path, engine=eng); break
                except TypeError:
                    break
                except Exception:
                    continue
            if self.net is None:
                self.net = cv.dnn.readNetFromONNX(path)
            self.backend = "opencv-dnn"
            print(f"atlas decoder: opencv-dnn (onnxruntime unavailable: {e}); "
                  f"on OpenCV 5 run 'python -m pip install onnxruntime'")
    def __call__(self, zs):
        zs = np.ascontiguousarray(zs.astype(np.float32))
        if self.backend == "onnxruntime":
            return self.sess.run([self.oname], {self.iname: zs})[0]
        self.net.setInput(zs, "z_latent")
        return self.net.forward("rendered_image").copy()   # (N,3,H,W) in [0,1]

class MockDecoder:
    def __init__(self, h=48):
        g = np.random.default_rng(99).standard_normal((3 * h * h, LATENT))
        self.W = (g / math.sqrt(LATENT)).astype(np.float32)
        self.h = h
    def __call__(self, zs):
        y = np.tanh(zs.astype(np.float32) @ self.W.T) * 0.5 + 0.5
        return y.reshape(len(zs), 3, self.h, self.h)

# ------------------------------------------------------------- z generators
def gen_survey(rng):
    """Yield (strategy, param, z) forever, interleaving strategies."""
    dirs = None
    while True:
        # prior shells
        for r in RADII:
            z = rng.standard_normal(LATENT) * r
            yield ("prior", r, z)
        # one ray step-set: a fresh direction, full outward walk
        d = rng.standard_normal(LATENT); d /= np.linalg.norm(d)
        for i in range(RAY_STEPS):
            r = RAY_MAX * (i + 1) / RAY_STEPS
            yield ("ray", r, d * r)
        # a random-walk burst: 64 consecutive steps of one walk
        if dirs is None or rng.random() < 0.02:
            walk_z = rng.standard_normal(LATENT) * 0.3
            dirs = True
        for _ in range(64):
            walk_z = walk_z + rng.standard_normal(LATENT) * WALK_STEP
            yield ("walk", float(np.linalg.norm(walk_z)), walk_z.copy())
        # slerp between two prior points
        a = rng.standard_normal(LATENT) * 0.6
        b = rng.standard_normal(LATENT) * 0.6
        na, nb = a / np.linalg.norm(a), b / np.linalg.norm(b)
        om = math.acos(float(np.clip(na @ nb, -1, 1)))
        for t in np.linspace(0, 1, 16):
            if om < 1e-4:
                z = (1 - t) * a + t * b
            else:
                z = (math.sin((1 - t) * om) * a + math.sin(t * om) * b) / math.sin(om)
            yield ("slerp", float(t), z)

# ------------------------------------------------------------- band stats
def band3(gray):
    """(low, mid, fine) from float gray [0,1], numpy-only box blurs."""
    def box(im, r):
        k = 2 * r + 1
        c = np.cumsum(np.cumsum(np.pad(im, ((1, 0), (1, 0))), 0), 1)
        o = (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)
        return np.pad(o, r, mode='edge')
    b1, b2 = box(gray, 6), box(gray, 2)
    return (float(b1.mean()), float((b2 - b1).std()),
            float((gray - box(gray, 1)).std()))

def to_thumbs(out):
    """(N,3,H,W) float -> (N,32,32,3) uint8 via area-ish mean pooling."""
    n, c, h, w = out.shape
    f = max(1, h // THUMB)
    hh = (h // f) * f
    t = out[:, :, :hh, :hh].reshape(n, c, h // f, f, h // f, f).mean((3, 5))
    if t.shape[2] != THUMB:                       # final resize if not exact
        import cv2 as cv
        t = np.stack([cv.resize(np.transpose(x, (1, 2, 0)), (THUMB, THUMB))
                      for x in t])
    else:
        t = np.transpose(t, (0, 2, 3, 1))
    return (t * 255).clip(0, 255).astype(np.uint8)

# ------------------------------------------------------------- dump
def sheet(imgs, cols=64):
    rows = int(math.ceil(len(imgs) / cols))
    g = np.zeros((rows * THUMB, cols * THUMB, 3), np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        g[r*THUMB:(r+1)*THUMB, c*THUMB:(c+1)*THUMB] = im[..., ::-1]  # BGR png
    return g

def dump(gb, dec):
    os.makedirs(OUTDIR, exist_ok=True)
    per_img = THUMB * THUMB * 3 + LATENT * 2      # uint8 thumb + f16 z
    n_total = int(gb * 1e9 / per_img)
    done = sorted(glob.glob(f"{OUTDIR}/shard_*_img.npy"))
    start_shard = len(done)
    n_have = start_shard * SHARD_N
    print(f"target {n_total} images ({gb} GB); have {n_have}; "
          f"{SHARD_N} per shard")
    rng = np.random.default_rng(RSEED + start_shard)   # resumable-ish
    gen = gen_survey(rng)
    t0 = time.time()
    shard = start_shard
    while n_have < n_total:
        imgs = np.zeros((SHARD_N, THUMB, THUMB, 3), np.uint8)
        zs   = np.zeros((SHARD_N, LATENT), np.float16)
        meta = []
        for b0 in range(0, SHARD_N, BATCH):
            batch = [next(gen) for _ in range(min(BATCH, SHARD_N - b0))]
            zb = np.stack([z for _, _, z in batch]).astype(np.float32)
            out = dec(zb)
            th = to_thumbs(out)
            for j, (strat, par, z) in enumerate(batch):
                i = b0 + j
                imgs[i] = th[j]; zs[i] = z.astype(np.float16)
                g = th[j].mean(2).astype(np.float32) / 255.0
                lo, mi, fi = band3(g)
                meta.append([strat, f"{par:.4f}", f"{lo:.4f}",
                             f"{mi:.4f}", f"{fi:.4f}"])
        tag = f"{shard:04d}"
        np.save(f"{OUTDIR}/shard_{tag}_img.npy", imgs)
        np.save(f"{OUTDIR}/shard_{tag}_z.npy", zs)
        with open(f"{OUTDIR}/shard_{tag}_meta.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["strategy","param","low","mid","fine"])
            w.writerows(meta)
        import cv2 as cv
        cv.imwrite(f"{OUTDIR}/sheet_{tag}.png", sheet(imgs))
        n_have += SHARD_N; shard += 1
        rate = n_have / max(1e-9, time.time() - t0)
        print(f"shard {tag}: {n_have}/{n_total}  ({rate:.0f} img/s, "
              f"eta {(n_total-n_have)/max(rate,1e-9)/60:.1f} min)")
    print("dump complete.")

# ------------------------------------------------------------- browse
def browse(dec):
    import cv2 as cv
    sheets = sorted(glob.glob(f"{OUTDIR}/sheet_*.png"))
    if not sheets:
        print("no atlas found — run --dump first"); return
    idx = 0
    win = "ATLAS  (click tile = full res + z | n/p sheet | q quit)"
    cv.namedWindow(win, cv.WINDOW_NORMAL)
    state = {"click": None}
    cv.setMouseCallback(win, lambda ev, x, y, fl, _:
                        state.update(click=(x, y)) if ev == cv.EVENT_LBUTTONDOWN else None)
    while True:
        tag = sheets[idx].split("sheet_")[1].split(".")[0]
        sh = cv.imread(sheets[idx])
        cv.imshow(win, sh)
        k = cv.waitKey(30) & 0xFF
        if k == ord('q'): break
        elif k == ord('n'): idx = (idx + 1) % len(sheets)
        elif k == ord('p'): idx = (idx - 1) % len(sheets)
        if state["click"]:
            x, y = state["click"]; state["click"] = None
            cols = sh.shape[1] // THUMB
            ti = (y // THUMB) * cols + (x // THUMB)
            zs = np.load(f"{OUTDIR}/shard_{tag}_z.npy")
            if ti < len(zs):
                z = zs[ti].astype(np.float32)
                out = dec(z[None])[0]
                im = (np.transpose(out, (1, 2, 0)) * 255).clip(0, 255).astype(np.uint8)
                im = cv.cvtColor(im, cv.COLOR_RGB2BGR)
                cv.imshow("full res", cv.resize(im, (512, 512),
                          interpolation=cv.INTER_CUBIC))
                np.save("picked_z.npy", z)
                print(f"tile {tag}/{ti}  |z|={np.linalg.norm(z):.2f}  "
                      f"-> picked_z.npy (surf/probe can start here)")
    cv.destroyAllWindows()

# ------------------------------------------------------------- analyze
def analyze():
    rows = []
    for f in sorted(glob.glob(f"{OUTDIR}/shard_*_meta.csv")):
        with open(f) as fh:
            rows += list(csv.DictReader(fh))
    if not rows:
        print("no metadata — run --dump first"); return
    print(f"{len(rows)} images.")
    # bin ray+prior by radius, report mean fine-band energy
    bins = {}
    for r in rows:
        if r["strategy"] not in ("ray", "prior"): continue
        key = (r["strategy"], round(float(r["param"]) * 2) / 2)
        bins.setdefault(key, []).append(float(r["fine"]))
    print(f"{'strategy':8} {'radius':>7} {'n':>6} {'fine-band':>10}")
    curve = {}
    for (s, rad), v in sorted(bins.items()):
        print(f"{s:8} {rad:7.1f} {len(v):6d} {np.mean(v):10.4f}")
        curve.setdefault(s, []).append((rad, np.mean(v)))
    # draw departure curve png with cv2 (no matplotlib dependency)
    import cv2 as cv
    W, H = 640, 360
    img = np.full((H, W, 3), 24, np.uint8)
    allpts = [p for c in curve.values() for p in c]
    mx_r = max(p[0] for p in allpts); mx_e = max(p[1] for p in allpts) + 1e-9
    colors = {"prior": (80, 200, 80), "ray": (80, 160, 255)}
    for s, pts in curve.items():
        pts = sorted(pts)
        pix = [(int(30 + r / mx_r * (W - 60)),
                int(H - 30 - e / mx_e * (H - 60))) for r, e in pts]
        for a, b in zip(pix, pix[1:]):
            cv.line(img, a, b, colors.get(s, (200, 200, 200)), 2, cv.LINE_AA)
        cv.putText(img, s, pix[-1], cv.FONT_HERSHEY_PLAIN, 1,
                   colors.get(s, (200, 200, 200)), 1)
    cv.putText(img, "fine-band energy vs |z|  (manifold departure curve)",
               (30, 20), cv.FONT_HERSHEY_PLAIN, 1, (220, 220, 220), 1)
    cv.imwrite(f"{OUTDIR}/departure_curve.png", img)
    print(f"wrote {OUTDIR}/departure_curve.png")

# ------------------------------------------------------------- selftest
def selftest():
    global OUTDIR, SHARD_N
    ok = True
    def check(name, cond, note=""):
        nonlocal ok; ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {note}")
    import tempfile
    OUTDIR = tempfile.mkdtemp(); SHARD_N = 256
    dec = MockDecoder()
    # generator produces all four strategies with sane shapes
    g = gen_survey(np.random.default_rng(0))
    seen = {}
    for _ in range(600):
        s, p, z = next(g); seen[s] = seen.get(s, 0) + 1
        if len(z) != LATENT: ok = False
    check("survey covers strategies",
          all(k in seen for k in ("prior", "ray", "walk", "slerp")), str(seen))
    dump(gb=(256 * (THUMB*THUMB*3 + 256)) / 1e9, dec=dec)   # exactly 1 shard
    imgs = np.load(f"{OUTDIR}/shard_0000_img.npy")
    zs = np.load(f"{OUTDIR}/shard_0000_z.npy")
    check("shard shapes", imgs.shape == (256, 32, 32, 3)
          and zs.shape == (256, LATENT), f"{imgs.shape} {zs.shape}")
    # z roundtrip: stored z re-renders to (nearly) the stored thumbnail
    out = dec(zs[:8].astype(np.float32))
    th = to_thumbs(out)
    err = np.abs(th.astype(int) - imgs[:8].astype(int)).mean()
    check("z -> thumb roundtrip", err < 2.0, f"mean|d| {err:.3f} (f16 z)")
    check("sheet exists", os.path.exists(f"{OUTDIR}/sheet_0000.png"))
    # meta rows align
    with open(f"{OUTDIR}/shard_0000_meta.csv") as f:
        n = sum(1 for _ in f) - 1
    check("meta rows", n == 256, str(n))
    analyze()
    check("departure curve", os.path.exists(f"{OUTDIR}/departure_curve.png"))
    # budget math: 2 GB at 3.3 KB/img ~ 600k images
    n = int(2e9 / (THUMB*THUMB*3 + LATENT*2))
    check("2 GB budget ~ 600k imgs", 550_000 < n < 650_000, str(n))
    print("selftest:", "ALL PASS" if ok else "FAILURES ABOVE")
    return 0 if ok else 1

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--browse", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--gb", type=float, default=2.0)
    a = ap.parse_args()
    if a.selftest: sys.exit(selftest())
    elif a.analyze: analyze()
    elif a.browse: browse(OnnxDecoder())
    elif a.dump: dump(a.gb, OnnxDecoder())
    else: print(__doc__ or "use --dump / --browse / --analyze / --selftest")
