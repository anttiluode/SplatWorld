#!/usr/bin/env python3
# app.py — SplatWorld: walk around inside a 7 MB wave-interference face field.
#
# One decoder (splat_decoder.onnx) maps a 128-D point z to 256 Gabor wave
# packets. Their interference IS the image. A face is a phase-locked standing
# wave; the "fire" you see between faces is those same waves decorrelating.
# This app is the instrument you explore that world with.
#
#   python app.py                      # launch (needs splat_decoder.onnx here)
#   python app.py --gallery 64         # headless: dump 64 faces to ./gallery/
#   python app.py --selftest           # no window; check the math
#
# ---- GLOBAL KEYS -----------------------------------------------------------
#   1  ZOOM mode        2  SURF mode        3  ATLAS info
#   H  theory (cycles pages)   B  bump a face gallery to disk
#   R  record video on/off     S  save this frame       Q  quit
#
# ---- ZOOM (automated Shepard dive, hands-free) -----------------------------
#   wheel / UP,DOWN  zoom speed (negative = reverse)
#   LEFT / RIGHT     skip to previous / next identity     SPACE  pause
#   The camera falls down one identity's ray until the face resolves, dwells,
#   then rises back into the fire and turns toward the next identity. The scale
#   reset is hidden in the fire (a visual Shepard tone) so the dive never ends.
#
# ---- SURF (free flight, no charts, no TAB) ---------------------------------
#   drag             morph the identity (rotate your ray through face-space)
#   wheel            dive toward the core (a face) or rise into the fire
#   SPACE            jump to a fresh random face
#   N                re-roll the drag plane (a new pair of morph directions)
#   Radius = how phase-locked the waves are (core=face, fire=soup).
#   Direction = which identity. They are decoupled on purpose.
#
# Honest notes:
#  - render is native model res (96px) upscaled with cubic to the window;
#    it is not true super-resolution.
#  - SURF's drag spans a 2-plane of the 128-D tangent space at a time; press N
#    for a fresh plane. It is a hang-glider, not a spaceship — that is the fun.

import argparse, glob, math, os, sys, time
import numpy as np

LATENT = 128
RSEED  = 7
WIN    = 768
R_ESC  = 45.0          # escape radius: deep fire
P_OCT  = 3.0           # octaves of zoom per identity cycle
WRAP_W = 0.12
DESC, DWEND = 0.35, 0.65
EPS = 1e-9

def smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)

def shell_name(zn):
    if zn < 15:  return "core"
    if zn < 35:  return "ghost"
    if zn < 100: return "fire"
    return "void"

# ----------------------------------------------------------------- decoders
class OnnxDecoder:
    def __init__(self, path="splat_decoder.onnx"):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Put the 7 MB splat_decoder.onnx next to this "
                f"file (see the README for where to get it).")
        self.backend = None
        # Prefer onnxruntime: OpenCV 5's dnn importer cannot parse this graph's
        # ConstantOfShape (dynamic batch) node and dies on batched forwards.
        try:
            import onnxruntime as ort
            self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self.iname = self.sess.get_inputs()[0].name
            self.oname = self.sess.get_outputs()[0].name
            self.backend = "onnxruntime"
            print("decoder backend: onnxruntime")
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
            print(f"decoder backend: opencv-dnn  (onnxruntime unavailable: {e})")
            print("  -> on OpenCV 5, install onnxruntime for full/batched support:")
            print("     python -m pip install onnxruntime")
    def __call__(self, zs):
        zs = np.ascontiguousarray(zs.astype(np.float32))
        if self.backend == "onnxruntime":
            return self.sess.run([self.oname], {self.iname: zs})[0]
        self.net.setInput(zs, "z_latent")
        return self.net.forward("rendered_image").copy()   # (N,3,H,W) in [0,1]

class MockDecoder:
    """A stand-in so --selftest runs without the real model or a GPU."""
    def __init__(self, h=48):
        g = np.random.default_rng(99).standard_normal((3 * h * h, LATENT))
        self.W = (g / math.sqrt(LATENT)).astype(np.float32)
        self.h = h
    def __call__(self, zs):
        y = np.tanh(zs.astype(np.float32) @ self.W.T) * 0.5 + 0.5
        return y.reshape(len(zs), 3, self.h, self.h)

# ================================================================= ZOOM ====
def prior_waypoints(rng, n=64):
    return [rng.standard_normal(LATENT).astype(np.float32) * 0.6 for _ in range(n)]

def atlas_waypoints(rng, n=64, outdir="atlas"):
    """Real core faces from a dumped atlas, if one exists."""
    import csv
    picks = []
    for mf in sorted(glob.glob(f"{outdir}/shard_*_meta.csv")):
        tag = mf.split("shard_")[1].split("_")[0]
        try:
            zs = np.load(f"{outdir}/shard_{tag}_z.npy")
        except OSError:
            continue
        with open(mf) as f:
            for i, row in enumerate(csv.DictReader(f)):
                if row["strategy"] == "prior" and float(row["param"]) <= 1.0:
                    picks.append(zs[i].astype(np.float32))
        if len(picks) > 4 * n:
            break
    if not picks:
        return prior_waypoints(rng, n)
    rng.shuffle(picks)
    return picks[:n]

class Journey:
    """z(cycle,phi): fire -> down identity K's ray -> dwell -> back to fire."""
    def __init__(self, waypoints):
        self.wp = waypoints
    def ident(self, k):
        return self.wp[k % len(self.wp)]
    def z(self, k, phi):
        K, N = self.ident(k), self.ident(k + 1)
        nK, nN = np.linalg.norm(K), np.linalg.norm(N)
        dK, dN = K / nK, N / nN
        if phi < DESC:
            t = smooth(phi / DESC)
            r = R_ESC + (nK - R_ESC) * t
            return (dK * r).astype(np.float32)
        elif phi < DWEND:
            return K.astype(np.float32)
        else:
            t = smooth((phi - DWEND) / (1.0 - DWEND))
            r = nK + (R_ESC - nK) * t
            om = math.acos(float(np.clip(dK @ dN, -1, 1)))
            if om < 1e-5:
                d = dK
            else:
                d = (math.sin((1 - t) * om) * dK + math.sin(t * om) * dN) / math.sin(om)
            d /= np.linalg.norm(d)
            return (d * r).astype(np.float32)

def voices(u):
    """Zoom coord u (cycles) -> [(k, phi, scale, weight)]; blends across wraps."""
    k = int(math.floor(u)); phi = u - k; out = []
    if phi < WRAP_W:
        a = smooth((phi + WRAP_W) / (2 * WRAP_W))
        out.append((k,     phi,       2.0 ** (P_OCT * phi),        a))
        out.append((k - 1, phi + 1.0, 2.0 ** (P_OCT * (phi + 1)),  1 - a))
    elif phi > 1.0 - WRAP_W:
        a = smooth((phi - (1.0 - WRAP_W)) / (2 * WRAP_W))
        out.append((k,     phi,       2.0 ** (P_OCT * phi),        1 - a))
        out.append((k + 1, phi - 1.0, 2.0 ** (P_OCT * (phi - 1)),  a))
    else:
        out.append((k, phi, 2.0 ** (P_OCT * phi), 1.0))
    return out

def compose_zoom(dec, jour, u, drift_rho=0.0):
    import cv2 as cv
    vs = voices(u)
    zs = np.stack([jour.z(k, phi) for k, phi, _, _ in vs])
    outs = dec(zs)
    acc = None
    for (k, phi, s, w), out in zip(vs, outs):
        im = np.transpose(out, (1, 2, 0)).astype(np.float32)
        h = im.shape[0]
        M = cv.getRotationMatrix2D((h / 2, h / 2), drift_rho * 360, s)
        warped = cv.warpAffine(im, M, (h, h), flags=cv.INTER_CUBIC,
                               borderMode=cv.BORDER_REFLECT)
        acc = warped * w if acc is None else acc + warped * w
    big = cv.resize(acc, (WIN, WIN), interpolation=cv.INTER_CUBIC)
    zn = float(np.linalg.norm(zs[0]))
    return np.clip(big, 0, 1), zn, vs[0][0], vs[0][1]

# ================================================================= SURF ====
def tangent_axes(d, e1, e2):
    """Two orthonormal directions in the tangent plane of unit vector d,
    built by projecting fixed globals e1,e2 off d (so dragging rotates the
    identity, it never just rescales |z|)."""
    a = e1 - (e1 @ d) * d
    a /= (np.linalg.norm(a) + EPS)
    b = e2 - (e2 @ d) * d - (e2 @ a) * a
    b /= (np.linalg.norm(b) + EPS)
    return a, b

DRAG_GAIN = 0.004
WHEEL_STEP = 1.5
SURF_SM = 0.25

class SurfFree:
    """Free flight: a unit direction (identity) + a radius (phase-lock depth)."""
    def __init__(self, rng):
        self.rng = rng
        g = rng.standard_normal((LATENT, LATENT))
        q, _ = np.linalg.qr(g)
        self.bank = q.T.astype(np.float32)
        self.e1, self.e2 = self.bank[0], self.bank[1]
        self.reseed()
    def reseed(self):
        d = self.rng.standard_normal(LATENT)
        self.td = (d / np.linalg.norm(d)).astype(np.float32)
        self.tr = 8.0                       # start just inside a face
        self.d = self.td.copy(); self.r = self.tr
    def reroll_plane(self):
        i = int(self.rng.integers(0, LATENT - 1))
        self.e1, self.e2 = self.bank[i], self.bank[(i + 1) % LATENT]
    def drag(self, dx, dy, fine=False):
        a, b = tangent_axes(self.td, self.e1, self.e2)
        g = DRAG_GAIN * (0.2 if fine else 1.0)
        nd = self.td + g * (dx * a - dy * b)
        self.td = (nd / np.linalg.norm(nd)).astype(np.float32)
    def wheel(self, notches):
        self.tr = float(np.clip(self.tr + notches * WHEEL_STEP, 1.0, 55.0))
    def tick(self):
        self.d += SURF_SM * (self.td - self.d)
        self.d /= (np.linalg.norm(self.d) + EPS)
        self.r += SURF_SM * (self.tr - self.r)
    def z(self):
        return (self.d * self.r).astype(np.float32)

def compose_surf(dec, surf):
    import cv2 as cv
    out = dec(surf.z()[None])[0]
    im = np.transpose(out, (1, 2, 0)).astype(np.float32)
    big = cv.resize(im, (WIN, WIN), interpolation=cv.INTER_CUBIC)
    return np.clip(big, 0, 1), float(np.linalg.norm(surf.z()))

# ================================================================ THEORY ===
THEORY = [
    ("WHAT IS THIS", [
        "SplatWorld: 202,599 CelebA faces living inside one",
        "7.2 MB decoder. It does NOT store pixels.",
        "",
        "A tiny MLP maps a 128-D point z to 256 Gabor 'atoms'.",
        "Each atom = position, size (sigma), orientation (theta),",
        "frequency, and a complex (a,b) amplitude per color:",
        "the cosine weight and the sine weight of a little wave.",
        "",
        "The picture is the SUM of 256 vibrating wave packets.",
        "A face is not painted. It is an interference pattern.",
        "Trained at 96x96 (a VRAM limit), 30k steps, ~5.7M params.",
    ]),
    ("FIRE vs FACE  (how it works)", [
        "Core  (|z| < 15): the atoms PHASE-LOCK. Peaks and troughs",
        "line up to cancel everywhere except along an eyebrow or a",
        "cheekbone. A standing wave. That is a face.",
        "",
        "Fire  (|z| > 35): no training data lives out here, so the",
        "decoder stops orchestrating. The atoms DECORRELATE, their",
        "envelopes wander and phases drift. That soup is the fire.",
        "",
        "ZOOM rides the radius |z|: dive from fire -> ghost -> face,",
        "then back out. The scale reset hides in the fire, so the",
        "dive loops forever (a Shepard tone for the eye).",
    ]),
    ("THE SPACE BETWEEN  (the theory)", [
        "Between two faces you see splats and soup. Why?",
        "Moving a feature from A to B is TRANSPORT. In a fixed",
        "additive basis, transport means: fade one atom out while",
        "fading another in. Mid-way both exist and their phases",
        "fight -> interference. That fight IS the fire.",
        "",
        "This is 1990s tech: eigenfaces did face-space by linear",
        "transfer and got exactly these ghostly double-exposures.",
        "",
        "The loophole: a complex (a,b) atom can TRANSLATE by rotating",
        "its phase (a Fourier shift) instead of crossfading. Phase-",
        "transport leaves no ghost. That is the direction worth chasing.",
        "",
        "Squeeze faces together (raise beta) -> smooth morphs but",
        "identities collapse to a mean. Push apart (low beta) -> crisp",
        "faces, vast fire between. The gap is a tug-of-war.",
    ]),
    ("CONTROLS", [
        "1 zoom   2 surf   3 atlas   H theory   B gallery",
        "R record   S save frame   Q quit",
        "",
        "ZOOM:  wheel = speed,  arrows = skip identity,  SPACE = pause",
        "SURF:  drag = morph identity,  wheel = dive(core)<->rise(fire),",
        "       SPACE = new face,  N = re-roll the drag plane",
        "",
        "ATLAS is a separate surveyor (splat_atlas.py). Run --dump",
        "once to bake thousands of thumbnails, then --browse to click",
        "through them (n/p flip pages). See the README.",
    ]),
]

def draw_panel(img, title, lines):
    import cv2 as cv
    ov = img.copy()
    cv.rectangle(ov, (0, 0), (WIN, WIN), (12, 12, 16), -1)
    cv.addWeighted(ov, 0.82, img, 0.18, 0, img)
    y = 64
    cv.putText(img, title, (44, y), cv.FONT_HERSHEY_DUPLEX, 1.0,
               (0, 255, 255), 1, cv.LINE_AA)
    y += 20
    cv.line(img, (44, y), (WIN - 44, y), (0, 120, 120), 1, cv.LINE_AA)
    y += 34
    for ln in lines:
        cv.putText(img, ln, (44, y), cv.FONT_HERSHEY_PLAIN, 1.15,
                   (210, 210, 210), 1, cv.LINE_AA)
        y += 30
    cv.putText(img, "H = next page   any mode key = leave",
               (44, WIN - 28), cv.FONT_HERSHEY_PLAIN, 1.0,
               (120, 200, 120), 1, cv.LINE_AA)
    return img

# =============================================================== GALLERY ===
def dump_gallery(dec, n=64, outdir="gallery", radius=0.6, seed=None):
    import cv2 as cv
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    zs = (rng.standard_normal((n, LATENT)) * radius).astype(np.float32)
    outs = dec(zs)
    tiles = []
    for i, out in enumerate(outs):
        im = (np.transpose(out, (1, 2, 0)) * 255).clip(0, 255).astype(np.uint8)
        im = cv.cvtColor(im, cv.COLOR_RGB2BGR)
        big = cv.resize(im, (256, 256), interpolation=cv.INTER_CUBIC)
        cv.imwrite(f"{outdir}/face_{i:03d}.png", big)
        tiles.append(cv.resize(im, (96, 96), interpolation=cv.INTER_CUBIC))
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    sheet = np.zeros((rows * 96, cols * 96, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r*96:(r+1)*96, c*96:(c+1)*96] = t
    cv.imwrite(f"{outdir}/contact_sheet.png", sheet)
    return outdir, n

# =============================================================== SELFTEST ==
def selftest():
    ok = True
    def check(name, cond, note=""):
        nonlocal ok; ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {note}")

    rng = np.random.default_rng(0)

    # zoom path is C0 across the whole cycle incl. the waypoint hand-off
    j = Journey(prior_waypoints(rng, 8))
    us = np.linspace(0.001, 2.999, 900); prev = None; mx = 0.0
    for u in us:
        k = int(u); z = j.z(k, u - k)
        if prev is not None: mx = max(mx, float(np.linalg.norm(z - prev)))
        prev = z
    check("zoom path C0", mx < 2.0, f"max step {mx:.3f}")
    check("dwell is identity K", np.allclose(j.z(0, 0.5), j.ident(0), atol=1e-6))
    gap = np.linalg.norm(j.z(0, 1 - 1e-9) - j.z(1, 0.0))
    check("z C0 across wrap", gap < 1e-3, f"gap {gap:.2e}")
    for u in (0.05, 0.5, 0.95, 1.0, 1.03):
        ws = sum(w for _, _, _, w in voices(u))
        check(f"voice weights sum to 1 @u={u}", abs(ws - 1) < 1e-6, f"{ws:.6f}")

    # surf: tangent axes are orthonormal and perpendicular to the ray;
    # dragging rotates the identity but keeps it a unit vector; wheel clamps
    s = SurfFree(np.random.default_rng(1))
    a, b = tangent_axes(s.td, s.e1, s.e2)
    check("tangent axes orthonormal",
          abs(a @ a - 1) < 1e-5 and abs(b @ b - 1) < 1e-5 and abs(a @ b) < 1e-5)
    check("tangent axes perp to ray", abs(a @ s.td) < 1e-5 and abs(b @ s.td) < 1e-5)
    d0 = s.td.copy(); s.drag(120, -40)
    check("drag rotates identity", not np.allclose(s.td, d0, atol=1e-4))
    check("identity stays unit", abs(np.linalg.norm(s.td) - 1) < 1e-5)
    s.wheel(1000); check("wheel clamps radius", s.tr == 55.0, f"{s.tr}")
    s.wheel(-1000); check("wheel clamps low", s.tr == 1.0, f"{s.tr}")
    [s.tick() for _ in range(40)]
    check("shown ray converges", np.allclose(s.d, s.td, atol=1e-3))

    # compositors render sane frames with the mock decoder
    dec = MockDecoder()
    fr, zn, k, phi = compose_zoom(dec, j, 0.97)
    check("zoom frame shape", fr.shape == (WIN, WIN, 3), str(fr.shape))
    check("zoom frame in range", 0 <= fr.min() and fr.max() <= 1.0)
    fr2, zn2 = compose_surf(dec, s)
    check("surf frame shape", fr2.shape == (WIN, WIN, 3), str(fr2.shape))
    check("shell classifier", shell_name(5) == "core" and shell_name(50) == "fire")

    # gallery dumps files + a contact sheet
    import tempfile
    gd = tempfile.mkdtemp()
    outdir, n = dump_gallery(dec, n=9, outdir=gd)
    check("gallery wrote faces", os.path.exists(f"{gd}/face_000.png") and n == 9)
    check("gallery contact sheet", os.path.exists(f"{gd}/contact_sheet.png"))

    check("theory pages present", len(THEORY) == 4 and all(t[1] for t in THEORY))
    print("selftest:", "ALL PASS" if ok else "FAILURES ABOVE")
    return 0 if ok else 1

# =================================================================== LIVE ==
def try_launch_atlas():
    """If an atlas exists, open its browser; else print how to build one."""
    if glob.glob("atlas/sheet_*.png"):
        import subprocess
        try:
            subprocess.Popen([sys.executable, "splat_atlas.py", "--browse"])
            return "launched splat_atlas.py --browse"
        except Exception as e:
            return f"could not launch atlas browser: {e}"
    return "no atlas yet -> run:  python splat_atlas.py --dump --gb 2"

def live():
    import cv2 as cv
    dec = OnnxDecoder()
    rng = np.random.default_rng()
    wps = atlas_waypoints(rng)
    jour = Journey(wps)
    surf = SurfFree(rng)

    st = {"mode": "zoom", "wheel": 0}
    u, vel, paused = 0.0, 0.10, False
    theory_page = 0
    atlas_msg = ""
    writer = None
    win = "SplatWorld  [1]zoom [2]surf [3]atlas [H]theory [B]gallery [Q]quit"
    cv.namedWindow(win, cv.WINDOW_NORMAL)

    state = {"btn": 0, "px": 0, "py": 0}
    def on_mouse(ev, x, y, flags, _):
        if ev == cv.EVENT_MOUSEWHEEL:
            st["wheel"] = 1 if flags > 0 else -1
        elif ev in (cv.EVENT_LBUTTONDOWN, cv.EVENT_RBUTTONDOWN):
            state["btn"] = 1 if ev == cv.EVENT_LBUTTONDOWN else 2
            state["px"], state["py"] = x, y
        elif ev in (cv.EVENT_LBUTTONUP, cv.EVENT_RBUTTONUP):
            state["btn"] = 0
        elif ev == cv.EVENT_MOUSEMOVE and state["btn"] and st["mode"] == "surf":
            surf.drag(x - state["px"], y - state["py"], fine=(state["btn"] == 2))
            state["px"], state["py"] = x, y
    cv.setMouseCallback(win, on_mouse)

    tprev = time.time()
    print("SplatWorld flying. 1=zoom 2=surf 3=atlas H=theory B=gallery Q=quit")
    while True:
        now = time.time(); dt = min(0.1, now - tprev); tprev = now
        wheel = st["wheel"]; st["wheel"] = 0

        if st["mode"] == "zoom":
            if wheel: vel += 0.02 * wheel
            if not paused: u += vel * dt
            fr, zn, k, phi = compose_zoom(dec, jour, u, drift_rho=0.02 * u)
            hud = (f"ZOOM  identity {k}  phi {phi:.2f}  |z| {zn:5.1f}  "
                   f"[{shell_name(zn)}]  vel {vel:+.2f}")
        elif st["mode"] == "surf":
            if wheel: surf.wheel(wheel)
            surf.tick()
            fr, zn = compose_surf(dec, surf)
            hud = f"SURF  |z| {zn:5.1f}  [{shell_name(zn)}]  drag=morph wheel=dive"
        else:  # atlas info screen
            fr = np.zeros((WIN, WIN, 3), np.float32)
            zn = 0.0; hud = "ATLAS"

        im = cv.cvtColor((fr * 255).astype(np.uint8), cv.COLOR_RGB2BGR)

        if st["mode"] == "atlas":
            draw_panel(im, "ATLAS  (separate surveyor)", [
                "The atlas bakes thousands of thumbnails to disk so you",
                "can eyeball the whole latent space and click any tile",
                "back to a full-res face + its z.",
                "",
                "  python splat_atlas.py --dump --gb 2     (build it once)",
                "  python splat_atlas.py --browse          (n/p flip pages)",
                "  python splat_atlas.py --analyze         (departure curves)",
                "",
                f"status: {atlas_msg or 'press A to open the browser if built'}",
            ])
        elif st["mode"] == "theory":
            draw_panel(im, *THEORY[theory_page])
        else:
            cv.putText(im, hud, (12, WIN - 14), cv.FONT_HERSHEY_PLAIN,
                       1.15, (0, 255, 0), 1, cv.LINE_AA)

        if writer is not None:
            writer.write(im)
        cv.imshow(win, im)
        key = cv.waitKeyEx(1)
        if key == -1:
            continue
        k = key & 0xFF
        if k == ord('q'):
            break
        elif k == ord('1'): st["mode"] = "zoom"
        elif k == ord('2'): st["mode"] = "surf"
        elif k == ord('3'): st["mode"] = "atlas"
        elif k == ord('h'):
            if st["mode"] == "theory":
                theory_page = (theory_page + 1) % len(THEORY)
            else:
                st["mode"] = "theory"; theory_page = 0
        elif k == ord('a') and st["mode"] == "atlas":
            atlas_msg = try_launch_atlas(); print(atlas_msg)
        elif k == ord('b'):
            outdir, n = dump_gallery(dec, n=64)
            print(f"bumped {n} faces -> ./{outdir}/  (+ contact_sheet.png)")
        elif k == ord('r'):
            if writer is None:
                fn = f"splatworld_{int(time.time())}.mp4"
                writer = cv.VideoWriter(fn, cv.VideoWriter_fourcc(*"mp4v"),
                                        30, (WIN, WIN))
                print("recording ->", fn)
            else:
                writer.release(); writer = None; print("recording stopped")
        elif k == ord('s'):
            fn = f"frame_{int(time.time())}.png"; cv.imwrite(fn, im); print("saved", fn)
        elif k == ord(' '):
            if st["mode"] == "zoom": paused = not paused
            elif st["mode"] == "surf": surf.reseed()
        elif k == ord('n') and st["mode"] == "surf":
            surf.reroll_plane(); print("surf plane re-rolled")
        elif key in (2490368,) and st["mode"] == "zoom": vel += 0.02   # UP
        elif key in (2621440,) and st["mode"] == "zoom": vel -= 0.02   # DOWN
        elif key in (2555904,) and st["mode"] == "zoom": u = math.floor(u) + 1.0
        elif key in (2424832,) and st["mode"] == "zoom": u = max(0.0, math.floor(u) - 1.0)

    if writer is not None:
        writer.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SplatWorld explorer")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--gallery", type=int, default=None, metavar="N",
                    help="headless: dump N faces to ./gallery/ and exit")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    elif a.gallery is not None:
        outdir, n = dump_gallery(OnnxDecoder(), n=a.gallery)
        print(f"wrote {n} faces -> ./{outdir}/ (+ contact_sheet.png)")
    else:
        live()
