#!/usr/bin/env python3
# app.py — SplatWorld on Hugging Face Spaces (Gradio, runs in a browser).
#
# The desktop explorer (explore_desktop.py) uses a live OpenCV window with
# mouse-drag surfing. Spaces have no display, so this version puts the same
# world behind sliders and buttons:
#
#   Surf    : radius = dive (core<->fire), morph sliders = drag, buttons = new face
#   Zoom    : scrub one identity's dive, or render the full Shepard-zoom video
#   Gallery : sample a grid of faces
#   About   : what this is and how it works
#
# Put splat_decoder.onnx (7.2 MB) next to this file. Without it the app still
# boots and shows mock noise, so the Space never hard-fails on startup.
#
#   python app.py            # launch locally at http://127.0.0.1:7860
#
# Loaded via onnxruntime (CPU); falls back to opencv's dnn, then to a mock.

import os, math, tempfile, threading
import numpy as np
import gradio as gr
from PIL import Image

LATENT   = 128
RSEED    = 7
DISPLAY  = 384          # on-screen render size (model is 96px, upscaled)
R_ESC    = 45.0
P_OCT    = 3.0
MODEL    = os.environ.get("SPLAT_MODEL", "splat_decoder.onnx")

def smooth(t):
    t = float(np.clip(t, 0.0, 1.0));  return t * t * (3 - 2 * t)

def shell_name(zn):
    if zn < 15:  return "core (a face)"
    if zn < 35:  return "ghost"
    if zn < 100: return "fire (the soup)"
    return "void"

# ------------------------------------------------------------------ decoder
class MockDecoder:
    def __init__(self, h=48):
        g = np.random.default_rng(99).standard_normal((3 * h * h, LATENT))
        self.W = (g / math.sqrt(LATENT)).astype(np.float32); self.h = h
    def __call__(self, zs):
        y = np.tanh(zs.astype(np.float32) @ self.W.T) * 0.5 + 0.5
        return y.reshape(len(zs), 3, self.h, self.h)

class Decoder:
    """onnxruntime -> opencv dnn (classic engine, thread-local) -> mock.

    OpenCV 5's new dnn engine asserts 'buf.u == m.u' when a Net built on one
    thread is run on another (which Gradio always does). onnxruntime avoids
    this entirely; if only OpenCV is available we build one Net per worker
    thread and prefer the classic engine."""
    def __init__(self, path=MODEL):
        self.backend = None; self.note = ""; self.path = path
        self._local = threading.local()
        self._engine = None
        if os.path.exists(path):
            try:
                import onnxruntime as ort
                self.sess = ort.InferenceSession(
                    path, providers=["CPUExecutionProvider"])
                self.iname = self.sess.get_inputs()[0].name
                self.oname = self.sess.get_outputs()[0].name
                self.backend = "onnxruntime"
            except Exception as e_ort:
                try:
                    import cv2 as cv
                    self._cv = cv
                    # prefer the classic engine; fall back to whatever exists
                    self._engine = getattr(cv.dnn, "ENGINE_CLASSIC", None)
                    self._net()                       # validate it builds
                    self.backend = "opencv-dnn"
                    self.note = ("onnxruntime not installed — using OpenCV dnn. "
                                 "For best results run:  pip install onnxruntime")
                except Exception as e_cv:
                    self.note = f"onnx load failed: onnxruntime=({e_ort}); opencv=({e_cv})"
        if self.backend is None:
            self.mock = MockDecoder(); self.backend = "mock"
            if not self.note:
                self.note = (f"'{path}' not found — showing mock noise. "
                             f"Add the model to see faces.")

    def _net(self):
        """A cv2 dnn Net local to the calling thread (dodges the cross-thread
        buffer assertion in OpenCV 5's new engine)."""
        n = getattr(self._local, "net", None)
        if n is None:
            cv = self._cv
            if self._engine is not None:
                try:
                    n = cv.dnn.readNetFromONNX(self.path, engine=self._engine)
                except TypeError:                     # OpenCV 4: no engine kw
                    n = cv.dnn.readNetFromONNX(self.path)
            else:
                n = cv.dnn.readNetFromONNX(self.path)
            self._local.net = n
        return n

    def __call__(self, zs):
        zs = np.ascontiguousarray(zs.astype(np.float32))
        if self.backend == "onnxruntime":
            return self.sess.run([self.oname], {self.iname: zs})[0]
        if self.backend == "opencv-dnn":
            net = self._net()
            net.setInput(zs, "z_latent")
            return net.forward("rendered_image").copy()
        return self.mock(zs)

DEC = Decoder()

def to_rgb(out_chw, size=DISPLAY):
    """(3,H,W) float [0,1] -> upscaled HxWx3 uint8 RGB via PIL bicubic."""
    im = np.transpose(np.clip(out_chw, 0, 1), (1, 2, 0))
    im = (im * 255).astype(np.uint8)
    return np.asarray(Image.fromarray(im).resize((size, size), Image.BICUBIC))

def decode_one(z, size=DISPLAY):
    return to_rgb(DEC(z[None])[0], size)

# ------------------------------------------------------------------- charts
def direction_bank(seed=RSEED):
    g = np.random.default_rng(seed).standard_normal((LATENT, LATENT))
    q, _ = np.linalg.qr(g)
    return q.T.astype(np.float32)
BANK = direction_bank()

def tangent_axes(d, e1, e2):
    """Three orthonormal directions in the tangent plane of unit vector d,
    so the morph sliders rotate the identity instead of rescaling |z|."""
    a = e1 - (e1 @ d) * d; a /= (np.linalg.norm(a) + 1e-9)
    b = e2 - (e2 @ d) * d - (e2 @ a) * a; b /= (np.linalg.norm(b) + 1e-9)
    e3 = BANK[2]
    c = e3 - (e3 @ d) * d - (e3 @ a) * a - (e3 @ b) * b
    c /= (np.linalg.norm(c) + 1e-9)
    return np.stack([a, b, c])

# =================================================================== SURF ===
def new_dir(seed):
    rng = np.random.default_rng(int(seed) & 0x7fffffff)
    d = rng.standard_normal(LATENT); d /= np.linalg.norm(d)
    return d.astype(np.float32)

def surf_render(state, radius, mx, my, mz):
    d = np.asarray(state["d"], np.float32)
    ax = tangent_axes(d, BANK[state["e"] % LATENT], BANK[(state["e"] + 1) % LATENT])
    d2 = d + mx * ax[0] + my * ax[1] + mz * ax[2]
    d2 /= (np.linalg.norm(d2) + 1e-9)
    z = (d2 * float(radius)).astype(np.float32)
    zn = float(np.linalg.norm(z))
    return decode_one(z), f"**|z| = {zn:5.1f}**   ·   {shell_name(zn)}"

def surf_new_face(state):
    state = dict(state); state["d"] = new_dir(np.random.randint(1 << 30))
    img, info = surf_render(state, 8.0, 0.0, 0.0, 0.0)
    return state, img, info, 8.0, 0.0, 0.0, 0.0

def surf_reroll(state):
    state = dict(state); state["e"] = (state["e"] + 3) % LATENT
    return state, "_morph axes re-rolled — the sliders now push new directions_"

# =================================================================== ZOOM ===
class Journey:
    def __init__(self, wps): self.wp = wps
    def ident(self, k): return self.wp[k % len(self.wp)]
    def z(self, k, phi):
        K, N = self.ident(k), self.ident(k + 1)
        nK, nN = np.linalg.norm(K), np.linalg.norm(N)
        dK, dN = K / nK, N / nN
        if phi < 0.35:
            t = smooth(phi / 0.35); r = R_ESC + (nK - R_ESC) * t
            return (dK * r).astype(np.float32)
        elif phi < 0.65:
            return K.astype(np.float32)
        else:
            t = smooth((phi - 0.65) / 0.35); r = nK + (R_ESC - nK) * t
            om = math.acos(float(np.clip(dK @ dN, -1, 1)))
            d = dK if om < 1e-5 else \
                (math.sin((1-t)*om)*dK + math.sin(t*om)*dN) / math.sin(om)
            d /= np.linalg.norm(d)
            return (d * r).astype(np.float32)

def zoom_waypoints(seed, n=48):
    rng = np.random.default_rng(int(seed) & 0x7fffffff)
    return [rng.standard_normal(LATENT).astype(np.float32) * 0.6 for _ in range(n)]

def zoom_scrub(state, phi):
    jour = Journey(state["wps"])
    z = jour.z(0, float(phi))
    zn = float(np.linalg.norm(z))
    stage = ("descending into the face" if phi < 0.35 else
             "dwelling inside the face"  if phi < 0.65 else
             "rising back into the fire")
    return decode_one(z), f"phi {float(phi):.2f}  ·  |z| {zn:5.1f}  ·  {shell_name(zn)}  ·  {stage}"

def zoom_new_ids(state):
    state = dict(state); state["wps"] = zoom_waypoints(np.random.randint(1 << 30))
    img, info = zoom_scrub(state, 0.0)
    return state, img, info, 0.0

def zoom_video(state, cycles, progress=gr.Progress()):
    """Render a short Shepard zoom to mp4: fall through several identities."""
    import imageio
    jour = Journey(state["wps"])
    fps, per = 24, 48
    frames = []
    total = int(cycles) * per
    for f in progress.tqdm(range(total), desc="rendering zoom"):
        u = f / per; k = int(u); phi = u - k
        frames.append(decode_one(jour.z(k, phi), size=256))
    path = os.path.join(tempfile.mkdtemp(), "splat_zoom.mp4")
    imageio.mimwrite(path, frames, fps=fps, codec="libx264",
                     quality=8, macro_block_size=None)
    return path

# ================================================================ GALLERY ===
def gallery(n, radius, seed, progress=gr.Progress()):
    rng = np.random.default_rng(int(seed) & 0x7fffffff)
    n = int(n)
    zs = (rng.standard_normal((n, LATENT)) * float(radius)).astype(np.float32)
    imgs = []
    for i in progress.tqdm(range(n), desc="sampling faces"):
        imgs.append(decode_one(zs[i], size=224))
    return imgs

# ================================================================== THEORY ==
ABOUT = """
## SplatWorld — 202,599 faces in a 7 MB wave field

This decoder doesn't store pixels. It maps a 128-D point **z** to **256 Gabor
wave packets** — each one a little oriented wave with a position, a size, an
orientation, a frequency, and a *complex* amplitude (a cosine weight and a sine
weight). The picture is the **sum of all 256 packets interfering**. A face is a
phase-locked standing wave.

**Fire vs face.** Near the origin (`|z| < 15`) the packets phase-lock: peaks and
troughs cancel everywhere except along an eyebrow or a cheekbone. Far out
(`|z| > 35`) there's no training data, so the decoder stops orchestrating — the
packets decorrelate into drifting "fire". Zoom rides the radius: dive from fire
into a face and back, forever (a Shepard tone for the eye).

**The space between faces.** Moving a feature from A to B is *transport*. In a
fixed additive basis the only way is to fade one atom out while fading another
in — mid-way both exist and their phases fight, and that fight *is* the fire.
This is 1990s technology: **eigenfaces** interpolated faces linearly in 1991 and
produced exactly these ghosts. The loophole: a complex atom can **translate by
rotating its phase** (a Fourier shift) instead of crossfading — phase-transport
leaves no ghost, and that's the direction this whole line of work points at.

**Honest notes.** 96×96 is a VRAM limit, not a taste; hair and fine detail
struggle and samples skew toward a mean face. The "standing wave" language is a
faithful description of a Gabor renderer, not a claim of new physics. Trained on
CelebA (non-commercial research use — check the dataset's terms).

*Do not hype. Do not lie. Just show.*
"""

# =================================================================== BUILD ==
def build():
    with gr.Blocks(title="SplatWorld") as demo:
        gr.Markdown("# 🌊 SplatWorld\n"
                    "**202,599 CelebA faces compressed into a 7 MB wave-interference "
                    "field.** Dive through it below.")
        if DEC.backend == "mock":
            gr.Markdown(f"> ⚠️ {DEC.note}")
        else:
            gr.Markdown(f"<sub>model backend: <code>{DEC.backend}</code></sub>")

        with gr.Tabs():
            # ---- SURF ----
            with gr.Tab("Surf"):
                s_state = gr.State(value={"d": new_dir(RSEED), "e": 0})
                with gr.Row():
                    with gr.Column(scale=1):
                        s_img = gr.Image(label="face", height=DISPLAY, width=DISPLAY)
                        s_info = gr.Markdown()
                    with gr.Column(scale=1):
                        s_r = gr.Slider(1, 55, value=8, step=0.5,
                                        label="radius · dive core (a face) ↔ rise into the fire")
                        gr.Markdown("**morph** — nudge the identity along three directions:")
                        s_x = gr.Slider(-1.5, 1.5, value=0, step=0.02, label="morph A")
                        s_y = gr.Slider(-1.5, 1.5, value=0, step=0.02, label="morph B")
                        s_z = gr.Slider(-1.5, 1.5, value=0, step=0.02, label="morph C")
                        with gr.Row():
                            s_new = gr.Button("🎲 new face")
                            s_roll = gr.Button("↻ re-roll morph axes")
                        s_roll_info = gr.Markdown()
                inp = [s_state, s_r, s_x, s_y, s_z]
                for ctrl in (s_r, s_x, s_y, s_z):
                    ctrl.change(surf_render, inp, [s_img, s_info])
                s_new.click(surf_new_face, s_state,
                            [s_state, s_img, s_info, s_r, s_x, s_y, s_z])
                s_roll.click(surf_reroll, s_state, [s_state, s_roll_info])
                demo.load(surf_render, inp, [s_img, s_info])

            # ---- ZOOM ----
            with gr.Tab("Zoom"):
                z_state = gr.State(value={"wps": zoom_waypoints(RSEED)})
                with gr.Row():
                    with gr.Column():
                        z_img = gr.Image(label="one identity's dive",
                                         height=DISPLAY, width=DISPLAY)
                        z_info = gr.Markdown()
                    with gr.Column():
                        z_phi = gr.Slider(0, 0.999, value=0, step=0.005,
                                          label="phi · scrub: fire → face → fire")
                        z_new = gr.Button("🎲 new identities")
                        gr.Markdown("**Full Shepard zoom** (falls through many faces, "
                                    "the scale reset hidden in the fire):")
                        z_cycles = gr.Slider(2, 8, value=4, step=1,
                                             label="identities to fall through")
                        z_go = gr.Button("🎬 render zoom video")
                        z_vid = gr.Video(label="Shepard zoom")
                z_phi.change(zoom_scrub, [z_state, z_phi], [z_img, z_info])
                z_new.click(zoom_new_ids, z_state, [z_state, z_img, z_info, z_phi])
                z_go.click(zoom_video, [z_state, z_cycles], z_vid)
                demo.load(zoom_scrub, [z_state, z_phi], [z_img, z_info])

            # ---- GALLERY ----
            with gr.Tab("Gallery"):
                with gr.Row():
                    g_n = gr.Slider(4, 64, value=24, step=4, label="how many faces")
                    g_r = gr.Slider(0.2, 3.0, value=0.6, step=0.1, label="radius (spread)")
                    g_seed = gr.Number(value=7, label="seed", precision=0)
                g_go = gr.Button("🖼️ sample faces")
                g_out = gr.Gallery(label="samples", columns=6, height=560)
                g_go.click(gallery, [g_n, g_r, g_seed], g_out)

            # ---- ABOUT ----
            with gr.Tab("About / Theory"):
                gr.Markdown(ABOUT)

    return demo

if __name__ == "__main__":
    demo = build()
    try:
        demo.launch(theme=gr.themes.Soft())     # Gradio 6: theme lives here
    except TypeError:
        demo.launch()                            # older Gradio: theme was on Blocks

