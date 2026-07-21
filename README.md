---
title: SplatWorld
emoji: 🌊
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: cc-by-nc-4.0
tags:
  - gabor
  - wave-optics
  - vae
  - generative
  - faces
  - celeba
  - onnx
---

<!-- ^^^ Hugging Face Space header. On GitHub, delete this block (or keep it —
     GitHub renders it as an empty comment). Adjust sdk_version to a version
     Hugging Face currently offers if the build complains. -->

# SplatWorld

**202,599 CelebA faces compressed into one 7.2 MB decoder — and it doesn't store a single pixel.**

SplatWorld is a small VAE whose decoder doesn't paint images. It maps a 128‑dimensional point `z` to **256 Gabor wave packets** and lets their interference *become* the picture. A face is a phase‑locked standing wave. The "fire" you see drifting between faces is those same waves coming unlocked. This repo gives you two ways to walk around inside that world.

---

## Two apps

| | run it | what it is |
|---|---|---|
| **`app.py`** (Gradio) | `python app.py` → opens in a browser | Runs anywhere, including **Hugging Face Spaces**. Surf with sliders, scrub the zoom, render a zoom video, sample a gallery. This is the Space entry point. |
| **`explore_desktop.py`** (OpenCV) | `python explore_desktop.py` | The richer **local** tool: a live window with real mouse‑drag surfing and a hands‑free automated zoom. Needs a display, so it can't run on Spaces. |

Both need `splat_decoder.onnx` (7.2 MB) sitting next to them. The Gradio app still boots without it (showing mock noise), so a Space never hard‑fails on startup — but add the model to see faces.

---

## Quick start (browser app)

```bash
pip install -r requirements.txt
# put splat_decoder.onnx next to app.py, then:
python app.py                 # http://127.0.0.1:7860
```

On **Hugging Face**: create a Gradio Space, upload these files plus `splat_decoder.onnx`, and it runs. The header block at the top of this README is the Space config.

## Quick start (desktop app)

```bash
pip install -r requirements-desktop.txt
python explore_desktop.py     # live OpenCV window
python explore_desktop.py --selftest
python explore_desktop.py --gallery 64   # headless: dump faces to ./gallery/
```

Desktop keys: `1` zoom · `2` surf · `3` atlas · `H` theory · `B` gallery · `R` record · `S` save · `Q` quit. In **surf**, drag to morph, wheel to dive/rise, `SPACE` for a new face, `N` to re‑roll the drag plane.

---

## The atlas (`splat_atlas.py`)

The atlas is a heavier, separate tool. It systematically **bakes thousands of thumbnails to disk** (with every `z` saved) so you can survey the whole space at a glance and click any tile back to a full‑res face.

```bash
python splat_atlas.py --dump --gb 2   # build it ONCE (writes ./atlas/, a few GB)
python splat_atlas.py --browse        # click a tile -> full res + z;  n/p flip pages
python splat_atlas.py --analyze       # fine-band energy vs |z|: the "departure curve"
```

`--browse` will say *"no atlas found — run --dump first"* until you've dumped. Once `./atlas/` exists, `app.py`'s zoom automatically uses the real core faces in it as waypoints.

---

## How it works

The decoder is a tiny MLP: `z (128-D) → 256 packets × 11 numbers`. Each packet is a Gabor atom — **position, size (σ), orientation (θ), frequency, and a complex `(a, b)` amplitude per color channel**, i.e. the cosine weight and the sine weight of a little oriented wave. The renderer drops all 256 vibrating drumheads on a 96×96 canvas and sums their ripples with explicit `env·cos` and `env·sin` math. No convolutions, no pixels — just interference.

- **Core** (`|z| < 15`): the atoms **phase‑lock**. Peaks and troughs align to cancel everywhere except along a sharp eyebrow or a soft cheekbone. A standing wave — a face.
- **Fire** (`|z| > 35`): no training data lives out here, so the decoder stops orchestrating. Envelopes wander, phases drift, packets beat against each other. That decorrelating soup is what you're flying through between faces.

Training is coarse‑to‑fine and you can watch it happen: the **envelope** code (where the mass sits) commits first, the **carrier** code (orientation and phase) commits second. Early in training the eyebrows are flat horizontal blobs (envelope only); once the coarse loss plateaus the model finally commits an orientation and the brows snap into their characteristic slant. Amplitude first, phase later.

---

## The space between faces (the theory)

Why is there *anything* between two faces? Because moving a feature from A to B is **transport**, and transport is where linear models break.

In a fixed additive basis, the only way to move an eyebrow is to fade one atom out while fading another in. Mid‑way both exist, their phases fight, and you get interference. **The fire is that fight made visible.** This is not new — it's 1990s technology. *Eigenfaces* built face‑space as linear combinations of basis faces in 1991, and interpolating between two of them produced exactly these ghostly double‑exposures. A purely additive "image model by linear transfer" is ancient, and the fire is its failure mode frozen in place.

There's a loophole worth naming, because it's the direction this whole line of work keeps pointing at. A complex `(a, b)` atom can **translate a feature by rotating its phase** — the Fourier shift theorem — instead of crossfading amplitudes. Phase‑transport moves the wave without ever destroying it, so it leaves no ghost. The atoms here already carry that complex structure; a "good" version of this world would move features by rotating phases, not sliding coefficients.

And the size of the gap is a tug‑of‑war you can feel in the trainer. Squeeze faces together (raise the VAE's β) and the morphs get buttery but identities collapse toward a bland mean. Push them apart (low β) and every face is crisp but the fire between them is vast. Crisp‑and‑dense at once — density without collapse — is the actual open problem.

> If you want the quantitative version: `splat_atlas.py --analyze` measures fine‑band energy as a function of `|z|`. That curve is "the splats appear and grow," plotted. The local geometry of the gaps (how fast the image changes per step of `z`, direction by direction) is a Riemannian pullback metric — mapping it would tell you exactly where the fire is thick and where faces slide smoothly into each other.

---

## Troubleshooting

**`cv2.error ... buf.u == m.u` on OpenCV 5.** OpenCV 5's new dnn engine crashes
on this graph when the browser app runs the model in a worker thread. Two fixes,
either works:

- **Install onnxruntime** (recommended, and what the Space uses):
  `pip install onnxruntime`. The app prefers it automatically and it sidesteps
  OpenCV entirely.
- If you only have OpenCV, the app now builds a **per-thread** net and prefers
  the **classic** engine, which avoids the assertion. Just update to this
  version of `app.py`.

The status line under the title tells you which backend loaded
(`onnxruntime`, `opencv-dnn`, or `mock`).

**The Space shows mock noise.** `splat_decoder.onnx` isn't next to `app.py`. Add
it (commit it to the Space repo, or use Git LFS).

---

## Model card / provenance

| | |
|---|---|
| **Data** | CelebA, 202,599 aligned face images |
| **Resolution** | 96×96 (chosen to fit in VRAM, not for quality) |
| **Latent** | 128‑D |
| **Packets** | 256 Gabor atoms, 11 params each |
| **Params** | ~5.74M |
| **Training** | 30,000 steps, batch 96, cosine LR, β warm‑up |
| **Export** | ONNX, 7.2 MB, dynamic batch axis (many `z` per forward) |
| **Inference** | CPU‑friendly via OpenCV `cv2.dnn`; no GPU or PyTorch required |

The trainer itself is not in this repo — this repo is the *explorer*. (Training code lives with the model line, e.g. TheSplat5.)

---

## Honest ledger

*Do not hype. Do not lie. Just show.*

**What's real and shown here**
- The decoder genuinely reconstructs and samples CelebA‑like faces from a 7.2 MB file, live.
- The core→fire shell structure, and the coarse‑to‑fine (envelope‑then‑carrier) commitment, are directly observable in the tools.
- Zoom's `z`‑path is C0 across identity hand‑offs and across the scale wrap; the surf tangent axes are provably orthonormal (see `--selftest`).

**What's honest to admit**
- 96×96 is a memory limit, not an aesthetic. Hair and fine detail struggle; sampled faces skew toward the population mean.
- The "phase‑locked standing wave" framing is a faithful description of a Gabor renderer, not a claim of new physics. The physics/AI resonance here is a *representation* echo (complex amplitudes = phasors), not a discovery about nature.
- SURF explores a 2‑plane of the tangent space at a time. It does not give you free access to all 128 dimensions at once; press `N` for a new plane.
- The phase‑transport loophole above is a *direction*, not something this particular model already does well.

**License / data terms.** CelebA is released for **non‑commercial research use** — check the dataset's own terms before you do anything with the outputs. This repo is offered under CC BY‑NC 4.0 to stay consistent with that.

---

## Credits

Built by [Antti Luode](https://github.com/anttiluode) with several AI systems as thinking partners over a couple of long sessions. The interference/eigenface/phase‑transport framing came out of those conversations; the code and the honest ledger are meant to let you check all of it yourself.
