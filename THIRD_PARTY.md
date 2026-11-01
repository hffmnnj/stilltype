# Third-party code and model notices

Stilltype's own app code is MIT (see [LICENSE](LICENSE)). Everything below keeps
its own terms. Nothing in this file relicenses anyone else's work — in
particular, model weights are **not** MIT solely because source code is, and
must not be treated as if they were.

## Recognition: USR 2.0 (2026 release)

| What | Where | From | Terms |
| --- | --- | --- | --- |
| USR 2.0 recognition code | `engine/` | [USR 2.0](https://github.com/ahaliassos/usr2) ([paper](https://arxiv.org/abs/2602.19316)) | MIT, copyright 2026 Alexandros Haliassos |
| ESPnet decoding and model modules (trimmed) | `engine/espnet/` | [ESPnet](https://github.com/espnet/espnet) | Apache 2.0 (per-file headers) |
| Mouth-ROI and preprocessing helpers | `engine/` | Original vendor sources for the pipeline above | Original notices preserved alongside the code |

Local changes to third-party files are kept minimal and noted where they occur.

## Model weights (never in git)

| What | How it arrives | Terms |
| --- | --- | --- |
| USR 2.0 Base+ high-resource full-model checkpoint (`models/usr2/baseplus.pth`, ~1.39 GB / ~1.3 GiB) | Downloaded only after explicit `--accept-license` consent in `engine/install.py` (fetched with `gdown`), or via the app's Models page | Pretrained research/data rights are **separate from the code licence** — check upstream before your intended use |
| Qwen3.5-0.8B text-cleanup GGUF (`models/cleanup/qwen3.5-0.8b-q4_k_m.gguf`, 579,615,840 bytes / ~580 MB) | Separate consent-gated pinned HTTPS download, SHA256-verified before use (`fb044e93939a70469c905781334f5de1e6c8b608ced6cbc8c9249bd4127d9526`); never in git | Apache-2.0 ([Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), created 2026-02-28; GGUF from [bartowski/Qwen_Qwen3.5-0.8B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.5-0.8B-GGUF) at revision `f36b1ea49a332ede8fe5f389bbf5b3575ef71f48`, text cleanup only, no image projector) |
| Qwen3.5-2B text-cleanup GGUF (1,396,198,496 bytes) | Separate consent-gated pinned HTTPS download; exact revision and SHA-256 in [BUILDING.md](docs/BUILDING.md); never in git | Apache-2.0 ([Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B); [GGUF packaging](https://huggingface.co/bartowski/Qwen_Qwen3.5-2B-GGUF), created 2026-03-01) |
| Qwen3.5-4B text-cleanup GGUF (3,013,027,808 bytes) | Separate consent-gated pinned HTTPS download; exact revision and SHA-256 in [BUILDING.md](docs/BUILDING.md); never in git | Apache-2.0 ([Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B); [GGUF packaging](https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF), created 2026-02-27) |

`engine/install.py` verifies the checkpoint SHA256
(`4ab1f3f230aa6a724bcc698db8a17f2f54919136d2885c1a07a6b69247012773`)
and writes `installed.json` as the runtime model marker. Do not reuse licence
restrictions from any other model for this checkpoint; its terms are whatever
upstream states for the USR 2.0 high-resource Base+ row.

The Android `usr2-ctc-int8.onnx` graph is a quantized encoder/CTC export of
that same checkpoint, not newly trained weights or a differently licensed
model. It is generated outside Git and bundled only when explicitly supplied
to the Android build. Exporting or quantizing does not remove upstream model
or training-data obligations; review those terms before distributing an APK
with weights. The copied token and mean-face data retain upstream provenance.

## Android runtime

- **ONNX Runtime Android 1.24.3** —
  [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime), MIT.
- **OpenCV Android 4.11.0** —
  [opencv/opencv](https://github.com/opencv/opencv/tree/4.11.0), Apache-2.0.
- **AndroidX, CameraX, Jetpack Compose and Material 3** — Apache-2.0;
  resolved versions are declared in `android/app/build.gradle.kts`.
- The APK includes ONNX Runtime, OpenCV, YuNet, USR2 and Geist notices in
  `android/app/src/main/assets/licenses/`. Android uses the same pinned
  YuNet helper described below; no model or font is fetched at runtime.

## Runtime pieces fetched at install or runtime

- **uv** (used to install the Python runtime; bundled into release helpers if
  the packager adds it) — [Astral](https://github.com/astral-sh/uv), MIT or Apache 2.0.
- **Python, PyTorch and the engine's other packages** (`torch`, `numpy`,
  `opencv-python-headless`, `six` — see [engine/requirements.txt](engine/requirements.txt)) —
  installed from PyPI/python-build-standalone; each keeps its own licence.
- **YuNet 2026may face-tracking artifact** (~230 kB ONNX, auxiliary
  camera face-box extraction — a helper component, not the 2026 recognition
  release) — OpenCV Zoo dynamic-input export redistributed by Pollen Robotics:
  [model card](https://huggingface.co/pollen-robotics/face_detection_yunet_2026may)
  (revision `2b8e922362946a0db67e861bae0f77826980effd`), MIT (see
  `public/vision/LICENSE.yunet.txt`). The 2026 export shares learned weights
  with the earlier YuNet architecture; it is not newly trained 2026 face
  weights. Prepared by `scripts/prepare-vision.mjs`, which verifies its SHA256
  (`ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0`);
  full provenance in `public/vision/NOTICE.md`.
- **ONNX Runtime Web WASM** (runs the face detector locally, no runtime CDN) —
  copied from the locked `onnxruntime-web` npm dependency by
  `scripts/prepare-vision.mjs`, MIT (see
  `public/vision/LICENSE.onnxruntime.txt`).
- **llama.cpp** (compiled into the app as the text-cleanup inference engine
  via the bindings below) — [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), MIT. Copyright 2023–2026 the ggml authors; [complete notice](src-tauri/licenses/llama.cpp-MIT.txt) is bundled with the app.
- **llama-cpp-2 0.1.160** (maintained safe Rust bindings for `llama.cpp`,
  exact pinned version) — [utilityai/llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs);
  MIT OR Apache-2.0. Distributed here under MIT, copyright Dial AI; [complete notice](src-tauri/licenses/llama-cpp-rs-MIT.txt) is bundled with the app.
- **Qwen3.5 0.8B / 2B / 4B** (optional text-only cleanup weights; Format or
  experimental Intent correction, raw recognition text always kept) —
  [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), Apache-2.0;
  GGUF packaging from
  [bartowski/Qwen_Qwen3.5-0.8B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.5-0.8B-GGUF).

## Fonts

**Geist Sans** (variable, wght 100–900) and **Geist Serif Regular** (static,
weight 400) ship with the app: web `woff2` in `public/fonts/`, Android `ttf`
in `android/app/src/main/res/font/` (`geist_sans_variable.ttf`,
`geist_serif_regular.ttf`). Copyright 2024 The Geist Project Authors
([vercel/geist-font](https://github.com/vercel/geist-font), release v1.9.0).
Licensed under the SIL Open Font License 1.1; the complete notice is in
[public/fonts/OFL-Geist.txt](public/fonts/OFL-Geist.txt) and
`android/app/src/main/assets/licenses/Geist-OFL.txt`. Only the static Regular
serif cut is used — Geist Serif's variable `ELSH` axis build is not shipped,
so no weight range is declared for the serif.

Legacy **Newsreader** variable Latin files remain on disk in `public/fonts/`
but are no longer loaded; candidate for removal in a later cleanup. Copyright
2020 The Newsreader Project Authors; SIL Open Font License 1.1, see
[public/fonts/OFL-Newsreader.txt](public/fonts/OFL-Newsreader.txt).

## Original brand artwork

Stilltype's raster logo and cover artwork in `public/brand/` were generated
with GPT Image 2.5. Desktop and Android icons are resized derivatives of
that original PNG artwork, not hand-coded SVG or Canvas logos. Functional
interface icons and camera guides remain native UI-kit/vector controls.
