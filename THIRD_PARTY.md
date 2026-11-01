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

`engine/install.py` verifies the checkpoint SHA256
(`4ab1f3f230aa6a724bcc698db8a17f2f54919136d2885c1a07a6b69247012773`)
and writes `installed.json` as the runtime model marker. Do not reuse licence
restrictions from any other model for this checkpoint; its terms are whatever
upstream states for the USR 2.0 high-resource Base+ row.

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
- **Ollama** (optional, user-installed, loopback only) — see
  [Ollama's releases](https://github.com/ollama/ollama); subject to its own terms.

## Fonts

**Newsreader** variable Latin fonts (normal and italic) are bundled in
`public/fonts/`. Copyright 2020 The Newsreader Project Authors. They are
licensed under the SIL Open Font License 1.1; the complete notice is in
[public/fonts/OFL-Newsreader.txt](public/fonts/OFL-Newsreader.txt).
