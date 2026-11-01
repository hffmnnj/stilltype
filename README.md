<p align="center">
  <img src="public/brand/stilltype-cover-generated.png" alt="Original generated Stilltype brand artwork" width="1000" />
</p>

<h1 align="center">Stilltype</h1>

<p align="center">Silent on-camera dictation that stays on your device.<br />Mouth it&ensp;·&ensp;read it&ensp;·&ensp;paste it.</p>

<p align="center">
  <a href="docs/BUILDING.md">Desktop build</a> ·
  <a href="docs/ANDROID.md">Android</a> ·
  <a href="CONTRIBUTING.md">Contributing</a> ·
  <a href="SECURITY.md">Security</a> ·
  <a href="THIRD_PARTY.md">Third-party notices</a>
</p>

---

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/screenshots/desktop-dark.png" />
  <img src="docs/screenshots/desktop-light.png" alt="Stilltype Dictate page with camera stage, readiness status and transcript" />
</picture>

<p align="center"><sub>Dictate on desktop, light and dark. Screenshots show the real app UI — not a model accuracy claim.</sub></p>

Stilltype reads silently mouthed words from your camera and turns them into editable text. No microphone, no accounts, no hosted inference, no telemetry. The desktop app pairs a Tauri&nbsp;2 / Rust / React shell with a private Python stdio visual-inference worker and compiled-in text cleanup; the Android companion is native Kotlin&nbsp;/ Compose&nbsp;Material&nbsp;3 with ONNX&nbsp;Runtime and OpenCV — no Python on the phone.

<p align="center">
  <img src="docs/screenshots/android-dictate.png" alt="Stilltype Android Dictate screen" width="230" />
  &ensp;
  <img src="docs/screenshots/android-capture.png" alt="Stilltype Android compact capture activity" width="230" />
  &ensp;
  <img src="docs/screenshots/android-settings.png" alt="Stilltype Android Settings screen" width="230" />
</p>

<p align="center"><sub>Android Dictate, compact capture, and Settings — real app states, shown at phone scale.</sub></p>

<details>
  <summary>Quick Settings shortcut and Android dark mode</summary>
  <p align="center">
    <img src="docs/screenshots/android-quick-settings.png" alt="Stilltype tile in Android Quick Settings" width="260" />
    <img src="docs/screenshots/android-dictate-dark.png" alt="Stilltype Android Dictate in dark mode" width="260" />
  </p>
</details>

## What you get

- **Two-minute dictation.** Record up to two minutes on every platform, with camera guidance, elapsed time and a configurable limit. Bounded inference windows keep long recordings manageable instead of discarding everything after eight seconds.
- **Native shortcut overlay.** A compact always-on-top capsule that follows dictation without pulling you out of your current app — opening, recording, processing, result and error states with Stop / Cancel / Dismiss plus Copy and Open where they apply.
- **Editable history.** Searchable, date-grouped dictations with raw and cleaned text, favorites, copy and delete. Nothing is ever seeded.
- **Vocabulary.** Case-insensitive word and phrase replacements with canonical spelling and capitalization.
- **Model consent and status.** Desktop model downloads and Android's bundled-model preparation require explicit consent and report real readiness.
- **Export and sharing.** History exports to TXT, JSON or CSV on desktop; local share and clipboard on Android. You deliberately share or copy every export.
- **Quiet controls.** Light, dark and system themes, retention window, inference-device selection (Auto / GPU / CPU), beam size, camera-preview preference — no telemetry anywhere.

```
camera → face + mouth alignment → visual recognition → text → optional local cleanup → copy / paste / export
```

The cleanup pass only tidies punctuation and capitalization, preserves your wording, and always keeps the raw text. There is no inference server to call.

## Quickstart

### Desktop

Prerequisites: Node 20+, Rust stable, `uv` on PATH. Per-OS packages and GPU notes live in [docs/BUILDING.md](docs/BUILDING.md).

```sh
npm ci
npm run desktop   # full Tauri app: camera, shortcut, local inference
```

Then open the **Models** page and consent to the recognition checkpoint: the pinned 2026 USR&nbsp;2 Base+ weights (~1.39&nbsp;GB), GPU (CUDA&nbsp;/ MPS) preferred with a real CPU mode. The optional cleanup model is a separate consent-gated download: compiled-in `llama.cpp` through `llama-cpp-2` 0.1.160 with a 2026 Q4 text model (~580&nbsp;MB). Build an installer on the host you ship for with `npm run desktop:build`.

### Android

The full flow, including the phone-model export, is in [docs/ANDROID.md](docs/ANDROID.md).

```sh
cd android
./gradlew :app:assembleDebug -PmobileModelDir=/absolute/path/to/export
```

The export derives a CTC-only mobile graph (~170&nbsp;MB INT8) from the same 2026 weights; weights are never committed to git. Built without the property, the APK honestly reports the model as not bundled. On the phone, a Quick&nbsp;Settings tile and a pinned Dictate shortcut open a small visible capture activity — no Accessibility or overlay permission, no background camera. The manifest requests camera only: no microphone, and the APK declares no Internet permission.

## Tested status

| Platform | State |
| --- | --- |
| Linux desktop | Builds, launches, and transcribes — an 8.28&nbsp;s public clip decoded at beam&nbsp;5 in 4.117&nbsp;s (CUDA) and 20.904&nbsp;s (CPU). Runtime smoke only, not an accuracy benchmark. Compiled-in cleanup returned identical punctuation on CPU and Vulkan. |
| Android | Native APK builds and runs. Real API&nbsp;36 x86_64 emulator inference: 633&nbsp;ms load, 2.77&nbsp;s warm for 200 frames; sampled process PSS about 706&nbsp;MiB. **Not physical-phone measurements or accuracy.** INT8 is 170&nbsp;MB but changed the fixture text versus FP32; [details and reproducible benchmark](docs/ANDROID.md). |
| macOS / Windows | Native-host execution still needed. macOS Fn is hold-to-dictate and may need Input&nbsp;Monitoring (restart after granting); most Windows&nbsp;/ Linux firmware reserves Fn, so Ctrl&nbsp;/ Cmd+Shift+Space is the fallback. Wayland may further limit global shortcuts, paste and overlay position — clipboard copy always works. |

<img src="docs/screenshots/overlay-light.png" alt="Stilltype native dictation overlay" width="360" />

Desktop beam-search quality does not transfer to CTC-only mobile decoding, and per-frame alignment differs — so paper figures are not repeated here as product accuracy or phone speed. History is stored as local plaintext; retention controls prune it but do not encrypt it.

The complete two-minute path was exercised too: 3,001 frames through the real
desktop recognizer and the Android recognizer, plus an actual Android camera
session that stopped at 2:00. The model tests repeat public-video crops;
they establish execution, not natural-speech accuracy.

## Docs and rights

- [docs/BUILDING.md](docs/BUILDING.md) — desktop prerequisites, GPU/CPU, data paths, uninstall.
- [docs/ANDROID.md](docs/ANDROID.md) — phone export, build variants, permissions, benchmark.
- [docs/RECOGNITION.md](docs/RECOGNITION.md) — raw word-error measurement, fixed evaluation sets and accuracy limits.
- [SECURITY.md](SECURITY.md) — what stays local, what is ephemeral, how to report issues.
- Recognition: [USR&nbsp;2.0 repository](https://github.com/ahaliassos/usr2) · [paper](https://arxiv.org/abs/2602.19316). Code is MIT; the pretrained checkpoint carries separate research&nbsp;/ data rights — check upstream before your intended use (see [THIRD_PARTY.md](THIRD_PARTY.md)).
- App code is MIT ([LICENSE](LICENSE)). No signing, no public prebuilt releases yet — build from source for now.
