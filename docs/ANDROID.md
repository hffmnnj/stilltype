# Stilltype for Android (companion)

Native Kotlin/Compose Material 3 companion to the desktop app. It mouths to
text fully on-device: camera frames are aligned locally, then decoded by a
CTC-only export of the desktop recognition checkpoint. No account, no network,
no telemetry.

> Status: no physical phone is attached. Host export/benchmark numbers below
> are fixture smoke and latency measurements from a desktop host, not word
> error rate and not phone measurements. The real Android runtime was also
> exercised on an API 36 x86_64 emulator; those results are labelled below.

See [BUILDING.md](BUILDING.md) for the desktop build; this file covers only
the Android companion.

## Desktop versus Android

| | Desktop | Android |
| --- | --- | --- |
| Shell | Tauri 2 Rust shell + React frontend (`src/`, `src-tauri/`) | Native Kotlin, Compose Material 3 (`android/app/src/main/java/com/stilltype/android/`) |
| Recognition | Private Python 3.12 worker over stdio (`engine/`), beam search, CUDA/MPS/CPU | On-device ONNX Runtime (`onnxruntime-android:1.24.3`), greedy CTC argmax in bounded 200-frame inference windows |
| Face tracking | JS/WASM YuNet in the browser view | Same pinned `yunet-2026may.onnx` via ORT + OpenCV (`engine/FaceAnalyzer.kt`), per-frame affine warp — no temporal smoothing (desktop smooths landmarks over a centered 13-frame window in `engine/inference.py:align_mouths`) |
| Speech gate | Desktop motion gate | Capture requires faces in >= 50% of observed frames (`ui/capture/CaptureViewModel.kt`), plus the shared motion-based speech span (`engine/AndroidRecognizer.kt:speechSpan`) |
| Compute setting | `settings.inferenceDevice` (`auto` \| `gpu` \| `cpu`) | `ComputeMode` (`AUTO` \| `CPU`) in the shared contract (`core/Contracts.kt`) |
| Storage | Tauri app-data directory | Local DataStore `stilltype/state.json`, FileProvider export from `cacheDir/exports` |

Shared contract: `android/app/src/main/java/com/stilltype/android/core/Contracts.kt`
defines settings, history, vocabulary, the mobile model manifest
(`usr2-baseplus-2026-ctc-int8`, 25 fps, 88px crops, 200 frames per model
window, blank 0, EOS 1048), and the `VisualRecognizer` interface. Recording
sessions support **two minutes**, independently of the internal window size.

## Model provenance

The phone model is a CTC-only graph derived from the existing trained
weights — no new training. Source is the USR2 Base+ high-resource release,
original 2026 checkpoint (SHA256
`4ab1f3f230aa6a724bcc698db8a17f2f54919136d2885c1a07a6b69247012773`),
exported with `tools/export_android.py` and exercised with
`tools/benchmark_mobile.py`. INT8 quality is not equivalent to desktop beam
decoding: on the host fixture, INT8 frame argmax agreement was 96.5% and the
decoded text changed.

Why this checkpoint: among the sources supplied for this task, no smaller
proven 2026 phone runtime with measured on-phone memory/latency was
established (a narrower statement than claiming none exists anywhere).
Reference points, kept short: the USR2 paper reports 24.8% VSR WER on LRS3
for Base+ high-resource with joint decoding (beam 40, CTC weight 0.1) on
spoken videos —
[repo](https://github.com/ahaliassos/usr2),
[paper](https://arxiv.org/html/2602.19316) — which does not transfer to this
mobile export or to silent mouthing; its 40x figure is H200 decoding-only,
not phone end-to-end. Older options are off-task: LiteVSR is 2023
([paper](https://arxiv.org/html/2312.09727v1)), MobiVSR is 2019 isolated
words ([paper](https://arxiv.org/abs/1905.03968)), not 2026 sentence
recognition. VALLR is original 2025, revised 2026
([paper](https://arxiv.org/html/2503.21408v2)); its official repo states
CC BY-NC 4.0 for its work with weights terms/date unresolved
([repo](https://github.com/MarshallT-99/VALLR)) — code licensing is not
derived here from an arXiv paper license.

## Preparing the mobile model (host)

Needs the host Python 3.12 runtime and the pinned checkpoint from the
desktop Models flow (`python3 engine/install.py --models /path/to/models
--accept-license`; consent refusal refuses the download — see
[BUILDING.md](BUILDING.md)). The export writes outside the source tree; never
commit weights.

```sh
# From the repository root. --checkpoint is the pinned baseplus .pth,
# --output is a directory outside the repo.
python -m pip install -r tools/mobile-requirements.txt
python tools/export_android.py --checkpoint /path/baseplus.pth --output /absolute/path/mobile-model
python tools/export_android.py --checkpoint /path/baseplus.pth --output /absolute/path/mobile-model --stage quantize
python tools/export_android.py --checkpoint /path/baseplus.pth --output /absolute/path/mobile-model --stage manifest
```

The three stages write `usr2-ctc-fp32.onnx`, then `usr2-ctc-int8.onnx`,
then `tokens.json`, `mean-face.json`, a copy of
`public/vision/face_detection_yunet_2026may.onnx`, and `manifest.json`.
`manifest()` refuses token lists that are not exactly 1049 entries.

Host comparison on a real aligned clip (fixture smoke/latency, not accuracy):

```sh
python tools/benchmark_mobile.py prepare --output /absolute/path/mobile-model --clip /path/to/clip-directory
python tools/benchmark_mobile.py torch --output /absolute/path/mobile-model --checkpoint /path/baseplus.pth
python tools/benchmark_mobile.py ort --output /absolute/path/mobile-model
```

Run the stages in order, in separate processes (the script docstring says so
to avoid keeping model copies alive). `prepare` needs `--clip` (a recorded
clip directory readable by `engine.inference.load_clip_records`);
`torch` needs `--checkpoint`; `ort` compares the staged FP32 and INT8
exports against the saved PyTorch logits and fails on FP32 parity drift.

Known measured host results (desktop CPU, four ORT threads; smoke fixture,
not WER, not phone):

| Artifact | Bytes |
| --- | --- |
| `usr2-ctc-fp32.onnx` | 449,293,501 |
| `usr2-ctc-int8.onnx` | 169,919,869 |

| Check (dynamic T=25 and T=200) | Result |
| --- | --- |
| FP32 vs PyTorch argmax text | Identical |
| FP32 maximum logit difference | 0.00117 |
| Four-thread host CPU warm T=200, coordinator rerun | FP32 702 ms vs INT8 542 ms |
| INT8 frame argmax agreement | 96.5%, decoded text changed |

## Building the app

Prerequisites: JDK 17 and the Android SDK (Gradle wrapper pins Gradle
8.13; `compileSdk`/`targetSdk` 36, `minSdk` 29; see
`android/app/build.gradle.kts` and `android/settings.gradle.kts`).

```sh
cd android
./gradlew assembleDebug     # without model assets: builds, model reports "not bundled"
./gradlew assembleRelease
```

A full build bundles generated assets by pointing at the export directory:

```sh
./gradlew assembleDebug -PmobileModelDir=/absolute/path/mobile-model
# or: export STILLTYPE_MOBILE_MODEL_DIR=/absolute/path/mobile-model
```

`stageBundledModels` copies exactly `manifest.json`, `usr2-ctc-int8.onnx`,
`tokens.json`, `yunet-2026may.onnx`, `mean-face.json` into the APK. When the
property/env var is unset, the APK builds without model assets; a supplied
nonexistent directory fails the build rather than silently dropping weights.
The data layer truthfully reports the
model as not bundled (`data/BundledModelRepository.kt`). The manifest is
validated on install (model id, 2026 release, `ctc-int8` kind, checkpoint
SHA, frame geometry, token ids, file sizes/hashes) and installation is
explicit — `prepareBundled()` runs only after UI consent, never on launch,
and never opens the camera. Debug APKs keep `arm64-v8a` + `x86_64` ABIs
(ORT/OpenCV ship both; duplicate `libc++_shared.so` is deduplicated).

## Permissions and data

Manifest (`app/src/main/AndroidManifest.xml`): **camera permission only**.
No microphone, no Internet, no Accessibility service, no overlay permission,
no background camera. The tile service declares only
`BIND_QUICK_SETTINGS_TILE`.

- History, vocabulary and settings live in app-private DataStore
  (`filesDir/stilltype/state.json`); retention 7–365 days prunes on load
  and on every write (`data/LocalAppRepository.kt`).
- Exports are local temporary files under `cacheDir/exports/`, shared
  through FileProvider authority `com.stilltype.android.files`
  (`res/xml/file_paths.xml`); CSV cells guard spreadsheet formula prefixes.
- The camera binds only while a capture owner is active
  (`ui/capture/CameraStage.kt`); a single `CaptureViewModel` owns sessions
  by token so late frames or results after cancel/pause are dropped without
  saving or copying.
- Fonts (Geist Sans variable, Geist Serif) and the launcher icon ship in
  `res/`; the app theme is defined in `ui/theme/StilltypeTheme.kt`.

## Quick capture entry points

- **Quick Settings tile** (`platform/StilltypeTileService.kt`): opens the
  visible capture activity over the previous app. Listening only reads model
  status for the tile subtitle; clicks never start the camera, recognizer,
  or model preparation. Adding from the app uses
  `QuickAccess.requestTile`, which needs API 33+; older devices get
  `UNSUPPORTED` so the UI shows manual add guidance instead of pretending.
  Manual path: swipe into Quick Settings, tap Edit, drag the Stilltype tile
  in. Opening the sheet never turns on the camera: tap Record explicitly.
- **Pinned shortcut** (`platform/QuickAccess.requestPinnedShortcut`): a
  "Dictate" launcher shortcut opening capture. Returns the system's
  acceptance, not the user's choice; `false` means pinning is unsupported.
  Call both only from a visible user action, never automatically on launch.

## Android runtime benchmark

Measured on the **Google API 36 x86_64 emulator**, two virtual CPU cores,
3 GiB guest RAM, CPU provider and four ORT intra-op threads:

| Measurement | Result |
| --- | --- |
| Fixture | 200 frames / approximately 8 seconds, public demo |
| INT8 model session load | 633 ms |
| Warm inference | 2,771 ms |
| Process PSS before / after inference | 112,452 / 722,843 KiB |

This is real Android ONNX execution, not a host Python timing. PSS is a
process sample, **not peak memory**, and includes the instrumentation process;
camera capture, face analysis, model preparation and the full UI journey are
not part of that inference timing. These numbers do not predict physical
phone speed, battery use or accuracy.

The full recognizer also passed a **120,000 ms / 3,001-frame** repeated
public-video timeline through `LongSessionTest`: 38,509 ms aggregate native
inference and 40,323 ms for the recognition pipeline, with sampled PSS
701,726 KiB. That pipeline timing excludes camera capture and model
preparation. It proves the entire long-session path executes, not the
accuracy of natural two-minute speech. A separate real camera session stayed
open for two minutes, stopped automatically, and correctly rejected the
emulator scene as having no face.

`app/src/androidTest/java/com/stilltype/android/engine/MobileBenchmarkTest.kt`
is a real-model CPU smoke test, not an accuracy test: it prepares the
bundled model, runs one warm INT8 session over the `benchmark-input.f32`
fixture (shape `[1, T, 88, 88]`, `float32-le`, T within 1–200), asserts
finite `[1, T, 1049]` CTC logits, and logs a one-line summary
(`stilltypeBenchmark`: device, SDK, ABI, provider, frames, load/warm ms,
PSS) to logcat and instrumentation status.

```sh
cd android
# Build/install the debug app and androidTest APK first, then supply the fixture.
adb shell run-as com.stilltype.android mkdir -p files
adb shell -T run-as com.stilltype.android tee files/benchmark-input.f32 < /absolute/path/mobile-model/benchmark-input.f32 > /dev/null
adb shell -T run-as com.stilltype.android tee files/benchmark-input.json < /absolute/path/mobile-model/benchmark-input.json > /dev/null
adb shell am instrument -w -r -e class com.stilltype.android.engine.MobileBenchmarkTest#bundledModelRunsRealFixtureWithFiniteCtcLogits com.stilltype.android.test/androidx.test.runner.AndroidJUnitRunner
# Full two-minute recognition pipeline over repeated public-video mouth crops:
adb shell am instrument -w -r -e class com.stilltype.android.engine.LongSessionTest com.stilltype.android.test/androidx.test.runner.AndroidJUnitRunner
adb logcat -s StilltypeBenchmark
```

Use `adb -s SERIAL` when more than one device is connected. The debug-only
`run-as` transfer keeps the public fixture out of the repository and APK.

## Compute notes

`AUTO` requests NNAPI with CPU fallback (`CPU (NNAPI unavailable)` when the
request fails); the provider string this produces is a request outcome, not
proof of GPU offload. Analysis and recognition sessions use four intra-op
threads. New recording sessions default to **120 seconds**, configurable down
to shorter limits. Up to 3,001 mouth crops are buffered (about 27 MiB of
pixel data); sequential inference windows stay within 8 seconds / 200 model
frames. Later windows are joined in order rather than truncated. Empty,
invalid and still-mouth recordings do not produce invented text. Long
recordings can take time to process, and chunk boundaries may affect words;
this is not a streaming-accuracy guarantee.
