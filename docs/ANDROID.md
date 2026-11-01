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
| Speech gate | Desktop motion gate | Capture requires faces in >= 50% of observed frames (`ui/capture/CaptureViewModel.kt`), then motion plus greedy CTC evidence for weak-motion windows (`engine/AndroidRecognizer.kt:activitySpans`) |
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

### Weak-motion regression (0.1.2)

The CTC evidence gate recovers some speech rejected by the old motion gate,
without admitting weak motion alone. Host WER and quiet-control results are in
[RECOGNITION.md](RECOGNITION.md#ctc-activity-gate-improvement); they are not phone
accuracy measurements. Two-minute sessions and CPU/Auto selection remain.

`MobileBenchmarkTest#weakMotionSpeechAndQuiet_useRealRecognizer` exercises the
real recognizer with two app-private fixtures. Transfer
`activity-gate-fixtures.json`, `activity-gate-speech.raw`, and
`activity-gate-quiet.raw` using the same `run-as` method above. Each metadata
entry (`speech`, `quiet`) has `format: "uint8-96x96"` and `timestampsMs` at
40 ms intervals, with 14–200 frames. The corresponding raw file concatenates
unsigned 96×96 grayscale frames. Both fixtures must pass motion 4 and fail
motion 8; the test rejects a fixture that does not exercise that regression.

The API 36 emulator passed speech acceptance, quiet rejection, and a repeated
3,001-frame / 120,000 ms session. Results are written to app-private
`activity-gate-results.json`, not logcat. This proves routing and bounds, not
correct words: the short fixture still had 5 word errors out of 6. The Pixel
9 Pro Fold has not run this test.

### Experimental encoder/decoder benchmark

`ExperimentalFlareBenchmarkTest#exportedGraphs_decodeRealFixtureOnCpu` runs the
research FLARE exports directly through the pinned Android ONNX Runtime. It is
**not** an application model switch, camera test, GPU test or phone accuracy
benchmark. These graphs are not bundled in the APK; the model's CC-BY-NC-4.0
terms still apply. See [RECOGNITION.md](RECOGNITION.md#portable-flare-research-utilities)
for the exporter, trusted-source requirements and separate Python environments.

After exporting, `tools/benchmark_flare.py` prepares a private directory with
`face.f32`, `lip.f32`, `manifest.json` and host results. The manifest carries
graph sizes/hashes, frame count and independently generated host token IDs.
The Android test validates them, decodes without consulting the reference,
then requires exact token equality. Missing fixtures, non-finite values, shape
mismatches and failure to generate EOS cause test failure, not a skipped pass.

Build/install the debug app and instrumentation APK first. Then transfer the
two graphs and three numeric fixtures to the target app's private storage:

```sh
adb shell run-as com.stilltype.android mkdir -p files/experimental-flare
for file in encoder.onnx decoder.onnx; do
  adb shell -T run-as com.stilltype.android tee "files/experimental-flare/$file" \
    < "/path/flare-onnx/$file" > /dev/null
done
for file in face.f32 lip.f32 manifest.json; do
  adb shell -T run-as com.stilltype.android tee "files/experimental-flare/$file" \
    < "/path/android-flare-fixture/$file" > /dev/null
done
adb shell am instrument -w -r \
  -e class 'com.stilltype.android.engine.ExperimentalFlareBenchmarkTest#exportedGraphs_decodeRealFixtureOnCpu' \
  com.stilltype.android.test/androidx.test.runner.AndroidJUnitRunner
```

Use `adb -s SERIAL` when needed. Token IDs are written only to app-private
`experimental-flare/results.json`, not logs. Status key
`stilltypeExperimentalFlare` reports device, provider, timings and sampled PSS.

Coordinator-run API 36 x86_64 emulator results (two virtual cores, 3 GiB guest
RAM, four ORT threads, CPU only) on one real 128-frame / 5.12-second input:

| Stage | Time |
| --- | ---: |
| Encoder session load | 2,471 ms |
| Decoder session load | 756 ms |
| Encoder execution | 5,303 ms |
| Full-prefix greedy decoder | 3,532 ms |
| Complete instrumentation test | 15.10 s |

All 33 generated tokens matched the host reference. Sampled process PSS was
112,868 KiB before loading and 1,216,371 KiB after decoding; this is not peak
memory. Stage timings exclude camera, face processing, fixture transfer and
graph hashing; the complete test includes its validation/setup work. This
demonstrates native graph compatibility, not the user's camera-to-text wait on
a Pixel. Greedy decoding also had worse host WER than the research beam-50
decoder, so the current app model remains unchanged.

### Cached decoder benchmark

Export with `--decoder-mode cached`, then run `tools/benchmark_flare.py` as
described above. Transfer `encoder.onnx`, `cross.onnx` and `decoder-cache.onnx`
from that export and `face.f32`, `lip.f32`, `manifest.json` from the resulting
fixture into app-private `files/experimental-flare-cache/`, using the same
`run-as` method. Run:

```sh
adb shell am instrument -w -r \
  -e class 'com.stilltype.android.engine.ExperimentalFlareBenchmarkTest#cachedGraphs_decodeRealFixtureOnCpu' \
  com.stilltype.android.test/androidx.test.runner.AndroidJUnitRunner
```

The native test checks graph consistency, cache shapes, finite values and
independent token parity. Zero-length initial caches worked with Android ORT
1.24.3. Results and tokens stay in private `experimental-flare-cache/results.json`;
the `stilltypeExperimentalFlareCache` status summary contains no token IDs.

On the same API 36 emulator configuration and 128-frame fixture, the published
cached graphs generated all 33 expected tokens exactly:

| Stage | Time |
| --- | ---: |
| Encoder / cross / decoder session loads | 2,264 / 35 / 549 ms |
| Encoder execution | 4,291 ms |
| Cross projections | 67 ms |
| Complete cached decoding loop, including validation | 1,035 ms |
| Native decoder calls within that loop | 715 ms |
| Complete instrumentation test | 10.80 s |

An earlier cached run took 834 ms for the complete decoding loop. The earlier
full-prefix run took 3,532 ms; these are separate runs, not a controlled phone
benchmark or camera-to-text latency. Sampled process PSS after the published
cached run was 1,175,788 KiB, not peak memory. Camera, face analysis, transfer,
model preparation and graph hashing are outside the stage timings. This is
still an ungated research decoder, not a new app default or a physical Pixel
GPU/NPU measurement.

### Batched beam benchmark

Export with `--decoder-mode beam`, then run `tools/benchmark_flare.py` to create
the private numeric fixture. Transfer `encoder.onnx`, `cross.onnx` and
`decoder-beam.onnx`, plus `face.f32`, `lip.f32` and `manifest.json`, to
`files/experimental-flare-beam/` using the same `run-as` method. Run:

```sh
adb shell am instrument -w -r \
  -e class 'com.stilltype.android.engine.ExperimentalFlareBeamBenchmarkTest#batchedGraphs_decodeRealFixtureOnCpu' \
  com.stilltype.android.test/androidx.test.runner.AndroidJUnitRunner
```

The test validates graph hashes and exact input/output geometry, starts with
zero-length self caches, expands from one row to five and reorders caches by
selected parent. Native results are copied before their owners close. Complete
tokens must match the independent reference; the status key
`stilltypeExperimentalFlareBeam` contains timings but no tokens. The private
`experimental-flare-beam/results.json` retains the numeric output.

The coordinator built and ran this test on the API 36 x86_64 emulator, two
virtual cores and 3 GiB guest RAM. All 29 generated tokens matched on a real
129-frame input:

| Stage | Time |
| --- | ---: |
| Encoder / cross / decoder session loads | 4,899 / 70 / 1,581 ms |
| Encoder execution | 7,285 ms |
| Cross projections | 53 ms |
| Complete beam-5 loop, including validation/copies/search | 1,894 ms |
| Native decoder calls within that loop | 911 ms |
| Complete instrumentation test | 23.16 s |

Sampled PSS after decoding was 1,254,360 KiB, not peak memory. A host CPU
evaluation was running concurrently, so comparison with earlier greedy timings
is not a controlled speed comparison. The stage timings exclude camera,
preprocessing, model preparation and graph hashing. This proves native cached
beam execution and token parity, **not** camera-to-text latency, physical Pixel
acceleration or beam-50 accuracy. The application and delivered APK retain their
existing model.

### Compact graph compatibility

`tools/pack_flare.py` can reduce the three beam-5 graph files from 666,988,252
to 338,099,826 bytes while retaining FP32 computation through weight Casts.
It does not add a native dependency or change the installed application's model.
Use the packed directory with `tools/benchmark_flare.py`, then transfer its
graphs and newly generated numeric fixture to `files/experimental-flare-beam/`
and run the same batched-beam test above.

The coordinator ran the final published packer and the actual API 36 emulator
test. All 29 tokens matched the independent reference on the 129-frame fixture;
the test completed in 11.52 seconds. Encoder execution was 4,796 ms and the
complete beam loop 1,856 ms. Those are compatibility-run stage timings, not a
controlled speed comparison or camera-to-text latency. Sampled process PSS was
1,290,081 KiB after decoding: **no memory reduction is claimed**. The emulator
was stopped after the check. Physical Pixel and camera-path compatibility remain
unverified, and the delivered APK still uses the existing model.

## Compute notes

`AUTO` requests NNAPI with `CPU_DISABLED`, excluding NNAPI's reference CPU
device while retaining ONNX Runtime CPU kernels for unsupported partitions.
The first recognition inference in each AUTO session is profiled locally.
Executed provider events produce `NNAPI execution`, `NNAPI execution + CPU`,
or `CPU`. If the trace is unavailable or incomplete, the result says
`NNAPI requested; execution unverified`. This describes observed ORT execution
providers, **not proof of GPU versus NPU use or a measured speedup**. It is a
sample of the first inference, not a hardware trace for every later window.
The old 0.1.0 label `NNAPI requested, CPU fallback` was unconditional after
session creation and did not establish CPU-only execution.

Profiling contains operator/provider metadata, not pixels or transcript text;
the temporary trace is removed after reading. Logcat tag `StilltypeCompute`
contains provider event counts only. Explicit `CPU` mode remains CPU-only and
does not enable profiling. Analysis and recognition use four intra-op threads.

The INT8 graph includes a 3D convolution and dynamically quantized matrix
operators. The [ORT NNAPI supported-op table](https://onnxruntime.ai/docs/execution-providers/NNAPI-ExecutionProvider.html)
supports only 2D convolution and does not list `MatMulInteger` or
`DynamicQuantizeLinear`; those operators cannot be assumed to offload. Other
supported subgraphs may still use NNAPI. Android also
[deprecated NNAPI in Android 15](https://developer.android.com/ndk/guides/neuralnetworks/migration-guide).
Enabling a provider or changing a label does not make the entire model GPU-compatible.

The real-model AUTO diagnostic on the API 36 x86_64 emulator observed 50 NNAPI
partition events and 1,228 ORT CPU node events on a 200-frame fixture, taking
3,196 ms for inference. These are event counts, not percentages of work, and
the emulator is not evidence of physical Pixel acceleration. Reproduce with:

```sh
adb shell am instrument -w -r \
  -e class 'com.stilltype.android.engine.MobileBenchmarkTest#autoRecognizer_reportsExecutionEvidenceForRealModel' \
  com.stilltype.android.test/androidx.test.runner.AndroidJUnitRunner
adb logcat -d -s StilltypeCompute:I StilltypeBenchmark:I
```

Use the same bundled app/test APK and staged public fixture described above.
Physical Pixel graph placement remains unverified until that device runs it.

New recording sessions default to **120 seconds**, configurable down
to shorter limits. Up to 3,001 mouth crops are buffered (about 27 MiB of
pixel data); sequential inference windows stay within 8 seconds / 200 model
frames. Later windows are joined in order rather than truncated. Empty,
invalid and still-mouth recordings do not produce invented text. Long
recordings can take time to process, and chunk boundaries may affect words;
this is not a streaming-accuracy guarantee.
