# Building Stilltype

Stilltype is a Tauri 2 app: a Rust shell (`src-tauri/`) hosting a React +
TypeScript frontend (`src/`) and managing a private Python 3.12 recognition
worker (`engine/`) over stdio. Build installers **on the host you ship for** —
Linux cannot cross-produce macOS DMG or Windows NSIS installers.

> Verified on Linux: frontend production build, Rust format/check and thirteen
> tests (twelve passed, one ignored real-model test), 32 Python tests, a Debian
> package build, and native app launch.
> A real 8.28-second public video exercised CUDA recognition in 4.1 seconds
> and CPU recognition in 20.9 seconds at beam 5. The JavaScript/WASM face
> detector also ran against the public video. The optional cleanup runtime
> produced identical punctuation on CPU and GPU. These are smoke checks, not
> accuracy benchmarks. Windows/macOS execution and the complete live webcam
> to paste workflow remain unverified; isolated browser camera access was denied.

## Prerequisites (all platforms)

| Tool | Version | Notes |
| --- | --- | --- |
| Node.js | 20 LTS or newer | `npm ci` for a reproducible install |
| Rust stable | via [rustup](https://rustup.rs) | Needed for `npm run desktop` and `npm run desktop:build` |
| uv | on PATH ([install](https://docs.astral.sh/uv/getting-started/installation/)) | Needed for source installs: the app uses it to set up the Python runtime |
| Python | 3.12 | Managed via uv; inference needs `torch`, `numpy`, `opencv-headless`, `six` per [engine/requirements.txt](../engine/requirements.txt) |
| GPU (optional, preferred) | CUDA on Linux/Windows, MPS on macOS | Auto mode prefers CUDA, then MPS, and falls back to CPU |
| CMake + C++ compiler | Needed for the bundled text-cleanup runtime (see below) |

A packaging helper may bundle `uv` with release artifacts later; for source
builds, install `uv` yourself first — it is required on PATH and is not
fetched automatically.

GPU wheels come from the standard PyTorch distribution; CPU-only machines use
the CPU fallback automatically. `settings.inferenceDevice` (`auto` | `gpu` |
`cpu`) selects auto (CUDA → MPS → CPU), explicit GPU-only, or CPU. The
default beam size is 5 (valid range 1–60): beam 30 exceeded the 6 GB GPU
memory on the tested CUDA host, while beam 5 transcribed an 8.28 s demo clip
in 4117 ms.

New settings default to a **120-second recording limit** on every platform;
older saved shorter choices remain valid. Desktop recognition processes long
recordings in sequential timestamp-based windows no longer than eight seconds,
then joins their text in order. The inference window is not the recording
limit. A real 120-second repeated public-video fixture (3,001 frames) ran on
CUDA in 45.048 seconds of inference, 54.59 seconds including setup/preprocessing.
This is a runtime smoke check, not natural-speech accuracy. CPU processing can
take several minutes; the private worker's response timeout is 900 seconds.

## Platform dependencies

### Linux (Debian/Ubuntu)

```sh
sudo apt install build-essential curl wget file \
  libssl-dev libgtk-3-dev libayatana-appindicator3-dev librsvg2-dev \
  libwebkit2gtk-4.1-dev
```

### Linux (Arch)

```sh
sudo pacman -S base-devel curl openssl gtk3 libappindicator-gtk3 \
  librsvg webkit2gtk-4.1
```

> **Wayland note:** global shortcuts and simulated paste may be limited under
> Wayland compositors. Clipboard copy always works — use it as the fallback
> when auto-paste is unavailable.

### Windows

- Microsoft Visual Studio C++ build tools (MSVC compiler + Windows SDK).
- WebView2 — preinstalled on Windows 10 1803+ and Windows 11; otherwise
  install the evergreen runtime from Microsoft.
- CUDA-capable GPU recommended for inference; CPU mode works where no GPU is
  available. Build and test on a Windows host.

### macOS

- Xcode command-line tools: `xcode-select --install`.
- The app prompts for **camera** and **accessibility** permissions on first
  use (camera for capture, accessibility for paste-by-keystroke). Deny either
  and the corresponding feature reports itself unavailable instead of failing
  silently.
- MPS is used for GPU inference where available, with CPU fallback. Build and
  test on a Mac.

## Native overlay and dictation shortcuts

The floating dictation capsule is a native always-on-top window. It starts
hidden and appears without activating the main window when the dictation
shortcut fires or tray Dictate is used. It reflects opening, recording,
processing, result and error states with Stop/Cancel/Dismiss plus Copy/Open
where they apply. A Preview overlay action opens the ready/blocked surface
without starting the camera. Capture has a single owner, so the overlay never
starts a second session.

- **Fn is always hold-to-dictate:** press while mouthing, release to
  transcribe.
- **macOS:** a native listen-only tap (flagsChanged/keycode 63) supports Fn
  and may require **Input Monitoring** permission, which reports status
  without prompting — restart after permission changes. Synthetic paste may
  still need Accessibility separately (see above). macOS execution is not
  verified from this Linux host. The transparent non-activating overlay uses
  a private API, so App Store distribution is not claimed.
- **Windows/Linux fallback:** most laptop Fn keys are handled by firmware and
  cannot be registered as a universal desktop hotkey. Use the existing
  configurable Ctrl/Cmd+Shift+Space shortcut, which retains Hold or Toggle
  behavior; the app reports Fn as unsupported there. There is no raw keyboard
  device access, no extra driver and no root setup.
- **Wayland:** global shortcuts, simulated paste and overlay positioning may
  be restricted by the compositor. Clipboard copy always works as the
  fallback.

## Build and run

```sh
npm ci
npm run typecheck     # tsc -b
npm run dev           # labeled browser preview; native recognition unavailable
npm run desktop       # full Tauri app
npm run desktop:build # platform installer for this host
```

The Android companion builds separately; see [ANDROID.md](ANDROID.md).

`npm run dev` and `npm run build` automatically run `prepare:vision` first,
which verifies the pinned YuNet 2026may face-tracking artifact by SHA256
(downloading it only when missing or mismatched, ~230 kB) and copies the
matching WebAssembly runtime from the locked `onnxruntime-web` npm
dependency. Network access is therefore needed only on first preparation;
the running app fetches nothing from a CDN. Full provenance lives in
[`public/vision/NOTICE.md`](../public/vision/NOTICE.md).

Install the recognition checkpoint from the app's **Models** page (requires
`uv` on PATH), or from the command line:

```sh
python3 engine/install.py --models /path/to/models --accept-license
```

Refusing `--accept-license` refuses the download. The installer fetches the
USR 2.0 Base+ high-resource checkpoint (~1.39 GB) with `gdown` into
`models/usr2/baseplus.pth`, verifies its SHA256
(`4ab1f3f230aa6a724bcc698db8a17f2f54919136d2885c1a07a6b69247012773`),
and writes `installed.json` as the runtime model marker. Checkpoint weights
are never committed to git, and their research/data rights are separate from
the MIT code licence — check upstream before your intended use (see
[THIRD_PARTY.md](../THIRD_PARTY.md)).

The worker speaks a private stdio protocol:

```sh
python -u engine/worker.py --models /path/to/models
```

## Bundled text-cleanup runtime (no external server)

Optional cleanup is compiled into the Rust application using exactly
`llama-cpp-2 =0.1.160` with default features off (`src-tauri/Cargo.toml`):
the `vulkan` feature on Linux/Windows, the `metal` feature on macOS — from <https://github.com/utilityai/llama-cpp-rs>
(maintained safe Rust bindings for MIT `llama.cpp`). There is no separate
cleanup install, process, model tag, endpoint, listening socket or inference
HTTP request — no external server is required.

- **Acceleration:** Vulkan on Linux/Windows, Metal on macOS, with a runtime
  CPU alternative and GPU-preferred Auto, over pthreads with no OpenMP
  requirement. GPU execution is not universal —
  where acceleration is unavailable the runtime falls back to CPU.
- **Extra build prerequisites:** a C/C++ compiler and CMake on all
  platforms, Clang/libclang for bindgen, plus Linux Vulkan headers/loader,
  SPIRV-Headers and `glslc` (or the Windows Vulkan SDK), and Xcode/Metal
  tooling on macOS. Bound both build-parallelism knobs to 4:
  `CMAKE_BUILD_PARALLEL_LEVEL=4` and `CARGO_BUILD_JOBS=4`.
- **The text model** is Qwen3.5-0.8B (created 2026-02-28, Apache-2.0; see
  <https://huggingface.co/Qwen/Qwen3.5-0.8B>). Stilltype uses the
  `bartowski/Qwen_Qwen3.5-0.8B-GGUF` GGUF at revision
  `f36b1ea49a332ede8fe5f389bbf5b3575ef71f48`, file
  `Qwen_Qwen3.5-0.8B-Q4_K_M.gguf` (579,615,840 bytes, ~580 MB; SHA256
  `fb044e93939a70469c905781334f5de1e6c8b608ced6cbc8c9249bd4127d9526`).
  Source: <https://huggingface.co/bartowski/Qwen_Qwen3.5-0.8B-GGUF>.
  The library ships with the app; the weights are a separate consent-gated,
  pinned HTTPS download verified by SHA256 before use and never committed to
  git. They land at `models/cleanup/qwen3.5-0.8b-q4_k_m.gguf` under the
  app-data directory. Text cleanup only — no image projector is downloaded.
- **Behavior:** cleanup stays off by default, preserves the raw recognition
  output, and only tidies punctuation/capitalization. If the runtime is
  unavailable, or if its output would change words, the raw recognition text
  is retained with a warning. No semantic look-alike-word correction is
  promised.

## Signing and releases

There are no signing certificates, no notarization and **no prebuilt releases
yet**. Unsigned local builds will trigger the usual OS warnings (Gatekeeper on
macOS, SmartScreen on Windows); that is expected until a release process with
proper signing exists. Do not promise signed artifacts that have not happened.

## App data and uninstall

The app identifier is `com.stilltype.desktop`. Settings, history, vocabulary and
models live under the Tauri app-data directory:

| OS | Location |
| --- | --- |
| Linux | `~/.local/share/com.stilltype.desktop` (or `$XDG_DATA_HOME/com.stilltype.desktop`) |
| macOS | `~/Library/Application Support/com.stilltype.desktop` |
| Windows | `%APPDATA%\com.stilltype.desktop` |

Capture clips are ephemeral and deleted after transcription or cancel; only
history you keep (subject to your retention setting) persists.

To fully uninstall: remove the app via the OS-standard method, then delete the
directory above. No registry keys, launch agents or background services are
installed, so nothing else needs cleaning.
