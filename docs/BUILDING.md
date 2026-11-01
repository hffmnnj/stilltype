# Building Stilltype

Stilltype is a Tauri 2 app: a Rust shell (`src-tauri/`) hosting a React +
TypeScript frontend (`src/`) and managing a private Python 3.12 recognition
worker (`engine/`) over stdio. Build installers **on the host you ship for** —
Linux cannot cross-produce macOS DMG or Windows NSIS installers.

> Verified on Linux: frontend production build, Rust format/check and four
> tests, 31 Python tests, a Debian package build, and native app launch.
> A real 8.28-second public video exercised CUDA recognition in 4.1 seconds
> and CPU recognition in 20.9 seconds at beam 5. The JavaScript/WASM face
> detector also ran against the public video. These are smoke checks, not
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

A packaging helper may bundle `uv` with release artifacts later; for source
builds, install `uv` yourself first — it is required on PATH and is not
fetched automatically.

GPU wheels come from the standard PyTorch distribution; CPU-only machines use
the CPU fallback automatically. `settings.inferenceDevice` (`auto` | `gpu` |
`cpu`) selects auto (CUDA → MPS → CPU), explicit GPU-only, or CPU. The
default beam size is 5 (valid range 1–60): beam 30 exceeded the 6 GB GPU
memory on the tested CUDA host, while beam 5 transcribed an 8.28 s demo clip
in 4117 ms.

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

## Build and run

```sh
npm ci
npm run typecheck     # tsc -b
npm run dev           # labeled browser preview; native recognition unavailable
npm run desktop       # full Tauri app
npm run desktop:build # platform installer for this host
```

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
