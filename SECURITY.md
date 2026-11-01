# Security policy

## What stays local

- **Camera frames and transcripts never leave your machine.** There is no
  account, no sync, no telemetry and no network recognition service. Inference
  runs in a private Python worker managed by the Tauri shell over local
  standard input/output on desktop, and inside ONNX Runtime on Android.
- **The only intentional downloads** are the USR 2.0 recognition checkpoint
  (after you explicitly accept its licence), the small YuNet face-tracking
  artifact fetched once by `scripts/prepare-vision.mjs` (SHA256-verified,
  ~230 kB), the optional Qwen text-cleanup GGUF (0.58–3.01 GB by tier, consent-gated and
  SHA256-verified before use — see [docs/BUILDING.md](docs/BUILDING.md)),
  and the Python runtime packages
  listed in `engine/requirements.txt`. Each is fetched over HTTPS from its
  published source and, for the checkpoint, face artifact and cleanup model,
  verified against its pinned SHA256. Nothing is fetched from a CDN while the
  app is running.
- **Inference itself is strictly local with no network traffic.** Visual
  recognition runs in the private Python worker over local standard
  input/output, and optional text cleanup runs inside the Rust application
  via its compiled-in runtime. There is no external cleanup server, no
  listening socket and no inference HTTP request — public model downloads
  above are the only network use, and they happen once at install time.
- **Android has no Internet or microphone permission.** Its recognition assets
  are bundled at build time, then copied into app-private storage with size
  and SHA256 verification after explicit preparation consent. The Quick
  Settings tile and pinned shortcut open a visible capture activity; they do
  not run a background camera or request Accessibility/overlay access.

## What is ephemeral

Capture frames are written to temporary clip files that are deleted after
transcription completes or when you cancel. Failed or still-mouth clips are
rejected before inference rather than retained. Nothing is cached for later
upload because there is nowhere to upload to.

Android keeps bounded mouth frames in memory rather than recording video to
disk. Pausing capture releases its camera and invalidates the current session;
late recognition results from a cancelled session are not saved.

## What is *not* protected

- **Local history is not encrypted.** Dictations, vocabulary and settings live
  as plaintext in the app-data directory for `com.stilltype.desktop`, or
  Android app-private DataStore for `com.stilltype.android` (see
  [docs/BUILDING.md](docs/BUILDING.md) and [docs/ANDROID.md](docs/ANDROID.md)).
  Retention controls delete old entries,
  but anyone with access to your user account can read the store. If that
  matters for your threat model, use OS-level full-disk encryption.
- **Lipreading is experimental.** Visually ambiguous words are a correctness
  property, not a security boundary — review pasted text before sending it
  anywhere sensitive.
- **Intent correction can change wording.** It is optional, off by default,
  and preserves the original recognition. Edit, number, negation and vocabulary
  guards reduce risk; they cannot prove that the meaning stayed the same.
- **Model loading is not a sandbox against your own account.** Pinned checksums
  verify downloaded bytes. A process with write access to app data can replace
  a model between verification and the native loader reopening its path, or
  modify other application files. Protect the account and app-data directory;
  checksums do not isolate Stilltype from another process running as you.
- **Copy and Share are intentional disclosures.** Copied text is available to
  the OS clipboard and apps allowed to read it. Android exports are shared
  through a temporary FileProvider URI permission to the app you choose.
  Recipient apps and their backups are outside Stilltype's control.

## Reporting a vulnerability

The project is young and has no dedicated security team or bounty. If you find
something that leaks user data off-device, bypasses licence consent, or
executes untrusted input, please report it privately to the repository
maintainers (via a private channel on the hosting service — do not file a
public issue for an active vulnerability). Include the version or commit, the
platform, and steps to reproduce. You can expect an acknowledgement and a fix
or mitigation plan, not a timeline promise.
