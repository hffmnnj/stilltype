# Security policy

## What stays local

- **Camera frames and transcripts never leave your machine.** There is no
  account, no sync, no telemetry and no network recognition service. Inference
  runs in a private Python worker managed by the Tauri shell over local
  standard input/output.
- **The only intentional downloads** are the USR 2.0 recognition checkpoint
  (after you explicitly accept its licence), the small YuNet face-tracking
  artifact fetched once by `scripts/prepare-vision.mjs` (SHA256-verified,
  ~230 kB), and the Python runtime packages
  listed in `engine/requirements.txt`. Each is fetched from its published
  source and, for the checkpoint and face artifact, verified against its
  pinned SHA256. Nothing is fetched from a CDN while the app is running.
- **The only intentional local-network traffic** is to Ollama at
  `http://127.0.0.1:11434`, and only when you enable cleanup yourself. It is
  never contacted otherwise.

## What is ephemeral

Capture frames are written to temporary clip files that are deleted after
transcription completes or when you cancel. Failed or still-mouth clips are
rejected before inference rather than retained. Nothing is cached for later
upload because there is nowhere to upload to.

## What is *not* protected

- **Local history is not encrypted.** Dictations, vocabulary and settings live
  as plaintext in the app-data directory for `com.stilltype.desktop` (see
  [docs/BUILDING.md](docs/BUILDING.md)). Retention controls delete old entries,
  but anyone with access to your user account can read the store. If that
  matters for your threat model, use OS-level full-disk encryption.
- **Lipreading is experimental.** Visually ambiguous words are a correctness
  property, not a security boundary — review pasted text before sending it
  anywhere sensitive.

## Reporting a vulnerability

The project is young and has no dedicated security team or bounty. If you find
something that leaks user data off-device, bypasses licence consent, or
executes untrusted input, please report it privately to the repository
maintainers (via a private channel on the hosting service — do not file a
public issue for an active vulnerability). Include the version or commit, the
platform, and steps to reproduce. You can expect an acknowledgement and a fix
or mitigation plan, not a timeline promise.
