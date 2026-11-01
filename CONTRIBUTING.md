# Contributing to Stilltype

Thanks for stopping by. Stilltype is a small, quiet codebase — the best
contributions are narrow, well-checked and honest about what was verified.

## What fits here

- Bug fixes with a known cause, camera-guidance and UI polish, docs corrections.
- Small, file-scoped improvements that follow the existing patterns
  (Tauri commands in `src-tauri/`, React surfaces in `src/`, engine modules in `engine/`).
- Tests for settled behavior, not for implementation internals.

Please don't open broad refactors, new dependencies, rewrites of vendored
inference code, or anything touching auth, secrets, migrations or production
operations without discussing it first. Frontend changes follow the design
tokens in `DESIGN.md` — parchment surfaces, vermilion reserved for identity
and recording, no audio waveforms, no invented accuracy figures, no seeded
demo dictations.

Recognition runs on the USR 2.0 Base+ high-resource checkpoint (2026 release);
engine changes must keep adapters separate so the selected model can change,
and must respect `settings.inferenceDevice` (`auto` | `gpu` | `cpu`):
auto prefers CUDA, then MPS, then CPU.

Model weights, virtualenvs, `dist/`, `target/` and local notes never go in a
pull request (see `.gitignore`).

## Checks

Run these from the repo root before asking for review:

```sh
npm ci
npm run typecheck   # tsc -b, must be clean
npm run build       # production frontend bundle, must succeed
```

For Rust changes, from `src-tauri/`:

```sh
cargo check
cargo test
```

Then smoke-test what you touched: `npm run desktop` for the full app, or
`npm run dev` for a labeled browser preview (native recognition is unavailable
there by design). For the model installer and worker:

```sh
python3 engine/install.py --help
python -u engine/worker.py --models /path/to/models
```

Engine tests live in `engine/tests/`; extend them when you change inference
behavior and run them with `python -m unittest discover -s engine/tests -v`
(with the engine requirements installed).

Report what you ran and what you saw, including the platform. Linux builds,
native launch, GPU/CPU inference and local face-detector execution have been
exercised. Windows/macOS execution and live webcam-to-paste behavior remain
unverified. Never present one platform's results as another's, or a public
video fixture as a real webcam session.
