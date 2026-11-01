# Stilltype engineering

## Product constraints

- Keep the repository self-branded. Do not add competitor names, comparison copy, or copied competitor code. Preserve the original upstream licenses for code actually used.
- Use 2026 recognition releases, prefer GPU, and retain the real CPU mode. Do not describe a newly exported utility model as newly trained weights; keep its provenance explicit.
- Never commit recognition weights, captured video, transcripts, Python environments, or local workflow plans.

## Integration seams

- Keep `src/lib/types.ts`, Rust camelCase serialization, and the Python worker protocol aligned. Worker stdout is JSONL only; route diagnostics to stderr.
- When moving or adding engine resources, update `src-tauri/tauri.conf.json`. The vendored modules live at `engine/espnet/`; `engine/usr2/` contains labels, the mean face and its license. Tauri rejects unmatched resource globs.
- Keep the camera import on `onnxruntime-web/wasm`. The prepared runtime assets do not include the WebGPU/JSEP module selected by the default entry point.
- Keep optional cleanup fixed to loopback, with proxies and redirects disabled. `engine/tests/test_privacy.py` exercises that boundary against a temporary local server.

## UI work

- Read `DESIGN.md` before UI changes, update it with token or behavior changes, and lint it with `npx @google/design.md lint DESIGN.md` from the repository root.
