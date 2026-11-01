#!/usr/bin/env python3
"""Stilltype engine worker: persistent local AVSR over stdio JSONL.

Rust spawns this process once per session::

    python -u engine/worker.py --models MODEL_DIR

then sends one JSON request per stdin line and reads exactly one JSON
response per stdout line. All logs (including vendored initialisation
prints) go to stderr; stdout carries only responses.

Request: {id: string, command: "health" | "transcribe",
          clipDir?: string, final?: boolean, settings?: object,
          vocabulary?: [{id, word, replacement, createdAt}]}
Response: {id, ok: true, result: object} | {id, ok: false, error: string}

- health -> {ready, device, modelInstalled, error}; never loads the model.
- transcribe -> {text, raw, durationMs, inferenceMs, frames, error, device};
  the model lazy-loads on first use and is reused. `device` is the actual
  torch device used (additive: Rust may cache it, the UI ignores it).
  `final: false` decodes only up to the first pause (interim preview);
  Rust owns the clip lifetime either way.

Smoke test (same protocol, exits after one request)::

    echo '{"id":"h","command":"health"}' | python engine/worker.py --once
    echo '{"id":"t","command":"transcribe","clipDir":"/tmp/clip"}' \\
        | python engine/worker.py --models MODEL_DIR --once
"""

import argparse
import json
import os
import sys

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
if ENGINE_DIR not in sys.path:
    sys.path.insert(0, ENGINE_DIR)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Stilltype local AVSR worker (stdio JSONL).")
    p.add_argument("--models", default=None, help="Model weights root (sets *_MODELS env).")
    p.add_argument(
        "--once",
        action="store_true",
        help="Handle a single JSON request from stdin, print the response, exit.",
    )
    return p.parse_args(argv)


def _warn(msg: str) -> None:
    print(f"stilltype: {msg}", file=sys.stderr)


def handle_line(line: str, inference) -> dict | None:
    """Handle one protocol line. Returns the response dict, or None for blank lines."""
    if not line.strip():
        return None
    try:
        req = json.loads(line)
    except json.JSONDecodeError as e:
        return {"id": None, "ok": False, "error": f"invalid-json: {e}"}
    if not isinstance(req, dict):
        return {"id": None, "ok": False, "error": "invalid-request: expected a JSON object"}
    rid = req.get("id")
    cmd = req.get("command")
    if cmd == "health":
        try:
            return {"id": rid, "ok": True, "result": inference.health_snapshot()}
        except Exception as e:  # noqa: BLE001
            return {"id": rid, "ok": False, "error": f"health-failed: {type(e).__name__}: {e}"}
    if cmd == "transcribe":
        clip_dir = req.get("clipDir")
        if not isinstance(clip_dir, str) or not clip_dir:
            return {"id": rid, "ok": False, "error": "missing-clip-dir"}
        settings = req.get("settings", {})
        if settings is None:
            settings = {}
        if not isinstance(settings, dict):
            return {"id": rid, "ok": False, "error": "invalid-settings: expected an object"}
        vocabulary = req.get("vocabulary", [])
        if vocabulary is None:
            vocabulary = []
        if not isinstance(vocabulary, list):
            return {"id": rid, "ok": False, "error": "invalid-vocabulary: expected an array"}
        final = req.get("final", True)
        try:
            result = inference.transcribe_clip(clip_dir, settings=settings, vocabulary=vocabulary,
                                               final=bool(final))
        except inference.ClipError as e:
            # Decode-stage rejection (e.g. no frame survived JPEG decode).
            result = {
                "text": "",
                "raw": "",
                "durationMs": 0,
                "inferenceMs": 0,
                "frames": 0,
                "error": e.code,
            }
        except inference.ModelUnavailable as e:
            return {"id": rid, "ok": False, "error": str(e)}
        except Exception as e:  # noqa: BLE001 -- traceback to stderr, short code to Rust
            import traceback

            traceback.print_exc(file=sys.stderr)
            return {"id": rid, "ok": False, "error": f"transcribe-failed: {type(e).__name__}: {e}"}
        return {"id": rid, "ok": True, "result": result}
    return {"id": rid, "ok": False, "error": f"unknown-command: {cmd!r}"}


def _emit(resp: dict) -> None:
    sys.stdout.write(json.dumps(resp) + "\n")
    sys.stdout.flush()


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.models:
        here = os.path.abspath(args.models)
        # Set before importing inference, which reads this for its weight root.
        os.environ["STILLTYPE_MODELS"] = here
    import inference

    if args.once:
        for line in sys.stdin:
            if not line.strip():
                continue
            _emit(handle_line(line, inference))
            return 0
        _warn("no request on stdin")
        return 2
    for line in sys.stdin:
        resp = handle_line(line, inference)
        if resp is not None:
            _emit(resp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
