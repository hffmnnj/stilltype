#!/usr/bin/env python3
"""Download Stilltype's lipreading model weights.

Fetches the single USR 2.0 Base+ high-resource checkpoint (released 2026)
from its official Google Drive link into ``<models>/usr2/baseplus.pth`` and
records ``<models>/installed.json`` so the engine can load it.

GPU is preferred and CPU is supported as an inference device mode using this
same checkpoint. The checkpoint is large (~1.4 GB); allow time and disk
space. The Google Drive fetch uses the battle-tested ``gdown`` package, and
the file is verified against a pinned SHA256 before use.

Usage:
    python engine/install.py --models PATH --accept-license
    python engine/install.py --models ~/.local/share/stilltype/models --accept-license
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

LICENSE_LINE = "Use under the upstream USR2 model terms; confirm intended use."
WEIGHT_NOTE = "Checkpoint is about 1.4 GB; allow time and disk space."
SOURCE_NOTE = (
    "Model: USR 2.0 Base+ high-resource (released 2026) from "
    "https://github.com/ahaliassos/usr2 (created 2026-01-30, arXiv 2602.19316). "
    "USR2 code is MIT; no separate upstream weight-license statement was "
    "found, so the weights are used under the upstream USR2 model terms. "
    "GPU preferred, CPU supported (same checkpoint, device mode only)."
)

MODEL_ID = "usr2-baseplus-2026"
RELEASE_YEAR = 2026
CHECKPOINT_URL = "https://drive.google.com/uc?id=18vmJjdem5XPOA8bmizybIW5sLJuHMdRR"
CHECKPOINT_SHA256 = "4ab1f3f230aa6a724bcc698db8a17f2f54919136d2885c1a07a6b69247012773"
CHECKPOINT_BYTES = 1389096590
CHECKPOINT_REL = "usr2/baseplus.pth"
MANIFEST_REL = "installed.json"

CHUNK_BYTES = 1 << 20  # 1 MiB per read; weights stream straight to disk.
CONNECT_TIMEOUT = 30.0  # seconds, kept for the injectable downloader seam.


@dataclass(frozen=True)
class ModelFile:
    """One artifact to fetch: URL, path under the models dir, digest (if any)."""

    url: str
    relpath: str
    sha256: str | None = None


MODEL_FILES: tuple[ModelFile, ...] = (
    ModelFile(CHECKPOINT_URL, CHECKPOINT_REL, CHECKPOINT_SHA256),
)

ProgressFn = Callable[[str, str, float], None]


class LicenseNotAccepted(Exception):
    """Raised when --accept-license was not passed."""


class DownloadError(Exception):
    """Raised when a download or verification step fails."""


class ChecksumMismatch(DownloadError):
    """Raised when the downloaded checkpoint fails its SHA256 check."""


def emit_progress(stage: str, message: str, progress: float) -> None:
    """Write one newline-JSON InstallProgress record to stdout."""
    print(json.dumps({"stage": stage, "message": message, "progress": progress}), flush=True)


def diag(message: str) -> None:
    """Write human diagnostics to stderr (stdout is the progress stream)."""
    print(message, file=sys.stderr, flush=True)


def sha256_of(path: Path) -> str:
    """Hex SHA256 of a file, streamed so multi-GB weights fit in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def final_is_good(final: Path, want: ModelFile) -> bool:
    """Trust the existing checkpoint only after re-hashing it, never by presence."""
    if not final.is_file() or final.stat().st_size == 0:
        return False
    if want.sha256 is None:
        return False
    try:
        return sha256_of(final) == want.sha256
    except OSError:
        return False


def download_with_gdown(
    url: str,
    part: Path,
    timeout: float,
    on_progress: Callable[[int, int | None], None],
) -> None:
    """Fetch ``url`` into ``part`` with gdown (created/overwritten, never resumed).

    ``timeout`` is accepted for seam compatibility and ignored: gdown manages
    its own HTTP session. gdown's own activity goes to stderr; progress JSON
    stays at honest file-level milestones (no invented byte percentages).
    """
    try:
        import gdown
    except ImportError as exc:
        raise DownloadError(
            "gdown is required to fetch the USR2 checkpoint: "
            "install engine/requirements.txt first"
        ) from exc
    part.parent.mkdir(parents=True, exist_ok=True)
    on_progress(0, None)
    try:
        gdown.download(url, str(part), quiet=False, fuzzy=True)
    except Exception as exc:
        raise DownloadError(f"download of {url} failed: {exc}") from exc
    if not part.is_file() or part.stat().st_size == 0:
        raise DownloadError(f"download of {url} produced no file; retry")
    on_progress(part.stat().st_size, None)


Downloader = Callable[[str, Path, float, Callable[[int, int | None], None]], None]


def write_manifest(models_dir: Path) -> None:
    """Atomically record which model is installed (native/inference contract)."""
    manifest = {
        "modelId": MODEL_ID,
        "releaseYear": RELEASE_YEAR,
        "checkpoint": CHECKPOINT_REL,
        "sha256": CHECKPOINT_SHA256,
    }
    target = models_dir / MANIFEST_REL
    tmp = target.with_name(target.name + ".tmp")
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    os.replace(tmp, target)


def install(
    models_dir: Path,
    *,
    accept_license: bool,
    files: Sequence[ModelFile] | None = None,
    timeout: float = CONNECT_TIMEOUT,
    downloader: Downloader = download_with_gdown,
    emit: ProgressFn = emit_progress,
) -> None:
    """Download and verify the checkpoint, then write installed.json; raise on failure.

    An existing checkpoint is re-hashed and only re-downloaded when it fails.
    A corrupt download is deleted and never moved over a known-good file; the
    move itself is atomic. The manifest is written atomically after the
    checkpoint verifies, on both the download and skip paths.
    """
    if not accept_license:
        raise LicenseNotAccepted(
            "Refusing to download: pass --accept-license to acknowledge the "
            f"model/upstream terms ({LICENSE_LINE} {WEIGHT_NOTE})"
        )
    wanted = list(MODEL_FILES if files is None else files)
    total = len(wanted)
    done = 0
    for index, want in enumerate(wanted):
        final = models_dir / want.relpath
        part = final.with_name(final.name + ".part")
        if final_is_good(final, want):
            done += 1
            emit("download", f"{want.relpath} already present and verified", done / total)
            continue
        emit("download", f"downloading {want.relpath}", index / total)

        def on_progress(_n: int, _total: int | None, _i: int = index) -> None:
            # Monotonic within this file would need its length; file-count
            # progress keeps the bar honest without guessing sizes.
            emit("download", f"downloading {want.relpath}", (_i + 0.5) / total)

        try:
            downloader(want.url, part, timeout, on_progress)
        except DownloadError:
            emit("error", f"{want.relpath} download failed; retry to resume", index / total)
            raise
        emit("verify", f"verifying {want.relpath}", (index + 0.5) / total)
        try:
            if want.sha256 is None:
                raise DownloadError(f"{want.relpath} has no pinned digest; refusing to trust it")
            if sha256_of(part) != want.sha256:
                raise ChecksumMismatch(
                    f"{want.relpath} failed its SHA256 check; "
                    "the download may be corrupt. Retry."
                )
        except (ChecksumMismatch, DownloadError):
            part.unlink(missing_ok=True)
            emit("error", f"{want.relpath} failed verification and was rejected", index / total)
            raise
        except OSError as exc:
            part.unlink(missing_ok=True)
            emit("error", f"{want.relpath} could not be verified: {exc}", index / total)
            raise DownloadError(f"{want.relpath} could not be verified: {exc}") from exc
        os.replace(part, final)
        done += 1
        emit("download", f"{want.relpath} installed", done / total)
    write_manifest(models_dir)
    emit("ready", "all models installed and verified", 1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download Stilltype's USR 2.0 Base+ (2026) checkpoint into --models. "
            "GPU preferred, CPU supported (same checkpoint)."
        ),
        epilog=(
            f"Model terms: {LICENSE_LINE} {SOURCE_NOTE} {WEIGHT_NOTE} "
            f"Checkpoint: {CHECKPOINT_REL} ({CHECKPOINT_BYTES} bytes, "
            f"SHA256 {CHECKPOINT_SHA256}). Manifest: {MANIFEST_REL}."
        ),
    )
    parser.add_argument(
        "--models",
        type=Path,
        required=True,
        help="Directory holding usr2/baseplus.pth and installed.json (created if missing).",
    )
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help=f"Acknowledge the model/upstream terms ({LICENSE_LINE}) and start downloading.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=CONNECT_TIMEOUT,
        help="Accepted for compatibility; the gdown fetch manages its own HTTP session.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        install(args.models, accept_license=args.accept_license, timeout=args.timeout)
    except LicenseNotAccepted as exc:
        emit_progress("error", str(exc), 0.0)
        diag(f"error: {exc}")
        return 2
    except DownloadError as exc:
        diag(f"error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
