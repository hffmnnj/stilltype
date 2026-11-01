"""Tests for engine/install.py.

No live network: downloads go through a fake downloader injected into
``install.install``, so these run offline and in milliseconds while still
exercising the real verify-and-replace logic. Payloads are tiny fakes served
through the ``files`` seam; the manifest contract assertions use the real
pinned constants from the module.
"""

import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import install
from install import ChecksumMismatch, DownloadError, LicenseNotAccepted, ModelFile


GOOD_WEIGHTS = b"fake-but-consistent weights payload"
GOOD_SHA = hashlib.sha256(GOOD_WEIGHTS).hexdigest()
TINY_FILES = (
    ModelFile("https://example.invalid/usr2/baseplus.pth", "usr2/baseplus.pth", GOOD_SHA),
)


def fake_downloader(payloads, calls=None):
    """Build an install.Downloader serving canned bytes (or raising canned errors)."""

    def download(url, part, timeout, on_progress):
        if calls is not None:
            calls.append(url)
        payload = payloads[url]
        if isinstance(payload, Exception):
            raise payload
        part.parent.mkdir(parents=True, exist_ok=True)
        with open(part, "wb") as handle:
            handle.write(payload)
        on_progress(len(payload), len(payload))

    return download


class Collector:
    """Captures InstallProgress records emitted during install."""

    def __init__(self):
        self.events = []

    def __call__(self, stage, message, progress):
        self.events.append({"stage": stage, "message": message, "progress": progress})

    @property
    def stages(self):
        return [e["stage"] for e in self.events]


def read_manifest(models):
    with open(models / "installed.json", encoding="utf-8") as handle:
        return json.load(handle)


class InstallTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.models = Path(self.tmp.name) / "models"
        self.emit = Collector()

    def tearDown(self):
        self.tmp.cleanup()

    def test_models_without_license_raise_before_any_download(self):
        calls = []
        downloader = fake_downloader(
            {"https://example.invalid/usr2/baseplus.pth": GOOD_WEIGHTS}, calls
        )
        with self.assertRaises(LicenseNotAccepted):
            install.install(
                self.models,
                accept_license=False,
                files=TINY_FILES,
                downloader=downloader,
                emit=self.emit,
            )
        self.assertEqual(calls, [])
        self.assertFalse(self.models.exists())

    def test_cli_without_accept_license_fails_before_files(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = install.main(["--models", str(self.models)])
        self.assertEqual(code, 2)
        record = json.loads(out.getvalue().strip().splitlines()[-1])
        self.assertEqual(record["stage"], "error")
        self.assertFalse(self.models.exists())

    def test_weights_with_wrong_checksum_are_rejected(self):
        downloader = fake_downloader(
            {"https://example.invalid/usr2/baseplus.pth": b"tampered bytes"}
        )
        with self.assertRaises(ChecksumMismatch):
            install.install(
                self.models,
                accept_license=True,
                files=TINY_FILES,
                downloader=downloader,
                emit=self.emit,
            )
        self.assertIn("error", self.emit.stages)
        self.assertFalse((self.models / "usr2" / "baseplus.pth").exists())
        self.assertFalse((self.models / "usr2" / "baseplus.pth.part").exists())
        self.assertFalse((self.models / "installed.json").exists())

    def test_corrupt_redownload_never_replaces_known_good_weights(self):
        final = self.models / "usr2" / "baseplus.pth"
        final.parent.mkdir(parents=True, exist_ok=True)
        final.write_bytes(b"previous release payload")
        downloader = fake_downloader(
            {"https://example.invalid/usr2/baseplus.pth": b"tampered bytes"}
        )
        with self.assertRaises(ChecksumMismatch):
            install.install(
                self.models,
                accept_license=True,
                files=TINY_FILES,
                downloader=downloader,
                emit=self.emit,
            )
        self.assertEqual(final.read_bytes(), b"previous release payload")

    def test_mismatched_existing_weights_are_redownloaded(self):
        final = self.models / "usr2" / "baseplus.pth"
        final.parent.mkdir(parents=True, exist_ok=True)
        final.write_bytes(b"stale payload from an older mirror")
        downloader = fake_downloader(
            {"https://example.invalid/usr2/baseplus.pth": GOOD_WEIGHTS}
        )
        install.install(
            self.models,
            accept_license=True,
            files=TINY_FILES,
            downloader=downloader,
            emit=self.emit,
        )
        self.assertEqual(final.read_bytes(), GOOD_WEIGHTS)

    def test_matching_weights_are_trusted_without_redownload(self):
        final = self.models / "usr2" / "baseplus.pth"
        final.parent.mkdir(parents=True, exist_ok=True)
        final.write_bytes(GOOD_WEIGHTS)
        calls = []

        def boom(url, part, timeout, on_progress):
            calls.append(url)
            raise AssertionError("network should not be touched")

        install.install(
            self.models, accept_license=True, files=TINY_FILES, downloader=boom, emit=self.emit
        )
        self.assertEqual(calls, [])
        self.assertEqual(self.emit.events[-1]["stage"], "ready")
        self.assertEqual(self.emit.events[-1]["progress"], 1.0)
        self.assertEqual(
            read_manifest(self.models),
            {
                "modelId": "usr2-baseplus-2026",
                "releaseYear": 2026,
                "checkpoint": "usr2/baseplus.pth",
                "sha256": install.CHECKPOINT_SHA256,
            },
        )

    def test_successful_install_reports_ready_at_progress_one(self):
        downloader = fake_downloader(
            {"https://example.invalid/usr2/baseplus.pth": GOOD_WEIGHTS}
        )
        install.install(
            self.models,
            accept_license=True,
            files=TINY_FILES,
            downloader=downloader,
            emit=self.emit,
        )
        self.assertEqual((self.models / "usr2" / "baseplus.pth").read_bytes(), GOOD_WEIGHTS)
        self.assertEqual(
            read_manifest(self.models),
            {
                "modelId": "usr2-baseplus-2026",
                "releaseYear": 2026,
                "checkpoint": "usr2/baseplus.pth",
                "sha256": install.CHECKPOINT_SHA256,
            },
        )
        progresses = [e["progress"] for e in self.emit.events]
        self.assertTrue(all(0.0 <= p <= 1.0 for p in progresses))
        self.assertEqual(progresses, sorted(progresses))
        self.assertEqual(self.emit.events[-1], {
            "stage": "ready",
            "message": "all models installed and verified",
            "progress": 1.0,
        })

    def test_default_model_is_the_single_usr2_baseplus_checkpoint(self):
        self.assertEqual(len(install.MODEL_FILES), 1)
        want = install.MODEL_FILES[0]
        self.assertEqual(want.relpath, "usr2/baseplus.pth")
        self.assertEqual(want.url, "https://drive.google.com/uc?id=18vmJjdem5XPOA8bmizybIW5sLJuHMdRR")
        self.assertEqual(
            want.sha256, "4ab1f3f230aa6a724bcc698db8a17f2f54919136d2885c1a07a6b69247012773"
        )
        self.assertEqual(install.MODEL_ID, "usr2-baseplus-2026")
        self.assertEqual(install.RELEASE_YEAR, 2026)
        help_text = install.build_parser().format_help()
        self.assertIn("USR", help_text)
        self.assertIn("installed.json", help_text)
        for old_name in ("Auto-AVSR", "HuggingFace", "Amanvir", "vsr/model.pth"):
            self.assertNotIn(old_name, help_text)
        source = Path(install.__file__).read_text(encoding="utf-8")
        for old_name in ("Auto-AVSR", "Amanvir", "LRS3", "lm_subword"):
            self.assertNotIn(old_name, source)


if __name__ == "__main__":
    unittest.main()
