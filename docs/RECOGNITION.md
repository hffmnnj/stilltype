# Measuring recognition

Stilltype's accuracy target is **word error rate (WER) at or below 10%** on a
fixed evaluation set. This is a target, not a measured product claim. A few
successful demo clips cannot establish general lipreading accuracy.

## What counts

- Score raw visual recognition, before vocabulary replacement or text cleanup.
- Never give the recognizer the reference transcript, audio, subtitles, or a
  benchmark-specific word list.
- Keep development and test clips separate. Freeze test IDs, references and
  source hashes before changing the model or decoder. Split by speaker/source
  when possible, not different cuts of the same recording.
- Report ordinary spoken videos separately from genuine silent mouthing.
  Removing audio from a spoken video does not make it silent-mouthing data.
- Record training overlap as unknown unless there is evidence otherwise.
- Include empty predictions and failed clips in the denominator. Do not report
  only the clips that succeeded.

WER is `(substitutions + deletions + insertions) / reference words`, aggregated
across the selected corpus, not an average of clip percentages. It can exceed
100%. Punctuation and case are ignored; internal apostrophes are retained.
Numbers are not expanded and semantically similar phrases still count as errors.
Sentence exact match, latency and word error rate are separate measurements.

## Local evaluation manifest

Keep videos, transcripts, prepared clips, models and result JSON **outside Git**.
Download only material whose terms permit the intended evaluation; a public
video URL alone is not a reuse license. Do not redistribute dataset material in
the app or repository.

The UTF-8 JSON manifest has `schemaVersion: 1`, a `datasetId` string and an
`items` array. Each item requires:

| Field | Meaning |
| --- | --- |
| `id` | Unique fixed clip ID |
| `split` | `dev` or `test` |
| `clipDir` | Directory containing `frames.jsonl`, relative to the manifest or absolute |
| `reference` | Exact human-checked words for this clip |
| `sourceUrl`, `sourceSha256` | Original video URL and its 64-character lowercase SHA-256 |
| `license` | Applicable evaluation terms and their source |
| `speakerGroup` | Source/speaker grouping used for the split |
| `speechMode` | `spoken` or `silent` |
| `trainingOverlap` | `unknown`, `known-unseen` or `known-seen` |

## Run a comparison

Use the prepared Python runtime described in [BUILDING.md](BUILDING.md). Run
from the repository root, with output paths outside the checkout:

```sh
python tools/prepare_evaluation.py --video /path/source.mp4 \
  --detector /path/mobile-model/yunet-2026may.onnx \
  --start-ms 0 --end-ms 6000 --fps 25 --output /path/evaluation/clip-001

python tools/evaluate_lipreading.py --manifest /path/evaluation/manifest.json \
  --split dev --decoder desktop --models /path/models --device gpu \
  --beam-size 5 --output /path/evaluation/desktop-beam5.json

python tools/evaluate_lipreading.py --manifest /path/evaluation/manifest.json \
  --split dev --decoder desktop --models /path/models --device gpu \
  --beam-size 40 --output /path/evaluation/desktop-beam40.json

python tools/evaluate_lipreading.py --manifest /path/evaluation/manifest.json \
  --split dev --decoder ctc-int8 --mobile-model-dir /path/mobile-model \
  --output /path/evaluation/ctc-int8.json
```

The preparer reads video only, keeps no-face observations, and uses the same
contain-fit, YuNet input and four-point landmark geometry as capture. It streams
JPEG frames rather than holding a full-resolution session in memory. File frame
timing is not wall-clock camera timing. Its metadata records the video/detector
hashes, exact selected range, observed frame rate and face count. Variable-frame-
rate sources should first be converted to a known constant rate outside Git;
this tool derives timestamps from the file's reported frame rate.

Use `--device cpu` for a real desktop CPU run. `ctc-fp32` selects the unquantized
export. CTC runs use host ORT CPU and desktop temporal mouth alignment: they are
**not Android-device accuracy or speed measurements**. See
[ANDROID.md](ANDROID.md) for the different phone preprocessing path.

Results contain raw hypotheses, word edit counts, per-mode/corpus metrics,
manifest and prepared-clip hashes, source provenance, decoder settings and
timings. Existing results are not overwritten. `targetMet` refers only to the
selected samples; it is not a claim of representative 90% accuracy.

For independently captured predictions, use `--score-predictions /path/preds.json`.
That file contains an `items` array with `id`, `raw` and `error` (null on success).
IDs must cover the selected split exactly. This mode needs no torch, ORT or
weights, and does not run inference.

```sh
python -m unittest discover -s tools -p 'test_evaluation.py' -v
python -m unittest discover -s tools -p 'test_prepare_evaluation.py' -v
```
