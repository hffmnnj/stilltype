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

### Experimental comparisons

`--motion-threshold` changes the host motion gate for an experiment only; the
production baseline is 8. `--ctc-beam-size 10 --ctc-token-top-k 20` selects a
token-pruned CTC prefix-marginal decoder. The default `--ctc-beam-size 1` keeps
greedy decoding. Prefix search sums paths that collapse to the same output;
it has no language model, word list, length bonus or reference input. It is
not the desktop joint attention/CTC decoder, and a higher-probability prefix
does not necessarily have fewer word errors.

```sh
python tools/evaluate_lipreading.py --manifest /path/evaluation/manifest.json \
  --split dev --decoder ctc-int8 --mobile-model-dir /path/mobile-model \
  --ctc-beam-size 10 --ctc-token-top-k 20 --motion-threshold 8 \
  --output /path/evaluation/ctc-prefix10.json
```

For independently captured predictions, use `--score-predictions /path/preds.json`.
That file contains an `items` array with `id`, `raw` and `error` (null on success).
IDs must cover the selected split exactly. This mode needs no torch, ORT or
weights, and does not run inference.

```sh
python -m unittest discover -s tools -p 'test_evaluation.py' -v
python -m unittest discover -s tools -p 'test_prepare_evaluation.py' -v
python -m unittest discover -s tools -p 'test_ctc_decode.py' -v
```

## First fixture comparison: GRID, November 2026

**The 10% WER target was not reached.** This small out-of-domain comparison
exposed substantial failures; it does not support a 90% product-accuracy claim.

Source: the [University of Sheffield GRID corpus](https://spandh.dcs.shef.ac.uk/gridcorpus/),
which states the recordings and transcriptions are freely available for research
use. No dataset content is redistributed here. GRID uses a constrained six-word
grammar, not natural open dictation. These are ordinary spoken recordings,
with no audio used by recognition. Training overlap is unknown, and two
speakers do not establish population-level performance.

The fixed set has 20 clips / 120 reference words from speaker 1 for development
and 20 / 120 from speaker 2 for testing. IDs were selected before inference:
sort the intersection of MPEG and alignment-file stems for each speaker, then
use `random.Random(20261101 + speaker).sample(ids, 20)` in Python 3.12 and sort
the result. Videos are the normal-quality archives `s1/video/s1.mpg_vcd.zip`
and `s2/video/s2.mpg_vcd.zip`, with references from `s1/align/s1.tar` and
`s2/align/s2.tar`, all linked by the corpus page. The local manifest SHA-256 is
`8fe59857e3159977088d4d98560436678ce3ac447c170b47c2b83c0f7b797c1a`.
Per-item source hashes refer to the extracted MPEG files. References, raw
hypotheses and prepared frames remain outside Git.

Preparation produced 3,000 frames at 25 fps; YuNet detected a face in 2,988.
All rows below use the original motion threshold of 8, no vocabulary or
cleanup, and include rejected clips as full deletions:

| Decoder | Development WER | Test WER |
| --- | ---: | ---: |
| Desktop Base+ joint beam 5 | 87.50% | 97.50% |
| Desktop Base+ joint beam 40 | 88.33% | 97.50% |
| Host CTC FP32 greedy | 88.33% | 95.83% |
| Host CTC INT8 greedy | 88.33% | 95.83% |
| Host CTC INT8 prefix beam 10, top 20 tokens | 88.33% | 95.00% |

No configuration produced an exact six-word sentence. The motion gate rejected
6/20 development and 15/20 test clips as `no-speech`, despite the supplied speech
annotations. This is a major pipeline weakness, not a reason to exclude those
clips from the score. Beam 40 roughly doubled development model-inference time
(7.0 to 14.9 seconds across the set) without improving WER. Prefix search
changed only one test word error overall and did not improve development WER;
it is not enabled in the app.

Lowering the gate to 4 admitted all development speech clips and reduced host
INT8 development WER to 77.50%, but also admitted **7/27 annotated non-speech
control segments**, versus 0/27 at the original threshold. Controls were natural
`sil`/`sp` regions of the same development videos, with alignment positions
interpreted at the corpus's 25 kHz audio rate, trimmed by 100 ms at the start and 40 ms at the end,
and required at least 13 video frames. Admission means a segment could reach
recognition, not that it generated a false transcript. That safety regression
was enough to reject the lower production threshold; it was not tuned further
against the test speaker.

Desktop inference used an RTX 4050 with CUDA. CTC used host CPU ORT 1.24.3,
four intra-op threads and one inter-op thread, with OpenCV 4.11 preprocessing.
These results are not Android device measurements, paper reproductions, or
silent-mouthing accuracy. A stronger open-speech evaluation and a separately
consented silent-mouthing set are needed before choosing new app defaults.
