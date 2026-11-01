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

`--activity-gate ctc-evidence` measures the Android 0.1.2 speech gate with
either CTC decoder. It uses motion threshold 4 for context, with the original
threshold 8 as the strong-speech check. A window that fails the strong check
also needs two collapsed nonblank/non-EOS greedy CTC emissions, mean emission
probability at least 0.30, and nonblank text. Prefix search does not supply this
evidence; it comes from the same model output's greedy path. Both motion spans
require 0.3 seconds of movement and use 0.4-second padding. Non-finite output
fails recognition. The evaluator defaults to `legacy` for reproducible baseline
comparisons; the flag is not supported for desktop or score-only runs.

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

## CTC activity-gate improvement

Android 0.1.2 adopts the evidence gate above, rather than admitting weak motion
alone. This addresses falsely rejected speech; it does not resolve visually
ambiguous words. The fixed original split and a further frozen 40-clip set gave:

| Host INT8 greedy | Development, 120 words | Test, 120 words | Confirmation, 240 words |
| --- | ---: | ---: | ---: |
| Original gate | 106/120 = 88.33% | 115/120 = 95.83% | 223/240 = 92.92% |
| CTC evidence gate | 93/120 = 77.50% | 97/120 = 80.83% | 204/240 = 85.00% |

All rejected clips remain in the scores. The candidate rejected no speech clips
in these sets and admitted none of the 27 development quiet controls. Those
controls are limited evidence, not a general false-activation guarantee. Test
word errors fell by 15.65%, and confirmation errors by 8.52% relative to the
original gate: **neither is the requested halving**. No exact sentence was
recognized in the original test split.

The additional set used 20 new IDs from each of the same two GRID speakers,
excluding the original 40 IDs, selected with `random.Random(20261102 + speaker)`
before candidate tuning. Its manifest SHA-256 is
`00cff9dab43546d9bc867603b31fc0a005803e7ad8e1e1c5a6139a11cd1ea021`.
The set is disjoint, not a new population of speakers. Model weights and text
references were not changed or supplied as recognition hints.

Real API 36 emulator QA also exercised a legacy-rejected speech ROI, a weak-motion
quiet ROI, and a 120-second / 3,001-frame repeated-ROI session through the Android
recognizer. Speech became nonempty, quiet remained `no-speech`, and the full
session completed. The short speech fixture still had 5 errors in 6 words; this
is pipeline evidence, not accurate dictation or physical-phone verification.

## Broader 2026 model experiments

These are research measurements, **not a new app engine or a 90%-accuracy
claim**. [FLARE](https://github.com/jxzhanggg/FLARE) combines full-face and lip
inputs with a Transformer decoder. Its [2026 paper, Table 2](https://arxiv.org/html/2610.19559v1)
distinguishes the Base and Large encoders from the separate 7B LLM variant.
The paper's best 13.2% LRS3 WER belongs to Large plus the LLM, not the Base
checkpoint used below. Neither a paper score nor these experiments establishes
performance on silent mouthing or a phone.

The [released checkpoints](https://huggingface.co/Anony94/FLARE-checkpoints)
state **CC-BY-NC-4.0**, separate from the repository's code license. Research
used the fine-tuned Base checkpoint at revision
`6187ad650dc1786a8efc8b6bfbee37bbf62c4b2e`, SHA-256
`fe8c8ecab633bd4e3e490924e2bc917420b4d18d7309a5cd612902ccc842853b`.
Weights, videos, prepared arrays and reference text are not redistributed here.
Loading used `weights_only=True`, an explicit three-class allowlist and strict
state-dict matching; missing training-only configuration was reconstructed from
the published configuration and launch scripts.

### Natural-speech diagnostic

A public-domain [White House weekly address](https://commons.wikimedia.org/wiki/File:2017-01-07_President_Obama%27s_Weekly_Address.webm)
provided a more natural-language diagnostic than GRID. Before inference, the
first 42 subtitle cues were grouped consecutively into at most 7.8-second clips:
8 development clips / 141 reference words, then 9 test clips / 148 words. The
manifest SHA-256 is
`2a7c7ce060a20badfcfd2c34c929941791fa6a0a4d0d7bab9624907429d59737`.
Both splits share one speaker and source; they are not independent populations.
Training overlap is unknown. Caption-derived words and timing have not been
independently audio-verified, so these scores are **provisional**. Audio,
captions and references were never inputs to recognition or correction.

| System | Development errors / words | Test errors / words |
| --- | ---: | ---: |
| Current desktop Base+, joint beam 5 | 70/141 = 49.65% | 70/148 = 47.30% |
| FLARE Base, beam 50 | 46/141 = 32.62% | 34/148 = 22.97% |
| FLARE Base + optional Qwen3.5 4B Intent | 45/141 = 31.91% | 32/148 = 21.62% |

The FLARE test error count is 51.43% below the desktop baseline on this source.
This does **not** mean the original GRID target was halved. Intent removed only
two further test errors, not half the remaining errors, and is not raw visual
accuracy. All three Qwen tiers were tested on development hypotheses; the two
smaller tiers left FLARE WER unchanged. One rejected 4B development correction
fell back to the raw text. No tier received the reference or a speaker hint.

Preprocessing is a material confound: the current desktop model with the
alternative lip crops scored 54/141 development errors instead of 70/141.
FLARE used approximate aligned full-face crops and an approximate MediaPipe
68-point mapping, rather than exact published preprocessing. It decoded prepared
JPEG frames. This is not a paper reproduction. The larger 342M-encoder
checkpoint scored 44/141 development errors, versus Base's 46/141; that small
gain did not justify a switch or another held-out comparison.

### Original-corpus and quiet checks

The same Base configuration was also measured on all 80 fixed GRID speech
clips and 27 development quiet regions, with no decoder tuning against their
references. The gate below applies the existing threshold-8 acceptance check
from desktop-aligned ROIs; accepted FLARE predictions use the full clip, not
the desktop decoder's trimmed speech span.

| FLARE Base | Development, 120 words | Test, 120 words | Confirmation, 240 words |
| --- | ---: | ---: | ---: |
| Ungated research decode | 88/120 = 73.33% | 83/120 = 69.17% | 171/240 = 71.25% |
| Existing motion-8 acceptance | 99/120 = 82.50% | 110/120 = 91.67% | 212/240 = 88.33% |

No preparation or model calls failed. **All 27 quiet regions produced nonempty
text without a gate**; the existing gate rejected all 27. Consequently an
ungated model switch is not safe to ship. The original fixed-set halving and
the 10% WER target remain unmet. A new model needs validated quiet detection,
portable runtime measurements, and stronger source/speaker coverage before
becoming an application default.

The weak-motion CTC-evidence idea was also tried with the existing desktop
joint-beam decoder on CPU. It recovered rejected clips but raised confirmation
WER from 236/240 to 244/240 errors. That decoder/gate combination was rejected;
the Android improvement does not establish a benefit for every decoder.

### Rejected silence-confidence experiment

A development probe found that FLARE's first-token EOS probability separated
20 speech clips from 27 short quiet controls. An EOS veto at probability 0.01
was fixed before measuring 60 other GRID speech clips and 53 new annotated
quiet regions from them. It retained all 60 speech clips, rejected all 53
quiet regions, and retained all nine natural-source test clips.

However, **the rule failed a duration stress test**. Extending each of the 53
quiet regions to 200 frames by repeating its final frame caused none to meet
the EOS veto. Two static-face controls also passed the EOS check. Combining
the veto with a motion threshold of 4 would admit 8/53 edge-padded controls.
These are synthetic stress inputs, not 55 new natural recordings; they expose
a duration dependence that the short-quiet comparison missed. The EOS veto
was rejected, not retuned against these results or shipped.

The existing INT8 CTC-evidence gate rejected all 53 fresh short quiet regions
and all 53 corresponding edge-padded controls in a separate host CPU probe.
Only 6 short and 8 padded regions passed its weak-motion check; CTC evidence
rejected the remainder. These probes use desktop-aligned ROIs sliced from
full-clip context, unlike the earlier separately prepared 27 regions. They
are neither physical-phone checks nor a general false-activation guarantee.

### Alternative-hypothesis selection

The unchanged beam-50 model returned 50 finite-scored hypotheses for each of
the eight natural development clips, with exact top-1 text parity and no
failures. Reference-selected best candidates within the top 5, 10 and 50
would leave 34, 32 and 26 errors out of 141 words, respectively, versus the
actual top-1 result of 46. These are **diagnostic oracle bounds**, not an
implemented reranker's accuracy. Even the top-50 bound cannot halve those
46 development errors through candidate selection alone.

A reference-free uniform word-edit consensus selector reduced development
errors to 43/141. Its rule was fixed before applying it to the nine existing
natural test clips: select the candidate with the lowest mean word-edit
distance to all 50 returned hypotheses, with visual-rank tie breaking. Test
errors rose from 34/148 to 38/148, so it was not adopted. Two posterior-weighted
development variants also failed to beat the uniform selector. This remains
one-source exploratory evidence, not a new independent speaker population.

## Portable FLARE research utilities

`tools/export_flare.py` exports the pinned Base encoder and next-token decoder
as FP32 ONNX opset-17 graphs. `tools/benchmark_flare.py` executes them using
ONNX Runtime CPU, without Torch or fairseq, and creates a private numeric fixture
for the Android benchmark in [ANDROID.md](ANDROID.md). Neither tool changes
the installed application's model or bypasses its quiet-input checks.

Use separate environments: the verified exporter used **Python 3.10**, Torch
2.7.1, fairseq 0.12.2, OmegaConf 2.0.6, Hydra 1.0.7, SentencePiece 0.2.2,
NumPy 1.26.4 and ONNX 1.20.1. Fairseq was built from its full v0.12.2 source
checkout, not the incomplete source distribution. Runtime checks used
**Python 3.12 and ORT 1.24.3**; that ORT release has no Python 3.10 wheel.
These legacy export dependencies are not added to the app's inference runtime.

Supply a trusted official FLARE checkout at revision
`7c8f76c788a01f9270a3badfd2eeab45a979c44a`, the checkpoint described above,
and a reference-free NPZ containing only `face` and `lip` uint8 arrays of equal
shape `[T,96,96]`, with 1–200 frames at 25 fps. Cropping and face tracking are
not supplied by these utilities; the preprocessing limitations above still
apply. Do not package non-commercial weights as unrestricted app assets.

```sh
# EXPORT_PY is the isolated Python 3.10 interpreter with the dependencies above.
"$EXPORT_PY" tools/export_flare.py \
  --checkpoint /path/flare-base-finetuned.pt --source /path/trusted-FLARE \
  --sample /path/visual-only.npz --output /path/flare-onnx \
  --device auto --accept-license

# RUNTIME_PY needs NumPy and ONNX Runtime, but no fairseq or Torch.
"$RUNTIME_PY" tools/benchmark_flare.py --model-dir /path/flare-onnx \
  --sample /path/visual-only.npz --output /path/android-flare-fixture
```

Outputs must be outside Git working trees and existing artifacts are not
overwritten. The exporter verifies the fixed checkpoint's size and SHA-256
before source import or deserialization. The source checkout is trusted
executable code; graph hashes establish file consistency, not independent
authenticity. The runtime fixture directory is owner-only (`0700`) because
normalized face/lip pixels and generated token IDs can be sensitive.

The graphs use batch size one, dynamic frame/prefix axes, a 512-dimensional
encoder output and a 1,000-token vocabulary. Metadata records ordered symbols,
geometry, normalization, provenance and graph hashes. The decoder's causal mask
is constructed dynamically; a cached trace-time mask is not valid for longer
prefixes. Greedy decoding starts with EOS, excludes PAD and first-step EOS,
and **fails** if the model does not terminate within 128 iterations. Callers
must enforce the 200-frame bound. Padding and exact-boundary parity remain
unverified; this is not a streaming or app-ready decoder.

On eight real development inputs, the exported graphs matched all complete
Torch greedy token sequences and all 24 next-token argmax checks at prefix
lengths 1, 5 and 10. Maximum encoder difference was 0.002622; maximum next-token
log-probability difference was 0.001739. Greedy WER was **55/141 = 39.01%**,
not the beam-50 model's 46/141. On a Ryzen 9 8945HS with four ORT threads,
encoder time was 0.92–1.42 seconds per clip and full-prefix decoder time was
0.25–0.99 seconds, excluding graph loading and preprocessing. The decoder
recomputes its prefix on every step; these are not phone speed measurements.

### Incremental decoder export

Add `--decoder-mode cached` to the export command to produce `encoder.onnx`,
`cross.onnx` and `decoder-cache.onnx` instead of the full-prefix decoder.
`tools/benchmark_flare.py` detects the mode from metadata; its command is
unchanged. The default remains `full`, and a full-mode regression still matched
the independent Torch sequence after adding the cached path.

The cross graph projects all six decoder layers' attention keys and values
once. Each decoding step supplies the full prefix for positional indexing,
but fairseq processes only the newest token and returns updated self-attention
caches. Each key/value has shape `[1,4,length,192]`; the first self cache has
length zero. Cache geometry and all three graph sizes/hashes are recorded in
metadata. The same EOS/PAD rules and 128-step failure limit apply.

The published cached exporter was run against the trusted pristine source,
then the runtime matched complete independent Torch token sequences on all
eight development clips. This preserves greedy output, not the better beam-50
WER. The prototype also matched 24 next-token argmax checks at prefix lengths
1, 5 and 10, with maximum log-probability difference 0.001740. A decoder-only
masked-tail check passed; padded visual-encoder parity remains unverified.

On the Ryzen host, one 128-frame input took about 0.95 seconds for the encoder,
0.007 seconds for cross projections and 0.19 seconds for cached decoding,
versus about 0.45 seconds for full-prefix decoding in separate prototype
measurements. These are CPU internal-stage timings, excluding graph loads,
camera and preprocessing. See [ANDROID.md](ANDROID.md#cached-decoder-benchmark)
for actual native execution and its different performance envelope.
