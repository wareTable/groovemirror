# Phase 2 instrument classifier

The pipeline compares two ways to label drum hits: a classical classifier and a small CNN. Both produce independent probabilities for kick, snare and hi-hat. A candidate onset can have multiple labels or none.

## Run it locally

Start in the project directory. These commands are for PowerShell with the environment from the [README](../README.md). The CPU PyTorch wheel is enough for this classifier. When using a different PyTorch build, record its version and device; do not assume CPU and GPU runs will give identical results.

```powershell
.\.venv\Scripts\python.exe -m pip install 'torch==2.15.0+cpu' --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e '.[dev,classifier]'
```

Follow the [dataset instructions](phase2-dataset.md) to prepare and checksum the original audio and annotation files. The source archive, extracted data, generated labels, model weights, and individual predictions remain local. The repository contains code and metadata only.

Set the thread counts for repeatable CPU training, then run with a new output directory:

```powershell
$env:OMP_NUM_THREADS='2'; $env:MKL_NUM_THREADS='2'; $env:OPENBLAS_NUM_THREADS='2'
.\.venv\Scripts\python.exe -m groovemirror_ml.phase2 train --manifest data/manifests/idmt-phase2-splits.json --data-root . --output artifacts/phase2-run --seed 0 --epochs 12 --batch-size 64
```

Training writes `classical.npz`, `cnn.npz`, `frozen.json` and its checksum in `frozen.sha256`. Both model files are local CPU-loadable numerical artifacts with no pickle loader. The frozen metadata records the selected candidate, artifact and manifest digests, preprocessing, tolerance, candidate selection rule, validation thresholds, training history, environment and source lineage. Each saved model is reloaded and checked against the original validation predictions.

For a prediction on a separate WAV, the classifier uses the Phase 1 onset detector and the same patch mapper used at training:

```powershell
.\.venv\Scripts\python.exe -m groovemirror_ml.phase2 predict --run artifacts/phase2-run --output artifacts/phase2-prediction.json path\to\recording.wav
```

Once the model, thresholds, code and preprocessing are frozen, record the printed SHA-256 separately. Pass that same digest into the holdout command. The evaluator checks the digest, artifact, data manifest and package source hashes before opening the held-out files. Keep the resulting report instead of overwriting it.

```powershell
.\.venv\Scripts\python.exe -m groovemirror_ml.phase2 evaluate --manifest data/manifests/idmt-phase2-splits.json --data-root . --run artifacts/phase2-run --frozen-sha256 '<recorded digest>' --output artifacts/phase2-run/heldout.json
```

## How the comparison works

- **Input.** Shared validation, mono conversion and resampling at 22,050 Hz; 40 mel bands × 24 frames, 1,024-point Hann FFT, 256-sample hop, 4 frame centers before and 20 from the event onward. Waveform edges are zero padded. Power is relative to 1.0 and floored at −100 dB. The patch transform fits nothing.
- **Targets.** One binary column per instrument, ordered kick, snare, hi-hat. Exactly equal reference timestamps share a patch and retain every label. Unequal timestamps stay separate, even if close.
- **Models.** The classical baseline fits class-balanced one-vs-rest logistic models to spectral summary features. The compact CNN has two convolutional blocks and three logits with independent sigmoid probabilities. Model normalization uses training patches only. CNN checkpoint selection uses validation loss.
- **Thresholds and selection.** For each model and class, maximize reference-patch validation F1 over thresholds 0.2 through 0.8 in steps of 0.1; ties prefer closest to 0.5, then higher. A validation class with only one target value keeps 0.5. Pick the model by detected-event validation macro F1, with classical winning ties. These outputs are model scores, not calibrated confidence estimates.
- **Two measurements.** Classification at known annotation times isolates the classifier. End-to-end event scoring first detects onsets in the WAV, then classifies those candidates. Per-instrument matches are one-to-one within 50 ms by default; matching stays within a recording. Report binary confusion counts, per-class precision/recall/F1, micro scores, and unweighted macro scores across all three classes. Absent denominators count as zero; inspect class support alongside the average. Signed timing is prediction minus reference in seconds.
- **Holdout.** The training command reads only train and validation audio and labels; it never scores a candidate on test. Only the selected frozen model is evaluated on test. A limited or unsupported test class cannot establish generalization.

## Results from the first CPU run

The checksum-verified `idmt-phase2-v1` manifest has 76 train, 9 validation and 1 test recording after excluding nine source annotation errors. See the [split audit](phase2-dataset.md) for why the source groups could not be split near 70/15/15 without compromising the experiment. The train/validation support counts below are event labels, not disjoint single-class patches.

| Model | Train patch macro F1 | Validation patch macro F1 | Validation event macro F1 | Validation event micro F1 | Thresholds: kick / snare / hi-hat |
| --- | ---: | ---: | ---: | ---: | --- |
| Classical | 0.5778 | 0.7542 | 0.8901 | 0.9033 | 0.5 / 0.7 / 0.3 |
| CNN (checkpoint 12) | 0.6657 | 0.7090 | 0.8691 | 0.8869 | 0.6 / 0.7 / 0.2 |

| Validation event class | Support | Classical precision / recall / F1 | CNN precision / recall / F1 |
| --- | ---: | --- | --- |
| Kick | 269 | 0.901 / 0.743 / 0.815 | 0.639 / 0.836 / 0.725 |
| Snare | 166 | 0.907 / 0.940 / 0.923 | 0.950 / 0.910 / 0.929 |
| Hi-hat | 639 | 0.945 / 0.920 / 0.933 | 0.931 / 0.977 / 0.953 |

The classical model won the predeclared comparison. The CNN caught more validation kicks (225 vs. 200), but also produced 127 extra kick predictions vs. 22 for the classical model. Hi-hat had 3,209 training references, compared with 1,691 kick and 1,194 snare. The models account for imbalance during fitting; the support and error counts still matter when interpreting the scores. Validation used a separate source group, but that group was already seen in Phase 1 development. It is not the test result.

The selected classical candidate was scored once on the held-out TechnoDrum02 clip at the frozen thresholds. End-to-end results:

| Test event class | Support | TP / FP / FN | Precision | Recall | F1 |
| --- | ---: | --- | ---: | ---: | ---: |
| Kick | 12 | 11 / 12 / 1 | 0.4783 | 0.9167 | 0.6286 |
| Snare | 8 | 0 / 0 / 8 | 0.0000 | 0.0000 | 0.0000 |
| Hi-hat | 0 | 0 / 31 / 0 | 0.0000 | 0.0000* | 0.0000 |
| Micro | 20 | 11 / 43 / 9 | 0.2037 | 0.5500 | 0.2973 |
| Macro | — | — | 0.1594 | 0.3056 | 0.2095 |

*Hi-hat recall is not measurable on this clip; the table uses the documented zero-denominator convention. Even at annotated times, all 8 snare labels were missed and all 20 patches received a false hi-hat label, while kick F1 was 1.0000. That shows part of the failure is classification, not just onset timing. The median absolute timing error was 11.61 ms across 11 matched events, all kicks; this is not a general timing result. The 0.75 held-out macro F1 target was not met. A single synth clip does not establish broader generalization. These results were not used to revise the candidate. Future work needs more independently sourced test groups with all three instruments.

### Reproduction evidence

Two training runs from clean commit `4b5409894f4b2aa2376fc367feaebd1b40a0f30b` on Windows with Python 3.12.14, PyTorch `2.15.0+cpu`, NumPy 2.5.4, librosa 0.11.0 and the thread settings above produced byte-identical artifacts and frozen metrics. Both artifacts reloaded with exactly equal validation predictions. This verifies repeatability on that setup, not across different devices or library versions.

| Local artifact | SHA-256 |
| --- | --- |
| `frozen.json` | `6909dd7c27da6c78fe765d98457f00d113069b1556a6b35d90d35be8fcf7a606` |
| `classical.npz` | `ce58cc80af64acefefb90f236587eae029db6da7a8bfd2dbb99177a50a4b756a` |
| `cnn.npz` | `9410fee5e582dc8f26b72ee62da9bf4e5a0bc687ed5a17bc1cffdab7a43f8dc5` |

## Prediction JSON (schema 1)

```json
{
  "schema_version": 1,
  "model_version": "cnn-sha256:<model digest>",
  "preprocessing_version": "logmel-v1+<configuration digest>",
  "events": [
    {"time_seconds": 1.25, "probabilities": {"kick": 0.91, "snare": 0.78, "hihat": 0.06}, "labels": ["kick", "snare"]}
  ]
}
```

The numbers are illustrative. For each event, compare each probability independently to its frozen threshold (inclusive). The timestamp is relative to the start of the original WAV. `frozen.json` carries the exact thresholds and DSP parameters behind the version strings.
