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
