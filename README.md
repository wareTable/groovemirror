# GrooveMirror

This project is a drum rhythm coach for my ML/MLOps portfolio. We need to be able to rerun an analysis and check where the results came from, so that's part of what I'm working on here.

For now, GrooveMirror takes a WAV file through the command line. Phase 1 checks when the hits happen, and Phase 2 adds probabilities for kick, snare and hi-hat. The [onset baseline report](docs/baseline-report.md) has the Phase 1 results. The [instrument classifier guide](docs/phase2-classifier.md) covers the next set of runs and how to try it yourself. The deployed app is still planned.

## Getting it running

The setup was tested with CPython 3.12.14 on Windows 11 x86-64. I'd use Python 3.12 for now since that's the environment that's been checked. Other platforms and environments haven't been verified yet. Keep the dependencies in a project-local environment so they don't affect your other Python projects.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest ml/tests -q --basetemp=ml/tests/.tmp-local
.\.venv\Scripts\python.exe -m ruff check ml/src ml/tests scripts
```

On Linux/macOS, use `.venv/bin/python` instead of `.venv\Scripts\python.exe`.

The exact dependency versions from the Windows run are in [requirements-phase1.lock.txt](requirements-phase1.lock.txt). To use that same dependency snapshot in a fresh Python 3.12 environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-phase1.lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
```

This pins the environment used for that run. It still isn't a hash-verified lockfile for every platform.

## Getting a small dataset to work with

Read the [dataset and licensing notes](docs/dataset-notes.md) before downloading. The script gets three unchanged IDMT-SMT-Drums MIX WAV/XML pairs using bounded ZIP byte ranges, so it skips downloading the whole 287 MB archive. It checks CRC and length, then creates local normalized JSON and a manifest with the metadata and checksums.

```powershell
.\.venv\Scripts\python.exe scripts/prepare_idmt.py --subset --normalize
```

Once the original files are there, `--normalize` on its own works offline. The raw audio/XML and normalized annotations stay local and ignored by Git. The source metadata and manifest are kept separately.

## Running the baseline

```powershell
.\.venv\Scripts\python.exe -m groovemirror_ml "data/raw/idmt/audio/RealDrum01_00#MIX.wav" --annotations data/processed/idmt/RealDrum01_00.json --config configs/audio.yaml --evaluation-config configs/evaluation.yaml --output-dir artifacts/baseline/realdrum01_00
```

The installed `groovemirror-baseline` command does the same thing. The CLI takes normalized JSON annotations. The ingestion library and preparation script also support the original IDMT XML format that was inspected.

This gives you:

- `report.json`: onset times and strengths, matching metrics, signed timing errors, the effective configuration, file/source fingerprints and dependency versions.
- `onsets.png`: the waveform and spectral flux, with reference and detected events plotted together.

To check the input checksums and run all three real recordings twice:

```powershell
.\.venv\Scripts\python.exe scripts/run_phase1_baselines.py
```

It saves `artifacts/verified-phase1/summary.json` along with the reports and plots for each recording. The repeated runs need to agree exactly on the metrics, onsets, configuration and lineage.

## Results so far

114 tests passed. Ruff checks for the source, tests and scripts passed too, along with the dependency consistency check. The tests use synthetic fixtures. The real recordings have their own results below.

| Recording | Precision | Recall | Onset F1 | Matched median error |
|---|---:|---:|---:|---:|
| RealDrum01_00 | 1.0000 | 0.5698 | 0.7259 | 8.013 ms |
| WaveDrum01_00 | 1.0000 | 0.7209 | 0.8378 | 12.733 ms |
| TechnoDrum01_00 | 1.0000 | 0.7293 | 0.8435 | 10.344 ms |

These are only three selected development recordings, so I'd treat the scores as a starting point. They aren't held-out benchmark results. The evaluator ignores instrument class and matches events one-to-one within 50 ms. Only reference timestamps that are exactly equal get combined. Hits with close but different timestamps stay separate, which adds to the missed-event counts. Musical timing calibration and instrument-classification scores aren't part of this baseline.

## Where things are kept

- `ml/src/groovemirror_ml/`: the reusable annotation loader, audio processing, onset evaluation and CLI code.
- `ml/tests/`: unit and integration tests. Any generated audio here is synthetic.
- `configs/`: preprocessing and evaluation settings.
- `scripts/`: getting the selected source files and checking whether the real-data runs repeat properly.
- `docs/`: dataset notes and measured baseline results.
- `data/manifests/`, `data/source/`: source metadata and checksums. The audio and annotation payloads are kept elsewhere.
- `data/raw/`, `data/processed/`: local source recordings and normalized annotations, ignored by Git.
- `artifacts/`: generated reports, plots and scratch runs, also ignored by Git.

Additional work: the backend, frontend, DVC, MLflow and deployment are still planned for later phases.
