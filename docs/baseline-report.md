# Phase 1 verified onset baseline

## Scope and implementation

Phase 1 baseline: inspect real IDMT data, load/validate WAV audio, mix channels by arithmetic mean, resample to 22,050 Hz, compute a librosa log-mel/spectral-flux onset baseline, match against annotations and save a report plus overlay plot. No classifier, full application, train/test split, DVC/MLflow deployment or later phase has been implemented.

The lead inspected the new source files (this workspace has no preexisting Git history), integrated packaging/configuration and independently ran tests and the real-data checks after the contributing agents froze their source.

Key implementation locations:

- [Annotation ingestion](../ml/src/groovemirror_ml/data/__init__.py).
- [Shared audio validation/preprocessing/onset detection](../ml/src/groovemirror_ml/features/audio.py).
- [Deterministic one-to-one event evaluation](../ml/src/groovemirror_ml/evaluation/onsets.py).
- [CLI and report/plot lineage](../ml/src/groovemirror_ml/cli.py).
- [Independent tests](../ml/tests/), [source preparation](../scripts/prepare_idmt.py), [real-data repeatability runner](../scripts/run_phase1_baselines.py).

## Dataset and protocol

Dataset: [IDMT-SMT-Drums, Zenodo 7544164](https://zenodo.org/records/7544164). See [dataset inspection and license notes](dataset-notes.md) for exact source formats and [the manifest](../data/manifests/idmt-phase1.json) for hashes and selection policy.

- Manifest version: `idmt-phase1-v1`.
- Manifest SHA256: `6fbfde74d7f5c8aa63d67c13a8db6adecd1ea82b0bd65e1355a4924c67b33726`.
- Subset: first numbered MIX recording in each of RealDrum01, WaveDrum01 and TechnoDrum01. All measured originals are mono PCM16, 44,100 Hz.
- Use: development smoke/evaluation only. These recordings are not held out and are not representative performance estimates.
- Normalized labels come from the actual inspected original XML, not synthetic annotation or guessed filename schemas. The original final XML event is retained; an apparent off-by-one in the supplied MATLAB example is documented separately.
- Exact equal annotation timestamps collapse to one instrument-agnostic onset; unequal timestamps, however close, remain separate. Prediction duplicates remain separate and compete one-to-one.
- Matching: maximum cardinality, then minimum total absolute timing error, via deterministic sparse ordered dynamic programming. Inclusive tolerance is 0.05 seconds, with only 1e-12 s arithmetic slack for nonzero tolerance.
- Signed errors are prediction minus reference; positive means late. No mean-error subtraction or recording-offset correction is applied. Empty timing statistics are null; zero denominators produce zero-valued precision/recall/F1.

## Effective configuration and environment

[Audio settings](../configs/audio.yaml): sample rate 22,050; hop length 256; FFT size 1,024; arithmetic mean channel mixing; no amplitude normalization; `soxr_hq` resampling, `fix=True`, `scale=False`; backtrack false. Maximum WAV size 100 MiB, duration 300 s; silence RMS threshold 1e-7. The report records full log-mel, decibel, spectral-flux and peak-picking parameters, not just these summary fields.

[Evaluation settings](../configs/evaluation.yaml): onset tolerance 0.05 s. Phase 4 on-time/early/late musical scoring is intentionally separate.

Verified environment: CPython 3.12.14, Windows-11-10.0.26200-SP0, x86-64 processor reported as `AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD`. Selected library versions: librosa 0.11.0, NumPy 2.5.4, SciPy 1.18.1, SoundFile 0.13.1, soxr 1.1.0, Matplotlib 3.11.2 and PyYAML 6.0.3. The [dependency snapshot](../requirements-phase1.lock.txt) includes the complete resolved noneditable environment; it is not a hash-verified universal lockfile.

Code/config provenance for all six verified real-data runs:

- Python package source SHA256: `362d2bf635be0cac1e93ad41001dd5a86e0859e883e13b0aac4713e80d49eed4` (canonical list of all Python source paths/hashes under the shared package).
- Effective audio/evaluation/detector configuration SHA256: `00a8c0e35001e969a52cc9c6fa0485e0c1fc0988b4582800d684482c56d809a8`.
- Code commit: **null**. Git status: **unversioned**. No Git repository, commit or remote has been created by the agents. Commit-based lineage remains a setup limitation; hashes are supplemental provenance, not a fabricated commit.

## Actual real-recording results

Lead command, executed successfully:

```bash
.venv/Scripts/python.exe scripts/run_phase1_baselines.py
```

The runner verifies original audio/XML and normalized JSON against manifest SHA256 values, starts a separate CLI process twice per recording, checks JSON/PNG artifacts, and compares metrics, onset arrays, effective configuration and lineage exactly. All six CLI processes exited 0; all three repeat comparisons passed.

| Recording | Duration s | Reference / detected | TP / FP / FN | Precision | Recall | F1 | Mean abs ms | Median abs ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RealDrum01_00 | 17.6824 | 86 / 49 | 49 / 0 / 37 | 1.0000 | 0.5698 | 0.7259 | 7.506 | 8.013 |
| WaveDrum01_00 | 21.3750 | 129 / 93 | 93 / 0 / 36 | 1.0000 | 0.7209 | 0.8378 | 12.764 | 12.733 |
| TechnoDrum01_00 | 19.1250 | 133 / 97 | 97 / 0 / 36 | 1.0000 | 0.7293 | 0.8435 | 10.198 | 10.344 |

Reported analysis timer first/repeat: RealDrum 2.119/2.079 s, WaveDrum 2.153/2.120 s, TechnoDrum 2.100/2.144 s. These are local single-recording internal analysis timings, excluding interpreter/module startup and final report publication. They are not the 30-second-recording service latency acceptance benchmark.

Generated evidence stays local and is ignored by Git. Run the repeatability command above to recreate these files after preparing the dataset; they are not downloadable repository attachments.

- Summary: `artifacts/verified-phase1/summary.json`.
- RealDrum: `artifacts/verified-phase1/RealDrum01_00/report.json` and `artifacts/verified-phase1/RealDrum01_00/onsets.png`.
- WaveDrum: `artifacts/verified-phase1/WaveDrum01_00/report.json` and `artifacts/verified-phase1/WaveDrum01_00/onsets.png`.
- TechnoDrum: `artifacts/verified-phase1/TechnoDrum01_00/report.json` and `artifacts/verified-phase1/TechnoDrum01_00/onsets.png`.

The lead opened the RealDrum overlay and verified its readable three-panel waveform, spectral flux and reference/detected timelines. The baseline agent separately ran and inspected its own RealDrum evidence; those artifacts remain separate.

## Tests and actual verification history

Final independent lead command:

```bash
mkdir -p .pytest_tmp
.venv/Scripts/python.exe -m pytest ml/tests -q --basetemp=.pytest_tmp/lead-final --tb=short
.venv/Scripts/python.exe -m ruff check ml/src ml/tests scripts
.venv/Scripts/python.exe -m pip check
```

Actual results: **114 passed in 5.26 s**; **All checks passed**; **No broken requirements found**. Independent QA final: 114 passed in 5.44 s with a separate fixture directory; baseline agent: 114 passed in 5.45 s. Synthetic fixtures test contracts/algorithms, not real-world DSP quality. The one-to-one matcher includes exhaustive tiny-case oracle comparison and ambiguous-cardinality/minimum-error regressions.

Failures encountered and resolved:

1. Initial agent/lead Git inspection exited 128 because this is not a Git repository; no Git mutation was performed.
2. Agent test discovery initially preceded completion of the implementation modules. Collection errors were rerun after implementation, rather than reported as successful tests.
3. Host default temporary-directory ACL errors affected early fixture creation. Agents used separate workspace-owned fixture locations without changing external directory permissions.
4. The lead's first independent fixture run omitted the parent of `.pytest_tmp/lead-final`: 45 passed and 69 setup errors. Creating `.pytest_tmp` resolved this; the final full command passed all 114 tests. The chained lint/dependency checks did not execute on that failed attempt.
5. QA caught a missing plot hop-length argument in a synthetic CLI smoke run (104 passed, 1 failed). The implementation owner fixed it; later CLI/full-suite regressions passed.
6. Review caught a distribution metadata name mismatch (`groovemirror-ml` versus the actual `groovemirror` distribution); corrected and covered by a version/lineage assertion.
7. A lead lint check ran during QA's test edits and observed one temporary unused import; the frozen source/tests/scripts all pass final lint.

## Limitations

- Recall is limited on these clips. Closely spaced, unequal instrument annotations can outnumber acoustically distinct peaks within a frame/peak-picking interval. The current exact-only reference policy is explicit; no post-hoc timestamp merging or tolerance changes were used to improve the reported scores.
- Only three development recordings were measured. No held-out instrument-onset macro F1, generalization claim, 99% job-success claim or production-service latency target is established.
- Frame-grid resolution, padding, peak picking and recording boundaries affect timing. Strength is spectral flux, not a calibrated instrument-confidence probability.
- Raw/derived dataset payloads and local plots are not committed or automatically published. CC BY-NC-ND terms, the archive's observed-versus-advertised count discrepancy, and unverified full-archive MD5 are documented in dataset notes.
- There is no code commit lineage until the owner establishes Git history; original/source/config/environment hashes are recorded in the meantime.
