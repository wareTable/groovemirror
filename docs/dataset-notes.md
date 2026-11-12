# Phase 1 dataset inspection and ingestion

## Scope and source

Only IDMT-SMT-Drums is used. This is an educational, local onset-baseline development subset, not a representative benchmark, training split, or held-out test set. No annotations were invented and no unrelated audio was substituted.

Official sources inspected during this run:

- [Zenodo record 7544164, version 1.0.0](https://zenodo.org/records/7544164), published 2023-01-17, DOI `10.5281/zenodo.7544164`.
- [Fraunhofer IDMT project page](https://www.idmt.fraunhofer.de/en/publications/datasets/drums.html).
- [CC BY-NC-ND 4.0 legal code](https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).

Attribution: Feliks Weber, Manuel Winges, Christian Dittmar, Daniel Gärtner, Fraunhofer IDMT. Reference: Christian Dittmar and Daniel Gärtner, *Real-time transcription and separation of drum recordings based on NMF decomposition*, DAFx-14, Erlangen, Germany, 2014.

## License and publication decision

Both current official pages explicitly specify **CC BY-NC-ND 4.0**; the project page says evaluation purpose. Section 2(a)(1)(A) permits reproducing/sharing original licensed material for noncommercial purposes. Section 2(a)(1)(B) permits producing/reproducing, **but not sharing**, adapted material for noncommercial purposes. Section 2(a)(4) permits necessary technical modifications. Educational use still needs to satisfy the noncommercial condition; education is not a blanket commercial-use grant.

For this Phase 1 run: original downloads remain immutable local files, resampling/onset analysis is local noncommercial evaluation, and generated normalized annotations remain local. The repository publishes code, source links, archive metadata and checksums, not source audio, source annotations, or normalized annotation payloads. Keep raw downloads under `data/raw/idmt/` and processed annotations under `data/processed/idmt/` ignored by Git. These local folders are recreated through the preparation commands below. Re-evaluate terms before publishing audio, altered recordings, annotation databases, derived dataset bundles, or commercial use. No general permission for commercial deployment is inferred.

## Archive inspection: observed versus advertised

The official ZIP is `IDMT-SMT-DRUMS-V2.zip`, **287,075,649 bytes** (287.1 MB decimal), above the preferred 250 MB download budget. Zenodo publishes MD5 `d2664b4c2aaa34b90ba2f57b389c5663`. The full archive was **not** downloaded, so that full-file MD5 is recorded as **unverified**.

HTTP byte ranges returned `206 Partial Content`. A 65,557-byte suffix locates EOCD; a 73,300-byte central directory then describes **942 members**. The [archive index](<../data/source/idmt-archive-index.json>) records every source member, compressed/uncompressed sizes, CRC32, compression method and source offset. Observed counts:

| Archive member group | Observed count |
|---|---:|
| `audio/*.wav` | 560 |
| `annotation_xml/*.xml` | 95 |
| `annotation_svl/*.svl` | 284 |
| MATLAB example/parser files | 3 |

**Important discrepancy:** the official web descriptions advertise 608 WAVs, 104 drum loops, 312 training files and 192 isolated reference tracks. The inspected current ZIP directory contains 560 WAVs and 95 XMLs instead. These are not silently reconciled or filled with fabricated files. Only three retrieved pairs are validated end-to-end here; archive membership inspection is not validation of every member's audio/annotations.

Observed archive folders: `audio`, `annotation_xml`, `annotation_svl`, plus `importAudioAndAnnotations.m`, `parseSVLAnnotations.m`, `parseXMLAnnotations.m` at archive root. Mixture filenames use `GROUP_NUMBER#MIX.wav`; XML has the same basename. MATLAB import code identifies track markers `MIX`, `HH`, `SD`, `KD`, with additional `#train` suffixes for training files. Related mixtures/training/isolated variants share group/item identity and should stay together in a future group-aware split. Phase 2 splitting is not implemented.

## Actual formats and parser decisions

Inspected original XML: local `data/raw/idmt/annotation_xml/RealDrum01_00#MIX.xml`, plus corresponding WaveDrum and TechnoDrum XML files. These source files are not distributed with the repository. Actual structure:

```xml
<instrumentRecording>
  <globalParameter>
    <audioFileName>RealDrum01_00#MIX.wav</audioFileName>
    <!-- artist, training filenames and mix flags also present -->
  </globalParameter>
  <transcription>
    <event>
      <pitch>0</pitch>
      <onsetSec>0.08585</onsetSec>
      <offsetSec>0.08585</offsetSec>
      <instrument>HH</instrument>
    </event>
  </transcription>
</instrumentRecording>
```

`onsetSec` is seconds; `KD` → `kick`, `SD` → `snare`, `HH` → `hihat`. All events are retained; pitch/offset are not used by the onset baseline. The original local MATLAB XML parser, `data/raw/idmt/parseXMLAnnotations.m` lines 25 to 38, uses `getLength-1`, which appears to omit its final event. Our parser does **not** copy that apparent off-by-one: the final RealDrum event is genuine XML at 17.2263 seconds, HH, and is retained.

A local SVL example, `data/raw/idmt/annotation_svl/RealDrum01_01#KD.svl`, was also downloaded and inspected. It is Sonic Visualiser XML: `sv/data/model` has `sampleRate="44100"`; `dataset/point` has integer `frame` and generic labels. Instrument identity comes from the per-instrument filename. Its model `end` is smaller than some actual point frames, so model extent should not replace audio duration validation. SVL is **not supported** by the Phase 1 loader; use the complete mixture XML rather than silently merging per-instrument SVLs.

### Public loader contract

[Implementation](<../ml/src/groovemirror_ml/data/__init__.py>): `from groovemirror_ml.data import load_annotations`.

```python
events = load_annotations(path, duration_seconds=17.682403628117914)
# list[dict]: [{"time": 0.08585, "instrument": "hihat"}, ...]
```

Supported inputs: original inspected `.xml` schema above, or `.json` object with required nonblank string `recording_id` and list `events`. Example normalized shape (illustrative only):

```json
{"recording_id":"example","events":[{"time":0.0,"instrument":"kick"}]}
```

- `duration_seconds` must be numeric, finite and strictly positive; bool is rejected.
- Times must be numeric, finite, nonnegative, and **<= duration_seconds**. JSON numeric strings and bool are rejected. XML decimal text is parsed as float then validated.
- Vocabulary is exactly lowercase `kick`, `snare`, `hihat`; raw XML accepts exactly `KD`, `SD`, `HH`.
- Exact duplicate `(float(time), instrument)` records raise `ValueError`. Distinct instruments at the same timestamp are preserved. No near-time merging, rounding, tolerance clustering or onset fabrication occurs.
- Return value is sorted by time, then instrument, with newly constructed dicts. Empty event lists are valid.
- Onset-only evaluation, not the loader, collapses identical timestamps across simultaneous instruments.
- Unknown formats, malformed JSON/XML, schema/value failures and XML DTD/entity declarations raise `ValueError`. Filesystem errors keep ordinary IO exception types.
- Extra JSON metadata is allowed; preparation adds source record/version/member and original XML SHA256.
- Raw XML declares audio filename, but caller-supplied audio duration drives loader validation; the preparation script pairs exact basenames. The current baseline CLI accepts normalized JSON only; XML support is in the data loader/preparation path. Matching an arbitrary audio/annotation pair remains the caller's responsibility.

## Retrieved real subset and lineage

The [public manifest](<../data/manifests/idmt-phase1.json>) contains source/version/license, source paths, original and normalized SHA256, measured WAV headers, event counts, and the explicit development-only selection policy.

| Recording ID | Duration seconds | Frames | kick/snare/hihat | Events |
|---|---:|---:|---:|---:|
| RealDrum01_00 | 17.682403628117914 | 779794 | 23 / 21 / 42 | 86 |
| WaveDrum01_00 | 21.375011337868482 | 942638 | 25 / 14 / 90 | 129 |
| TechnoDrum01_00 | 19.125011337868482 | 843413 | 26 / 14 / 93 | 133 |

All three measured originals are **44,100 Hz, mono, 16-bit PCM WAV**. Each has as many unique exact timestamps as events; real near-simultaneous instruments often have slightly different annotations. That distinction is preserved. This is not a claim that every dataset recording has the same format or event statistics.

The three compressed audio members total **3,710,076 bytes**; XMLs are a few KB compressed each. No full ZIP transfer occurred. Each extracted member was verified against central-directory CRC32 and uncompressed length; SHA256 was computed on original extracted bytes. The selected WAVs total 5,138,198 bytes uncompressed. Original MATLAB import/XML parsers and one SVL example were also fetched unchanged for inspection. No source audio transformation was saved.

## Reproduce and actual checks

[Preparation script](<../scripts/prepare_idmt.py>) uses Python's standard library only; no dataset-specific dependency installation is needed. Run from repository root with the project-local Python environment:

```bash
python scripts/prepare_idmt.py --inspect
python scripts/prepare_idmt.py --subset --normalize
# Offline regeneration after the source pairs exist:
python scripts/prepare_idmt.py --normalize
```

`--subset` gets exactly the three listed MIX WAV/XML pairs plus original MATLAB examples. Existing local files must match source directory CRC/length; mismatches error rather than overwrite. Range responses are bounded at 8 MB per request and checked for HTTP206, correct Content-Range and length; a server that ignores ranges produces an explicit error without intentional full download. ZIP64 and unsupported compression are not general-purpose supported features. The known archive uses ordinary deflate.

Actually executed:

1. Bounded range inspection, selective downloads and CRC/length checks: succeeded.
2. `python scripts/prepare_idmt.py --subset --normalize`: succeeded, three JSON/XML round-trip equality assertions, manifest written. Cached rerun transferred 138,857 response-body bytes (directory inspection only).
3. Dataset agent independently executed `.venv/Scripts/python.exe -m pytest ml/tests/test_annotations.py ml/tests/test_idmt_xml.py -q --basetemp=data/.tmp-pytest --tb=short`: **46 passed in 0.09 s**. QA owns the tests and separately reported the same 46-pass result with its own `--basetemp=ml/tests/.tmp-qa-data`. The dataset agent's scratch folder is excluded by [data ignore rules](<../data/.gitignore>).
4. Initial `git status --short` / `git rev-parse`: exit 128, workspace not a Git repository; no Git initialization, commit or push performed. No commit ID is asserted.

The measured onset results, repeatability checks and config/tolerance/dependency lineage for these three manifest pairs are in the [baseline report](<baseline-report.md>). No classification training or later-phase infrastructure was added.
