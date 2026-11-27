# Phase 2 IDMT mixture corpus and split

[Canonical manifest](../data/manifests/idmt-phase2-splits.json), schema `idmt-phase2-v1`. It contains source paths, metadata, aggregate label counts and checksums, not audio or annotation payloads. The split was assigned from IDs, lineage, source types, sizes and class support only. No model was trained and no predictions were inspected during preparation.

## Source and local preparation

Use [IDMT-SMT-Drums 1.0.0, Zenodo 7544164](https://zenodo.org/records/7544164) (DOI `10.5281/zenodo.7544164`) by Feliks Weber, Manuel Winges, Christian Dittmar, and Daniel Gärtner, Fraunhofer IDMT. Both the Zenodo record and [Fraunhofer project page](https://www.idmt.fraunhofer.de/en/publications/datasets/drums.html) specify [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode); the project page describes evaluation use. Follow [the project's licensing notes](dataset-notes.md): keep source files unchanged, keep derived labels local, and review permission before wider sharing or commercial use.

The full `IDMT-SMT-DRUMS-V2.zip` (287,075,649 bytes) was reconstructed from validated HTTP 206 ranges and matched Zenodo's full-file MD5 `d2664b4c2aaa34b90ba2f57b389c5663`. No partial-download verification claim substitutes for that checksum. The archive has 942 members, including 560 WAVs and 95 MIX XML files, not the advertised 608 WAVs and 104 loops. All 95 MIX WAV filenames have a matching XML filename and vice versa. All 95 original pairs were materialized locally; the other WAVs were read and hashed in the verified archive only.

From the checkout containing the code, with the project environment (PowerShell example):

```powershell
$env:PYTHONPATH = 'ml/src'
& C:/POLY/DrumProj/.venv/Scripts/python.exe scripts/prepare_phase2.py --data-root C:/POLY/DrumProj --manifest data/manifests/idmt-phase2-splits.json --download --normalize
```

`--data-root` is the directory **above** `data/`, not `data/raw/` or the worktree. Adjust the paths for your checkout. `--download` is optional; it fetches the archive only when absent. You can also put the official ZIP in `<root>/data/raw/` yourself. Run without `--download` to regenerate offline. The script checks the archive's full length and MD5, reads members with ZIP CRC checks, rejects modified cached copies instead of overwriting them, and writes unchanged pairs under `<root>/data/raw/idmt/`. `--normalize` optionally writes JSON for the included pairs under `<root>/data/processed/idmt/phase2/`; Phase 1 normalized files are untouched. Both locations are ignored. The ZIP and copies of individual WAVs/XML also remain local.

The Python boundary is `groovemirror_ml.data.phase2.load_split_manifest(path) -> dict`. It validates the version, ID and IDMT-relative paths, nonempty splits, SHA-256 syntax, unique IDs/paths, and separation of known lineage and duplicate hashes. It does not open audio or annotations. Resolve paths against the same data root and verify file hashes before running an experiment. Read the original XML with `groovemirror_ml.data.load_annotations(path, duration_seconds=...)`.

## Lineage audit

- Every loop, associated training WAV and isolated instrument WAV inherits its numbered IDMT source group (e.g. `WaveDrum02`). Whole source groups stay together. Across all 95 XML documents, `globalParameter` contains `audioFileName`, `recordingArtist`, two mix-ground-truth flags and three `audioTrainingFileName` references (one per instrument). There is no session, kit ID or more detailed acquisition field. Each declared training reference resolves to an archive member, with the one documented alias below.
- `RealDrum01_13` is named that way in its archive WAV, XML, three training WAVs and SVLs. Its XML names `RealDrum02_00` for the audio and all three training stems; none of those internally named files exists. The preparation code accepts this exact alias, checks the corresponding external filenames and unions `RealDrum01` with `RealDrum02`. No other mixture name disagreed. None of the other XML references crosses an IDMT source group.
- SHA-256 across all 560 WAV members finds five identical-byte sets: the three training stems reused between `RealDrum01_10` and `RealDrum01_11`, and the kick and snare isolated stems reused between `WaveDrum02_55` and `WaveDrum02_56`. All five sets are within source groups; no duplicate mixture or cross-group WAV hashes were found. The algorithm unions source groups connected by exact WAV bytes or XML identities, including transitive links.
- The same credited artist appears across different acquisition types and source groups. `RealDrum01` has two distinct credited artists including the alias; each other source group has one. An artist credit is not a session identifier and was not used to collapse those different sources. This is not artist-disjoint, and exact hashes do not detect transformed copies or guarantee distinct sample banks, performances or underlying rhythms. There is no independent kit/session inventory in the metadata. No obvious cross-split links were found under the checks above.

## Shape, exclusions and support

All 95 original mixtures are mono PCM16 at 44,100 Hz. The unchanged annotation loader rejected nine: `WaveDrum01_08` has an exact duplicate snare event; `WaveDrum02_05`, `_08`, `_11`, `_14`, `_23`, `_26`, `_29`, `_32` contain one or two negative labeled onsets (all within 0.205 ms before time zero). Version 1 deliberately excludes those pairs, retains their original bytes and audits their lineage before splitting. The manifest lists the excluded IDs, reason codes and hashes. No onset clipping, duplicate removal or label edits were made.

| Split | Source groups (usable loops) | Loops | % of 86 | Audio seconds | Kick / snare / hi-hat events |
| --- | --- | ---: | ---: | ---: | ---: |
| Train | RealDrum01 incl. alias (14); TechnoDrum01 (10); WaveDrum02 (52) | 76 | 88.37 | 1123.97 | 1691 / 1194 / 3209 |
| Validation | WaveDrum01 (9) | 9 | 10.47 | 169.45 | 269 / 166 / 639 |
| Test | TechnoDrum02 (1) | 1 | 1.16 | 7.62 | 12 / 8 / 0 |

The deterministic assignment searches whole-group allocations with nonempty splits and blocks the three Phase 1 development groups, plus linked identities, from test. It ranks by training then validation class coverage, then training source-type coverage, then squared distance to the target 70/15/15 by recording count; it breaks ties by test class coverage and lexical split order. The only eligible test groups are the 52 usable WaveDrum02 loops as a block or the one TechnoDrum02 loop. Using WaveDrum02 for test would put at least 60% of the corpus in test. A ratio-only alternative using the singleton test could move TechnoDrum01 to validation (66/19/1), leaving training without any synth recording. The chosen assignment preserves all three acquisition types in training and all three classes in training and validation. Validation contains a Phase 1 development group; it is not a new held-out benchmark.

The test split is one synth clip with zero hi-hat references. Report that support explicitly. This split does not establish held-out hi-hat recall or generalization to new acoustic sessions, artists, sample libraries, or the corpus as a whole. Stronger evaluation would need more independently documented, unexposed groups with all three classes, not a random split of WaveDrum02.
