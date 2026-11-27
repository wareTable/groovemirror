"""Manifest validation and deterministic source-group splitting for IDMT MIX data."""
from collections import Counter, defaultdict
from itertools import product
import json
from pathlib import Path, PureWindowsPath
import re

from . import INSTRUMENTS

MANIFEST_VERSION = 'idmt-phase2-v1'
SPLITS = ('train', 'val', 'test')
DEVELOPMENT_GROUPS = frozenset({'RealDrum01', 'WaveDrum01', 'TechnoDrum01'})
# External archive names for this exact lineage correction are authoritative.
XML_RECORDING_ALIASES = {'RealDrum02_00': 'RealDrum01_13'}
PROTECTED_LINEAGE = DEVELOPMENT_GROUPS | {'RealDrum02'}
GROUP_PATTERN = r'(?:RealDrum|WaveDrum|TechnoDrum)[0-9]+'
RECORDING_PATTERN = rf'({GROUP_PATTERN})_[0-9]+'
SHA256_PATTERN = r'[0-9a-f]{64}'


def _relative_path(value: object, suffix: str) -> bool:
    return (isinstance(value, str) and '\\' not in value and '\x00' not in value
            and ':' not in value and all(part not in {'', '.', '..'} for part in value.split('/'))
            and not any(PureWindowsPath(part).is_reserved() or part.endswith((' ', '.'))
                        for part in value.split('/'))
            and value.endswith(suffix))


def load_split_manifest(path: str | Path) -> dict:
    """Read and validate metadata; caller resolves and checks files under its data root."""
    doc = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(doc, dict) or doc.get('manifest_version') != MANIFEST_VERSION:
        raise ValueError(f'Expected manifest_version {MANIFEST_VERSION!r}')
    records = doc.get('recordings')
    if not isinstance(records, list) or not records:
        raise ValueError('Manifest recordings must be a nonempty list')
    used = {key: set() for key in ('recording_id', 'audio_path', 'raw_annotation_path')}
    assignments: dict[tuple[str, str], str] = {}
    component_groups: dict[str, tuple[str, ...]] = {}
    component_owner: dict[str, str] = {}
    for row in records:
        if not isinstance(row, dict):
            raise ValueError('Each recording must be an object')
        for key in ('recording_id', 'recording_group', 'split'):
            if not isinstance(row.get(key), str) or not row[key] or row[key] != row[key].strip():
                raise ValueError(f'Invalid {key}')
        rid, group, split = (row[key] for key in ('recording_id', 'recording_group', 'split'))
        if split not in SPLITS:
            raise ValueError(f'{rid}: invalid split {split!r}')
        for key, suffix in (('audio_path', '.wav'), ('raw_annotation_path', '.xml')):
            if not _relative_path(row.get(key), suffix):
                raise ValueError(f'{rid}: invalid relative {key}')
        for key in used:
            value = row[key].casefold()
            if value in used[key]:
                raise ValueError(f'{rid}: duplicate {key}')
            used[key].add(value)
        for key in ('audio_sha256', 'raw_annotation_sha256'):
            if not isinstance(row.get(key), str) or not re.fullmatch(SHA256_PATTERN, row[key]):
                raise ValueError(f'{rid}: invalid {key}')
        source_match = re.fullmatch(RECORDING_PATTERN, rid)
        source_groups = row.get('source_groups', [group])
        if (not isinstance(source_groups, list) or not source_groups
                or any(not isinstance(g, str) or not re.fullmatch(GROUP_PATTERN, g)
                       for g in source_groups)
                or sorted(set(source_groups)) != source_groups or group != source_groups[0]
                or not source_match or source_match[1] not in source_groups):
            raise ValueError(f'{rid}: invalid source-group lineage')
        for key, directory, suffix in (('audio_path', 'audio', 'wav'),
                                       ('raw_annotation_path', 'annotation_xml', 'xml')):
            if row[key] != f'data/raw/idmt/{directory}/{rid}#MIX.{suffix}':
                raise ValueError(f'{rid}: {key} does not match recording identity')
        if component_groups.setdefault(group, tuple(source_groups)) != tuple(source_groups):
            raise ValueError(f'{rid}: inconsistent source-group component')
        if rid in XML_RECORDING_ALIASES.values() and 'RealDrum02' not in source_groups:
            raise ValueError(f'{rid}: missing known XML alias group')
        if split == 'test' and PROTECTED_LINEAGE.intersection(source_groups):
            raise ValueError(f'{rid}: Phase 1 lineage in test')
        for source_group in source_groups:
            if component_owner.setdefault(source_group, group) != group:
                raise ValueError(f'{rid}: source group in overlapping components')
        for kind, value in [('audio hash', row['audio_sha256']),
                            ('annotation hash', row['raw_annotation_sha256']),
                            ('source group', group), *(('source group', g) for g in source_groups)]:
            if assignments.setdefault((kind, value), split) != split:
                raise ValueError(f'{rid}: {kind} crosses splits')
    if {row['split'] for row in records} != set(SPLITS):
        raise ValueError('Train, val and test must all be nonempty')
    return doc


def assign_splits(records: list[dict]) -> dict[str, str]:
    """Exhaustively rank whole-component assignments using only metadata and labels."""
    grouped = defaultdict(list)
    for row in records:
        grouped[row['recording_group']].append(row)
    groups = sorted(grouped)
    blocked = {g for g in groups if any(PROTECTED_LINEAGE.intersection(row['source_groups'])
                                      for row in grouped[g])}
    counts = {g: len(grouped[g]) for g in groups}
    types = {g: {r['source_type'] for r in grouped[g]} for g in groups}
    classes = {g: {c for r in grouped[g] for c, n in r['event_counts'].items() if n}
               for g in groups}
    total = len(records)
    best = None
    for choice in product(SPLITS, repeat=len(groups)):
        if set(choice) != set(SPLITS) or any(g in blocked and s == 'test'
                                            for g, s in zip(groups, choice)):
            continue
        size = Counter()
        coverage = {s: set() for s in SPLITS}
        train_types = set()
        for g, s in zip(groups, choice):
            size[s] += counts[g]
            coverage[s].update(classes[g])
            if s == 'train':
                train_types.update(types[g])
        # Keep classes learnable/tunable; don't carve out individual loops for ratio or class balance.
        # Prefer all acquisition types in training before optimizing the recording-count ratio.
        rank = (len(INSTRUMENTS - coverage['train']), len(INSTRUMENTS - coverage['val']),
                -len(train_types),
                sum((100 * size[s] - goal * total) ** 2
                    for s, goal in zip(SPLITS, (70, 15, 15))),
                len(INSTRUMENTS - coverage['test']), choice)
        if best is None or rank < best:
            best = rank
    if best is None:
        raise ValueError('No nonempty split assignment can protect Phase 1 groups')
    return dict(zip(groups, best[-1]))
