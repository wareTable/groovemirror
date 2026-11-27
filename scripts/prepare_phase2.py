"""Verify official IDMT archive, materialize MIX pairs, and rebuild split metadata."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import tempfile
from urllib.request import urlopen
import wave
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from groovemirror_ml.data import INSTRUMENTS, load_annotations
from groovemirror_ml.data.phase2 import (
    DEVELOPMENT_GROUPS,
    MANIFEST_VERSION,
    RECORDING_PATTERN,
    SPLITS,
    XML_RECORDING_ALIASES,
    assign_splits,
    load_split_manifest,
)

RECORD_URL = 'https://zenodo.org/records/7544164'
ARCHIVE_NAME = 'IDMT-SMT-DRUMS-V2.zip'
ARCHIVE_URL = f'https://zenodo.org/api/records/7544164/files/{ARCHIVE_NAME}/content'
ARCHIVE_SIZE = 287_075_649
ARCHIVE_MD5 = 'd2664b4c2aaa34b90ba2f57b389c5663'
RAW = Path('data/raw')
LOCAL_IDMT = RAW / 'idmt'
MEMBER_ID = re.compile(rf'({RECORDING_PATTERN})#(?:MIX|(?:HH|KD|SD)(?:#train)?)\.wav')
SOURCE_TYPES = {'RealDrum': 'acoustic', 'WaveDrum': 'sample_library', 'TechnoDrum': 'synthesizer'}
# Version 1 uses the existing strict annotation loader without repairing raw labels.
KNOWN_REJECTIONS = {'WaveDrum01_08': ('duplicate_exact_event', 'Duplicate annotation event'), **{
    f'WaveDrum02_{n:02}': ('negative_onset', 'outside recording duration')
    for n in (5, 8, 11, 14, 23, 26, 29, 32)
}}


def verify_archive(path: Path) -> None:
    if path.stat().st_size != ARCHIVE_SIZE:
        raise ValueError(f'{path}: wrong archive size (expected {ARCHIVE_SIZE})')
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle, lambda: hashlib.md5(usedforsecurity=False)).hexdigest()
    if digest != ARCHIVE_MD5:
        raise ValueError(f'{path}: wrong official Zenodo archive MD5')


def download_archive(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        candidate = Path(tmp) / ARCHIVE_NAME
        with urlopen(ARCHIVE_URL, timeout=90) as response, candidate.open('wb') as handle:
            left = ARCHIVE_SIZE + 1
            while left and (part := response.read(min(left, 1024 * 1024))):
                handle.write(part)
                left -= len(part)
        verify_archive(candidate)
        # Existing sources are checked, not replaced. The temporary file is on the same volume.
        path.hardlink_to(candidate)


def materialize(zipped: ZipFile, member: str, root: Path) -> tuple[Path, str]:
    raw = zipped.read(member)  # ZipFile verifies the member CRC on full read.
    path = root / LOCAL_IDMT / member
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('xb') as handle:
            handle.write(raw)
    except FileExistsError:
        if path.read_bytes() != raw:
            raise ValueError(f'Existing source differs from verified archive: {path}') from None
    return path, hashlib.sha256(raw).hexdigest()


def wav_identity(name: str) -> tuple[str, str]:
    match = MEMBER_ID.fullmatch(name)
    if not match:
        raise ValueError(f'Unexpected WAV identity {name!r}')
    return match[1], match[2]


def read_lineage(path: Path, wav_name: str, wave_names: set[str]) -> tuple[str, list[str], set[str], bool]:
    text = path.read_text(encoding='utf-8-sig')
    if '<!DOCTYPE' in text.upper() or '<!ENTITY' in text.upper():
        raise ValueError(f'{path}: XML declarations not supported')
    xml = ET.fromstring(text)
    param = xml.findall('globalParameter')
    if xml.tag != 'instrumentRecording' or len(param) != 1:
        raise ValueError(f'{path}: expected one IDMT globalParameter')
    fields = defaultdict(list)
    for element in param[0]:
        if len(element) or not (element.text or '').strip():
            raise ValueError(f'{path}: non-scalar or empty globalParameter field')
        fields[element.tag].append(element.text.strip())
    singleton = {'audioFileName', 'recordingArtist', 'mixGroundTrouthAvailable',
                 'mixGroundTrouthHasInterference'}
    if set(fields) != singleton | {'audioTrainingFileName'} or any(len(fields[k]) != 1 for k in singleton):
        raise ValueError(f'{path}: unexpected lineage metadata, review before splitting')
    if any(fields[key][0] not in {'true', 'false'} for key in singleton if key.startswith('mix')):
        raise ValueError(f'{path}: unexpected mix ground truth flag')
    declared = fields['audioFileName'][0]
    references = fields['audioTrainingFileName']
    if len(references) != 3 or {r.partition('#')[2] for r in references} != {
        'HH#train.wav', 'KD#train.wav', 'SD#train.wav'
    }:
        raise ValueError(f'{path}: expected one training reference per instrument')
    declared_id, declared_group = wav_identity(declared)
    expected_id, _ = wav_identity(wav_name)
    alias = declared != wav_name
    # Only the reviewed external alias is accepted; the declared members are absent.
    if alias and not (declared.endswith('#MIX.wav') and XML_RECORDING_ALIASES.get(declared_id) == expected_id
                      and declared not in wave_names):
        raise ValueError(f'{path}: unexpected audioFileName {declared!r}')
    resolved = []
    for name in references:
        recording_id, _ = wav_identity(name)
        if alias and recording_id == declared_id:
            if name in wave_names:
                raise ValueError(f'{path}: unexpected existing alias {name!r}')
            name = expected_id + '#' + name.partition('#')[2]
        if name not in wave_names:
            raise ValueError(f'{path}: absent training reference {name!r}')
        resolved.append(name)
    groups = {declared_group, *(wav_identity(n)[1] for n in references + resolved)}
    return fields['recordingArtist'][0], resolved, groups, alias


def build_manifest(archive: Path, root: Path, *, normalize: bool = False) -> dict:
    verify_archive(archive)
    records = []
    links: dict[str, set[str]] = defaultdict(set)
    hashes: dict[str, list[str]] = defaultdict(list)
    artists: dict[str, set[str]] = defaultdict(set)
    aliases = []
    rejected = []

    def link(groups: set[str]) -> None:
        for group in groups:
            links[group].update(groups)

    with ZipFile(archive) as zipped:
        names = zipped.namelist()
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError('Duplicate ZIP member paths')
        wave_names = {n.removeprefix('audio/') for n in names if n.startswith('audio/') and n.endswith('.wav')}
        wave_stems = {n.removesuffix('.wav') for n in wave_names if n.endswith('#MIX.wav')}
        xml_stems = {n.removeprefix('annotation_xml/').removesuffix('.xml') for n in names
                     if n.startswith('annotation_xml/') and n.endswith('.xml')}
        if wave_stems != xml_stems:
            raise ValueError('Expected equal sets of MIX WAV and annotation XML stems')
        # Hash all advertised WAV variants, not just the MIX inputs; shared bytes in
        # training stems/isolated sources also link the corresponding MIX groups.
        for name in sorted(wave_names):
            _, group = wav_identity(name)
            link({group})
            with zipped.open(f'audio/{name}') as handle:
                hashes[hashlib.file_digest(handle, 'sha256').hexdigest()].append(name)
        hash_by_name = {name: sha for sha, members in hashes.items() for name in members}
        for members in hashes.values():
            link({wav_identity(n)[1] for n in members})
        for stem in sorted(wave_stems):
            wav_name = f'{stem}.wav'
            rid, source_group = wav_identity(wav_name)
            wav_member, xml_member = f'audio/{wav_name}', f'annotation_xml/{stem}.xml'
            audio, audio_sha = materialize(zipped, wav_member, root)
            xml, xml_sha = materialize(zipped, xml_member, root)
            if audio_sha != hash_by_name[wav_name]:
                raise ValueError(f'{wav_name}: inconsistent archive hash')
            artist, sources, lineage, alias = read_lineage(xml, wav_name, wave_names)
            artists[source_group].add(artist)
            link({source_group, *lineage, *(wav_identity(n)[1] for n in sources)})
            if alias:
                aliases.append(rid)
            with wave.open(str(audio), 'rb') as handle:
                if (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) != (1, 2, 44_100):
                    raise ValueError(f'{audio}: expected mono PCM16 44,100 Hz WAV')
                frames = handle.getnframes()
                if len(handle.readframes(frames)) != frames * 2:
                    raise ValueError(f'{audio}: short PCM payload')
            try:
                events = load_annotations(xml, duration_seconds=frames / 44_100)
            except ValueError as exc:
                expected = KNOWN_REJECTIONS.get(rid)
                if expected is None or expected[1] not in str(exc):
                    raise ValueError(f'{rid}: {exc}') from exc
                rejected.append({'recording_id': rid, 'reason': expected[0],
                                 'raw_annotation_sha256': xml_sha, 'audio_sha256': audio_sha})
                continue
            if rid in KNOWN_REJECTIONS:
                raise ValueError(f'{rid}: expected annotation rejection did not occur')
            if normalize:
                out = root / 'data/processed/idmt/phase2' / f'{rid}.json'
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(json_bytes({'recording_id': rid, 'raw_annotation_sha256': xml_sha,
                                            'events': events}))
            counts = Counter(event['instrument'] for event in events)
            source_type = SOURCE_TYPES[re.match(r'[A-Za-z]+', source_group)[0]]
            records.append({'recording_id': rid, 'source_group': source_group, 'source_type': source_type,
                            'audio_path': (LOCAL_IDMT / wav_member).as_posix(),
                            'raw_annotation_path': (LOCAL_IDMT / xml_member).as_posix(),
                            'audio_sha256': audio_sha, 'raw_annotation_sha256': xml_sha,
                            'frames': frames, 'sample_rate': 44_100,
                            'event_counts': {key: counts[key] for key in sorted(INSTRUMENTS)}})
    # Components are transitive over IDMT group prefixes, XML identities and content links.
    for row in records:
        connected = set()
        remaining = {row['source_group']}
        while remaining:
            group = remaining.pop()
            connected.add(group)
            remaining.update(links[group] - connected)
        row['source_groups'] = sorted(connected)
        row['recording_group'] = row['source_groups'][0]
    by_group = assign_splits(records)
    for row in records:
        row['split'] = by_group[row['recording_group']]
    return {
        'manifest_version': MANIFEST_VERSION,
        'source': {'record_url': RECORD_URL, 'dataset_version': '1.0.0', 'license': 'CC-BY-NC-ND-4.0',
                   'archive': ARCHIVE_NAME, 'archive_bytes': ARCHIVE_SIZE, 'archive_md5': ARCHIVE_MD5},
        'split_policy': {'target_percent': dict(zip(SPLITS, (70, 15, 15))),
                         'phase1_groups_excluded_from_test': sorted(DEVELOPMENT_GROUPS)},
        'audit': {'archive_members': len(names), 'wav_members_hashed': len(wave_names),
                  'matched_mix_pairs': len(wave_stems), 'excluded_pairs': rejected,
                  'duplicate_wav_hash_sets': sum(len(m) > 1 for m in hashes.values()),
                  'distinct_artist_counts_by_source_group': {g: len(v) for g, v in sorted(artists.items())},
                  'xml_alias_recording_ids': aliases},
        'recordings': records,
    }


def json_bytes(doc: dict) -> bytes:
    return (json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path, help='Root containing data/raw and data/processed')
    parser.add_argument('--manifest', required=True, type=Path, help='Output manifest (root-relative paths inside)')
    parser.add_argument('--download', action='store_true', help='Download full ZIP only if absent')
    parser.add_argument('--normalize', action='store_true', help='Write local normalized labels under data/processed')
    args = parser.parse_args()
    root = args.data_root.resolve()
    archive = root / RAW / ARCHIVE_NAME
    if args.manifest.resolve().is_relative_to((root / RAW).resolve()):
        parser.error('Manifest output must not be inside data/raw')
    if not archive.exists() and args.download:
        download_archive(archive)
    try:
        document = build_manifest(archive, root, normalize=args.normalize)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.manifest.parent) as tmp:
            candidate = Path(tmp) / 'manifest.json'
            candidate.write_bytes(json_bytes(document))
            load_split_manifest(candidate)
            candidate.replace(args.manifest)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'Preparation failed: {exc}\n')
    print(dict(sorted(Counter(row['split'] for row in document['recordings']).items())))


if __name__ == '__main__':
    main()
