"""Fetch an unchanged IDMT educational subset with bounded HTTP ZIP ranges.

Run from repository root: python scripts/prepare_idmt.py --subset --normalize
No third-party dependencies; normalized annotation content stays local.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import sys
import urllib.request
import wave
import zipfile
import zlib

URL = 'https://zenodo.org/records/7544164/files/IDMT-SMT-DRUMS-V2.zip'
RECORDINGS = ('RealDrum01_00', 'WaveDrum01_00', 'TechnoDrum01_00')
TRANSFER_BYTES = 0


def get_range(start: int | None, end: int) -> tuple[bytes, str]:
    global TRANSFER_BYTES
    value = f'bytes=-{end}' if start is None else f'bytes={start}-{end}'
    req = urllib.request.Request(URL + '?download=1&range=' + value.replace('=', ''), headers={'Range': value, 'User-Agent': 'GrooveMirror-educational-baseline/0.1'})
    with urllib.request.urlopen(req, timeout=90) as response:
        if response.status != 206:
            raise RuntimeError(f'Range unsupported: HTTP {response.status}; full download not attempted')
        content_range = response.headers.get('Content-Range', '')
        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', content_range)
        if not match:
            raise RuntimeError('Missing or invalid Content-Range')
        first, last, total = map(int, match.groups())
        if (start is not None and (first != start or last != end)) or (start is None and first != max(0, total - end)):
            raise RuntimeError('Server returned the wrong byte range')
        data = response.read(8_000_001)
        if len(data) > 8_000_000 or len(data) != last - first + 1:
            raise RuntimeError('Range response length invalid or exceeded 8 MB')
        TRANSFER_BYTES += len(data)
        return data, content_range


def directory() -> tuple[zipfile.ZipFile, str]:
    tail, content_range = get_range(None, 65557)
    pos = tail.rfind(b'PK\x05\x06')
    if pos < 0:
        raise RuntimeError('ZIP end record absent')
    fields = struct.unpack_from('<4s4H2LH', tail, pos)
    size, offset = fields[5:7]
    central, _ = get_range(offset, offset + size - 1)
    eocd = bytearray(tail[pos:])
    struct.pack_into('<L', eocd, 16, 0)
    return zipfile.ZipFile(io.BytesIO(central + eocd)), content_range


def extract_member(info: zipfile.ZipInfo) -> Path:
    relative = Path(info.filename)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe ZIP member path')
    target = Path('data/raw/idmt') / relative
    if target.exists():
        raw = target.read_bytes()
    else:
        header, _ = get_range(info.header_offset, info.header_offset + 29)
        if header[:4] != b'PK\x03\x04':
            raise RuntimeError('Invalid local ZIP header')
        name_len, extra_len = struct.unpack_from('<HH', header, 26)
        offset = info.header_offset + 30 + name_len + extra_len
        packed, _ = get_range(offset, offset + info.compress_size - 1)
        if info.compress_type not in (0, 8):
            raise RuntimeError('Only stored/deflated members supported')
        raw = zlib.decompress(packed, -15) if info.compress_type == 8 else packed
    if len(raw) != info.file_size or zlib.crc32(raw) != info.CRC:
        raise RuntimeError(f'ZIP member CRC/length verification failed: {info.filename}')
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_bytes(raw)
    print(json.dumps({'path': target.as_posix(), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}))
    return target


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize() -> None:
    sys.path.insert(0, str(Path('ml/src').resolve()))
    from groovemirror_ml.data import load_annotations
    rows = []
    for recording in RECORDINGS:
        stem = recording + '#MIX'
        audio = Path('data/raw/idmt/audio') / (stem + '.wav')
        xml = Path('data/raw/idmt/annotation_xml') / (stem + '.xml')
        with wave.open(str(audio), 'rb') as wav:
            rate, frames, channels, width = wav.getframerate(), wav.getnframes(), wav.getnchannels(), wav.getsampwidth()
        duration = frames / rate
        events = load_annotations(xml, duration_seconds=duration)
        annotation = Path('data/processed/idmt') / (recording + '.json')
        annotation.parent.mkdir(parents=True, exist_ok=True)
        source = {'dataset': 'IDMT-SMT-Drums', 'record_url': 'https://zenodo.org/records/7544164', 'version': '1.0.0', 'archive_member': 'annotation_xml/' + stem + '.xml', 'sha256': digest(xml)}
        document = {'recording_id': recording, 'source': source, 'events': events}
        annotation.write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')
        assert load_annotations(annotation, duration_seconds=duration) == events
        row = {'recording_id': recording, 'recording_group': recording.split('_')[0], 'split': 'development_smoke_only', 'audio_path': audio.as_posix(), 'raw_annotation_path': xml.as_posix(), 'annotation_path': annotation.as_posix(), 'audio_sha256': digest(audio), 'raw_annotation_sha256': digest(xml), 'annotation_sha256': digest(annotation), 'sample_rate': rate, 'frames': frames, 'channels': channels, 'sample_width_bytes': width, 'duration_seconds': duration, 'event_count': len(events), 'unique_onset_count': len({event['time'] for event in events}), 'instrument_counts': {instrument: sum(e['instrument'] == instrument for e in events) for instrument in ('kick', 'snare', 'hihat')}}
        rows.append(row)
    manifest = {'manifest_version': 'idmt-phase1-v1', 'source_record': 'https://zenodo.org/records/7544164', 'archive_url': URL, 'archive_bytes': 287075649, 'published_archive_md5': 'd2664b4c2aaa34b90ba2f57b389c5663', 'full_archive_md5_verified': False, 'license': 'CC-BY-NC-ND-4.0', 'usage': 'Local noncommercial educational evaluation; no source audio or derived annotations redistributed', 'selection': 'First numbered MIX recording in each of RealDrum01, WaveDrum01, TechnoDrum01; not representative benchmark or held-out test', 'recordings': rows}
    target = Path('data/manifests/idmt-phase1.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'manifest': target.as_posix(), 'recordings': rows}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--subset', action='store_true', help='Fetch three MIX WAV/XML pairs and original MATLAB examples')
    parser.add_argument('--normalize', action='store_true', help='Create local JSON annotations and public metadata manifest')
    parser.add_argument('--member', action='append', default=[], help='Exact additional archive member to fetch unchanged')
    args = parser.parse_args()
    if args.inspect or args.subset or args.member:
        archive, content_range = directory()
        listing = [{'name': i.filename, 'compressed_bytes': i.compress_size, 'bytes': i.file_size, 'offset': i.header_offset, 'crc32': f'{i.CRC:08x}', 'method': i.compress_type} for i in archive.infolist()]
        target = Path('data/source/idmt-archive-index.json')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({'url': URL, 'content_range': content_range, 'members': listing}, indent=2) + '\n', encoding='utf-8')
        names = list(args.member)
        if args.subset:
            names += [f'{folder}/{recording}#MIX.{suffix}' for recording in RECORDINGS for folder, suffix in [('audio', 'wav'), ('annotation_xml', 'xml')]]
            names += ['parseXMLAnnotations.m', 'importAudioAndAnnotations.m']
        for name in dict.fromkeys(names):
            extract_member(archive.getinfo(name))
        print(json.dumps({'member_count': len(listing), 'http_response_body_bytes_this_run': TRANSFER_BYTES}))
    if args.normalize:
        normalize()
    if not any((args.inspect, args.subset, args.member, args.normalize)):
        parser.print_help()


if __name__ == '__main__':
    main()
