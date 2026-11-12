"""Verify manifest inputs and run/repeat three real Phase 1 development baselines.

Run from the repository root after scripts/prepare_idmt.py --subset --normalize.
Outputs stay under ignored artifacts/. These are not held-out benchmark results.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts' / 'verified-phase1'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest_path = ROOT / 'data/manifests/idmt-phase1.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['manifest_version'] != 'idmt-phase1-v1' or len(manifest['recordings']) != 3:
        raise ValueError('Expected the documented three-recording development subset')
    summary = {
        'scope': 'phase1_development_smoke_only_not_held_out_benchmark',
        'manifest_version': manifest['manifest_version'],
        'manifest_sha256': sha256(manifest_path),
        'python': platform.python_version(),
        'platform': platform.platform(),
        'processor': platform.processor(),
        'recordings': [],
    }
    for row in manifest['recordings']:
        for path_key, hash_key in [('audio_path', 'audio_sha256'),
                                   ('raw_annotation_path', 'raw_annotation_sha256'),
                                   ('annotation_path', 'annotation_sha256')]:
            if sha256(ROOT / row[path_key]) != row[hash_key]:
                raise ValueError(f'Manifest checksum mismatch: {row[path_key]}')
        reports = []
        for suffix in ('', '-repeat'):
            target = OUT / (row['recording_id'] + suffix)
            command = [sys.executable, '-m', 'groovemirror_ml', row['audio_path'],
                       '--annotations', row['annotation_path'],
                       '--config', 'configs/audio.yaml',
                       '--evaluation-config', 'configs/evaluation.yaml',
                       '--recording-id', row['recording_id'],
                       '--output-dir', str(target)]
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
            print(result.stdout, end='')
            if result.returncode:
                raise RuntimeError(f'CLI exited {result.returncode}: {result.stderr}')
            report = json.loads((target / 'report.json').read_text(encoding='utf-8'))
            if not (target / 'onsets.png').read_bytes().startswith(bytes([137, 80, 78, 71])):
                raise ValueError('Expected PNG overlay artifact')
            if report['lineage']['audio']['sha256'] != row['audio_sha256']:
                raise ValueError('CLI source audio lineage differs from manifest')
            if report['lineage']['annotations']['sha256'] != row['annotation_sha256']:
                raise ValueError('CLI annotation lineage differs from manifest')
            if report['lineage']['software'].get('groovemirror') != '0.1.0':
                raise ValueError('Local package distribution version missing from report')
            reports.append(report)
        first, repeat = reports
        for key in ('metrics', 'onsets', 'configuration', 'lineage'):
            if first[key] != repeat[key]:
                raise ValueError(f'Repeated baseline differs in {key}: {row["recording_id"]}')
        entry = {
            'recording_id': row['recording_id'],
            'duration_seconds': row['duration_seconds'],
            'metrics': first['metrics'],
            'analysis_duration_seconds': first['analysis_duration_seconds'],
            'repeat_analysis_duration_seconds': repeat['analysis_duration_seconds'],
            'source_sha256': first['lineage']['code']['source_sha256'],
            'effective_config_sha256': first['configuration']['effective_sha256'],
            'git_commit': first['lineage']['code']['git_commit'],
            'git_status': first['lineage']['code']['git_status'],
            'repeated_metrics_onsets_configuration_lineage_equal': True,
        }
        summary['recordings'].append(entry)
        print(json.dumps({'verified': row['recording_id'], 'precision': first['metrics']['precision'],
                          'recall': first['metrics']['recall'], 'f1': first['metrics']['f1'],
                          'median_absolute_error_ms': None if first['metrics']['median_absolute_error_seconds'] is None
                          else 1000 * first['metrics']['median_absolute_error_seconds'],
                          'repeat_equal': True}))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(f'Verified summary: {OUT / "summary.json"}')


if __name__ == '__main__':
    main()
