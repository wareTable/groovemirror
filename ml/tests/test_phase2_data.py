"""Synthetic metadata fixtures; no licensed WAVs/XML are needed for these tests."""
import hashlib
import json

import pytest

from groovemirror_ml.data.phase2 import MANIFEST_VERSION, assign_splits, load_split_manifest


def recording(rid, split, *, linked=None):
    source = rid.split('_')[0]
    lineage = sorted({source, *(linked or [])})
    return {
        'recording_id': rid,
        'recording_group': lineage[0],
        'source_groups': lineage,
        'split': split,
        'audio_path': f'data/raw/idmt/audio/{rid}#MIX.wav',
        'raw_annotation_path': f'data/raw/idmt/annotation_xml/{rid}#MIX.xml',
        'audio_sha256': hashlib.sha256(('wav' + rid).encode()).hexdigest(),
        'raw_annotation_sha256': hashlib.sha256(('xml' + rid).encode()).hexdigest(),
        'source_type': source.rstrip('0123456789'),
        'event_counts': {'kick': 1, 'snare': 1, 'hihat': 1},
    }


@pytest.fixture
def document():
    return {
        'manifest_version': MANIFEST_VERSION,
        'recordings': [recording('RealDrum01_00', 'train', linked=['RealDrum02']),
                       recording('WaveDrum01_00', 'val'), recording('TechnoDrum02_00', 'test')],
    }


def load(tmp_path, doc):
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(doc), encoding='utf-8')
    return load_split_manifest(path)


def test_load_manifest_preserves_metadata(tmp_path, document):
    assert load(tmp_path, document) == document


@pytest.mark.parametrize(('field', 'value', 'reason'), [
    ('recording_id', True, 'recording_id'),
    ('recording_group', '  ', 'recording_group'),
    ('source_groups', ['WaveDrum02'], 'lineage'),
    ('split', 'validation', 'split'),
    ('audio_sha256', 'a' * 63, 'audio_sha256'),
    ('raw_annotation_sha256', 'F' * 64, 'raw_annotation_sha256'),
    ('audio_path', '../corpus/a.wav', 'relative'),
    ('audio_path', 'C:/corpus/a.wav', 'relative'),
    ('audio_path', 'data\\audio.wav', 'relative'),
    ('audio_path', '/corpus/a.wav', 'relative'),
    ('audio_path', 'data/CON/a.wav', 'relative'),
    ('audio_path', 'data/. /a.wav', 'relative'),
    ('audio_path', 'data/other.wav', 'recording identity'),
    ('raw_annotation_path', 'data/a.json', 'relative'),
])
def test_invalid_record(tmp_path, document, field, value, reason):
    document['recordings'][0][field] = value
    with pytest.raises(ValueError, match=reason):
        load(tmp_path, document)


@pytest.mark.parametrize(('key', 'label'), [('audio_sha256', 'audio'), ('raw_annotation_sha256', 'annotation')])
def test_rejects_duplicate_bytes_across_splits(tmp_path, document, key, label):
    rows = document['recordings']
    rows[2][key] = rows[0][key]
    with pytest.raises(ValueError, match=f'{label} hash crosses splits'):
        load(tmp_path, document)
    rows[2]['split'] = 'train'
    rows.append(recording('WaveDrum02_00', 'test'))
    assert load(tmp_path, document) == document


def test_rejects_linked_phase1_lineage_in_test(tmp_path, document):
    document['recordings'][2] = recording('RealDrum02_00', 'test', linked=['RealDrum01'])
    with pytest.raises(ValueError, match='Phase 1 lineage in test'):
        load(tmp_path, document)


def test_rejects_source_groups_crossing_splits(tmp_path, document):
    document['recordings'].append(recording('RealDrum02_01', 'val', linked=['RealDrum01']))
    with pytest.raises(ValueError, match='source group crosses splits'):
        load(tmp_path, document)


def test_rejects_overlapping_component_across_splits(tmp_path, document):
    document['recordings'].append(recording('WaveDrum02_01', 'val', linked=['TechnoDrum02']))
    document['recordings'][2] = recording('TechnoDrum02_00', 'test', linked=['RealDrum03'])
    with pytest.raises(ValueError, match='overlapping components'):
        load(tmp_path, document)


def test_rejects_duplicates_and_empty_splits(tmp_path, document):
    recs = document['recordings']
    recs.append(recs[0].copy())
    with pytest.raises(ValueError, match='duplicate recording_id'):
        load(tmp_path, document)
    del recs[1:]
    with pytest.raises(ValueError, match='all be nonempty'):
        load(tmp_path, document)


def test_rejects_unknown_manifest_version(tmp_path, document):
    document['manifest_version'] = 'future'
    with pytest.raises(ValueError, match='manifest_version'):
        load(tmp_path, document)


def test_assignment_reproducible_and_honors_phase1_links(document):
    rows = document['recordings']
    rows += [recording('TechnoDrum01_00', 'unused'), recording('WaveDrum02_00', 'unused')]
    rows += [recording(f'WaveDrum02_{i:02}', 'unused') for i in range(1, 30)]
    assert assign_splits(rows) == assign_splits(rows[::-1])
    assignment = assign_splits(rows)
    assert assignment['RealDrum01'] == assignment['TechnoDrum01'] == 'train'
    assert assignment['WaveDrum02'] == 'train'
    assert assignment['WaveDrum01'] == 'val'
    assert assignment['TechnoDrum02'] == 'test'


def test_assignment_fails_instead_of_splitting_inseparable_groups(document):
    rows = document['recordings'][:2] + [recording('TechnoDrum02_00', 'test', linked=['TechnoDrum01'])]
    with pytest.raises(ValueError, match='No nonempty split'):
        assign_splits(rows)
