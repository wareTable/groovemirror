"""Synthetic XML follows the inspected IDMT instrumentRecording schema.

Original local XML was inspected for field names; tests need no downloaded audio.
"""
import pytest

from groovemirror_ml.data import load_annotations


def document(events):
    return ('<?xml version="1.0" encoding="UTF-8"?>'
            '<instrumentRecording><globalParameter><audioFileName>synthetic.wav</audioFileName>'
            '</globalParameter><transcription>' + events + '</transcription></instrumentRecording>')


def event(time, instrument):
    return f'<event><pitch>0</pitch><onsetSec>{time}</onsetSec><offsetSec>{time}</offsetSec><instrument>{instrument}</instrument></event>'


def load_xml(tmp_path, text):
    path = tmp_path / 'synthetic-idmt.xml'
    path.write_text(text, encoding='utf-8')
    return load_annotations(path, duration_seconds=1)


def test_idmt_xml_maps_codes_and_preserves_simultaneous(tmp_path):
    result = load_xml(tmp_path, document(event(0.5, 'SD') + event(0, 'KD') + event(0.5, 'HH')))
    assert result == [{'time': 0.0, 'instrument': 'kick'}, {'time': 0.5, 'instrument': 'hihat'},
                      {'time': 0.5, 'instrument': 'snare'}]


@pytest.mark.parametrize('body', [event(-0.1, 'KD'), event(1.1, 'KD'), event('NaN', 'SD'),
    event('Infinity', 'HH'), event(0.5, 'TOM'), event(0.5, 'KD') * 2,
    '<event><instrument>KD</instrument></event>',
    '<event><onsetSec>0.1</onsetSec></event>'])
def test_idmt_xml_invalid_events(tmp_path, body):
    with pytest.raises(ValueError):
        load_xml(tmp_path, document(body))


@pytest.mark.parametrize('text', ['<wrong/>', '<instrumentRecording>',
    '<instrumentRecording><transcription/></instrumentRecording>',
    '<!DOCTYPE instrumentRecording [<!ENTITY x "value">]><instrumentRecording/>'])
def test_idmt_xml_invalid_structure(tmp_path, text):
    with pytest.raises(ValueError):
        load_xml(tmp_path, text)
