"""Validated normalized JSON and inspected IDMT-SMT-Drums XML ingestion.

Exact duplicate (time, instrument) pairs are errors. Simultaneous distinct
instruments are retained; onset-only evaluation owns timestamp deduplication.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

INSTRUMENTS = frozenset({'kick', 'snare', 'hihat'})
IDMT_INSTRUMENTS = {'KD': 'kick', 'SD': 'snare', 'HH': 'hihat'}


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{label} must be a finite number')
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f'{label} must be a finite number') from exc
    if not math.isfinite(result):
        raise ValueError(f'{label} must be finite')
    return result


def _xml_events(text: str) -> list[dict]:
    # IDMT XML uses no DTD/entities. Reject these rather than expanding input.
    if '<!DOCTYPE' in text.upper() or '<!ENTITY' in text.upper():
        raise ValueError('IDMT XML must not contain DTD or entity declarations')
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ValueError(f'Invalid annotation XML: {exc}') from exc
    if root.tag != 'instrumentRecording':
        raise ValueError('Expected IDMT instrumentRecording XML root')
    audio_name = root.findtext('globalParameter/audioFileName')
    transcriptions = root.findall('transcription')
    if not audio_name or not audio_name.strip() or len(transcriptions) != 1:
        raise ValueError('IDMT XML requires audioFileName and one transcription')
    result = []
    for index, event in enumerate(transcriptions[0]):
        if event.tag != 'event':
            raise ValueError(f'Unexpected transcription element: {event.tag}')
        onsets, instruments = event.findall('onsetSec'), event.findall('instrument')
        if len(onsets) != 1 or len(instruments) != 1:
            raise ValueError(f'XML event {index} requires one onsetSec and instrument')
        code = (instruments[0].text or '').strip()
        if code not in IDMT_INSTRUMENTS:
            raise ValueError(f'XML event {index} unknown instrument {code!r}; expected KD/SD/HH')
        try:
            time = float(onsets[0].text or '')
        except ValueError as exc:
            raise ValueError(f'XML event {index} invalid onsetSec') from exc
        result.append({'time': time, 'instrument': IDMT_INSTRUMENTS[code]})
    return result


def load_annotations(path: str | Path, *, duration_seconds: float) -> list[dict]:
    """Load JSON {recording_id, events} or original IDMT .xml events.

    Times are seconds in the inclusive interval [0, duration_seconds]. Duration
    must be finite and positive. JSON numeric strings and booleans are rejected.
    Returns new dicts sorted by time then instrument. Other suffixes (including
    .svl) are unsupported; no annotation format is guessed from filenames.
    IO errors retain their standard exception types; content errors are ValueError.
    """
    duration = _number(duration_seconds, 'duration_seconds')
    if duration <= 0:
        raise ValueError('duration_seconds must be positive')
    source = Path(path)
    text = source.read_text(encoding='utf-8-sig')
    if source.suffix.lower() == '.json':
        try:
            document = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f'Invalid annotation JSON: {exc}') from exc
        if not isinstance(document, dict):
            raise ValueError('Annotations must be an object with recording_id and events')
        recording_id = document.get('recording_id')
        if not isinstance(recording_id, str) or not recording_id.strip():
            raise ValueError('recording_id must be a nonempty string')
        events = document.get('events')
        if not isinstance(events, list):
            raise ValueError('events must be a list')
    elif source.suffix.lower() == '.xml':
        events = _xml_events(text)
    else:
        raise ValueError(f'Unsupported annotation format {source.suffix!r}; expected .json or IDMT .xml')
    result = []
    seen = set()
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(f'Event {index} must be an object')
        time = _number(event.get('time'), f'Event {index} time')
        if time < 0 or time > duration:
            raise ValueError(f'Event {index} time {time} outside recording duration [0, {duration}]')
        instrument = event.get('instrument')
        if not isinstance(instrument, str) or instrument not in INSTRUMENTS:
            raise ValueError(f'Event {index} instrument must be kick, snare, or hihat')
        key = (time, instrument)
        if key in seen:
            raise ValueError(f'Duplicate annotation event {key!r}')
        seen.add(key)
        result.append({'time': time, 'instrument': instrument})
    return sorted(result, key=lambda event: (event['time'], event['instrument']))


__all__ = ['load_annotations', 'INSTRUMENTS']
