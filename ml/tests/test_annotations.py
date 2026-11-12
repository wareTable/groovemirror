"""Normalized synthetic annotation validation; no dataset schema is invented here."""
import json

import pytest

from groovemirror_ml.data import load_annotations


def test_sorted_simultaneous_events_and_duration_boundary(annotations_file):
    path = annotations_file([
        {"time": 1, "instrument": "snare"},
        {"time": 0, "instrument": "kick"},
        {"time": 1, "instrument": "hihat"},
    ])
    assert load_annotations(path, duration_seconds=1) == [
        {"time": 0.0, "instrument": "kick"},
        {"time": 1.0, "instrument": "hihat"},
        {"time": 1.0, "instrument": "snare"},
    ]


def test_empty_events_are_valid(annotations_file):
    assert load_annotations(annotations_file([]), duration_seconds=1) == []


@pytest.mark.parametrize("time", [-0.001, 1.001, float("nan"), float("inf"), -float("inf"), True, "0.5", None])
def test_invalid_event_times(annotations_file, time):
    with pytest.raises(ValueError):
        load_annotations(annotations_file([{"time": time, "instrument": "kick"}]), duration_seconds=1)


@pytest.mark.parametrize("instrument", ["tom", "hi-hat", "Kick", "", None, 1])
def test_invalid_instrument(annotations_file, instrument):
    with pytest.raises(ValueError):
        load_annotations(annotations_file([{"time": 0.5, "instrument": instrument}]), duration_seconds=1)


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf"), True])
def test_invalid_duration(annotations_file, duration):
    with pytest.raises(ValueError):
        load_annotations(annotations_file([]), duration_seconds=duration)


def test_exact_duplicate_annotation_rejected(annotations_file):
    event = {"time": 0.5, "instrument": "kick"}
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        load_annotations(annotations_file([event, event]), duration_seconds=1)


@pytest.mark.parametrize("payload", [[], {}, {"recording_id": "x"}, {"events": []},
    {"recording_id": "", "events": []}, {"recording_id": 7, "events": []},
    {"recording_id": "x", "events": {}},
    {"recording_id": "x", "events": [None]},
    {"recording_id": "x", "events": [{"time": 0.5}]},
    {"recording_id": "x", "events": [{"instrument": "kick"}]}])
def test_invalid_json_schema(tmp_path, payload):
    path = tmp_path / "synthetic-invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_annotations(path, duration_seconds=1)


def test_malformed_json(tmp_path):
    path = tmp_path / "synthetic-corrupt.json"
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError):
        load_annotations(path, duration_seconds=1)
