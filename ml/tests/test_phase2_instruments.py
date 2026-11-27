"""Synthetic contract checks, not measured instrument classification performance."""
import json

import numpy as np
import pytest

from groovemirror_ml.evaluation.instruments import (
    INSTRUMENTS,
    dataset_event_metrics,
    event_metrics,
    patch_metrics,
    prediction_document,
)


def document(times, probabilities, thresholds=(0.5, 0.5, 0.5)):
    return prediction_document(
        times, probabilities, thresholds, model_version='synthetic-v1', preprocessing_version='patch-v1',
    )


def ref(time, instrument):
    return {'time': time, 'instrument': instrument}


def counts(metrics):
    return tuple(metrics[key] for key in ('tp', 'fp', 'fn', 'support'))


def test_json_schema_independent_inclusive_thresholds_and_input_order():
    prediction = document(
        np.array([2, 1, 1]),
        np.array([[0.375, 0.875, 0.25], [0.5, 0.75, 0.25], [0, 0, 0]], dtype=np.float32),
        {'hihat': 0.25, 'snare': 0.875, 'kick': 0.5},
    )
    assert json.loads(json.dumps(prediction, allow_nan=False)) == prediction
    assert prediction['schema_version'] == 1
    assert prediction['model_version'] == 'synthetic-v1'
    assert prediction['preprocessing_version'] == 'patch-v1'
    assert [e['time_seconds'] for e in prediction['events']] == [2, 1, 1]
    assert [e['labels'] for e in prediction['events']] == [['snare', 'hihat'], ['kick', 'hihat'], []]
    assert tuple(prediction['events'][0]['probabilities']) == INSTRUMENTS
    assert prediction['events'][0]['probabilities']['kick'] == 0.375


def test_patch_binary_confusion_macro_and_micro():
    result = patch_metrics([[1, 1, 0], [0, 1, 0], [0, 0, 0]], [[0.5, 0.5, 0], [1, 0, 0], [0, 0, 0]],
                           [0.5, 0.5, 0.5])
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert [(counts(row), row['tn']) for row in result['per_class'].values()] == [
        ((1, 1, 0, 1), 1), ((1, 0, 1, 2), 1), ((0, 0, 0, 0), 3),
    ]
    assert counts(result['micro']) == (2, 1, 1, 3)
    assert result['micro']['tn'] == 5
    assert result['micro']['f1'] == pytest.approx(2 / 3)
    assert result['macro']['f1'] == pytest.approx(4 / 9)


def test_event_simultaneous_labels_duplicates_wrong_class_and_signed_timing():
    truth = [ref(1, 'kick'), ref(1, 'snare'), ref(2, 'kick'), ref(3, 'hihat')]
    prediction = document([0.98, 0.98, 2.04, 3, 9], [[1, 1, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]])
    result = event_metrics(truth, prediction)
    assert [counts(row) for row in result['per_class'].values()] == [(2, 1, 0, 2), (1, 1, 0, 1), (0, 0, 1, 1)]
    assert counts(result['micro']) == (3, 2, 1, 4)
    assert result['micro']['f1'] == pytest.approx(2 / 3)
    assert result['macro']['f1'] == pytest.approx((0.8 + 2 / 3) / 3)
    kick = result['per_class']['kick']
    assert kick['signed_errors_seconds'] == pytest.approx([-0.02, 0.04])
    assert kick['median_signed_error_seconds'] == pytest.approx(0.01)
    assert kick['median_absolute_error_seconds'] == pytest.approx(0.03)
    assert result['per_class']['hihat']['median_absolute_error_seconds'] is None


def test_matcher_keeps_nearby_references_and_uses_global_cardinality():
    truth = [ref(0.1, 'kick'), ref(0.2, 'kick'), ref(0.2, 'kick')]
    prediction = document([0.04, 0.14], [[1, 0, 0], [1, 0, 0]])
    result = event_metrics(truth, prediction, tolerance_seconds=0.06)
    assert counts(result['per_class']['kick']) == (2, 0, 0, 2)


def test_dataset_aggregation_never_matches_across_recordings():
    empty = document([], np.empty((0, 3)))
    truth = [ref(1, 'kick')]
    prediction = document([1], [[1, 0, 0]])
    result = dataset_event_metrics(iter([(truth, empty), ([], prediction), (truth, prediction)]))
    assert counts(result['micro']) == (1, 1, 1, 2)
    assert result['micro']['signed_errors_seconds'] == [0.0]
    assert result['micro']['median_absolute_error_seconds'] == 0.0


def test_empty_inputs():
    empty = np.empty((0, 3))
    assert patch_metrics(empty, empty, [0.5] * 3)['micro']['tn'] == 0
    assert event_metrics([], document([], empty))['micro']['median_signed_error_seconds'] is None
    assert dataset_event_metrics([])['macro'] == {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}


@pytest.mark.parametrize('bad', [[0.5], [0.5, 0.5, 1.01], [0.5, 0.5, np.nan], {'kick': 0.5}])
def test_invalid_thresholds(bad):
    with pytest.raises(ValueError, match='thresholds'):
        document([1], [[1, 0, 0]], bad)


@pytest.mark.parametrize('bad', [[0, 0, 0], [[0, 0]], [[0, np.inf, 0]], [[-0.1, 0, 0]], [['1', '0', '0']]])
def test_invalid_probabilities(bad):
    with pytest.raises(ValueError, match='probabilities'):
        document([1], bad)


@pytest.mark.parametrize('bad', [[], [-1], [np.nan], [True]])
def test_invalid_timestamps(bad):
    with pytest.raises(ValueError, match='timestamps'):
        document(bad, [[1, 0, 0]])


def test_bad_target_shape_label_and_schema():
    with pytest.raises(ValueError, match='same shape'):
        patch_metrics([[1, 0, 0]], np.empty((0, 3)), [0.5] * 3)
    with pytest.raises(ValueError, match='binary'):
        patch_metrics([[0.2, 0, 0]], [[1, 0, 0]], [0.5] * 3)
    prediction = document([1], [[1, 0, 0]])
    with pytest.raises(ValueError, match='instrument'):
        event_metrics([ref(1, 'ride')], prediction)
    prediction['events'][0]['labels'] = ['kick', 'kick']
    with pytest.raises(ValueError, match='event labels'):
        event_metrics([], prediction)
    with pytest.raises(ValueError, match='schema_version'):
        event_metrics([], prediction | {'schema_version': 2})
