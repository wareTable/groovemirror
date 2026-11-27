"""Pure multi-label prediction serialization and instrument evaluation.

Probability and binary target matrices use (N, 3) in INSTRUMENTS order.
Thresholds are a length-three vector or a mapping with exactly those keys;
independent decisions are inclusive (probability >= threshold).
"""
from collections.abc import Mapping
from statistics import median

import numpy as np

from .onsets import EvaluationConfig, evaluate_onsets

INSTRUMENTS = ('kick', 'snare', 'hihat')
SCHEMA_VERSION = 1


def _numeric(values, name, *, allow_bool=False):
    array = np.asarray(values)
    if array.dtype.kind not in ('biuf' if allow_bool else 'iuf') or not np.isfinite(array).all():
        raise ValueError(f'{name} must contain finite numbers')
    return array


def _matrix(values, name, *, binary=False):
    array = _numeric(values, name, allow_bool=binary)
    if array.ndim != 2 or array.shape[1] != len(INSTRUMENTS):
        raise ValueError(f'{name} must have shape (N, 3)')
    if binary and not np.all((array == 0) | (array == 1)):
        raise ValueError(f'{name} must be binary')
    if not binary and np.any((array < 0) | (array > 1)):
        raise ValueError(f'{name} must be in [0, 1]')
    return array


def _decisions(probabilities, thresholds):
    if isinstance(thresholds, Mapping):
        if set(thresholds) != set(INSTRUMENTS):
            raise ValueError('thresholds must have exactly the canonical instrument keys')
        thresholds = [thresholds[name] for name in INSTRUMENTS]
    limits = _numeric(thresholds, 'thresholds')
    if limits.shape != (len(INSTRUMENTS),) or np.any((limits < 0) | (limits > 1)):
        raise ValueError('thresholds must have shape (3,) and be in [0, 1]')
    return probabilities >= limits


def _timestamps(values, size):
    times = _numeric(values, 'timestamps').astype(float)
    if times.shape != (size,) or not np.isfinite(times).all() or np.any(times < 0):
        raise ValueError('timestamps must be finite, nonnegative and aligned with the events')
    return times


def prediction_document(times_seconds, probabilities, thresholds, *, model_version, preprocessing_version):
    """Build a JSON-ready schema v1 document, preserving row order and duplicates.

    An event is one candidate patch, even if it has no labels. Probabilities
    are named native floats; labels are in canonical order and do not compete.
    """
    if not all(isinstance(v, str) and v.strip() for v in (model_version, preprocessing_version)):
        raise ValueError('model_version and preprocessing_version must be nonempty strings')
    scores = _matrix(probabilities, 'probabilities')
    decisions = _decisions(scores, thresholds)
    times = _timestamps(times_seconds, len(scores))
    return {
        'schema_version': SCHEMA_VERSION,
        'model_version': model_version,
        'preprocessing_version': preprocessing_version,
        'events': [
            {
                'time_seconds': float(time),
                'probabilities': dict(zip(INSTRUMENTS, map(float, row), strict=True)),
                'labels': [name for name, active in zip(INSTRUMENTS, chosen, strict=True) if active],
            }
            for time, row, chosen in zip(times, scores, decisions, strict=True)
        ],
    }


def _scores(tp, fp, fn):
    support = tp + fn
    predicted = tp + fp
    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'support': support,
        'precision': tp / predicted if predicted else 0.0,
        'recall': tp / support if support else 0.0,
        'f1': 2 * tp / (support + predicted) if support + predicted else 0.0,
    }


def _summary(per_class):
    rows = per_class.values()
    micro = _scores(*(sum(row[key] for row in rows) for key in ('tp', 'fp', 'fn')))
    macro = {key: sum(row[key] for row in rows) / len(INSTRUMENTS) for key in ('precision', 'recall', 'f1')}
    return {'per_class': per_class, 'micro': micro, 'macro': macro}


def patch_metrics(targets, probabilities, thresholds):
    """Report classwise binary confusion and micro/macro scores for aligned patches.

    Macro averages the three fixed classes, with zero for undefined ratios.
    Empty input matrices must have shape (0, 3).
    """
    truth = _matrix(targets, 'targets', binary=True).astype(bool)
    scores = _matrix(probabilities, 'probabilities')
    if truth.shape != scores.shape:
        raise ValueError('targets and probabilities must have the same shape')
    predicted = _decisions(scores, thresholds)
    rows = {}
    for i, name in enumerate(INSTRUMENTS):
        y, p = truth[:, i], predicted[:, i]
        rows[name] = _scores(
            int(np.count_nonzero(y & p)), int(np.count_nonzero(~y & p)), int(np.count_nonzero(y & ~p)),
        )
        rows[name]['tn'] = int(np.count_nonzero(~y & ~p))
    result = _summary(rows)
    result['micro']['tn'] = sum(row['tn'] for row in rows.values())
    return result


def _recording_matches(references, prediction, config):
    refs = {name: [] for name in INSTRUMENTS}
    preds = {name: [] for name in INSTRUMENTS}
    for ref in references:
        name = ref['instrument']
        if name not in refs:
            raise ValueError(f'unknown reference instrument: {name!r}')
        refs[name].append(ref['time'])
    if type(prediction.get('schema_version')) is not int or prediction['schema_version'] != SCHEMA_VERSION:
        raise ValueError('unsupported prediction schema_version')
    events = prediction['events']
    times = _timestamps([event['time_seconds'] for event in events], len(events))
    for time, event in zip(times, events, strict=True):
        labels = event['labels']
        if not isinstance(labels, list) or labels != [name for name in INSTRUMENTS if name in labels]:
            raise ValueError('event labels must be unique canonical instruments in canonical order')
        for name in labels:
            preds[name].append(time)
    return {name: evaluate_onsets(refs[name], preds[name], config) for name in INSTRUMENTS}


def _timing(errors):
    return {
        'signed_errors_seconds': errors,
        'median_signed_error_seconds': median(errors) if errors else None,
        'median_absolute_error_seconds': median(map(abs, errors)) if errors else None,
    }


def dataset_event_metrics(recordings, *, tolerance_seconds=0.05):
    """Score iterable of (references, prediction document) pairs independently.

    Sum TP/FP/FN before computing scores; pool matched timing errors rather
    than averaging per-recording medians. No event true negatives are defined.
    Only the emitted labels and timestamps are consumed, not probabilities.
    Exact duplicate reference times collapse within each class per the onset
    matcher; predictions at the same time remain separate candidates.
    """
    config = EvaluationConfig(onset_tolerance_seconds=tolerance_seconds)
    counts = {name: [0, 0, 0] for name in INSTRUMENTS}
    errors = {name: [] for name in INSTRUMENTS}
    for references, prediction in recordings:
        for name, match in _recording_matches(references, prediction, config).items():
            counts[name][0] += match.true_positives
            counts[name][1] += match.false_positives
            counts[name][2] += match.false_negatives
            errors[name].extend(match.signed_errors_seconds)
    result = _summary({name: _scores(*counts[name]) | _timing(errors[name]) for name in INSTRUMENTS})
    result['micro'].update(_timing([error for name in INSTRUMENTS for error in errors[name]]))
    result['tolerance_seconds'] = float(config.onset_tolerance_seconds)
    return result


def event_metrics(references, prediction, *, tolerance_seconds=0.05):
    """Score one recording with references {'time': seconds, 'instrument': name}."""
    return dataset_event_metrics([(references, prediction)], tolerance_seconds=tolerance_seconds)
