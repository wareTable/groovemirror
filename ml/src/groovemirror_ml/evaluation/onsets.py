"""Deterministic, one-to-one, instrument-agnostic onset evaluation.

Exact duplicate reference timestamps represent a single onset, including
simultaneous hits of distinct instruments. Predictions are never deduplicated:
repeated detections compete for the same reference and can be false positives.
Matching maximizes cardinality, then minimizes total absolute error, rather
than greedily assigning each event its nearest neighbour. Signed errors are
prediction minus reference (positive = late), without offset correction.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
import math
from numbers import Real
from typing import Any, Iterable

import numpy as np


MAX_EVENTS = 100_000
MAX_CANDIDATE_PAIRS = 1_000_000
# Only accommodates arithmetic roundoff at an inclusive tolerance boundary.
BOUNDARY_EPSILON_SECONDS = 1e-12


@dataclass(frozen=True)
class EvaluationConfig:
    onset_tolerance_seconds: float = 0.05

    def __post_init__(self) -> None:
        value = self.onset_tolerance_seconds
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(value) or value < 0):
            raise ValueError("onset_tolerance_seconds must be finite and nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OnsetMatch:
    reference_index: int
    prediction_index: int
    reference_time: float
    prediction_time: float
    error_seconds: float


@dataclass(frozen=True)
class EvaluationResult:
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    reference_count: int
    prediction_count: int
    matches: tuple[OnsetMatch, ...]
    unmatched_reference_indices: tuple[int, ...]
    unmatched_prediction_indices: tuple[int, ...]
    signed_errors_seconds: tuple[float, ...]
    mean_absolute_error_seconds: float | None
    median_absolute_error_seconds: float | None
    onset_tolerance_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision": self.precision, "recall": self.recall, "f1": self.f1,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "reference_count": self.reference_count, "prediction_count": self.prediction_count,
            "matches": [asdict(match) for match in self.matches],
            "unmatched_reference_indices": list(self.unmatched_reference_indices),
            "unmatched_prediction_indices": list(self.unmatched_prediction_indices),
            "signed_errors_seconds": list(self.signed_errors_seconds),
            "mean_absolute_error_seconds": self.mean_absolute_error_seconds,
            "median_absolute_error_seconds": self.median_absolute_error_seconds,
            "onset_tolerance_seconds": self.onset_tolerance_seconds,
        }


def _timestamps(values: Iterable[float], name: str) -> list[tuple[float, int]]:
    result = []
    for index, value in enumerate(values):
        if index >= MAX_EVENTS:
            raise ValueError(f"{name} exceeds the {MAX_EVENTS} event processing limit")
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not math.isfinite(value) or value < 0):
            raise ValueError(f"{name}[{index}] must be a finite nonnegative timestamp")
        result.append((float(value), index))
    return sorted(result)


def evaluate_onsets(
    reference_times: Iterable[float],
    predicted_times: Iterable[float],
    config: EvaluationConfig = EvaluationConfig(),
) -> EvaluationResult:
    """Compute maximum-cardinality, minimum-absolute-error temporal matching.

    Stable sorting preserves original indices. Duplicate references keep their
    first input index; returned unmatched indices refer to original inputs (not
    a newly indexed sorted array). Exact remaining objective ties select the
    earliest generated DP node, traversing reference time then prediction time
    then original input index. Empty precision/recall denominators yield 0;
    timing statistics with no matches are None, never fabricated zeros.

    Sparse ordered dynamic programming with a Fenwick prefix-best tree takes
    O(E log P) time and O(E+P) storage, where E is the number of pairs within
    tolerance. Cardinality is the primary objective and total absolute error
    the secondary objective. In 1D with a symmetric constant tolerance, crossing
    pairs can be uncrossed without losing feasibility or increasing absolute
    cost, so an optimal order-preserving matching exists. Updates are deferred
    per reference to prevent reuse. Explicit limits bound pathological dense
    candidate sets before allocation; no approximate fallback is used.
    """
    references_all = _timestamps(reference_times, "reference_times")
    predictions = _timestamps(predicted_times, "predicted_times")
    references = []
    for pair in references_all:
        if not references or references[-1][0] != pair[0]:
            references.append(pair)
    pred_values = [time for time, _ in predictions]
    tolerance = float(config.onset_tolerance_seconds)
    # Do not add epsilon at tolerance=0: exact matching must remain exact.
    slack = BOUNDARY_EPSILON_SECONDS if tolerance > 0 else 0.0
    windows = []
    edge_count = 0
    for time, _ in references:
        left = bisect_left(pred_values, time - tolerance - slack)
        right = bisect_right(pred_values, time + tolerance + slack)
        edge_count += right - left
        if edge_count > MAX_CANDIDATE_PAIRS:
            raise ValueError(f"Matching exceeds {MAX_CANDIDATE_PAIRS} candidate pairs; reduce tolerance or recording/event count")
        windows.append((left, right))

    # Sentinel empty chain, then compact parallel arrays for DP reconstruction.
    counts = [0]
    costs = [0.0]
    parents = [-1]
    reference_positions = [-1]
    prediction_positions = [-1]
    tree = [0] * (len(predictions) + 1)

    def better(a: int, b: int) -> int:
        if counts[a] != counts[b]:
            return a if counts[a] > counts[b] else b
        if costs[a] != costs[b]:
            return a if costs[a] < costs[b] else b
        return min(a, b)

    def query(length: int) -> int:
        best = 0
        while length > 0:
            best = better(best, tree[length])
            length -= length & -length
        return best

    def update(position: int, node: int) -> None:
        position += 1
        while position < len(tree):
            tree[position] = better(tree[position], node)
            position += position & -position

    for ref_position, (left, right) in enumerate(windows):
        pending = []
        ref_time = references[ref_position][0]
        for pred_position in range(left, right):
            parent = query(pred_position)  # Strictly earlier prediction positions.
            node = len(counts)
            counts.append(counts[parent] + 1)
            costs.append(costs[parent] + abs(pred_values[pred_position] - ref_time))
            parents.append(parent)
            reference_positions.append(ref_position)
            prediction_positions.append(pred_position)
            pending.append((pred_position, node))
        for pred_position, node in pending:
            update(pred_position, node)

    node = query(len(predictions))
    matched = []
    while node:
        ref_time, ref_index = references[reference_positions[node]]
        pred_time, pred_index = predictions[prediction_positions[node]]
        matched.append(OnsetMatch(ref_index, pred_index, ref_time, pred_time, pred_time - ref_time))
        node = parents[node]
    matches = tuple(reversed(matched))
    matched_refs = {match.reference_index for match in matches}
    matched_preds = {match.prediction_index for match in matches}
    tp, n_ref, n_pred = len(matches), len(references), len(predictions)
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_ref if n_ref else 0.0
    f1 = 2 * tp / (n_ref + n_pred) if n_ref + n_pred else 0.0
    errors = tuple(match.error_seconds for match in matches)
    absolute_errors = np.abs(np.asarray(errors, dtype=float))
    return EvaluationResult(
        precision, recall, f1, tp, n_pred - tp, n_ref - tp, n_ref, n_pred,
        matches,
        tuple(index for _, index in references if index not in matched_refs),
        tuple(index for _, index in predictions if index not in matched_preds),
        errors,
        float(np.mean(absolute_errors)) if tp else None,
        float(np.median(absolute_errors)) if tp else None,
        tolerance,
    )
