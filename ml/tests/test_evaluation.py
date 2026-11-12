"""Onset-only matching tests, deliberately independent of DSP accuracy."""
import itertools

import pytest

from groovemirror_ml.evaluation import EvaluationConfig, evaluate_onsets


def score(refs, preds, tolerance=0.05):
    return evaluate_onsets(refs, preds, EvaluationConfig(onset_tolerance_seconds=tolerance))


def test_exact_matches():
    result = score([0, 0.5, 1], [0, 0.5, 1])
    assert (result.true_positives, result.false_positives, result.false_negatives) == (3, 0, 0)
    assert (result.precision, result.recall, result.f1) == (1, 1, 1)
    assert result.mean_absolute_error_seconds == 0


def test_early_late_signed_errors_and_no_automatic_centering():
    result = score([1, 2], [0.98, 2.04])
    assert [match.error_seconds for match in result.matches] == pytest.approx([-0.02, 0.04])
    assert result.mean_absolute_error_seconds == pytest.approx(0.03)
    shifted = score([1, 2], [1.04, 2.04])
    assert [m.error_seconds for m in shifted.matches] == pytest.approx([0.04, 0.04])


def test_missed_extra_metrics():
    result = score([0.1, 0.5, 0.9], [0.1, 0.5, 1.5, 1.7])
    assert (result.true_positives, result.false_positives, result.false_negatives) == (2, 2, 1)
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(2 / 3)
    assert result.f1 == pytest.approx(4 / 7)


@pytest.mark.parametrize("refs,preds,expected", [([], [], (0, 0, 0)), ([1], [], (0, 0, 1)), ([], [1], (0, 1, 0))])
def test_empty_sets(refs, preds, expected):
    result = score(refs, preds)
    assert (result.true_positives, result.false_positives, result.false_negatives) == expected
    assert (result.precision, result.recall, result.f1) == (0, 0, 0)
    assert result.mean_absolute_error_seconds is None
    assert result.median_absolute_error_seconds is None


def test_simultaneous_references_collapse_but_duplicate_predictions_are_extra():
    result = score([0.5, 0.5], [0.5, 0.5])
    assert (result.true_positives, result.false_positives, result.false_negatives) == (1, 1, 0)


def test_maximum_cardinality_beats_nearest_first_greedy():
    result = score([0.1, 0.2], [0.04, 0.14], tolerance=0.061)
    assert result.true_positives == 2
    assert {(m.reference_index, m.prediction_index) for m in result.matches} == {(0, 0), (1, 1)}


def test_minimum_error_among_maximum_cardinality_matches():
    result = score([0.1, 0.2], [0.02, 0.13, 0.21], tolerance=0.1)
    assert result.true_positives == 2
    assert sum(abs(m.error_seconds) for m in result.matches) == pytest.approx(0.04)


def test_zero_tolerance_requires_exact_timestamp():
    assert score([1], [1], tolerance=0).true_positives == 1
    assert score([1], [1 + 1e-13], tolerance=0).true_positives == 0


def test_reference_deduplication_retains_first_original_index():
    result = score([0.5, 0.1, 0.5], [0.5])
    assert result.matches[0].reference_index == 0
    assert result.to_dict()["unmatched_reference_indices"] == [1]


def test_original_indices_and_determinism_on_unsorted_inputs():
    first = score([0.8, 0.2], [0.21, 0.81])
    assert {(m.reference_index, m.prediction_index) for m in first.matches} == {(0, 1), (1, 0)}
    assert first.to_dict() == score([0.8, 0.2], [0.21, 0.81]).to_dict()


@pytest.mark.parametrize("pred,expected", [(0.9375, 1), (1.0625, 1), (0.93749, 0), (1.06251, 0)])
def test_inclusive_tolerance_boundary(pred, expected):
    assert score([1], [pred], tolerance=0.0625).true_positives == expected


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf"), -float("inf"), True, "0.5", None])
@pytest.mark.parametrize("side", ["reference", "prediction"])
def test_invalid_timestamps(bad, side):
    with pytest.raises(ValueError):
        score([bad] if side == "reference" else [1], [bad] if side == "prediction" else [1])


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf")])
def test_invalid_tolerance(bad):
    with pytest.raises(ValueError):
        score([1], [1], tolerance=bad)


def test_dense_candidate_processing_limit():
    references = [index / 2000 for index in range(1001)]
    predictions = [index / 2000 for index in range(1000)]
    with pytest.raises(ValueError, match="candidate pairs"):
        score(references, predictions, tolerance=1)


def test_event_processing_limit():
    with pytest.raises(ValueError, match="event processing limit"):
        score((index / 1000 for index in range(100001)), [])


def brute_force_optimum(refs, preds, tolerance):
    """Tiny exhaustive oracle: cardinality first, then absolute-error cost."""
    best = (0, 0.0)
    for size in range(1, min(len(refs), len(preds)) + 1):
        for indices in itertools.combinations(range(len(refs)), size):
            for predictions in itertools.permutations(range(len(preds)), size):
                errors = [abs(refs[i] - preds[j]) for i, j in zip(indices, predictions)]
                if all(error <= tolerance for error in errors):
                    candidate = (size, sum(errors))
                    if candidate[0] > best[0] or candidate[0] == best[0] and candidate[1] < best[1]:
                        best = candidate
    return best


def test_small_matching_cases_against_exhaustive_oracle():
    grid = [0.125, 0.25, 0.375, 0.5]
    subsets = [list(s) for size in range(4) for s in itertools.combinations(grid, size)]
    for refs in subsets:
        for preds in subsets:
            predictions = [time + 0.03125 for time in reversed(preds)]
            result = score(refs, predictions, tolerance=0.125)
            count, cost = brute_force_optimum(refs, predictions, 0.125)
            assert result.true_positives == count
            assert sum(abs(m.error_seconds) for m in result.matches) == pytest.approx(cost)
            assert len({m.reference_index for m in result.matches}) == count
            assert len({m.prediction_index for m in result.matches}) == count
