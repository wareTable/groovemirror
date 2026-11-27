"""Synthetic verification; these fixtures do not measure real drum performance."""

import json

import numpy as np
import pytest

from groovemirror_ml.instrument_models import fit_classical, fit_cnn, load_model, save_model


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(42)
    # Include overlapping instruments, blank rows, and one rarer class.
    patterns = np.asarray([[k, s, h] for k in (0, 1) for s in (0, 1) for h in (0, 1)])
    y = np.repeat(patterns, [12, 3, 12, 3, 12, 3, 12, 3], axis=0)
    X = rng.normal(0, 0.05, (len(y), 12, 8)).astype(np.float32)
    for k in range(3):
        X[:, 4*k:4*(k + 1), :] += 2 * y[:, k, None, None]
    return X, y


@pytest.fixture
def cpu_torch():
    torch = pytest.importorskip('torch')
    # Limit oversubscription for this tiny fixture on large dev machines.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield torch
    finally:
        torch.set_num_threads(previous)


def test_classical_learns_independent_classes_and_round_trips(synthetic, tmp_path):
    X, y = synthetic
    model, history = fit_classical(X, y, seed=42)
    again, again_history = fit_classical(X, y, seed=42)
    probability = model.predict_proba(X)
    assert probability.shape == y.shape and probability.dtype == np.float32
    np.testing.assert_array_equal(probability >= 0.5, y)
    np.testing.assert_array_equal(probability, again.predict_proba(X))
    assert history == again_history
    assert history['train_positive_counts'] == {'kick': 30, 'snare': 30, 'hihat': 12}
    assert json.loads(json.dumps(history, allow_nan=False)) == history
    path = tmp_path / 'classical.npz'
    save_model(model, path)
    restored = load_model(path)
    assert restored.model_type == 'classical'
    np.testing.assert_array_equal(restored.predict_proba(X), probability)
    assert restored.predict_proba(X[:0]).shape == (0, 3)
    with pytest.raises(ValueError, match='Expected patch shape'):
        restored.predict_proba(X[:, :, :-1])


@pytest.mark.parametrize('kind', ['classical', 'cnn'])
def test_constant_classes_are_train_only_and_survive_round_trip(kind, synthetic, request, tmp_path):
    X, y = synthetic
    y[:, 1:] = [0, 1]
    if kind == 'cnn':
        request.getfixturevalue('cpu_torch')
        model, history = fit_cnn(X, y, X, 1 - y, epochs=2, batch_size=16)
    else:
        model, history = fit_classical(X, y)
    assert history['constant_classes'] == {'snare': 0, 'hihat': 1}
    path = tmp_path / kind  # Save must use the exact path, not append .npz.
    save_model(model, path)
    restored = load_model(path)
    assert restored.model_type == kind
    probability = restored.predict_proba(X)
    np.testing.assert_array_equal(probability, model.predict_proba(X))
    np.testing.assert_array_equal(probability[:, 1:], y[:, 1:])
    assert (probability[:, 0] > 0).all() and (probability[:, 0] < 1).all()


def test_cnn_determinism_learning_checkpoint_and_artifact(synthetic, cpu_torch, tmp_path):
    X, y = synthetic
    # Alternate examples to keep every pattern in the synthetic train and val slices.
    args = X[::2], y[::2], X[1::2], y[1::2]
    kw = dict(seed=4, epochs=12, batch_size=8, learning_rate=0.015)
    state = cpu_torch.random.get_rng_state().clone()
    model, history = fit_cnn(*args, **kw)
    assert cpu_torch.equal(state, cpu_torch.random.get_rng_state())
    again, again_history = fit_cnn(*args, **kw)
    probability = model.predict_proba(X)
    np.testing.assert_array_equal(probability, again.predict_proba(X))
    assert history == again_history
    assert json.loads(json.dumps(history, allow_nan=False)) == history
    assert len(history['epochs']) == kw['epochs']
    best = min(history['epochs'], key=lambda row: row['val_loss'])
    assert history['best_epoch'] == best['epoch']
    assert best['val_loss'] < history['epochs'][0]['val_loss']
    # Returned weights must correspond to the selected epoch (not just the final one).
    with cpu_torch.inference_mode():
        logits = model.network(model._inputs(args[2])).numpy()
    val_bce = np.logaddexp(0, (1 - 2 * args[3]) * logits).mean()
    assert val_bce == pytest.approx(best['val_loss'], abs=1e-6)
    for k in range(3):
        assert probability[y[:, k] == 1, k].mean() > probability[y[:, k] == 0, k].mean()
    assert (probability[y.sum(axis=1) == 3] > 0.5).all()
    path = tmp_path / 'cnn.npz'
    save_model(model, path)
    restored = load_model(path)
    np.testing.assert_array_equal(restored.predict_proba(X), probability)
    assert restored.predict_proba(X[:0]).shape == (0, 3)


def test_cnn_validation_changes_monitoring_not_weight_updates(synthetic, cpu_torch):
    X, y = synthetic
    train = X[::2], y[::2]
    val = X[1::2], y[1::2]
    # With one epoch, model selection cannot choose a different train checkpoint.
    first, baseline = fit_cnn(*train, *val, epochs=1, batch_size=16)
    second, shifted = fit_cnn(*train, val[0] + 10, 1 - val[1], epochs=1, batch_size=16)
    assert baseline['epochs'][0]['val_loss'] != shifted['epochs'][0]['val_loss']
    assert baseline['epochs'][0]['train_loss'] == shifted['epochs'][0]['train_loss']
    np.testing.assert_array_equal(first.predict_proba(X), second.predict_proba(X))


def test_all_constant_cnn_skips_optimization(cpu_torch):
    X = np.zeros((3, 1, 1), dtype=np.float32)
    y = np.tile([1, 1, 0], (3, 1))
    model, history = fit_cnn(X, y, X, 1 - y, epochs=2)
    assert history['best_epoch'] is None
    assert history['epochs'] == []
    np.testing.assert_array_equal(model.predict_proba(X), y)


@pytest.mark.parametrize('kind', ['wrong_shape', 'empty', 'nan', 'object', 'labels'])
def test_reject_bad_training_arrays(kind, synthetic):
    X, y = synthetic
    if kind == 'wrong_shape':
        X = X[:, 0]
    elif kind == 'empty':
        X, y = X[:0], y[:0]
    elif kind == 'nan':
        X[0, 0, 0] = np.nan
    elif kind == 'object':
        X = X.astype(object)
    else:
        y[0, 0] = 2
    with pytest.raises(ValueError):
        fit_classical(X, y)


@pytest.mark.parametrize('kwargs', [{'epochs': 101}, {'batch_size': 0}, {'seed': True},
                                    {'learning_rate': np.inf}])
def test_cnn_rejects_invalid_controls(kwargs, synthetic):
    X, y = synthetic
    with pytest.raises(ValueError):
        fit_cnn(X, y, X, y, **kwargs)


@pytest.mark.parametrize('field', ['object', 'shape', 'type', 'order', 'nonfinite', 'scale', 'extra'])
def test_model_archive_rejects_malformed_parameters_and_schema(field, synthetic, tmp_path):
    path = tmp_path / 'classical.npz'
    source, _ = fit_classical(*synthetic)
    save_model(source, path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = dict(archive)
    if field == 'object':
        arrays['mean'] = arrays['mean'].astype(object)
    elif field == 'shape':
        arrays['coefficients'] = arrays['coefficients'][:2]
    elif field in {'type', 'order'}:
        meta = json.loads(arrays['metadata'].item())
        if field == 'type':
            meta['model_type'] = 'unknown'
        else:
            meta['class_order'].reverse()
        arrays['metadata'] = np.asarray(json.dumps(meta))
    elif field == 'nonfinite':
        arrays['intercept'][0] = np.inf
    elif field == 'scale':
        arrays['scale'][0] = 0.0
    else:
        arrays['extra'] = np.zeros(1)
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match='Invalid model artifact'):
        load_model(path)


def test_cnn_archive_rejects_wrong_weight_dtype(synthetic, cpu_torch, tmp_path):
    X, y = synthetic
    path = tmp_path / 'cnn.npz'
    model, _ = fit_cnn(X, y, X, y, epochs=1)
    save_model(model, path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = dict(archive)
    key = next(name for name in arrays if name.startswith('weights.'))
    arrays[key] = arrays[key].astype(np.float64)
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match='Malformed model parameter'):
        load_model(path)
