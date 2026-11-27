"""Independent kick, snare and hi-hat classifiers for fixed (N, F, T) patches.

Only training data sets normalization, class weights and learned parameters.
Classes with no positive examples predict 0; classes with no negatives predict 1.
Balanced training objectives do not guarantee calibrated probabilities.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar
from zipfile import BadZipFile

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

CLASS_ORDER = ('kick', 'snare', 'hihat')
_FORMAT_VERSION = 1
_PREDICT_BATCH = 64


def _patches(X, *, shape=None, nonempty=False):
    X = np.asarray(X)
    if X.ndim != 3 or min(X.shape[1:]) < 1 or (nonempty and len(X) == 0):
        raise ValueError('X must have shape (N, F, T) with positive patch dimensions')
    if shape is not None and X.shape[1:] != shape:
        raise ValueError(f'Expected patch shape {shape}, got {X.shape[1:]}')
    if X.dtype.kind not in 'fiu':
        raise ValueError('X must contain finite real numbers')
    with np.errstate(over='ignore', invalid='ignore'):
        X = np.asarray(X, dtype=np.float32, order='C')
    if not np.isfinite(X).all():
        raise ValueError('X must contain finite float32 values')
    return X


def _targets(y, n):
    y = np.asarray(y)
    if y.shape != (n, len(CLASS_ORDER)) or y.dtype.kind not in 'bfiu':
        raise ValueError(f'y must be a binary (N, 3) array ordered {CLASS_ORDER}')
    if not np.all((y == 0) | (y == 1)):
        raise ValueError('y must contain only 0 and 1')
    return y.astype(np.float32)


def _integer(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= value <= high):
        raise ValueError(f'{name} must be an integer in [{low}, {high}]')
    return int(value)


def _positive_real(value, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))):
        raise ValueError(f'{name} must be a finite positive number')
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a finite positive number')
    return value


def _class_setup(y, seed):
    counts = y.sum(axis=0, dtype=np.int64)
    constant = np.where(counts == 0, 0, np.where(counts == len(y), 1, -1)).astype(np.int8)
    history = {
        'seed': seed,
        'train_positive_counts': dict(zip(CLASS_ORDER, map(int, counts))),
        'constant_classes': {
            name: int(value) for name, value in zip(CLASS_ORDER, constant) if value != -1
        },
    }
    return counts, constant, history


def _summary(X):
    # Preserve position with global, per-frequency and per-time mean/std.
    X = X.astype(np.float64)
    return np.concatenate([
        X.mean((1, 2))[:, None], X.std((1, 2))[:, None],
        X.mean(2), X.std(2), X.mean(1), X.std(1),
    ], axis=1)


@dataclass
class ClassicalModel:
    model_type: ClassVar[str] = 'classical'
    patch_shape: tuple[int, int]
    constant: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray

    def predict_proba(self, X) -> np.ndarray:
        X = _patches(X, shape=self.patch_shape)
        z = (_summary(X) - self.mean) / self.scale
        probabilities = expit(z @ self.coefficients.T + self.intercept)
        fixed = self.constant != -1
        probabilities[:, fixed] = self.constant[fixed]
        return probabilities.astype(np.float32)


def fit_classical(X_train, y_train, *, seed=0, l2=1.0) -> tuple[ClassicalModel, dict]:
    """Fit independent L2-regularized logistic models with balanced row weights.

    Returns (model, JSON-compatible history). Optimization starts at zero and
    is deterministic; seed is recorded for consistency with the CNN interface.
    """
    seed = _integer(seed, 'seed', 0, 2**32 - 1)
    l2 = _positive_real(l2, 'l2')
    X = _patches(X_train, nonempty=True)
    y = _targets(y_train, len(X))
    counts, constant, history = _class_setup(y, seed)
    history.update(model_type='classical', l2=l2, classes={})
    z = _summary(X)
    mean = z.mean(axis=0)
    scale = z.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (z - mean) / scale
    coefficients = np.zeros((len(CLASS_ORDER), z.shape[1]))
    intercept = np.zeros(len(CLASS_ORDER))
    for k in np.flatnonzero(constant == -1):
        target = y[:, k]
        # Each label value contributes half the total weight for this class.
        weights = np.where(target == 1, 0.5 / counts[k], 0.5 / (len(X) - counts[k]))

        def objective(params):
            beta = params[:-1]
            logits = z @ beta + params[-1]
            loss = weights @ np.logaddexp(0, (1 - 2 * target) * logits)
            error = weights * (expit(logits) - target)
            gradient = np.r_[z.T @ error + l2 * beta, error.sum()]
            return loss + (0.5 * l2) * (beta @ beta), gradient

        result = minimize(objective, np.zeros(z.shape[1] + 1), jac=True,
                          method='L-BFGS-B', options={'maxiter': 300})
        if not result.success or not np.isfinite(result.fun) or not np.isfinite(result.x).all():
            raise RuntimeError(f'Logistic fit failed for {CLASS_ORDER[k]}: {result.message}')
        coefficients[k] = result.x[:-1]
        intercept[k] = result.x[-1]
        history['classes'][CLASS_ORDER[k]] = {
            'iterations': int(result.nit), 'train_objective': float(result.fun),
        }
    return ClassicalModel(X.shape[1:], constant, mean, scale, coefficients, intercept), history


def _torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name != 'torch':
            raise
        raise ImportError('The CNN requires PyTorch; the classical model does not.') from exc
    return torch


def _cnn_network(torch, seed):
    nn = torch.nn
    # Force CPU/float32 even if the caller has set PyTorch defaults. Do not
    # advance the caller's torch RNG; the shuffle generator is separate.
    kw = {'device': 'cpu', 'dtype': torch.float32}
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, **kw), nn.ReLU(),
            nn.AvgPool2d(2, ceil_mode=True),
            nn.Conv2d(8, 12, 3, padding=1, **kw), nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 2)), nn.Flatten(), nn.Linear(12 * 3 * 2, 3, **kw),
        )


@dataclass
class CNNModel:
    model_type: ClassVar[str] = 'cnn'
    patch_shape: tuple[int, int]
    constant: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    network: Any

    def _inputs(self, X):
        array = ((X.astype(np.float64) - self.mean) / self.scale).astype(np.float32)
        return _torch().from_numpy(array[:, None])

    def predict_proba(self, X) -> np.ndarray:
        X = _patches(X, shape=self.patch_shape)
        torch = _torch()
        probabilities = np.empty((len(X), len(CLASS_ORDER)), dtype=np.float32)
        fixed = self.constant != -1
        if not fixed.all():
            with torch.inference_mode():
                for start in range(0, len(X), _PREDICT_BATCH):
                    stop = start + _PREDICT_BATCH
                    logits = self.network(self._inputs(X[start:stop]))
                    probabilities[start:stop] = logits.sigmoid().numpy()
        probabilities[:, fixed] = self.constant[fixed]
        return probabilities


def fit_cnn(X_train, y_train, X_val, y_val, *, seed=0, epochs=12, batch_size=64,
            learning_rate=0.001) -> tuple[CNNModel, dict]:
    """Fit on CPU and return the best validation checkpoint and JSON history.

    Epochs (1..100) and batch size (1..512) bound the run. Train loss is
    class-balanced BCE on nonconstant classes; unweighted validation BCE on
    those same classes selects the earliest lowest-loss epoch. If all classes
    are constant, no optimization is done and best_epoch is None.
    Reproducibility is expected for a fixed CPU/library/thread environment.
    """
    seed = _integer(seed, 'seed', 0, 2**32 - 1)
    epochs = _integer(epochs, 'epochs', 1, 100)
    batch_size = _integer(batch_size, 'batch_size', 1, 512)
    learning_rate = _positive_real(learning_rate, 'learning_rate')
    X = _patches(X_train, nonempty=True)
    y = _targets(y_train, len(X))
    V = _patches(X_val, shape=X.shape[1:], nonempty=True)
    v = _targets(y_val, len(V))
    counts, constant, history = _class_setup(y, seed)
    history.update(model_type='cnn', epochs=[], best_epoch=None, requested_epochs=epochs,
                   batch_size=batch_size, learning_rate=learning_rate)
    torch = _torch()
    network = _cnn_network(torch, seed)
    mean, scale = X.mean(dtype=np.float64), X.std(dtype=np.float64)
    model = CNNModel(X.shape[1:], constant, mean, scale if scale >= 1e-12 else 1.0,
                     network.eval())
    active = np.flatnonzero(constant == -1)
    if not len(active):
        return model, history
    labels = torch.from_numpy(y[:, active])
    val_labels = torch.from_numpy(v[:, active])
    positive = torch.from_numpy((len(X) / (2 * counts[active])).astype(np.float32))
    negative = torch.from_numpy((len(X) / (2 * (len(X) - counts[active]))).astype(np.float32))
    generator = torch.Generator(device='cpu').manual_seed(seed)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    loss = torch.nn.functional.binary_cross_entropy_with_logits
    best_loss = float('inf')
    best_weights = None
    for epoch in range(1, epochs + 1):
        network.train()
        train_loss = 0.0
        order = torch.randperm(len(X), generator=generator, device='cpu').numpy()
        for start in range(0, len(X), batch_size):
            indices = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = network(model._inputs(X[indices]))[:, active]
            target = labels[indices]
            batch_loss = loss(logits, target, weight=torch.where(target == 1, positive, negative))
            batch_loss.backward()
            optimizer.step()
            train_loss += batch_loss.item() * len(indices)
        network.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for start in range(0, len(V), batch_size):
                stop = start + batch_size
                logits = network(model._inputs(V[start:stop]))[:, active]
                val_loss += loss(logits, val_labels[start:stop], reduction='sum').item()
        train_loss /= len(X)
        val_loss /= len(V) * len(active)
        if not np.isfinite([train_loss, val_loss]).all():
            raise RuntimeError(f'CNN loss became nonfinite at epoch {epoch}')
        history['epochs'].append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = {name: value.clone() for name, value in network.state_dict().items()}
            history['best_epoch'] = epoch
    network.load_state_dict(best_weights)
    return model, history


def save_model(model: ClassicalModel | CNNModel, path: str | Path) -> None:
    """Write a versioned, local NPZ artifact to exactly path (no implied suffix)."""
    if type(model) not in (ClassicalModel, CNNModel):
        raise TypeError('Expected a fitted ClassicalModel or CNNModel')
    metadata = json.dumps({'format_version': _FORMAT_VERSION, 'model_type': model.model_type,
                           'class_order': list(CLASS_ORDER), 'patch_shape': model.patch_shape})
    arrays = dict(metadata=np.asarray(metadata), constant=model.constant,
                  mean=np.asarray(model.mean), scale=np.asarray(model.scale))
    if isinstance(model, ClassicalModel):
        arrays.update(coefficients=model.coefficients, intercept=model.intercept)
    else:
        arrays.update({f'weights.{name}': value.detach().numpy()
                       for name, value in model.network.state_dict().items()})
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError('Object arrays are not valid model parameters')
    with Path(path).open('wb') as file:
        np.savez_compressed(file, **arrays)


def _parameter(archive, name, dtype, shape):
    result = archive[name]
    if result.dtype != np.dtype(dtype) or result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'Malformed model parameter: {name}')
    return result


def load_model(path: str | Path) -> ClassicalModel | CNNModel:
    """Load a local NPZ without pickle, validating version, schema and parameters."""
    try:
        with Path(path).open('rb') as file:
            archive = np.load(file, allow_pickle=False)
            if not isinstance(archive, np.lib.npyio.NpzFile):
                raise ValueError('Expected an NPZ model archive')
            with archive:
                return _load_archive(archive)
    except (BadZipFile, EOFError, KeyError, ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f'Invalid model artifact: {exc}') from exc


def _load_archive(archive):
    descriptor = archive['metadata']
    if descriptor.shape != () or descriptor.dtype.kind != 'U' or descriptor.nbytes > 4096:
        raise ValueError('Malformed model metadata')
    meta = json.loads(descriptor.item())
    keys = {'format_version', 'model_type', 'class_order', 'patch_shape'}
    if (not isinstance(meta, dict) or set(meta) != keys
            or type(meta['format_version']) is not int or meta['format_version'] != _FORMAT_VERSION
            or meta['class_order'] != list(CLASS_ORDER)):
        raise ValueError('Unknown model format or class order')
    shape = meta['patch_shape']
    if (not isinstance(shape, list) or len(shape) != 2
            or any(type(dim) is not int or dim < 1 for dim in shape)):
        raise ValueError('Invalid patch shape')
    shape = tuple(shape)
    common = {'metadata', 'constant', 'mean', 'scale'}
    size = len(CLASS_ORDER)
    constant = _parameter(archive, 'constant', np.int8, (size,))
    if not np.isin(constant, [-1, 0, 1]).all():
        raise ValueError('Invalid constant classes')
    if meta['model_type'] == 'classical':
        expected = common | {'coefficients', 'intercept'}
        statistic_shape = (2 + 2 * sum(shape),)
    elif meta['model_type'] == 'cnn':
        network = _cnn_network(_torch(), seed=0)
        weights = network.state_dict()
        expected = common | {f'weights.{name}' for name in weights}
        statistic_shape = ()
    else:
        raise ValueError('Unknown model type')
    if len(archive.files) != len(expected) or set(archive.files) != expected:
        raise ValueError('Unexpected or missing model archive entries')
    mean = _parameter(archive, 'mean', np.float64, statistic_shape)
    scale = _parameter(archive, 'scale', np.float64, statistic_shape)
    if (scale <= 0).any():
        raise ValueError('Model scale must be positive')
    if meta['model_type'] == 'classical':
        coefficients = _parameter(archive, 'coefficients', np.float64, (size, *statistic_shape))
        intercept = _parameter(archive, 'intercept', np.float64, (size,))
        return ClassicalModel(shape, constant, mean, scale, coefficients, intercept)
    torch = _torch()
    network.load_state_dict({
        name: torch.from_numpy(_parameter(archive, f'weights.{name}', np.float32, tuple(t.shape)))
        for name, t in weights.items()
    })
    return CNNModel(shape, constant, mean, scale, network.eval())
