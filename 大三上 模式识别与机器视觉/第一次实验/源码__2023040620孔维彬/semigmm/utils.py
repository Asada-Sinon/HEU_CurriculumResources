import numpy as np

_EPS = 1e-12


def logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.maximum(_EPS, np.sum(np.exp(a - m), axis=axis, keepdims=True)))).squeeze(axis)


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(logits, axis=axis, keepdims=True)
    e = np.exp(logits - m)
    return e / np.maximum(_EPS, np.sum(e, axis=axis, keepdims=True))


def log_gaussian_diag(X: np.ndarray, MU: np.ndarray, LOGVAR: np.ndarray) -> np.ndarray:
    """
    Compute log N(x|mu, diag(var)) for all x and all components.
    X: (N, D), MU: (K, D), LOGVAR: (K, D)
    Returns: (N, K) of log-probabilities
    """
    N, D = X.shape
    # Expand dims for broadcasting: (N, 1, D) - (1, K, D)
    x = X[:, None, :]  # (N,1,D)
    mu = MU[None, :, :]  # (1,K,D)
    logvar = LOGVAR[None, :, :]  # (1,K,D)
    inv_var = np.exp(-logvar)
    diff2 = (x - mu) ** 2
    # log N = -0.5*(D*log(2pi) + sum_d(logvar + diff2*inv_var))
    const = -0.5 * D * np.log(2 * np.pi)
    log_prob = const - 0.5 * np.sum(logvar + diff2 * inv_var, axis=-1)  # (N,K)
    return log_prob


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((y.shape[0], n_classes), dtype=float)
    out[np.arange(y.shape[0]), y] = 1.0
    return out
