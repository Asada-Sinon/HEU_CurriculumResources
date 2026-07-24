from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans

from .utils import log_gaussian_diag, softmax, logsumexp


@dataclass
class SemiGMMConfig:
    n_classes: int
    components_per_class: int = 1
    reg_covar: float = 1e-6
    random_state: int = 0


class SemiGMM:
    """
    Semi-supervised GMM with per-class component groups.
    - E-step masks components for labeled samples to their class group only.
    - Two training modes:
      * EM closed-form updates (diag covariance)
      * Gradient-based M-step on Q with fixed responsibilities
    """

    def __init__(self, config: SemiGMMConfig):
        self.cfg = config
        self.C = config.n_classes
        self.M = config.components_per_class
        self.K = self.C * self.M
        self.rng = np.random.RandomState(config.random_state)
        # Parameters
        self.mu_: Optional[np.ndarray] = None  # (K,D)
        self.logvar_: Optional[np.ndarray] = None  # (K,D)
        self.alpha_: Optional[np.ndarray] = None  # logits for pi (K,)

    # --------- helpers ---------
    def _class_components(self, c: int) -> np.ndarray:
        return np.arange(c * self.M, (c + 1) * self.M, dtype=int)

    def _init_params(self, X: np.ndarray):
        N, D = X.shape
        # KMeans for means
        km = KMeans(n_clusters=self.K, random_state=self.cfg.random_state, n_init=5)
        labels = km.fit_predict(X)
        mu = np.zeros((self.K, D), dtype=float)
        for k in range(self.K):
            if np.any(labels == k):
                mu[k] = X[labels == k].mean(axis=0)
            else:
                mu[k] = X[self.rng.randint(0, N)]
        # Variances as overall variance
        var = np.var(X, axis=0) + self.cfg.reg_covar
        logvar = np.log(np.tile(var[None, :], (self.K, 1)))
        alpha = np.zeros(self.K, dtype=float)  # softmax -> uniform
        self.mu_, self.logvar_, self.alpha_ = mu, logvar, alpha

    def _e_step(self, X: np.ndarray, y: Optional[np.ndarray]) -> np.ndarray:
        """Compute responsibilities r (N,K); y may contain -1 for unknown."""
        assert self.mu_ is not None and self.logvar_ is not None and self.alpha_ is not None
        N = X.shape[0]
        log_pi = self.alpha_ - logsumexp(self.alpha_, axis=0)
        log_p = log_gaussian_diag(X, self.mu_, self.logvar_) + log_pi[None, :]
        # Mask for labeled samples
        if y is not None:
            for i in range(N):
                if y[i] >= 0:
                    c = int(y[i])
                    allowed = self._class_components(c)
                    # set disallowed to -inf
                    mask = np.ones(self.K, dtype=bool)
                    mask[allowed] = False
                    log_p[i, mask] = -np.inf
        # Normalize
        r = np.exp(log_p - logsumexp(log_p, axis=1)[:, None])
        r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        return r

    # --------- EM training ---------
    def fit_em(self, X_l: np.ndarray, y_l: np.ndarray, X_u: np.ndarray, max_iter=100, tol=1e-4, verbose=False):
        X = np.vstack([X_l, X_u])
        y = np.concatenate([y_l, np.full(X_u.shape[0], -1, dtype=int)])
        if self.mu_ is None:
            self._init_params(X)
        prev_ll = -np.inf
        for it in range(max_iter):
            r = self._e_step(X, y)
            # M-step closed-form (diag)
            Nk = r.sum(axis=0) + 1e-12
            pi = Nk / Nk.sum()
            mu = (r.T @ X) / Nk[:, None]
            # diag variances
            diff = X[:, None, :] - mu[None, :, :]
            var = (r[:, :, None] * (diff ** 2)).sum(axis=0) / Nk[:, None]
            var = np.maximum(var, self.cfg.reg_covar)
            logvar = np.log(var)
            # update params
            self.mu_, self.logvar_, self.alpha_ = mu, logvar, np.log(pi + 1e-12)
            # Compute log-likelihood of unified objective
            ll_l = np.log(np.maximum(1e-300, self._class_mixture_density(X_l, y_l))).sum()
            ll_u = np.log(np.maximum(1e-300, self._mixture_density(X_u))).sum()
            ll = ll_l + ll_u
            if verbose:
                print(f"[EM] iter {it:03d} ll={ll:.3f}")
            if np.abs(ll - prev_ll) < tol * (1 + np.abs(prev_ll)):
                break
            prev_ll = ll
        return self

    # --------- GD-based M-step training ---------
    def fit_gd(self, X_l: np.ndarray, y_l: np.ndarray, X_u: np.ndarray, max_iter=50, tol=1e-4,
               gd_steps=5, lr_mu=0.1, lr_var=0.05, lr_pi=0.1, verbose=False):
        X = np.vstack([X_l, X_u])
        y = np.concatenate([y_l, np.full(X_u.shape[0], -1, dtype=int)])
        if self.mu_ is None:
            self._init_params(X)
        prev_ll = -np.inf
        for it in range(max_iter):
            # E-step
            r = self._e_step(X, y)
            N, D = X.shape
            Nk = r.sum(axis=0) + 1e-12
            # GD on Q with fixed r
            for _ in range(gd_steps):
                pi = softmax(self.alpha_, axis=0)
                s2 = np.exp(self.logvar_)
                # grads
                # alpha (pi): g = sum_n r_nk - N*pi_k
                g_alpha = Nk - N * pi
                # mu: sum_n r_nk * (x - mu)/s2
                diff = X[:, None, :] - self.mu_[None, :, :]
                g_mu = (r[:, :, None] * (diff / s2[None, :, :])).sum(axis=0)
                # gamma (logvar): 0.5*sum_n r_nk*((x-mu)^2/s2 - 1)
                g_logvar = 0.5 * (r[:, :, None] * ((diff ** 2) / s2[None, :, :] - 1.0)).sum(axis=0)
                # step (normalize by Nk to stabilize)
                self.alpha_ += lr_pi * g_alpha / N
                self.mu_ += lr_mu * (g_mu / Nk[:, None])
                self.logvar_ += lr_var * (g_logvar / Nk[:, None])
                # clamp logvar to avoid too small variances
                self.logvar_ = np.clip(self.logvar_, np.log(self.cfg.reg_covar), 10.0)
            # objective
            ll_l = np.log(np.maximum(1e-300, self._class_mixture_density(X_l, y_l))).sum()
            ll_u = np.log(np.maximum(1e-300, self._mixture_density(X_u))).sum()
            ll = ll_l + ll_u
            if verbose:
                print(f"[GD] iter {it:03d} ll={ll:.3f}")
            if np.abs(ll - prev_ll) < tol * (1 + np.abs(prev_ll)):
                break
            prev_ll = ll
        return self

    # --------- densities & predict ---------
    def _mixture_density(self, X: np.ndarray) -> np.ndarray:
        log_pi = self.alpha_ - logsumexp(self.alpha_, axis=0)
        log_p = log_gaussian_diag(X, self.mu_, self.logvar_) + log_pi[None, :]
        return np.exp(logsumexp(log_p, axis=1))

    def _class_mixture_density(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        # For each sample i, sum only components of its class
        log_pi = self.alpha_ - logsumexp(self.alpha_, axis=0)
        log_p = log_gaussian_diag(X, self.mu_, self.logvar_) + log_pi[None, :]
        N = X.shape[0]
        out = np.zeros(N, dtype=float)
        for i in range(N):
            c = int(y[i])
            ks = self._class_components(c)
            out[i] = np.exp(logsumexp(log_p[i, ks], axis=0))
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Score per class by summing component densities of that class
        log_pi = self.alpha_ - logsumexp(self.alpha_, axis=0)
        log_p = log_gaussian_diag(X, self.mu_, self.logvar_) + log_pi[None, :]
        scores = np.zeros((X.shape[0], self.C), dtype=float)
        for c in range(self.C):
            ks = self._class_components(c)
            scores[:, c] = np.exp(logsumexp(log_p[:, ks], axis=1))
        return np.argmax(scores, axis=1)
