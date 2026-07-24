from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans


@dataclass
class KMeansNCCConfig:
    n_classes: int
    components_per_class: int = 1
    mode: str = "per-class"  # "per-class" or "global"
    random_state: Optional[int] = None


class KMeansNCC:
    """
    基准方法：KMeans + 最近质心分类（Nearest Centroid Classification）。

    两种模式：
    - per-class: 对每个类别的已标注子集分别做 KMeans，得到每类 K=components_per_class 个质心；预测时取最近质心的类别。
    - global: 在训练折（有/无标签混合）上整体做 KMeans（K=C×M），再用已标注子集多数投票将簇映射到类别；预测时使用映射后的簇标签。
    """

    def __init__(self, cfg: KMeansNCCConfig):
        self.cfg = cfg
        self.centers_: Optional[np.ndarray] = None  # [K, D]
        self.center_labels_: Optional[np.ndarray] = None  # [K]

    def fit(self, X_l: np.ndarray, y_l: np.ndarray, X_all_train: np.ndarray) -> "KMeansNCC":
        if self.cfg.mode == "per-class":
            self._fit_per_class(X_l, y_l)
        elif self.cfg.mode == "global":
            self._fit_global(X_l, y_l, X_all_train)
        else:
            raise ValueError("KMeansNCC.mode must be 'per-class' or 'global'")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.centers_ is None or self.center_labels_ is None:
            raise RuntimeError("Model not fitted")
        # squared Euclidean distances
        d2 = _cdist_sq(X, self.centers_)  # [N, K]
        idx = np.argmin(d2, axis=1)
        return self.center_labels_[idx]

    # ---- internal helpers ----
    def _fit_per_class(self, X_l: np.ndarray, y_l: np.ndarray):
        centers = []
        labels = []
        rs = self.cfg.random_state
        for c in range(self.cfg.n_classes):
            Xc = X_l[y_l == c]
            if Xc.shape[0] == 0:
                raise ValueError(f"No labeled samples for class {c} in per-class mode")
            k = max(1, min(self.cfg.components_per_class, Xc.shape[0]))
            if k == 1:
                # 单中心（避免 L < k 导致失败）
                ctr = Xc.mean(axis=0, keepdims=True)
            else:
                km = KMeans(n_clusters=k, n_init=10, random_state=rs)
                km.fit(Xc)
                ctr = km.cluster_centers_
            centers.append(ctr)
            labels.append(np.full((ctr.shape[0],), c, dtype=np.int64))
        self.centers_ = np.vstack(centers).astype(np.float32)
        self.center_labels_ = np.concatenate(labels)

    def _fit_global(self, X_l: np.ndarray, y_l: np.ndarray, X_all: np.ndarray):
        # 1) 全体样本做 KMeans，K = C*M
        K = int(self.cfg.n_classes * self.cfg.components_per_class)
        K = max(1, min(K, max(1, X_all.shape[0])))
        km = KMeans(n_clusters=K, n_init=10, random_state=self.cfg.random_state)
        km.fit(X_all)
        centers = km.cluster_centers_.astype(np.float32)  # [K, D]

        # 2) 用已标注样本做簇到类别的多数投票映射
        if X_l.shape[0] == 0:
            raise ValueError("Global mode requires at least some labeled samples for cluster-to-class mapping")
        # 找每个 labeled 最近的簇索引
        d2 = _cdist_sq(X_l, centers)  # [L, K]
        nearest = np.argmin(d2, axis=1)
        votes = np.zeros((K, self.cfg.n_classes), dtype=np.int64)
        for idx, lab in zip(nearest, y_l):
            votes[idx, int(lab)] += 1
        # 若某些簇没有任何投票，回退为“最近的 labeled 样本类别”
        map_labels = np.full((K,), -1, dtype=np.int64)
        for k in range(K):
            if votes[k].sum() > 0:
                map_labels[k] = int(np.argmax(votes[k]))
            else:
                # 选距离该中心最近的 labeled 样本
                d2_k = _cdist_sq(centers[k:k+1], X_l)[0]  # [L]
                j = int(np.argmin(d2_k))
                map_labels[k] = int(y_l[j])

        self.centers_ = centers
        self.center_labels_ = map_labels


def _cdist_sq(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance matrix between rows of A and rows of B."""
    # (a-b)^2 = a^2 + b^2 - 2ab
    a2 = np.sum(A * A, axis=1, keepdims=True)  # [Na, 1]
    b2 = np.sum(B * B, axis=1, keepdims=True).T  # [1, Nb]
    ab = A @ B.T
    d2 = a2 + b2 - 2.0 * ab
    # 数值稳定（可能出现极小的负数）
    np.maximum(d2, 0.0, out=d2)
    return d2
