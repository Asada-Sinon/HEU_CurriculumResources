from typing import List, Tuple

import numpy as np


def select_labeled_per_class(y: np.ndarray, n_classes: int, per_class: int, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given full training indices (implicit 0..N-1) and labels y, sample exactly per_class labeled indices per class.
    Returns labeled_idx, unlabeled_idx.
    """
    labeled_idx: List[int] = []
    for c in range(n_classes):
        idx_c = np.where(y == c)[0]
        if idx_c.size < per_class:
            raise ValueError(f"Class {c} has only {idx_c.size} samples < {per_class}")
        sel = rng.choice(idx_c, size=per_class, replace=False)
        labeled_idx.extend(sel.tolist())
    labeled_idx = np.array(sorted(labeled_idx), dtype=np.int64)
    all_idx = np.arange(y.shape[0], dtype=np.int64)
    mask = np.ones_like(all_idx, dtype=bool)
    mask[labeled_idx] = False
    unlabeled_idx = all_idx[mask]
    return labeled_idx, unlabeled_idx
