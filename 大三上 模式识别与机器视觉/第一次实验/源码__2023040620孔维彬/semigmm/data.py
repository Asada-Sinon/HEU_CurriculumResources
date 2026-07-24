import os
import pickle
import struct
from typing import Tuple

import numpy as np


def _read_idx_images(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid MNIST image file magic: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num, rows * cols)
        return data


def _read_idx_labels(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid MNIST label file magic: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data


def load_mnist(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Expect files: train-images-idx3-ubyte, train-labels-idx1-ubyte,
                  t10k-images-idx3-ubyte, t10k-labels-idx1-ubyte
    Returns X_train, y_train, X_test, y_test with X in float32 scaled to [0,1].
    """
    paths = {
        "train_images": os.path.join(data_dir, "train-images-idx3-ubyte"),
        "train_labels": os.path.join(data_dir, "train-labels-idx1-ubyte"),
        "test_images": os.path.join(data_dir, "t10k-images-idx3-ubyte"),
        "test_labels": os.path.join(data_dir, "t10k-labels-idx1-ubyte"),
    }
    for k, p in paths.items():
        if not os.path.isfile(p):
            raise FileNotFoundError(f"MNIST file missing: {p}")
    X_train = _read_idx_images(paths["train_images"]).astype(np.float32) / 255.0
    y_train = _read_idx_labels(paths["train_labels"]).astype(np.int64)
    X_test = _read_idx_images(paths["test_images"]).astype(np.float32) / 255.0
    y_test = _read_idx_labels(paths["test_labels"]).astype(np.int64)
    return X_train, y_train, X_test, y_test


def _unpickle_cifar(file: str):
    with open(file, "rb") as fo:
        dict_ = pickle.load(fo, encoding="latin1")
    return dict_


def load_cifar10(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Expect CIFAR-10 python version: files data_batch_1..5, test_batch under data_dir.
    Returns flattened X in float32 scaled to [0,1].
    """
    train_data = []
    train_labels = []
    for i in range(1, 6):
        path = os.path.join(data_dir, f"data_batch_{i}")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"CIFAR-10 file missing: {path}")
        batch = _unpickle_cifar(path)
        train_data.append(batch["data"])  # shape (10000, 3072)
        train_labels.extend(batch["labels"])
    X_train = np.concatenate(train_data, axis=0).astype(np.float32) / 255.0
    y_train = np.array(train_labels, dtype=np.int64)

    test_path = os.path.join(data_dir, "test_batch")
    if not os.path.isfile(test_path):
        raise FileNotFoundError(f"CIFAR-10 file missing: {test_path}")
    test_batch = _unpickle_cifar(test_path)
    X_test = test_batch["data"].astype(np.float32) / 255.0
    y_test = np.array(test_batch["labels"], dtype=np.int64)

    return X_train, y_train, X_test, y_test


def generate_synthetic(n_classes=3, samples_per_class=200, dim=10, seed=0):
    rng = np.random.RandomState(seed)
    means = rng.randn(n_classes, dim) * 2.0
    covs = np.array([np.diag(rng.rand(dim) * 0.5 + 0.2) for _ in range(n_classes)])
    X_list = []
    y_list = []
    for c in range(n_classes):
        Xc = rng.multivariate_normal(means[c], covs[c], size=samples_per_class)
        yc = np.full(samples_per_class, c, dtype=np.int64)
        X_list.append(Xc)
        y_list.append(yc)
    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list)
    # Simple train/test split
    idx = rng.permutation(X.shape[0])
    n_train = int(0.8 * X.shape[0])
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]
