import os
import sys
import argparse
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Robust import of local package
try:
    import semigmm as sg
except ModuleNotFoundError:
    import importlib.util as _ilu
    pkg_init = os.path.join(ROOT, 'semigmm', '__init__.py')
    spec = _ilu.spec_from_file_location('semigmm', pkg_init)
    if spec is None or spec.loader is None:
        raise
    sg = _ilu.module_from_spec(spec)
    spec.loader.exec_module(sg)


def load_dataset(name: str, data_root: str):
    name = name.lower()
    if name == 'mnist':
        return sg.load_mnist(os.path.join(data_root, 'mnist'))
    if name == 'cifar10':
        return sg.load_cifar10(os.path.join(data_root, 'cifar-10-batches-py'))
    raise ValueError(f'Unsupported dataset: {name}')


def pca_reduce_2d(
    X: np.ndarray,
    n_components: int = 2,
    random_state: int = 0,
    whiten: bool = False,
):
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components, random_state=random_state, whiten=whiten)
        X2 = pca.fit_transform(X)
        return X2.astype(np.float32), pca
    except Exception:
        rng = np.random.RandomState(random_state)
        W = rng.randn(X.shape[1], n_components)
        X2 = X @ W
        return X2.astype(np.float32), None


def tsne_reduce_2d(
    X: np.ndarray,
    random_state: int = 0,
    perplexity: float = 30.0,
    pca_init_dim: int = 50,
):
    """Reduce to 2D using t-SNE; optionally pre-reduce with PCA to 50 dims for speed."""
    try:
        # optional pre-PCA for speed/stability
        D = X.shape[1]
        if D > pca_init_dim:
            Xp, _ = pca_reduce_2d(X, n_components=pca_init_dim, random_state=random_state, whiten=False)
        else:
            Xp = X
        from sklearn.manifold import TSNE
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init='pca',
            learning_rate='auto',
            random_state=random_state,
            n_iter=1000,
            verbose=0,
        )
        X2 = tsne.fit_transform(Xp)
        return X2.astype(np.float32)
    except Exception:
        # Fall back to random projection if TSNE not available
        rng = np.random.RandomState(random_state)
        W = rng.randn(X.shape[1], 2)
        return (X @ W).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description='2D feature scatter plot by true labels (PCA / PCA-whiten / t-SNE)')
    parser.add_argument('--dataset', type=str, required=True, choices=['mnist', 'cifar10'])
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--limit', type=int, default=8000, help='max total points to plot for clarity')
    parser.add_argument('--limit-per-class', type=int, default=None, help='balanced sampling per class (overrides --limit if set)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--method', type=str, default='pca', choices=['pca', 'pca-whiten', 'tsne'])
    parser.add_argument('--perplexity', type=float, default=30.0, help='t-SNE perplexity')
    args = parser.parse_args()

    Xtr, ytr, Xte, yte = load_dataset(args.dataset, args.data_dir)
    X = np.vstack([Xtr, Xte]).astype(np.float32)
    y = np.concatenate([ytr, yte]).astype(int)

    # Balanced subsampling (before reduction to speed up t-SNE)
    rng = np.random.RandomState(args.seed)
    if args.limit_per_class is not None:
        classes = np.unique(y)
        idx_list = []
        for c in classes:
            idx_c = np.where(y == c)[0]
            take = min(len(idx_c), args.limit_per_class)
            if take > 0:
                idx_pick = rng.choice(idx_c, size=take, replace=False)
                idx_list.append(idx_pick)
        if idx_list:
            idx = np.concatenate(idx_list)
        else:
            idx = np.arange(X.shape[0])
    else:
        n = X.shape[0]
        if args.limit and n > args.limit:
            idx = rng.choice(n, size=args.limit, replace=False)
        else:
            idx = np.arange(n)

    X = X[idx]
    y = y[idx]

    if args.method == 'tsne':
        X2 = tsne_reduce_2d(X, random_state=args.seed, perplexity=args.perplexity)
        method_title = f't-SNE (perp={args.perplexity:g})'
        suffix = 'tsne'
    else:
        whiten = (args.method == 'pca-whiten')
        X2, _ = pca_reduce_2d(X, n_components=2, random_state=args.seed, whiten=whiten)
        method_title = 'PCA (whitened)' if whiten else 'PCA'
        suffix = 'pca_whiten' if whiten else 'pca'

    cmap = plt.get_cmap('tab10')
    colors = [cmap(int(c) % 10) for c in y]

    plt.figure(figsize=(6.2, 5.2))
    plt.scatter(X2[:, 0], X2[:, 1], c=colors, s=8, alpha=0.85, linewidths=0)
    plt.title(f'{args.dataset.upper()} 2D ({method_title}) — scatter by true label')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    os.makedirs('figures', exist_ok=True)
    out_path = args.out or os.path.join('figures', f'{suffix}_{args.dataset}_scatter.png')
    plt.savefig(out_path, dpi=180)
    plt.close()
    print('Saved:', out_path)


if __name__ == '__main__':
    main()
