import os
import sys
import argparse
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


# Ensure project root (workspace root) is importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import local package robustly
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
    if name == 'synthetic':
        return sg.generate_synthetic(n_classes=3, samples_per_class=200, dim=2, seed=0)
    raise ValueError(f'Unsupported dataset: {name}')


def pca_reduce_2d(X: np.ndarray, n_components: int = 2, random_state: int = 0) -> Tuple[np.ndarray, Optional[object]]:
    if X.shape[1] == n_components:
        return X.astype(np.float32), None
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components, random_state=random_state)
        X2 = pca.fit_transform(X)
        return X2.astype(np.float32), pca
    except Exception:
        rng = np.random.RandomState(random_state)
        W = rng.randn(X.shape[1], n_components)
        X2 = X @ W
        return X2.astype(np.float32), None


def compute_responsibilities(model, X: np.ndarray) -> np.ndarray:
    # Use internal E-step without label masking for clean visualization
    return model._e_step(X, None)


def class_from_component(model, k: int) -> int:
    return k // model.M


def class_responsibilities_from_r(model, r: np.ndarray) -> np.ndarray:
    C, M = model.C, model.M
    N = r.shape[0]
    R = np.zeros((N, C), dtype=float)
    for c in range(C):
        ks = slice(c * M, (c + 1) * M)
        R[:, c] = r[:, ks].sum(axis=1)
    return R


def draw_cov_ellipse_axis_aligned(ax, mu: np.ndarray, var_diag: np.ndarray, color, lw=1.2, alpha=0.25, nsig: float = 1.0):
    w = 2.0 * nsig * float(np.sqrt(max(1e-12, var_diag[0])))
    h = 2.0 * nsig * float(np.sqrt(max(1e-12, var_diag[1])))
    e = Ellipse(xy=(float(mu[0]), float(mu[1])), width=w, height=h, angle=0.0, facecolor=color, edgecolor=color, lw=lw, alpha=alpha)
    ax.add_patch(e)


def draw_cov_ellipse_rotated(ax, mu: np.ndarray, cov2: np.ndarray, color, lw=1.2, alpha=0.25, nsig: float = 1.0):
    # cov2: 2x2 positive semi-definite
    try:
        vals, vecs = np.linalg.eigh(cov2)
        vals = np.clip(vals, 1e-12, None)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
        w = 2.0 * nsig * float(np.sqrt(vals[0]))
        h = 2.0 * nsig * float(np.sqrt(vals[1]))
        e = Ellipse(xy=(float(mu[0]), float(mu[1])), width=w, height=h, angle=angle, facecolor=color, edgecolor=color, lw=lw, alpha=alpha)
        ax.add_patch(e)
    except Exception:
        # Fallback to axis-aligned if covariance is not well-conditioned
        draw_cov_ellipse_axis_aligned(ax, mu, np.array([cov2[0, 0], cov2[1, 1]]), color=color, lw=lw, alpha=alpha, nsig=nsig)


def mix_colors(weights: np.ndarray, base_colors: np.ndarray):
    weights = np.maximum(0.0, weights)
    if weights.sum() <= 0:
        return (0.5, 0.5, 0.5)
    w = weights / weights.sum()
    col = (w[:, None] * base_colors).sum(axis=0)
    col = np.clip(col, 0.0, 1.0)
    return tuple(col.tolist())


def main():
    parser = argparse.ArgumentParser(description='Plot 2D covariance ellipses and color by posteriors')
    parser.add_argument('--dataset', type=str, default='synthetic', choices=['synthetic', 'mnist', 'cifar10'])
    parser.add_argument('--data-dir', type=str, default='data', help='dataset root directory')
    parser.add_argument('--solver', type=str, default='em', choices=['em', 'gd'])
    parser.add_argument('--m', type=int, default=2, help='components per class')
    parser.add_argument('--L', type=int, default=5, help='labels per class')
    parser.add_argument('--pca-dim', type=int, default=2, help='reduce to this dim for training/plotting (use 2 for plotting)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--limit', type=int, default=3000, help='max points to scatter for clarity')
    parser.add_argument('--u-max', type=int, default=5000, help='max unlabeled points used for training (subsample if larger)')
    parser.add_argument('--color-mode', type=str, default='argmax', choices=['argmax', 'mix'])
    parser.add_argument('--ellipse-mode', type=str, default='empirical', choices=['empirical', 'model'], help='empirical: r-weighted 2x2 covariance (rotated); model: axis-aligned from diag variances')
    parser.add_argument('--sigma-levels', type=float, nargs='+', default=[1.0, 2.0], help='ellipse sigma radii to draw')
    parser.add_argument('--show-means', action='store_true', help='draw component means as small crosses')
    parser.add_argument('--topk-per-class', type=int, default=None, help='only draw top-k weighted components per class to reduce clutter')
    parser.add_argument('--alpha-base', type=float, default=0.12, help='base alpha for low-weight components')
    parser.add_argument('--alpha-scale', type=float, default=0.30, help='additional alpha scaled by component weight')
    parser.add_argument('--point-mode', type=str, default='posterior', choices=['posterior', 'true'], help='color points by model posterior (argmax/mix) or true label')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    # Load data and form a single pool
    Xtr, ytr, Xte, yte = load_dataset(args.dataset, args.data_dir)
    X = np.vstack([Xtr, Xte]).astype(np.float32)
    y = np.concatenate([ytr, yte]).astype(np.int64)
    n_classes = int(y.max() + 1)

    # 2D projection for plotting
    if X.shape[1] != 2 or args.pca_dim != 2:
        X2, _ = pca_reduce_2d(X, n_components=2, random_state=args.seed)
    else:
        X2 = X

    # Sample labeled for visualization
    rng = np.random.RandomState(args.seed)
    lab_idx, unlab_idx = sg.select_labeled_per_class(y, n_classes, args.L, rng)
    # Subsample unlabeled for training to keep runtime manageable
    if args.u_max and unlab_idx.size > args.u_max:
        sel_u = rng.choice(unlab_idx, size=args.u_max, replace=False)
        unlab_idx = np.sort(sel_u)
    X_l, y_l = X2[lab_idx], y[lab_idx]
    X_u = X2[unlab_idx]

    cfg = sg.SemiGMMConfig(n_classes=n_classes, components_per_class=args.m, random_state=args.seed)
    model = sg.SemiGMM(cfg)
    if args.solver == 'em':
        model.fit_em(X_l, y_l, X_u, max_iter=100, tol=1e-4, verbose=False)
    else:
        model.fit_gd(X_l, y_l, X_u, max_iter=60, tol=1e-4, gd_steps=5, lr_mu=0.1, lr_var=0.05, lr_pi=0.1, verbose=False)

    # Colors
    cmap = plt.get_cmap('tab10')
    class_colors = np.array([cmap(i % 10)[:3] for i in range(n_classes)])

    plt.figure(figsize=(6, 5))
    ax = plt.gca()

    N = X2.shape[0]
    if args.limit and N > args.limit:
        sub_idx = np.random.RandomState(args.seed + 7).choice(N, size=args.limit, replace=False)
    else:
        sub_idx = np.arange(N)

    X_sub = X2[sub_idx]
    y_sub = y[sub_idx]
    # Compute responsibilities only for the subset to reduce cost
    r_sub = compute_responsibilities(model, X_sub)
    Rc_sub = class_responsibilities_from_r(model, r_sub)

    if args.point_mode == 'true':
        colors = class_colors[y_sub % n_classes]
    else:
        if args.color_mode == 'argmax':
            c_pred = Rc_sub.argmax(axis=1)
            colors = class_colors[c_pred]
        else:
            colors = np.array([mix_colors(Rc_sub[i], class_colors) for i in range(Rc_sub.shape[0])])

    labeled_mask_sub = np.isin(sub_idx, lab_idx)
    ul_mask = ~labeled_mask_sub
    ax.scatter(X_sub[ul_mask, 0], X_sub[ul_mask, 1], c=colors[ul_mask], s=12, alpha=0.8, linewidths=0)
    ax.scatter(X_sub[labeled_mask_sub, 0], X_sub[labeled_mask_sub, 1],
               c=class_colors[y_sub[labeled_mask_sub] % n_classes], s=40, marker='o', edgecolors='k', linewidths=0.7, alpha=0.95,
               label='Labeled')

    s2 = np.exp(model.logvar_)
    Nk_sub = r_sub.sum(axis=0) + 1e-12
    Nk_ratio = (Nk_sub / Nk_sub.max()) if Nk_sub.max() > 0 else Nk_sub
    # optional: only draw top-k per class
    draw_mask = np.ones(model.K, dtype=bool)
    if args.topk_per_class is not None and args.topk_per_class > 0:
        for c in range(model.C):
            ks = np.arange(c * model.M, (c + 1) * model.M)
            top_idx = ks[np.argsort(Nk_ratio[ks])[::-1][:args.topk_per_class]]
            mask_c = np.zeros_like(ks, dtype=bool)
            mask_c[np.isin(ks, top_idx)] = True
            draw_mask[ks] = mask_c

    for k in range(model.K):
        if not draw_mask[k]:
            continue
        c = class_from_component(model, k)
        base_col = np.array(class_colors[c])
        mu_k = model.mu_[k]

        # Determine opacity from component weight on the plotted subset
        weight = float(np.clip(Nk_ratio[k], 0.05, 1.0))
        edge_lw = 0.8 + 0.8 * weight

        if args.ellipse_mode == 'empirical':
            # r-weighted empirical covariance on the subset (2x2)
            w = r_sub[:, k]
            w_sum = float(w.sum())
            if w_sum <= 1e-12:
                cov2 = np.diag(s2[k][:2])
            else:
                xc = X_sub - mu_k[None, :]
                # Weighted covariance: (X^T W X) / sum(w)
                cov2 = (xc.T * w) @ xc / w_sum
            for ns in args.sigma_levels:
                alpha_s = args.alpha_base + args.alpha_scale * weight
                draw_cov_ellipse_rotated(ax, mu_k, cov2, color=tuple(base_col), alpha=alpha_s, lw=edge_lw, nsig=float(ns))
        else:
            var_k = s2[k]
            for ns in args.sigma_levels:
                alpha_s = args.alpha_base + args.alpha_scale * weight
                draw_cov_ellipse_axis_aligned(ax, mu_k, var_k, color=tuple(base_col), alpha=alpha_s, lw=edge_lw, nsig=float(ns))

    if args.show_means:
        ax.scatter(model.mu_[:, 0], model.mu_[:, 1], c=class_colors[[class_from_component(model, k) for k in range(model.K)]],
                   marker='+', s=60, linewidths=1.2, alpha=0.9, label='Component means')

    ax.set_title(f"{args.dataset.upper()} | {args.solver.upper()} | m={args.m} | L={args.L} | pca=2 | {args.ellipse_mode}")
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.grid(True, alpha=0.25)

    os.makedirs('figures', exist_ok=True)
    out_path = args.out or os.path.join('figures', f"ellipses_{args.dataset}_{args.solver}_m{args.m}_pca2_L{args.L}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print('Saved:', out_path)


if __name__ == '__main__':
    main()
