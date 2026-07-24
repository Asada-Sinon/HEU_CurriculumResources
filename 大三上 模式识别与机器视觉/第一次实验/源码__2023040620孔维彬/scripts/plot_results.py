import os
import csv
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def load_results(results_dir: str):
    rows = []
    for fn in os.listdir(results_dir):
        if not fn.startswith("results_") or not fn.endswith(".csv"):
            continue
        path = os.path.join(results_dir, fn)
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                r['mean_acc'] = float(r['mean_acc'])
                r['var_acc'] = float(r['var_acc'])
                r['label_per_class'] = int(r['label_per_class'])
                rows.append(r)
    return rows


def load_results_multi(results_dirs):
    rows = []
    for d in results_dirs:
        if not os.path.isdir(d):
            continue
        rows.extend(load_results(d))
    return rows


def _dedup_by_label(items):
    """If multiple rows share the same label_per_class, keep the one with highest mean_acc.
    This helps remove stale / anomalous duplicates when results CSVs accumulate multiple runs."""
    best = {}
    for it in items:
        L = it['label_per_class']
        if (L not in best) or (it['mean_acc'] > best[L]['mean_acc']):
            best[L] = it
    return list(best.values())


def plot_by_dataset_solver(rows, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    # group by (dataset, solver, m, pca_dim)
    groups = defaultdict(list)
    for r in rows:
        key = (r['dataset'], r['solver'], r['m'], r['pca_dim'])
        groups[key].append(r)
    images = []
    for (dataset, solver, m, pca), items in groups.items():
        # Deduplicate potential multiple entries (keep highest accuracy per label)
        items = _dedup_by_label(items)
        items = sorted(items, key=lambda x: x['label_per_class'])
        xs = [it['label_per_class'] for it in items]
        ys = [it['mean_acc'] for it in items]
        stds = [np.sqrt(it['var_acc']) for it in items]
        plt.figure(figsize=(5, 3.2))
        plt.errorbar(xs, ys, yerr=stds, fmt='-o', capsize=4)
        plt.ylim(0.0, 1.0)
        plt.xlabel('Labeled per class')
        plt.ylabel('Accuracy')
        plt.title(f"{dataset.upper()} | {solver.upper()} | m={m} | pca={pca}")
        fn = f"plot_{dataset}_{solver}_m{m}_pca{pca}.png"
        out_path = os.path.join(out_dir, fn)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        images.append(out_path)
    return images


def plot_compare_across_solvers(rows, out_dir: str):
    """
    Draw comparison plots by (dataset, m, pca_dim), with multiple solvers
    (e.g., 'em', 'gd', 'baseline') in the same figure as L varies.
    Saves to: figures/plot_{dataset}_compare_m{m}_pca{p}.png
    """
    # group by (dataset, m, pca_dim) first, then split lines by solver
    groups = {}
    for r in rows:
        key = (r['dataset'], r['m'], r['pca_dim'])
        groups.setdefault(key, []).append(r)

    images = []
    for (dataset, m, pca), items in groups.items():
        # organize by solver
        by_solver = {}
        for it in items:
            by_solver.setdefault(it['solver'], []).append(it)

        plt.figure(figsize=(6, 4))
        for solver, solver_items in sorted(by_solver.items()):
            solver_items = _dedup_by_label(solver_items)
            solver_items = sorted(solver_items, key=lambda x: x['label_per_class'])
            xs = [it['label_per_class'] for it in solver_items]
            ys = [it['mean_acc'] for it in solver_items]
            stds = [np.sqrt(it['var_acc']) for it in solver_items]
            plt.errorbar(xs, ys, yerr=stds, marker='o', capsize=3, label=solver.upper())

        plt.xlabel('Labels per class (L)')
        plt.ylabel('Accuracy')
        plt.title(f"{dataset.upper()} | Compare solvers | m={m} | pca={pca}")
        plt.grid(True, alpha=0.3)
        plt.legend()

        os.makedirs(out_dir, exist_ok=True)
        fn = f"plot_{dataset}_compare_m{m}_pca{pca}.png"
        path = os.path.join(out_dir, fn)
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        images.append(path)

    return images


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', nargs='+', default=['results'], help='One or more result directories to merge')
    parser.add_argument('--out-dir', type=str, default='figures')
    args = parser.parse_args()
    # args.results_dir is a list (supports single dir as well)
    rows = load_results_multi(args.results_dir)
    if not rows:
        print('No results found.')
        return
    images_single = plot_by_dataset_solver(rows, args.out_dir)
    images_compare = plot_compare_across_solvers(rows, args.out_dir)
    print('Saved figures (single-solver):')
    for p in images_single:
        print(' ', p)
    print('Saved figures (compare):')
    for p in images_compare:
        print(' ', p)


if __name__ == '__main__':
    main()
