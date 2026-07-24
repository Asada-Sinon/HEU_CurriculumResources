"""\n1.py  —— 课程项目主运行脚本（主入口）\n================================================\n功能概述：\n1. 读取或生成数据集（MNIST / CIFAR-10 / Synthetic）。\n2. 可选进行 PCA 降维。\n3. 按照“5 次 5 折”重复分层交叉验证划分数据。\n4. 在每个训练折里：为每个类别随机挑选 L 个有标签样本，其余当无标签。\n5. 根据参数选择训练：\n   - 半监督 GMM（EM）\n   - 半监督 GMM（GD 替代 M 步）\n   - 无监督基线：KMeans + NCC\n6. 在测试折上评估准确率，聚合所有折的均值与方差，写入 CSV。\n7. 具备“断点续跑”机制：同一个 (dataset, solver, m, pca_dim) 下已存在的某个 L 结果不会重复计算。\n8. 提供 quick 模式：抽样少量数据 + 减少迭代，加速冒烟测试。\n\n输出：results/results_{dataset}_{solver}_m{m}_pca{p}.csv\n列含义：dataset, solver, m, pca_dim, label_per_class, mean_acc, var_acc, time_s\n\n阅读提示：本文件结构尽量线性化，适合由上到下快速理解；核心训练逻辑在 run_experiment 中。\n"""

import os
import argparse
import time
from typing import List

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold

from semigmm import (
    SemiGMM,
    SemiGMMConfig,
    load_mnist,
    load_cifar10,
    generate_synthetic,
    select_labeled_per_class,  # 挑选每类 L 个标签的工具函数
)
from semigmm.utils import accuracy
from semigmm.baseline import KMeansNCC, KMeansNCCConfig


def load_dataset(name: str, data_dir: str):
    """根据名称加载数据集。

    返回 (X_train, y_train, X_test, y_test)。\n+    Synthetic 为程序生成（适合验证正确性）。\n+    注意：Synthetic 这里设置 n_classes=3（可根据需要改为 2）。\n+    """
    name = name.lower()
    if name == "mnist":
        return load_mnist(data_dir)
    elif name == "cifar10":
        return load_cifar10(data_dir)
    elif name == "synthetic":
        return generate_synthetic(n_classes=3, samples_per_class=120, dim=20, seed=0)
    else:
        raise ValueError(f"Unsupported dataset: {name}")


def prepare_data(X_train, y_train, X_test, y_test, pca_dim: int):
    """对训练/测试数据做可选的 PCA 降维（仅在训练集 fit，测试集 transform）。

    条件：pca_dim 需小于原始维度且 > 0，否则直接返回原始数据。\n+    返回的矩阵强制转成 float32，节省内存。\n+    """
    if pca_dim is not None and pca_dim > 0 and pca_dim < X_train.shape[1]:
        pca = PCA(n_components=pca_dim, whiten=False, random_state=0)
        Xtr = pca.fit_transform(X_train)
        Xte = pca.transform(X_test)
        return Xtr.astype(np.float32), y_train, Xte.astype(np.float32), y_test
    return X_train.astype(np.float32), y_train, X_test.astype(np.float32), y_test


def run_experiment(
    dataset: str,
    data_dir: str,
    solver: str,
    components_per_class: int,
    pca_dim: int,
    repeats: int,
    folds: int,
    label_budgets: List[int],
    max_iter: int,
    tol: float,
    gd_steps: int,
    lr_mu: float,
    lr_var: float,
    lr_pi: float,
    results_dir: str,
    quick: bool,
    baseline_mode: str,
):
    """主实验函数：组织交叉验证、半监督划分、训练与结果写入。

    参数说明（常用）：
    - dataset: 数据集名称
    - solver: 'em' / 'gd' / 'baseline'
    - components_per_class: 每个类别的混合分量数 M
    - label_budgets: 需要评测的每类标签数量列表，如 [5,10,20]
    - quick: True 时走“快速模式”抽样少量数据做冒烟测试
    - baseline_mode: 基线中 KMeans 的模式（按类 / 全局）
    """
    # 1) 读取（或生成）数据
    X_train, y_train, X_test, y_test = load_dataset(dataset, data_dir)
    n_classes = int(y_train.max() + 1)

    # 2) 合并 train+test 然后做重复分层 K 折划分
    #    好处：每个折都能覆盖原始数据分布；之后再切分成“训练折”和“测试折”
    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    # 3) Quick 模式：减少数据规模、迭代次数，只保留最小的一个标签预算
    if quick:
        repeats = 1               # 只做 1 次重复
        folds = 2                 # 2 折即可
        label_budgets = [min(label_budgets[0], 5)]  # 只保留第一个 L
        max_iter = min(max_iter, 10)
        quick_train_n = 5000      # 训练子采样上限
        quick_test_n = 2000       # 测试子采样上限
    else:
        quick_train_n = None
        quick_test_n = None

    rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=0)

    # 4) 准备输出文件路径（一个 solver + pca_dim + m 一份 CSV）
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(
        results_dir,
        f"results_{dataset}_{solver}_m{components_per_class}_pca{pca_dim}.csv",
    )

    # 5) 断点续跑：如果该 CSV 已存在，就把已经完成的 label_per_class 收集起来
    existing_labels = set()
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        try:
            with open(out_path, "r", encoding="utf-8") as rf:
                lines = rf.read().strip().splitlines()
            for line in lines[1:]:  # 跳过表头
                parts = line.split(",")
                if len(parts) >= 5:
                    try:
                        existing_labels.add(int(parts[4]))
                    except Exception:
                        pass
        except Exception:
            pass  # 若损坏则忽略，重新写

    write_header = not (os.path.exists(out_path) and os.path.getsize(out_path) > 0)
    mode = "a" if not write_header else "w"

    # 6) 外层按不同 L 循环
    with open(out_path, mode, encoding="utf-8") as f:
        if write_header:
            f.write("dataset,solver,m,pca_dim,label_per_class,mean_acc,var_acc,time_s\n")
        for L in label_budgets:
            if L in existing_labels:
                # 已经有这个 L 的结果了 → 跳过
                continue
            accs = []  # 收集所有折的准确率
            t0 = time.time()

            # 7) 交叉验证循环
            for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
                Xtr_raw, ytr_raw = X[train_idx], y[train_idx]
                Xte_raw, yte = X[test_idx], y[test_idx]

                # Quick 模式下对每个折的训练/测试再做一次随机子采样
                if quick:
                    rng_q = np.random.RandomState(1234 + fold_idx)
                    if quick_train_n and len(Xtr_raw) > quick_train_n:
                        sel_tr = rng_q.choice(len(Xtr_raw), size=quick_train_n, replace=False)
                        Xtr_raw, ytr_raw = Xtr_raw[sel_tr], ytr_raw[sel_tr]
                    if quick_test_n and len(Xte_raw) > quick_test_n:
                        sel_te = rng_q.choice(len(Xte_raw), size=quick_test_n, replace=False)
                        Xte_raw, yte = Xte_raw[sel_te], yte[sel_te]

                # 8) PCA 仅在训练折拟合，再 transform 测试折（防止数据泄漏）
                Xtr, ytr, Xte, yte = prepare_data(Xtr_raw, ytr_raw, Xte_raw, yte, pca_dim)

                # 9) 半监督：每个类别随机选 L 个有标签样本，其余为无标签
                rng = np.random.RandomState(42 + fold_idx + L)
                lab_idx, unlab_idx = select_labeled_per_class(ytr, n_classes, L, rng)
                X_l, y_l = Xtr[lab_idx], ytr[lab_idx]
                X_u = Xtr[unlab_idx]

                # 10) 根据 solver 训练
                if solver in {"em", "gd"}:
                    cfg = SemiGMMConfig(
                        n_classes=n_classes,
                        components_per_class=components_per_class,
                        random_state=fold_idx,
                    )
                    model = SemiGMM(cfg)
                    if solver == "em":
                        # EM：交替 E/M 步，使用闭式更新
                        model.fit_em(
                            X_l,
                            y_l,
                            X_u,
                            max_iter=max_iter,
                            tol=tol,
                            verbose=False,
                        )
                    else:
                        # GD：用梯度替换 M 步的小循环
                        model.fit_gd(
                            X_l,
                            y_l,
                            X_u,
                            max_iter=max_iter,
                            tol=tol,
                            gd_steps=gd_steps,
                            lr_mu=lr_mu,
                            lr_var=lr_var,
                            lr_pi=lr_pi,
                            verbose=False,
                        )
                    y_pred = model.predict(Xte)
                elif solver == "baseline":
                    # 基准方法：用少量标签指导 KMeans 聚类与 NCC 判别
                    bcfg = KMeansNCCConfig(
                        n_classes=n_classes,
                        components_per_class=components_per_class,
                        mode=baseline_mode,
                        random_state=fold_idx,
                    )
                    base = KMeansNCC(bcfg).fit(X_l, y_l, Xtr)
                    y_pred = base.predict(Xte)
                else:
                    raise ValueError("solver must be 'em', 'gd', or 'baseline'")

                # 11) 评估本折准确率
                acc = accuracy(yte, y_pred)
                accs.append(acc)

            # 12) 汇总本 L 的统计量
            t1 = time.time()
            mean_acc = float(np.mean(accs))
            var_acc = float(np.var(accs))
            f.write(
                f"{dataset},{solver},{components_per_class},{pca_dim},{L},{mean_acc:.4f},{var_acc:.6f},{t1-t0:.1f}\n"
            )
            f.flush()  # 及时写盘，防止中途意外退出

    # 13) 最终提示路径
    print(f"结果已保存到: {out_path}")


if __name__ == "__main__":
    # 命令行参数解析：保持与报告说明一致，便于复现
    parser = argparse.ArgumentParser(description="Semi-supervised GMM experiments")
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=["synthetic", "mnist", "cifar10"],
        help="选择数据集",
    )
    parser.add_argument("--data-dir", type=str, default="data", help="数据根目录")
    parser.add_argument(
        "--solver",
        type=str,
        default="em",
        choices=["em", "gd", "baseline"],
        help="训练方式：EM / GD / baseline",
    )
    parser.add_argument(
        "--components-per-class", type=int, default=2, help="每个类别的混合分量数 M"
    )
    parser.add_argument(
        "--pca-dim", type=int, default=50, help="PCA 降维后的维度 (<= 原始维度)"
    )
    parser.add_argument("--repeats", type=int, default=5, help="重复次数 (外层) ")
    parser.add_argument("--folds", type=int, default=5, help="交叉验证折数")
    parser.add_argument(
        "--label-budgets",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="每类标签数量列表",
    )
    parser.add_argument("--max-iter", type=int, default=50, help="最大迭代次数 (EM/GD 外层)")
    parser.add_argument("--tol", type=float, default=1e-4, help="收敛阈值")
    parser.add_argument("--gd-steps", type=int, default=5, help="GD 模式每轮内部梯度步数")
    parser.add_argument("--lr-mu", type=float, default=0.1, help="GD：均值学习率")
    parser.add_argument("--lr-var", type=float, default=0.05, help="GD：方差(对数)学习率")
    parser.add_argument("--lr-pi", type=float, default=0.1, help="GD：混合权重学习率")
    parser.add_argument("--results-dir", type=str, default="results", help="结果输出目录")
    parser.add_argument(
        "--quick", action="store_true", help="快速模式（抽样数据 + 减少迭代）"
    )
    parser.add_argument(
        "--baseline-mode",
        type=str,
        default="per-class",
        choices=["per-class", "global"],
        help="baseline 聚类方式：按类分别 / 全局一次",
    )

    args = parser.parse_args()

    run_experiment(
        dataset=args.dataset,
        data_dir=args.data_dir,
        solver=args.solver,
        components_per_class=args.components_per_class,
        pca_dim=args.pca_dim,
        repeats=args.repeats,
        folds=args.folds,
        label_budgets=args.label_budgets,
        max_iter=args.max_iter,
        tol=args.tol,
        gd_steps=args.gd_steps,
        lr_mu=args.lr_mu,
        lr_var=args.lr_var,
        lr_pi=args.lr_pi,
        results_dir=args.results_dir,
        quick=args.quick,
        baseline_mode=args.baseline_mode,
    )
