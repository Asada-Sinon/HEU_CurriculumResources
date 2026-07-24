# Semi-GMM 半监督高斯混合模型课程项目

> 课程：PRCV 第一次课程项目  
> 截止：2025-09-27 23:59  
> 主题：在少量有标签 + 大量无标签场景下，用半监督高斯混合模型进行模式分类

## 目录
- 项目简介
- 快速上手 (Quick Start)
- 文件结构
- 主要方法说明
- 运行实验
- 结果与典型表现
- 常见问题 (FAQ)
- 许可与引用

## 项目简介
本项目实现一个半监督 Gaussian Mixture Model (GMM) 分类框架，支持两种参数学习方式：
- EM (Expectation-Maximization) 闭式更新
- GD (Gradient Descent) 梯度更新（模拟无闭式解的混合模型情形）
并提供一个无监督基准：KMeans + 最近类中心 (NCC)。项目包含数据准备、PCA / t-SNE 可视化、协方差椭圆绘制、结果自动汇总。

支持数据集：
- MNIST（手写数字）
- CIFAR-10（彩色小图）

## 快速上手
### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 准备数据
将 MNIST idx 文件放入 `data/mnist/`，CIFAR-10 Python 版本批次放入 `data/cifar-10-batches-py/`。仓库已含示例结构。Synthetic 数据自动生成。

### 3. 运行一个快速实验
```bash
# 合成数据（快速模式）
python 1.py --dataset synthetic --solver em --pca-dim 10 --label-budgets 5 --quick

# MNIST 半监督 (EM)
python 1.py --dataset mnist --solver em --pca-dim 50 --label-budgets 5 10 20

# CIFAR-10 半监督 (GD)
python 1.py --dataset cifar10 --solver gd --pca-dim 100 --label-budgets 5 10 20
```

### 4. 生成图表
使用 VS Code 任务（终端→运行任务）或手动：
```bash
python scripts/plot_results.py --results-dir results --out-dir figures
```

## 文件结构
```
PRCV/
  1.py                  # 主运行脚本（控制实验流程）
  semigmm/              # 半监督 GMM 与基线实现
    semigmm.py          # SemiGMM (EM / GD) 主类
    baseline.py         # KMeans + NCC 基线
    data.py             # 数据加载与 synthetic 生成
    utils.py            # 工具函数（logsumexp / accuracy 等）
  scripts/
    plot_results.py     # 汇总结果折线图
    plot_ellipses.py    # 2D 椭圆与责任可视化
    plot_pca_features.py# PCA / t-SNE 散点
  results/              # CSV 结果（mean_acc / var_acc）
  figures/              # 输出图像（折线/散点/椭圆）
  data/                 # 数据目录（MNIST, CIFAR-10, synthetic）
  EXPERIMENT_REPORT.md  # 实验报告（含原理/结果/结论）
  requirements.txt      # 依赖列表
```

## 主要方法说明
### 半监督统一目标
划分有标签集 L 和无标签集 U：
```
L(Θ) = Σ_(x,y∈L) log Σ_{k∈K_y} π_k N(x|μ_k,Σ_k) + Σ_{x∈U} log Σ_{k∈K} π_k N(x|μ_k,Σ_k)
```
有标签样本只在所属类别的分量集合归一化；无标签样本在全部分量归一化。

### EM 关键步骤
1. E 步：计算责任 r_{nk}
2. M 步：
```
π_k = N_k / Σ_j N_j
μ_k = (Σ_n r_{nk} x_n) / N_k
σ^2_{kd} = (Σ_n r_{nk} (x_{nd}-μ_{kd})^2)/N_k + ε
```

### GD 版更新
保持 E 步相同，用梯度上升替代 M 步：
```
∂Q/∂α_k = Σ_n r_{nk} - N π_k
∂Q/∂μ_{kd} = Σ_n r_{nk} (x_{nd}-μ_{kd}) / σ^2_{kd}
∂Q/∂γ_{kd} = 0.5 Σ_n r_{nk} [ (x_{nd}-μ_{kd})^2/σ^2_{kd} - 1 ]
```
其中 π=softmax(α)，σ^2=exp(γ)。

### 基线方法（KMeans+NCC）
1. KMeans 聚类（按类或全局）
2. 用少量有标签样本为簇投票
3. 使用簇中心最近类或中心平均分类

## 运行实验（常用参数）
| 参数 | 说明 | 示例 |
|------|------|------|
| --dataset | 数据集 | mnist / cifar10 / synthetic |
| --solver | 训练方式 | em / gd / baseline |
| --pca-dim | PCA 降维后维数 | 50 / 100 / 10 |
| --label-budgets | 每类标签数 | 5 10 20 |
| --repeats / --folds | 交叉验证控制 | 5 / 5 |
| --quick | 快速模式（采样少量数据） | --quick |

## 结果与典型表现
- MNIST：EM 随 L 增大缓慢提升（约 54%→58%）。
- CIFAR-10：GD 在 L=10 达平台 (~56.7%)，优于基线 (~13%) 很多。
- Synthetic：EM 明显优于 GD，验证实现正确。
- 责任+椭圆：帮助理解模型在哪些区域不确定。

## 常见问题 (FAQ)
| 问题 | 解决 |
|------|------|
| 准确率太低 | 增加标签数或提升 --pca-dim |
| 图没更新 | 删除旧 figures/ 再跑绘图脚本 |
| 运行慢 | 加 --quick 或减少 label_budgets |
| t-SNE 很慢 | 减少采样或先 PCA 再 t-SNE |
