# 项目2: 基础矩阵估计与图像对校正

## 项目结构

```
proj2-part2/
├── fundmatrixest.py          # 任务1.1: 基础矩阵估计实现
├── imagerectify.py           # 任务1.2: 图像对校正实现
├── epipolar_utils.py         # 工具函数
├── run_all_tasks.py          # 运行所有任务的主程序
├── data/                     # 数据文件夹
│   ├── set1/                 # 数据集1
│   └── set2/                 # 数据集2
├── 实验报告.md                # 任务1.1详细报告
└── 任务2_图像校正实验报告.md  # 任务1.2详细报告
```

## 任务说明

### 任务1.1: 基础矩阵估计
实现两种基础矩阵估计算法：
1. **线性最小二乘八点法 (LLS)**
2. **规范化八点法 (Normalized)**

#### 实现的函数（fundmatrixest.py）
- `lls_eight_point_alg(points1, points2)` - 线性最小二乘八点法
- `normalized_eight_point_alg(points1, points2)` - 规范化八点法
- `compute_distance_to_epipolar_lines(points1, points2, F)` - 计算极线距离

### 任务1.2: 图像对校正
基于基础矩阵估计结果，计算单应矩阵H1和H2，使极线水平化。

#### 实现的函数（imagerectify.py）
- `compute_epipole(points1, points2, F)` - 计算极点
- `compute_matching_homographies(e2, F, im2, points1, points2)` - 计算单应矩阵
- `compute_matching_homographies_opencv(F, im_shape, points1, points2)` - 使用OpenCV的稳健实现

## 快速开始

### 环境要求
```bash
pip install numpy matplotlib scikit-image opencv-python
```

### 运行程序
```bash
python run_all_tasks.py
```

该命令将：
1. 对两个数据集计算基础矩阵（两种方法）
2. 计算并显示极线距离
3. 计算极点和单应矩阵
4. 生成校正后的图像
5. 保存所有结果图像到当前目录

### 单独运行某个任务

**运行任务1.1（基础矩阵估计）：**
```bash
python fundmatrixest.py
```

**运行任务1.2（图像校正）：**
```bash
python imagerectify.py
```

## 实验结果

### 任务1.1结果摘要

| 数据集 | 方法 | 图像1距离 | 图像2距离 | 平均距离 |
|--------|------|----------|----------|----------|
| set1 | LLS | 80.73 | 89.48 | 85.10 |
| set1 | 规范化 | 60.87 | 68.34 | 64.61 |
| set2 | LLS | 33.60 | 30.17 | 31.89 |
| set2 | 规范化 | 32.42 | 32.73 | 32.57 |

**结论**：规范化八点法在精度上优于线性最小二乘法。

### 任务1.2结果摘要

| 数据集 | 校正前平均距离 | 校正后平均距离 | y坐标平均差异 | 改善率 |
|--------|---------------|---------------|--------------|--------|
| set1 | 64.61 pixels | 0.86 pixels | 0.86 pixels | 98.7% |
| set2 | 32.57 pixels | 0.89 pixels | 0.88 pixels | 97.3% |

**结论**：图像校正使极线距离减小约98%，对应点y坐标误差小于1像素。

## 生成的文件

### 任务1.1输出
- `task1_set1_lls_epipolar.png` - set1的LLS方法极线图
- `task1_set1_normalized_epipolar.png` - set1的规范化方法极线图
- `task1_set2_lls_epipolar.png` - set2的LLS方法极线图
- `task1_set2_normalized_epipolar.png` - set2的规范化方法极线图

### 任务1.2输出
- `task2_set1_rectified_epipolar.png` - set1校正后的极线图
- `task2_set1_comparison.png` - set1校正前后对比图
- `task2_set2_rectified_epipolar.png` - set2校正后的极线图
- `task2_set2_comparison.png` - set2校正前后对比图

## 核心算法

### 1. 线性最小二乘八点法
1. 构建矩阵A（N×9），每行对应一个点对约束
2. SVD分解求解Af=0
3. 将解重构为3×3基础矩阵
4. 施加秩2约束（SVD后将最小奇异值置零）

### 2. 规范化八点法
1. 计算规范化变换矩阵T₁和T₂
2. 规范化点：使质心在原点，平均距离为√2
3. 用规范化后的点计算基础矩阵F'
4. 反规范化：F = T₂ᵀF'T₁
5. 施加秩2约束

### 3. 图像对校正
1. 计算极点：通过SVD求解F<sup>T</sup>e=0
2. 计算H₂：将极点映射到无穷远点
3. 计算H₁：使对应点y坐标对齐（使用OpenCV的stereoRectifyUncalibrated）
4. 应用单应变换校正图像

## 技术要点

- **数值稳定性**：规范化八点法显著改善条件数
- **秩约束**：基础矩阵必须满足秩2约束
- **极点位置**：极点在图像外部时需要特殊处理
- **OpenCV优化**：使用OpenCV的实现提高稳健性

## 参考文献
1. Hartley, R. I., & Zisserman, A. (2003). Multiple View Geometry in Computer Vision.
2. Hartley, R. I. (1997). In defense of the eight-point algorithm. IEEE TPAMI.
3. OpenCV Documentation: stereoRectifyUncalibrated

## 注意事项
1. 确保data文件夹包含set1和set2的图像和点文件
2. 基础矩阵的秩2约束是必须的
3. 规范化八点法通常比LLS更稳定
4. 校正后的图像对可大幅简化立体匹配

