"""
项目2 - 任务1.1和任务1.2的完整实现

任务1.1: 基础矩阵估计（八点法和规范化八点法）
任务1.2: 图像对校正

运行此脚本将：
1. 计算基础矩阵（使用两种方法）
2. 计算并显示极线距离
3. 计算极点
4. 计算校正单应矩阵
5. 生成校正后的图像
6. 保存所有结果图像
"""

import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from fundmatrixest import *
from epipolar_utils import *
from imagerectify import *
import cv2

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("项目2: 基础矩阵估计与图像对校正")
print("="*80)
print()

# 处理数据集
for im_set in ['data/set1', 'data/set2']:
    set_name = im_set.split('/')[-1]
    
    print('='*80)
    print(f"处理数据集: {set_name}")
    print('='*80)
    print()
    
    # ========== 读取数据 ==========
    im1 = imread(im_set+'/image1.jpg')
    im2 = imread(im_set+'/image2.jpg')
    points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
    points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
    
    # ========== 任务1.1: 基础矩阵估计 ==========
    print("【任务1.1】基础矩阵估计")
    print("-" * 40)
    
    # 线性最小二乘八点法
    F_lls = lls_eight_point_alg(points1, points2)
    print("1. 线性最小二乘八点法 (LLS)")
    print("   基础矩阵 F:")
    print(f"   {F_lls}")
    dist_lls_1 = compute_distance_to_epipolar_lines(points1, points2, F_lls)
    dist_lls_2 = compute_distance_to_epipolar_lines(points2, points1, F_lls.T)
    print(f"   图像1极线距离: {dist_lls_1:.4f} pixels")
    print(f"   图像2极线距离: {dist_lls_2:.4f} pixels")
    print()
    
    # 规范化八点法
    F_normalized = normalized_eight_point_alg(points1, points2)
    print("2. 规范化八点法 (Normalized)")
    print("   基础矩阵 F:")
    print(f"   {F_normalized}")
    dist_norm_1 = compute_distance_to_epipolar_lines(points1, points2, F_normalized)
    dist_norm_2 = compute_distance_to_epipolar_lines(points2, points1, F_normalized.T)
    print(f"   图像1极线距离: {dist_norm_1:.4f} pixels")
    print(f"   图像2极线距离: {dist_norm_2:.4f} pixels")
    print()
    
    # 保存任务1.1的结果图像
    plot_epipolar_lines_on_images(points1, points2, im1, im2, F_lls)
    plt.suptitle(f'{set_name} - 线性最小二乘八点法', fontsize=16)
    plt.savefig(f'task1_{set_name}_lls_epipolar.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    plot_epipolar_lines_on_images(points1, points2, im1, im2, F_normalized)
    plt.suptitle(f'{set_name} - 规范化八点法', fontsize=16)
    plt.savefig(f'task1_{set_name}_normalized_epipolar.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ========== 任务1.2: 图像对校正 ==========
    print("【任务1.2】图像对校正")
    print("-" * 40)
    
    # 使用规范化八点法的结果
    F = F_normalized
    
    # 计算极点
    e1 = compute_epipole(points1, points2, F)
    e2 = compute_epipole(points2, points1, F.T)
    print(f"1. 极点计算")
    print(f"   极点 e1 (第一幅图像): [{e1[0]:.2f}, {e1[1]:.2f}, {e1[2]:.2f}]")
    print(f"   极点 e2 (第二幅图像): [{e2[0]:.2f}, {e2[1]:.2f}, {e2[2]:.2f}]")
    print()
    
    # 计算单应矩阵（使用OpenCV方法）
    pts1 = points1[:, :2].astype(np.float32).reshape(-1, 1, 2)
    pts2 = points2[:, :2].astype(np.float32).reshape(-1, 1, 2)
    retval, H1, H2 = cv2.stereoRectifyUncalibrated(
        pts1, pts2, F.astype(np.float64), (im1.shape[1], im1.shape[0])
    )
    
    print(f"2. 单应矩阵计算")
    print(f"   H1:")
    print(f"   {H1}")
    print(f"   H2:")
    print(f"   {H2}")
    print()
    
    # 图像校正
    rectified_im1 = cv2.warpPerspective(im1, H1, (im1.shape[1], im1.shape[0]))
    rectified_im2 = cv2.warpPerspective(im2, H2, (im2.shape[1], im2.shape[0]))
    
    # 变换对应点
    new_points1 = (H1 @ points1.T).T
    new_points2 = (H2 @ points2.T).T
    new_points1 = new_points1 / new_points1[:, 2:3]
    new_points2 = new_points2 / new_points2[:, 2:3]
    
    # 评估校正效果
    F_new = normalized_eight_point_alg(new_points1, new_points2)
    dist_rect_1 = compute_distance_to_epipolar_lines(new_points1, new_points2, F_new)
    dist_rect_2 = compute_distance_to_epipolar_lines(new_points2, new_points1, F_new.T)
    y_diffs = np.abs(new_points1[:, 1] - new_points2[:, 1])
    
    print(f"3. 校正效果评估")
    print(f"   校正后极线距离 - 图像1: {dist_rect_1:.4f} pixels")
    print(f"   校正后极线距离 - 图像2: {dist_rect_2:.4f} pixels")
    print(f"   对应点y坐标平均差异: {np.mean(y_diffs):.4f} pixels")
    print(f"   对应点y坐标最大差异: {np.max(y_diffs):.4f} pixels")
    print()
    
    # 保存任务1.2的结果图像
    # 校正后的极线图
    plot_epipolar_lines_on_images(new_points1, new_points2, rectified_im1, rectified_im2, F_new)
    plt.suptitle(f'{set_name} - 校正后的图像对与极线', fontsize=16)
    plt.savefig(f'task2_{set_name}_rectified_epipolar.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 校正前后对比图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 原始图像1
    axes[0, 0].imshow(im1, cmap='gray')
    axes[0, 0].set_title('原始图像1', fontsize=14)
    axes[0, 0].axis('off')
    for i in range(points1.shape[0]):
        axes[0, 0].plot(points1[i, 0], points1[i, 1], '*b', markersize=6)
    
    # 原始图像2
    axes[0, 1].imshow(im2, cmap='gray')
    axes[0, 1].set_title('原始图像2', fontsize=14)
    axes[0, 1].axis('off')
    for i in range(points2.shape[0]):
        axes[0, 1].plot(points2[i, 0], points2[i, 1], '*b', markersize=6)
    
    # 校正后图像1
    axes[1, 0].imshow(rectified_im1, cmap='gray')
    axes[1, 0].set_title('校正后图像1', fontsize=14)
    axes[1, 0].axis('off')
    for i in range(new_points1.shape[0]):
        if 0 <= new_points1[i, 0] < im1.shape[1] and 0 <= new_points1[i, 1] < im1.shape[0]:
            axes[1, 0].plot(new_points1[i, 0], new_points1[i, 1], '*r', markersize=6)
    # 绘制水平参考线
    for i in range(0, min(10, new_points1.shape[0]), 2):
        y = new_points1[i, 1]
        if 0 <= y < rectified_im1.shape[0]:
            axes[1, 0].axhline(y=y, color='g', alpha=0.3, linewidth=1)
    
    # 校正后图像2
    axes[1, 1].imshow(rectified_im2, cmap='gray')
    axes[1, 1].set_title('校正后图像2', fontsize=14)
    axes[1, 1].axis('off')
    for i in range(new_points2.shape[0]):
        if 0 <= new_points2[i, 0] < im2.shape[1] and 0 <= new_points2[i, 1] < im2.shape[0]:
            axes[1, 1].plot(new_points2[i, 0], new_points2[i, 1], '*r', markersize=6)
    # 绘制水平参考线
    for i in range(0, min(10, new_points2.shape[0]), 2):
        y = new_points2[i, 1]
        if 0 <= y < rectified_im2.shape[0]:
            axes[1, 1].axhline(y=y, color='g', alpha=0.3, linewidth=1)
    
    plt.suptitle(f'{set_name} - 图像校正前后对比', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'task2_{set_name}_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ {set_name} 处理完成，结果已保存")
    print()

print("="*80)
print("所有任务完成！")
print("="*80)
print()
print("生成的文件:")
print("  任务1.1 (基础矩阵估计):")
print("    - task1_set1_lls_epipolar.png")
print("    - task1_set1_normalized_epipolar.png")
print("    - task1_set2_lls_epipolar.png")
print("    - task1_set2_normalized_epipolar.png")
print()
print("  任务1.2 (图像对校正):")
print("    - task2_set1_rectified_epipolar.png")
print("    - task2_set1_comparison.png")
print("    - task2_set2_rectified_epipolar.png")
print("    - task2_set2_comparison.png")
print()

