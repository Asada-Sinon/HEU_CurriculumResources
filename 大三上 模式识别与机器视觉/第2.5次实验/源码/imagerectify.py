import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from fundmatrixest import *
from epipolar_utils import *
import cv2

'''
COMPUTE_EPIPOLE computes the epipole in homogenous coordinates
given matching points in two images and the fundamental matrix
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    F - the Fundamental matrix such that (points1)^T * F * points2 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    epipole - the homogenous coordinates [x y 1] of the epipole in the first image
'''
def compute_epipole(points1, points2, F):
    """
    计算极点（所有极线的交点）
    
    原理：
    极点是所有极线的交点。对于基础矩阵 F，第一幅图像中的极点 e1 满足：
    F^T * e1 = 0
    
    这是因为对于任意点 p2，极线 l1 = F * p2，而 e1 在所有极线上，即：
    e1^T * l1 = e1^T * F * p2 = 0 对所有 p2 成立
    因此 e1^T * F = 0，即 F^T * e1 = 0
    
    求解：通过对 F^T 进行 SVD 分解，最小奇异值对应的右奇异向量即为极点
    """
    # 对 F^T 进行 SVD 分解
    # F^T * e1 = 0，e1 是 F^T 的零空间
    U, S, Vt = np.linalg.svd(F.T)
    
    # 最小奇异值对应的右奇异向量（Vt 的最后一行）即为极点
    e = Vt[-1]
    
    # 归一化为齐次坐标（最后一维为1）
    e = e / e[-1]
    
    return e
    
'''
COMPUTE_MATCHING_HOMOGRAPHIES determines homographies H1 and H2 such that they
rectify a pair of images. Do not divide the homographies by their 2,2 entry.
Arguments:
    e2 - the second epipole
    F - the Fundamental matrix
    im2 - the second image
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
Returns:
    H1 - the homography associated with the first image
    H2 - the homography associated with the second image
'''
def compute_matching_homographies(e2, F, im2, points1, points2):
    """
    计算用于图像对校正的单应矩阵 H1 和 H2
    
    原理：
    图像对校正的目标是使极线变为水平线，即平行于 x 轴
    
    参考 Hartley & Zisserman《Multiple View Geometry》Algorithm 11.1
    
    算法步骤：
    1. 计算 H2，将第二个极点 e2 映射到无穷远点
    2. 计算 H1，使得对应点在 y 方向上对齐
    """
    # 图像的宽度和高度
    height, width = im2.shape[:2]
    
    # ========== 计算 H2 ==========
    # 使用简化的方法：将 e2 映射到 (1, 0, 0)^T（无穷远点）
    
    # 平移矩阵，将图像中心作为参考
    T = np.array([
        [1, 0, -width/2],
        [0, 1, -height/2],
        [0, 0, 1]
    ])
    
    e2_t = T @ e2
    
    # 旋转矩阵，使 e2 对齐到 x 轴
    e2_len = np.sqrt(e2_t[0]**2 + e2_t[1]**2)
    if e2_len > 1e-6:
        cos_theta = e2_t[0] / e2_len
        sin_theta = e2_t[1] / e2_len
    else:
        cos_theta = 1
        sin_theta = 0
    
    R = np.array([
        [cos_theta, sin_theta, 0],
        [-sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])
    
    # 投影变换，将点发送到无穷远
    e2_rt = R @ e2_t
    f = e2_rt[0] if abs(e2_rt[0]) > 1e-6 else width
    
    G = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [-1/f, 0, 1]
    ])
    
    # H2 = G * R * T
    H2 = G @ R @ T
    
    # ========== 计算 H1 ==========
    # 根据 Hartley & Zisserman，使用公式：
    # H1 = H_A * H0
    # 其中 H0 与基础矩阵相关，H_A 是仿射调整
    
    # 构建 H0 = [e2]_x * F + e2 * v^T
    # 其中 [e2]_x 是反对称矩阵，v 是任意向量
    
    # 反对称矩阵 [e2]_x
    e2_skew = np.array([
        [0, -e2[2], e2[1]],
        [e2[2], 0, -e2[0]],
        [-e2[1], e2[0], 0]
    ])
    
    # 选择 v，使矩阵可逆
    # 通常选择 v = [1, 0, 0] 或中心点
    v = np.array([width/2, height/2, 1])
    
    # H0 = [e2]_x * F + e2 * v^T
    H0 = e2_skew @ F + np.outer(e2, v)
    
    # 规范化 H0
    H0 = H0 / H0[2, 2]
    
    # 用 H2 和 H0 变换对应点
    points2_H2 = (H2 @ points2.T).T
    points2_H2 = points2_H2 / points2_H2[:, 2:3]
    
    points1_H0 = (H0 @ points1.T).T
    points1_H0 = points1_H0 / points1_H0[:, 2:3]
    
    # 计算仿射变换 H_A，使对应点的 y 坐标对齐
    # H_A 的形式: [[a, b, c], [d, e, f], [0, 0, 1]]
    
    N = points1.shape[0]
    
    # 构建最小二乘问题
    A = np.column_stack([points1_H0[:, 0], points1_H0[:, 1], np.ones(N)])
    
    # 求解 x 坐标的映射
    b_x = points2_H2[:, 0]
    params_x = np.linalg.lstsq(A, b_x, rcond=None)[0]
    
    # 求解 y 坐标的映射
    b_y = points2_H2[:, 1]
    params_y = np.linalg.lstsq(A, b_y, rcond=None)[0]
    
    # 构建 H_A
    H_A = np.array([
        [params_x[0], params_x[1], params_x[2]],
        [params_y[0], params_y[1], params_y[2]],
        [0, 0, 1]
    ])
    
    # 最终的 H1
    H1 = H_A @ H0
    
    # 规范化
    H1 = H1 / H1[2, 2]
    H2 = H2 / H2[2, 2]
    
    return H1, H2

def compute_matching_homographies_opencv(F, im_shape, points1, points2):
    """
    使用 OpenCV 的 stereoRectifyUncalibrated 计算校正单应矩阵
    这是一个更稳健的实现，推荐使用
    
    Arguments:
        F - 基础矩阵
        im_shape - 图像形状 (height, width, ...)
        points1 - 第一幅图像中的点
        points2 - 第二幅图像中的点
    
    Returns:
        H1, H2 - 两个单应矩阵
    """
    height, width = im_shape[:2]
    
    # 转换点格式为 OpenCV 格式 (N, 1, 2)
    pts1 = points1[:, :2].astype(np.float32).reshape(-1, 1, 2)
    pts2 = points2[:, :2].astype(np.float32).reshape(-1, 1, 2)
    
    # 使用 OpenCV 进行立体校正
    retval, H1, H2 = cv2.stereoRectifyUncalibrated(
        pts1, pts2, F.astype(np.float64), (width, height)
    )
    
    if not retval:
        raise Exception("OpenCV 立体校正失败")
    
    return H1, H2

if __name__ == '__main__':
    # Read in the data
    im_set = 'data/set1'
    im1 = imread(im_set+'/image1.jpg')
    im2 = imread(im_set+'/image2.jpg')
    points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
    points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
    assert (points1.shape == points2.shape)

    F = normalized_eight_point_alg(points1, points2)
    e1 = compute_epipole(points1, points2, F)
    e2 = compute_epipole(points2, points1, F.transpose())
    print("e1", e1)
    print("e2", e2)

    # Find the homographies needed to rectify the pair of images
    # 推荐使用 OpenCV 方法（更稳健）
    try:
        H1, H2 = compute_matching_homographies_opencv(F, im1.shape, points1, points2)
        print("使用 OpenCV 方法")
    except:
        # 如果 OpenCV 方法失败，使用手工实现
        H1, H2 = compute_matching_homographies(e2, F, im2, points1, points2)
        print("使用手工实现方法")
    
    print("H1:\n", H1)
    print('')
    print("H2:\n", H2)

    # Transforming the images by the homographies
    new_points1 = H1.dot(points1.T)
    new_points2 = H2.dot(points2.T)
    new_points1 /= new_points1[2,:]
    new_points2 /= new_points2[2,:]
    new_points1 = new_points1.T
    new_points2 = new_points2.T
    
    # 使用 OpenCV 进行图像变换（更高效）
    rectified_im1 = cv2.warpPerspective(im1, H1, (im1.shape[1], im1.shape[0]))
    rectified_im2 = cv2.warpPerspective(im2, H2, (im2.shape[1], im2.shape[0]))
    
    # Plotting the image
    F_new = normalized_eight_point_alg(new_points1, new_points2)
    
    # 计算校正效果
    dist1 = compute_distance_to_epipolar_lines(new_points1, new_points2, F_new)
    dist2 = compute_distance_to_epipolar_lines(new_points2, new_points1, F_new.T)
    print(f"\n校正后极线距离 - 图像1: {dist1:.4f} pixels")
    print(f"校正后极线距离 - 图像2: {dist2:.4f} pixels")
    
    # 检查 y 坐标差异
    y_diffs = np.abs(new_points1[:, 1] - new_points2[:, 1])
    print(f"对应点 y 坐标平均差异: {np.mean(y_diffs):.4f} pixels")
    print(f"对应点 y 坐标最大差异: {np.max(y_diffs):.4f} pixels")
    
    plot_epipolar_lines_on_images(new_points1, new_points2, rectified_im1, rectified_im2, F_new)
    plt.show()
