import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from epipolar_utils import *

'''
LLS_EIGHT_POINT_ALG  computes the fundamental matrix from matching points using 
linear least squares eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points1)^T * F * points2 = 0
Please see project notes and slides to see how the linear least squares eight
point algorithm works
'''
def lls_eight_point_alg(points1, points2):
    """
    线性最小二乘八点法实现基础矩阵估计
    
    原理：
    对于对应点 p1 和 p2，满足约束 p2^T * F * p1 = 0
    其中 p1 = [x1, y1, 1]^T, p2 = [x2, y2, 1]^T
    
    展开后得到：x2*x1*f11 + x2*y1*f12 + x2*f13 + y2*x1*f21 + y2*y1*f22 + y2*f23 + x1*f31 + y1*f32 + f33 = 0
    
    对于 N 个点对，构建矩阵 A，使得 A*f = 0，其中 f 是 F 按行展开的 9x1 向量
    通过 SVD 分解 A，最小奇异值对应的右奇异向量即为 f 的解
    
    最后对 F 施加秩 2 约束：对 F 进行 SVD 分解，将最小奇异值置为 0
    """
    # 获取点的数量
    N = points1.shape[0]
    
    # 构建矩阵 A (N x 9)
    # 每一行对应一个点对的约束方程
    A = np.zeros((N, 9))
    for i in range(N):
        x1, y1, _ = points1[i]
        x2, y2, _ = points2[i]
        # [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
        A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    
    # 对 A 进行 SVD 分解
    # A = U * S * V^T
    # 最小奇异值对应的右奇异向量（V 的最后一列）即为 f 的解
    U, S, Vt = np.linalg.svd(A)
    
    # V 的最后一行（Vt 的最后一行）对应最小奇异值
    f = Vt[-1]
    
    # 将 f 重构为 3x3 的基础矩阵 F
    F = f.reshape(3, 3)
    
    # 对 F 施加秩 2 约束
    # 对 F 进行 SVD 分解
    U_f, S_f, Vt_f = np.linalg.svd(F)
    
    # 将最小奇异值置为 0
    S_f[-1] = 0
    
    # 重构基础矩阵
    F = U_f @ np.diag(S_f) @ Vt_f
    
    return F

'''
NORMALIZED_EIGHT_POINT_ALG  computes the fundamental matrix from matching points
using the normalized eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points1)^T * F * points2 = 0
Please see project notes and slides to see how the normalized eight
point algorithm works
'''
def normalized_eight_point_alg(points1, points2):
    """
    规范化八点法实现基础矩阵估计
    
    原理：
    在使用八点法之前，先对点进行规范化处理，使得点的中心在原点，平均距离为 sqrt(2)
    这样可以提高数值稳定性和精度
    
    步骤：
    1. 计算规范化变换矩阵 T1 和 T2
    2. 对点进行规范化：p1_norm = T1 * p1, p2_norm = T2 * p2
    3. 使用规范化后的点计算基础矩阵 F_norm
    4. 反规范化得到原始基础矩阵：F = T2^T * F_norm * T1
    """
    
    def normalize_points(points):
        """
        规范化点集
        返回：规范化后的点和规范化变换矩阵 T
        """
        # 计算点的中心（质心）
        centroid = np.mean(points[:, :2], axis=0)
        
        # 将点平移到原点
        points_centered = points.copy()
        points_centered[:, 0] -= centroid[0]
        points_centered[:, 1] -= centroid[1]
        
        # 计算点到原点的平均距离
        distances = np.sqrt(points_centered[:, 0]**2 + points_centered[:, 1]**2)
        mean_distance = np.mean(distances)
        
        # 计算缩放因子，使得平均距离为 sqrt(2)
        scale = np.sqrt(2) / mean_distance
        
        # 构建规范化变换矩阵 T
        # T = [[s, 0, -s*cx],
        #      [0, s, -s*cy],
        #      [0, 0,   1  ]]
        T = np.array([
            [scale, 0, -scale * centroid[0]],
            [0, scale, -scale * centroid[1]],
            [0, 0, 1]
        ])
        
        # 应用规范化变换
        points_normalized = (T @ points.T).T
        
        return points_normalized, T
    
    # 对两组点进行规范化
    points1_norm, T1 = normalize_points(points1)
    points2_norm, T2 = normalize_points(points2)
    
    # 使用规范化后的点计算基础矩阵
    N = points1_norm.shape[0]
    
    # 构建矩阵 A
    A = np.zeros((N, 9))
    for i in range(N):
        x1, y1, _ = points1_norm[i]
        x2, y2, _ = points2_norm[i]
        A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    
    # SVD 分解
    U, S, Vt = np.linalg.svd(A)
    f = Vt[-1]
    F_norm = f.reshape(3, 3)
    
    # 对 F_norm 施加秩 2 约束
    U_f, S_f, Vt_f = np.linalg.svd(F_norm)
    S_f[-1] = 0
    F_norm = U_f @ np.diag(S_f) @ Vt_f
    
    # 反规范化得到原始基础矩阵
    # F = T2^T * F_norm * T1
    F = T2.T @ F_norm @ T1
    
    return F

'''
PLOT_EPIPOLAR_LINES_ON_IMAGES given a pair of images and corresponding points,
draws the epipolar lines on the images
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    im1 - a HxW(xC) matrix that contains pixel values from the first image 
    im2 - a HxW(xC) matrix that contains pixel values from the second image 
    F - the fundamental matrix such that (points1)^T * F * points2 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    Nothing; instead, plots the two images with the matching points and
    their corresponding epipolar lines. See Figure 1 within the project note
    for an example
'''
def plot_epipolar_lines_on_images(points1, points2, im1, im2, F):

    def plot_epipolar_lines_on_image(points1, points2, im, F):
        im_height = im.shape[0]
        im_width = im.shape[1]
        lines = F.dot(points2.T)
        plt.imshow(im, cmap='gray')
        for line in lines.T:
            a,b,c = line
            xs = [1, im.shape[1]-1]
            ys = [(-c-a*x)/b for x in xs]
            plt.plot(xs, ys, 'r')
        for i in range(points1.shape[0]):
            x,y,_ = points1[i]
            plt.plot(x, y, '*b')
        plt.axis([0, im_width, im_height, 0])

    # We change the figsize because matplotlib has weird behavior when 
    # plotting images of different sizes next to each other. This
    # fix should be changed to something more robust.
    new_figsize = (8 * (float(max(im1.shape[1], im2.shape[1])) / min(im1.shape[1], im2.shape[1]))**2 , 6)
    fig = plt.figure(figsize=new_figsize)
    plt.subplot(121)
    plot_epipolar_lines_on_image(points1, points2, im1, F)
    plt.axis('off')
    plt.subplot(122)
    plot_epipolar_lines_on_image(points2, points1, im2, F.T)
    plt.axis('off')

'''
COMPUTE_DISTANCE_TO_EPIPOLAR_LINES  computes the average distance of a set a 
points to their corresponding epipolar lines. Compute just the average distance
from points1 to their corresponding epipolar lines (which you get from points2).
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    F - the fundamental matrix such that (points1)^T * F * points2 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    average_distance - the average distance of each point to the epipolar line
'''
def compute_distance_to_epipolar_lines(points1, points2, F):
    """
    计算点到对应极线的平均距离
    
    原理：
    对于点 p2，其在图像 1 中对应的极线为 l = F * p2
    极线方程为 ax + by + c = 0，其中 l = [a, b, c]^T
    点 p1 = [x1, y1, 1]^T 到极线的距离为：
    d = |ax1 + by1 + c| / sqrt(a^2 + b^2)
    
    计算所有点到对应极线的距离，并返回平均值
    """
    N = points1.shape[0]
    distances = np.zeros(N)
    
    for i in range(N):
        # 计算点 p2 对应的极线 l = F * p2
        line = F @ points2[i]
        a, b, c = line
        
        # 计算点 p1 到极线的距离
        x1, y1, _ = points1[i]
        
        # 点到直线距离公式：d = |ax + by + c| / sqrt(a^2 + b^2)
        distance = np.abs(a * x1 + b * y1 + c) / np.sqrt(a**2 + b**2)
        distances[i] = distance
    
    # 返回平均距离
    average_distance = np.mean(distances)
    
    return average_distance

if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set2']:
        print('-'*80)
        print("Set:", im_set)
        print('-'*80)

        # Read in the data
        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
        points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
        assert (points1.shape == points2.shape)

        # Running the linear least squares eight point algorithm
        F_lls = lls_eight_point_alg(points1, points2)
        print("Fundamental Matrix from LLS  8-point algorithm:\n", F_lls)
        print("Distance to lines in image 1 for LLS:", \
            compute_distance_to_epipolar_lines(points1, points2, F_lls))
        print("Distance to lines in image 2 for LLS:", \
            compute_distance_to_epipolar_lines(points2, points1, F_lls.T))

        # Running the normalized eight point algorithm
        F_normalized = normalized_eight_point_alg(points1, points2)

        pFp = [points1[i].dot(F_normalized.dot(points2[i]))
            for i in range(points1.shape[0])]
        print("p'^T F p =", np.abs(pFp).max())
        print("Fundamental Matrix from normalized 8-point algorithm:\n", \
            F_normalized)
        print("Distance to lines in image 1 for normalized:", \
            compute_distance_to_epipolar_lines(points1, points2, F_normalized))
        print("Distance to lines in image 2 for normalized:", \
            compute_distance_to_epipolar_lines(points2, points1, F_normalized.T))

        # Plotting the epipolar lines
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_lls)
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_normalized)

        plt.show()
