"""
project3_pipeline.py

该脚本实现了课程项目第 1 部分的完整流水线：
- 加载数据集（images/ 与 landmarks/，默认按编号排序）
- 将前 800 张作为训练集，后 200 张作为测试集
- 计算训练集的平均形状（mean shape）并基于 mean shape 计算 Delaunay 三角划分
- 将训练集图像 warp 到 mean shape，从而在统一姿态下计算 appearance PCA（在 HSV 的 V 通道）
- 在关键点向量上计算 geometry PCA（eigen-warpings）
- 在测试集上评估重建误差并输出误差曲线与示例重建图像
- 基于前 10 个几何特征和前 50 个表观特征生成 50 张合成脸

输出位于 `outputs/` 目录，包含误差曲线、eigenfaces/eigenwarps、重建对比图以及合成图像。

注意：脚本依赖 `mywarper.py` 中的 warp 与 Delaunay 函数来完成形变与对齐。
"""
import os
import sys
import numpy as np
import cv2
import scipy.io
import matplotlib.pyplot as plt
from math import sqrt

# Ensure project folder in path to import mywarper
sys.path.append(os.path.dirname(__file__))
import mywarper


IMG_DIR = os.path.join(os.path.dirname(__file__), 'images')
LAND_DIR = os.path.join(os.path.dirname(__file__), 'landmarks')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE = 128

def list_ids():
    files = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith('.jpg') or f.lower().endswith('.png')])
    ids = [os.path.splitext(f)[0] for f in files]
    return ids

def load_landmark(mat_path):
    """
    从 .mat 文件中读取 landmark 点。

    返回：Nx2 的 float32 数组（x,y）。
    支持的 .mat 变量名为 'lms'，如果不存在则使用第一个非 __ 开头的变量。
    处理步骤：
    - 读取变量并转为 numpy
    - 若形状为 (2, N) 的转置到 (N,2)
    - 最终 reshape 保证 (N,2)
    """
    m = scipy.io.loadmat(mat_path)
    # variable is 'lms' in provided mats
    if 'lms' in m:
        lms = m['lms']
    else:
        keys = [k for k in m.keys() if not k.startswith('__')]
        lms = m[keys[0]]
    lms = np.array(lms)
    if lms.ndim == 2 and lms.shape[0] == 2 and lms.shape[1] != 2:
        lms = lms.T
    lms = lms.reshape(-1,2).astype(np.float32)
    return lms


def ensure_square_image(img, size):
    """
    将输入图像缩放并裁剪/填充为指定大小的方形图像（size x size）。

    注意: 在本 pipeline 中通常直接通过 mywarper.warp 中的 Delaunay 处理，
    但在预处理/可视化时可能需要保证图像为指定方形尺寸。
    """
    return cv2.resize(img, (size, size))

def rgb_to_v_channel(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    # V in 0..255
    V = hsv[:,:,2] / 255.0
    return V

def v_channel_to_bgr(mean_h, mean_s, v_channel):
    # mean_h,s in same scale as OpenCV HSV (H:0..179, S:0..255)
    hsv = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    hsv[:,:,0] = np.clip(mean_h, 0, 179).astype(np.uint8)
    hsv[:,:,1] = np.clip(mean_s, 0, 255).astype(np.uint8)
    hsv[:,:,2] = np.clip((v_channel*255.0), 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def compute_pca(X, k_max=None):
    """
    对数据矩阵 X 计算 PCA（均值、主成分向量、特征值）。

    参数：
    - X: shape (n_samples, dim)
    - k_max: 如果指定，返回前 k_max 个主成分与对应的特征值

    返回：mean (dim,), components (k, dim), eigvals (k,)

    实现说明：
    - 当 n_samples < dim 时，先在 Gram 矩阵 (n x n) 上做特征分解以节省计算开销（经典 trick），
      再将结果映射回原空间得到主成分向量。
    - 否则直接对中心化数据做 SVD 得到主成分。
    """
    # X: (n_samples, dim)
    n, d = X.shape
    mean = np.mean(X, axis=0)
    Xc = X - mean

    # 当样本数量小于向量维度（例如像素向量）时，直接对协方差矩阵 SVD 代价大，
    # 此处采用 Gram 矩阵 (n x n) 的特征分解技巧来减少计算开销。
    if n < d:
        # Use eigen-decomposition of the Gram matrix (n x n)
        # C = Xc @ Xc.T / (n-1)
        C = np.dot(Xc, Xc.T) / (n - 1)
        S, U = np.linalg.eigh(C)  # eigenvalues ascending
        # sort descending
        idx = np.argsort(S)[::-1]
        S = S[idx]
        U = U[:, idx]
        # convert to principal components in original space
        # components = (Xc.T @ U) / sqrt(eigvals * (n-1))
        eigvals = S
        # avoid division by zero
        nonzero = eigvals > 1e-12
        components = np.zeros((len(eigvals), d))
        for i in range(len(eigvals)):
            if nonzero[i]:
                components[i] = (Xc.T @ U[:, i]) / (np.sqrt(eigvals[i] * (n - 1)))
            else:
                components[i] = 0
    else:
        # d <= n: SVD on Xc is manageable
        U_s, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        eigvals = (S**2) / (n - 1)
        components = Vt

    if k_max is not None:
        # trim to k_max components and eigvals
        k_trim = min(k_max, components.shape[0])
        components = components[:k_trim]
        eigvals = eigvals[:k_trim]

    return mean, components, eigvals

def project_and_reconstruct(x, mean, components, k):
    # x: (dim,) single sample
    comps = components[:k]
    x_c = x - mean
    coeffs = np.dot(x_c, comps.T)
    recon = mean + np.dot(coeffs, comps)
    return recon, coeffs

def evaluate_image_reconstruction(test_X, mean, components, ks):
    n_test = test_X.shape[0]
    errors = []
    for k in ks:
        err_sum = 0.0
        for i in range(n_test):
            recon, _ = project_and_reconstruct(test_X[i], mean, components, k)
            diff = (recon - test_X[i])
            err_sum += np.sum(diff*diff)
        # normalize by pixel count and average
        pix = test_X.shape[1]
        errors.append(err_sum / (n_test * pix))
    return np.array(errors)

def evaluate_landmark_reconstruction(test_L, meanL, compsL, ks):
    n_test = test_L.shape[0]
    errors = []
    for k in ks:
        err_sum = 0.0
        for i in range(n_test):
            recon, _ = project_and_reconstruct(test_L[i], meanL, compsL, k)
            diff = recon - test_L[i]
            # per-point euclidean
            diff = diff.reshape(-1,2)
            err_sum += np.mean(np.sqrt(np.sum(diff*diff, axis=1)))
        errors.append(err_sum / n_test)
    return np.array(errors)

def main():
    ids = list_ids()
    if len(ids) < 1000:
        print('Warning: less than 1000 images found (%d). Proceeding with available.' % len(ids))
    # use up to 1000
    ids = ids[:1000]
    train_ids = ids[:800]
    test_ids = ids[800:1000]

    # Load landmarks and images for train
    print('Loading training data...')
    train_imgs = []
    train_vs = []
    train_h = []
    train_s = []
    train_lms = []
    for id0 in train_ids:
        img_path = os.path.join(IMG_DIR, id0 + '.jpg')
        lm_path = os.path.join(LAND_DIR, id0 + '.mat')
        img = cv2.imread(img_path)
        if img is None:
            print('skip missing image', img_path); continue
        lm = load_landmark(lm_path)
        train_lms.append(lm.reshape(-1))
    train_lms = np.array(train_lms)

    # compute mean shape
    mean_shape = np.mean(train_lms, axis=0).reshape(-1,2)
    print('Mean shape computed')

    # compute Delaunay on mean shape via provided function
    rect = (0,0,IMG_SIZE,IMG_SIZE)
    delaunay = mywarper.calculateDelaunayTriangles(rect, mean_shape.astype(int).tolist())
    print('Delaunay triangles:', len(delaunay))

    # Warp train images to mean shape and collect HSV channels
    print('Warping training images to mean shape...')
    aligned_V = []
    aligned_H = []
    aligned_S = []
    aligned_color_mean = None
    for id0 in train_ids:
        img_path = os.path.join(IMG_DIR, id0 + '.jpg')
        lm_path = os.path.join(LAND_DIR, id0 + '.mat')
        img = cv2.imread(img_path)
        if img is None:
            continue
        lm = load_landmark(lm_path)
        warped = mywarper.warp(img, lm, mean_shape)
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV).astype(np.float32)
        H = hsv[:,:,0]
        S = hsv[:,:,1]
        V = hsv[:,:,2] / 255.0
        aligned_H.append(H)
        aligned_S.append(S)
        aligned_V.append(V.flatten())

    aligned_V = np.array(aligned_V)
    aligned_H = np.array(aligned_H)
    aligned_S = np.array(aligned_S)
    print('Aligned train images:', aligned_V.shape)

    mean_H = np.mean(aligned_H, axis=0)
    mean_S = np.mean(aligned_S, axis=0)
    mean_V = np.mean(aligned_V, axis=0)

    # Appearance PCA on V channel
    print('Computing appearance PCA (V channel)...')
    mean_app, comps_app, eigvals_app = compute_pca(aligned_V, k_max=50)

    # Geometry PCA on landmarks
    print('Computing geometry PCA (landmarks)...')
    mean_geom, comps_geom, eigvals_geom = compute_pca(train_lms, k_max=50)

    # Prepare test data
    print('Preparing test data and evaluating...')
    test_vs = []
    test_lms = []
    original_test_images = []
    for id0 in test_ids:
        img_path = os.path.join(IMG_DIR, id0 + '.jpg')
        lm_path = os.path.join(LAND_DIR, id0 + '.mat')
        img = cv2.imread(img_path)
        if img is None:
            continue
        lm = load_landmark(lm_path)
        original_test_images.append(img)
        test_lms.append(lm.reshape(-1))
        warped = mywarper.warp(img, lm, mean_shape)
        V = rgb_to_v_channel(warped).flatten()
        test_vs.append(V)

    test_vs = np.array(test_vs)
    test_lms = np.array(test_lms)
    print('Test aligned V shape:', test_vs.shape, 'Test landmarks:', test_lms.shape)

    ks = list(range(1,51,4))
    # ensure last value 50 included
    if 50 not in ks:
        ks.append(50)

    app_errors = evaluate_image_reconstruction(test_vs, mean_app, comps_app, ks)
    geom_errors = evaluate_landmark_reconstruction(test_lms, mean_geom, comps_geom, ks)

    # Save error curves
    plt.figure()
    plt.plot(ks, app_errors, '-o')
    plt.xlabel('K (appearance eigenfaces)')
    plt.ylabel('MSE (pixel)')
    plt.title('Appearance reconstruction error')
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, 'appearance_error.png'))

    plt.figure()
    plt.plot(ks, geom_errors, '-o')
    plt.xlabel('K (geom eigenwarp)')
    plt.ylabel('Mean per-point Euclidean error')
    plt.title('Geometry reconstruction error')
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, 'geometry_error.png'))

    # Joint reconstruction: geom K=10, appearance up to 50
    print('Performing joint reconstruction (geom K=10, app K up to 50)')
    Kgeom = 10
    Kapp = 50
    comps_app_k = comps_app[:Kapp]
    comps_geom_k = comps_geom[:Kgeom]

    # For representative 20 test images, reconstruct and save comparisons
    n_show = min(20, len(test_ids))
    recon_errors = []
    for i in range(len(test_vs)):
        # reconstruct landmarks
        recon_lm_flat, coeffs_g = project_and_reconstruct(test_lms[i], mean_geom, comps_geom, Kgeom)
        recon_lm = recon_lm_flat.reshape(-1,2)

        # reconstruct appearance on aligned image
        recon_v_flat, coeffs_a = project_and_reconstruct(test_vs[i], mean_app, comps_app, Kapp)
        recon_v = recon_v_flat.reshape(IMG_SIZE, IMG_SIZE)

        # create aligned reconstructed color image by combining mean H,S with recon V
        aligned_recon_bgr = v_channel_to_bgr(mean_H, mean_S, recon_v)

        # warp back to reconstructed landmarks
        rec_warped = mywarper.warp(aligned_recon_bgr, recon_lm, (test_lms[i].reshape(-1,2)))

        # original image for comparison
        orig = original_test_images[i]
        # compute pixel MSE (normalized)
        diff = (rec_warped.astype(np.float32) - orig.astype(np.float32))
        mse = np.mean(np.sum(diff*diff, axis=2))
        recon_errors.append(mse)

        if i < n_show:
            # save side-by-side comparison
            cmp = np.hstack((orig, rec_warped))
            cv2.imwrite(os.path.join(OUT_DIR, f'recon_cmp_{i:02d}.jpg'), cmp)

    # plot recon error bar
    plt.figure()
    plt.plot(range(len(recon_errors)), recon_errors, '-o')
    plt.xlabel('Test image index')
    plt.ylabel('Pixel MSE')
    plt.title('Joint reconstruction pixel MSE (geom10+app50)')
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, 'joint_recon_errors.png'))

    # Save eigenfaces (first 10)
    print('Saving first 10 eigenfaces...')
    for i in range(10):
        ef = comps_app[i].reshape(IMG_SIZE, IMG_SIZE)
        ef_min, ef_max = ef.min(), ef.max()
        ef_img = 255.0 * (ef - ef_min) / (ef_max - ef_min + 1e-8)
        ef_img = ef_img.astype(np.uint8)
        cv2.imwrite(os.path.join(OUT_DIR, f'eigenface_{i+1:02d}.png'), ef_img)

    # Save eigen-warpings (first 10) as visualized shapes
    print('Saving first 10 eigen-warpings...')
    mean_shape_pts = mean_geom.reshape(-1,2)
    for i in range(10):
        vec = comps_geom[i].reshape(-1,2)
        # visualize mean +/- scaled eigenvector
        plt.figure(figsize=(3,3))
        plt.scatter(mean_shape_pts[:,0], mean_shape_pts[:,1], c='k', s=5)
        plt.quiver(mean_shape_pts[:,0], mean_shape_pts[:,1], vec[:,0], vec[:,1], angles='xy', scale_units='xy', scale=1)
        plt.gca().invert_yaxis()
        plt.axis('off')
        plt.savefig(os.path.join(OUT_DIR, f'eigenwarp_{i+1:02d}.png'), bbox_inches='tight', pad_inches=0)
        plt.close()

    # Generate 50 synthetic faces
    print('Generating 50 synthetic faces...')
    rng = np.random.RandomState(1234)
    n_gen = 50
    for i in range(n_gen):
        z_g = rng.randn(Kgeom) * np.sqrt(np.maximum(eigvals_geom[:Kgeom], 0))
        z_a = rng.randn(Kapp) * np.sqrt(np.maximum(eigvals_app[:Kapp], 0))
        # reconstruct landmarks
        recon_lm_flat = mean_geom + np.dot(z_g, comps_geom[:Kgeom])
        recon_lm = recon_lm_flat.reshape(-1,2)
        # reconstruct appearance
        recon_v_flat = mean_app + np.dot(z_a, comps_app[:Kapp])
        recon_v = recon_v_flat.reshape(IMG_SIZE, IMG_SIZE)
        aligned_recon_bgr = v_channel_to_bgr(mean_H, mean_S, recon_v)
        # warp aligned_recon to target recon_lm (warp from mean_shape to recon_lm)
        gen = mywarper.warp(aligned_recon_bgr, mean_shape, recon_lm)
        cv2.imwrite(os.path.join(OUT_DIR, f'generated_{i+1:02d}.jpg'), gen)

    print('Pipeline finished. Outputs in', OUT_DIR)

if __name__ == '__main__':
    main()