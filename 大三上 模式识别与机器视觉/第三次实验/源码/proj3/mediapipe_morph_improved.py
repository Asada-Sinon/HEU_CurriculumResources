import sys
import os
import cv2
import numpy as np
import mediapipe as mp
import mywarper
import imageio

"""
mediapipe_morph_improved.py

本脚本使用 MediaPipe 检测面部关键点，并生成从源脸 (z.jpg) 到目标脸 (r.jpg) 的
过渡动画（morph）。相比简单的线性混合，该脚本包含：
- 每帧对两张图像进行三角形仿射 warp（使用 mywarper.warp）
- 颜色迁移（Reinhard 在 Lab 空间）以减小颜色跳变
- 基于 Delaunay 三角区域的羽化掩膜（Gaussian feathering）以平滑边界
- 导出 GIF 与 MP4 两种格式，支持调节帧数、分辨率与羽化半径

注：为了简单演示，使用 MediaPipe 的 468 点人脸网格并用均匀采样降到 68 点。
如果需要严格的 68 点语义对应，请使用 face_alignment 或 dlib。
"""

IMG_A = os.path.join(os.path.dirname(__file__), 'r.jpg')  # target
IMG_B = os.path.join(os.path.dirname(__file__), 'z.jpg')  # source
OUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh

# Detect landmarks (returns Nx2 float coords)
def detect_landmarks(image):
    """
    使用 MediaPipe FaceMesh 检测面部网格点（468 点），并返回像素坐标形式的点数组 (468,2)。

    输入:
    - image: BGR numpy 数组
    返回:
    - pts: (468,2) float32 数组（x,y）或 None（未检测到人脸）
    """
    h, w = image.shape[:2]
    # 使用 MediaPipe FaceMesh 进行单张图片检测（静态模式）
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=False) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        lm = results.multi_face_landmarks[0]
        pts = []
        # 将归一化坐标转换为像素坐标
        for p in lm.landmark:
            pts.append([p.x * w, p.y * h])
        return np.array(pts, dtype=np.float32)

# reduce 468 to 68 by chosen indices (uniform sampling)
REDUCE_TO = 68
indices = np.linspace(0, 467, REDUCE_TO).astype(int)

# Reinhard color transfer (simple) in Lab space
# map source color distribution to target
def color_transfer_reinhard(src, target):
    """
    在 Lab 颜色空间中执行 Reinhard 颜色迁移：将 src 的颜色分布映射到 target 的颜色分布。

    输入/输出均为 BGR 图像（numpy uint8）。该方法简单且速度快，适合实时帧处理。
    """
    # 将图像转换到 Lab 空间（OpenCV 的 L:0..255, a,b:0..255）
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    tar_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    # compute mean and std per channel
    src_mean, src_std = cv2.meanStdDev(src_lab)
    tar_mean, tar_std = cv2.meanStdDev(tar_lab)
    # transfer (广播计算)：对每个通道做 (x - mean_src) * (std_tar / std_src) + mean_tar
    result = (src_lab - src_mean.reshape(1,1,3)) * (tar_std.reshape(1,1,3) / (src_std.reshape(1,1,3) + 1e-8)) + tar_mean.reshape(1,1,3)
    result = np.clip(result, 0, 255).astype(np.uint8)
    result_bgr = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result_bgr

# create feathered mask from triangular region (delaunay)
def create_feather_mask(shape, tri_indices, points, feather=15):
    """
    基于三角形索引列表构建区域掩膜，并对掩膜边缘做高斯羽化（平滑）。

    参数:
    - shape: (H,W)
    - tri_indices: Delaunay 三角索引列表，每个元素为 (i,j,k)
    - points: 三角顶点坐标数组，与 tri_indices 对应
    - feather: 羽化半径（像素），值越大边界越柔和

    返回:
    - mask_f: 浮点掩膜，范围 [0,1]
    """
    mask = np.zeros((shape[0], shape[1]), dtype=np.uint8)
    # fill triangles
    for tri in tri_indices:
        pts = np.array([points[tri[0]], points[tri[1]], points[tri[2]]], dtype=np.int32)
        cv2.fillConvexPoly(mask, pts, 255)
    # feather via gaussian blur: 将二值掩码平滑成 0..1 的浮点掩码
    k = max(3, int(feather) // 2 * 2 + 1)
    mask_f = cv2.GaussianBlur(mask.astype(np.float32), (k,k), sigmaX=feather)
    mask_f = (mask_f / mask_f.max()).astype(np.float32)
    return mask_f


def make_morph_frames_improved(imgA, imgB, lmA, lmB, n_frames=48, out_size=256, feather=25):
    # resize and scale
    H = out_size
    imgA_rs = cv2.resize(imgA, (H,H))
    imgB_rs = cv2.resize(imgB, (H,H))
    scale_x_A = H / imgA.shape[1]
    scale_y_A = H / imgA.shape[0]
    scale_x_B = H / imgB.shape[1]
    scale_y_B = H / imgB.shape[0]
    lmA_rs = np.stack([lmA[:,0]*scale_x_A, lmA[:,1]*scale_y_A], axis=1)[indices]
    lmB_rs = np.stack([lmB[:,0]*scale_x_B, lmB[:,1]*scale_y_B], axis=1)[indices]

    # compute delaunay on mean shape (用于构建三角剖分以便多边形羽化)
    mean_shape = ((lmA_rs + lmB_rs) / 2.0)
    rect = (0,0,H,H)
    delaunay = mywarper.calculateDelaunayTriangles(rect, mean_shape.astype(int).tolist())

    frames = []
    for t_i in range(n_frames+1):
        t = t_i / n_frames
        inter = (1-t)*lmB_rs + t*lmA_rs
        # warp both images to inter
        warpB = mywarper.warp(imgB_rs, lmB_rs, inter)
        warpA = mywarper.warp(imgA_rs, lmA_rs, inter)
    # color transfer: 将 warpB 的色彩统计映射到 warpA，使得颜色过渡更平滑
    warpB_col = color_transfer_reinhard(warpB, warpA)
    # create feather mask from triangles: 得到 0..1 的浮点掩码
    mask = create_feather_mask((H,H), delaunay, inter, feather=feather)
    mask3 = np.stack([mask,mask,mask], axis=2)
    # blend with feathering + cross-dissolve:
    # - 在三角形内部使用羽化后的 mask 去混合两张对齐后的图像
    # - 在掩码外部直接做像素级的线性交叉淡入淡出
    for t_i in range(n_frames+1):
        t = t_i / n_frames
        inter = (1-t)*lmB_rs + t*lmA_rs
        # warp both images to inter
        warpB = mywarper.warp(imgB_rs, lmB_rs, inter)
        warpA = mywarper.warp(imgA_rs, lmA_rs, inter)
        # color transfer: 将 warpB 的色彩统计映射到 warpA，使得颜色过渡更平滑
        warpB_col = color_transfer_reinhard(warpB, warpA)
        # create feather mask from triangles: 得到 0..1 的浮点掩码
        mask = create_feather_mask((H,H), delaunay, inter, feather=feather)
        mask3 = np.stack([mask,mask,mask], axis=2)
        # blend with feathering + cross-dissolve:
        # - 在三角形内部使用羽化后的 mask 去混合两张对齐后的图像
        # - 在掩码外部直接做像素级的线性交叉淡入淡出
        mixed = ((1-t) * (warpB_col * mask3) + t * (warpA * mask3) + ((1-mask3) * ((1-t)*warpB_col + t*warpA)))
        mixed = np.clip(mixed, 0, 255).astype(np.uint8)
        frames.append(mixed)
    return frames


def main():
    imgA = cv2.imread(IMG_A)
    imgB = cv2.imread(IMG_B)
    if imgA is None or imgB is None:
        print('Missing input images'); return
    lmA = detect_landmarks(imgA)
    lmB = detect_landmarks(imgB)
    if lmA is None or lmB is None:
        print('Face not detected in one of images'); return
    frames = make_morph_frames_improved(imgA, imgB, lmA, lmB, n_frames=48, out_size=256, feather=25)
    gif_path = os.path.join(OUT_DIR, 'morph_z_to_r_improved.gif')
    mp4_path = os.path.join(OUT_DIR, 'morph_z_to_r_improved.mp4')
    # save gif (fast)
    imageio.mimsave(gif_path, frames, duration=0.04)  # 25 fps
    # save mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(mp4_path, fourcc, 25.0, (w,h))
    for f in frames:
        out.write(f)
    out.release()
    print('Saved', gif_path, mp4_path)

if __name__ == '__main__':
    main()