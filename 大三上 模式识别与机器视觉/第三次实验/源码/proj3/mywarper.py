# -*- coding: utf-8 -*-
"""
mywarper.py

本文件包含用于三角形仿射变形（triangle-to-triangle warping）的一组工具函数。
主要功能：
- 计算并应用仿射变换（applyAffineTransform）
- 基于 Delaunay 三角剖分，对人脸图像按三角形进行局部仿射变换（warpTriangle、warp）
- 计算 Delaunay 三角（calculateDelaunayTriangles）以及一些辅助函数

这些函数被用于：将图像从一组关键点（landmarks）几何变换到另一组关键点，常用在
人脸对齐（warp 到 mean shape）、两幅人脸之间的变形（morph）等任务。

约定：
- 图像采用 HxWxC 的 numpy 数组（BGR，OpenCV 风格）
- 关键点为 (N,2) 的数组或 list，坐标格式为 [x,y]

作者/修改：根据课程项目需要做了注释与部分实现补全。
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import datetime
import imageio

##################################
# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.

def applyAffineTransform(src, srcTri, dstTri, size) :
    """
    对 src 矩形区域应用从 srcTri -> dstTri 的仿射变换并返回目标大小的图像块。

    参数:
    - src: 源图像块（通常是从原图切下的小矩形），numpy 数组
    - srcTri: 源三角形的三个点，相对于 src 矩形（局部坐标）
    - dstTri: 目标三角形的三个点，相对于目标矩形（局部坐标）
    - size: 输出大小 (width, height)

    返回值:
    - dst: 经过仿射变换后的图像块，dtype=float32，shape=(height, width[, channels])

    说明:
    - 使用 OpenCV 的 getAffineTransform 得到 2x3 仿射矩阵，然后使用 warpAffine 执行变换。
    - borderMode 使用 BORDER_REFLECT_101 来减少边界伪影。
    """
    # Convert triangles to numpy float32
    srcTri_np = np.array(srcTri, dtype=np.float32)
    dstTri_np = np.array(dstTri, dtype=np.float32)

    # 给定三角形对，计算仿射变换矩阵（3 点 -> 仿射变换）
    warpMat = cv2.getAffineTransform(srcTri_np, dstTri_np)

    # 将仿射变换应用到 src 图像块，得到指定大小的输出
    dst = cv2.warpAffine(src, warpMat, (int(size[0]), int(size[1])), None,
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # 返回 float32 以便后续与 mask 相乘不出现截断
    return dst.astype(np.float32)


# Check if a point is inside a rectangle
def rectContains(rect, point) :
    """
    判断一个点是否位于给定矩形内（包含边界）。

    参数:
    - rect: (x, y, w, h) 的元组或列表，表示矩形左上角和宽高。
    - point: (x, y) 的点坐标。

    返回:
    - True 如果点位于矩形内（含边界），否则 False。

    注意: 该函数只做简单的坐标比较，不做浮点容差处理。
    """
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[0] + rect[2] :
        return False
    elif point[1] > rect[1] + rect[3] :
        return False
    return True

#calculate delanauy triangle
def calculateDelaunayTriangles(rect, points):
    """
    基于 OpenCV Subdiv2D 计算 Delaunay 三角剖分，并将三角顶点坐标映射回原始点索引。

    参数:
    - rect: 用于 Subdiv2D 的矩形 (x, y, w, h)，通常是整张图的边界。
    - points: 点坐标列表或数组，元素为 [x,y]。函数会尝试把 Subdiv2D 返回的三角顶点与 points 列表匹配，
      并返回对应的索引三元组 (i,j,k)。

    返回:
    - delaunayTri: 列表，每个元素为三元组 (i,j,k)，表示 points 列表中的索引，适用于后续按三角形进行局部仿射变换。

    说明:
    - Subdiv2D 返回的三角顶点是坐标形式，本函数通过在 points 中以小阈值匹配坐标来寻找对应索引；
      因此要求 points 中包含与 Subdiv2D 插入时相同的离散化坐标（通常为整数）。
    """
    # create subdiv
    subdiv = cv2.Subdiv2D(rect);

    # Insert points into subdiv (整数化坐标)
    for p in points:
        p1 = (int(p[0]), int(p[1]))
        # 简单的边界检查，避免插入越界点
        if p1[1] <= rect[1] + rect[3] - 1 and p1[0] <= rect[0] + rect[2] - 1 and p1[1] >= rect[1] and p1[0] >= rect[0]:
            subdiv.insert(p1)

    triangleList = subdiv.getTriangleList();

    delaunayTri = []

    pt = []

    for t in triangleList:
        # t 是 6 元素的向量 (x1,y1,x2,y2,x3,y3)
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        # 仅在三个顶点均在矩形内时考虑
        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            # 将 Subdiv2D 的顶点坐标与原始 points 列表进行匹配，找到索引
            for j in range(0, 3):
                for k in range(0, len(points)):
                    if (abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)
            # 若成功匹配到 3 个索引，则保存该三角
            if len(ind) == 3:
                delaunayTri.append((ind[0], ind[1], ind[2]))

        pt = []

    return delaunayTri

# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2) :
    """
    将 img1 中由三角形 t1 指定的区域仿射变换并写入 img2 中由 t2 指定的对应区域。

    参数:
    - img1: 源图像（numpy 数组），通常为 BGR 或灰度，shape=(H,W,channels)
    - img2: 目标图像（会被修改），与 img1 尺寸相同
    - t1: 源三角形顶点列表 [(x1,y1),(x2,y2),(x3,y3)]，以 img1 的坐标系为基准
    - t2: 目标三角形顶点列表，坐标系为 img2

    处理流程:
    1. 计算源/目标三角形的包围矩形（bounding rect），并把三角形顶点转换为矩形内的局部坐标。
    2. 使用 applyAffineTransform 对源矩形块进行仿射变换得到目标矩形大小的图块。
    3. 构建三通道的遮罩（mask），仅保留三角形内部像素，将变换后的图块融合回目标图像的对应矩形区域。

    注意: 所有中间运算使用 float32，最后直接在 img2 上覆盖写入（更快，避免额外拷贝）。
    """

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    w, h, num_chans = img1.shape
    # Get mask by filling triangle (mask size: r2.h x r2.w)
    mask = np.zeros((r2[3], r2[2], num_chans), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    if num_chans == 1:
        img2Rect = np.reshape(img2Rect, (r2[3], r2[2], num_chans))

    # 仅保留三角形内部的像素
    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    if num_chans == 1:
        img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * (1.0 - mask)
    else:
        img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ((1.0, 1.0, 1.0) - mask)

    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect

###################################
def warp(Image,sc,tc):
    '''
    Image: the image to be warped
    sc: original landmarks
    tc: warped landmarks
    '''
    """
    将图像 Image 从源 landmarks sc 仿射变换（按三角形分块）为目标 landmarks tc。

    参数:
    - Image: numpy 数组，BGR 图像，shape=(H,W,channels)，要求为方形（在本项目中通常为 HxH）。
    - sc: 源人脸关键点数组，shape=(N,2) 或能转换为 list 的点集合，单位像素坐标。
    - tc: 目标（或参考）人脸关键点数组，shape=(N,2)。sc 与 tc 应该一一对应。

    返回:
    - imgWarped: 仿射对齐后的图像（与 Image 同类型）。

    说明:
    - 函数内部会在点集合后附加四个角点以保证 Delaunay 覆盖整个图像区域。
    - 使用 mywarper.calculateDelaunayTriangles 计算三角剖分，然后对每个三角形调用 warpTriangle 完成局部仿射变换。
    """

    HW,_,_=Image.shape
    cornerps=[[0,0],[0,HW-1],[HW-1,0],[HW-1,HW-1]]
    #cornerps=[[0,0],[0,HW-1],[HW-1,0],[HW-1,HW-1],[0,np.floor(HW/2)],[np.floor(HW/2),0],[HW-1,np.floor(HW/2)],[np.floor(HW/2),HW-1]]

    scl=sc.astype(np.int64).tolist()+cornerps
    tcl=tc.astype(np.int64).tolist()+cornerps
    imgWarped = np.copy(Image);    
    rect = (0, 0, HW, HW)
    dt = calculateDelaunayTriangles(rect,tcl)
# Apply affine transformation to Delaunay triangles
    for i in range(0, len(dt)):
        t1 = []
        t2 = []
        
        #get points for img1, img2 corresponding to the triangles
        # dt contains indices into the point lists (tcl/scl)
        tri = dt[i]
        for j in range(0, 3):
            # index into the landmark+corner lists
            idx = tri[j]
            # scl and tcl are Python lists of [x,y]
            t1.append(scl[idx])
            t2.append(tcl[idx])
        
        warpTriangle(Image, imgWarped, t1, t2)
    return imgWarped

#########################################
def plot(samples,Nh,Nc,channel,IMG_HEIGHT, IMG_WIDTH):
    fig = plt.figure(figsize=(Nc, Nh))
    plt.clf()
    gs = gridspec.GridSpec(Nh, Nc)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples[0:Nh*Nc,:,:,:]):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if channel==1:
            image=sample.reshape(IMG_HEIGHT, IMG_WIDTH)
            immin=(image[:,:]).min()
            immax=(image[:,:]).max()
            image=(image-immin)/(immax-immin+1e-8)
            plt.imshow(image,cmap ='gray')
        else:
            image=sample.reshape(IMG_HEIGHT, IMG_WIDTH,channel)
            immin=(image[:,:,:]).min()
            immax=(image[:,:,:]).max()
            image=(image-immin)/(immax-immin+1e-8)
            plt.imshow(image)
    return fig 

