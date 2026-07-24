import cv2
import numpy as np

# 定义一个函数，用于获取鱼眼相机的内参矩阵K和畸变系数D
def  get_K_and_D(checkerboard, imgsPath, images_num, checkerboard_size):
    # checkerboard: 棋盘格的尺寸（内部角点的行数和列数）
    # imgsPath: 包含标定图像的文件夹路径
    # images_num: 标定图像的数量
    # checkerboard_size: 棋盘格每个方格的物理尺寸（例如，毫米）

    CHECKERBOARD = checkerboard
    # objp是世界坐标系中棋盘格角点的坐标
    objp = np.zeros((1,CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)*checkerboard_size
    _img_shape = None
    objpoints = [] # 存储世界坐标系中的点
    imgpoints = [] # 存储图像坐标系中的点

    # 遍历所有标定图像
    for i in range(images_num):
        img = cv2.imread(imgsPath+f'/{i+1}'+'.jpg')
        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            # 确保所有图像尺寸相同
            assert _img_shape == img.shape[:2], "所有图像必须具有相同的尺寸。"

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # 寻找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret == True:
            objpoints.append(objp)
            
            # 提高角点检测的精度
            cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
            imgpoints.append(corners)

            # 在图像上绘制并显示角点
            cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
            cv2.imshow(f'{i+1}'+'_corner.jpg', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        else:
            print(f"在 {i+1}.jpg' 中无法检测到角点")
    
    N_OK = len(objpoints)
    K = np.zeros((3, 3)) # 初始化内参矩阵
    D = np.zeros((4, 1)) # 初始化畸变系数
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)] # 旋转向量
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)] # 平移向量
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW#+cv2.fisheye.CALIB_CHECK_COND
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1e-6)

    # TODO: 执行鱼眼相机的相机标定，
    #  使用 'cv2.fisheye.calibrate' 函数完成以下代码。(一行代码)
    ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(objpoints, imgpoints, gray.shape[::-1], K, D, rvecs, tvecs, flags, criteria)
    DIM = _img_shape[::-1]
    print("内参矩阵:")
    print(str(K.tolist()))
    print("畸变系数:")
    print(str(D.tolist()))
    print("图像尺寸:")
    print(str(_img_shape[::-1])+"\n")

    err=0.0
    total_err=0.0
    print("每张图像的标定误差:")
    for j in range(images_num):
        # TODO: 使用 'cv2.fisheye.projectPoints' 函数，
        #  计算“每张图像的标定误差”。
        #  完成以下代码 (两行代码)。
        imgpoints2, _ = cv2.fisheye.projectPoints(objp.reshape(-1, 1, 3), rvecs[j], tvecs[j], K, D)
        err = cv2.norm(imgpoints[j], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_err+=err
        print(f"图像 {j+1} 的平均误差:"+str(err)+"像素")
    print("总体平均误差:"+str(total_err/images_num)+"像素")
    return DIM, K, D


def undistort(img_path,K,D,DIM,scale=0.85):
    # img_path: 待校正图像的路径
    # K: 内参矩阵
    # D: 畸变系数
    # DIM: 图像尺寸
    # scale: 缩放比例，用于控制校正后图像的裁剪程度

    # TODO: 读取并显示测试图像
    #  完成以下代码(两行代码)
    img = cv2.imread(img_path)
    cv2.imshow('test_distorted', img)
    dim1 = img.shape[:2][::-1]  # dim1是输入图像的尺寸
    # 确保待校正图像与标定图像具有相同的宽高比
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "待校正图像需要与标定图像具有相同的宽高比"
    if dim1[0]!=DIM[0]:
        img = cv2.resize(img,DIM,interpolation=cv2.INTER_AREA)
    Knew = K.copy()
    # 调整新的内参矩阵以进行缩放
    Knew[(0,0)] *= scale
    Knew[(1,1)] *= scale
    # TODO: 使用 ‘cv2.fisheye.initUndistortRectifyMap’ 和
    #  'cv2.remap' 函数来校正图像。完成以下代码(两行代码)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Knew, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow('test_undistorted', undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return undistorted_img

if __name__ == '__main__':
    # TODO: 根据训练数据中的标定板，
    #  完成 'checkerboard' 的尺寸（行和列的交点数）
    #  (一行代码)
    checkerboard = (11, 8)
    # 要校正鱼眼相机的畸变，第一步是
    # 执行相机标定以获得内参矩阵
    # 和畸变参数
    DIM, K, D = get_K_and_D(checkerboard, 'images', 20, 30.0)
    # 校正图像。
    undistort('test.jpg', K, D, DIM)
    img = undistort('./test.jpg',K,D,DIM)
    # 将校正后的图像保存到 result 文件夹
    cv2.imwrite('./result/test_undistorted.jpg', img)
