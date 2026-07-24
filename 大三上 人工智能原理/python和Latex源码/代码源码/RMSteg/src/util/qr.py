"""
QR 码生成与处理工具
==================
提供 QR 码生成、高斯核、随机消息生成等辅助功能

作者: RMSteg 项目
"""

import qrcode
import string
import torch
import random
from torchvision import transforms
import util.util as util
import numpy as np
import scipy.stats as st


def get_gaussian_kernel(size, sigma=1.0):
    """
    生成高斯卷积核（用于 QR 码模块平均池化）
    
    Args:
        size: 核大小 (size x size)
        sigma: 高斯标准差
        
    Returns:
        torch.Tensor: 归一化的高斯核 [1, 1, size, size]
        
    应用: 将 QR 码下采样到模块级别（每个模块平均为一个值）
    """
    x = np.linspace(-sigma, sigma, size + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return torch.from_numpy(kern2d/kern2d.sum()).float().unsqueeze(0).unsqueeze(0)


def get_random_message(length):
    """
    生成随机字符串消息
    
    Args:
        length: 字符串长度
        
    Returns:
        str: 包含小写字母、数字和特殊符号的随机字符串
        
    用于训练时生成多样化的 QR 码内容
    """
    letters = string.ascii_lowercase + "0123456789:;'\\,.<>[]{}-=_+|?!@#$%^&*()"
    rand_string = ''.join(random.choice(letters) for i in range(length))
    return rand_string


def save_qr_code(save_dir, version=5, box_size=20, max_text=40, size=(256, 256), message=None):
    """
    生成 QR 码并保存为图像
    
    Args:
        save_dir: 保存路径
        version: QR 码版本 (1-40，越大容量越大)
                 Version 5 = 37x37 模块
        box_size: 每个模块的像素大小（生成时）
        max_text: 随机消息的最大长度
        size: 最终输出图像尺寸 (H, W)
        message: 嵌入的文本内容，None 则随机生成
        
    QR 码规格:
        - 纠错级别: H (High, ~30% 可恢复)
        - 边框: 0 (无白边)
        - 输出: 二值图像 (0=白, 1=黑)
    """
    QRC = qrcode.QRCode(
        version=version,
        error_correction=qrcode.constants.ERROR_CORRECT_H,  # 高纠错级别
        box_size=box_size,
        border=0  # 去除白边，适合嵌入
    )
    
    # 生成或使用指定消息
    if message is None:
        message = get_random_message(random.randint(max_text - 10, max_text))
        
    QRC.add_data(message)
    QRC.make(fit=False)
    qr = QRC.make_image()

    # 转为 Tensor 并 resize
    qr = torch.from_numpy(np.array(qr)).unsqueeze(0).unsqueeze(0).float()
    qr = transforms.Resize(size)(qr)
    util.save_image_from_tensor(qr, save_dir)
    