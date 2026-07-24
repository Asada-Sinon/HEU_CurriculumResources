"""
RMSteg 工具函数库
=================
提供图像处理、指标计算、QR 码评估等常用工具函数

作者: RMSteg 项目
"""

import torch
from torchvision import transforms, utils
from PIL import Image
from easydict import EasyDict
import yaml
import matplotlib.pyplot as plt
import torch.nn as nn
import util.qr as uqr
from einops import rearrange
try:
    from pyzbar import pyzbar
except Exception:
    pyzbar = None


def resize_qr_unit(qr_unit, factor):
    """
    将 QR 码模块上采样（每个模块复制为 factor x factor 像素块）
    
    Args:
        qr_unit: QR 模块级 Tensor [B, C, num_module, num_module]
        factor: 上采样倍数
        
    Returns:
        上采样后的 Tensor [B, C, num_module*factor, num_module*factor]
        
    应用: 将低分辨率的 QR 码模块矩阵放大为高分辨率图像
    """
    unit_size = qr_unit.shape[-1]
    qr_unit = qr_unit.reshape(qr_unit.shape[0], qr_unit.shape[1], -1, 1)
    qr_unit = qr_unit.repeat(1, 1, 1, factor * factor)
    qr_unit = rearrange(qr_unit, 'b c (h w) (f1 f2) -> b c (h f1) (w f2)', h=unit_size, w=unit_size, f1=factor, f2=factor)
    return qr_unit


def save_image_from_tensor(t, filename):
    """
    将 Tensor 保存为图像文件
    
    Args:
        t: 图像 Tensor，支持多种形状:
           - [C, H, W]
           - [B, C, H, W]
           - [H, W, C] (会自动转置)
        filename: 保存路径
        
    自动处理:
    - 维度扩展/调整
    - CPU 转移
    - 数值范围限制 [0, 1]
    """
    while len(t.shape) < 4:
        t = t.unsqueeze(0)
    if t.shape[-1] == 1 or t.shape[-1] == 3:
        t = t.permute(0, 3, 1, 2)
    t = t.clone().detach()
    t = t.to(torch.device('cpu'))
    utils.save_image(t, filename)
    
    
def image_to_tensor(img_path):
    """
    加载图像文件为 Tensor
    
    Args:
        img_path: 图像文件路径
        
    Returns:
        Tensor: [1, C, H, W]，范围 [0, 1]
        
    自动处理:
    - RGBA 转 RGB (丢弃 Alpha 通道)
    - 归一化到 [0, 1]
    """
    rgb = transforms.ToTensor()(Image.open(img_path))
    rgb = rgb.unsqueeze(0)
    if rgb.shape[1] == 4:
        rgb = rgb[:, :3, ...]  # 去除透明通道
    return rgb
        
    
def calc_psnr(tensor1, tensor2):
    """
    计算峰值信噪比 (Peak Signal-to-Noise Ratio)
    
    Args:
        tensor1, tensor2: 待比较的图像 Tensor
        
    Returns:
        float: PSNR 值（越高越好）
        
    公式: PSNR = 20 * log10(MAX / MSE^0.5)
    """
    l1_loss = nn.L1Loss()
    l1 = l1_loss(tensor1, tensor2)
    psnr = 20 * torch.log10(1 / l1)
    return psnr
    
    
def calc_emr(qr, decode_qr, num_module):
    """
    计算 QR 码模块错误率 (Error Module Ratio)
    
    Args:
        qr: 原始 QR 码 [B, 1, H, W]
        decode_qr: 解码 QR 码 [B, 1, H, W]
        num_module: QR 版本的模块数 (如 Version 5 = 37)
        
    Returns:
        float: 错误率百分比 (0-100)
        
    流程:
    1. Resize 到 num_module*5 分辨率
    2. 高斯池化下采样到模块级 (5x5 -> 1 值)
    3. 四舍五入为二值
    4. 计算不匹配比例
    """
    qr_trans = transforms.Resize((num_module * 5, num_module * 5))
    qr = qr_trans(qr)[:, :1, ...]
    decode_qr = qr_trans(decode_qr)[:, :1, ...]
    gs_kernel = uqr.get_gaussian_kernel(5, 1.0)
    gs_conv = nn.Conv2d(1, 1, 5, padding=0, stride=5).requires_grad_(False)
    gs_conv.weight.data = gs_kernel
    gs_conv = gs_conv.to(qr.device)
    qr = gs_conv(qr.clamp(0.0, 1.0))
    decode_qr = gs_conv(decode_qr.clamp(0.0, 1.0))
    error_map = torch.round(torch.abs(qr - decode_qr))
    emr = error_map.mean().reshape(-1).item()
    return emr * 100  # 百分比


def calc_tra(decode_qr):
    """
    计算 Token 恢复准确率 (Token Recovery Accuracy)
    
    Args:
        decode_qr: 解码的 QR 码 Tensor [1, 1, H, W]
        
    Returns:
        int: 1 表示可成功解码，0 表示解码失败
        
    使用 pyzbar 库尝试解码 QR 码，成功则返回 1，失败则返回 0
    """
    decode_qr = torch.cat([decode_qr, decode_qr, decode_qr], dim=1).clamp(0.0, 1.0)[0].cpu()[:3, ...]
    decode_qr = transforms.ToPILImage()(decode_qr)
    try:
        if pyzbar is None:
            return 0
        _ = pyzbar.decode(decode_qr, symbols=[pyzbar.ZBarSymbol.QRCODE])[0].data.decode("utf-8")
        return 1
    except Exception:
        return 0


    
def get_error_map(gt, pred, num_module):
    """
    生成 QR 码错误可视化图
    
    Args:
        gt: 真实 QR 码
        pred: 预测 QR 码
        num_module: 模块数
        
    Returns:
        可视化 Tensor: 绿色=正确，红色=错误
        
    用于直观展示 QR 解码的错误位置
    """
    red = transforms.Resize((args.train.img_size, args.train.img_size))(image_to_tensor('./test_img/misc/red.png')).to(pred.device)
    green = transforms.Resize((args.train.img_size, args.train.img_size))(image_to_tensor('./test_img/misc/green.png')).to(pred.device)
    error = torch.abs(gt - pred)
    error = transforms.Resize((num_module, num_module))(error)
    error = resize_qr_unit(error, factor=7)
    error = transforms.Resize((args.train.img_size, args.train.img_size))(error)
    error_map = torch.round(error) * red + (1.0 - torch.round(error)) * green
    return error_map


# 加载全局配置
with open('config.yaml', 'r') as f:
    args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))
    
    
if __name__ == '__main__':
    print(args)