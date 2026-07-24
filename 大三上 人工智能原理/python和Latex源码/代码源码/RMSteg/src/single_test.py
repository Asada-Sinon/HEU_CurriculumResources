"""
RMSteg 单图测试脚本
==================
演示完整的隐写-攻击-解码流程：
1. 加载预训练模型
2. 将 QR 码嵌入封面图像生成隐写图
3. 模拟攻击（扭曲/噪声）
4. 从攻击后的图像解码 QR 码
5. 保存所有中间结果供可视化

输出目录: ./result/
- test_host.png: 原始封面图像
- test_qr.png: 原始 QR 码
- test_steg.png: 隐写图像
- steg_res.png: 隐写残差（放大显示差异）
- test_trans_qr.png: 变换后的 QR 码
- test_distort.png: 攻击后的隐写图
- test_decode_qr.png: 解码出的 QR 码
- test_qr_error.png: QR 解码错误图（红色=错误，绿色=正确）

作者: RMSteg 项目
"""

import torch
import torch.nn as nn
from util.util import args
from util import util
import util.qr as uqr
from model.attnflow import AttnFlow
from torchvision import transforms
import os

# 配置
device = 'cpu'        # 使用 CPU 推理（可改为 'cuda' 加速）
num_module = 37       # QR 码 Version 5 的模块数 (37x37)
img_size = 224        # 统一图像尺寸

if __name__ == '__main__':
    # 创建输出目录
    os.makedirs('./result/', exist_ok=True)
    
    # 1. 加载预训练模型
    print("Loading model...")
    net = AttnFlow(block_num=4, use_itf=True, use_qr_trans=True, num_module=num_module)
    net.load_state_dict((torch.load('./pretrained/rmsteg.pth')))
    net = net.to(device)
    net.eval()

    # 2. 加载封面图像
    print("Loading cover image...")
    host_image = util.image_to_tensor('./test_img/test_host.png').to(device)
    host_image = transforms.Resize((img_size, img_size))(host_image)
    
    # 3. 生成 QR 码
    print("Generating QR code...")
    uqr.save_qr_code(
        save_dir='./result/qr.png',
        version=5,              # QR 码版本（决定容量）
        message='rmsteg'        # 嵌入的消息内容
    )
    qr = util.image_to_tensor('./result/qr.png').to(device)
    qr = transforms.Resize((img_size, img_size))(qr)[:, :1, ...]  # 取单通道
    
    # 4. 隐写编码：将 QR 码嵌入封面图像
    print("Encoding (embedding QR into image)...")
    with torch.no_grad():
        steg, trans_qr = net.encode(torch.cat([host_image, qr], dim=1))
    
    # 5. 模拟攻击（扭曲、噪声、压缩等）
    print("Applying distortions...")
    with torch.no_grad():
        distort = net.distort(steg)
    
    # 6. 解码：从攻击后的图像恢复 QR 码
    print("Decoding QR from distorted image...")
    with torch.no_grad():
        decode_qr = net.decode(distort)
    
    # 7. 保存所有结果
    print("Saving results to ./result/ ...")
    util.save_image_from_tensor(host_image, './result/test_host.png')
    util.save_image_from_tensor(qr, './result/test_qr.png')
    util.save_image_from_tensor(steg, './result/test_steg.png')
    util.save_image_from_tensor(torch.abs(steg - host_image), './result/steg_res.png')  # 残差可视化
    util.save_image_from_tensor(trans_qr, './result/test_trans_qr.png')
    util.save_image_from_tensor(distort, './result/test_distort.png')
    util.save_image_from_tensor(decode_qr, './result/test_decode_qr.png')
    util.save_image_from_tensor(util.get_error_map(qr, decode_qr, num_module=num_module), './result/test_qr_error.png')
    
    print("Done! Check ./result/ for output images.")