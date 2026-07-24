"""
RMSteg 数据加载器
================
用于训练的 PyTorch Dataset，负责加载封面图像和 QR 码配对数据。

特性:
- 随机裁剪数据增强
- 自动处理灰度图转 RGB
- 随机配对图像与 QR 码

作者: RMSteg 项目
"""

from torch.utils.data import Dataset
from torchvision import transforms
from util.util import args
from util import util
import torch
import glob
import random


class StegDataset(Dataset):
    """
    隐写术训练数据集
    
    加载封面图像 (Cover Images) 和 QR 码，进行随机配对用于隐写训练。
    
    Args:
        img_dir: 封面图像目录的 glob 模式 (如 '../../datasets/coco/train2017/*')
        qr_dir: QR 码图像目录的 glob 模式 (如 '../../datasets/qr_code/v5_new/*')
    
    数据增强:
        - 封面图像: 先 resize 到 1.5 倍，再随机裁剪到目标尺寸 (增加多样性)
        - QR 码: 直接 resize 到目标尺寸
    """
    
    def __init__(self, img_dir, qr_dir):
        self.img_size = (args.train.img_size, args.train.img_size)
        self.img_dir = img_dir
        self.qr_dir = qr_dir
        
        # 加载所有图像和 QR 码路径
        self.img_name_list = glob.glob(self.img_dir)
        self.qr_name_list = glob.glob(self.qr_dir)
        
        # 数据增强变换
        self.img_transform = transforms.Compose([
            transforms.Resize(int(self.img_size[0] * 1.5)),  # 先放大到 1.5 倍
            transforms.RandomCrop(self.img_size)              # 随机裁剪回目标尺寸
        ])
        self.qr_transform = transforms.Resize(self.img_size)
            
    
    def __len__(self):
        """数据集大小（以封面图像数量为准）"""
        return len(self.img_name_list)
    
    
    def __repr__(self):
        """数据集信息字符串"""
        return f'img num: {len(self.img_name_list)}\n' \
               f'qr num: {len(self.qr_name_list)}'
    
    
    def __getitem__(self, idx):
        """
        获取一对训练样本
        
        Args:
            idx: 封面图像索引
            
        Returns:
            dict: {'img': Tensor[3,H,W], 'qr': Tensor[1,H,W]}
                - img: 封面图像 (RGB)
                - qr: QR 码 (单通道)
        
        处理流程:
        1. 加载第 idx 张封面图像
        2. 随机选择一张 QR 码
        3. 灰度图转 RGB (如需要)
        4. 应用数据增强
        """
        # 加载封面图像
        img = util.image_to_tensor(self.img_name_list[idx])[0]
        
        # 随机选择 QR 码（不与图像固定配对，增加训练多样性）
        qr_idx = random.randint(0, len(self.qr_name_list) - 1)
        qr = util.image_to_tensor(self.qr_name_list[qr_idx])[0, :1, ...]  # 仅取第一通道
        
        # 灰度图转 RGB
        if (img.shape[0] == 1):
            img = torch.cat([img, img, img], dim=0)
        
        # 应用数据增强
        img = self.img_transform(img)
        qr = self.qr_transform(qr)[:1, ...]  # 确保 QR 为单通道
        
        return {'img': img, 'qr': qr}
            
        
        
