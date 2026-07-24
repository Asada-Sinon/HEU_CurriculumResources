"""
RMSteg轻量化模型训练脚本

本脚本实现了轻量化QR码隐写模型的分布式训练，包含以下特性：
1. 多GPU分布式数据并行 (DDP)
2. 自动混合精度训练 (AMP)
3. 梯度累积
4. TensorBoard可视化
5. 可配置的模型架构

优化特性:
- 混合精度训练：减少50%显存，加速计算
- 数据加载优化：pin_memory + num_workers
- 梯度累积：支持大批次训练
- 非阻塞传输：CPU到GPU的异步传输

使用方法:
    单GPU: python train_lite.py --local_rank 0
    多GPU: python -m torch.distributed.launch --nproc_per_node=4 train_lite.py

作者: RMSteg Team
日期: 2025-11-26
版本: v1.0 Lite
"""

import torch
import torch.nn as nn
from util.util import args
from util import util
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import StegDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import time
import argparse
import os
from model.gan import Discriminator
from util.qr import get_gaussian_kernel
from model.attnflow_lite import AttnFlowLite
import lpips
from tqdm import tqdm

# 任务名称，用于保存checkpoint和日志
task_name = 'rmsteg_lite'
os.environ["CUDA_VISIBLE_DEVICES"] = args.train.cuda_devices
writer = SummaryWriter(log_dir=f'log/{task_name}/')

if __name__ == '__main__':
    # ========== 参数解析 ==========
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=-1, type=int, help="DDP参数，自动设置")
    parser.add_argument("--config", default="config_lite.yaml", type=str, help="配置文件路径")
    FLAGS = parser.parse_args()
    local_rank = FLAGS.local_rank
    
    # ========== 初始化分布式训练 ==========
    torch.cuda.set_device(local_rank)  # 设置当前进程使用的GPU
    dist.init_process_group(backend='nccl')  # 初始化进程组，使用NCCL后端（GPU通信）
    
    # ========== 创建输出目录 ==========
    os.makedirs('./result/', exist_ok=True)  # 保存训练过程中的图像结果
    os.makedirs('./log/', exist_ok=True)  # TensorBoard日志
    os.makedirs('./checkpoints/', exist_ok=True)  # 模型checkpoint
    
    # ========== 读取模型配置 ==========
    # 从config_lite.yaml中读取轻量化模型参数
    model_config = args.model if hasattr(args, 'model') else {}
    block_num = model_config.get('block_num', 2)  # 流模块数量，默认2
    vit_depth = model_config.get('vit_depth', 1)  # ViT深度，默认1
    dim = model_config.get('dim', 384)  # 隐藏维度，默认384
    dim_mlp = model_config.get('dim_mlp', 1024)  # MLP维度，默认1024
    latent_dim = model_config.get('latent_dim', 32)  # 潜在维度，默认32
    patch_size = model_config.get('patch_size', 16)  # Patch大小，默认16
    use_itf = model_config.get('use_itf', True)  # 是否使用Token Shuffle
    use_qr_trans = model_config.get('use_qr_trans', True)  # 是否使用QR码变换
    num_module = model_config.get('num_module', 37)  # QR码模块数
    
    # ========== 混合精度和梯度累积配置 ==========
    use_mixed_precision = args.train.get('use_mixed_precision', False) if hasattr(args.train, 'get') else False
    gradient_accumulation_steps = args.train.get('gradient_accumulation_steps', 1) if hasattr(args.train, 'get') else 1
    
    # ========== 创建轻量化模型 ==========
    net = AttnFlowLite(
        block_num=block_num,
        use_itf=use_itf,
        use_qr_trans=use_qr_trans,
        num_module=num_module,
        dim=dim,
        dim_mlp=dim_mlp,
        latent_dim=latent_dim,
        vit_depth=vit_depth,
        patch_size=patch_size
    ).to(local_rank)
    
    # ========== 加载预训练模型（可选）==========
    # 如果有预训练模型，可以取消下面的注释
    # net.load_state_dict((torch.load('./pretrained/rmsteg_lite.pth', map_location=f'cuda:{local_rank}')))
    
    # ========== 分布式数据并行包装 ==========
    net = DDP(net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    # ========== 打印模型信息 ==========
    # 只在主进程（rank 0）打印，避免重复输出
    if dist.get_rank() == 0:
        total_params = sum(p.numel() for p in net.parameters())
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"=" * 60)
        print(f"轻量化模型统计:")
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  模型大小 (FP32): {total_params * 4 / 1024 / 1024:.2f} MB")
        if use_mixed_precision:
            print(f"  模型大小 (FP16): {total_params * 2 / 1024 / 1024:.2f} MB")
        print(f"=" * 60)
    
    # ========== GAN判别器（可选）==========
    # 用于对抗训练，提升隐写图像的视觉质量
    use_gan = args.train.use_gan
    if use_gan:
        dis = Discriminator().to(local_rank)
        dis = DDP(dis, device_ids=[local_rank], output_device=local_rank, 
                 find_unused_parameters=True, broadcast_buffers=False)
    
    # ========== 感知损失计算器 ==========
    # LPIPS用于计算感知相似度（比像素级损失更符合人眼感知）
    calc_lpips = lpips.LPIPS(net='vgg').to(local_rank)
    
    # ========== 混合精度训练的缩放器 ==========
    # GradScaler用于防止FP16训练时的数值下溢
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
    
    # ========== 准备数据集和数据加载器 ==========
    dataset = StegDataset(img_dir=args.data.train_img_dir, qr_dir=args.data.train_qr_dir)
    if dist.get_rank() == 0:
        print(dataset)
    
    # 分布式采样器：确保每个GPU处理不同的数据
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    
    # 数据加载器优化：
    # - num_workers=4: 使用4个进程并行加载数据
    # - pin_memory=True: 将数据固定在内存中，加速CPU到GPU传输
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.train.batch_size, 
        sampler=sampler, 
        num_workers=4, 
        pin_memory=True
    )
    
    # ========== 优化器配置 - 主网络 ==========
    params_trainable_net = list(filter(lambda p: p.requires_grad, list(net.parameters())))
    optimizer_net = optim.AdamW(
        params_trainable_net,
        args.train.lr,  # 学习率
        betas=(args.train.betas1, args.train.betas2),  # Adam的beta参数
        weight_decay=args.train.weight_decay  # L2正则化
    )
    # 学习率调度器：每optim_step个epoch衰减一次
    lr_scheduler_net = optim.lr_scheduler.StepLR(
        optimizer_net, 
        args.train.optim_step, 
        gamma=args.train.optim_gamma
    )
    
    # ========== 优化器配置 - GAN判别器 ==========
    if use_gan:
        params_trainable_gan = (list(filter(lambda p: p.requires_grad, list(dis.parameters()))))
        optimizer_gan = optim.AdamW(
            params_trainable_gan,
            args.train.lr,
            betas=(args.train.betas1, args.train.betas2),
            weight_decay=args.train.weight_decay
        )
        lr_scheduler_gan = optim.lr_scheduler.StepLR(
            optimizer_net, 
            args.train.optim_step, 
            gamma=args.train.optim_gamma
        )
    
    # ========== 训练主循环 ==========
    iter_idx = 1  # 全局迭代计数器（用于TensorBoard）
    for epoch_idx in range(1, args.train.epoch_num + 1):
        net.train()  # 设置为训练模式
        dataloader.sampler.set_epoch(epoch_idx)  # 设置epoch，确保每个epoch的数据shuffle不同
        
        # 初始化epoch级别的统计量
        epoch_steg_loss = 0.0  # 隐写损失累计
        epoch_qr_loss = 0.0  # QR码重建损失累计
        epoch_qr_fusion_loss = 0.0  # QR码融合损失累计
        epoch_loss_dis = 0.0  # 判别器损失累计
        epoch_loss_dis_steg = 0.0  # 生成器对抗损失累计
        epoch_lpips = 0.0  # 感知损失累计
        epoch_ssim = 0.0  # SSIM累计
        epoch_s_time = time.time()  # 记录epoch开始时间
        
        # ========== 批次训练循环 ==========
        for data_idx, data in tqdm(enumerate(dataloader)):
            
            # 数据加载：使用non_blocking=True实现异步传输
            img = data['img'].to(local_rank, non_blocking=True)  # 封面图像 [B, 3, H, W]
            qr = data['qr'].to(local_rank, non_blocking=True)  # QR码 [B, 1, H, W]
            
            # ========== 第一步：训练GAN判别器（如果启用）==========
            if use_gan:
                # 使用混合精度的上下文管理器
                with torch.cuda.amp.autocast(enabled=use_mixed_precision):
                    # 前向传播
                    steg, distort, decode_qr, fusion_qr = net(torch.cat([img, qr], dim=1))
                    
                    # 判别器判断：真实图像应该输出1，生成图像应该输出0
                    dis_result_real = dis(torch.cat([img], dim=1))  # 真实图像
                    dis_result_fake = dis(torch.cat([steg.detach()], dim=1))  # 生成图像（detach阻止梯度回传）
                    dis_loss = dis.module.calc_loss_dis(dis_result_real, dis_result_fake)
                
                epoch_loss_dis += dis_loss.item() * args.train.batch_size
                
                # 反向传播和优化
                optimizer_gan.zero_grad()
                if use_mixed_precision:
                    scaler.scale(dis_loss).backward()  # 缩放损失
                    scaler.step(optimizer_gan)  # 更新参数
                else:
                    dis_loss.backward()
                    optimizer_gan.step()
            
            # ========== 第二步：训练主网络（编码器-解码器）==========
            with torch.cuda.amp.autocast(enabled=use_mixed_precision):
                # 前向传播：完整的隐写和提取流程
                steg, distort, decode_qr, fusion_qr = net(torch.cat([img, qr], dim=1))
                
                # 计算各项损失
                steg_loss, ssim_loss, qr_loss, qr_fusion_loss, refine_qr = net.module.calc_loss(
                    img, steg, qr, decode_qr, fusion_qr
                )
                
                # 计算感知损失（LPIPS）
                lpips_loss = calc_lpips(img, steg).reshape(-1).mean()
            
                # 总损失：加权组合各项损失
                # 权重设计：
                # - steg_loss × 5.0: 保证隐写图像与封面图像相似
                # - ssim_loss × 0.1: 保持结构相似度
                # - qr_loss × 20.0: 最重要，确保QR码能正确提取
                # - lpips_loss × 4.0: 保持感知质量
                total_loss = steg_loss * 5.0 + ssim_loss * 0.1 + qr_loss * 20.0 + lpips_loss * 4.0
                
                # 如果使用GAN，添加对抗损失
                if use_gan:
                    dis_result = dis(torch.cat([steg], dim=1))
                    dis_loss_steg = dis.module.calc_loss_net(dis_result)
                    epoch_loss_dis_steg += dis_loss_steg.item() * args.train.batch_size
                    total_loss += dis_loss_steg * 0.15  # 对抗损失权重较小
            
            # ========== 梯度累积 ==========
            # 将损失除以累积步数，实现梯度累积
            total_loss = total_loss / gradient_accumulation_steps
            
            # 反向传播和参数更新
            if use_mixed_precision:
                # 混合精度训练流程
                scaler.scale(total_loss).backward()  # 缩放损失并反向传播
                # 每accumulation_steps次更新一次参数
                if (data_idx + 1) % gradient_accumulation_steps == 0:
                    scaler.step(optimizer_net)  # 更新参数
                    scaler.update()  # 更新缩放因子
                    optimizer_net.zero_grad()  # 清空梯度
            else:
                # 标准训练流程
                total_loss.backward()
                if (data_idx + 1) % gradient_accumulation_steps == 0:
                    optimizer_net.step()
                    optimizer_net.zero_grad()
            
            # ========== 统计损失 ==========
            # 累积各项损失用于epoch级别的统计
            epoch_steg_loss += steg_loss.item() * img.shape[0]
            epoch_qr_loss += qr_loss.item() * img.shape[0]
            epoch_lpips += lpips_loss.item() * img.shape[0]
            epoch_ssim += (1.0 - ssim_loss.item()) * img.shape[0]  # SSIM越大越好
            epoch_qr_fusion_loss += qr_fusion_loss.item() * img.shape[0]
            
            # ========== 保存中间结果 ==========
            # 每20个batch保存一次图像，用于可视化训练过程
            if dist.get_rank() == 0 and data_idx % 20 == 0:
                # 每500个batch保存一次checkpoint
                if data_idx % 500 == 0:
                    torch.save(net.module.state_dict(), f'./checkpoints/{task_name}_{data_idx}.pth')
                    
                # 保存各种中间图像
                util.save_image_from_tensor(img, f'./result/epoch_{epoch_idx}_{data_idx}_cover.png')
                util.save_image_from_tensor(fusion_qr, f'./result/epoch_{epoch_idx}_{data_idx}_trans_qr.png')
                util.save_image_from_tensor(refine_qr, f'./result/epoch_{epoch_idx}_{data_idx}_refine_qr.png')
                util.save_image_from_tensor(steg, f'./result/epoch_{epoch_idx}_{data_idx}_steg.png')
                util.save_image_from_tensor(distort, f'./result/epoch_{epoch_idx}_{data_idx}_distort.png')
                util.save_image_from_tensor(qr, f'./result/epoch_{epoch_idx}_{data_idx}_qr.png')
                util.save_image_from_tensor(decode_qr , f'./result/epoch_{epoch_idx}_{data_idx}_decode_qr.png')
                util.save_image_from_tensor(util.get_error_map(qr, refine_qr, num_module=num_module) , 
                                          f'./result/epoch_{epoch_idx}_{data_idx}_qr_error_map.png')
               
            # ========== TensorBoard日志记录 ==========
            # 记录各项指标到TensorBoard，方便实时监控训练过程
            writer.add_scalar('Loss/Steg Loss', epoch_steg_loss / (data_idx * args.train.batch_size + img.shape[0]), iter_idx)
            writer.add_scalar('Loss/QR Loss', epoch_qr_loss / (data_idx * args.train.batch_size + img.shape[0]), iter_idx)
            writer.add_scalar('Loss/QR Fusion Loss', epoch_qr_fusion_loss / (data_idx * args.train.batch_size + img.shape[0]), iter_idx)
            if use_gan:
                writer.add_scalar('Loss/Dis Loss', epoch_loss_dis / (data_idx * args.train.batch_size + img.shape[0]), iter_idx)
                writer.add_scalar('Loss/Dis Steg Loss', epoch_loss_dis_steg / (data_idx * args.train.batch_size + img.shape[0]), iter_idx)
                
            writer.add_scalar('Metrices/SSIM', epoch_ssim / (data_idx * args.train.batch_size + img.shape[0]), iter_idx)
            writer.add_scalar('Metrices/LPIPS', epoch_lpips / (data_idx * args.train.batch_size + img.shape[0]), iter_idx)
            
            iter_idx += 1  # 全局迭代计数器
            
        # ========== Epoch结束处理 ==========
        # 更新学习率
        lr_scheduler_net.step()
        if use_gan:
            lr_scheduler_gan.step()
        
        epoch_t_time = time.time()  # 记录epoch结束时间
        
        # ========== 保存checkpoint ==========
        # 每save_freq个epoch保存一次，或者第一个epoch一定保存
        if dist.get_rank() == 0 and epoch_idx % args.train.save_freq == 0 or epoch_idx == 1:
            torch.save(net.module.state_dict(), f'./checkpoints/{task_name}_epoch_{epoch_idx}.pth')
            if use_gan:
                torch.save(dis.module.state_dict(), f'./checkpoints/{task_name}_dis_epoch_{epoch_idx}.pth')
        
        # ========== 打印训练统计信息 ==========
        # 只在主进程打印，避免多GPU重复输出
        if dist.get_rank() == 0:
            print(f'Epoch {epoch_idx}/{args.train.epoch_num} -- ' \
                f'LR: {optimizer_net.state_dict()["param_groups"][0]["lr"]:.8f}  ' \
                f'Steg Loss: {epoch_steg_loss * torch.cuda.device_count() / len(dataset):.8f}  ' \
                f'SSIM: {epoch_ssim * torch.cuda.device_count() / len(dataset):.8f}  ' \
                f'QR Loss: {epoch_qr_loss * torch.cuda.device_count() / len(dataset):.8f}  ' \
                f'LPIPS: {epoch_lpips * torch.cuda.device_count() / len(dataset):.8f}  ' \
                f'Time: {epoch_t_time - epoch_s_time:.1f}s')
