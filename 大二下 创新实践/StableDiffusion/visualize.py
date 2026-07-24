import os
import paddle
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tools import visualize_samples, font, my_logger
from models import UNet_conditional
from StableDiffusion import Diffusion

# 设置绘图参数
plt.rcParams['figure.figsize'] = (15, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

def generate_and_save_samples(model_path, epoch, output_dir="samples"):
    """
    生成并保存样本图像
    :param model_path: 模型路径
    :param epoch: 当前epoch
    :param output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    model = UNet_conditional(num_classes=6)
    model.set_state_dict(paddle.load(model_path))
    
    # 初始化扩散模型
    diffusion = Diffusion(img_size=64, device="cuda")
    
    # 类别名称和标签
    name = ["建筑物", "森林", "冰川", "山峰", "大海", "街道"]
    labels = paddle.to_tensor([0, 1, 2, 3, 4, 5]).astype("int64")
    
    # 生成图像
    my_logger.info(f"正在生成第 {epoch} 轮的样本图像...")
    
    # 使用两种不同的标签引导强度
    for cfg_scale in [7, 10]:
        sampled_images = diffusion.sample(model, n=len(labels), labels=labels, cfg_scale=cfg_scale)
        fig = visualize_samples(sampled_images, labels, name, cfg_scale)
        
        # 保存图像
        save_path = os.path.join(output_dir, f"samples_epoch_{epoch}_cfg_{cfg_scale}.png")
        fig.savefig(save_path)
        plt.close(fig)
        
        my_logger.info(f"样本已保存到: {save_path}")

def create_training_gif(sample_dir="samples", output_path="training_progress.gif", pattern="samples_epoch_*_cfg_7.png"):
    """
    创建训练进度GIF
    :param sample_dir: 样本目录
    :param output_path: 输出GIF路径
    :param pattern: 文件匹配模式
    """
    import glob
    from PIL import Image
    
    # 获取所有样本图像
    sample_files = sorted(glob.glob(os.path.join(sample_dir, pattern)),
                         key=lambda x: int(x.split("_epoch_")[1].split("_")[0]))
    
    if not sample_files:
        my_logger.error(f"未找到匹配的样本图像: {os.path.join(sample_dir, pattern)}")
        return
    
    # 加载图像
    images = [Image.open(file) for file in sample_files]
    
    # 创建GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=500,  # 每帧持续时间（毫秒）
        loop=0  # 0表示无限循环
    )
    
    my_logger.info(f"训练进度GIF已保存到: {output_path}")

if __name__ == "__main__":
    # 测试生成样本
    model_path = "weight/sd_unet_450.pdparams"
    generate_and_save_samples(model_path, 450)
    
    # 如果有多个epoch的样本，可以创建GIF
    # create_training_gif()