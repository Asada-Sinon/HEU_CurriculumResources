import os                         # 导入操作系统接口模块，用于文件路径操作
import logging                    # 导入日志记录模块
from datetime import datetime     # 导入日期时间模块，用于获取当前时间
import paddle                     # 导入 PaddlePaddle 框架 (虽然在此代码段未直接使用，但可能在其他函数中使用)
import numpy as np                # 导入 NumPy 库，用于数值计算
import matplotlib.pyplot as plt   # 导入 Matplotlib 绘图库
from PIL import Image             # 导入 Pillow 库，用于图像处理
from matplotlib.font_manager import FontProperties # 导入 Matplotlib 字体管理器，用于支持中文显示

# 定义字体路径，确保路径正确
font_path = "C:/Windows/Fonts/msyh.ttc"  # 微软雅黑字体路径（Windows 系统），用于 Matplotlib 显示中文
# 检查字体文件是否存在，如果不存在则记录警告
if not os.path.exists(font_path):
    logging.warning(f"字体文件未找到: {font_path}，Matplotlib 可能无法正确显示中文。")
    font = FontProperties() # 使用默认字体
else:
    font = FontProperties(fname=font_path)  # 创建 FontProperties 对象，指定中文字体

# --- 日志记录器设置 ---
# 基本配置：设置日志级别为 INFO，定义日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# 获取名为 "StableDiffusion" 的日志记录器实例
my_logger = logging.getLogger("StableDiffusion")
# 防止重复添加处理器（如果脚本被多次导入）
if not my_logger.handlers:
    # 添加文件处理器：将日志写入到 'training.log' 文件
    file_handler = logging.FileHandler('training.log')
    file_handler.setLevel(logging.INFO) # 设置文件处理器的日志级别
    # 定义写入文件的日志格式
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter) # 应用格式
    my_logger.addHandler(file_handler) # 将文件处理器添加到记录器

    # 添加控制台处理器：将日志输出到控制台（屏幕）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO) # 设置控制台处理器的日志级别
    # 定义输出到控制台的日志格式
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter) # 应用格式
    my_logger.addHandler(console_handler) # 将控制台处理器添加到记录器

# --- 时间工具 ---
class MyTime:
    """
    一个简单的类，用于获取格式化的当前时间。
    """
    @staticmethod # 静态方法，可以直接通过类名调用，无需创建实例
    def get_time():
        """获取当前时间的字符串表示，格式为 YYYY-MM-DD HH:MM:SS"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 创建 MyTime 类的实例，方便调用 get_time 方法
my_time = MyTime()

# --- 检查点工具 ---
def get_checkponit_epoch(checkpoint_path):
    """
    从检查点文件名中提取训练的 epoch 数。
    假设文件名格式类似 '..._epoch数.pdparams'，例如 'sd_unet_60.pdparams'。
    :param checkpoint_path: 检查点文件的完整路径或文件名
    :return: 提取的 epoch 数 (整数)，如果提取失败则返回 0
    """
    try:
        # 获取路径中的文件名部分
        filename = os.path.basename(checkpoint_path)
        # 按下划线分割文件名，取最后一部分
        # 再按点分割，取第一部分（即 epoch 数）
        epoch = int(filename.split('_')[-1].split('.')[0])
        return epoch
    except Exception as e:
        # 如果在提取过程中发生任何错误（例如文件名格式不符）
        my_logger.error(f"无法从检查点路径 '{checkpoint_path}' 中提取 epoch：{e}")
        return 0 # 返回 0 表示无法确定起始 epoch

# --- 可视化工具 ---
def visualize_samples(sampled_images, labels, name, cfg_scale, fixed_size=(256, 256)):
    """
    使用 Matplotlib 可视化一批生成的图像及其对应的标签。
    :param sampled_images: 生成的图像张量 (PaddlePaddle 张量或 NumPy 数组)，形状 (N, C, H, W)，值范围 [0, 255]
    :param labels: 图像对应的标签索引列表或数组 (N,)
    :param name: 标签索引到类别名称的映射字典或列表
    :param cfg_scale: 生成时使用的 Classifier-Free Guidance 尺度因子
    :param fixed_size: 可视化时将图像调整到的固定尺寸 (宽度, 高度)
    :return: Matplotlib 的 Figure 对象，包含了绘制的图像
    """
    # 创建一个新的 Matplotlib 图形窗口
    plt.figure(figsize=(15, 3)) # 设置图形大小
    # 设置图形的总标题，显示 CFG scale
    plt.suptitle(f"CFG Scale: {cfg_scale}", fontsize=16)

    # 遍历这批图像和标签
    for j in range(len(labels)):
        # 1. 准备图像数据：
        #    - 将图像从 (C, H, W) 转换为 (H, W, C) 以便 Matplotlib 显示
        #    - 如果是 Paddle 张量，先转为 NumPy 数组
        if isinstance(sampled_images, paddle.Tensor):
            img = sampled_images[j].numpy().transpose([1, 2, 0])
        else: # 假设已经是 NumPy 数组
            img = sampled_images[j].transpose([1, 2, 0])
        #    - 确保数据类型为 uint8 (0-255 整数)
        img = np.array(img).astype("uint8")
        #    - 使用 PIL 将 NumPy 数组转换为图像对象，并调整大小
        img = Image.fromarray(img).resize(fixed_size)

        # 2. 创建子图：在一行中显示所有图像
        #    参数：总行数，总列数，当前子图索引 (从 1 开始)
        plt.subplot(1, len(labels), j + 1)
        # 3. 显示调整大小后的图像
        plt.imshow(img)
        # 4. 设置子图标题为对应的类别名称，使用指定的中文字体
        plt.title(name[labels[j]], fontproperties=font)
        # 5. 关闭坐标轴显示
        plt.axis('off')

    # 自动调整子图布局，防止重叠
    plt.tight_layout()
    # 调整总标题的位置，避免与子图标题重叠
    plt.subplots_adjust(top=0.85)
    # 返回当前的 Matplotlib Figure 对象，方便外部调用者保存或进一步处理
    return plt.gcf()

# --- 损失追踪器 ---
class LossTracker:
    """
    一个简单的类，用于记录和绘制训练过程中的损失值。
    """
    def __init__(self):
        """初始化，创建空列表来存储 epoch 和对应的损失值"""
        self.epochs = []
        self.losses = []

    def add(self, epoch, loss):
        """
        添加一个 epoch 的损失记录。
        :param epoch: 当前的 epoch 数
        :param loss: 当前 epoch 的平均损失值
        """
        self.epochs.append(epoch)
        self.losses.append(loss)

    def plot(self, save_path=None):
        """
        使用 Matplotlib 绘制损失曲线图。
        :param save_path: 如果提供路径，则将图表保存到该文件；否则不保存。
        :return: Matplotlib 的 Figure 对象
        """
        # 创建一个新的 Matplotlib 图形窗口
        plt.figure(figsize=(10, 5)) # 设置图形大小
        # 绘制损失曲线，x 轴为 epoch，y 轴为 loss，蓝色实线 ('b-')
        plt.plot(self.epochs, self.losses, 'b-')
        # 设置图表标题，使用指定的中文字体
        plt.title('训练损失曲线', fontproperties=font)
        # 设置 x 轴标签，使用指定的中文字体
        plt.xlabel('Epoch', fontproperties=font)
        # 设置 y 轴标签，使用指定的中文字体
        plt.ylabel('Loss', fontproperties=font)
        # 显示网格线
        plt.grid(True)
        # 如果指定了保存路径
        if save_path:
            # 保存图表到文件
            plt.savefig(save_path)
            # 记录保存信息
            my_logger.info(f"损失曲线已保存到: {save_path}")
            # 关闭当前图形，释放内存 (如果连续绘制多个图，这很重要)
            plt.close()
        # 返回当前的 Matplotlib Figure 对象 (如果未保存，可以用于显示或进一步处理)
        # 注意：如果调用了 plt.close()，这里返回的 Figure 可能已关闭
        # 如果需要在保存后仍能操作 Figure，可以考虑不调用 plt.close() 或在调用前返回
        # return plt.gcf() # 原始代码返回了 gcf，但在保存后关闭可能导致问题
        # 更好的做法是，如果需要返回，就在 plt.close() 之前返回
        fig = plt.gcf()
        if save_path:
             plt.close(fig) # 显式关闭返回的 figure
        return fig # 返回 figure 对象