import paddle                     # 导入 PaddlePaddle 深度学习框架
import paddle.nn as nn            # 导入 PaddlePaddle 的神经网络模块
import paddle.optimizer as optimizer # 导入 PaddlePaddle 的优化器模块
from paddle.io import DataLoader  # 导入 PaddlePaddle 的数据加载器
import os                         # 导入操作系统接口模块，用于文件和目录操作
import numpy as np                # 导入 NumPy 库，用于数值计算，特别是随机数生成
import copy                       # 导入 copy 模块，用于深度复制对象（如模型）
from models import UNet_conditional, EMA # 从 models.py 导入条件 UNet 模型和 EMA 类
from dataset import TrainData     # 从 dataset.py 导入自定义的数据集类
from tqdm import tqdm             # 导入 tqdm 库，用于显示训练进度条
from tools import my_logger, my_time, get_checkponit_epoch, LossTracker # 从 tools.py 导入日志、时间、检查点工具和损失追踪器
from StableDiffusion import Diffusion # 从 StableDiffusion.py 导入扩散模型类
from visualize import generate_and_save_samples, create_training_gif # 从 visualize.py 导入可视化函数

# 训练配置字典：集中管理训练过程中的超参数和设置
config = {
    "epochs": 500,                # 总训练轮次 (epoch)：整个数据集将被遍历多少次
    "batch_size": 4,             # 批次大小 (batch size)：每次训练迭代处理的样本数量
    "accum_steps": 4,             # 梯度累积步数：模拟更大的批次大小 (实际批次大小 = batch_size * accum_steps = 16)
                                  # 在显存不足时有用，每 accum_steps 次迭代才更新一次模型参数
    "num_classes": 6,             # 数据集中的类别数量，用于条件生成
    "lr": 1.5e-4,                 # 学习率 (learning rate)：控制模型参数更新的步长
    "image_size": 64,             # 训练图像的尺寸 (调整后的方形尺寸)
    "load_checkpoints": True,     # 是否加载预训练的模型检查点以继续训练
    "load_checkpoints_path": "weight/sd_unet_120.pdparams",  # 要加载的检查点文件路径
    "weight_dir": "weight",        # 保存模型权重（检查点）的目录名称
    "sample_dir": "samples",       # 保存生成的样本图像和损失曲线图的目录名称
    "save_interval": 20,           # 每隔多少个 epoch 保存一次模型检查点
    "sample_interval": 50,         # 每隔多少个 epoch 生成并保存一次样本图像
    "ema_decay": 0.995             # EMA (指数移动平均) 的衰减率，用于平滑模型参数
}

# 初始化目录：如果指定的目录不存在，则创建它们
os.makedirs(config["weight_dir"], exist_ok=True) # 创建权重保存目录
os.makedirs(config["sample_dir"], exist_ok=True) # 创建样本保存目录

# 定义训练函数
def train():
    # 设置设备：优先使用 CUDA (GPU)，如果不可用则使用 CPU
    device = "cuda" if paddle.device.is_compiled_with_cuda() else "cpu"
    my_logger.info(f"使用设备: {device}") # 记录使用的设备

    # 加载数据集
    my_logger.info("加载数据集...")
    dataset = TrainData() # 创建自定义数据集类的实例 (默认使用 data.txt)
    # 创建数据加载器，负责批量加载、打乱数据
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    my_logger.info(f"数据集加载完成，共 {len(dataset)} 个样本") # 记录数据集大小

    # 初始化模型
    start_epoch = 0 # 默认从第 0 轮开始
    # 检查是否需要加载检查点以及检查点文件是否存在
    if config["load_checkpoints"] and os.path.exists(config["load_checkpoints_path"]):
        # 从检查点文件名中提取之前训练到的 epoch 数
        start_epoch = get_checkponit_epoch(config["load_checkpoints_path"])
        # 创建条件 UNet 模型实例
        model = UNet_conditional(num_classes=config["num_classes"], device=device)
        my_logger.info(f"加载模型检查点: {config['load_checkpoints_path']}")
        # 加载检查点文件中的参数
        params = paddle.load(config["load_checkpoints_path"])
        # 将加载的参数设置到模型中
        model.set_state_dict(params)
        my_logger.info(f"模型加载成功! 从第 {start_epoch} 轮开始训练")
    else:
        # 如果不加载检查点或文件不存在，则从头开始训练
        my_logger.info("从头开始训练")
        # 创建新的条件 UNet 模型实例
        model = UNet_conditional(num_classes=config["num_classes"], device=device)

    # 优化器和损失函数
    # 使用 Adam 优化器，传入学习率和模型参数
    opt = optimizer.Adam(learning_rate=config["lr"], parameters=model.parameters())
    # 使用均方误差损失函数 (Mean Squared Error)，用于比较预测噪声和实际噪声
    mse = nn.MSELoss()

    # 初始化扩散模型：创建 Diffusion 类的实例，传入图像大小和设备信息
    diffusion = Diffusion(img_size=config["image_size"], device=device)

    # 设置 EMA (指数移动平均)
    ema = EMA(config["ema_decay"]) # 创建 EMA 类的实例，传入衰减率
    # 创建一个模型的深拷贝作为 EMA 模型 (影子模型)
    ema_model = copy.deepcopy(model)
    # 将 EMA 模型设置为评估模式，因为它不参与梯度计算，只用于参数平滑
    ema_model.eval()

    # 损失追踪器：用于记录和绘制训练过程中的损失变化
    loss_tracker = LossTracker()

    # --- 训练循环开始 ---
    # 从 start_epoch 开始，训练到 config["epochs"] 结束
    for epoch in range(start_epoch, config["epochs"]):
        model.train() # 将主模型设置为训练模式 (启用 Dropout 等)
        cache_loss = 0 # 用于累积一个 epoch 内的损失值
        # 使用 tqdm 创建进度条，迭代 dataloader 中的数据批次
        pbar = tqdm(dataloader, desc=f"[{my_time.get_time()}] Epoch {epoch}", position=0, leave=True)

        # --- 批次循环开始 ---
        # enumerate 提供批次索引 i 和批次数据 (images, labels)
        for i, (images, labels) in enumerate(pbar):
            # 获取当前批次的实际大小 (可能小于 config["batch_size"]，尤其在最后一个批次)
            B = images.shape[0]  # images 形状为 [B, C, H, W]

            # 1. 采样时间步长：为批次中的每个样本随机选择一个扩散时间步 t
            t = diffusion.sample_timesteps(B) # t 的形状为 [B,]

            # 2. 添加噪声：根据采样的时间步 t，向原始图像 images 添加噪声
            #    得到加噪后的图像 x_t 和实际添加的噪声 noise
            x_t, noise = diffusion.noise_images(images, t)

            # 3. 随机丢弃类别标签 (Classifier-Free Guidance 训练技巧)
            #    以 10% 的概率将 labels 设置为 None，让模型学会在没有条件的情况下预测噪声
            if np.random.random() < 0.1:
                labels = None

            # 4. 预测噪声：将加噪图像 x_t、时间步 t 和标签 labels 输入模型，得到预测的噪声
            predicted_noise = model(x_t, t, labels)

            # 5. 计算损失：计算预测噪声 predicted_noise 和实际添加的噪声 noise 之间的均方误差
            loss = mse(noise, predicted_noise)

            # 6. 反向传播和优化 (考虑梯度累积)
            # 将损失除以梯度累积步数，使得每次累积的梯度贡献是平均的
            loss = loss / config["accum_steps"]
            loss.backward() # 计算损失相对于模型参数的梯度

            # 只有当迭代次数是 accum_steps 的倍数时，才执行优化器步骤和梯度清零
            if (i + 1) % config["accum_steps"] == 0:
                opt.step() # 使用优化器根据梯度更新模型参数
                opt.clear_grad() # 清除之前的梯度，为下一次计算做准备

                # 更新 EMA 模型：在主模型参数更新后，平滑更新 EMA 模型的参数
                ema.step_ema(ema_model, model)

            # 更新进度条显示：显示当前的批次损失值
            # 注意：这里显示的是未除以 accum_steps 的损失，更直观反映单次预测的误差
            cache_loss += loss.item() * config["accum_steps"] # 累加未除以 accum_steps 的损失
            pbar.set_postfix(MSE=loss.item() * config["accum_steps"])
        # --- 批次循环结束 ---

        # 计算并记录当前 epoch 的平均损失
        avg_loss = cache_loss / len(dataloader.dataset) # 总损失除以总样本数
        my_logger.info(f"[{my_time.get_time()}] Epoch {epoch} 平均损失: {avg_loss:.6f}")

        # 将当前 epoch 的平均损失添加到追踪器
        loss_tracker.add(epoch, avg_loss)

        # 定期保存模型检查点 (使用 EMA 模型的参数)
        # 在指定的间隔或最后一个 epoch 保存
        if epoch % config["save_interval"] == 0 or epoch == config["epochs"] - 1:
            # 构造保存路径
            save_path = os.path.join(config["weight_dir"], f"sd_unet_{epoch}.pdparams")
            # 保存 EMA 模型的参数字典
            paddle.save(ema_model.state_dict(), save_path)
            my_logger.info(f"EMA 模型已保存到: {save_path}")

            # 保存当前的损失曲线图
            loss_tracker.plot(os.path.join(config["sample_dir"], f"loss_curve_{epoch}.png"))

        # 定期生成并保存样本图像 (使用 EMA 模型)
        # 在指定的间隔或最后一个 epoch 生成
        if epoch % config["sample_interval"] == 0 or epoch == config["epochs"] - 1:
            # 构造检查点路径 (虽然刚保存过，但这里是为了调用生成函数)
            save_path = os.path.join(config["weight_dir"], f"sd_unet_{epoch}.pdparams")
            # 调用函数生成并保存样本图像
            generate_and_save_samples(save_path, epoch, config["sample_dir"])
    # --- 训练循环结束 ---

    # 训练完成后，创建展示训练过程样本变化的 GIF 动图
    my_logger.info("训练完成，正在创建训练进度 GIF...")
    create_training_gif(config["sample_dir"])

# Python 入口点：当直接运行此脚本时执行
if __name__ == '__main__':
    my_logger.info("开始 Stable Diffusion 模型训练...")
    train() # 调用训练函数