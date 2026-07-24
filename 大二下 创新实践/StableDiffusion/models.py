import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# EMA (Exponential Moving Average) 类：用于平滑模型参数，提高模型在评估时的稳定性和性能
class EMA:
    """
    模型参数的指数移动平均。
    在训练过程中维护一个模型的影子副本，其参数是原始模型参数的平滑平均值。
    这通常能在评估时提供更好的性能。
    """
    def __init__(self, beta):
        """
        初始化 EMA 对象。
        :param beta: 平滑因子 (decay rate)，通常接近 1.0，例如 0.999。
        """
        super().__init__()
        self.beta = beta  # 保存平滑因子
        self.step = 0     # 初始化内部计数器，用于跟踪更新次数

    def update_model_average(self, ma_model, current_model):
        """
        使用当前模型的参数更新 EMA 模型（影子模型）的参数。
        :param ma_model: EMA 模型 (影子模型)
        :param current_model: 当前正在训练的模型
        """
        # 遍历当前模型和 EMA 模型的对应参数
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            # 获取旧的 EMA 参数和新的当前模型参数
            old_weight, up_weight = ma_params.data, current_params.data # 使用 .data 避免梯度追踪
            # 计算新的 EMA 参数并更新
            ma_params.data = self.update_average(old_weight, up_weight) # 直接更新 .data

    def update_average(self, old, new):
        """
        计算单个参数的指数移动平均值。
        公式: new_average = old_average * beta + new_value * (1 - beta)
        :param old: 旧的 EMA 参数值
        :param new: 新的当前模型参数值
        :return: 更新后的 EMA 参数值
        """
        if old is None: # 如果 EMA 参数尚未初始化
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=1000):
        """
        执行一步 EMA 更新。通常在每个训练步骤后调用。
        :param ema_model: EMA 模型
        :param model: 当前训练模型
        :param step_start_ema: 在多少步之后开始应用 EMA 更新（默认为 1000 步）
        """
        # 如果当前步数小于开始 EMA 的步数
        if self.step < step_start_ema:
            # 将 EMA 模型的参数重置为当前模型的参数
            self.reset_parameters(ema_model, model)
            self.step += 1 # 增加步数计数
            return # 提前返回，不进行平均更新
        # 如果达到开始 EMA 的步数，则执行正常的 EMA 更新
        self.update_model_average(ema_model, model)
        self.step += 1 # 增加步数计数

    def reset_parameters(self, ema_model, model):
        """
        将 EMA 模型的参数完全设置为当前模型的参数。
        通常在 EMA 开始阶段或需要重置时使用。
        :param ema_model: EMA 模型
        :param model: 当前训练模型
        """
        # 直接复制当前模型的 state_dict (包含所有参数和缓冲区) 到 EMA 模型
        ema_model.set_state_dict(model.state_dict())

# --- 下面是 UNet 模型及其组件的定义 ---

# 自注意力机制模块
class SelfAttention(nn.Layer):
    """
    自注意力模块。允许模型在处理图像的不同部分时关注其他相关部分。
    常用于捕捉长距离依赖关系。
    """
    def __init__(self, channels):
        """
        初始化自注意力模块。
        :param channels: 输入特征图的通道数。
        """
        super(SelfAttention, self).__init__()
        self.channels = channels
        # 使用 PaddlePaddle 内置的多头注意力机制，头数设为 4
        self.mha = nn.MultiHeadAttention(embed_dim=channels, num_heads=4)
        # 层归一化 (Layer Normalization)，对通道维度进行归一化
        self.ln = nn.LayerNorm(channels)
        # 前馈网络 (Feed Forward Network)，包含两层线性变换和 GELU 激活函数
        self.ff_self = nn.Sequential(
            nn.LayerNorm(channels),           # 先进行层归一化
            nn.Linear(channels, channels),    # 线性变换
            nn.GELU(),                        # GELU 激活函数
            nn.Linear(channels, channels),    # 再次线性变换
        )

    def forward(self, x):
        """
        前向传播。
        :param x: 输入张量，形状为 (batch_size, channels, height, width)
        :return: 输出张量，形状与输入相同
        """
        batch_size, channels, height, width = x.shape
        # 1. 调整形状以适应 MultiHeadAttention：
        #    (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        x = x.reshape([batch_size, channels, height * width]).transpose([0, 2, 1])
        # 2. 对输入进行层归一化
        x_ln = self.ln(x)
        # 3. 计算多头自注意力：Query, Key, Value 都来自 x_ln
        #    输出 attention_value 的形状为 (B, H*W, C)
        attention_value = self.mha(x_ln, x_ln, x_ln) # Q=K=V
        # 4. 残差连接：将注意力输出与原始输入相加
        attention_value = attention_value + x
        # 5. 通过前馈网络，并再次进行残差连接
        attention_value = self.ff_self(attention_value) + attention_value
        # 6. 恢复形状：
        #    (B, H*W, C) -> (B, C, H*W) -> (B, C, H, W)
        return attention_value.transpose([0, 2, 1]).reshape([batch_size, channels, height, width])

# 双卷积块
class DoubleConv(nn.Layer):
    """
    包含两个卷积层、归一化层和激活函数的块。UNet 的基本构建单元。
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        """
        初始化双卷积块。
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param mid_channels: 中间卷积层的通道数。如果为 None，则默认为 out_channels。
        :param residual: 是否使用残差连接 (将输入加到输出上)。
        """
        super().__init__()
        self.residual = residual
        if not mid_channels: # 如果未指定中间通道数
            mid_channels = out_channels
        # 定义包含两个卷积序列的 Sequential 容器
        self.double_conv = nn.Sequential(
            # 第一个卷积层：输入通道 -> 中间通道，3x3 卷积核，padding=1 保持尺寸，无偏置
            nn.Conv2D(in_channels, mid_channels, kernel_size=3, padding=1, bias_attr=False),
            # 组归一化 (Group Normalization)，这里组数为 1，等效于层归一化 (Layer Normalization)
            nn.GroupNorm(1, mid_channels),
            # GELU 激活函数
            nn.GELU(),
            # 第二个卷积层：中间通道 -> 输出通道，3x3 卷积核，padding=1 保持尺寸，无偏置
            nn.Conv2D(mid_channels, out_channels, kernel_size=3, padding=1, bias_attr=False),
            # 组归一化
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        """
        前向传播。
        :param x: 输入张量
        :return: 输出张量
        """
        if self.residual: # 如果使用残差连接
            # 将输入 x 与双卷积块的输出相加，然后应用 GELU 激活函数
            return F.gelu(x + self.double_conv(x))
        else: # 如果不使用残差连接
            # 直接返回双卷积块的输出
            return self.double_conv(x)

# UNet 下采样块
class Down(nn.Layer):
    """
    UNet 中的下采样（编码器）部分。
    包含一个最大池化层和两个双卷积块。
    同时融合了时间步长嵌入信息。
    """
    def __init__(self, in_channels, out_channels, emb_dim=256):
        """
        初始化下采样块。
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param emb_dim: 时间步长嵌入向量的维度
        """
        super().__init__()
        # 定义下采样和卷积序列
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2D(2), # 最大池化层，将特征图尺寸减半
            DoubleConv(in_channels, in_channels, residual=True), # 第一个双卷积块，带残差连接
            DoubleConv(in_channels, out_channels),             # 第二个双卷积块，改变通道数
        )

        # 定义用于处理时间步长嵌入的层
        self.emb_layer = nn.Sequential(
            nn.Silu(), # SiLU (Swish) 激活函数
            nn.Linear(emb_dim, out_channels), # 线性层，将时间嵌入维度映射到输出通道数
        )

    def forward(self, x, t):
        """
        前向传播。
        :param x: 输入特征图张量
        :param t: 时间步长嵌入张量
        :return: 下采样后的特征图张量
        """
        # 1. 通过最大池化和卷积块进行下采样和特征提取
        x = self.maxpool_conv(x)
        # 2. 处理时间嵌入：
        #    - 通过线性层和激活函数
        #    - 增加维度 (B, C) -> (B, C, 1, 1)
        #    - 使用 tile 复制嵌入，使其空间尺寸与 x 匹配 (B, C, H, W)
        emb = self.emb_layer(t)[:, :, None, None].tile([1, 1, x.shape[-2], x.shape[-1]])
        # 3. 将处理后的时间嵌入加到特征图上（特征融合）
        return x + emb

# UNet 上采样块
class Up(nn.Layer):
    """
    UNet 中的上采样（解码器）部分。
    包含一个上采样层、与编码器对应层特征图的拼接、以及两个双卷积块。
    同样融合了时间步长嵌入信息。
    """
    def __init__(self, in_channels, out_channels, emb_dim=256):
        """
        初始化上采样块。
        :param in_channels: 输入通道数 (来自上一层解码器和对应编码器的拼接)
        :param out_channels: 输出通道数
        :param emb_dim: 时间步长嵌入向量的维度
        """
        super().__init__()

        # 上采样层：使用双线性插值将特征图尺寸放大两倍
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # 定义卷积序列
        self.conv = nn.Sequential(
            # 第一个双卷积块，带残差连接，通道数不变
            DoubleConv(in_channels, in_channels, residual=True),
            # 第二个双卷积块，改变通道数，中间通道数设为 in_channels // 2
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        # 定义用于处理时间步长嵌入的层 (与 Down 块类似)
        self.emb_layer = nn.Sequential(
            nn.Silu(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, skip_x, t):
        """
        前向传播。
        :param x: 来自上一层解码器的输入特征图
        :param skip_x: 来自对应编码器层的特征图 (跳跃连接)
        :param t: 时间步长嵌入张量
        :return: 上采样后的特征图张量
        """
        # 1. 对输入 x 进行上采样，放大尺寸
        x = self.up(x)
        # 2. 拼接 (Concatenate)：将上采样后的 x 与来自编码器的 skip_x 在通道维度 (axis=1) 拼接起来
        #    这是 UNet 的核心特征，允许解码器利用编码器的低层特征
        x = paddle.concat([skip_x, x], axis=1)
        # 3. 通过卷积块进行特征提取和通道数调整
        x = self.conv(x)
        # 4. 处理时间嵌入并加到特征图上 (与 Down 块类似)
        emb = self.emb_layer(t)[:, :, None, None].tile([1, 1, x.shape[-2], x.shape[-1]])
        return x + emb

# UNet 模型主体
class UNet(nn.Layer):
    """
    UNet 模型架构。常用于图像分割和生成任务（如扩散模型）。
    包含编码器（下采样）、瓶颈和解码器（上采样）部分，并带有跳跃连接。
    """
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda"):
        """
        初始化 UNet 模型。
        :param c_in: 输入图像的通道数 (例如 RGB 为 3)
        :param c_out: 输出图像的通道数 (例如 RGB 为 3)
        :param time_dim: 时间步长嵌入的维度
        :param device: 指定模型运行的设备 ('cuda' 或 'cpu')，在此代码中未直接使用 paddle 参数
        """
        super().__init__()
        self.device = device # 保存设备信息 (虽然 PaddlePaddle 通常自动处理)
        self.time_dim = time_dim # 保存时间嵌入维度

        # --- 编码器部分 ---
        self.inc = DoubleConv(c_in, 64)      # 初始卷积块
        self.down1 = Down(64, 128)           # 第一个下采样块
        self.sa1 = SelfAttention(128)        # 第一个自注意力块
        self.down2 = Down(128, 256)          # 第二个下采样块
        self.sa2 = SelfAttention(256)        # 第二个自注意力块
        self.down3 = Down(256, 256)          # 第三个下采样块
        self.sa3 = SelfAttention(256)        # 第三个自注意力块

        # --- 瓶颈部分 ---
        self.bot1 = DoubleConv(256, 512)     # 瓶颈部分的第一个双卷积块
        self.bot2 = DoubleConv(512, 512)     # 瓶颈部分的第二个双卷积块
        self.bot3 = DoubleConv(512, 256)     # 瓶颈部分的第三个双卷积块 (通道数减少)

        # --- 解码器部分 ---
        self.up1 = Up(512, 128)              # 第一个上采样块 (输入通道数为 256(来自bot3)+256(来自down3的skip)=512)
        self.sa4 = SelfAttention(128)        # 第四个自注意力块
        self.up2 = Up(256, 64)               # 第二个上采样块 (输入通道数为 128(来自up1)+128(来自down2的skip)=256)
        self.sa5 = SelfAttention(64)         # 第五个自注意力块
        self.up3 = Up(128, 64)               # 第三个上采样块 (输入通道数为 64(来自up2)+64(来自down1的skip)=128)
        self.sa6 = SelfAttention(64)         # 第六个自注意力块

        # --- 输出层 ---
        self.outc = nn.Conv2D(64, c_out, kernel_size=1) # 最后的 1x1 卷积层，将通道数调整为输出通道数

    def pos_encoding(self, t, channels):
        """
        计算时间步长 t 的位置编码 (Positional Encoding)。
        使用 sin 和 cos 函数将标量时间步长映射到一个高维向量。
        这是 Transformer 中常用的技术，用于向模型提供序列（或时间）信息。
        :param t: 时间步长张量，形状为 (batch_size, 1)
        :param channels: 位置编码的维度 (必须是偶数)
        :return: 位置编码张量，形状为 (batch_size, channels)
        """
        # 计算频率倒数，频率随维度增加而指数减小
        inv_freq = 1.0 / (
            10000
            ** (paddle.arange(0, channels, 2).astype(paddle.float32) / channels)
        )
        # 计算 sin 部分 (应用于偶数维度)
        pos_enc_a = paddle.sin(t.tile([1, channels // 2]) * inv_freq)
        # 计算 cos 部分 (应用于奇数维度)
        pos_enc_b = paddle.cos(t.tile([1, channels // 2]) * inv_freq)
        # 将 sin 和 cos 部分拼接起来
        pos_enc = paddle.concat([pos_enc_a, pos_enc_b], axis=-1)
        return pos_enc

    def unet_forward(self, x, t):
        """
        UNet 的核心前向传播逻辑（不包括时间编码）。
        :param x: 输入图像张量
        :param t: 经过位置编码和处理后的时间嵌入张量
        :return: UNet 的输出张量
        """
        # --- 编码器 ---
        x1 = self.inc(x)    # (B, 64, H, W)
        x2 = self.down1(x1, t) # (B, 128, H/2, W/2)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t) # (B, 256, H/4, W/4)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t) # (B, 256, H/8, W/8)
        x4 = self.sa3(x4)

        # --- 瓶颈 ---
        x4 = self.bot1(x4) # (B, 512, H/8, W/8)
        x4 = self.bot2(x4) # (B, 512, H/8, W/8)
        x4 = self.bot3(x4) # (B, 256, H/8, W/8)

        # --- 解码器 ---
        # 上采样，拼接来自 x3 的跳跃连接
        x = self.up1(x4, x3, t) # (B, 128, H/4, W/4)
        x = self.sa4(x)
        # 上采样，拼接来自 x2 的跳跃连接
        x = self.up2(x, x2, t) # (B, 64, H/2, W/2)
        x = self.sa5(x)
        # 上采样，拼接来自 x1 的跳跃连接
        x = self.up3(x, x1, t) # (B, 64, H, W)
        x = self.sa6(x)
        # 输出层
        output = self.outc(x) # (B, c_out, H, W)
        return output

    def forward(self, x, t):
        """
        UNet 模型的完整前向传播。
        :param x: 输入图像张量 (B, c_in, H, W)
        :param t: 时间步长张量 (B,)，包含每个样本的时间步长标量
        :return: UNet 的输出张量 (B, c_out, H, W)
        """
        # 1. 将时间步长 t 增加一个维度并转换为 float32 类型 (B,) -> (B, 1)
        t = t.unsqueeze(-1).astype(paddle.float32)
        # 2. 计算时间步长的位置编码
        t = self.pos_encoding(t, self.time_dim) # (B, time_dim)
        # 3. 调用核心 unet_forward 方法，传入图像和处理后的时间嵌入
        return self.unet_forward(x, t)

# 条件 UNet 模型
class UNet_conditional(UNet):
    """
    带条件的 UNet 模型。在标准 UNet 的基础上增加了类别条件。
    允许模型根据给定的类别标签生成不同的输出。
    """
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, device="cuda"):
        """
        初始化条件 UNet 模型。
        :param c_in: 输入通道数
        :param c_out: 输出通道数
        :param time_dim: 时间和类别嵌入的维度
        :param num_classes: 数据集中的类别总数。如果为 None，则不使用类别条件。
        :param device: 设备信息
        """
        # 调用父类 UNet 的初始化方法
        super().__init__(c_in=c_in, c_out=c_out, time_dim=time_dim, device=device)

        # 如果指定了类别数，则创建类别嵌入层
        if num_classes is not None:
            # nn.Embedding 用于将离散的类别索引映射到稠密的嵌入向量
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def forward(self, x, t, y):
        """
        条件 UNet 的前向传播。
        :param x: 输入图像张量 (B, c_in, H, W)
        :param t: 时间步长张量 (B,)
        :param y: 类别标签张量 (B,)。如果模型是无条件的，可以传入 None。
        :return: UNet 的输出张量 (B, c_out, H, W)
        """
        # 1. 处理时间步长 t (与父类 UNet 相同)
        t = t.unsqueeze(-1).astype(paddle.float32)
        t = self.pos_encoding(t, self.time_dim) # (B, time_dim)

        # 2. 处理类别标签 y (如果提供了)
        if y is not None:
            # 使用类别嵌入层将类别索引 y 转换为嵌入向量 (B, time_dim)
            # 并将其加到时间嵌入 t 上，实现条件注入
            t += self.label_emb(y)

        # 3. 调用核心 unet_forward 方法，传入图像和融合了类别信息的时间嵌入
        return self.unet_forward(x, t)