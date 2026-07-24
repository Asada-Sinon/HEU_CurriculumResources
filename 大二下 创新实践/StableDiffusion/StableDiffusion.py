import paddle
from tqdm import tqdm
from tools import * # 假设 tools.py 包含一些辅助函数，例如保存图像等

# 定义扩散模型的核心逻辑类
class Diffusion:
    # 初始化方法，设置扩散过程的超参数
    def __init__(self, noise_steps=500, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        """
        初始化扩散模型参数。
        :param noise_steps: 总的扩散步数 (T)
        :param beta_start: 噪声调度表中 beta 的起始值
        :param beta_end: 噪声调度表中 beta 的结束值
        :param img_size: 图像的尺寸 (假设为方形)
        :param device: 计算设备 ('cuda' 或 'cpu')，在此代码中未直接使用 paddle 参数
        """
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # 1. 准备噪声调度表 (beta)
        # beta 是每一步添加噪声的方差，通常从一个较小值线性增加到一个较大值
        self.beta = self.prepare_noise_schedule() # (T,)

        # 2. 计算 alpha 和 alpha_hat
        # alpha = 1 - beta
        self.alpha = 1. - self.beta # (T,)
        # alpha_hat 是 alpha 的累积乘积 (α_t_hat = α_1 * α_2 * ... * α_t)
        # 它表示从原始图像 x_0 直接加噪到第 t 步 x_t 的缩放因子
        self.alpha_hat = paddle.cumprod(self.alpha, dim=0) # (T,)

        self.img_size = img_size
        self.device = device # 保存设备信息

    # 创建噪声调度表 (beta)
    def prepare_noise_schedule(self):
        """
        生成一个从 beta_start 到 beta_end 的线性序列，包含 noise_steps 个值。
        :return: beta 调度表张量
        """
        return paddle.linspace(self.beta_start, self.beta_end, self.noise_steps)

    # 前向过程：向图像添加噪声
    def noise_images(self, x, t):
        """
        根据给定的时间步 t，向一批图像 x 添加噪声。
        使用公式: x_t = sqrt(alpha_hat_t) * x_0 + sqrt(1 - alpha_hat_t) * ε
        其中 x_0 是原始图像，ε 是标准高斯噪声。
        :param x: 原始图像张量 (B, C, H, W)
        :param t: 时间步张量 (B,)，包含每个图像对应的时间步索引
        :return: 加噪后的图像 x_t 和添加的噪声 ε
        """
        # 获取对应时间步 t 的 sqrt(alpha_hat_t) 值
        # [:, None, None, None] 用于将 (B,) 形状扩展为 (B, 1, 1, 1) 以匹配图像张量形状进行广播
        sqrt_alpha_hat = paddle.sqrt(self.alpha_hat[t])[:, None, None, None]
        # 获取对应时间步 t 的 sqrt(1 - alpha_hat_t) 值
        sqrt_one_minus_alpha_hat = paddle.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        # 生成与 x 形状相同的高斯噪声 ε
        Ɛ = paddle.randn(shape=x.shape)
        # 计算加噪后的图像 x_t
        noisy_image = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ
        return noisy_image, Ɛ # 返回加噪图像和所加噪声

    # 随机采样时间步
    def sample_timesteps(self, n):
        """
        为一批大小为 n 的数据随机采样时间步。
        采样范围从 1 到 noise_steps - 1 (包含)。
        :param n: 批次大小
        :return: 包含 n 个随机时间步的张量 (n,)
        """
        # 从 [1, noise_steps) 区间内随机抽取 n 个整数
        return paddle.randint(low=1, high=self.noise_steps, shape=(n,))

    # 反向过程：从噪声生成图像 (采样)
    def sample(self, model, n, labels, cfg_scale=3):
        """
        使用训练好的模型从纯噪声生成 n 张图像。
        这是扩散模型的反向过程（去噪过程）。
        :param model: 训练好的 UNet 模型 (通常是条件 UNet)
        :param n: 要生成的图像数量
        :param labels: 对应的类别标签张量 (n,)，用于条件生成
        :param cfg_scale: Classifier-Free Guidance (CFG) 的尺度因子。
                          控制生成结果与条件的符合程度以及多样性。
                          cfg_scale=0 表示无条件生成。
        :return: 生成的图像张量，像素值范围 [0, 255]
        """
        # 将模型设置为评估模式 (不计算梯度，关闭 Dropout 等)
        model.eval()
        # 使用 paddle.no_grad() 上下文管理器，禁用梯度计算，节省内存和计算
        with paddle.no_grad():
            # 1. 初始化：从标准高斯分布生成纯噪声图像 x_T
            x = paddle.randn((n, 3, self.img_size, self.img_size))
            # 2. 迭代去噪：从 T-1 步循环到 1 步
            # tqdm 用于显示进度条
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0, desc=f"denoising, wait for {self.noise_steps} steps ", unit=' steps',ncols=80):
                # 当前时间步 t (从 T-1 递减到 1)
                # 创建一个包含 n 个当前时间步 i 的张量
                t = paddle.to_tensor([i] * x.shape[0]).astype("int64")

                # 3. 预测噪声：使用 UNet 模型预测当前图像 x 在时间步 t 下的噪声
                # a. 条件预测：使用给定的类别标签 labels
                predicted_noise = model(x, t, labels)

                # b. Classifier-Free Guidance (CFG)
                if cfg_scale > 0:
                    # 无条件预测：将标签设置为 None，预测无条件下的噪声
                    uncond_predicted_noise = model(x, t, None)
                    # CFG 公式：最终预测 = 无条件预测 + cfg_scale * (条件预测 - 无条件预测)
                    # 等价于线性插值：lerp(uncond, cond, weight) = uncond + weight * (cond - uncond)
                    # 这里 weight 就是 cfg_scale
                    cfg_scale_tensor = paddle.to_tensor(cfg_scale).astype("float32") # 确保类型匹配
                    # 使用 lerp 进行插值，得到结合了 CFG 的最终预测噪声
                    predicted_noise = paddle.lerp(uncond_predicted_noise, predicted_noise, cfg_scale_tensor)

                # 4. 计算去噪一步：根据预测的噪声 predicted_noise 计算 x_{t-1}
                # 获取当前时间步 t 对应的 alpha, alpha_hat, beta 值
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                # 采样噪声 z (用于引入随机性)
                if i > 1:
                    # 如果不是最后一步 (t > 1)，则采样高斯噪声
                    noise = paddle.randn(shape=x.shape)
                else:
                    # 如果是最后一步 (t = 1)，则不添加噪声 (z=0)
                    noise = paddle.zeros_like(x)

                # DDPM 采样公式（简化版）:
                # x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_hat_t)) * predicted_noise) + sqrt(beta_t) * z
                # 这里使用了稍微不同的形式，但效果类似：
                # x_{t-1} = (1/sqrt(alpha_t)) * (x_t - ((1 - alpha_t) / sqrt(1 - alpha_hat_t)) * predicted_noise) + sqrt(beta_t) * z
                # 计算去噪后的图像 x_{t-1}
                term1 = 1 / paddle.sqrt(alpha)
                term2 = (x - ((1 - alpha) / (paddle.sqrt(1 - alpha_hat))) * predicted_noise)
                term3 = paddle.sqrt(beta) * noise
                x = term1 * term2 + term3 # 更新 x 为 x_{t-1}

        # 将模型切换回训练模式
        model.train()
        # 后处理：将生成的图像 x_0 (值范围理论上接近 [-1, 1]) 转换回 [0, 255] 的像素值
        # 1. 裁剪到 [-1, 1] 范围
        x = (x.clip(-1, 1) + 1) / 2 # 映射到 [0, 1]
        # 2. 缩放到 [0, 255]
        x = (x * 255)
        # 返回最终生成的图像
        return x