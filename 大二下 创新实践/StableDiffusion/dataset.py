import paddle.vision as V  # 导入 PaddlePaddle 的视觉库，用于图像变换
from PIL import Image  # 导入 PIL (Pillow) 库，用于图像文件的读取和处理
from paddle.io import Dataset, DataLoader  # 从 PaddlePaddle 导入 Dataset 和 DataLoader 类，用于构建和加载数据集
from tqdm import tqdm  # 导入 tqdm 库，用于显示进度条（虽然在此代码段中未直接使用，但常用于数据加载循环）

# 数据变换：定义一系列图像预处理操作
# 这些操作会按顺序应用到每一张图片上
transforms = V.transforms.Compose([
    # 1. 调整图像大小：将图像的最短边调整到 80 像素，另一边按比例缩放
    V.transforms.Resize(80),
    # 2. 随机裁剪并调整大小：
    #    - 从调整大小后的图像中随机裁剪出一个区域。
    #    - 裁剪区域的大小是原图大小的 80% 到 100% (scale=(0.8, 1.0))。
    #    - 将裁剪出的区域调整为 64x64 像素。这有助于增加模型的鲁棒性（数据增强）。
    V.transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
    # 3. 转换为张量：将 PIL 图像对象（像素值范围 0-255）转换为 PaddlePaddle 张量。
    #    - 形状从 (H, W, C) 变为 (C, H, W)。
    #    - 像素值会被缩放到 0.0 到 1.0 之间。
    V.transforms.ToTensor(),
    # 4. 归一化：将张量中的像素值进行归一化。
    #    - 使用公式: output = (input - mean) / std
    #    - 这里 mean 和 std 都设为 (0.5, 0.5, 0.5)，意味着将像素值从 [0, 1] 范围转换到 [-1, 1] 范围。
    #    - (0.5, 0.5, 0.5) 分别对应 R, G, B 三个通道的均值和标准差。
    V.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 自定义数据集类，用于加载训练数据
# 需要继承 paddle.io.Dataset 类，并实现 __init__, __getitem__, __len__ 三个方法
class TrainData(Dataset):
    # 类的初始化方法，在创建数据集对象时调用
    def __init__(self, txt_path="data.txt"):
        """
        初始化数据集
        :param txt_path: 包含图像路径和标签的文本文件路径 (例如 "data.txt")
        """
        # 使用 'with open' 安全地打开指定的文本文件
        # 'r' 表示以只读模式打开
        with open(txt_path, "r") as f:
            # 读取文件中的所有行，每行作为一个字符串存入列表 data
            data = f.readlines()
        # 列表推导式：遍历 data 中的每一行 (line)
        # line.strip() 会移除行首和行尾的空白字符（包括换行符）
        # if line.strip() 判断处理后的行是否为空，如果不为空，则将其保留
        # 这样可以过滤掉原始文件中的空行
        self.image_paths = [line for line in data if line.strip()]

    # 获取数据集中单个样本的方法，DataLoader 会调用这个方法来获取数据
    def __getitem__(self, index):
        """
        根据索引获取数据
        :param index: 数据在列表中的索引号
        :return: 经过预处理的图像张量和对应的整数标签
        """
        # 1. 根据传入的索引，从 self.image_paths 列表中获取对应的行
        #    .strip() 再次确保移除可能存在的首尾空白
        line = self.image_paths[index].strip()
        # 2. 分割字符串：假设文件中的格式是 "图片路径 标签"，使用空格作为分隔符
        #    将分割后的两部分分别赋值给 image_path 和 label
        image_path, label = line.split(" ")

        # 3. 打开图像文件：使用 PIL 的 Image.open 读取指定路径的图像文件
        #    .convert("RGB") 确保图像是 RGB 格式（3个颜色通道），防止灰度图或带 alpha 通道的图导致后续处理出错
        image = Image.open(image_path).convert("RGB")
        # 4. 应用预处理：将之前定义的 transforms 应用到加载的图像上
        #    包括 Resize, RandomResizedCrop, ToTensor, Normalize
        image = transforms(image)

        # 5. 转换标签类型：将从文件中读取到的标签字符串转换为整数类型
        label = int(label)

        # 6. 返回处理好的图像张量和整数标签
        return image, label

    # 获取数据集总样本数的方法
    def __len__(self):
        """
        获取数据集的大小
        :return: 数据集中的样本数量 (即 self.image_paths 列表的长度)
        """
        return len(self.image_paths)

# --- 下面是使用这个自定义数据集类的示例 ---

# 创建数据集实例：使用默认的 "data.txt" 文件路径来初始化 TrainData 对象
dataset = TrainData()

# 创建数据加载器 (DataLoader)
# DataLoader 可以方便地实现数据的批量加载、打乱顺序、多进程加载等功能
dataloader = DataLoader(
    dataset,          # 指定要加载的数据集
    batch_size=64,    # 每个批次加载 64 个样本
    shuffle=True      # 在每个 epoch 开始时打乱数据顺序，有助于模型训练
)

# 验证步骤：打印一些信息来检查数据集和数据加载器是否正常工作
print(f"数据集样本数：{len(dataset)}") # 打印数据集的总样本数

# 遍历 DataLoader，获取批次数据
# enumerate 会同时提供索引 i 和对应的批次数据 (images, labels)
for i, (images, labels) in enumerate(dataloader):
    # 只处理前 2 个批次作为示例
    if i >= 2:
        break
    # 打印当前批次的索引（从1开始计数）
    # 打印图像张量的形状 (batch_size, channels, height, width)
    # 打印标签张量的形状 (batch_size,)
    print(f"批次 {i+1} - 图像形状：{images.shape}，标签形状：{labels.shape}")
    # 打印当前批次中前 5 个样本的标签值
    # .numpy() 将 PaddlePaddle 张量转换为 NumPy 数组，方便查看
    print(f"标签示例：{labels[:5].numpy()}（前 5 个标签）\n")
