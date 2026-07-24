import paddlehub as hub
import os
from PIL import Image
import io
import numpy as np
from tools import my_logger # 从 tools.py 导入日志记录器
import time

# 确保输出目录存在
output_dir = "stable_diffusion_output"
os.makedirs(output_dir, exist_ok=True)

# 加载模型 (PaddleHub 版本)
my_logger.info("加载StableDiffusion模型 (PaddleHub)...")
# 使用 PaddleHub 加载预训练的 stable_diffusion 模型
# 注意：这可能与 Hugging Face 的 diffusers 库版本不同
model = hub.Module(name="stable_diffusion")

# 类别名称和对应的英文提示词 (Prompts)
categories = ["建筑物", "森林", "冰川", "山峰", "大海", "街道"]
prompts = [
    "a photo of buildings, architecture, city buildings", # 建筑物的提示词
    "a photo of forest, trees, woods, dense forest",    # 森林的提示词
    "a photo of glacier, ice mountains, frozen landscape", # 冰川的提示词
    "a photo of mountains, peaks, rocky mountains",     # 山峰的提示词
    "a photo of sea, ocean, water, beach, seascape",    # 大海的提示词
    "a photo of street, road, urban, city street"       # 街道的提示词
]

# 简化图像提取函数 - 聚焦已知属性 (针对 PaddleHub 模型输出)
def extract_and_save_images_focused():
    """
    尝试从 PaddleHub Stable Diffusion 模型的输出中提取并保存图像。
    该模型可能返回 PIL 图像或 DocArray 对象，此函数尝试处理这两种情况。
    """
    # 遍历每个类别和对应的提示词
    for i, (category, prompt) in enumerate(zip(categories, prompts)):
        my_logger.info(f"使用 PaddleHub 生成 {category} 类别的图像...")

        try:
            # 调用 PaddleHub 模型生成图像
            result = model.generate_image(prompt)

            # 打印模型返回结果的类型，用于调试
            my_logger.info(f"结果类型: {type(result)}")

            # 情况 1: 如果结果直接是 PIL 图像对象
            if isinstance(result, Image.Image):
                # 构造保存路径
                save_path = os.path.join(output_dir, f"{category}_direct.png")
                # 保存图像
                result.save(save_path)
                my_logger.info(f"直接保存PIL图像: {save_path}")
                continue # 处理下一个类别

            # 情况 2: 如果结果是 DocArray 对象 (一种数据结构)
            # 使用字符串检查类型，避免直接导入 DocArray 可能引发的问题
            if str(type(result)).find('DocumentArray') >= 0:
                my_logger.info(f"尝试访问DocArray内部数据, 长度: {len(result)}")

                # 遍历 DocArray 中的每个 Document 对象
                for j, doc in enumerate(result):
                    my_logger.info(f"处理Document {j}")
                    image_saved = False # 标记当前 Document 是否已成功保存图像

                    # 尝试 1: 访问 'tensor' 属性
                    try:
                        # 检查是否存在 'tensor' 属性且不为空
                        if hasattr(doc, 'tensor') and doc.tensor is not None:
                            data = doc.tensor # 获取 tensor 数据
                            # 检查是否为 NumPy 数组且形状合理 (3维)
                            if isinstance(data, np.ndarray) and len(data.shape) == 3:
                                my_logger.info(f"找到 tensor 数据: shape={data.shape}, dtype={data.dtype}")
                                # 如果是 CHW 格式，转换为 HWC 格式
                                if data.shape[0] == 3: data = data.transpose(1, 2, 0)
                                # 如果值范围是 [0, 1]，则缩放到 [0, 255]
                                if data.max() <= 1.0: data = (data * 255).astype(np.uint8)
                                else: data = data.astype(np.uint8) # 否则直接转为 uint8

                                # 从 NumPy 数组创建 PIL 图像
                                img = Image.fromarray(data)
                                # 构造保存路径
                                save_path = os.path.join(output_dir, f"{category}_tensor_{j}.png")
                                # 保存图像
                                img.save(save_path)
                                my_logger.info(f"从 tensor 保存图像: {save_path}")
                                image_saved = True # 标记成功
                                continue # 处理下一个 Document
                    except Exception as e:
                        my_logger.error(f"处理 tensor 失败: {e}")

                    # 尝试 2: 访问 'blob' 属性 (如果 tensor 失败)
                    if not image_saved:
                        try:
                            # 检查是否存在 'blob' 属性且不为空
                            if hasattr(doc, 'blob') and doc.blob is not None:
                                data = doc.blob # 获取 blob 数据 (通常是字节流)
                                # 检查是否为字节串且长度大于某个阈值 (简单判断是否为有效图像数据)
                                if isinstance(data, bytes) and len(data) > 1000:
                                    my_logger.info(f"找到 blob 数据，长度: {len(data)}")
                                    # 使用 BytesIO 将字节流包装成文件对象，然后用 PIL 打开
                                    img = Image.open(io.BytesIO(data))
                                    # 构造保存路径
                                    save_path = os.path.join(output_dir, f"{category}_blob_{j}.png")
                                    # 保存图像
                                    img.save(save_path)
                                    my_logger.info(f"从 blob 保存图像: {save_path}")
                                    image_saved = True # 标记成功
                                    continue # 处理下一个 Document
                        except Exception as e:
                            my_logger.error(f"处理 blob 失败: {e}")

                    # 尝试 3: 访问 'matches' 属性 (如果以上都失败)
                    # 'matches' 可能包含与原始文档相关的其他文档，其中也可能包含图像
                    if not image_saved:
                        try:
                            # 检查是否存在 'matches' 属性且不为空
                            if hasattr(doc, 'matches') and doc.matches is not None:
                                matches = doc.matches # 获取 matches 列表
                                if len(matches) > 0:
                                    my_logger.info(f"找到 matches 数据，长度: {len(matches)}")
                                    # 遍历 matches 中的每个 match 文档
                                    for k, match in enumerate(matches):
                                        # 尝试从 match 文档中提取 tensor
                                        if hasattr(match, 'tensor') and match.tensor is not None:
                                            data = match.tensor
                                            if isinstance(data, np.ndarray) and len(data.shape) == 3:
                                                if data.shape[0] == 3: data = data.transpose(1, 2, 0) # CHW -> HWC
                                                if data.max() <= 1.0: data = (data * 255).astype(np.uint8)
                                                else: data = data.astype(np.uint8)

                                                img = Image.fromarray(data)
                                                save_path = os.path.join(output_dir, f"{category}_match_{k}_{j}.png")
                                                img.save(save_path)
                                                my_logger.info(f"从 match {k} 保存图像: {save_path}")
                                                image_saved = True # 标记已保存 (即使只保存了一个 match)
                                                # 注意：这里不 continue，因为一个 doc 可能有多个有效的 matches 图像
                        except Exception as e:
                            my_logger.error(f"处理 matches 失败: {e}")

                    # 尝试 4: 访问 'uri' 属性 (如果以上都失败)
                    # 'uri' 可能包含指向本地图像文件的路径
                    if not image_saved:
                        try:
                            # 检查是否存在 'uri' 属性，且不为空，是字符串，并且该路径确实存在
                            if hasattr(doc, 'uri') and doc.uri and isinstance(doc.uri, str) and os.path.exists(doc.uri):
                                my_logger.info(f"找到 uri 数据: {doc.uri}")
                                # 直接用 PIL 打开 uri 指向的文件
                                img = Image.open(doc.uri)
                                # 构造保存路径
                                save_path = os.path.join(output_dir, f"{category}_uri_{j}.png")
                                # 保存图像
                                img.save(save_path)
                                my_logger.info(f"从 uri 保存图像: {save_path}")
                                image_saved = True # 标记成功
                                continue # 处理下一个 Document
                        except Exception as e:
                            my_logger.error(f"处理 uri 失败: {e}")

                    # 如果所有尝试都失败了
                    if not image_saved:
                         my_logger.warning(f"无法从Document {j} 的已知属性中提取图像")

            # 情况 3: 未知的返回类型
            else:
                my_logger.error(f"未知的结果类型，无法处理: {type(result)}")

        # 捕获处理单个类别时可能发生的任何顶层错误
        except Exception as e:
            my_logger.error(f"处理 {category} 时发生顶层错误: {str(e)}")
            # 打印详细的错误堆栈信息，帮助调试
            import traceback
            my_logger.error(traceback.format_exc())


# 强烈推荐：使用 Hugging Face 的 diffusers 库
def use_diffusers():
    """
    使用 Hugging Face 的 diffusers 库生成图像。
    这是一个更标准、更常用的 Stable Diffusion 实现方式。
    """
    try:
        # 动态检查是否安装了必要的库 (diffusers 和 torch)
        import importlib
        if importlib.util.find_spec("diffusers") is None or importlib.util.find_spec("torch") is None:
            my_logger.error("未安装 diffusers 或 torch 库，跳过此方法")
            my_logger.info("请安装: pip install diffusers transformers torch")
            return # 如果库未安装，则退出函数

        # 导入必要的库
        import torch
        from diffusers import StableDiffusionPipeline

        my_logger.info("使用 Hugging Face 的 diffusers 库生成图像...")

        # 创建用于存放 diffusers 生成图像的子目录
        diffusers_dir = os.path.join(output_dir, "diffusers")
        os.makedirs(diffusers_dir, exist_ok=True)

        # 加载 Stable Diffusion Pipeline 模型
        my_logger.info("正在加载 diffusers 模型 (首次运行可能需要下载)...")
        # 从 Hugging Face Hub 加载预训练的 "runwayml/stable-diffusion-v1-5" 模型
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            # 如果 CUDA 可用，使用 float16 以节省显存和加速；否则使用 float32
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        my_logger.info("diffusers 模型加载完成.")

        # 如果 CUDA (GPU) 可用，将模型移动到 GPU 上运行
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")

        # 遍历每个类别和对应的提示词
        for category, prompt in zip(categories, prompts):
            my_logger.info(f"使用 diffusers 生成 {category} 图像...")

            # 生成图像
            # 使用 torch.no_grad() 上下文管理器，在推理时禁用梯度计算，节省显存
            with torch.no_grad():
                # 调用 pipeline 生成图像，结果包含一个图像列表，取第一个图像
                image = pipe(prompt).images[0]

            # 构造保存路径
            save_path = os.path.join(diffusers_dir, f"{category}.png")
            # 保存生成的 PIL 图像
            image.save(save_path)
            my_logger.info(f"使用 diffusers 保存图像: {save_path}")

    # 捕获使用 diffusers 过程中可能发生的任何错误
    except Exception as e:
        my_logger.error(f"使用 diffusers 失败: {e}")
        # 打印详细的错误堆栈信息
        import traceback
        my_logger.error(traceback.format_exc())

# --- 执行脚本的主要逻辑 ---

# 1. 尝试使用 PaddleHub 模型和聚焦已知属性的方法 (默认注释掉)
# my_logger.info("\n--- 正在尝试 PaddleHub 模型和聚焦提取方法 ---")
# extract_and_save_images_focused()

# 2. 强烈建议：如果上述方法失败或效果不佳，请使用 diffusers 库
my_logger.info("\n--- 正在尝试使用 diffusers 库 ---")
use_diffusers()  # 调用 use_diffusers 函数来生成图像

# 脚本执行完毕
my_logger.info("处理完成!")