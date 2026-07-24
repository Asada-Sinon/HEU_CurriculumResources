# Lite-RMSteg: Efficient Robust Message Steganography via Lightweight Attention Flow

**《人工智能原理》大作业**

*   **学院**: 智能科学与工程学院
*   **专业**: 人工智能专业
*   **学生姓名**: 孔维彬
*   **学号**: 2023040620
*   **任课教师**: 张中平、董福王

---

## 1. 项目介绍 

本项目基于 CVPR 2025 的 **RMSteg** 模型进行复现与轻量化改进，提出了 **Lite-RMSteg**。
原模型虽然在隐写鲁棒性上表现出色，但基于 Transformer 的架构导致参数量大且推理速度较慢。本项目旨在通过数学理论分析，对 Attention 机制和 Flow 模型进行剪枝与优化，在保持高质量隐写效果的同时，显著提升推理速度并降低显存占用，使其更适合在边缘设备部署。

### 主要改进点 :
1.  **维度剪枝 (Dimension Pruning)**: 将 Hidden Dimension 从 768 压缩至 714，减少冗余特征。
2.  **深度压缩 (Shallow Flow Depth)**: 将 Flow Blocks 从 4 层减为 2 层，ViT 提取器深度从 2 层减为 1 层。
3.  **结构简化 (Simplified Architecture)**: 优化 QR Transition 模块，将卷积流从 3 层减为 1 层。
4.  **工程优化 (Training Efficiency)**: 引入 AMP (Automatic Mixed Precision) 混合精度训练与梯度累积技术。

---

## 2. 环境搭建

本项目运行环境基于 Python 3.9 和 PyTorch 1.13。请按照以下步骤配置 Conda 环境。

### 2.1 创建虚拟环境
```bash
conda create -n steg_project python=3.9
conda activate steg_project
```

### 2.2 安装 PyTorch
根据你的 CUDA 版本选择合适的安装命令（本项目在 CUDA 11.8 下测试通过）：

**CUDA 11.8 (推荐)**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CPU 版本（仅用于测试）**
```bash
pip install torch torchvision torchaudio
```

> 其他 CUDA 版本请访问 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选择对应安装命令

### 2.3 一键安装所有依赖（推荐）
项目提供了 `requirements.txt` 文件，可一键安装所有必需的依赖包：

```bash
pip install -r requirements.txt
```

**手动安装依赖**（如果不使用 requirements.txt）
```bash
pip install lpips tqdm tensorboard opencv-python matplotlib scipy qrcode[pil] easydict pyyaml einops pyzbar pillow numpy
```

**加速下载提示**  
如果下载速度较慢，可使用清华镜像源：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 3. 数据集准备
本项目使用 COCO 2017 数据集作为 Cover Image。

下载 COCO 2017 Train/Val 图像：COCO Dataset

解压并将图片放入 datasets 文件夹（或在配置文件中修改路径）。

文件结构示例：

```
Project_Root/
├── src/
├── datasets/
│   └── train2017/
│       ├── 000000000009.jpg
│       └── ...
├── pretrained/
└── README.md
```

---

## 4. 快速开始 

### 4.1 运行推理/测试 
本项目提供了预训练权重（位于 src/pretrained/），可直接运行测试脚本生成隐写图像并查看 QR 码解码结果。

```bash
cd src
python single_test.py
```

输出结果: 运行完成后，请查看 `result/` 文件夹，其中包含：

- `org_img`: 原始图像
- `steg_img`: 隐写后的图像
- `decoded_qr`: 解码出的 QR 码
- `residual`: 隐写残差图

然后我自己训练的轻量化模型也在 `src/pretrained`，如果想用我训练好的轻量化模型进行测试，可以在 `single_test.py` 里把加载模型的路径改成 `trained2023040620.pth`。这样可以看到我训练好的轻量化模型的效果。
### 4.2 训练 Lite-RMSteg 模型 
如果要复现本项目的轻量化训练结果，请先前往Coco官网下载Coco2017数据集并解压至 `datasets/` 目录，(我上交的这个文件夹里面没有这个数据集，因为这个数据集有18G大小，智慧树上传不上去)然后记得要改一下config_lite.yaml中的数据集路径，确保指向正确的本地路径，然后下面的训练参数记得根据本地电脑情况改一下，不然可能会因为CPU显存不足而死机，运行优化后的训练脚本：

```bash
cd src
python train_lite.py
```

关键参数说明 (`src/config.yaml` 或代码内配置):

- `hidden_dim`: 714 (原版为 768)
- `flow_blocks`: 2 (原版为 4)
- `vit_depth`: 1 (原版为 2)
- `use_mixed_precision`: True (开启 AMP 加速)

可以使用 TensorBoard 监控训练过程：

```bash
tensorboard --logdir logs/
```

---

## 5. 实验结果 
我们在 NVIDIA T4 服务器上进行了对比实验，Lite-RMSteg 在保持高视觉质量的同时，显著提升了推理速度。

### 5.1 性能对比 

| Model | Hidden Dim | Blocks | Params (M) | FPS (Inf) | Speedup |
|-------|------------|--------|------------|-----------|----------|
| RMSteg (Baseline) | 768 | 4 | ~86.4 | 59.90 | 1.0x |
| Lite-RMSteg (Ours) | 714 | 2 | ~81.5 | 65.58 | 1.2x |
### 5.2 质量与鲁棒性 

| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ | TRA (Clean) | TRA (Noise) |
|-------|--------|--------|---------|-------------|-------------|
| RMSteg | 32.88 | 0.911 | 0.070 | 100% | 99.5% |
| Lite-RMSteg | 30.12 | 0.890 | 0.081 | 99.8% | 96.2% |

**TRA (Token Recovery Accuracy)**: QR 码 token 恢复准确率。

即便在进行高斯噪声攻击 (σ=0.01) 和 JPEG 压缩 (Q=80) 下，Lite 模型依然保持了 >92% 的解码准确率。

---

## 6. 文件结构说明 

```
├── src/
│   ├── model/
│   │   ├── attnflow_lite.py   # [改进] 重构后的轻量化 Flow & Attention 定义
│   │   ├── common.py          # 基础组件
│   │   └── ...
│   ├── util/                  # 工具类
│   ├── config.yaml            # 配置文件
│   ├── train_lite.py          # [改进] 包含 AMP 和梯度累积的训练脚本
│   ├── single_test.py         # 推理测试脚本
│   └── ...
├── result/                    # 输出结果目录
├── requirements.txt           # 依赖列表
└── README.md                  # 项目说明文档
```

---

## 7. 致谢与引用 

本项目基于 CVPR 2025 RMSteg 论文复现，部分代码参考了原作者实现。

- **Paper**: Robust Message Embedding via Attention Flow-Based Steganography
- **Repo**: https://github.com/huayuan4396/RMSteg

---

*本项目仅用于课程作业，请勿用于其他用途。*