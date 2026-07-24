"""
RMSteg 模型评估脚本
==================
用于对比多个预训练模型的性能指标，包括：
- 推理速度 (FPS, 平均时延)
- 模型大小 (文件体积, 参数量)
- 图像质量 (SSIM, LPIPS)

支持快速模式（--quick），在 2-3 分钟内完成评测，适合 RTX 4060 等中端 GPU。

作者: 改进自 RMSteg 项目
日期: 2025-11-27
"""

import os
import time
import glob
import argparse
import random
import datetime
from typing import List, Tuple, Dict

import torch
from torchvision import transforms
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from util import util
import util.qr as uqr
from model.attnflow import AttnFlow
from model.pytorch_ssim import ssim as ssim_fn

try:
    import lpips  # pip install lpips
except Exception as e:
    lpips = None


def find_images(img_root: str) -> List[str]:
    """
    递归查找指定目录下的所有图像文件
    
    Args:
        img_root: 图像根目录路径或 glob 模式
        
    Returns:
        去重后的图像文件路径列表
        
    支持格式: jpg, jpeg, png, bmp, webp
    """
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    paths = []
    if os.path.isdir(img_root):
        for ext in exts:
            paths.extend(glob.glob(os.path.join(img_root, ext)))
            paths.extend(glob.glob(os.path.join(img_root, "**", ext), recursive=True))
    else:
        # treat as glob pattern
        paths = glob.glob(img_root)
    # keep unique while preserving order
    seen = set()
    uniq = []
    for p in paths:
        if p.lower() not in seen:
            uniq.append(p)
            seen.add(p.lower())
    return uniq


def load_model(weights_path: str, device: torch.device, num_module: int = 37) -> torch.nn.Module:
    """
    加载预训练的 RMSteg 模型
    
    Args:
        weights_path: 权重文件路径 (.pth)
        device: 目标设备 (cpu/cuda)
        num_module: QR 码模块数量 (Version 5 = 37x37)
        
    Returns:
        加载完成并设置为评估模式的模型
    """
    net = AttnFlow(block_num=4, use_itf=True, use_qr_trans=True, num_module=num_module)
    state = torch.load(weights_path, map_location=device)
    net.load_state_dict(state)
    net = net.to(device)
    net.eval()
    return net


@torch.inference_mode()
def evaluate_model(
    model_path: str,
    images: List[str],
    device: torch.device,
    img_size: int = 224,
    qr_version: int = 5,
    batch_size: int = 8,
    max_images: int = 300,
    subset: str = "random",
    warmup_batches: int = 1,
    lpips_ratio: float = 0.25,
    use_amp: bool = True,
    out_dir: str = "./result",
) -> Dict[str, float]:
    """
    评估单个模型的性能指标
    
    Args:
        model_path: 模型权重文件路径
        images: 待评测的图像路径列表
        device: 计算设备
        img_size: 图像统一尺寸 (默认 224)
        qr_version: QR 码版本 (默认 5)
        batch_size: 批处理大小 (默认 8)
        max_images: 最大评测图像数 (None=全部)
        subset: 采样方式 ("random" 或 "first")
        warmup_batches: 预热批次数 (不计入时间统计)
        lpips_ratio: LPIPS 计算比例 (0.25=仅计算 25% 批次，节省时间)
        use_amp: 是否启用混合精度加速
        out_dir: 输出目录
        
    Returns:
        包含以下指标的字典：
        - size_mb: 模型文件大小 (MB)
        - params_m: 参数量 (百万)
        - avg_time_ms: 单张图像平均编码时间 (毫秒)
        - fps: 每秒处理帧数
        - lpips: LPIPS 感知损失 (越低越好)
        - ssim: 结构相似度 (越高越好)
        - count: 实际评测图像数
        
    设计目标: 在中端 GPU 上 2-3 分钟完成评测
    优化策略:
    - 图像子采样 (max_images)
    - 批量推理 (batch_size)
    - 混合精度 (use_amp)
    - 部分 LPIPS 计算 (lpips_ratio)
    """

    # 获取模型文件大小 (MB)
    size_mb = os.path.getsize(model_path) / (1024 ** 2)

    # 加载模型
    net = load_model(model_path, device)

    # 统计参数量 (百万)
    params_m = sum(p.numel() for p in net.parameters()) / 1e6

    # 准备 LPIPS 评估器 (感知损失)
    calc_lpips = None
    if lpips is not None:
        try:
            calc_lpips = lpips.LPIPS(net="vgg").to(device)
            calc_lpips.eval()
        except Exception:
            calc_lpips = None

    resize_img = transforms.Resize((img_size, img_size))

    # 生成固定内容的 QR 码（保证可重复性）
    os.makedirs(out_dir, exist_ok=True)
    qr_png = os.path.join(out_dir, "_eval_qr.png")
    uqr.save_qr_code(qr_png, version=qr_version, size=(img_size, img_size), message="rmsteg-eval")
    qr_1 = util.image_to_tensor(qr_png).to(device)[:, :1, ...]  # shape [1,1,H,W]

    # 图像子采样：随机或顺序选取指定数量的图像
    all_images = images[:]
    if subset == "random":
        random.shuffle(all_images)
    sel_images = all_images[: max_images] if (max_images is not None and max_images > 0) else all_images

    # 初始化统计变量
    n_total = 0          # 总图像数
    elapsed = 0.0        # 总推理时间
    lpips_sum = 0.0      # LPIPS 累计值
    lpips_count = 0      # LPIPS 计算次数
    ssim_sum = 0.0       # SSIM 累计值

    use_cuda = device.type == "cuda"
    amp_enabled = use_cuda and use_amp

    # 批量迭代处理图像
    for b_start in range(0, len(sel_images), batch_size):
        b_paths = sel_images[b_start : b_start + batch_size]
        if len(b_paths) == 0:
            continue

        # 加载当前批次的图像
        hosts = []
        for p in b_paths:
            try:
                h = util.image_to_tensor(p)
                # 灰度图转 RGB
                if h.shape[1] == 1:
                    h = h.repeat(1, 3, 1, 1)
                h = resize_img(h)
                hosts.append(h)
            except Exception:
                # 跳过损坏的图像
                pass

        if len(hosts) == 0:
            continue

        # 组装批次数据
        host_b = torch.cat(hosts, dim=0).to(device)
        qr_b = qr_1.repeat(host_b.shape[0], 1, 1, 1)
        x_b = torch.cat([host_b, qr_b], dim=1)  # [B, 4, H, W]

        # 预热批次：执行但不计时（避免首次推理的初始化开销）
        if warmup_batches > 0:
            with torch.amp.autocast(device_type='cuda' if use_cuda else 'cpu', enabled=amp_enabled):
                _ = net.encode(x_b)
            warmup_batches -= 1
            continue

        # 计时的前向推理
        if use_cuda:
            torch.cuda.synchronize()  # 确保 GPU 操作完成
        t0 = time.time()
        with torch.amp.autocast(device_type='cuda' if use_cuda else 'cpu', enabled=amp_enabled):
            steg_b, _ = net.encode(x_b)
        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.time()
        elapsed += (t1 - t0)

        bs = host_b.shape[0]
        n_total += bs

        # 计算批次指标
        # SSIM: 结构相似度，批次平均后按样本数加权
        # 转为 float32 以兼容 SSIM 模块
        host_clamped = host_b.clamp(0.0, 1.0).float()
        steg_clamped = steg_b.clamp(0.0, 1.0).float()
        ssim_val = ssim_fn(host_clamped, steg_clamped).item()
        ssim_sum += ssim_val * bs

        # LPIPS: 感知损失，为节省时间仅对部分批次计算
        if calc_lpips is not None and random.random() < lpips_ratio:
            # 转为 float32 以提高 LPIPS 稳定性
            lp_vals = calc_lpips(host_b.float(), steg_b.float()).reshape(-1)
            lpips_sum += lp_vals.mean().item() * bs
            lpips_count += bs

    # 若无有效图像，返回空结果
    if n_total == 0:
        return {
            "size_mb": size_mb,
            "params_m": params_m,
            "avg_time_ms": float("nan"),
            "fps": float("nan"),
            "lpips": float("nan"),
            "ssim": float("nan"),
            "count": 0,
        }

    # 计算平均指标
    avg_time = elapsed / n_total if n_total > 0 else float("inf")
    lpips_avg = (lpips_sum / lpips_count) if (lpips_count > 0) else float("nan")
    return {
        "size_mb": size_mb,
        "params_m": params_m,
        "avg_time_ms": avg_time * 1000.0,
        "fps": (1.0 / avg_time) if avg_time > 0 else float("inf"),
        "lpips": lpips_avg,
        "ssim": ssim_sum / n_total,
        "count": n_total,
    }


def _ensure_out_dir(out_dir: str | None) -> str:
    """
    确保输出目录存在，若未指定则创建时间戳命名的新目录
    
    Args:
        out_dir: 指定的输出目录，None 则自动生成
        
    Returns:
        实际使用的输出目录路径
    """
    if out_dir is None or len(out_dir.strip()) == 0:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("./result", f"eval_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _save_plots(results: List[tuple], out_dir: str):
    """
    生成模型对比图表并保存
    
    Args:
        results: 模型评估结果列表 [(model_path, metrics_dict), ...]
        out_dir: 输出目录
        
    生成内容:
        - metrics_overview.png: 包含模型大小、FPS、SSIM、LPIPS 四个子图
    """
    if plt is None:
        print("matplotlib not installed; skipping plots.")
        return

    names = [os.path.basename(mp) for mp, _ in results]
    size_mb = [r["size_mb"] for _, r in results]
    fps = [r["fps"] for _, r in results]
    ssim_vals = [r["ssim"] for _, r in results]
    lpips_vals = [r["lpips"] if (r["lpips"] == r["lpips"]) else None for _, r in results]  # NaN 安全处理

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("RMSteg Model Evaluation", fontsize=14)

    def _bar(ax, vals, title, ylabel, invert_better=False):
        ax.bar(names, vals, color="#4C78A8")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(names, rotation=15, ha="right")
        for i, v in enumerate(vals):
            if v is None:
                txt = "nan"
                val = 0
            else:
                txt = f"{v:.2f}"
                val = v
            ax.text(i, val, txt, ha='center', va='bottom', fontsize=9)
        if invert_better:
            ax.invert_yaxis()

    _bar(axes[0, 0], size_mb, "Model Size", "MB")
    _bar(axes[0, 1], fps, "Throughput", "FPS")
    _bar(axes[1, 0], ssim_vals, "SSIM (higher better)", "SSIM")
    # handle None in LPIPS by mapping to 0 for bar height but label as nan
    lpips_vals_plot = [0 if v is None else v for v in lpips_vals]
    _bar(axes[1, 1], lpips_vals_plot, "LPIPS (lower better)", "LPIPS", invert_better=False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(out_dir, "metrics_overview.png"), dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate pretrained RMSteg models")
    parser.add_argument("--img_root", type=str, required=True, help="Image folder or glob, e.g. E:\\dataset_for_train")
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--models", type=str, nargs="*", default=None, help="Paths to .pth files; default scans ./pretrained/")
    # quick options
    parser.add_argument("--quick", action="store_true", help="Enable lightweight evaluation for ~2-3 minutes runtime")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for encode timing (default depends on quick)")
    parser.add_argument("--max_images", type=int, default=None, help="Max images to evaluate (default depends on quick)")
    parser.add_argument("--subset", type=str, choices=["random", "first"], default="random")
    parser.add_argument("--lpips_ratio", type=float, default=None, help="Fraction of batches to compute LPIPS on (0..1)")
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP mixed precision on CUDA")
    parser.add_argument("--out_dir", type=str, default=None, help="Directory to save all evaluation artifacts; default is a new timestamped folder under ./result/")

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    images = find_images(args.img_root)
    if len(images) == 0:
        print(f"No images found under: {args.img_root}")
        return

    model_paths: List[str]
    if args.models:
        model_paths = args.models
    else:
        model_paths = []
        for name in ["rmsteg.pth", "trained2023040620.pth"]:
            p = os.path.join("./pretrained", name)
            if os.path.exists(p):
                model_paths.append(p)

    if len(model_paths) == 0:
        print("No model weights provided or found in ./pretrained")
        return

    # defaults for quick mode tuned for ~2-3 minutes on RTX 4060
    quick = bool(args.quick)
    batch_size = args.batch_size if args.batch_size is not None else (8 if quick else 4)
    max_images = args.max_images if args.max_images is not None else (300 if quick else None)
    lpips_ratio = args.lpips_ratio if args.lpips_ratio is not None else (0.25 if quick else 1.0)
    use_amp = not args.no_amp
    out_dir = _ensure_out_dir(args.out_dir)

    print(
        f"Found {len(images)} images. Device: {device}. Quick={quick} | "
        f"batch_size={batch_size} | max_images={max_images} | lpips_ratio={lpips_ratio} | AMP={'on' if use_amp else 'off'} | "
        f"out_dir={os.path.abspath(out_dir)}"
    )

    results = []
    for mp in model_paths:
        r = evaluate_model(
            mp,
            images,
            device,
            img_size=args.img_size,
            batch_size=batch_size,
            max_images=max_images if max_images is not None else (len(images)),
            subset=args.subset,
            warmup_batches=1,
            lpips_ratio=lpips_ratio,
            use_amp=use_amp,
            out_dir=out_dir,
        )
        results.append((mp, r))

    # print summary and save csv
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "eval_summary.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("model,size_mb,params_m,avg_time_ms,fps,lpips,ssim,count\n")
        for mp, r in results:
            f.write(
                f"{os.path.basename(mp)},{r['size_mb']:.3f},{r['params_m']:.3f},{r['avg_time_ms']:.3f},{r['fps']:.3f},{r['lpips'] if r['lpips']==r['lpips'] else 'nan'},{r['ssim']:.4f},{r['count']}\n"
            )

    # plots
    _save_plots(results, out_dir)

    print("\nEvaluation Summary:")
    for mp, r in results:
        print(
            f"- {os.path.basename(mp)} | size: {r['size_mb']:.2f} MB | params: {r['params_m']:.2f} M | "
            f"avg: {r['avg_time_ms']:.2f} ms | FPS: {r['fps']:.2f} | SSIM: {r['ssim']:.4f} | LPIPS: {r['lpips'] if r['lpips']==r['lpips'] else 'nan'}"
        )
    print(f"Artifacts saved to: {os.path.abspath(out_dir)}")
    print(f"CSV saved to: {os.path.abspath(csv_path)}")


if __name__ == "__main__":
    main()
