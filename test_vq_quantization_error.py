"""
VQ 量化误差测试脚本

测试目的：
1. 评估 VQ Tokenizer 的量化误差（有损压缩）
2. 验证量化误差对时序预测的影响
3. 为使用 MSE Loss 而非交叉熵提供实验依据

运行命令：
cd /Users/tian/Desktop/VisionTS/VisionTSRAR
python test_vq_quantization_error.py
"""

import sys
import os
from pathlib import Path

# 设置路径
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / 'visiontsrar'))

import torch
import torch.nn.functional as F
import numpy as np
import einops

from visiontsrar import VisionTSRAR


def test_vq_quantization_error():
    """测试 VQ 量化误差"""
    print("=" * 70)
    print("VQ Tokenizer 量化误差测试")
    print("=" * 70)
    
    # ========== 1. 加载模型 ==========
    print("\n[1/7] 加载 VisionTSRAR 模型...")
    try:
        # 使用项目根目录的 ckpt 文件夹
        ckpt_dir = project_root / 'ckpt'
        vm = VisionTSRAR(
            arch='rar_l_0.3b',
            finetune_type='ln',
            ckpt_dir=str(ckpt_dir),
            load_ckpt=True,
        )
        vm.eval()
        print("  ✅ 模型加载成功")
        print(f"  权重目录：{ckpt_dir}")
    except Exception as e:
        print(f"  ❌ 模型加载失败：{e}")
        print(f"  请确保已下载 VQ 和 RAR 权重文件到 {ckpt_dir} 目录")
        print("  首次运行会自动从 HuggingFace 下载权重文件")
        return None
    
    # ========== 2. 准备测试数据 ==========
    print("\n[2/7] 准备测试数据...")
    batch_size = 4
    context_len = 96  # 回看窗口
    pred_len = 96     # 预测窗口
    nvars = 7         # 变量数量
    
    # 模拟 ETT 数据（归一化后的分布）
    torch.manual_seed(42)
    x_enc = torch.randn(batch_size, context_len, nvars) * 0.5 + 1.0
    print(f"  输入形状：{x_enc.shape}")
    print(f"  数据范围：[{x_enc.min():.3f}, {x_enc.max():.3f}]")
    
    # ========== 3. 初始化模型配置 ==========
    print("\n[3/7] 初始化模型配置...")
    periodicity = 24  # 日周期
    vm.update_config(
        context_len=context_len,
        pred_len=pred_len,
        periodicity=periodicity,
    )
    print(f"  周期性：{periodicity}")
    print(f"  输入 patch 数：{vm.num_patch_input}")
    print(f"  输出 patch 数：{vm.num_patch_output}")
    print(f"  可见 token 数：{vm.num_visible_tokens}")
    
    # ========== 4. 时序 → 图像（前 3 步） ==========
    print("\n[4/7] 转换时序为图像（VisionTSRAR forward 前 3 步）...")
    
    # 第 1 步：Normalization
    means = x_enc.mean(1, keepdim=True).detach()
    x_norm = x_enc - means
    stdev = torch.sqrt(
        torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5
    )
    stdev /= vm.norm_const
    x_norm = x_norm / stdev
    
    # Channel Independent: [b, s, n] -> [b, n, s]
    x_norm = einops.rearrange(x_norm, 'b s n -> b n s')
    
    # 第 2 步：Segmentation
    x_pad = F.pad(x_norm, (vm.pad_left, 0), mode='replicate')
    x_2d = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=vm.periodicity)
    
    # 第 3 步：Render & Alignment
    x_resize = vm.input_resize(x_2d)
    masked = torch.zeros(
        (x_2d.shape[0], 1, vm.image_size, vm.num_patch_output * vm.patch_size),
        device=x_2d.device, dtype=x_2d.dtype,
    )
    x_concat_with_masked = torch.cat([x_resize, masked], dim=-1)
    image_input = einops.repeat(x_concat_with_masked, 'b 1 h w -> b c h w', c=3)
    
    print(f"  图像形状：{image_input.shape}")
    print(f"  图像范围：[{image_input.min():.3f}, {image_input.max():.3f}]")
    
    # ========== 5. VQ Encode → Decode ==========
    print("\n[5/7] VQ 编码 → 解码...")
    
    with torch.no_grad():
        rar_wrapper = vm.rar_wrapper
        
        # 使用完整的 RARWrapper.forward 进行重建（包含值域对齐）
        # num_visible_tokens = context_len / patch_size = 48
        num_visible_tokens = vm.num_visible_tokens
        reconstructed_image, _ = rar_wrapper(image_input, num_visible_tokens)
    
    print(f"  重建图像形状：{reconstructed_image.shape}")
    print(f"  重建图像范围：[{reconstructed_image.min():.3f}, {reconstructed_image.max():.3f}]")
    
    # ========== 6. 计算图像空间误差 ==========
    print("\n[6/7] 计算重建误差...")
    
    image_mse = ((image_input - reconstructed_image) ** 2).mean().item()
    image_mae = torch.abs(image_input - reconstructed_image).mean().item()
    image_psnr = 10 * np.log10(1.0 ** 2 / max(image_mse, 1e-8)) / np.log(10)
    
    # 相对误差：使用 mean(image_input) 作为分母，避免除零问题
    image_input_mean = torch.abs(image_input).mean().item()
    image_rel = (torch.abs(image_input - reconstructed_image).mean().item() / 
                 (image_input_mean + 1e-8))
    
    print("\n  图像空间重建质量:")
    print(f"    MSE:      {image_mse:.6f}")
    print(f"    MAE:      {image_mae:.6f}")
    print(f"    PSNR:     {image_psnr:.2f} dB")
    print(f"    相对误差：{image_rel:.4f} ({image_rel*100:.2f}%)")
    
    # ========== 7. 计算时序空间误差 ==========
    print("\n[7/7] 转换回时序并计算误差（VisionTSRAR forward 后 2 步）...")
    
    # 第 5 步：Forecasting
    y_grey = torch.mean(reconstructed_image, 1, keepdim=True)
    y_segmentations = vm.output_resize(y_grey)
    y_flatten = einops.rearrange(
        y_segmentations,
        '(b n) 1 f p -> b (p f) n',
        b=batch_size, f=vm.periodicity,
    )
    
    # 提取预测窗口
    y_pred = y_flatten[
        :,
        vm.pad_left + vm.context_len: vm.pad_left + vm.context_len + vm.pred_len,
        :,
    ]
    
    # 第 6 步：Denormalization
    y_pred = y_pred * (stdev.repeat(1, vm.pred_len, 1))
    y_pred = y_pred + (means.repeat(1, vm.pred_len, 1))
    
    # 计算时序误差
    time_mse = ((x_enc - y_pred) ** 2).mean().item()
    time_mae = torch.abs(x_enc - y_pred).mean().item()
    time_psnr = 10 * np.log10(1.0 ** 2 / max(time_mse, 1e-8)) / np.log(10)
    time_rel = (torch.abs(x_enc - y_pred) / 
                (torch.abs(x_enc) + 1e-8)).mean().item()
    
    print("\n  时序空间重建质量:")
    print(f"    MSE:      {time_mse:.6f}")
    print(f"    MAE:      {time_mae:.6f}")
    print(f"    PSNR:     {time_psnr:.2f} dB")
    print(f"    相对误差：{time_rel:.4f} ({time_rel*100:.2f}%)")
    
    # ========== 评估结论 ==========
    print("\n" + "=" * 70)
    print("评估结论")
    print("=" * 70)
    
    if image_mse < 0.01:
        print("  ✅ 图像量化误差很小 (< 0.01)，可接受")
    elif image_mse < 0.05:
        print("  ⚠️  图像量化误差中等 (< 0.05)，需注意")
    else:
        print("  ❌ 图像量化误差较大 (> 0.05)，可能影响预测")
    
    if time_mae < 0.1:
        print("  ✅ 时序重建误差很小 (< 0.1)，对预测影响小")
    elif time_mae < 0.3:
        print("  ⚠️  时序重建误差中等 (< 0.3)，可能影响精度")
    else:
        print("  ❌ 时序重建误差较大 (> 0.3)，严重影响预测")
    
    print("\n" + "=" * 70)
    print("建议")
    print("=" * 70)
    
    if image_mse < 0.01 and time_mae < 0.1:
        print("  ✅ VQ 量化误差可接受，建议使用 MSE Loss 进行端到端优化")
        print("  ✅ 交叉熵（token 空间）不如 MSE（时序空间）直接有效")
    else:
        print("  ⚠️  VQ 量化误差较大，可考虑：")
        print("     1. 使用更大的 VQ 码本（如 32768）")
        print("     2. 混合 Loss（MSE + 小权重交叉熵）")
        print("     3. 使用连续表示替代离散 token")
    
    print("\n" + "=" * 70)
    
    # 保存结果
    results = {
        'image_mse': image_mse,
        'image_mae': image_mae,
        'image_psnr': image_psnr,
        'image_rel': image_rel,
        'time_mse': time_mse,
        'time_mae': time_mae,
        'time_psnr': time_psnr,
        'time_rel': time_rel,
    }
    
    output_path = project_root / 'vq_quantization_results.txt'
    with open(output_path, 'w') as f:
        f.write("VQ Quantization Error Test Results\n")
        f.write("=" * 70 + "\n\n")
        f.write("Image Space Metrics:\n")
        f.write(f"  MSE:      {image_mse:.6f}\n")
        f.write(f"  MAE:      {image_mae:.6f}\n")
        f.write(f"  PSNR:     {image_psnr:.2f} dB\n")
        f.write(f"  Rel Error:{image_rel:.4f}\n")
        f.write("\nTime Series Space Metrics:\n")
        f.write(f"  MSE:      {time_mse:.6f}\n")
        f.write(f"  MAE:      {time_mae:.6f}\n")
        f.write(f"  PSNR:     {time_psnr:.2f} dB\n")
        f.write(f"  Rel Error:{time_rel:.4f}\n")
    
    print(f"\n结果已保存到：{output_path}")
    
    return results


if __name__ == '__main__':
    results = test_vq_quantization_error()
