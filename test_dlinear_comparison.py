"""
DLinear 对比测试脚本

测试目的：
1. 使用与 VisionTSRAR 相同的模拟数据
2. 对比 DLinear（最差线性模型）与 VisionTSRAR 的性能差距
3. 验证深度模型相对于线性模型的改进

运行命令：
cd /Users/tian/Desktop/VisionTS/VisionTSRAR
python test_dlinear_comparison.py
"""

import sys
import os
from pathlib import Path

# 设置路径
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / 'visiontsrar'))
sys.path.insert(0, str(project_root / '../Time-Series-Library'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops

from models.DLinear import Model as DLinearModel
from visiontsrar import VisionTSRAR


def test_dlinear(batch_size, context_len, pred_len, nvars):
    """测试 DLinear 模型"""
    print("\n" + "=" * 70)
    print("DLinear 模型测试")
    print("=" * 70)
    
    # ========== 1. 创建 DLinear 模型 ==========
    print("\n[1/3] 创建 DLinear 模型...")
    
    class Config:
        pass
    
    config = Config()
    config.seq_len = context_len
    config.pred_len = pred_len
    config.enc_in = nvars
    config.dec_in = nvars
    config.c_out = nvars
    config.task_name = 'long_term_forecast'
    config.moving_avg = 25
    config.individual = False
    
    dlinear = DLinearModel(config)
    dlinear.eval()
    print("  ✅ DLinear 模型创建成功")
    
    # ========== 2. 准备测试数据 ==========
    print("\n[2/3] 准备测试数据...")
    torch.manual_seed(42)
    x_enc = torch.randn(batch_size, context_len, nvars) * 0.5 + 1.0
    print(f"  输入形状：{x_enc.shape}")
    print(f"  数据范围：[{x_enc.min():.3f}, {x_enc.max():.3f}]")
    
    # ========== 3. 运行 DLinear 预测 ==========
    print("\n[3/3] 运行 DLinear 预测...")
    
    with torch.no_grad():
        # DLinear 期望输入 [B, L, D]
        dec_out = dlinear(x_enc, None, None, None)
        dlinear_pred = dec_out[:, -pred_len:, :]
    
    print(f"  预测形状：{dlinear_pred.shape}")
    
    # 计算误差
    time_mse = ((x_enc - dlinear_pred) ** 2).mean().item()
    time_mae = torch.abs(x_enc - dlinear_pred).mean().item()
    time_psnr = 10 * np.log10(1.0 ** 2 / max(time_mse, 1e-8)) / np.log(10)
    time_rel = (torch.abs(x_enc - dlinear_pred) / 
                (torch.abs(x_enc) + 1e-8)).mean().item()
    
    print("\n  DLinear 时序预测质量:")
    print(f"    MSE:      {time_mse:.6f}")
    print(f"    MAE:      {time_mae:.6f}")
    print(f"    PSNR:     {time_psnr:.2f} dB")
    print(f"    相对误差：{time_rel:.4f} ({time_rel*100:.2f}%)")
    
    return {
        'time_mse': time_mse,
        'time_mae': time_mae,
        'time_psnr': time_psnr,
        'time_rel': time_rel,
    }


def test_visiontsrar(batch_size, context_len, pred_len, nvars):
    """测试 VisionTSRAR 模型"""
    print("\n" + "=" * 70)
    print("VisionTSRAR 模型测试")
    print("=" * 70)
    
    # ========== 1. 加载模型 ==========
    print("\n[1/3] 加载 VisionTSRAR 模型...")
    try:
        ckpt_dir = project_root / 'ckpt'
        vm = VisionTSRAR(
            arch='rar_l_0.3b',
            finetune_type='ln',
            ckpt_dir=str(ckpt_dir),
            load_ckpt=True,
        )
        vm.eval()
        print("  ✅ 模型加载成功")
    except Exception as e:
        print(f"  ❌ 模型加载失败：{e}")
        return None
    
    # ========== 2. 初始化配置 ==========
    print("\n[2/3] 初始化模型配置...")
    periodicity = 24
    vm.update_config(
        context_len=context_len,
        pred_len=pred_len,
        periodicity=periodicity,
    )
    print(f"  周期性：{periodicity}")
    print(f"  可见 token 数：{vm.num_visible_tokens}")
    
    # ========== 3. 准备测试数据并运行 ==========
    print("\n[3/3] 准备测试数据并运行...")
    torch.manual_seed(42)
    x_enc = torch.randn(batch_size, context_len, nvars) * 0.5 + 1.0
    print(f"  输入形状：{x_enc.shape}")
    print(f"  数据范围：[{x_enc.min():.3f}, {x_enc.max():.3f}]")
    
    # 时序 → 图像
    means = x_enc.mean(1, keepdim=True).detach()
    x_norm = x_enc - means
    stdev = torch.sqrt(
        torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5
    )
    stdev /= vm.norm_const
    x_norm = x_norm / stdev
    
    x_norm = einops.rearrange(x_norm, 'b s n -> b n s')
    x_pad = F.pad(x_norm, (vm.pad_left, 0), mode='replicate')
    x_2d = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=vm.periodicity)
    x_resize = vm.input_resize(x_2d)
    masked = torch.zeros(
        (x_2d.shape[0], 1, vm.image_size, vm.num_patch_output * vm.patch_size),
        device=x_2d.device, dtype=x_2d.dtype,
    )
    x_concat_with_masked = torch.cat([x_resize, masked], dim=-1)
    image_input = einops.repeat(x_concat_with_masked, 'b 1 h w -> b c h w', c=3)
    
    # 运行 VisionTSRAR
    with torch.no_grad():
        reconstructed_image, _ = vm.rar_wrapper(image_input, vm.num_visible_tokens)
    
    # 图像 → 时序
    y_grey = torch.mean(reconstructed_image, 1, keepdim=True)
    y_segmentations = vm.output_resize(y_grey)
    y_flatten = einops.rearrange(
        y_segmentations,
        '(b n) 1 f p -> b (p f) n',
        b=batch_size, f=vm.periodicity,
    )
    y_pred = y_flatten[
        :,
        vm.pad_left + vm.context_len: vm.pad_left + vm.context_len + vm.pred_len,
        :,
    ]
    y_pred = y_pred * (stdev.repeat(1, vm.pred_len, 1))
    y_pred = y_pred + (means.repeat(1, vm.pred_len, 1))
    
    print(f"  预测形状：{y_pred.shape}")
    
    # 计算误差
    time_mse = ((x_enc - y_pred) ** 2).mean().item()
    time_mae = torch.abs(x_enc - y_pred).mean().item()
    time_psnr = 10 * np.log10(1.0 ** 2 / max(time_mse, 1e-8)) / np.log(10)
    time_rel = (torch.abs(x_enc - y_pred) / 
                (torch.abs(x_enc) + 1e-8)).mean().item()
    
    print("\n  VisionTSRAR 时序预测质量:")
    print(f"    MSE:      {time_mse:.6f}")
    print(f"    MAE:      {time_mae:.6f}")
    print(f"    PSNR:     {time_psnr:.2f} dB")
    print(f"    相对误差：{time_rel:.4f} ({time_rel*100:.2f}%)")
    
    return {
        'time_mse': time_mse,
        'time_mae': time_mae,
        'time_psnr': time_psnr,
        'time_rel': time_rel,
    }


def main():
    """主函数"""
    print("=" * 70)
    print("DLinear vs VisionTSRAR 性能对比测试")
    print("=" * 70)
    
    # 测试参数
    batch_size = 4
    context_len = 96
    pred_len = 96
    nvars = 7
    
    print(f"\n测试参数：")
    print(f"  Batch Size: {batch_size}")
    print(f"  Context Length: {context_len}")
    print(f"  Prediction Length: {pred_len}")
    print(f"  Variables: {nvars}")
    
    # 测试 DLinear
    dlinear_results = test_dlinear(batch_size, context_len, pred_len, nvars)
    
    # 测试 VisionTSRAR
    visiontsrar_results = test_visiontsrar(batch_size, context_len, pred_len, nvars)
    
    # ========== 性能对比 ==========
    print("\n" + "=" * 70)
    print("性能对比总结")
    print("=" * 70)
    
    if dlinear_results is None or visiontsrar_results is None:
        print("  ❌ 测试失败")
        return
    
    print("\n  指标对比表：")
    print(f"  {'指标':<15} {'DLinear':<15} {'VisionTSRAR':<15} {'改进':<15}")
    print(f"  {'-'*60}")
    
    # MSE
    dlinear_mse = dlinear_results['time_mse']
    visiontsrar_mse = visiontsrar_results['time_mse']
    mse_improvement = ((dlinear_mse - visiontsrar_mse) / dlinear_mse * 100) if dlinear_mse > 0 else 0
    print(f"  {'MSE':<15} {dlinear_mse:<15.6f} {visiontsrar_mse:<15.6f} {mse_improvement:<15.2f}%")
    
    # MAE
    dlinear_mae = dlinear_results['time_mae']
    visiontsrar_mae = visiontsrar_results['time_mae']
    mae_improvement = ((dlinear_mae - visiontsrar_mae) / dlinear_mae * 100) if dlinear_mae > 0 else 0
    print(f"  {'MAE':<15} {dlinear_mae:<15.6f} {visiontsrar_mae:<15.6f} {mae_improvement:<15.2f}%")
    
    # PSNR
    dlinear_psnr = dlinear_results['time_psnr']
    visiontsrar_psnr = visiontsrar_results['time_psnr']
    psnr_improvement = visiontsrar_psnr - dlinear_psnr
    print(f"  {'PSNR (dB)':<15} {dlinear_psnr:<15.2f} {visiontsrar_psnr:<15.2f} {psnr_improvement:<15.2f} dB")
    
    # Rel Error
    dlinear_rel = dlinear_results['time_rel']
    visiontsrar_rel = visiontsrar_results['time_rel']
    rel_improvement = ((dlinear_rel - visiontsrar_rel) / dlinear_rel * 100) if dlinear_rel > 0 else 0
    print(f"  {'Rel Error %':<15} {dlinear_rel*100:<15.2f} {visiontsrar_rel*100:<15.2f} {rel_improvement:<15.2f}%")
    
    # ========== 结论 ==========
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    
    if visiontsrar_mse < dlinear_mse:
        improvement_pct = (dlinear_mse - visiontsrar_mse) / dlinear_mse * 100
        print(f"  ✅ VisionTSRAR 在 MSE 上比 DLinear 降低了 {improvement_pct:.2f}%")
    else:
        degradation_pct = (visiontsrar_mse - dlinear_mse) / dlinear_mse * 100
        print(f"  ⚠️  VisionTSRAR 在 MSE 上比 DLinear 高了 {degradation_pct:.2f}%")
    
    if visiontsrar_mae < dlinear_mae:
        improvement_pct = (dlinear_mae - visiontsrar_mae) / dlinear_mae * 100
        print(f"  ✅ VisionTSRAR 在 MAE 上比 DLinear 降低了 {improvement_pct:.2f}%")
    else:
        degradation_pct = (visiontsrar_mae - dlinear_mae) / dlinear_mae * 100
        print(f"  ⚠️  VisionTSRAR 在 MAE 上比 DLinear 高了 {degradation_pct:.2f}%")
    
    if visiontsrar_psnr > dlinear_psnr:
        print(f"  ✅ VisionTSRAR 在 PSNR 上比 DLinear 高了 {visiontsrar_psnr - dlinear_psnr:.2f} dB")
    else:
        print(f"  ⚠️  VisionTSRAR 在 PSNR 上比 DLinear 低了 {dlinear_psnr - visiontsrar_psnr:.2f} dB")
    
    print("\n" + "=" * 70)
    print("注意：")
    print("  - 本测试使用随机模拟数据，仅作对比参考")
    print("  - 实际性能需在真实数据集上验证")
    print("  - DLinear 是最简单的线性模型，作为下界参考")
    print("=" * 70)


if __name__ == '__main__':
    main()
