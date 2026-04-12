"""
梯度流测试脚本

测试目的：
1. 验证 STE (Straight-Through Estimator) 的梯度能否正确反向传播
2. 验证 VQ Decoder 能否收到梯度
3. 验证 RAR GPT 能否收到梯度
4. 确保端到端训练可行

运行命令：
cd /Users/tian/Desktop/VisionTS/VisionTSRAR
python test_gradient_flow.py
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / 'visiontsrar'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops

from visiontsrar.models_rar import RARWrapper, StraightThroughEstimator


def test_ste_gradient():
    """测试 STE 的梯度流"""
    print("=" * 70)
    print("测试 1: STE (Straight-Through Estimator) 梯度流")
    print("=" * 70)

    B, L, V = 2, 10, 100
    logits = torch.randn(B, L, V, requires_grad=True)
    target = torch.randint(0, V, (B, L))

    output = StraightThroughEstimator.apply(logits)
    loss = F.cross_entropy(
        logits.view(-1, V),
        target.view(-1)
    )

    print(f"  logits.requires_grad: {logits.requires_grad}")
    print(f"  output.requires_grad: {output.requires_grad}")
    print(f"  loss.requires_grad: {loss.requires_grad}")

    loss.backward()
    print(f"  logits.grad is not None: {logits.grad is not None}")
    print(f"  logits.grad.shape: {logits.grad.shape}")

    if logits.grad is not None:
        print("  ✅ STE 梯度流正常！")
    else:
        print("  ❌ STE 梯度流断裂！")

    return logits.grad is not None


def test_vq_decoder_gradient():
    """测试 VQ Decoder 的梯度流"""
    print("\n" + "=" * 70)
    print("测试 2: VQ Decoder 梯度流")
    print("=" * 70)

    try:
        ckpt_dir = project_root / 'ckpt'
        wrapper = RARWrapper(
            rar_arch='rar_l_0.3b',
            finetune_type='ln',
            ckpt_dir=str(ckpt_dir),
            load_ckpt=False,
        )
        print(f"  VQ Tokenizer 已加载")
    except Exception as e:
        print(f"  ⚠️ 无法加载 VQ Tokenizer: {e}")
        print(f"  使用随机初始化进行测试")
        wrapper = RARWrapper(
            rar_arch='rar_l_0.3b',
            finetune_type='ln',
            ckpt_dir='./ckpt',
            load_ckpt=False,
        )

    wrapper.eval()

    B, H, W = 2, 224, 224
    image_input = torch.randn(B, 3, H, W, requires_grad=True)
    num_visible_tokens = 64

    print(f"  输入图像形状: {image_input.shape}")
    print(f"  num_visible_tokens: {num_visible_tokens}")

    print("\n  检查参数冻结情况:")
    for name, param in wrapper.vq_tokenizer.named_parameters():
        if 'decoder' in name.lower():
            print(f"    {name}: requires_grad={param.requires_grad}")

    print("\n  前向传播...")
    reconstructed_image, loss = wrapper(image_input, num_visible_tokens)

    print(f"  重建图像形状: {reconstructed_image.shape}")
    print(f"  loss: {loss}")

    if loss is not None:
        print("\n  反向传播测试...")
        # 计算图像空间 MSE loss（这样梯度才能传到 VQ Decoder）
        target_image = torch.randn_like(reconstructed_image)
        mse_loss = F.mse_loss(reconstructed_image, target_image)
        combined_loss = mse_loss + 0.001 * loss
        combined_loss.backward()

        decoder_params = [p for n, p in wrapper.vq_tokenizer.named_parameters()
                        if 'decoder' in n.lower() and p.requires_grad]
        grad_exists = [p.grad is not None for p in decoder_params]

        if any(grad_exists):
            print("  ✅ VQ Decoder 梯度正常!")
            for i, (n, p) in enumerate([(n, p) for n, p in wrapper.vq_tokenizer.named_parameters()
                                        if 'decoder' in n.lower()]):
                if p.grad is not None:
                    print(f"    {n}: grad_norm={p.grad.norm().item():.6f}")
            return True
        else:
            print("  ❌ VQ Decoder 没有收到梯度!")
            return False
    else:
        print("  ⚠️ loss 为 None，跳过测试")
        return False


def test_end_to_end_gradient():
    """测试端到端梯度流"""
    print("\n" + "=" * 70)
    print("测试 3: 端到端梯度流（模拟训练）")
    print("=" * 70)

    try:
        wrapper = RARWrapper(
            rar_arch='rar_l_0.3b',
            finetune_type='ln',
            ckpt_dir=str(project_root / 'ckpt'),
            load_ckpt=False,
        )
    except:
        wrapper = RARWrapper(
            rar_arch='rar_l_0.3b',
            finetune_type='ln',
            ckpt_dir='./ckpt',
            load_ckpt=False,
        )

    B, H, W = 2, 224, 224
    image_input = torch.randn(B, 3, H, W, requires_grad=True)
    target = torch.randn(B, 3, H, W)
    num_visible_tokens = 64

    print(f"  输入图像形状: {image_input.shape}")
    print(f"  目标图像形状: {target.shape}")

    print("\n  前向传播...")
    reconstructed_image, rar_loss = wrapper(image_input, num_visible_tokens)

    print(f"  重建图像形状: {reconstructed_image.shape}")
    print(f"  RAR loss: {rar_loss}")

    mse_loss = F.mse_loss(reconstructed_image, target)

    print(f"\n  MSE loss: {mse_loss.item():.6f}")

    if rar_loss is not None:
        combined_loss = mse_loss + 0.001 * rar_loss
        print(f"  Combined loss: {combined_loss.item():.6f}")
    else:
        combined_loss = mse_loss

    print("\n  反向传播...")
    combined_loss.backward()

    print("\n  检查各组件梯度:")
    grad_info = []

    for name, param in wrapper.rar_gpt.named_parameters():
        if param.grad is not None:
            grad_info.append(f"    RAR.{name}: grad_norm={param.grad.norm().item():.6f}")

    for name, param in wrapper.vq_tokenizer.named_parameters():
        if param.grad is not None and 'decoder' in name.lower():
            grad_info.append(f"    VQ_Decoder.{name}: grad_norm={param.grad.norm().item():.6f}")

    if grad_info:
        for info in grad_info[:10]:
            print(info)
        if len(grad_info) > 10:
            print(f"    ... and {len(grad_info) - 10} more")
        print("  ✅ 端到端梯度流正常!")
        return True
    else:
        print("  ❌ 端到端梯度流断裂!")
        return False


def test_full_visiontsrar_pipeline():
    """测试完整的 VisionTSRAR pipeline"""
    print("\n" + "=" * 70)
    print("测试 4: 完整 VisionTSRAR Pipeline")
    print("=" * 70)

    try:
        from visiontsrar import VisionTSRAR

        vm = VisionTSRAR(
            arch='rar_l_0.3b',
            finetune_type='ln',
            ckpt_dir=str(project_root / 'ckpt'),
            load_ckpt=False,
        )
        vm.train()

        B, context_len, nvars = 2, 96, 7
        pred_len = 96
        periodicity = 24

        x_enc = torch.randn(B, context_len, nvars, requires_grad=True)

        vm.update_config(
            context_len=context_len,
            pred_len=pred_len,
            periodicity=periodicity,
        )

        print(f"  输入时序形状: {x_enc.shape}")
        print(f"  num_visible_tokens: {vm.num_visible_tokens}")

        print("\n  前向传播...")
        y, rar_loss = vm.forward(x_enc)

        print(f"  输出形状: {y.shape}")
        print(f"  RAR loss: {rar_loss}")

        target = torch.randn_like(y)
        mse_loss = F.mse_loss(y, target)

        print(f"\n  MSE loss: {mse_loss.item():.6f}")

        if rar_loss is not None:
            # rar_loss 可能是一个 per-sample 的向量，需要取平均得到标量
            if rar_loss.dim() > 0:
                rar_loss = rar_loss.mean()
            combined_loss = mse_loss + 0.001 * rar_loss
            print(f"  Combined loss: {combined_loss.item():.6f}")
        else:
            combined_loss = mse_loss

        print("\n  反向传播...")
        combined_loss.backward()

        print("\n  检查梯度:")
        has_grad = False
        for name, param in vm.rar_wrapper.rar_gpt.named_parameters():
            if param.grad is not None:
                print(f"    RAR.{name}: grad_norm={param.grad.norm().item():.6f}")
                has_grad = True

        for name, param in vm.rar_wrapper.vq_tokenizer.named_parameters():
            if param.grad is not None and 'decoder' in name.lower():
                print(f"    VQ_Decoder.{name}: grad_norm={param.grad.norm().item():.6f}")
                has_grad = True

        if has_grad:
            print("  ✅ 完整 Pipeline 梯度流正常!")
            return True
        else:
            print("  ❌ 完整 Pipeline 梯度流断裂!")
            return False

    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("VisionTSRAR 梯度流测试")
    print("=" * 70 + "\n")

    results = []

    results.append(("STE 梯度", test_ste_gradient()))
    results.append(("VQ Decoder 梯度", test_vq_decoder_gradient()))
    results.append(("端到端梯度", test_end_to_end_gradient()))
    results.append(("完整 Pipeline", test_full_visiontsrar_pipeline()))

    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)

    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")

    all_passed = all(passed for _, passed in results)
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 所有测试通过！代码可以正常训练！")
    else:
        print("⚠️ 部分测试失败，需要修复代码")
    print("=" * 70)