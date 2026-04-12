"""
ETTh1 真实数据训练模式测试

测试目的：
1. 验证 VisionTSRAR 能用真实数据跑通训练模式
2. 验证梯度流正常

运行命令：
cd /Users/tian/Desktop/VisionTS/VisionTSRAR
python test_real_etth1.py
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / 'visiontsrar'))

import torch
import pandas as pd
import torch.nn.functional as F
from visiontsrar import VisionTSRAR

print("=" * 70)
print("ETTh1 真实数据测试（训练模式）")
print("=" * 70)

print("\n[1/4] 加载 ETTh1 数据集...")
data_path = project_root / 'long_term_tsf' / 'dataset' / 'ETT-small' / 'ETTh1.csv'
df = pd.read_csv(data_path)
full_data = df.drop(columns=['date']).values
full_data = torch.from_numpy(full_data).float()[:500]
print(f"  数据形状: {full_data.shape}")

print("\n[2/4] 加载 VisionTSRAR 模型...")
vm = VisionTSRAR(
    arch='rar_l_0.3b',
    finetune_type='ln',
    ckpt_dir=str(project_root / 'ckpt'),
    load_ckpt=True,
)
vm.train()
print("  ✅ 模型加载成功")

print("\n[3/4] 准备测试数据...")
context_len = 96
pred_len = 96
periodicity = 24
batch_size = 2

vm.update_config(
    context_len=context_len,
    pred_len=pred_len,
    periodicity=periodicity,
)

x_enc = full_data[:context_len, :].unsqueeze(0).repeat(batch_size, 1, 1)
print(f"  输入形状: {x_enc.shape}")

print("\n[4/4] 测试训练模式前向传播和梯度流...")
x_train = x_enc.clone().requires_grad_(True)

y_train, rar_loss = vm.forward(x_train)
print(f"  输出形状: {y_train.shape}")
print(f"  RAR loss: {rar_loss}")

y_true = full_data[context_len:context_len + pred_len, :].unsqueeze(0).repeat(batch_size, 1, 1)
y_true = y_true[:, :y_train.shape[1], :]

loss = F.mse_loss(y_train, y_true)
if rar_loss is not None:
    loss = loss + 0.001 * rar_loss.mean()
print(f"  MSE Loss: {loss.item():.6f}")

loss.backward()

print("\n  梯度检查:")
has_grad = x_train.grad is not None
print(f"    x_enc.grad exists: {has_grad}")

rar_grads = [(n, p.grad.norm().item()) for n, p in vm.rar_wrapper.rar_gpt.named_parameters() if p.grad is not None]
print(f"    RAR GPT 有梯度的参数: {len(rar_grads)}")

decoder_grads = [(n, p.grad.norm().item()) for n, p in vm.rar_wrapper.vq_tokenizer.named_parameters()
                if 'decoder' in n.lower() and p.grad is not None]
print(f"    VQ Decoder 有梯度的参数: {len(decoder_grads)}")

if has_grad and len(rar_grads) > 0 and len(decoder_grads) > 0:
    print("\n" + "=" * 70)
    print("✅ 所有测试通过！可以开始训练！")
    print("=" * 70)
else:
    print("\n" + "=" * 70)
    print("❌ 梯度流有问题!")
    print("=" * 70)