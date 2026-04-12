import sys
from pathlib import Path
sys.path.insert(0, str(Path('.') / 'visiontsrar'))
import torch
import pandas as pd
import torch.nn.functional as F
from visiontsrar import VisionTSRAR

print("=" * 70)
print("检查 MSE loss 高的原因")
print("=" * 70)

# 加载数据
df = pd.read_csv('long_term_tsf/dataset/ETT-small/ETTh1.csv')
full_data = df.drop(columns=['date']).values
full_data = torch.from_numpy(full_data).float()[:500]
print(f'\n原始数据范围: [{full_data.min():.3f}, {full_data.max():.3f}]')

# 加载模型
vm = VisionTSRAR(arch='rar_l_0.3b', finetune_type='ln', ckpt_dir='./ckpt', load_ckpt=True)
vm.train()
vm.update_config(context_len=96, pred_len=96, periodicity=24)

# 准备数据
x_enc = full_data[:96, :].unsqueeze(0).repeat(2, 1, 1)
print(f'输入 x_enc 范围: [{x_enc.min():.3f}, {x_enc.max():.3f}]')

# 前向传播
y_train, rar_loss = vm.forward(x_enc)
print(f'模型输出 y_train 范围: [{y_train.min():.3f}, {y_train.max():.3f}]')

# 真实值
y_true = full_data[96:192, :].unsqueeze(0).repeat(2, 1, 1)
y_true = y_true[:, :y_train.shape[1], :]
print(f'真实值 y_true 范围: [{y_true.min():.3f}, {y_true.max():.3f}]')

# 直接 MSE
mse = F.mse_loss(y_train, y_true).item()
print(f'\n直接 MSE: {mse:.6f}')

# 如果先标准化再算 MSE
x_mean = x_enc.mean()
x_std = x_enc.std()
y_train_norm = (y_train - x_mean) / (x_std + 1e-5)
y_true_norm = (y_true - x_mean) / (x_std + 1e-5)
mse_norm = F.mse_loss(y_train_norm, y_true_norm).item()
print(f'标准化后 MSE: {mse_norm:.6f}')

# 检查模型的 normalized 输出
print(f'\n模型是否做了 normalization?')
print(f'  y_train mean: {y_train.mean():.6f}')
print(f'  y_train std: {y_train.std():.6f}')

# 对比 DLinear 风格的 MSE (直接用原始数据)
print(f'\n对比:')
print(f'  DLinear MSE: ~0.4 (在标准化数据上)')
print(f'  我们的 MSE: {mse:.6f} (在原始数据上)')

# 检查 VisionTS 的输出是否做了反标准化
print(f'\n问题: 模型输出可能没有正确反标准化')