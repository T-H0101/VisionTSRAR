import sys
from pathlib import Path
sys.path.insert(0, str(Path('.') / 'visiontsrar'))
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from visiontsrar import VisionTSRAR

print("=" * 70)
print("用标准化数据测试 MSE")
print("=" * 70)

# 加载数据
df = pd.read_csv('long_term_tsf/dataset/ETT-small/ETTh1.csv')
full_data = df.drop(columns=['date']).values
print(f'原始数据形状: {full_data.shape}')

# 标准化
scaler = StandardScaler()
data_scaled = scaler.fit_transform(full_data)
data_scaled = torch.from_numpy(data_scaled).float()
print(f'标准化后数据范围: [{data_scaled.min():.3f}, {data_scaled.max():.3f}]')

# 加载模型
vm = VisionTSRAR(arch='rar_l_0.3b', finetune_type='ln', ckpt_dir='./ckpt', load_ckpt=True)
vm.train()
vm.update_config(context_len=96, pred_len=96, periodicity=24)

# 准备标准化后的数据
x_enc = data_scaled[:96, :].unsqueeze(0).repeat(2, 1, 1)
y_true = data_scaled[96:192, :].unsqueeze(0).repeat(2, 1, 1)

print(f'输入 x_enc 范围: [{x_enc.min():.3f}, {x_enc.max():.3f}]')
print(f'真实值 y_true 范围: [{y_true.min():.3f}, {y_true.max():.3f}]')

# 前向传播
y_train, rar_loss = vm.forward(x_enc)
print(f'模型输出 y_train 范围: [{y_train.min():.3f}, {y_train.max():.3f}]')

# MSE (在标准化空间)
mse = F.mse_loss(y_train, y_true).item()
print(f'\n标准化空间 MSE: {mse:.6f}')

# 对比 DLinear
print(f'\n对比:')
print(f'  DLinear MSE: ~0.4')
print(f'  我们 MSE: {mse:.6f}')

if mse < 0.4:
    print(f'\n✅ 我们比 DLinear 更好!')
elif mse < 0.5:
    print(f'\n⚠️ 我们与 DLinear 接近')
else:
    print(f'\n❌ 我们比 DLinear 差')