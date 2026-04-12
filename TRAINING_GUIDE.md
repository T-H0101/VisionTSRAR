# VisionTSRAR 训练与优化指南

## 📋 目录
1. [快速开始](#快速开始)
2. [训练配置](#训练配置)
3. [训练流程](#训练流程)
4. [优化技巧](#优化技巧)
5. [问题排查](#问题排查)

---

## 快速开始

### 1. 环境准备

```bash
# 确保已安装依赖
cd /Users/tian/Desktop/VisionTS/VisionTSRAR

# 检查 Python 环境
/opt/miniconda3/envs/1d_tokenizer/bin/python --version
```

### 2. 模型权重检查

```bash
# 确保权重文件已下载
ls -lh /Users/tian/Desktop/VisionTS/VisionTSRAR/ckpt/

# 预期文件：
# - vq_ds16_c2i.pt
# - randar_0.3b_llamagen_360k_bs_1024_lr_0.0004.safetensors
```

### 3. 开始训练

```bash
cd /Users/tian/Desktop/VisionTS/VisionTSRAR/long_term_tsf

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model VisionTSRAR \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --rar_arch rar_l_0.3b \
  --ft_type ln \
  --train_epochs 20 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --des 'Exp'
```

---

## 训练配置

### 核心参数

| 参数 | 值 | 说明 |
|------|-----|------|
| **is_training** | 1 | 训练模式 |
| **root_path** | ./dataset/ETT-small/ | 数据集路径 |
| **data_path** | ETTh1.csv | 数据文件 |
| **model_id** | ETTh1_96_96 | 实验 ID |
| **model** | VisionTSRAR | 模型名称 |
| **data** | ETTh1 | 数据集名称 |
| **features** | M | 多变量预测 |
| **seq_len** | 96 | 输入长度 |
| **pred_len** | 96 | 预测长度 |
| **rar_arch** | rar_l_0.3b | RandAR 架构 |
| **ft_type** | ln | 微调类型 |
| **train_epochs** | 20 | 训练轮数 |
| **batch_size** | 32 | 批大小 |
| **learning_rate** | 0.0001 | 学习率 |

### 可调参数

#### 1. 学习率

```bash
# 小数据集（ETTh1/ETTh2）
--learning_rate 0.0001

# 中等数据集（ETTm1/ETTm2）
--learning_rate 0.00005

# 大数据集（Weather/Exchange）
--learning_rate 0.00001
```

#### 2. 批大小

```bash
# 小显存（24GB）
--batch_size 16

# 中等显存（40GB）
--batch_size 32

# 大显存（80GB）
--batch_size 64
```

#### 3. 训练轮数

```bash
# 快速验证
--train_epochs 5

# 正式训练
--train_epochs 20

# 精细调优
--train_epochs 50
```

---

## 训练流程

### 阶段 1：初始化（Epoch 1-2）

#### 预期现象
```
Epoch 1:
  iters: 1, epoch: 1 | loss: 0.234567
  [Loss] MSE: 0.234567, CE: 0.000123, Total: 0.234690

Epoch 2:
  iters: 1, epoch: 2 | loss: 0.212345
  [Loss] MSE: 0.212345, CE: 0.000112, Total: 0.212457
```

#### 关注点
- ✅ Loss 是否下降
- ✅ MSE/CE 比例是否正常（MSE:CE ≈ 1000:1）
- ❌ 如果 Loss 不下降 → 检查学习率
- ❌ 如果 Loss = NaN → 检查数据

---

### 阶段 2：快速下降（Epoch 3-10）

#### 预期现象
```
Epoch 5:
  iters: 1, epoch: 5 | loss: 0.123456
  [Loss] MSE: 0.123456, CE: 0.000098, Total: 0.123554

Epoch 10:
  iters: 1, epoch: 10 | loss: 0.067890
  [Loss] MSE: 0.067890, CE: 0.000087, Total: 0.067977
```

#### 关注点
- ✅ Loss 是否稳定下降
- ✅ MSE 是否主导（MSE > 99% of Total）
- ⚠️ 如果 MSE 停滞 → 检查模型容量
- ⚠️ 如果 CE 增大 → 检查 VQ 退化

---

### 阶段 3：精细优化（Epoch 11-20）

#### 预期现象
```
Epoch 15:
  iters: 1, epoch: 15 | loss: 0.034567
  [Loss] MSE: 0.034567, CE: 0.000078, Total: 0.034645

Epoch 20:
  iters: 1, epoch: 20 | loss: 0.023456
  [Loss] MSE: 0.023456, CE: 0.000067, Total: 0.023523
```

#### 关注点
- ✅ Loss 是否收敛
- ✅ 验证集性能是否提升
- ⚠️ 如果过拟合 → 添加正则化
- ⚠️ 如果欠拟合 → 增加模型容量

---

## 优化技巧

### 技巧 1：学习率预热

```bash
# 在 exp_long_term_forecasting.py 中添加
# 训练前 100 iterations 使用小学习率
if iter_count < 100:
    current_lr = base_lr * (iter_count / 100)
    for param_group in model_optim.param_groups:
        param_group['lr'] = current_lr
```

### 技巧 2：学习率衰减

```bash
# 使用余弦退火
--learning_rate 0.0001
--train_epochs 20

# 在训练过程中动态调整
if epoch > 10:
    current_lr = base_lr * 0.1
```

### 技巧 3：梯度裁剪

```python
# 在 exp_long_term_forecasting.py 中添加
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 技巧 4：数据增强

```python
# 在数据加载时添加噪声
x_enc = x_enc + torch.randn_like(x_enc) * 0.01
```

### 技巧 5：混合精度训练

```python
# 在 exp_long_term_forecasting.py 中启用
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    outputs, model_loss = self._model_forward(...)
    loss = criterion(outputs, batch_y) + 0.001 * model_loss

scaler.scale(loss).backward()
scaler.step(model_optim)
scaler.update()
```

---

## 问题排查

### 问题 1：Loss = NaN

#### 可能原因
1. 学习率过大
2. 数据包含 NaN
3. 梯度爆炸

#### 解决方案
```bash
# 降低学习率
--learning_rate 0.00001

# 检查数据
/opt/miniconda3/envs/1d_tokenizer/bin/python -c "
import torch
data = torch.load('data.pt')
print('NaN count:', torch.isnan(data).sum())
"

# 添加梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

### 问题 2：Loss 不下降

#### 可能原因
1. 学习率过小
2. 模型容量不足
3. 数据预处理错误

#### 解决方案
```bash
# 增大学习率
--learning_rate 0.0005

# 增加模型容量
--rar_arch rar_b_1.0b  # 从 rar_l_0.3b 改为 rar_b_1.0b

# 检查数据预处理
# 运行 test_vq_quantization_error.py
```

---

### 问题 3：过拟合

#### 可能原因
1. 模型容量过大
2. 训练轮数过多
3. 数据量不足

#### 解决方案
```bash
# 减少模型容量
--rar_arch rar_s_0.1b  # 从 rar_l_0.3b 改为 rar_s_0.1b

# 减少训练轮数
--train_epochs 10

# 添加正则化
--dropout 0.1
--weight_decay 0.01
```

---

### 问题 4：验证集性能差

#### 可能原因
1. 训练集和验证集分布不一致
2. 模型未收敛
3. 过拟合

#### 解决方案
```bash
# 增加训练轮数
--train_epochs 50

# 使用早停
# 在 exp_long_term_forecasting.py 中添加早停逻辑

# 数据增强
# 在数据加载时添加噪声
```

---

### 问题 5：CE 损失异常

#### 可能原因
1. VQ 退化（码本使用率低）
2. Token 分布不均衡
3. 模型未学习

#### 解决方案
```bash
# 检查码本使用率
# 运行 test_vq_quantization_error.py
# 查看 Token 范围是否覆盖全部 vocab

# 增大 CE 权重
# 在 exp_long_term_forecasting.py 中
ce_weight = 0.01  # 从 0.001 改为 0.01

# 使用更大的 VQ 码本
# 在 randar/tokenizer.py 中
num_embeddings = 32768  # 从 16384 改为 32768
```

---

## 性能评估

### 训练完成后评估

```bash
# 运行测试脚本
cd /Users/tian/Desktop/VisionTS/VisionTSRAR

/opt/miniconda3/envs/1d_tokenizer/bin/python test_vq_quantization_error.py

# 对比 DLinear
/opt/miniconda3/envs/1d_tokenizer/bin/python test_dlinear_comparison.py
```

### 评估指标

| 指标 | 优秀 | 良好 | 一般 | 差 |
|------|------|------|------|----|
| **MSE** | < 0.01 | < 0.05 | < 0.1 | > 0.1 |
| **MAE** | < 0.1 | < 0.2 | < 0.3 | > 0.3 |
| **PSNR** | > 20 dB | > 10 dB | > 5 dB | < 5 dB |

---

## 最佳实践

### 1. 实验记录

```bash
# 每次训练记录参数
echo "Epochs: 20, LR: 0.0001, BS: 32" > experiments/exp1.txt

# 保存模型权重
cp ckpt/*.pt experiments/exp1_weights/
```

### 2. 可视化

```python
# 使用 tensorboard
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/exp1')
writer.add_scalar('Loss/MSE', mse_loss, global_step)
writer.add_scalar('Loss/CE', ce_loss, global_step)
writer.close()
```

### 3. 模型保存

```python
# 在 exp_long_term_forecasting.py 中
if (epoch + 1) % 5 == 0:
    torch.save(model.state_dict(), f'ckpt/model_epoch_{epoch+1}.pt')
```

---

## 常见问题

### Q1: 为什么 CE 权重这么小（0.001）？

**A**: CE 权重小是为了：
1. MSE 主导训练（优化时序预测）
2. CE 仅提供轻微正则化（防止 VQ 退化）
3. 避免 CE 主导导致 token 精度高但时序精度低

### Q2: 什么时候需要增大 CE 权重？

**A**: 当出现以下情况时：
1. VQ 退化（码本使用率低）
2. Token 分布不均衡
3. 重建质量差

### Q3: 如何判断模型是否收敛？

**A**: 观察以下指标：
1. Loss 曲线是否平滑下降
2. 验证集性能是否稳定
3. MSE/CE 比例是否合理（1000:1）

### Q4: 可以使用更大的 VQ 码本吗？

**A**: 可以，但要注意：
1. 更大码本 → 更高内存占用
2. 更大码本 → 更长训练时间
3. 建议从 32768 开始尝试

---

## 总结

### 训练流程
1. ✅ 初始化（Epoch 1-2）：Loss 快速下降
2. ✅ 快速下降（Epoch 3-10）：Loss 稳定下降
3. ✅ 精细优化（Epoch 11-20）：Loss 收敛

### 关键参数
- **learning_rate**: 0.0001（可调）
- **batch_size**: 32（可调）
- **train_epochs**: 20（可调）
- **ce_weight**: 0.001（固定）

### 问题排查
- Loss = NaN → 降低学习率 + 梯度裁剪
- Loss 不下降 → 增大学习率 + 检查数据
- 过拟合 → 减少模型容量 + 添加正则化
- 验证集差 → 增加训练轮数 + 数据增强

---

**最后更新**：2026-04-12  
**版本**：VisionTSRAR v0.4.0 (Optimized)
