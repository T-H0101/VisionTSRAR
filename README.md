# VisionTSRAR 项目总结

## 📋 目录
1. [项目概述](#项目概述)
2. [架构设计](#架构设计)
3. [核心改进](#核心改进)
4. [训练方案](#训练方案)
5. [测试结果](#测试结果)

---

## 项目概述

**VisionTSRAR** 是一个结合 **RandAR**（视觉生成）和 **VisionTS**（时序预测）的混合模型，用于长时序预测任务。

### 技术栈
- **RandAR**：Decoder-only 自回归视觉生成模型
- **VisionTS**：时序→图像转换框架
- **VQ Tokenizer**：离散化图像表示
- **MAE + MSE Loss**：像素级重建损失

---

## 架构设计

### 训练流程

```
时序输入 [b, 192, 1]
    ↓
时序 → 2D图像
    ↓ VQ encode
tokens [b, 256]
    ↓
RAR GPT forward（冻结但参与，计算token顺序）
    ↓
logits → argmax → predicted tokens
    ↓ VQ decode
重建图像 → 2D图像 → 时序
    ↓
loss = MAE(预测, 真实) + MSE(预测, 真实)
```

### 推理流程

```
历史时序 [b, 96, 1] + 占位符
    ↓
时序 → 2D图像
    ↓ VQ encode
visible tokens
    ↓
RAR GPT generate（自回归生成未来tokens）
    ↓ VQ decode
重建图像 → 2D图像 → 预测时序 [b, 96, 1]
```

### 核心特点

| 特性 | 说明 |
|------|------|
| **骨干冻结** | RAR GPT 参数冻结（`ft_type=In`），只训 VQ Tokenizer |
| **Loss 类型** | MAE + MSE（像素级重建） |
| **Token 顺序** | 历史 raster，未来 random（由 RAR GPT 处理） |
| **训练目标** | 预测的未来 vs 真实的未来 |

---

## 核心改进

### 1. 训练流程优化

#### 改动前（旧版）
- RAR GPT 参与训练 + 更新参数
- Loss = CE（交叉熵）
- 混合 Loss = MSE (99.9%) + CE (0.1%)

#### 改动后（新版）
- RAR GPT 参与 forward 但冻结参数
- Loss = MAE + MSE（像素级）
- 历史和未来区域分开计算 Loss

### 2. 值域对齐

VQ 解码器输出 `[-1, 1]`，通过归一化对齐到输入值域：

```python
recon = (recon - recon_mean) / recon_std
recon = recon * img_std + img_mean
```

### 3. 梯度流控制

| 组件 | 训练时 | 说明 |
|------|--------|------|
| RAR GPT | 冻结（grad=0） | 参与 forward 但不更新 |
| VQ Tokenizer | 可训练（grad≠0） | 唯一更新的部分 |

---

## 训练方案

### 快速开始

```bash
cd VisionTSRAR/long_term_tsf

# ETTh1 96→96 预测（3轮，无中间测试）
python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ETTh1_96_96 \
  --model VisionTSRAR \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --freq h \
  --rar_arch rar_l_0.3b \
  --train_epochs 3 \
  --batch_size 1 \
  --learning_rate 0.0001 \
  --use_gpu True \
  --gpu 0 \
  --ft_type In \
  --test_freq 0 \
  > training_etth1.log 2>&1 &
```

### 关键参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--ft_type` | `In` | Inpainting 模式，冻结 RAR |
| `--train_epochs` | 3-20 | 先跑 3 轮看效果 |
| `--batch_size` | 1 | RTX 3090 显存限制 |
| `--test_freq` | 0 | 只在最后 test（加快训练） |
| `--learning_rate` | 0.0001 | 学习率 |

### ft_type 选项

| ft_type | 说明 | 训练对象 |
|---------|------|----------|
| `In` | Inpainting 模式 | 只训练 VQ Tokenizer |
| `ln` | LayerNorm 可训练 | RAR GPT 部分可训练 |
| `full` | 所有参数可训练 | 全部可训练 |

---

## 测试结果

### 训练测试（随机数据）

```
Train forward: OK
Loss: ~1.12 (MAE + MSE)
RAR grad: 0 ✅
VQ grad: 196 ✅
```

### ETTh1 数据集结果

| 指标 | 修改前 | 目标 |
|------|--------|------|
| MSE | 19.97 | 下降 |
| MAE | 2.86 | 下降 |

---

## 文件结构

```
VisionTSRAR/
├── visiontsrar/
│   ├── model.py              # 主模型
│   ├── models_rar.py         # RAR 封装
│   └── randar/
│       ├── rar_gpt.py       # RAR GPT
│       └── tokenizer.py      # VQ Tokenizer
├── long_term_tsf/
│   ├── run.py               # 训练入口
│   └── exp/                 # 实验类
├── TRAINING_GUIDE.md         # 训练指南
└── CHANGELOG.md             # 改动记录
```

---

## 改动记录

详细改动请参考 [CHANGELOG.md](./CHANGELOG.md)

### 核心改动（2026-04-13）

1. Loss 从 CE + MSE 改为纯 MAE + MSE
2. RAR GPT 完全冻结（`ft_type=In`）
3. Loss 计算分开：历史 MAE + 未来 MAE + 历史 MSE + 未来 MSE
4. 添加 `--test_freq` 参数控制测试频率
