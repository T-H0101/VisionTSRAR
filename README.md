# VisionTSRAR 项目文档

## 目录
1. [项目概述](#项目概述)
2. [架构设计](#架构设计)
3. [核心实现](#核心实现)
4. [训练方案](#训练方案)
5. [测试方案](#测试方案)
6. [优化建议](#优化建议)
7. [文件结构](#文件结构)

---

## 项目概述

**VisionTSRAR** 是一个结合 **RandAR**（视觉生成）和 **VisionTS**（时序预测）的混合模型，用于长时序预测任务。

### 核心思想

```
时序数据 → 伪图像 → RAR生成 → 伪图像 → 时序预测
```

### 技术栈

| 组件 | 说明 |
|------|------|
| **RandAR GPT** | Decoder-only 自回归视觉生成模型（预训练，冻结） |
| **VQ Tokenizer** | 图像离散化编码器（冻结） |
| **Lightweight Decoder** | 轻量级解码器（可训练，替换原VQ Decoder） |
| **VisionTS** | 时序→图像转换框架 |

### 模型参数

| 组件 | 参数量 | 状态 |
|------|--------|------|
| RAR GPT (0.3B) | ~372M | 冻结 |
| VQ Encoder | ~360M | 冻结 |
| Lightweight Decoder | ~10M | **可训练** |
| **总计** | ~382M | 2.6%可训练 |

---

## 架构设计

### 训练流程

```
时序输入 [b, seq_len, nvars]
    ↓ Normalization + Segmentation + Render
2D伪图像 [b*nvars, 3, 256, 256]
    ↓ VQ Encode (冻结)
tokens [b*nvars, 256]
    ↓ RAR GPT Forward (冻结)
token_logits [b*nvars, 256, vocab_size]
    ↓ Schedule Sampling (Teacher Forcing / Generate)
predicted_tokens [b*nvars, 256]
    ↓ Lightweight Decoder (可训练)
重建图像 [b*nvars, 3, 256, 256]
    ↓ De-render + Denormalization
预测时序 [b, pred_len, nvars]
    ↓
loss = MAE + MSE (历史 + 未来分开计算)
```

### 推理流程

```
历史时序 [b, seq_len, nvars]
    ↓ Normalization + Segmentation + Render
2D伪图像 (左半真实，右半占位)
    ↓ VQ Encode
visible_tokens [b*nvars, 128]
    ↓ RAR GPT Generate (自回归生成)
generated_tokens [b*nvars, 256]
    ↓ Lightweight Decoder
重建图像 [b*nvars, 3, 256, 256]
    ↓ De-render + Denormalization
预测时序 [b, pred_len, nvars]
```

### 核心特点

| 特性 | 说明 |
|------|------|
| **骨干冻结** | RAR GPT + VQ Encoder 冻结，只训练 Lightweight Decoder |
| **Loss 类型** | MAE + MSE（像素级重建，历史+未来分开） |
| **Token 顺序** | 历史 raster，未来 random |
| **值域对齐** | 归一化对齐 VQ 输出 `[-1,1]` 到输入值域 |

---

## 核心实现

### 1. 轻量级 Decoder

**目的**：替换原 VQ Decoder（~360M参数），减少显存占用和checkpoint大小。

**实现位置**：`visiontsrar/randar/lightweight_decoder.py`

**架构**：
```python
LightweightDecoder(
    z_channels=256,           # VQ latent 维度
    base_channels=64,         # 基础通道数
    ch_mult=[1, 2, 4],        # 通道倍增（3层上采样）
    num_res_blocks=2,         # 每层残差块数
    out_channels=3,           # 输出RGB通道
)
```

**参数量**：~10M（原 VQ Decoder ~360M）

**Checkpoint 大小**：~38MB（原 ~1.5GB）

### 2. Schedule Sampling

**目的**：解决训练-测试分布不一致问题（Exposure Bias）。

**实现位置**：`visiontsrar/models_rar.py` → `_forward_train()`

**策略**：
```python
# Teacher Forcing Ratio 调度
if epoch <= 2:
    teacher_forcing_ratio = 1.0      # 100% Teacher Forcing
elif epoch <= 6:
    teacher_forcing_ratio = 1.0 - (epoch - 2) / 4 * 0.8  # 线性下降 1.0→0.2
else:
    teacher_forcing_ratio = 0.2      # 保持 20%
```

**训练时行为**：
- Teacher Forcing：使用 RAR GPT Forward（快速，token准确）
- Generate：使用 RAR GPT Generate（慢，模拟测试分布）

### 3. 梯度流

```
loss.backward()
    ↓
recon_image
    ↓
LightweightDecoder(quant)  ← Decoder参数获得梯度 ✓
    ↓
quant = embedding[predicted_tokens]  ← 离散lookup，梯度断开
    ↓
predicted_tokens  ← 无梯度
```

**关键**：Decoder参数可以直接从loss获得梯度，不需要通过codebook传递。

### 4. 值域对齐

VQ Decoder 输出 `[-1, 1]`，需要归一化到输入值域：

```python
def _normalize_image(recon, target):
    recon_mean = recon.mean(dim=[1,2,3], keepdim=True)
    recon_std = recon.std(dim=[1,2,3], keepdim=True) + 1e-8
    img_mean = target.mean(dim=[1,2,3], keepdim=True)
    img_std = target.std(dim=[1,2,3], keepdim=True) + 1e-8
    
    recon = (recon - recon_mean) / recon_std
    recon = recon * img_std + img_mean
    return recon
```

---

## 训练方案

### 快速开始

```bash
cd VisionTSRAR/long_term_tsf

python -m run \
    --task_name long_term_forecast \
    --is_training 1 \
    --model VisionTSRAR \
    --model_id light_decoder_ETTh1_sl96_pl96 \
    --data ETTh1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --seasonal_patterns Monthly \
    --use_lightweight_decoder \
    --lightweight_decoder_channels 64 \
    --ft_type In \
    --use_amp \
    --batch_size 16 \
    --itr 1 \
    --train_epochs 10 \
    --learning_rate 0.0001 \
    --lradj cosine \
    --skip_validation 1 \
    --skip_test 1 \
    --des light_decoder_ch64_schedule_sampling
```

### 关键参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--ft_type` | `In` | Inpainting 模式，冻结 RAR GPT |
| `--use_lightweight_decoder` | 必须添加 | 启用轻量 Decoder |
| `--lightweight_decoder_channels` | 64 | Decoder 基础通道数 |
| `--use_amp` | 建议添加 | 混合精度训练，减少显存 |
| `--batch_size` | 16 | RTX 3090 可支持 |
| `--lradj` | `cosine` | 余弦退火学习率 |
| `--skip_validation` | 1 | 跳过验证，加快训练 |
| `--skip_test` | 1 | 训练时跳过测试 |

### ft_type 选项

| ft_type | 说明 | 训练对象 |
|---------|------|----------|
| `In` | Inpainting 模式 | 只训练 Lightweight Decoder |
| `ln` | LayerNorm 可训练 | RAR GPT LayerNorm + Decoder |
| `full` | 所有参数可训练 | 全部可训练（不推荐） |

### 显存占用

| 配置 | Batch Size | 显存 |
|------|------------|------|
| 原版 VQ Decoder | 1 | ~22GB |
| Lightweight Decoder | 16 | ~21GB |
| Lightweight Decoder + AMP | 16 | ~18GB |

---

## 测试方案

### 训练后测试

```bash
python -m run \
    --task_name long_term_forecast \
    --is_training 0 \
    --model VisionTSRAR \
    --model_id light_decoder_ETTh1_sl96_pl96 \
    --data ETTh1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --seasonal_patterns Monthly \
    --use_lightweight_decoder \
    --lightweight_decoder_channels 64 \
    --ft_type In \
    --batch_size 16
```

### 本地验证（MacBook）

```bash
cd VisionTSRAR
conda run -n 1d_tokenizer python scripts/local_test.py
```

**验证内容**：
- 模型初始化
- 前向传播
- Checkpoint 保存/加载

---

## 优化建议

### 1. torch.compile（暂不可用）

**状态**：已实现但与 Schedule Sampling 不兼容

**原因**：Schedule Sampling 使用动态控制流 `torch.rand(1).item()`，导致 `torch.compile` 图断裂。

**解决方案**（未来）：
```python
# 方案A：将随机采样移到训练循环外
# 方案B：使用 torch.cond 等编译友好的控制流
```

**启用方式**（暂不推荐）：
```bash
--use_torch_compile
```

### 2. 潜在优化方向

| 优化项 | 说明 | 预期收益 |
|--------|------|----------|
| **KV Cache 优化** | 减少 RAR GPT Generate 显存 | 降低推理显存 |
| **Gradient Checkpointing** | 换时间换空间 | 支持更大 batch |
| **Flash Attention** | 加速 Transformer | 训练加速 20-30% |
| **DeepSpeed/FSDP** | 分布式训练 | 支持多卡 |

### 3. Decoder 架构调优

**当前配置**：
```python
ch_mult=[1, 2, 4]  # 3层上采样
base_channels=64
```

**可选配置**：
```python
# 更强 Decoder（更多参数）
ch_mult=[1, 2, 4, 4]  # 4层
base_channels=128

# 更轻量 Decoder（更快训练）
ch_mult=[1, 2, 4]
base_channels=32
```

---

## 文件结构

```
VisionTSRAR/
├── visiontsrar/
│   ├── model.py                    # VisionTSRAR 主模型
│   ├── models_rar.py               # RARWrapper 封装
│   └── randar/
│       ├── randar_gpt.py           # RAR GPT 实现
│       ├── tokenizer.py            # VQ Tokenizer
│       ├── lightweight_decoder.py  # 轻量级 Decoder
│       └── llamagen_gpt.py         # KV Cache 实现
├── long_term_tsf/
│   ├── run.py                      # 训练入口
│   ├── models/
│   │   └── VisionTSRAR.py          # 模型适配层
│   ├── exp/
│   │   ├── exp_basic.py            # 基础实验类
│   │   └── exp_long_term_forecasting.py  # 训练循环
│   └── utils/
│       └── tools.py                # EarlyStopping, Checkpoint
├── scripts/
│   ├── local_test.py               # 本地验证脚本
│   ├── train_commands.md           # 训练命令文档
│   └── train_balanced.md           # 平衡版训练文档
├── README.md                       # 本文档
└── CHANGELOG.md                    # 改动记录
```

---

## 常见问题

### Q1: 训练 loss 下降但测试效果差？

**原因**：训练时使用 Teacher Forcing，测试时使用 Generate，分布不一致。

**解决**：使用 Schedule Sampling（已实现）。

### Q2: Checkpoint 太大？

**原因**：保存了所有参数（包括冻结的 RAR GPT）。

**解决**：只保存可训练参数（已实现，~38MB）。

### Q3: 显存不足？

**解决方案**：
1. 使用 `--use_amp` 混合精度
2. 减小 `--batch_size`
3. 使用 `--skip_validation 1`

### Q4: 如何确认 Decoder 在训练？

**验证**：
```python
# 检查梯度
for name, param in model.named_parameters():
    if param.requires_grad and param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm().item():.6f}")
```

---

## 更新日志

### 2026-04-17

1. **Schedule Sampling**：解决 Exposure Bias 问题
2. **轻量 Decoder**：3层架构，~10M 参数
3. **torch.compile 支持**：已实现但暂不可用
4. **修复**：`current_epoch` 参数传递链

### 2026-04-13

1. Loss 从 CE + MSE 改为纯 MAE + MSE
2. RAR GPT 完全冻结（`ft_type=In`）
3. Loss 计算分开：历史 + 未来
4. 添加 `--skip_validation` 参数
