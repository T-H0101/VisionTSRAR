# VisionTSRAR 迁移对比文档

本文档梳理 VisionTS → VisionTSRAR 的完整迁移对照，包括模型类、数据集适配、权重来源、架构差异等。

## 1. 项目总览

| 维度 | VisionTS（原始） | VisionTSRAR（新增） |
|------|------------------|---------------------|
| 视觉骨干 | MAE (Encoder-Decoder, 单次前向) | RAR GPT (自回归 Decoder-only, 迭代生成) |
| 图像表示 | 连续像素值 (RGB, 3通道) | 离散 token 索引 (codebook_size=16384) |
| 掩码补全 | 固定掩码，左可见右掩码 | 自回归生成右侧 token 序列 |
| 推理复杂度 | O(1) 前向传播 | O(num_output_tokens) 自回归步数 |
| 预训练权重 | Facebook MAE ImageNet | RandAR ImageNet + VQ Tokenizer |

## 2. 架构对比

### 2.1 6步流水线对比

VisionTSRAR 继承 VisionTS 的6步流水线，仅第4步不同：

| 步骤 | VisionTS | VisionTSRAR | 变化程度 |
|------|----------|-------------|----------|
| 1. Normalization | min-max归一化 | min-max归一化 | ✅ 完全一致 |
| 2. Segmentation | 周期折叠切分 | 周期折叠切分 | ✅ 完全一致 |
| 3. Render & Alignment | 渲染为224×224灰度图 | 渲染为224×224灰度图 | ✅ 完全一致 |
| 4. **Reconstruction** | **MAE.forward(image, mask)** | **VQ encode → RAR GPT → VQ decode** | ❌ **完全重写** |
| 5. Forecasting | 从重建图像提取预测值 | 从重建图像提取预测值 | ✅ 完全一致 |
| 6. Denormalization | 反归一化 | 反归一化 | ✅ 完全一致 |

### 2.2 第4步详细对比

**原始 MAE 流程：**
```
image_input [bs, 3, 224, 224]
  → MAE.forward(image_input, fixed_mask)  # 单次前向传播
  → reconstructed_image [bs, 3, 224, 224]  # 直接输出重建图像
```

**新 RAR 流程：**
```
image_input [bs, 3, 224, 224]
  → VQTokenizer.encode_indices(image_input)  # 图像→离散token [bs, 256]
  → visible_tokens = left_half_tokens         # 左半部分作为条件
  → RandAR.generate(visible_tokens)           # 自回归生成全部token
  → all_generated_tokens [bs, 256]            # 包含完整的256个token
  → VQTokenizer.decode_tokens_to_image(tokens) # token→重建图像
  → reconstructed_image [bs, 3, 224, 224]
```

## 3. 文件对照表

### 3.1 核心库 (visionts/ → visiontsrar/)

| 文件 | VisionTS | VisionTSRAR | 说明 |
|------|----------|-------------|------|
| `__init__.py` | 导出 VisionTS, VisionTSpp | 导出 VisionTSRAR, VisionTSRARpp | 模型类名变更 |
| `model.py` | VisionTS + VisionTSpp | VisionTSRAR + VisionTSRARpp | 第4步 MAE→RAR |
| `models_mae.py` | MAE 模型封装 | ❌ 不再需要 | 替换为 models_rar.py |
| `models_rar.py` | ❌ 无 | RARWrapper 封装层 | **新增**：VQ+GPT统一接口 |
| `pos_embed.py` | 2D正弦-余弦位置编码 | 2D正弦-余弦位置编码 | ✅ 直接复用 |
| `util.py` | 下载、Resize、频率映射 | 下载、Resize、频率映射 + RAR/VQ权重下载 | 新增下载函数 |

### 3.2 RandAR 核心组件 (visiontsrar/randar/) — 全部新增

| 文件 | 来源 | 说明 |
|------|------|------|
| `__init__.py` | 新建 | 包导入导出 |
| `randar_gpt.py` | RandAR/model/randar_gpt.py | 核心GPT模型，**改造generate()支持inpainting** |
| `tokenizer.py` | RandAR/model/tokenizer.py | VQ变分自编码器，**新增便捷接口** |
| `llamagen_gpt.py` | RandAR/model/llamagen_gpt.py | 基础组件(RMSNorm, FFN, RotaryEmb等) |
| `generate.py` | RandAR/model/generate.py | 采样逻辑(top_k/top_p, parallel decode) |
| `utils.py` | RandAR/model/utils.py | 工具函数(DropPath, interleave_tokens等) |

### 3.3 实验框架 (long_term_tsf/)

| 文件 | 修改内容 |
|------|----------|
| `run.py` | 新增 RAR 专用命令行参数 |
| `models/VisionTSRAR.py` | **新增**：TSL框架适配器 |
| `exp/exp_basic.py` | 注册 VisionTSRAR 模型 |
| `exp/exp_long_term_forecasting.py` | 处理 RAR 的训练loss和推理generate |
| 其他文件 | 直接复用，无修改 |

## 4. 新增命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--rar_arch` | `rar_l_0.3b` | RAR GPT 架构选择 |
| `--vq_ckpt` | `None` | VQ Tokenizer 权重路径 |
| `--rar_ckpt` | `None` | RAR GPT 权重路径 |
| `--num_inference_steps` | `88` | RAR 推理步数 |
| `--position_order` | `raster` | token 位置顺序 |
| `--temperature` | `1.0` | RAR 采样温度 |
| `--top_k` | `0` | Top-k 采样 |
| `--top_p` | `1.0` | Top-p 采样 |
| `--cfg_scales` | `1.0,1.0` | CFG 缩放因子 |

## 5. 权重来源

| 权重 | HuggingFace 路径 | 文件名 | 大小 |
|------|-----------------|--------|------|
| VQ Tokenizer | `yucornell/RandAR` | `vq_ds16_c2i.pt` | ~200MB |
| RandAR GPT 0.3B | `yucornell/RandAR` | `rbrar_l_0.3b_c2i.safetensors` | ~600MB |

权重下载方式：
1. **自动下载**：首次运行时自动从 HuggingFace 下载
2. **手动下载**：`python ckpt/download_ckpt.py`
3. **huggingface-cli**：`huggingface-cli download yucornell/RandAR <filename> --local-dir ckpt/`

## 6. 关键设计决策

### 6.1 RARWrapper 封装层

`models_rar.py` 中的 `RARWrapper` 是核心新增组件，将 VQ Tokenizer + RandAR GPT 封装为统一的 "图像→图像" 接口：

```python
# 训练模式
reconstructed_image, loss = rar_wrapper(image_input, num_visible_tokens)

# 推理模式
reconstructed_image = rar_wrapper.generate(image_input, num_visible_tokens)
```

这种设计使得 `model.py` 中除了第4步调用方式不同外，其余5步可以完全保持与 VisionTS 一致。

### 6.2 Token 顺序选择

选择 **raster（光栅顺序）** 而非 random（随机顺序），原因：
- 时序预测需要空间连续性：左半部分是输入，右半部分是预测目标
- 光栅顺序保证了"先左后右"的自然空间顺序
- 随机顺序会破坏时序的空间结构

### 6.3 类别条件处理

RAR 原本使用 ImageNet 类别标签作为条件（class-to-image），时序预测不需要类别条件：
- 使用固定值 `cond_idx=0` 作为条件输入
- 通过 `LabelEmbedder` 的 dropout 机制处理无条件生成

### 6.4 微调策略

与 VisionTS 一致，采用 **仅微调 LayerNorm** 的策略：
- VQ Tokenizer：完全冻结
- RandAR GPT：仅 LayerNorm/RMSNorm 参数可训练
- 其他参数全部冻结，保持预训练知识

## 7. 性能预期

| 指标 | VisionTS (MAE) | VisionTSRAR (RAR) |
|------|----------------|-------------------|
| 单次前向推理 | ~0.05s (GPU) | ~5-20s (GPU, 取决于steps) |
| 显存占用 | ~2GB | ~6GB (0.3B模型) |
| 训练速度 | 快 | 慢约3-5x (teacher forcing) |
| 预测精度 | 基线 | 待验证 |

> ⚠️ RAR 的自回归特性使推理速度比 MAE 慢约 10-50 倍，这是架构特性的固有代价，需要在精度和速度之间权衡。
