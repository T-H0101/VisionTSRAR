# VisionTSRAR 实现指南

> 本文档详细说明 VisionTSRAR 的实现架构、参数配置、轻量化改造及训练策略。

---

## 目录

1. [项目架构概览](#1-项目架构概览)
2. [核心文件说明](#2-核心文件说明)
   - [2.1 visiontsrar/model.py](#21-visiontsrarmodelpy)
   - [2.2 visiontsrar/models_rar.py](#22-visiontsrarmodels_rarpy)
   - [2.3 visiontsrar/randar/randar_gpt.py](#23-visiontsrarrandarrandar_gptpy)
   - [2.4 visiontsrar/randar/llamagen_gpt.py](#24-visiontsrarrandarllamagen_gptpy)
   - [2.5 visiontsrar/randar/lightweight_decoder.py](#25-visiontsrarrandarlightweight_decoderpy)
   - [2.6 visiontsrar/util.py](#26-visiontsrarutilpy)
   - [2.7 visiontsrar/__init__.py](#27-visiontsrar__init__py)
   - [2.8 visiontsrar/randar/tokenizer.py](#28-visiontsrarrandartokenizerpy)
   - [2.9 long_term_tsf/models/VisionTSRAR.py](#29-long_term_tsfmodelsvisiontsrarpy)
   - [2.10 long_term_tsf/exp/exp_basic.py](#210-long_term_tsfexpexp_basicpy)
   - [2.11 long_term_tsf/exp/exp_long_term_forecasting.py](#211-long_term_tsfexpexp_long_term_forecastingpy)
3. [run.py 参数详解](#3-runpy-参数详解)
4. [轻量化改造](#4-轻量化改造)
5. [冻结策略与梯度管理](#5-冻结策略与梯度管理)
6. [训练流程](#6-训练流程)
7. [关键设计决策](#7-关键设计决策)

---

## 1. 项目架构概览

### 1.1 目录结构

```
VisionTSRAR/
├── visiontsrar/                    # 核心模型实现
│   ├── __init__.py
│   ├── model.py                   # VisionTSRAR 主类（入口）
│   ├── models_rar.py              # RARWrapper（RARGPT + VQTokenizer 封装）
│   ├── pos_embed.py               # 位置编码
│   ├── util.py                    # 工具函数（下载权重等）
│   └── randar/                    # RandAR 核心组件
│       ├── randar_gpt.py          # RandARTransformer 主模型
│       ├── llamagen_gpt.py        # 基础组件（Attention, FFN, KVCache, RMSNorm）
│       ├── generate.py            # 生成逻辑
│       ├── tokenizer.py           # VQ-VAE Tokenizer
│       ├── lightweight_decoder.py  # 轻量级解码器
│       └── utils.py               # 工具函数
├── long_term_tsf/                 # 时间序列训练框架
│   ├── run.py                     # 训练入口
│   ├── exp/                       # 实验管理
│   │   ├── exp_basic.py           # 实验基类
│   │   └── exp_long_term_forecasting.py  # 长期预测实验
│   ├── models/                    # 模型适配器
│   │   └── VisionTSRAR.py         # Time-Series-Library 适配器
│   └── data_provider/             # 数据加载
└── ckpt/                          # 权重存储目录
```

### 1.2 核心组件关系

```
run.py (训练入口)
    ↓
exp_long_term_forecasting.py (实验管理)
    ↓
models/VisionTSRAR.py (适配器)
    ↓
visiontsrar/model.py (VisionTSRAR 主类)
    ↓
visiontsrar/models_rar.py (RARWrapper)
    ├── VQ Tokenizer (tokenizer.py)
    │   ├── Encoder (冻结)
    │   ├── Quantize (冻结)
    │   └── Decoder / LightweightDecoder (可训练)
    └── RandAR Transformer (randar_gpt.py)
        ├── Attention + RoPE
        ├── FFN (SwiGLU)
        └── KV Cache
```

---

## 2. 核心文件说明

### 2.1 visiontsrar/model.py

VisionTSRAR 主模型类，继承自 `nn.Module`。

#### 类：`VisionTSRAR`

**初始化参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `arch` | str | `'rar_l_0.3b'` | RAR Transformer 架构选择，可选 `'rar_l_0.3b'`、`'rar_l_35m'` |
| `finetune_type` | str | `'ln'` | 微调策略：`'ln'`（仅 LayerNorm）、`'In'`（Inpainting 模式）、`'full'`（全训练）等 |
| `ckpt_dir` | str | `'./ckpt/'` | 预训练权重存储目录 |
| `load_ckpt` | bool | `True` | 是否加载预训练权重 |
| `vq_ckpt` | str | `None` | VQ Tokenizer 权重路径（None 则自动下载） |
| `rar_ckpt` | str | `None` | RAR Transformer 权重路径 |
| `num_inference_steps` | int | `88` | 并行解码步数 |
| `position_order` | str | `'random'` | Token 顺序策略：`'random'` 或 `'raster'` |
| `use_lightweight_decoder` | bool | `False` | 是否使用轻量级解码器 |
| `lightweight_decoder_channels` | int | `64` | 轻量解码器基础通道数 |

**关键方法：**

```python
def update_config(self, context_len, pred_len, periodicity, interpolation, norm_const, align_const)
```

配置时序数据的图像渲染参数：
- `context_len`: 输入序列长度（回看窗口）
- `pred_len`: 预测序列长度
- `periodicity`: 周期模式（用于图像布局）
- `interpolation`: 插值模式（`'bilinear'` 或 `'bicubic'`）
- `norm_const`: 归一化常数
- `align_const`: 对齐常数

---

### 2.2 visiontsrar/models_rar.py

RARWrapper 封装层，将 VQ Tokenizer + RandAR Transformer 封装为统一的"图像→图像"接口。

#### 架构配置表 `RAR_ARCH_CONFIG`

```python
RAR_ARCH_CONFIG = {
    "rar_l_0.3b": {          # 预训练模型（约 0.3B 参数）
        "n_layer": 24,        # Transformer 层数
        "n_head": 16,         # 注意力头数
        "n_kv_head": 16,      # KV 头数（GQA）
        "dim": 1024,          # 隐藏维度
        "vocab_size": 16384,  # VQ 码本大小
        "block_size": 256,    # Token 序列长度（16×16）
        "rar_ckpt": "rbrar_l_0.3b_c2i.safetensors",  # 预训练权重文件
    },
    "rar_l_35m": {           # 轻量模型（约 35M 参数）
        "n_layer": 8,        # Transformer 层数
        "n_head": 12,        # 注意力头数
        "n_kv_head": 6,      # KV 头数
        "dim": 768,          # 隐藏维度
        "vocab_size": 16384, # VQ 码本大小
        "block_size": 256,   # Token 序列长度
        "rar_ckpt": None,    # 随机初始化，不加载预训练
    },
}
```

#### 轻量模型参数估算

| 配置 | dim | layers | heads | 参数量估算 |
|------|-----|--------|-------|-----------|
| rar_l_0.3b | 1024 | 24 | 16 | ~300M |
| rar_l_35m | 768 | 8 | 12 | ~35M |

#### 类：`RARWrapper`

**初始化流程：**

```python
def __init__(self, rar_arch, finetune_type, ckpt_dir, load_ckpt, ...):
    # 1. 检查架构配置
    arch_config = RAR_ARCH_CONFIG[rar_arch].copy()

    # 2. 判断是否预训练模型
    self.is_pretrained = arch_config.get('rar_ckpt') is not None

    # 3. 创建 RandARTransformer
    self.rar_gpt = RandARTransformer(**arch_config)

    # 4. 加载预训练权重（仅预训练模型）
    if load_ckpt and self.is_pretrained:
        self._load_rar_ckpt(rar_ckpt_path)

    # 5. 应用冻结策略
    self._apply_finetune_strategy(finetune_type)
```

**关键属性：**

| 属性 | 类型 | 说明 |
|------|------|------|
| `is_pretrained` | bool | 是否为预训练模型（决定冻结策略） |
| `use_lightweight_decoder` | bool | 是否使用轻量解码器 |
| `block_size` | int | Token 序列长度（固定 256） |
| `vq_downsample_ratio` | int | VQ 下采样率（固定 16） |

**关键方法：**

```python
def encode_image(self, image_input: torch.Tensor) -> torch.Tensor
"""
将图像编码为离散 token 索引序列。

Args:
    image_input: [bs, 3, H, W] 输入图像
Returns:
    token_indices: [bs, 256] 离散 token 索引（1D 展平）
"""

def decode_tokens(self, token_indices: torch.Tensor, image_size: int = 256) -> torch.Tensor
"""
将离散 token 索引序列解码为图像。

Args:
    token_indices: [bs, 256] 离散 token 索引
    image_size: 输出图像尺寸（默认 256）
Returns:
    reconstructed_image: [bs, 3, 256, 256] 重建图像，值域 [-1, 1]
"""

def forward(self, image_input, num_visible_tokens, current_epoch, use_teacher_forcing)
"""
训练/推理统一接口。

训练模式：
    1. VQ encode → all_tokens
    2. RAR GPT generate → generated_tokens
    3. VQ decode → reconstructed_image
    4. 计算 MAE + MSE loss

推理模式：
    1. VQ encode（可见区）→ visible_tokens
    2. RAR GPT generate → generated_tokens
    3. VQ decode → reconstructed_image
"""
```

#### 冻结策略 `_apply_finetune_strategy()`

```python
def _apply_finetune_strategy(self, finetune_type):
    # 轻量模型（随机初始化）：全部可训练
    if not self.is_pretrained:
        for param in self.rar_gpt.parameters():
            param.requires_grad = True
        return

    # 预训练模型：根据 finetune_type 决定冻结策略
    if finetune_type == 'In':
        # Inpainting 模式：冻结 RAR GPT，训练 VQ Decoder
        for param in self.rar_gpt.parameters():
            param.requires_grad = False
        # VQ Decoder 可训练
        ...
```

#### `_forward_train()` 训练前向传播

```python
def _forward_train(self, image_resized, all_tokens, image_input, num_visible_tokens, vq_input_size, current_epoch):
    """
    训练模式前向传播。
    核心：100% generate，不使用 teacher forcing。
    """
    bs = all_tokens.shape[0]
    total_tokens = all_tokens.shape[1]
    query_len = total_tokens - num_visible_tokens

    # 生成全部 query tokens
    with torch.no_grad():
        generated_tokens = self.rar_gpt.generate(
            cond=cond_idx,
            token_order="random",       # 训练用 random（泛化能力）
            visible_tokens=visible_tokens,
            max_new_tokens=query_len,  # 生成全部 query tokens
            num_inference_steps=44,    # 训练步数（加速）
            kv_window_size=64,          # KV window 限制
        )

    # VQ decode
    recon_image = self.decode_tokens(generated_tokens, image_size=vq_input_size)

    # 计算 MAE + MSE loss（历史 + 未来）
    loss_mae = F.l1_loss(history_recon, history_input) + F.l1_loss(future_recon, future_input)
    loss_mse = F.mse_loss(history_recon, history_input) + F.mse_loss(future_recon, future_input)
    loss = loss_mae + loss_mse

    return recon_image, loss
```

---

### 2.3 visiontsrar/randar/randar_gpt.py

RandARTransformer 核心模型实现。

#### 类：`RandARTransformer`

**初始化参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dim` | int | 4096 | 隐藏维度 |
| `n_layer` | int | 32 | Transformer 层数 |
| `n_head` | int | 32 | 注意力头数 |
| `n_kv_head` | int | None | KV 头数（用于 GQA，None 则等于 n_head） |
| `vocab_size` | int | 16384 | VQ 码本大小 |
| `block_size` | int | 256 | Token 序列长度（16×16） |
| `rope_base` | float | 10000 | RoPE 基础频率 |
| `drop_path_rate` | float | 0.0 | DropPath 概率 |
| `num_inference_steps` | int | 88 | 推理步数 |
| `position_order` | str | 'random' | Token 顺序策略 |

**架构组成：**

```python
# 条件嵌入（类别条件）
self.cls_embedding = LabelEmbedder(num_classes, dim)

# Token 嵌入
self.tok_embeddings = nn.Embedding(vocab_size, dim)

# Transformer 层
self.layers = torch.nn.ModuleList([
    TransformerBlock(dim, n_head, n_kv_head, ...)
    for _ in range(n_layer)
])

# 输出层
self.norm = RMSNorm(dim)
self.output = nn.Linear(dim, vocab_size, bias=False)

# 位置指令嵌入
self.pos_instruct_embeddings = nn.Parameter(...)  # 256 × dim

# 2D RoPE 频率
self.freqs_cis = precompute_freqs_cis_2d(grid_size, dim // n_head, rope_base)
```

**关键方法：**

```python
def setup_caches(self, max_batch_size, max_seq_length, dtype, kv_window_size=None)
"""
设置 KV Cache，为推理做准备。

Args:
    max_batch_size: 最大批次大小
    max_seq_length: 最大序列长度
    dtype: 缓存数据类型
    kv_window_size: KV window 大小（None 表示不限制）
"""

def generate(self, cond, token_order=None, cfg_scales=(1.0, 1.0),
             num_inference_steps=88, temperature=1.0, top_k=0, top_p=1.0,
             visible_tokens=None, max_new_tokens=None, kv_window_size=None)
"""
RandAR 并行解码生成函数。

核心流程：
    1. 准备 token 顺序和位置指令
    2. 处理 CFG（可选）
    3. 设置 KV Cache
    4. 预填充：输入条件 token
    5. 循环解码（并行解码多 token）
    6. 逆排列恢复光栅顺序

Args:
    cond: [bsz, cls_token_num] 条件 token
    token_order: [bsz, block_size] 或 "raster"/"random" 每个 token 的位置顺序
    cfg_scales: (起始CFG, 结束CFG) 线性插值
    num_inference_steps: 推理步数
    temperature: 采样温度
    visible_tokens: [bsz, num_visible] 已知 token（inpainting 模式）
    max_new_tokens: 最多生成的新 token 数
    kv_window_size: KV window 大小

Returns:
    result_indices: [bsz, block_size] 生成的 token 索引（光栅顺序）
"""
```

---

### 2.4 visiontsrar/randar/llamagen_gpt.py

基础组件实现。

#### 类：`RMSNorm`

均方根归一化（LLaMA 风格）：

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps) * self.weight
```

#### 类：`FeedForward`

SwiGLU 前馈网络（LLaMA 风格）：

```python
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_dim_multiplier, multiple_of, ffn_dropout_p):
        hidden_dim = 4 * dim
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # 门控路径
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # 值路径
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # 下投影

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

#### 类：`KVCache`

键值缓存（加速推理）：

```python
class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype, kv_window_size=None):
        self.register_buffer("k_cache", torch.zeros(...))
        self.register_buffer("v_cache", torch.zeros(...))
        self.kv_window_size = kv_window_size  # KV window 大小

    def update(self, input_pos, k_val, v_val):
        """更新 KV Cache 并返回完整 Key/Value"""
        # 如果设置 kv_window_size，只保留最近 K 个 token
```

---

### 2.5 visiontsrar/randar/lightweight_decoder.py

轻量级 VQ 解码器，用于降低显存占用。

#### 类：`LightweightDecoder`

**设计目标：** 在 4090 24G 显卡上将 batch_size 从 2 提升到更大值。

**核心优化策略：**

| 特性 | 原版 Decoder | 轻量级 Decoder |
|------|-------------|----------------|
| base_channels | 128 | 64（或更小） |
| ch_mult | [1,1,2,2,4] (5层) | [1,2,4] (3层) |
| num_res_blocks | 3 | 2 |
| 注意力 | 每层都有 | 仅最后层（可选） |
| 参数量 | ~45M | ~8M（↓82%） |

**初始化参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `z_channels` | int | 256 | 潜空间通道数 |
| `base_channels` | int | 64 | 基础通道数 |
| `ch_mult` | list | [1, 2, 4] | 每层通道倍数 |
| `num_res_blocks` | int | 2 | 每层残差块数量 |
| `use_attn_last_only` | bool | True | 是否仅在最后层使用注意力 |
| `use_depthwise` | bool | False | 是否使用深度可分离卷积 |

**架构：**

```
Input (256, 16×16)
  └─ conv_in (256→256)
  └─ MidBlock × 2 (ResBlock × 2)
  └─ [ResBlock×2 + Upsample] × 3  (通道: 256→128→64)
  └─ final_upsample (bilinear 2x)
  └─ Conv (64→3)
Output: (3, 256×256)
```

**集成方式：**

```python
# models_rar.py 中
if self.use_lightweight_decoder:
    self._setup_lightweight_decoder()

def _setup_lightweight_decoder(self):
    self.lightweight_decoder = LightweightDecoder(
        z_channels=256,
        base_channels=self.lightweight_decoder_channels,  # 默认 64
        ch_mult=[1, 2, 4],
        num_res_blocks=2,
    )
    # 直接替换 vq_tokenizer.decoder
    self.vq_tokenizer.decoder = self.lightweight_decoder
```

---

## 3. run.py 参数详解

### 3.1 基础参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--task_name` | str | 必需 | 任务类型，如 `'long_term_forecast'` |
| `--is_training` | int | 必需 | 1=训练模式，0=推理模式 |
| `--model_id` | str | 必需 | 实验 ID，用于标识实验 |
| `--model` | str | 必需 | 模型名称，VisionTSRAR 使用 `'VisionTSRAR'` |
| `--save_dir` | str | `'.'` | 保存目录 |

### 3.2 数据参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data` | str | 必需 | 数据集类型，如 `'ETTh1'`、`'ETTm1'`、`'ECL'` 等 |
| `--root_path` | str | `'./data/ETT/'` | 数据根目录 |
| `--data_path` | str | `'ETTh1.csv'` | 数据文件名 |
| `--features` | str | `'M'` | 特征类型：`'M'`（多变量）、`'S'`（单变量）、`'MS'`（多变量输出单变量） |
| `--target` | str | `'OT'` | 目标特征（仅 S/MS 模式） |
| `--freq` | str | `'h'` | 时间特征编码频率：`'h'`（小时）、`'t'`（分钟）等 |
| `--seasonal_patterns` | str | `'Monthly'` | M4 季节模式 |

### 3.3 序列长度参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--seq_len` | int | `96` | 输入序列长度（回看窗口） |
| `--label_len` | int | `48` | 解码器起始 token 长度 |
| `--pred_len` | int | `96` | 预测序列长度 |

### 3.4 模型结构参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enc_in` | int | `7` | 编码器输入维度（变量数） |
| `--dec_in` | int | `7` | 解码器输入维度 |
| `--c_out` | int | `7` | 输出维度 |
| `--d_model` | int | `512` | Transformer 模型维度 |
| `--n_heads` | int | `8` | 注意力头数 |
| `--e_layers` | int | `2` | 编码器层数 |
| `--d_layers` | int | `1` | 解码器层数 |
| `--d_ff` | int | `2048` | FFN 隐藏维度 |
| `--factor` | int | `1` | 注意力因子 |
| `--dropout` | float | `0.1` | Dropout 概率 |
| `--activation` | str | `'gelu'` | 激活函数 |

### 3.5 VisionTSRAR 特定参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--vm_pretrained` | int | `1` | 是否加载 VisionMAE 预训练权重 |
| `--vm_ckpt` | str | `"../ckpt/"` | VisionMAE 权重目录 |
| `--vm_arch` | str | `'mae_base'` | VisionMAE 架构 |
| `--ft_type` | str | `'ln'` | 微调策略 |
| `--periodicity` | int | `0` | 周期模式（用于图像布局） |
| `--interpolation` | str | `'bilinear'` | 插值模式 |
| `--norm_const` | float | `0.4` | 归一化常数 |
| `--align_const` | float | `0.4` | 对齐常数 |

### 3.6 RAR 模型参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--rar_arch` | str | `'rar_l_0.3b'` | RAR Transformer 架构：`'rar_l_0.3b'`（预训练）或 `'rar_l_35m'`（轻量） |
| `--vq_ckpt` | str | `None` | VQ Tokenizer 权重路径（None 自动下载） |
| `--rar_ckpt` | str | `None` | RAR Transformer 权重路径 |
| `--num_inference_steps` | int | `88` | 并行解码步数（训练用 44，推理用 88） |
| `--position_order` | str | `'random'` | Token 顺序：`'random'`（训练）或 `'raster'`（推理） |
| `--temperature` | float | `1.0` | 生成温度 |
| `--rar_top_k` | int | `0` | Top-k 采样 |
| `--rar_top_p` | float | `1.0` | Top-p 采样 |
| `--cfg_scales` | str | `'1.0,1.0'` | CFG 缩放因子 |

### 3.7 轻量解码器参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use_lightweight_decoder` | flag | `False` | 是否使用轻量级解码器 |
| `--lightweight_decoder_channels` | int | `64` | 轻量解码器基础通道数（越小越轻量） |

### 3.8 训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--train_epochs` | int | `10` | 训练轮数 |
| `--batch_size` | int | `32` | 批次大小 |
| `--gradient_accumulation_steps` | int | `1` | 梯度累积步数（等效增大 batch_size） |
| `--learning_rate` | float | `0.0001` | 学习率 |
| `--lradj` | str | `'type1'` | 学习率调整策略：`'cosine'`、`'type1'`、`'type2'`、`'fixed'` |
| `--use_amp` | flag | `False` | 是否使用自动混合精度训练 |
| `--patience` | int | `3` | 早停耐心值 |
| `--num_workers` | int | `10` | DataLoader 工作进程数 |
| `--itr` | int | `1` | 实验重复次数 |
| `--des` | str | `'test'` | 实验描述 |

### 3.9 调试与快速测试参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--fast_train_batches` | int | `0` | 快速训练模式（仅训练前 N 批，0=禁用） |
| `--fast_eval_batches` | int | `0` | 快速评估模式（仅评估前 N 批） |
| `--skip_validation` | int | `0` | 跳过验证阶段 |
| `--skip_test` | int | `0` | 跳过测试阶段 |
| `--test_freq` | int | `1` | 每 N 轮测试一次（0=仅最后测试） |

---

## 4. 轻量化改造

### 4.1 轻量 Transformer 骨干 (rar_l_35m)

**目标：** 替代 0.3B 预训练模型，减少训练时间和显存占用。

**配置对比：**

| 配置项 | rar_l_0.3b（预训练） | rar_l_35m（轻量） |
|--------|---------------------|-------------------|
| dim | 1024 | 768 |
| n_layer | 24 | 8 |
| n_head | 16 | 12 |
| n_kv_head | 16 | 6 |
| 参数量 | ~300M | ~35M |
| 初始化 | 预训练权重 | 随机 |
| 可训练参数 | 仅 Decoder | 全部 |
| 训练时间 | ~37 min/epoch | 预期 ~10-15 min/epoch |
| 显存占用 | ~20GB | 预期 ~8-12GB |

**实现方式：**

1. **添加架构配置**（models_rar.py）：

```python
RAR_ARCH_CONFIG = {
    "rar_l_35m": {
        "n_layer": 8,
        "n_head": 12,
        "n_kv_head": 6,
        "dim": 768,
        "vocab_size": 16384,
        "block_size": 256,
        "rar_ckpt": None,  # 随机初始化
        ...
    },
}
```

2. **注册架构**（model.py）：

```python
RAR_ARCH = {
    "rar_l_0.3b": {...},
    "rar_l_35m": {
        "factory": models_rar.RARWrapper,
        "rar_ckpt": None,
    },
}
```

3. **加载逻辑修改**（models_rar.py）：

```python
# 检测是否为预训练模型
self.is_pretrained = arch_config.get('rar_ckpt') is not None

# 预训练模型：加载权重
if load_ckpt and self.is_pretrained:
    self._load_rar_ckpt(...)

# 轻量模型：随机初始化，全部可训练
else:
    for param in self.rar_gpt.parameters():
        param.requires_grad = True
```

### 4.2 轻量解码器 (LightweightDecoder)

**目标：** 减少 VQ Decoder 的参数量，降低显存占用。

**使用方式：**

```bash
# 启用轻量解码器
--use_lightweight_decoder --lightweight_decoder_channels 64
```

**参数影响：**

| channels | 参数量 | 显存节省 | 重建质量 |
|----------|--------|---------|----------|
| 128 | ~12M | 基准 | 最高 |
| 64 | ~8M | ~30% | 较高 |
| 32 | ~4M | ~50% | 中等 |

---

## 5. 冻结策略与梯度管理

### 5.1 冻结策略概览

| 组件 | rar_l_0.3b（预训练） | rar_l_35m（轻量） |
|------|---------------------|-------------------|
| VQ Encoder | 冻结 | 冻结 |
| VQ Quantize | 冻结 | 冻结 |
| VQ Decoder | 可训练 | 可训练 |
| RAR Transformer | 冻结 | 全部可训练 |

### 5.2 代码实现

**微调策略类型**（models_rar.py）：

```python
finetune_type = {
    'ln':      # 仅 LayerNorm（RMSNorm）可训练
    'bias':    # 仅偏置可训练
    'none':    # 全部冻结
    'full':    # 全部可训练
    'In':      # Inpainting 模式（冻结 GPT，训练 Decoder）
    'In_light': # 轻量 Inpainting（仅训练 post_quant_conv）
}
```

**冻结逻辑**：

```python
def _apply_finetune_strategy(self, finetune_type):
    # ===== VQ Tokenizer 冻结策略 =====
    if finetune_type == 'In':
        if self.use_lightweight_decoder:
            # 冻结整个 VQ，只训练轻量 Decoder
            for param in self.vq_tokenizer.parameters():
                param.requires_grad = False
            for param in self.lightweight_decoder.parameters():
                param.requires_grad = True

    # ===== RAR GPT 冻结策略 =====
    if not self.is_pretrained:
        # 轻量模型：全部可训练
        for param in self.rar_gpt.parameters():
            param.requires_grad = True
    else:
        # 预训练模型：冻结 RAR GPT
        for param in self.rar_gpt.parameters():
            param.requires_grad = False
```

### 5.3 梯度流动

```
输入图像
    ↓
VQ Encoder（冻结，无梯度）
    ↓
VQ Quantize（冻结，无梯度）
    ↓
量化 token
    ↓
RAR Transformer（前向传播，反向传播取决于模型类型）
    ↓
VQ Decoder / LightweightDecoder（可训练，有梯度）
    ↓
重建图像
    ↓
Loss 计算（MAE + MSE）
    ↓
反向传播 → Decoder → Decoder 梯度更新
```

### 5.4 关键设计决策

**为什么冻结 RAR GPT？**
- 预训练的 RAR GPT 已经学会生成合理的 token 序列
- 只需训练 Decoder 将 token 正确解码为图像
- 避免训练整个 0.3B 模型，显著降低训练成本

**为什么轻量模型需要全部可训练？**
- 轻量模型是随机初始化，没有预训练知识
- 如果冻结，模型无法学习任何东西
- 必须全部可训练才能让模型学习生成和重建

---

## 6. 训练流程

### 6.1 端到端流程

```
时序数据 [B, L, D]
    ↓
图像渲染（Segmentation + Render）
    ↓
VQ Encoder → token 序列 [B, 256]
    ↓
Visible tokens [B, 48] + Query tokens [B, 208]
    ↓
RAR GPT generate → 生成 tokens [B, 208]
    ↓
VQ Decoder → 重建图像 [B, 3, 256, 256]
    ↓
图像反渲染 → 预测时序 [B, L, D]
    ↓
Loss 计算（MAE + MSE）
    ↓
反向传播 + 优化器更新
```

### 6.2 训练配置

**标准训练命令（预训练模型）：**

```bash
python run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model VisionTSRAR \
    --data ETTh1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --features M \
    --seq_len 96 --label_len 48 --pred_len 96 \
    --use_lightweight_decoder \
    --lightweight_decoder_channels 64 \
    --ft_type In \
    --use_amp \
    --batch_size 8 \
    --lradj cosine \
    --train_epochs 10 \
    --learning_rate 0.0001 \
    --skip_validation 1
```

**轻量模型训练命令：**

```bash
python run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model VisionTSRAR \
    --rar_arch rar_l_35m \
    --data ETTh1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --features M \
    --seq_len 96 --label_len 48 --pred_len 96 \
    --use_lightweight_decoder \
    --lightweight_decoder_channels 32 \
    --ft_type In \
    --use_amp \
    --batch_size 16 \
    --lradj cosine \
    --train_epochs 5 \
    --learning_rate 0.0002 \
    --skip_validation 1
```

### 6.3 推理流程

```bash
python run.py \
    --is_training 0 \
    --model VisionTSRAR \
    --data ETTh1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --features M \
    --seq_len 96 --label_len 48 --pred_len 96 \
    --use_lightweight_decoder \
    --lightweight_decoder_channels 64 \
    --num_inference_steps 88 \
    --position_order raster
```

---

## 7. 关键设计决策

### 7.1 为什么使用 RAR 而非 MAE？

| 特性 | MAE | RAR |
|------|-----|-----|
| 生成方式 | 一次性解码 | 自回归生成 |
| 训练-推理一致性 | 存在差异 | 一致 |
| Inpainting 能力 | 有限 | 原生支持 |
| 序列建模 | 无 | 有 |

### 7.2 为什么 100% generate？

- **训练-推理一致性**：训练时用 generate，推理时也用 generate，分布匹配
- **误差累积容忍**：时序任务的误差累积是正常的，不需要 teacher forcing 兜底
- **收敛稳定**：100% generate 比混合训练更稳定

### 7.3 为什么训练用 random、推理用 raster？

- **训练 random**：让模型学习任意位置的依赖关系，增强泛化能力
- **推理 raster**：保持空间局部性，保证生成质量

### 7.4 为什么 KV window 限制？

- **显存优化**：只保留最近 K 个 token 的 KV，减少显存占用
- **局部依赖**：图像生成主要依赖局部特征，长距离依赖通过多层堆叠学习

---

### 2.6 visiontsrar/util.py

工具函数模块，包含权重下载和频率转换等。

#### 权重下载

```python
# HuggingFace 仓库配置
RAR_HF_REPO_ID = "ziqipang/RandAR"
VQ_HF_REPO_ID = "FoundationVision/LlamaGen"

def download_rar_ckpt(ckpt_name, ckpt_dir='./')
"""
从 HuggingFace 下载 RAR GPT 预训练权重。

Args:
    ckpt_name: RAR 架构名称（如 "rar_l_0.3b"）
    ckpt_dir: 本地存储目录

Returns:
    ckpt_path: 下载后的本地文件路径
"""

def download_vq_ckpt(ckpt_dir='./')
"""
从 HuggingFace 下载 VQ Tokenizer 预训练权重。

Returns:
    ckpt_path: 下载后的本地文件路径
"""
```

#### 频率转换

```python
# 时间频率→可能的周期性映射
POSSIBLE_SEASONALITIES = {
    "S": [3600],   # 秒级: 1小时
    "T": [1440, 10080],  # 分钟: 1天, 1周
    "H": [24, 168],      # 小时: 1天, 1周
    "D": [7, 30, 365],   # 天: 1周, 1月, 1年
    "W": [52, 4],        # 周: 1年, 1月
    "M": [12, 6, 3],      # 月: 1年, 半年, 1季度
    "B": [5],             # 工作日
    "Q": [4, 2],         # 季度
}

def freq_to_seasonality_list(freq: str, mapping_dict=None) -> int
"""
根据时间序列频率返回可能的周期性列表。

Args:
    freq: 频率字符串（如 "H"、"D"、"M"、"3H"、"15min" 等）

Returns:
    periodicity_list: 可能的周期值列表，如 [24, 168, 1]
"""
```

#### 其他工具

```python
def safe_resize(size, interpolation)
"""
安全的 Resize 构造函数，兼容不同版本 torchvision。

支持 antialias 参数的自动处理。
"""
```

---

### 2.7 visiontsrar/__init__.py

包入口文件，导出公开接口。

```python
from .model import VisionTSRAR, VisionTSRARpp
from .util import freq_to_seasonality_list

__version__ = "0.1.0"
__all__ = ["VisionTSRAR", "VisionTSRARpp", "freq_to_seasonality_list"]
```

---

### 2.8 visiontsrar/randar/tokenizer.py

VQ-VAE Tokenizer，将图像编码为离散 token 并解码重建。

#### VQ-VAE 架构概览

```
输入图像 [B, 3, 256, 256]
    ↓
Encoder (卷积编码器)
    ↓
潜空间特征 [B, 256, 16, 16]
    ↓
quant_conv (256→8)
    ↓
VectorQuantizer (连续→离散)
    ↓
码本索引 [B, 16, 16] → 量化向量 [B, 8, 16, 16]
    ↓
post_quant_conv (8→256)
    ↓
Decoder / LightweightDecoder
    ↓
重建图像 [B, 3, 256, 256]
```

#### 类：`VQModel`

**初始化参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `codebook_size` | int | 16384 | 码本大小（词汇表大小） |
| `codebook_embed_dim` | int | 8 | 码本嵌入维度 |
| `codebook_l2_norm` | bool | True | 是否对码本向量 L2 归一化 |
| `commit_loss_beta` | float | 0.25 | Commitment loss 权重 |
| `encoder_ch_mult` | list | [1,1,2,2,4] | 编码器通道倍数 |
| `decoder_ch_mult` | list | [1,1,2,2,4] | 解码器通道倍数 |
| `z_channels` | int | 256 | 潜空间通道数 |

**核心方法：**

```python
def encode_to_tokens(self, x: torch.Tensor) -> torch.Tensor
"""
将图像编码为 1D token 索引序列（VisionTSRAR 专用接口）。

Args:
    x: 输入图像 [B, 3, H, W]
Returns:
    token_indices: [B, L] 其中 L = (H/16)*(W/16)
    例如 256×256 图像 → L = 16×16 = 256 个 token
"""

def decode_tokens_to_image(self, tokens: torch.Tensor, image_size: int) -> torch.Tensor
"""
将 1D token 索引序列解码为图像（VisionTSRAR 专用接口）。

Args:
    tokens: token 索引序列 [B, L]
    image_size: 输出图像尺寸（如 256）
Returns:
    重建图像 [B, 3, image_size, image_size]，值域 [-1, 1]
"""

def encode(self, x) -> Tuple[quant, emb_loss, info]
"""
完整编码流程：图像 → 量化潜空间表示。

Returns:
    quant: 量化后的潜空间特征 [B, 8, 16, 16]
    emb_loss: VQ 损失元组
    info: 附加信息
"""

def decode(self, quant) -> torch.Tensor
"""
解码：量化潜空间 → 重建图像。

Args:
    quant: [B, 8, 16, 16] 量化特征
Returns:
    重建图像 [B, 3, H, W]
"""
```

#### VectorQuantizer 内部机制

```python
class VectorQuantizer(nn.Module):
    def __init__(self, n_e=16384, e_dim=8, beta=0.25, ...):
        self.embedding = nn.Embedding(n_e, e_dim)  # 16384 × 8 的码本

    def forward(self, z):
        # z: [B, 8, 16, 16] → [B, 16, 16, 8]
        # 计算到所有码本向量的距离
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())

        # 最近邻查找
        min_encoding_indices = torch.argmin(d, dim=1)
        min_encodings = F.one_hot(min_encoding_indices, num_classes=self.n_e)

        # 获取量化向量
        z_q = self.embedding(min_encoding_indices)

        # Straight-Through Estimator (STE)
        # 前向：argmin（离散索引）
        # 反向：梯度直接传给被选中位置
        z_q = z + (z_q - z).detach()

        return z_q, loss, perplexity
```

---

### 2.9 long_term_tsf/models/VisionTSRAR.py

Time-Series-Library 适配器，将 VisionTSRAR 核心模型封装为统一接口。

#### 类：`Model`（VisionTSRAR 适配器）

**初始化参数（来自 config）：**

| config 属性 | 说明 |
|------------|------|
| `rar_arch` | RAR GPT 架构（`'rar_l_0.3b'` 或 `'rar_l_35m'`） |
| `ft_type` | 微调策略（`'ln'`、``'In'` 等） |
| `vm_ckpt` | 权重文件目录 |
| `vm_pretrained` | 是否加载预训练（1/0） |
| `vq_ckpt` | VQ Tokenizer 权重路径 |
| `rar_ckpt` | RAR GPT 权重路径 |
| `num_inference_steps` | 推理步数 |
| `position_order` | Token 顺序 |
| `use_lightweight_decoder` | 是否使用轻量解码器 |
| `lightweight_decoder_channels` | 轻量解码器通道数 |
| `seq_len` | 输入序列长度 |
| `pred_len` | 预测序列长度 |
| `periodicity` | 周期长度 |
| `interpolation` | 插值模式 |
| `norm_const` | 归一化常数 |
| `align_const` | 对齐常数 |

**核心属性：**

```python
self.vm = VisionTSRAR(...)  # VisionTSRAR 核心模型
self.pred_len = config.pred_len
self.seq_len = config.seq_len
self.temperature = config.temperature
self.cfg_scales = (start_scale, end_scale)  # CFG 参数
```

**核心方法：**

```python
def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask, current_epoch, use_teacher_forcing)
"""
统一前向接口，根据任务类型分发到对应方法。

处理流程：
    1. 如果是 forecast 任务，调用 self.forecast()
    2. 训练时返回 (prediction, loss)
    3. 测试时返回 prediction

Args:
    x_enc: [B, L, D] 编码器输入
    x_dec: [B, L, D] 解码器输入（VisionTSRAR 不使用）
    current_epoch: 当前 epoch（用于 schedule sampling）
    use_teacher_forcing: 是否使用 teacher forcing

Returns:
    训练时: (prediction[:, -pred_len:, :], loss)
    测试时: prediction[:, -pred_len:, :]
"""

def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, current_epoch, use_teacher_forcing)
"""
时序预测入口。

直接调用 self.vm.forward()，VisionTSRAR 内部处理：
    训练时：forward() → _forward_train() → generate → decode → loss
    测试时：forward() → generate() → decode → prediction
"""
```

**任务分发逻辑：**

```python
def forward(self, ...):
    if self.task_name == 'long_term_forecast':
        dec_out = self.forecast(...)
        if isinstance(dec_out, tuple):
            pred, loss = dec_out
            return pred[:, -self.pred_len:, :], loss
        return dec_out[:, -self.pred_len:, :]
    elif self.task_name == 'imputation':
        return self.imputation(...)
    elif self.task_name == 'anomaly_detection':
        return self.anomaly_detection(...)
    elif self.task_name == 'classification':
        return self.classification(...)
```

---

### 2.10 long_term_tsf/exp/exp_basic.py

实验基类，管理模型注册和设备分配。

#### 类：`Exp_Basic`

**model_dict 模型注册：**

```python
class Exp_Basic(object):
    def __init__(self, args):
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            'VisionTSRAR': VisionTSRAR,  # ← 注册在这里
        }
```

**设备分配 `_acquire_device()`：**

```python
def _acquire_device(self):
    if self.args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
        device = torch.device(f'cuda:{self.args.gpu}')
    else:
        device = torch.device('cpu')
```

**torch.compile 加速（可选）：**

```python
if getattr(args, 'use_torch_compile', False):
    self.model = torch.compile(self.model)
```

---

### 2.11 long_term_tsf/exp/exp_long_term_forecasting.py

长期预测实验类，管理训练、验证、测试流程。

#### 类：`Exp_Long_Term_Forecast`

**初始化：**

```python
def __init__(self, args):
    super().__init__(args)
    self._is_visiontsrar = (args.model == 'VisionTSRAR')

    # VisionTSRAR 自动配置 periodicity
    if self._is_visiontsrar and getattr(args, 'periodicity', 0) == 0:
        args.periodicity = self._infer_periodicity(args.freq)
```

**periodicity 自动推断：**

```python
@staticmethod
def _infer_periodicity(freq):
    freq_map = {
        'h': 24,        # 小时 → 日周期
        'min': 96,      # 15min间隔 → 日周期
        't': 96,        # minutely 同上
        'd': 7,         # 天 → 周周期
        'b': 5,         # 工作日 → 周期5
        'w': 52,        # 周 → 年周期
        'm': 12,        # 月 → 年周期
        'q': 4,         # 季度 → 年周期
        '10min': 144,   # 10min → 日周期
        '15min': 96,
        '30min': 48,
        's': 86400,     # 秒 → 日周期
    }
```

**模型构建 `_build_model()`：**

```python
def _build_model(self):
    # 从 model_dict 获取适配器类，调用其 Model() 构造方法
    model = self.model_dict[self.args.model].Model(self.args).float()

    # 多 GPU 支持
    if self.args.use_multi_gpu and self.args.use_gpu:
        model = nn.DataParallel(model, device_ids=self.args.device_ids)
    return model
```

**优化器选择 `_select_optimizer()`：**

```python
def _select_optimizer(self):
    model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
    return model_optim
```

**损失函数选择 `_select_criterion()`：**

```python
def _select_criterion(self):
    criterion = nn.MSELoss()
    return criterion
```

**前向传播处理 `_model_forward()`：**

```python
def _model_forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark,
                  current_epoch=0, use_teacher_forcing=None):
    """
    统一前向调用接口，处理不同模型的输出格式。

    关键逻辑：
    - VisionTSRAR 训练时返回 (prediction, loss) 元组
    - 其他模型返回 prediction

    Returns:
        outputs: 模型预测输出
        model_loss: VisionTSRAR 的内部 loss（其他模型为 None）
    """
    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                         current_epoch=current_epoch, use_teacher_forcing=use_teacher_forcing)

    # VisionTSRAR 训练时拆包
    if self._is_visiontsrar and isinstance(outputs, tuple):
        outputs, model_loss = outputs
        return outputs, model_loss

    return outputs, None
```

**训练流程 `train()`：**

```python
def train(self, setting):
    train_data, train_loader = self._get_data(flag='train')
    vali_data, vali_loader = self._get_data(flag='val')
    test_data, test_loader = self._get_data(flag='test')

    model_optim = self._select_optimizer()
    criterion = self._select_criterion()

    # 梯度累积
    accumulation_steps = getattr(self.args, 'gradient_accumulation_steps', 1)

    for epoch in range(self.args.train_epochs):
        self.model.train()

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            # 梯度清零（仅在累积开始时）
            if i % accumulation_steps == 0:
                model_optim.zero_grad()

            # 数据移到 GPU
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            # 前向传播
            outputs, model_loss = self._model_forward(...)

            # 损失计算
            if model_loss is not None:
                loss = model_loss  # VisionTSRAR 内部 loss
            else:
                loss = criterion(outputs, batch_y)

            # 梯度累积：loss 除以累积步数
            loss = loss / accumulation_steps

            # 反向传播
            if self.args.use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # 参数更新（仅在累积完成时）
            if (i + 1) % accumulation_steps == 0:
                if self.args.use_amp:
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    model_optim.step()
```

**验证流程 `vali()`：**

```python
def vali(self, vali_data, vali_loader, criterion):
    self.model.eval()
    total_loss = []

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
            # 前向传播
            outputs, model_loss = self._model_forward(...)

            # 损失计算
            if model_loss is not None:
                loss = model_loss
            else:
                loss = criterion(outputs.detach(), batch_y)

            total_loss.append(loss.item())

    avg_loss = np.average(total_loss)
    self.model.train()
    return avg_loss
```

**测试流程 `test()`：**

```python
def test(self, setting, test_data, test_loader):
    self.model.eval()

    preds = []
    trues = []

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            # 推理时不需要 loss
            outputs, _ = self._model_forward(...)
            preds.append(outputs)
            trues.append(batch_y)

    # 计算指标
    mae, mse,rmse, mape, mspe = metric(np.array(preds), np.array(trues))
```

---

## 附录：参数速查表

### 模型切换

| 目标 | 命令参数 |
|------|---------|
| 预训练模型 | `--rar_arch rar_l_0.3b` |
| 轻量模型 | `--rar_arch rar_l_35m` |

### 解码器切换

| 目标 | 命令参数 |
|------|---------|
| 原版解码器 | 不加 `--use_lightweight_decoder` |
| 轻量解码器 | `--use_lightweight_decoder --lightweight_decoder_channels 32/64` |

### 微调策略

| 策略 | 命令参数 | 可训练部分 |
|------|---------|-----------|
| Inpainting | `--ft_type In` | Decoder |
| 轻量 Inpainting | `--ft_type In_light` | post_quant_conv |
| 全训练 | `--ft_type full` | 全部 |

---

> 文档版本：v1.0（2026-04-20）
