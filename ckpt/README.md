# Ckpt 目录结构说明

## 📂 当前目录结构

```
VisionTSRAR/
├── ckpt/                          # ✅ 根目录 ckpt（预训练权重目录）
│   ├── download_ckpt.py           # 预训练权重下载脚本
│   ├── README.md                  # 本说明文档
│   ├── randar_0.3b_*.safetensors  # RAR GPT 预训练权重 (~600MB)
│   └── vq_ds16_c2i.pt             # VQ Tokenizer 预训练权重 (~100MB)
│
└── long_term_tsf/
    ├── run.py                     # 训练/测试入口
    ├── models/
    │   ├── VisionTS.py            # VisionTS 模型适配器
    │   └── VisionTSRAR.py         # VisionTSRAR 模型适配器
    └── ckpt/                      # ⚠️ 已删除重复的预训练权重
        └── (此目录已删除)
```

## 🎯 目录用途说明

### 1. 根目录 `ckpt/` - 预训练权重目录

**用途**：
- 存放**预训练权重**（Pre-trained Weights）
- 这些权重是在大规模数据集（如 ImageNet）上预先训练好的
- 用于**迁移学习**，提供强大的特征提取能力

**包含文件**：
- `randar_0.3b_*.safetensors` - RAR GPT 预训练权重（替代 MAE）
- `vq_ds16_c2i.pt` - VQ Tokenizer 预训练权重
- （如果使用 VisionTS）还会有 MAE 预训练权重

**特点**：
- ✅ **跨任务共享**：多个任务（长短期预测、填补、异常检测等）可以共享同一份预训练权重
- ✅ **跨数据集共享**：不同数据集（ETTm1、ETTh1、Weather 等）使用相同的预训练权重
- ✅ **集中管理**：由 `download_ckpt.py` 统一下载和管理
- ✅ **平台无关**：可以在 macOS、Windows、Linux 之间直接迁移

### 2. `long_term_tsf/ckpt/` - 已删除 ⚠️

**原本用途**：
- 存放**训练好的时序模型 checkpoint**
- 例如：在 ETTm1 数据集上训练 10 个 epoch 后的 VisionTSRAR 模型

**问题**：
- ❌ 错误地存放了预训练权重（与根目录 `ckpt/` 重复）
- ❌ 浪费存储空间（每份权重 ~700MB）
- ❌ 管理混乱（多个目录存放相同文件）

**解决方案**：
- ✅ 已删除此目录中的预训练权重
- ✅ 修改 `run.py` 和 `VisionTSRAR.py` 中的默认路径从 `./ckpt/` 改为 `../ckpt/`
- ✅ 此目录现在可以重命名为 `checkpoints/` 或 `saved_models/`，用于保存训练好的模型

## 🔄 权重流向

```
预训练权重（根目录 ckpt/）
    ↓
    │ 下载（download_ckpt.py）
    ↓
VisionTSRAR 模型加载
    ↓
    │ 在特定数据集上训练
    ↓
训练好的模型 checkpoint（long_term_tsf/checkpoints/）
    ↓
    │ 保存最佳模型
    ↓
推理/测试时加载
```

## 📝 修改记录

### 2024-XX-XX - 修复重复权重问题

**修改文件**：
1. `long_term_tsf/run.py`
   - 修改 `--vm_ckpt` 参数默认值：`"./ckpt/"` → `"../ckpt/"`

2. `long_term_tsf/models/VisionTSRAR.py`
   - 修改 `ckpt_dir` 默认值：`'./ckpt/'` → `'../ckpt/'`

3. `long_term_tsf/ckpt/` 目录
   - 删除重复的预训练权重文件
   - 建议重命名为 `checkpoints/` 或 `saved_models/`

## 🚀 使用示例

### 下载预训练权重

```bash
cd VisionTSRAR/ckpt
python download_ckpt.py
```

这会将权重下载到 **根目录的 `ckpt/`**：
```
ckpt/
├── randar_0.3b_*.safetensors
└── vq_ds16_c2i.pt
```

### 训练模型

```bash
cd VisionTSRAR/long_term_tsf
python run.py --task_name long_term_forecast \
              --is_training 1 \
              --model VisionTSRAR \
              --data ETTm1 \
              --seq_len 96 \
              --pred_len 96 \
              --vm_ckpt ../ckpt/ \  # 指向根目录的预训练权重
              --checkpoints ./checkpoints/  # 保存训练好的模型
```

训练完成后，保存的模型结构：
```
long_term_tsf/
├── checkpoints/
│   └── VisionTSRAR_ETTm1/
│       └── checkpoint.pth  # 这是在 ETTm1 上训练好的模型
└── run.py
```

### 测试模型

```bash
python run.py --task_name long_term_forecast \
              --is_training 0 \
              --model VisionTSRAR \
              --data ETTm1 \
              --ckpt_path checkpoints/VisionTSRAR_ETTm1/checkpoint.pth
```

## ⚠️ 注意事项

### 1. 路径问题

**在 `long_term_tsf/` 目录下运行命令时**：
- ✅ 使用 `../ckpt/` 访问根目录的预训练权重
- ✅ 使用 `./checkpoints/` 保存训练好的模型

**在根目录下运行命令时**：
- ✅ 使用 `./ckpt/` 访问预训练权重
- ✅ 使用 `./long_term_tsf/checkpoints/` 保存训练好的模型

### 2. Windows 迁移

如果迁移到 Windows，路径分隔符会自动处理（使用 `pathlib.Path` 或 `os.path.join`）：

```python
# 正确做法（跨平台）
from pathlib import Path
ckpt_dir = Path("../ckpt")

# 错误做法（仅限 Linux/macOS）
ckpt_dir = "../ckpt"  # Windows 可能需要 "..\\ckpt"
```

### 3. 权重文件不要重复

- ❌ 不要在多个目录存放相同的预训练权重
- ✅ 使用统一的 `ckpt/` 目录集中管理
- ✅ 通过修改路径参数来访问

## 📊 存储空间对比

### 修复前（重复存储）

```
根目录 ckpt/           ~700MB
long_term_tsf/ckpt/    ~700MB
总计：~1.4GB（重复）
```

### 修复后（集中管理）

```
根目录 ckpt/           ~700MB
long_term_tsf/checkpoints/  ~100MB（训练好的模型，可选）
总计：~700MB（节省 50%）
```

## 🎓 最佳实践

### 1. 预训练权重管理

```bash
# 所有预训练权重放在根目录 ckpt/
VisionTSRAR/
└── ckpt/
    ├── vq_ds16_c2i.pt
    ├── randar_0.3b_*.safetensors
    └── mae_base.pth  # 如果使用 VisionTS
```

### 2. 训练模型保存

```bash
# 每个任务/数据集单独保存
long_term_tsf/
└── checkpoints/
    ├── ETTm1_VisionTSRAR/
    ├── ETTh1_VisionTSRAR/
    ├── Weather_VisionTS/
    └── Electricity_VisionTSRAR/
```

### 3. 版本控制

```bash
# .gitignore 应该忽略
.gitignore:
ckpt/*.pt
ckpt/*.safetensors
checkpoints/
```

## 🔗 相关文档

- [Windows 迁移指南.md](./Windows 迁移指南.md) - 完整的 Windows 迁移教程
- [download_ckpt.py](./download_ckpt.py) - 预训练权重下载脚本
- [long_term_tsf/README.md](./long_term_tsf/README.md) - Time-Series-Library 使用指南

## ❓ 常见问题

### Q: 为什么有两个 ckpt 目录？

A: 这是历史遗留问题。原本设计是：
- 根目录 `ckpt/`：预训练权重
- `long_term_tsf/ckpt/`：训练好的模型 checkpoint

但实际操作中误将预训练权重也下载到了 `long_term_tsf/ckpt/`，导致重复。

### Q: 可以删除根目录的 ckpt 吗？

A: **不可以**！根目录的 `ckpt/` 存放的是预训练权重，是模型的核心。删除后需要重新下载（约 700MB，10-15 分钟）。

### Q: 可以删除 long_term_tsf/ckpt 吗？

A: **可以且应该删除**！这个目录里的预训练权重是重复的。删除后修改路径参数指向根目录的 `ckpt/` 即可。

### Q: 训练好的模型保存在哪里？

A: 使用 `--checkpoints` 参数指定保存目录，建议使用 `./long_term_tsf/checkpoints/`。

---

**最后更新**: 2024-XX-XX  
**维护者**: VisionTSRAR Team
