# VisionTSRAR 训练优化日志

## 改动时间
2026-04-18

## 优化目标
- 降低训练 loss（mse/mae）
- 缩短训练时间
- 控制显存峰值（支持 bs=16 等效训练）

---

## 优化历程

### 阶段 1：显存优化（梯度累积）

**问题：** bs=16 时显存溢出（~30GB）

**解决方案：** 实施梯度累积方案

| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size | 8 | 实际 batch_size |
| gradient_accumulation_steps | 2 | 累积 2 次，等效 bs=16 |

**改动文件：**
- `long_term_tsf/run.py`：添加 `--gradient_accumulation_steps` 参数
- `long_term_tsf/exp/exp_long_term_forecasting.py`：实现梯度累积逻辑
  - 只在累积开始时清零梯度
  - loss 除以累积步数
  - 只在累积完成时更新参数

**效果：**
- 显存从 ~30GB 降至 ~17.4GB
- 等效 batch_size = 16
- loss 无影响（数学等价）

---

### 阶段 2：训练质量优化（generate 比例调整）

**问题：** 初始 generate 比例 30% 过低，导致训练/推理分布不一致

**改动：**

| 版本 | generate 比例 | num_inference_steps | 效果 |
|------|--------------|---------------------|------|
| v1 | 30% | 88 | mse: 0.999, mae: 0.660 |
| v2 | 50% | 44 | mse: 0.899, mae: 0.644 |
| v3 | 40% | 44 | 待测试 |

**改动文件：**
- `visiontsrar/models_rar.py`：
  - `_forward_train` 方法中调整 generate 比例
  - 降低 num_inference_steps 从 88 到 44

**效果（v2）：**
- mse 降低 10%（0.999 → 0.899）
- mae 降低 2.4%（0.660 → 0.644）
- 显存增加 3.6GB（17.4GB → 21GB）
- 训练时间增加 ~2 分钟

---

### 阶段 3：精确控制 generate 频率

**问题：** 随机决策导致显存使用不稳定

**方案：** 外部控制 teacher forcing vs generate 的频率

**改动文件：**
- `long_term_tsf/run.py`：添加 `--generate_frequency` 参数
- `long_term_tsf/exp/exp_long_term_forecasting.py`：
  - 训练循环中实现 `use_teacher_forcing = (i % generate_frequency != 0)`
  - `_model_forward` 方法接受 `use_teacher_forcing` 参数
- `long_term_tsf/models/VisionTSRAR.py`：
  - `forecast` 和 `forward` 方法接受 `use_teacher_forcing` 参数
- `visiontsrar/model.py`：
  - `forward` 方法接受并传递 `use_teacher_forcing` 参数
- `visiontsrar/models_rar.py`：
  - `forward` 和 `_forward_train` 方法接受 `use_teacher_forcing` 参数

**效果：**
- 每 4 个 batch 用 1 次 generate（75% TF, 25% generate）
- 显存使用更可预测
- 训练节奏更稳定

---

### 阶段 4：恢复 Schedule Sampling（当前方案）

**问题：** 外部控制覆盖了原始的 schedule sampling 设计

**原始 Schedule Sampling 设计：**
```
Epoch 0-2: 100% TF（学习基础模式）
Epoch 3:   80% TF / 20% generate
Epoch 4:   60% TF / 40% generate
Epoch 5:   40% TF / 60% generate
Epoch 6:   20% TF / 80% generate
Epoch 7+:  20% TF / 80% generate（稳定阶段）
```

**当前配置：**
- 移除外部 generate 频率控制
- 恢复内部 schedule sampling 逻辑
- generate 比例调整为 40%（平衡质量和显存）
- 训练 10 轮（完整覆盖 schedule sampling）

**改动文件：**
- `long_term_tsf/exp/exp_long_term_forecasting.py`：
  - 恢复 `use_teacher_forcing=None`（让内部逻辑控制）
- `visiontsrar/models_rar.py`：
  - generate 比例从 50% 降至 40%
- `train_optimized.sh`：
  - 移除 `--generate_frequency` 参数
  - 训练轮数从 6 增至 10

---

## 当前配置（v3）

| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size | 8 | 实际 batch_size |
| gradient_accumulation_steps | 2 | 等效 bs=16 |
| generate 比例 | 40% | 每次 generate 生成 40% 的 query tokens |
| num_inference_steps | 44 | 推理步数（从 88 降低） |
| lradj | cosine | 余弦退火学习率 |
| train_epochs | 10 | 完整覆盖 schedule sampling |
| use_teacher_forcing | None | 使用内部 schedule sampling 逻辑 |

---

## 文件改动清单

### 1. long_term_tsf/run.py
- 添加 `--gradient_accumulation_steps` 参数
- 添加 `--generate_frequency` 参数（后续移除）

### 2. long_term_tsf/exp/exp_long_term_forecasting.py
- 实现梯度累积逻辑
- `_model_forward` 方法接受 `use_teacher_forcing` 参数
- 训练循环传递 `use_teacher_forcing` 参数

### 3. long_term_tsf/models/VisionTSRAR.py
- `forecast` 方法接受 `use_teacher_forcing` 参数
- `forward` 方法接受 `use_teacher_forcing` 参数

### 4. visiontsrar/model.py
- `forward` 方法接受 `use_teacher_forcing` 参数

### 5. visiontsrar/models_rar.py
- `forward` 方法接受 `use_teacher_forcing` 参数
- `_forward_train` 方法接受 `use_teacher_forcing` 参数
- generate 比例从 30% → 50% → 40%
- num_inference_steps 从 88 → 44

---

## 训练效果对比

| 版本 | 轮数 | mse | mae | 显存 | 训练时间 |
|------|------|-----|-----|------|----------|
| v1 (30%) | 6 | 0.999 | 0.660 | 17.4GB | 基准 |
| v2 (50%) | 6 | 0.899 | 0.644 | 21GB | +2 分钟 |
| v3 (40%) | 10 | 待测试 | 待测试 | 待测试 | 待测试 |

---

## 关键设计决策

### 1. 为什么使用梯度累积？
- 显存限制无法直接训练 bs=16
- 梯度累积数学等价于大 batch_size
- 不影响 loss 收敛

### 2. 为什么调整 generate 比例？
- 30% 过低，训练/推理分布不一致
- 50% 显存过高
- 40% 是平衡点

### 3. 为什么降低 num_inference_steps？
- 88 步训练时间过长
- 44 步足够生成合理的 token 序列
- 训练时间缩短 ~50%（generate 路径）

### 4. 为什么恢复 schedule sampling？
- 外部控制覆盖了原始设计
- schedule sampling 有理论依据
- 10 轮训练能完整覆盖 schedule

---

## 下一步计划

1. 训练 10 轮，观察 loss 收敛曲线
2. 如果 loss 继续下降，考虑增加训练轮数
3. 如果显存仍然过高，考虑降低 generate 比例到 30%
4. 如果训练时间过长，考虑添加 torch.compile
