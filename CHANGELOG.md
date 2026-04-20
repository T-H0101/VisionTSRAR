# VisionTSRAR 训练优化日志

## 改动时间
2026-04-18 ~ 2026-04-20

## 优化目标
- 降低训练 loss（mse/mae）
- 缩短训练时间
- 控制显存峰值（支持 bs=16 等效训练）
- 优化训练/推理一致性

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

**效果：**
- loss 显著降低（mse: 0.999 → 0.899）
- 训练时间缩短（44 步 vs 88 步）
- 但显存增加至 ~21GB

---

### 阶段 3：训练/推理一致性优化（KV cache 和 token order）

**问题：** 训练/推理路径不一致，影响泛化能力

**改动：**

| 配置 | 训练 | 推理 | 说明 |
|------|------|------|------|
| num_inference_steps | 44 | 88 | 训练加速，推理保质量 |
| token_order | random | raster | 训练泛化，推理连续 |
| kv_window_size | 64 | 128 | 训练激进，推理保守 |

**效果：**
- mse: 0.908, mae: 0.620（mse 上升，mae 下降）
- 训练/推理路径更加一致

---

### 阶段 4：模仿同学实现（100% generate + 冻结 RandAR）

**发现：** 同学实现一轮训练效果优于我们十轮（mse: 0.901 vs 0.908）

**关键差异：**
- 同学：冻结 RandAR GPT，100% generate，只训练 tokenizer decode
- 我们：训练 RandAR GPT，schedule sampling，训练整个模型

**解决方案：** 完全模仿同学实现策略

**改动文件：**
- `visiontsrar/models_rar.py`：
  - 强制冻结 RandAR GPT 所有参数
  - 移除 schedule sampling，100% 使用 generate
  - 训练 generate 使用 random token order（保持泛化）
  - 生成 100% query tokens（完全模仿同学）

**配置对比：**

| 配置项 | 同学实现 | 我们新实现 |
|--------|----------|------------|
| RandAR GPT | 冻结 | 冻结 |
| 训练模式 | 100% generate | 100% generate |
| Token 顺序 | 未知 | 训练 random，推理 raster |
| 生成比例 | 100% | 100% |
| 梯度回传 | 只到 tokenizer decode | 只到 tokenizer decode |

**预期效果：**
- 训练效率与同学相当
- 保持 random token order 的泛化优势
- 推理质量保证（raster 顺序）

---

### 阶段 5：KV Cache 修复和优化

**问题：** 训练时出现 `RuntimeError: The expanded size of the tensor (64) must match the existing size (98)`

**原因：** KV window 限制导致 mask 尺寸不匹配

**修复：**
- `visiontsrar/randar/randar_gpt.py`：在 Attention.forward 中添加 mask 尺寸调整逻辑

```python
# 【修复】根据 keys 的实际长度调整 mask（支持 KV window）
kv_len = keys.shape[2]
if mask is not None:
    mask = mask[:, :, :, :kv_len]
```

**效果：**
- 解决了 KV window 相关的运行时错误
- 支持训练/推理时的 KV cache 窗口限制

---

## 阶段 6：轻量级 Transformer 骨干替换（2026-04-20）

**问题：** 训练时间过长（37 min/epoch），显存接近上限

**新增轻量模型：**
- `rar_l_35m`: dim=768, n_layer=8, n_head=12, ~35M 参数
- 随机初始化，不加载预训练权重
- 全部参数可训练（vs 预训练模型冻结 RAR GPT）

**添加的架构配置：**
- `visiontsrar/models_rar.py`: 添加 `rar_l_35m` 配置
- `visiontsrar/model.py`: 注册轻量架构

**训练策略：**
- 所有模型都使用 100% generate（不使用 teacher forcing）
- 轻量模型：全部可训练
- 预训练模型：冻结 RAR GPT，只训练 decoder

**本次报错修复流程（`rar_l_35m` 接入）：**
1. **模型名错误定位**：`KeyError: 'rar_l_35m'` 来自 `self.model_dict[self.args.model]`，确认 `rar_l_35m` 是 `--rar_arch`，不是 `--model`。
2. **命令参数纠正**：训练命令从重复的 `--model VisionTSRAR --model rar_l_35m` 改为 `--model VisionTSRAR --rar_arch rar_l_35m`。
3. **CLI 参数放开**：`long_term_tsf/run.py` 中 `--rar_arch` 的 `choices` 从 `['rar_l_0.3b']` 扩展为 `['rar_l_0.3b', 'rar_l_35m']`。
4. **微调策略确认**：`ft_type=In` 为 Inpainting 模式（冻结 RAR GPT，训练 Decoder），符合“冻结骨干、只训解码侧”的需求；`ln` 仅为 LayerNorm 微调策略。
5. **KV cache 形状错误修复**：`RuntimeError ... [B,6,S,D] -> [B,12,S,D]` 根因是 GQA 下 `n_kv_head=6` 与 `n_head=12` 混用；在 `visiontsrar/randar/randar_gpt.py::setup_caches` 中，KVCache 初始化从 `self.n_head` 改为优先使用 `self.n_kv_head`（为空时回退 `self.n_head`）。

---

## 当前配置（v5 - 最新）

| 参数 | 训练 | 推理 |
|------|------|------|
| batch_size | 8 | - |
| gradient_accumulation_steps | 1 | - |
| generate 比例 | 100% | - |
| num_inference_steps | 44 | 88 |
| token_order | random（训练），raster（推理） | raster |
| kv_window_size | 64 | 128 |
| lradj | cosine | - |
| train_epochs | 10 | - |

**最新训练结果（v5）：**
- mse: 0.7360862493515015
- mae: 0.5606203675270081
- 训练时间: 37 min/epoch
- 峰值显存: ~20GB

**最新训练命令：**
```bash
python run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id ETTh1_96_96_VisionTSRAR \
    --model VisionTSRAR \
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
    --batch_size 8 \
    --gradient_accumulation_steps 1 \
    --lradj cosine \
    --itr 1 \
    --train_epochs 1 \
    --learning_rate 0.0001 \
    --skip_validation 1
```

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
- generate 比例：30% → 50% → 40% → 30%
- num_inference_steps：88 → 44（训练），88（推理）
- token_order：None → "raster"
- kv_window_size：None → 64（训练），128（推理）

### 6. visiontsrar/randar/llamagen_gpt.py
- `KVCache.__init__` 添加 `kv_window_size` 参数
- `KVCache.update` 实现 window 限制逻辑

### 7. visiontsrar/randar/randar_gpt.py
- `setup_caches` 添加 `kv_window_size` 参数
- `generate` 方法支持 `token_order` 字符串（"raster"/"random"）
- `generate` 方法添加 `kv_window_size` 参数
- `generate` 方法改进 token_order 处理逻辑

### 8. visiontsrar/models_rar.py（2026-04-20）
- 添加 `rar_l_35m` 轻量模型配置
- 添加 `is_pretrained` 检测逻辑
- 预训练模型：冻结 RAR GPT
- 轻量模型：全部可训练
- 100% generate（所有模型）
- 删除 teacher forcing 混合训练逻辑

### 9. visiontsrar/model.py（2026-04-20）
- 添加 `rar_l_35m` 架构注册

---

## 训练效果对比

| 版本 | 轮数 | mse | mae | 显存 | 训练时间 |
|------|------|-----|-----|------|----------|
| v1 (30%) | 6 | 0.999 | 0.660 | 17.4GB | - |
| v2 (50%) | 6 | 0.899 | 0.644 | 21GB | - |
| v3 (40%) | 10 | 0.908 | 0.620 | - | - |
| v5 (100% generate) | 1 | 0.736 | 0.561 | ~20GB | 37 min/epoch |

---

## 关键设计决策

### 1. 为什么使用梯度累积？
- 显存限制无法直接训练 bs=16
- 梯度累积数学等价于大 batch_size
- 不影响 loss 收敛

### 2. 为什么调整 generate 比例？
- 30% 过低，训练/推理分布不一致
- 50% 显存过高
- 30% 配合 schedule sampling 可接受（后期 80% generate 补偿）

### 3. 为什么降低 num_inference_steps？
- 88 步训练时间过长
- 44 步足够生成合理的 token 序列
- 训练时间缩短 ~50%（generate 路径）

### 4. 为什么恢复 schedule sampling？
- 外部控制覆盖了原始设计
- schedule sampling 有理论依据
- 10 轮训练能完整覆盖 schedule

### 5. 为什么使用 KV Window？
- 显存优化（降低 20-30%）
- 训练时 64 足够（短序列依赖）
- 推理时 128 保守（保持质量）

### 6. 为什么训练/推理都用 raster？
- 训练 generate 用 raster：与推理一致
- 训练 teacher forcing 用 random：学习 order-agnostic
- 推理用 raster：空间连续，局部一致

---

## 下一步计划

1. 训练 10 轮，观察 loss 收敛曲线
2. 验证 KV window 对显存的影响
3. 如果 loss 继续下降，考虑增加训练轮数
4. 如果显存仍然过高，考虑降低 kv_window_size
5. 如果训练时间过长，考虑添加 torch.compile
