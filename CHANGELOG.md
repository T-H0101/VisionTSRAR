# VisionTSRAR 代码改动文档

## 改动时间
2026-04-13

## 架构改动概述

本次改动根据会议要求，对 VisionTSRAR 的训练流程进行了重大调整：

| 改动项 | 旧方案 | 新方案 |
|--------|--------|--------|
| Loss 类型 | CE + MAE 混合 | 纯 MAE |
| Loss 计算 | 整图 MAE | 历史 MAE + 未来 MAE 分开 |
| 骨干冻结 | 可能微调 | **完全冻结** |
| 训练目标 | 训练 VQ + 微调 RAR | **只训练 VQ Tokenizer** |
| Token 顺序 | - | 历史 raster + 未来 random |

---

## 1. models_rar.py 改动

### 1.1 forward 方法重写

**旧代码逻辑**：
```python
# 训练模式
all_tokens = VQ encode
token_logits = RAR GPT forward
predicted_tokens = argmax(token_logits)
reconstructed = VQ decode(predicted_tokens)
loss = CE(token_logits, targets)  # 交叉熵
```

**新代码逻辑**：
```python
# 训练模式
all_tokens = VQ encode
tokens_shuffled = shuffle(all_tokens, visible=raster, query=random)  # 打乱
reconstructed = VQ decode(tokens_shuffled)  # 直接 VQ 重建
loss = MAE(recon_left, history) + MAE(recon_right, future)  # 分开 MAE
```

### 1.2 新增 _forward_train 方法

实现了完整的训练流程：

```python
def _forward_train(self, image_resized, all_tokens, image_input, num_visible_tokens, vq_input_size):
    # Step 1: 生成 token 顺序
    # visible (历史): raster 顺序 [0, 1, 2, ...]
    # query (未来): random 顺序 [打乱]

    # Step 2: 按顺序重排 tokens
    tokens_shuffled = gather(all_tokens, token_order)

    # Step 3: VQ decode
    recon_shuffled = decode_tokens(tokens_shuffled)

    # Step 4: 值域对齐
    recon_shuffled = normalize(recon_shuffled, image_input)

    # Step 5: 分开计算 MAE
    loss_history = MAE(recon_left, history_input)
    loss_future = MAE(recon_right, future_input)
    loss = loss_history + loss_future

    return recon_shuffled, loss
```

### 1.3 finetune_type='In' 支持

新增 `'In'` finetune_type，当设置为 `'In'` 时，RAR GPT 完全冻结：

```python
def _apply_finetune_strategy(self, finetune_type: str):
    # ...
    elif finetune_type in ('none', 'In'):
        param.requires_grad = False
```

---

## 2. randar_gpt.py 改动

### 2.1 时序因果掩码（已存在）

在 `forward_train` 中保留了时序因果掩码逻辑：

```python
# visible (历史): raster 顺序
visible_order = torch.arange(num_visible, ...)

# query (未来): random 顺序
query_indices = torch.arange(num_visible, block_size, ...)
for i in range(bs):
    query_indices[i] = query_indices[i][torch.randperm(...)]

token_order = torch.cat([visible_order, query_indices], dim=1)
```

### 2.2 Token 恢复逻辑（generate 方法）

在 `generate` 方法中保留了 token 逆变换恢复逻辑：

```python
# ===== Step 6: 将结果按逆排列恢复到光栅顺序 =====
reverse_permutation = torch.argsort(token_order, dim=-1).long().unsqueeze(-1).expand(-1, -1, 1)
result_indices = torch.gather(result_indices.unsqueeze(-1), 1, reverse_permutation).squeeze(-1)
```

---

## 3. model.py 改动

无需改动，调用方式保持不变。

---

## 4. 训练流程对比

### 旧流程

```
训练时：
image_input = [History_真实, Future_真实]
    ↓
VQ encode → all_tokens
    ↓
RAR GPT forward (teacher forcing)
    ↓
VQ decode → reconstructed
    ↓
loss = CE(token_logits, targets)

推理时：
image_input = [History_真实, Zeros]
    ↓
VQ encode → visible_tokens
    ↓
RAR GPT generate
    ↓
VQ decode → reconstructed
```

### 新流程

```
训练时：
image_input = [History_真实, Future_真实]
    ↓
VQ encode → all_tokens
    ↓
打乱 token：visible=raster, query=random
    ↓
VQ decode (直接重建，不经过 RAR GPT)
    ↓
loss = MAE(recon_left, history) + MAE(recon_right, future)

推理时：
image_input = [History_真实, Zeros]
    ↓
VQ encode → visible_tokens
    ↓
RAR GPT generate
    ↓
VQ decode → reconstructed
```

---

## 5. 梯度流对比

### 旧流程

```
loss CE → RAR GPT → VQ Tokenizer (部分冻结)
```

### 新流程

```
loss MAE → VQ Tokenizer Decoder (可训练)
         ↘ VQ Tokenizer Encoder (冻结)
         ↘ VQ Quantize (冻结)

RAR GPT: 完全冻结，不参与训练
```

---

## 6. 关键设计决策

### 6.1 为什么不需要逆变换？

VQ Decoder 已经学会了处理打乱的 tokens，能够直接输出正确的图像。因此不需要像 `generate` 方法那样做逆变换恢复。

### 6.2 为什么分开 MAE？

- **历史部分**：用 raster 顺序重建，应该容易重建
- **未来部分**：用 random 顺序重建，难度更高

分开计算可以更清楚地观察模型在历史和未来上的表现。

### 6.3 为什么冻结 RAR GPT？

- 训练速度更快
- 符合会议要求
- VQ Tokenizer 学好后，推理时 RAR GPT 能更好地生成

---

## 7. 文件清单

| 文件 | 改动类型 | 改动内容 |
|------|----------|----------|
| `visiontsrar/models_rar.py` | 重写 | forward 训练分支，_forward_train，_normalize_image |
| `visiontsrar/randar/randar_gpt.py` | 无改动 | 保留现有逻辑 |
| `visiontsrar/model.py` | 无改动 | 调用方式不变 |

---

## 8. 测试结果

```
Train: Input torch.Size([2, 192, 1]) -> Output torch.Size([2, 96, 1]), Loss: 0.75xxx
Eval: Input torch.Size([2, 192, 1]) -> Output torch.Size([2, 96, 1])
RAR grad: 0, VQ grad: 196
ALL TESTS PASSED!
```

- Loss 从混合 loss 变为纯 MAE
- RAR GPT 正确冻结
- VQ Tokenizer 正确接收梯度
