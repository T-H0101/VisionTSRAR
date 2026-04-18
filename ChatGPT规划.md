# VisionTS + RAR 训练加速方案（torch.compile 分层优化版）  
  
## 一、目标  
  
在不破坏现有训练逻辑的前提下，实现：  
  
- 单 epoch 训练时间 ≤ 10 min  
- 推理 ≤ 20 min  
- 保持 loss 收敛稳定  
- 避免 torch.compile 带来的 graph break / retrace  
  
---  
  
## 二、核心问题（现状）  
  
当前实现存在 3 个与 torch.compile 强冲突点：  
  
### 1. 动态控制流（最大问题）  
- `_forward_train()` 内：  
  - `torch.rand().item()`（Python side）  
  - `if use_teacher_forcing`  
- → 导致 graph break / retrace  
  
---  
  
### 2. forward 混合两种模式  
- 训练 + generate 混在一个函数  
- shape / execution path 不稳定  
  
---  
  
### 3. generate 本身不可编译  
- Python for-loop  
- KV cache 动态更新  
- token 顺序动态变化  
  
---  
  
## 三、总体策略（核心思想）  
  
> **分层编译 + 动态控制流外提 + 只编译稳定子图**  
  
---  
  
## 四、最终架构设计  
  
### 训练路径拆分  
  
```text  
原：  
forward()  
  └── _forward_train()  
        ├── teacher forcing  
        └── generate  
  
改为：  
forward()  
  ├── forward_train_teacher_forcing()  
  ├── forward_train_generate()  
  └── decode_and_loss()
```

---

## 五、Step-by-Step 改造方案

---

# Step 1：外提 Schedule Sampling（关键）

## ❌ 原问题
```python
use_teacher_forcing = torch.rand(1).item() < ratio  
if use_teacher_forcing:  
    ...  
else:  
    ...
```
问题：
- Python 分支
- compile 无法静态图

---

## ✅ 改造（在训练循环中）

# exp_long_term_forecasting.py  
  
# 1. 计算 ratio（epoch级） 
```python
if epoch <= 2:  
    ratio = 1.0  
elif epoch <= 6:  
    ratio = 1.0 - (epoch - 2) / 4 * 0.8  
else:  
    ratio = 0.2  
```

# 2. batch级决策  
```python
use_teacher_forcing = (step % 4 != 0)   # 75% TF  
```
# 3. 显式调用  
```python
if use_teacher_forcing:  
    output, loss = model.forward_train_teacher_forcing(...)  
else:  
    output, loss = model.forward_train_generate(...)
```

---
# Step 2：拆分训练路径

## 2.1 Teacher Forcing 路径（可 compile）
```python
def forward_train_teacher_forcing(...):  
    token_logits, _, _ = self.rar_gpt(...)  
    predicted_tokens = ste(token_logits)  
    return self._decode_and_compute_loss(predicted_tokens)
```

---

## 2.2 Generate 路径（保持 eager）
```python
def forward_train_generate(...):  
    with torch.no_grad():  
        generated_tokens = self.rar_gpt.generate(...)  
  
    predicted_tokens = self._combine_with_gt(generated_tokens)  
    return self._decode_and_compute_loss(predicted_tokens)
```

---

## 2.3 Decode + Loss（可 compile）
```python
def _decode_and_compute_loss(tokens):  
    recon = self.decode_tokens(tokens)  
    loss = ...  
    return recon, loss
```

---

# Step 3：限制 generate 计算量（必须做）

## ❌ 当前问题

- 每次 generate 全 256 tokens

---

## ✅ 改造
```python
query_len = total_tokens - num_visible_tokens  
max_gen = int(query_len * 0.3)  
max_gen = max(1, max_gen)
```

---

### 拼接策略
```python
generated = generated_tokens[:, :num_visible_tokens + max_gen]  
  
remaining = gt_tokens[:, num_visible_tokens + max_gen:]  
  
predicted = torch.cat([generated, remaining], dim=1)
```

---
## 效果

|策略|计算量|
|---|---|
|全生成|100%|
|30%生成|↓ 60~70%|

---

# Step 4：torch.compile 使用策略

---

## 4.1 只编译 Teacher Forcing
```python
model.forward_train_teacher_forcing = torch.compile(  
    model.forward_train_teacher_forcing,  
    mode="max-autotune",  
    backend="inductor"  
)
```


---

## 4.2 编译 decode + loss

```python
model._decode_and_compute_loss = torch.compile(  
    model._decode_and_compute_loss  
)
```

---

## ❌ 不要编译

```python
model.forward()  
model.generate()
```

---

# Step 5：CLI 控制（强烈建议）
```python
--use_torch_compile  
--compile_scope tf_only|decode_only|off  
--compile_mode default|max-autotune  
--compile_backend inductor
```


---

## 示例

```python
if args.use_torch_compile:  
    if args.compile_scope == "tf_only":  
        model.forward_train_teacher_forcing = torch.compile(...)  
    elif args.compile_scope == "decode_only":  
        model._decode_and_compute_loss = torch.compile(...)
```

---

# Step 6：多卡策略

## ❌ 当前

nn.DataParallel

问题：

- compile 效果差
- overhead 大

---

## ✅ 建议

DistributedDataParallel (DDP)

---

## 最低要求

- 先单卡验证 compile
- 再迁移 DDP

---

# Step 7：性能优化优先级（重要）

|优先级|优化项|提升|
|---|---|---|
|⭐⭐⭐⭐⭐|限制 generate 长度|最大|
|⭐⭐⭐⭐|降低 generate 频率|很大|
|⭐⭐⭐|torch.compile TF路径|中等|
|⭐⭐|KV cache|中等|
|⭐|compile generate|几乎无|

---

# Step 8：验证流程

---

## 1. 功能一致性

- compile on/off loss 是否一致
- 输出图像分布是否一致

---

## 2. 性能

记录：

step time  
epoch time  
GPU 利用率  
显存

---

## 3. 图稳定性

观察：

- 是否频繁 recompilation
- 是否 graph break

---

## 4. 收敛性

必须跑：

epoch 0 → 6

因为：

- 覆盖 schedule sampling 变化区间

---

# Step 9：最终推荐配置（直接用）

# generate 频率  
use_generate = (step % 4 == 0)  
  
# generate 长度  
max_new_tokens = int(query_len * 0.3)  
  
# teacher forcing  
epoch 0~2: 1.0  
epoch 3~6: 0.5  
epoch >=7: 0.2

---

# Step 10：预期收益

|优化项|提升|
|---|---|
|generate 限制|↓ 2~3x 时间|
|compile TF|+10~30%|
|控制 flow|避免退化|

---

## 最终效果（合理预期）

原：15~25 min / epoch  
优化后：6~10 min / epoch

---

# 一句话总结

> **compile 不是核心加速点，真正决定速度的是：**
> 
> - generate 用多少
> - generate 生成多长
> 
> compile 只是“锦上添花”，不是“雪中送炭”。