# VisionTSRAR 训练命令记录

## 历史训练记录

### 第 1 次训练（v1 - 30% generate）
**配置：**
- batch_size: 8
- gradient_accumulation_steps: 2
- generate 比例: 30%
- num_inference_steps: 88
- train_epochs: 6

**命令：**
```bash
source /opt/miniconda3/bin/activate 1d_tokenizer
cd /Users/tian/Desktop/VisionTS/VisionTSRAR/long_term_tsf
python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_96_96 \
    --model VisionTSRAR \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 6 \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --lradj cosine \
    --learning_rate 0.0001 \
    --skip_validation 1
```

**结果：**
```
test shape: (2785, 96, 7) (2785, 96, 7)
mse: 0.9986847639083862, mae: 0.6595329642295837
峰值显存: 17827 MiB
```

---

### 第 2 次训练（v2 - 50% generate）
**配置：**
- batch_size: 8
- gradient_accumulation_steps: 2
- generate 比例: 50%
- num_inference_steps: 44
- train_epochs: 6

**结果：**
```
test shape: (2785, 96, 7) (2785, 96, 7)
mse: 0.8991530537605286, mae: 0.6438161134719849
峰值显存: 21000 MiB
```

---

### 第 3 次训练（v3 - 40% generate + KV cache）
**配置：**
- batch_size: 8
- gradient_accumulation_steps: 2
- generate 比例: 40%
- num_inference_steps: 44（训练），88（推理）
- token_order: random（训练），raster（推理）
- kv_window_size: 64（训练），128（推理）
- train_epochs: 10

**结果：**
```
test shape: (2785, 96, 7) (2785, 96, 7)
mse: 0.9081096053123474, mae: 0.6195420622825623
```

---

## 最新训练命令（v4 - 100% generate + 冻结 RandAR）

### 配置说明
**核心改动：**
- **100% generate**：移除 schedule sampling，训练/推理一致
- **冻结 RandAR GPT**：梯度只回传到 tokenizer decode，模仿同学实现
- **训练 random，推理 raster**：保持泛化能力，保证推理质量
- **KV cache 优化**：训练 KV window=64，推理 KV window=128

**完整配置：**
| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size | 8 | 实际 batch_size |
| gradient_accumulation_steps | 2 | 等效 batch_size=16 |
| train_epochs | 10 | 完整训练周期 |
| learning_rate | 0.0001 | 学习率 |
| lradj | cosine | 余弦退火学习率 |
| generate 比例 | 100% | 完全模仿同学实现 |
| num_inference_steps | 44（训练），88（推理） | 训练加速，推理保质量 |
| token_order | random（训练），raster（推理） | 训练泛化，推理连续 |
| kv_window_size | 64（训练），128（推理） | 训练激进，推理保守 |

### 训练命令
```bash
source /opt/miniconda3/bin/activate 1d_tokenizer
cd /Users/tian/Desktop/VisionTS/VisionTSRAR/long_term_tsf
python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_96_96 \
    --model VisionTSRAR \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 10 \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --lradj cosine \
    --learning_rate 0.0001 \
    --skip_validation 1
```

### 预期效果
- **训练效率**：与同学实现相当（100% generate + 冻结 RandAR）
- **泛化能力**：保持 random token order 的优势
- **推理质量**：用 raster 顺序保证空间连续性
- **显存优化**：KV window 限制显存使用

### 验证方法
1. **训练启动测试**：检查是否能正常训练，没有错误
2. **显存使用监控**：验证显存占用是否合理
3. **性能对比**：与同学实现进行性能对比（mse/mae）

---

## 关键发现总结

1. **100% generate 策略**：比 schedule sampling 更有效
2. **冻结 RandAR GPT**：显著提高训练效率
3. **训练/推理一致性**：random→raster 策略平衡泛化和质量
4. **KV window 限制**：有效控制显存使用

## 下一步
- 运行最新训练命令验证效果
- 与同学实现进行性能对比
- 根据测试结果进一步优化参数

---

### 第 2 次训练（v2 - 50% generate）
**配置：**
- batch_size: 8
- gradient_accumulation_steps: 2
- generate 比例: 50%
- num_inference_steps: 44
- train_epochs: 6

**结果：**
```
test shape: (2785, 96, 7) (2785, 96, 7)
mse: 0.8991530537605286, mae: 0.6438161134719849
峰值显存: 21000 MiB
```

---

### 第 3 次训练（v3 - 40% generate + schedule sampling）
**配置：**
- batch_size: 8
- gradient_accumulation_steps: 2
- generate 比例: 40%
- num_inference_steps: 44
- train_epochs: 10

**结果：**
```
test shape: (2785, 96, 7) (2785, 96, 7)
mse: 0.9081096053123474, mae: 0.6195420622825623
```

---

## 当前训练命令（v5 - 最新）

### 配置说明
| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size | 8 | 实际 batch_size |
| gradient_accumulation_steps | 2 | 等效 bs=16 |
| generate 比例 | 30% | 配合 schedule sampling |
| num_inference_steps (训练) | 44 | 训练加速 |
| num_inference_steps (推理) | 88 | 推理保质量 |
| token_order (训练 generate) | raster | 与推理一致 |
| token_order (推理) | raster | 空间连续 |
| kv_window_size (训练) | 64 | 显存优化 |
| kv_window_size (推理) | 128 | 质量优先 |
| train_epochs | 10 | 完整覆盖 schedule sampling |
| lradj | cosine | 余弦退火学习率 |

### 训练命令

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

**结果（v5 - 100% generate + lightweight_decoder）：**
```
mse: 0.7360862493515015
mae: 0.5606203675270081
峰值显存: ~20GB
训练时间: 37 min/epoch
```

---

## 新增：轻量模型训练命令（rar_l_35m）

### 配置说明
| 参数 | 值 | 说明 |
|------|-----|------|
| model | rar_l_35m | 轻量级 Transformer |
| dim | 768 | 隐藏维度 |
| n_layer | 8 | 层数 |
| n_head | 12 | 头数 |
| 参数量 | ~35M | 轻量模型 |
| 初始化 | 随机 | 不加载预训练权重 |
| 可训练参数 | 全部 | vs 预训练模型冻结 RAR GPT |

### 训练命令
```bash
python run.py \
     --task_name long_term_forecast \
     --is_training 1 \
     --model_id ETTh1_96_96_VisionTSRAR \
     --model VisionTSRAR \
     --rar_arch rar_l_35m \
     --data ETTh1 \
     --root_path ./dataset/ETT-small/ \
     --data_path ETTh1.csv \
     --features M \
     --seq_len 96 \
     --label_len 48 \
     --pred_len 96 \
     --seasonal_patterns Monthly \
     --use_lightweight_decoder \
     --lightweight_decoder_channels 32 \
     --ft_type In \
     --use_amp \
     --batch_size 16 \
     --lradj cosine \
     --itr 1 \
     --train_epochs 5 \
     --learning_rate 0.0002 \
     --skip_validation 1
```

### 对比原版（rar_l_0.3b）
```bash
python run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id ETTh1_96_96_VisionTSRAR \
    --model VisionTSRAR \
    --rar_arch rar_l_0.3b \
    --use_lightweight_decoder \
    --ft_type In \
    --use_amp \
    --batch_size 8 \
    --gradient_accumulation_steps 1 \
    --lradj cosine \
    --itr 1 \
    --train_epochs 5 \
    --learning_rate 0.0001 \
    --skip_validation 1
```

---

## 预期效果

- 显存降低 ~20-30%（KV window 限制）
- 训练时间缩短 ~15%（44 步 + 30% generate）
- 推理质量保持（88 步 + 128 window）
- loss 进一步降低（schedule sampling 完整覆盖）

---

## 记录格式（供后续使用）

### 第 N 次训练（版本描述）
**配置：**
- 关键参数列表

**命令：**
```bash
完整命令
```

**结果：**
```
mse: xxx, mae: xxx
峰值显存: xxx MiB
训练时间: xxx
```

**备注：**
- 特殊说明
