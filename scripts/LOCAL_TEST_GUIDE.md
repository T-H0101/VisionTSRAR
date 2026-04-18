# 本地测试指南

## 为什么需要本地测试？

在服务器上训练前，先在本地MacBook上验证：
1. 代码能否正常运行
2. Checkpoint大小是否合理
3. 模型结构是否正确

## 快速测试

### 步骤1: 运行本地测试脚本

```bash
cd /Users/tian/Desktop/VisionTS/VisionTSRAR
python scripts/local_test.py
```

### 步骤2: 检查输出

期望输出：
```
✓ 模型初始化成功
✓ 前向传播成功
✓ Checkpoint保存成功
  只保存可训练参数: 48.2 MB
✓ Checkpoint加载成功

✓ 所有测试通过！可以安全地在服务器上训练。
```

如果checkpoint大小超过100MB，说明保存了不必要的参数。

## 完整测试（可选）

如果需要测试完整训练流程：

```bash
# 使用CPU，小batch_size，快速验证
python -m run \
    --task_name long_term_forecast \
    --is_training 1 \
    --model VisionTSRAR \
    --model_id local_test \
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
    --batch_size 2 \
    --itr 1 \
    --train_epochs 1 \
    --learning_rate 0.0001 \
    --lradj cosine \
    --skip_validation 1 \
    --skip_test 1 \
    --use_gpu 0 \
    --fast_train_batches 10 \
    --des local_test
```

关键参数：
- `--use_gpu 0`: 使用CPU
- `--batch_size 2`: 小batch_size
- `--train_epochs 1`: 只训练1个epoch
- `--fast_train_batches 10`: 只训练10个batch

## 检查checkpoint大小

```bash
# 查看checkpoint大小
ls -lh ./checkpoints/*/checkpoint.pth
```

期望大小：
- 平衡版（通道64）: 约50MB
- 增强版（通道128）: 约100MB

如果超过1GB，说明保存了冻结的RAR GPT参数。

## 常见问题

### Q: 本地测试很慢怎么办？
A: 使用`--fast_train_batches 10`只训练10个batch，快速验证代码逻辑。

### Q: 内存不足怎么办？
A: 降低batch_size到1，或使用更小的模型。

### Q: 如何确认checkpoint只保存了可训练参数？
A: 运行`python scripts/local_test.py`，会显示checkpoint大小。

## 测试清单

在服务器上训练前，确认：

- [ ] 本地测试脚本通过
- [ ] Checkpoint大小合理（<100MB）
- [ ] 模型参数统计正确
- [ ] 前向传播正常
- [ ] Checkpoint加载正常

全部通过后，再上传到服务器训练！
