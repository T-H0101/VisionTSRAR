# VisionTSRAR 预测试对比指南

本文档描述如何对 VisionTSRAR 进行预测试，与 DLinear 基线模型进行对比，验证模型的基本功能正确性。

## 1. 环境准备

### 1.1 安装依赖

```bash
# 进入项目目录
cd /Users/tian/Desktop/VisionTS/VisionTSRAR

# 安装依赖
pip install -r requirements.txt

# 额外安装 HuggingFace 下载工具
pip install huggingface_hub safetensors
```

### 1.2 下载预训练权重

```bash
# 方法1：使用下载脚本（推荐）
python ckpt/download_ckpt.py --ckpt_dir ./ckpt/

# 方法2：手动下载
# 从 https://huggingface.co/yucornell/RandAR 下载以下文件到 ckpt/ 目录：
# - vq_ds16_c2i.pt
# - rbrar_l_0.3b_c2i.safetensors
```

### 1.3 准备数据集

数据集将自动下载到 `dataset/` 目录。首次运行时需要网络连接。

## 2. 测试命令

### 2.1 Mac MPS 设备

```bash
# ETTh1 数据集测试
python -u long_term_tsf/run.py \
  --model VisionTSRAR \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --rar_arch rar_l_0.3b \
  --num_inference_steps 88 \
  --position_order raster \
  --batch_size 8 \
  --num_workers 0 \
  --device mps
```

### 2.2 Linux CUDA 设备

```bash
# ETTh1 数据集测试
python -u long_term_tsf/run.py \
  --model VisionTSRAR \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --rar_arch rar_l_0.3b \
  --num_inference_steps 88 \
  --position_order raster \
  --batch_size 16 \
  --num_workers 4 \
  --device cuda \
  --use_gpu \
  --gpu 0
```

### 2.3 CPU 回退模式（调试用）

```bash
python -u long_term_tsf/run.py \
  --model VisionTSRAR \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --rar_arch rar_l_0.3b \
  --num_inference_steps 88 \
  --batch_size 2 \
  --num_workers 0 \
  --device cpu
```

## 3. 各数据集测试参数

| 数据集 | seq_len | pred_len | periodicity | batch_size(Mac) |
|--------|---------|----------|-------------|-----------------|
| ETTh1 | 96 | 96/192/336/720 | 24 | 8 |
| ETTh2 | 96 | 96/192/336/720 | 24 | 8 |
| ETTm1 | 96 | 96/192/336/720 | 96 | 8 |
| ETTm2 | 96 | 96/192/336/720 | 96 | 8 |
| Weather | 96 | 96/192/336/720 | 144 | 4 |
| Illness | 36 | 24/36/48/60 | 52 | 8 |

## 4. 与 DLinear 对比

### 4.1 运行 DLinear 基线

```bash
# 使用 Time-Series-Library 中的 DLinear
cd /Users/tian/Desktop/VisionTS/Time-Series-Library

python -u run.py \
  --model DLinear \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --batch_size 32
```

### 4.2 对比指标

| 指标 | 说明 | 越小越好 |
|------|------|----------|
| MSE | 均方误差 | ✅ |
| MAE | 平均绝对误差 | ✅ |

预期结果格式：
```
ETTh1  pred_len=96   MSE:0.xxx  MAE:0.xxx
```

### 4.3 预期结果范围

基于 VisionTS 论文的基准结果，VisionTSRAR 的预期范围：

| 数据集 | pred_len | VisionTS MSE | DLinear MSE | VisionTSRAR 预期 |
|--------|----------|-------------|-------------|-----------------|
| ETTh1 | 96 | 0.386 | 0.386 | 接近 VisionTS |
| ETTh1 | 192 | 0.427 | 0.449 | 接近 VisionTS |
| ETTh1 | 336 | 0.442 | 0.482 | 接近 VisionTS |
| ETTh1 | 720 | 0.463 | 0.515 | 接近 VisionTS |
| Weather | 96 | 0.159 | 0.196 | 接近 VisionTS |

> ⚠️ 由于 RAR 替换 MAE 后的精度尚未验证，以上为参考范围。实际结果可能因 RAR 生成质量而有所不同。

## 5. 常见问题排查

### 5.1 权重下载失败

**症状**: `Failed to download from HuggingFace`

**解决方案**:
1. 检查网络连接
2. 设置代理: `export HF_ENDPOINT=https://hf-mirror.com`
3. 手动从浏览器下载权重文件
4. 使用镜像站: `huggingface-cli download --repo-id yucornell/RandAR --local-dir ckpt/ --endpoint https://hf-mirror.com`

### 5.2 内存不足 (OOM)

**症状**: `RuntimeError: MPS backend out of memory` 或 `CUDA out of memory`

**解决方案**:
1. 减小 batch_size: `--batch_size 4` 或 `--batch_size 2`
2. 减小 seq_len: `--seq_len 48`
3. 使用 CPU 回退: `--device cpu`
4. 减小推理步数: `--num_inference_steps 44`（牺牲精度换速度）

### 5.3 MPS 不可用

**症状**: `MPS device not available`

**解决方案**:
1. 确认 macOS 版本 ≥ 12.3
2. 确认 PyTorch 版本 ≥ 2.0: `python -c "import torch; print(torch.backends.mps.is_available())"`
3. 回退 CPU: `--device cpu`

### 5.4 推理速度过慢

**症状**: 单次预测耗时超过 60 秒

**解决方案**:
1. 确认使用了 GPU/MPS: 检查 `--device` 参数
2. 减小推理步数: `--num_inference_steps 44`
3. 降低采样温度: `--temperature 0.5`（更确定性的生成，减少重复采样）
4. CPU 模式下减少 batch_size

### 5.5 训练 loss 不下降

**症状**: 训练多个 epoch 后 loss 无明显下降

**可能原因**:
1. 学习率过大/过小 → 尝试 `--learning_rate 1e-4` 或 `1e-5`
2. LayerNorm 未正确解冻 → 检查 `--finetune_type ln`
3. VQ Tokenizer 输出不正确 → 检查权重是否完整下载

### 5.6 预测结果全为常数

**症状**: 预测值几乎没有变化

**可能原因**:
1. 类别条件 (cond_idx) 处理不当 → 检查 `models_rar.py` 中的条件生成逻辑
2. 温度参数过低 → 尝试 `--temperature 1.0`
3. VQ decode 失败 → 检查 tokenizer 输出是否正确

## 6. 快速验证清单

在正式测试前，建议按以下清单逐项验证：

- [ ] Python 环境: `python -c "import torch; print(torch.__version__)"`
- [ ] 设备检测: `python -c "import torch; print(torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else 'N/A')"`  
- [ ] 权重文件: `ls -la ckpt/vq_ds16_c2i.pt ckpt/rbrar_l_0.3b_c2i.safetensors`
- [ ] 数据集: `ls dataset/ETTh1/`
- [ ] 模型加载: `python -c "from visiontsrar import VisionTSRAR; m = VisionTSRAR(); print('OK')"`
- [ ] 单样本推理: 运行一次 batch_size=1 的推理，确认无报错
