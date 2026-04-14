# VisionTSRAR GPU 服务器训练指南

## 📋 目录
1. [环境准备](#1-环境准备)
2. [代码改动说明](#2-代码改动说明)
3. [权重下载](#3-权重下载)
4. [依赖安装](#4-依赖安装)
5. [训练命令](#5-训练命令)
6. [监控训练](#6-监控训练)
7. [常见问题](#7-常见问题)

---

## 1. 环境准备

### 服务器信息
- **GPU**: NVIDIA RTX 3090 (24GB)
- **CUDA**: 13.0
- **PyTorch**: 2.9.0
- **Python**: 3.11.14

### 检查环境
```bash
nvidia-smi
python --version
pip show torch
```

---

## 2. 代码改动说明

### 架构改动（2026-04-13）

本次训练使用新版架构，核心特点：

| 特性 | 说明 |
|------|------|
| **训练目标** | 只训练 VQ Tokenizer，RAR GPT 完全冻结 |
| **Loss 类型** | MAE（像素级重建），不再使用交叉熵 |
| **Loss 计算** | 历史 MAE + 未来 MAE 分开计算 |
| **Token 顺序** | 历史用 raster 顺序，未来用 random 顺序打乱 |

### 训练流程

```
输入图像: [History_真实, Future_真实]
    ↓ VQ encode
tokens: [History_tokens, Future_tokens]
    ↓ 打乱 token（历史 raster，未来 random）
tokens_shuffled
    ↓ VQ decode
reconstructed_image
    ↓
loss = MAE(recon_history, History) + MAE(recon_future, Future)
```

### 推理流程

```
输入图像: [History_真实, Zeros]
    ↓ VQ encode
visible_tokens
    ↓ RAR GPT generate
generated_tokens
    ↓ VQ decode
reconstructed_image
```

---

## 3. 权重下载

权重文件约 2GB，需要下载到 `ckpt/` 目录：

```bash
cd VisionTSRAR
mkdir -p ckpt

# 方法1: 使用项目内的下载脚本
python ckpt/download_ckpt.py

# 方法2: 手动下载（如果上面的脚本失败）
# VQ Tokenizer
wget -O ckpt/vq_ds16_c2i.pt https://huggingface.co/Quantact/VQGAN/resolve/main/vq_ds16_c2i.pt

# RAR GPT
wget -O ckpt/randar_0.3b_llamagen_360k_bs_1024_lr_0.0004.safetensors \
  https://huggingface.co/GenAI4Sci/RandAR/resolve/main/randar_0.3b_llamagen_360k_bs_1024_lr_0.0004.safetensors
```

**验证权重文件**:
```bash
ls -lh ckpt/
# 应该看到:
# vq_ds16_c2i.pt (~50MB)
# randar_0.3b_llamagen_360k_bs_1024_lr_0.0004.safetensors (~2GB)
```

---

## 4. 依赖安装

```bash
cd VisionTSRAR

# 方法1: 使用 requirements.txt（推荐）
pip install -r requirements.txt

# 方法2: 如果 torch 已安装，只安装其他依赖
pip install einops omegaconf safetensors huggingface_hub pandas scikit-learn tqdm
```

### 验证 PyTorch CUDA
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

---

## 5. 训练命令

### 5.1 ETTh1 数据集 (96→96)

```bash
cd VisionTSRAR/long_term_tsf

nohup python3 run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ETTh1_96_96 \
  --model VisionTSRAR \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --freq h \
  --periodicity 24 \
  --rar_arch rar_l_0.3b \
  --train_epochs 3 \
  --batch_size 1 \
  --learning_rate 0.0001 \
  --use_gpu True \
  --gpu 0 \
  --ft_type In \
  --test_freq 0 \
  > training_etth1_96_96.log 2>&1 &
```

### 5.2 ETTh2 数据集 (96→96)

```bash
nohup python3 run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ETTh2_96_96 \
  --model VisionTSRAR \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --freq h \
  --rar_arch rar_l_0.3b \
  --train_epochs 20 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --use_gpu True \
  --gpu 0 \
  --ft_type In \
  > training_etth2.log 2>&1 &
```

### 5.3 ETTm1 数据集 (96→96)

```bash
nohup python3 run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ETTm1_96_96 \
  --model VisionTSRAR \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --freq t \
  --rar_arch rar_l_0.3b \
  --train_epochs 20 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --use_gpu True \
  --gpu 0 \
  --ft_type In \
  > training_ettm1.log 2>&1 &
```

### 5.4 长预测窗口 ETTh1 (96→720)

```bash
nohup python3 run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ETTh1_96_720 \
  --model VisionTSRAR \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --freq h \
  --rar_arch rar_l_0.3b \
  --train_epochs 20 \
  --batch_size 4 \
  --learning_rate 0.0001 \
  --use_gpu True \
  --gpu 0 \
  --ft_type In \
  > training_etth1_720.log 2>&1 &
```

---

## 6. 监控训练

### 查看训练进程
```bash
# 查看后台运行的训练任务
jobs -l

# 或者
ps aux | grep python
```

### 查看训练日志
```bash
# 实时查看日志
tail -f training_etth1.log

# 查看最新输出
tail -n 50 training_etth1.log
```

### 监控 GPU 使用
```bash
watch -n 1 nvidia-smi
```

### 预期 Loss 曲线
- **初始 MAE Loss**: ~0.7-0.8 (标准化空间)
- **DLinear 对比**: ~0.4
- **我们的模型**: 应该稳步下降

### 停止训练
```bash
# 找到进程 ID
ps aux | grep python

# 强制停止
kill -9 PROCESS_ID

# 或者 pkill
pkill -f "python3 run.py"
```

---

## 7. 常见问题

### Q1: CUDA out of memory
**解决**: 减小 batch_size
```bash
--batch_size 4  # 原先是 8
```

### Q2: 权重文件下载失败
**解决**: 使用代理或手动下载
```bash
# 设置代理
export http_proxy=http://proxy:port
export https_proxy=http://proxy:port

# 手动下载到本地，然后 scp 上传
```

### Q3: 训练 loss 为 None
**解决**: 检查代码是否为最新版本，确保 `ft_type In` 设置正确

### Q4: 模型加载失败
**解决**: 检查 ckpt 目录
```bash
ls -la ckpt/
```

### Q5: Loss 不下降
**解决**: 这是正常的，MAE loss 初期可能较高，模型会逐渐学习

---

## 8. 重要参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--batch_size` | 批大小，RTX 3090 建议 8 | 8 |
| `--learning_rate` | 学习率 | 0.0001 |
| `--train_epochs` | 训练轮数 | 20 |
| `--seq_len` | 输入序列长度 | 96 |
| `--pred_len` | 预测序列长度 | 96/720 |
| `--rar_arch` | RAR 模型架构 | rar_l_0.3b |
| `--ft_type` | **微调类型，重要！** | `In`（Inpainting 模式，冻结 RAR） |

### ft_type 选项说明

| ft_type | 说明 | 训练对象 |
|---------|------|----------|
| `In` | **Inpainting 模式** | 只训练 VQ Tokenizer |
| `ln` | 仅 RMSNorm 可训练 | RAR GPT 部分可训练 |
| `full` | 所有参数可训练 | RAR GPT + VQ Tokenizer |

**重要**: 请使用 `--ft_type In` 进行训练

---

## 9. 训练输出

训练完成后，结果保存在:
```
VisionTSRAR/long_term_tsf/checkpoints/
VisionTSRAR/long_term_tsf/results/
```

---

## 10. 快速验证

在开始正式训练前，可以先运行测试脚本验证:
```bash
cd VisionTSRAR
python test_real_etth1.py
```

应该看到:
```
✅ 所有测试通过！可以开始训练！
```

---

## 11. 代码改动记录

详细改动请参考 [CHANGELOG.md](./CHANGELOG.md)

### 核心改动：
1. Loss 从 CE + MAE 混合改为纯 MAE
2. Loss 计算从整图 MAE 改为历史 + 未来分开 MAE
3. 训练时打乱 token 顺序（历史 raster，未来 random）
4. RAR GPT 完全冻结（`ft_type=In`）
5. 只训练 VQ Tokenizer 的 decoder 部分
