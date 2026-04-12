# VisionTSRAR GPU 服务器训练指南

## 📋 目录
1. [环境准备](#1-环境准备)
2. [代码上传](#2-代码上传)
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

## 2. 代码上传

### 方式一：GitHub
```bash
# 在本地创建 git 仓库（如果还没有）
cd VisionTSRAR
git init
git add .
git commit -m "VisionTSRAR: 端到端时序预测模型"

# 添加远程仓库（替换为你的仓库地址）
git remote add origin https://github.com/YOUR_USERNAME/VisionTSRAR.git
git push -u origin main
```

### 方式二：SCP 直接上传
```bash
# 在本地执行
scp -r /Users/tian/Desktop/VisionTS/VisionTSRAR user@server:/path/to/VisionTSRAR
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
  --rar_arch rar_l_0.3b \
  --train_epochs 20 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --use_gpu True \
  --gpu 0 \
  > training_etth1.log 2>&1 &
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
- **初始 MSE Loss**: ~0.26 (标准化空间)
- **DLinear 对比**: ~0.4
- **我们的模型**: 应该 < 0.4

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
**解决**: 已修复，确保代码是最新的

### Q4: 模型加载失败
**解决**: 检查 ckpt 目录
```bash
ls -la ckpt/
```

### Q5: 梯度流问题
**解决**: 已修复，使用最新的 model.py 和 models_rar.py

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
