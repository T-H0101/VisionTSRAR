# VisionTSRAR 服务器部署指南

## 📦 1. 上传项目到服务器

### 方式 1：使用 scp 命令（推荐）
```bash
# 在本地电脑执行
cd /Users/tian/Desktop/VisionTS
zip -r VisionTSRAR.zip VisionTSRAR -x "VisionTSRAR/.git/*" "VisionTSRAR/long_term_tsf/dataset/*" "VisionTSRAR/ckpt/*"

# 上传到服务器（替换 your_username 和 server_ip）
scp VisionTSRAR.zip your_username@server_ip:/home/your_username/
```

### 方式 2：直接拖拽
- 如果使用 VSCode Remote：直接拖拽文件到服务器
- 如果使用 Xshell/FinalShell：使用 SFTP 功能上传

---

## 🔧 2. 服务器环境配置

### 步骤 1：登录服务器并解压
```bash
# SSH 登录服务器
ssh your_username@server_ip

# 解压项目
unzip VisionTSRAR.zip
cd VisionTSRAR
```

### 步骤 2：一键自动配置（推荐）
```bash
# 运行自动配置脚本
chmod +x deploy/setup_server.sh
bash deploy/setup_server.sh
```

脚本会自动：
1. 检查 Python/pip 环境
2. 配置 PyPI 清华源（加速下载）
3. 安装 PyTorch（根据你的选择）
4. 安装所有依赖
5. 下载模型权重

### 手动安装（可选）

如果不想用脚本，可以手动执行：

```bash
# 1. 配置清华源
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << 'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF

# 2. 安装 PyTorch（选择你的 CUDA 版本）
# CUDA 11.8（新 GPU，推荐）
pip3 install torch torchvision torchaudio --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# CUDA 11.7（旧 GPU）
pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# CPU 版本（仅测试）
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 安装依赖
cd long_term_tsf
pip3 install -r requirements.txt

# 4. 下载模型权重
cd ..
python3 ckpt/download_ckpt.py
```

---

## 🚀 3. 开始训练

### 简单训练命令（ETTh1 数据集）
```bash
cd long_term_tsf

python3 run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_visiontsrar_96_96 \
  --model VisionTSRAR \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --rar_arch rar_l_0.3b \
  --ft_type ln \
  --train_epochs 20 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --des 'Training'
```

### 后台运行（推荐，防止断线）
```bash
# 使用 nohup 后台运行
nohup python3 run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_visiontsrar_96_96 \
  --model VisionTSRAR \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --rar_arch rar_l_0.3b \
  --ft_type ln \
  --train_epochs 20 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --des 'Training' > training.log 2>&1 &

# 查看日志
tail -f training.log

# 查看进程
ps aux | grep python
```

### 使用 tmux（更推荐的后台管理方式）
```bash
# 创建新会话
tmux new -s train

# 在 tmux 中运行训练
python3 run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_visiontsrar_96_96 \
  --model VisionTSRAR \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --rar_arch rar_l_0.3b \
  --ft_type ln \
  --train_epochs 20 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --des 'Training'

# 按 Ctrl+B 然后按 D 退出 tmux（保持运行）

# 重新连接会话
tmux attach -t visiontsrar

# 关闭会话
tmux kill-session -t visiontsrar
```

---

## 📊 4. 查看训练结果

### 查看日志
```bash
# 实时查看训练日志
tail -f long_term_tsf/logs/ETTh1_visiontsrar_96_96/log.txt

# 查看训练结果
cat long_term_tsf/logs/ETTh1_visiontsrar_96_96/log.txt
```

### 查看模型权重
```bash
# 查看保存的模型
ls -lh long_term_tsf/checkpoints/

# 下载模型权重到本地（在本地电脑执行）
scp your_username@server_ip:/home/your_username/VisionTSRAR/long_term_tsf/checkpoints/*.pt ./
```

---

## 🛠️ 5. 常见问题

### Q1: CUDA out of memory
```bash
# 解决方案：减小 batch_size
--batch_size 16  # 或更小
```

### Q2: 训练太慢
```bash
# 解决方案：使用混合精度训练（如果代码支持）
# 或者减小模型大小
--rar_arch rar_s_0.1b  # 使用更小的模型
```

### Q3: 数据集不存在
```bash
# 确保数据集已下载
ls long_term_tsf/dataset/ETT-small/

# 如果没有，需要下载 ETT 数据集
```

### Q4: 权限问题
```bash
# 给脚本执行权限
chmod +x run.sh
```

---

## 📝 6. 完整命令示例

### 一键训练脚本
创建 `train.sh`：
```bash
#!/bin/bash

# 开始训练
cd long_term_tsf

python3 run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_visiontsrar_96_96 \
  --model VisionTSRAR \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --rar_arch rar_l_0.3b \
  --ft_type ln \
  --train_epochs 20 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --des 'ETTh1_Full_Training'
```

使用：
```bash
chmod +x train.sh
./train.sh
```

---

## 💡 7. 最佳实践

### 1. 使用虚拟环境
```bash
# 始终使用 Conda 环境
conda activate visiontsrar
```

### 2. 后台运行
```bash
# 使用 tmux 或 nohup，避免断线导致训练中断
```

### 3. 定期保存
```bash
# 确保模型定期保存（代码中已实现）
# 检查点目录：long_term_tsf/checkpoints/
```

### 4. 监控 GPU
```bash
# 查看 GPU 使用情况
watch -n 1 nvidia-smi

# 查看 GPU 内存
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### 5. 结果备份
```bash
# 定期备份结果到本地
scp your_username@server_ip:/home/your_username/VisionTSRAR/long_term_tsf/results/*.txt ./
```

---

## 🎯 总结

### 最少命令（推荐新手）
```bash
# 1. 登录服务器
ssh your_username@server_ip

# 2. 解压
unzip VisionTSRAR.zip
cd VisionTSRAR

# 3. 一键配置（自动安装所有依赖）
chmod +x deploy/setup_server.sh
bash deploy/setup_server.sh

# 4. 训练（使用 tmux）
tmux new -s train
cd long_term_tsf
python3 run.py --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_96_96 --model VisionTSRAR --data ETTh1 --features M --seq_len 96 --pred_len 96 --rar_arch rar_l_0.3b --ft_type ln --train_epochs 20 --batch_size 32 --learning_rate 0.0001 --des 'Exp'
# 按 Ctrl+B 然后 D 退出

# 5. 重新连接
tmux attach -t train
```

完成！🎉
