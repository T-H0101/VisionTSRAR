# VisionTSRAR 服务器部署 - 极简指南

## 🚀 5 步快速部署

### 步骤 1：在本地打包
```bash
cd /Users/tian/Desktop/VisionTS/VisionTSRAR
bash package_for_server.sh
```

### 步骤 2：上传到服务器
```bash
scp VisionTSRAR_deploy.zip your_username@server_ip:/home/your_username/
```

### 步骤 3：登录服务器并解压
```bash
ssh your_username@server_ip
unzip VisionTSRAR_deploy.zip
cd VisionTSRAR
```

### 步骤 4：一键自动配置
```bash
chmod +x deploy/setup_server.sh
bash deploy/setup_server.sh
```

脚本会自动：
- ✅ 检查 Python/pip 环境
- ✅ 配置 PyPI 清华源（下载速度提升 10 倍+）
- ✅ 安装 PyTorch（选择 CUDA 版本）
- ✅ 安装所有 Python 依赖
- ✅ 下载模型权重

### 步骤 5：开始训练
```bash
# 使用 tmux 后台运行（推荐）
tmux new -s train
cd long_term_tsf
python3 run.py --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_96_96 --model VisionTSRAR --data ETTh1 --features M --seq_len 96 --pred_len 96 --rar_arch rar_l_0.3b --ft_type ln --train_epochs 20 --batch_size 32 --learning_rate 0.0001 --des 'Exp'

# 按 Ctrl+B 然后 D 退出 tmux
```

---

## 📋 常用命令

### 查看训练进度
```bash
# 重新连接 tmux
tmux attach -t train

# 查看日志
tail -f long_term_tsf/logs/ETTh1_96_96/log.txt

# 监控 GPU
watch -n 1 nvidia-smi
```

### 训练其他数据集
```bash
cd long_term_tsf

# ETTh2
python3 run.py --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model_id ETTh2_96_96 --model VisionTSRAR --data ETTh2 --features M --seq_len 96 --pred_len 96 --rar_arch rar_l_0.3b --ft_type ln --train_epochs 20 --batch_size 32 --learning_rate 0.0001 --des 'Exp'

# ETTm1
python3 run.py --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id ETTm1_96_96 --model VisionTSRAR --data ETTm1 --features M --seq_len 96 --pred_len 96 --rar_arch rar_l_0.3b --ft_type ln --train_epochs 20 --batch_size 32 --learning_rate 0.0001 --des 'Exp'
```

### 下载训练结果
```bash
# 在本地电脑执行
scp your_username@server_ip:/home/your_username/VisionTSRAR/long_term_tsf/logs/ETTh1_96_96/log.txt ./
scp your_username@server_ip:/home/your_username/VisionTSRAR/long_term_tsf/checkpoints/ETTh1_96_96/*.pt ./
```

---

## 💡 常见问题

### Q1: 没有 Conda 怎么办？
**A**: 不需要 Conda！我们使用全局 Python + pip3 安装。

### Q2: 为什么要用清华源？
**A**: 服务器通常在国内，清华源比 PyPI 官方源快 10 倍以上。

### Q3: 训练中断了怎么办？
**A**: 使用 tmux 就不会中断，即使断开 SSH 连接也会继续运行。

### Q4: GPU 内存不够怎么办？
**A**: 减小 batch_size：`--batch_size 16` 或 `--batch_size 8`

### Q5: 如何停止训练？
**A**: 
```bash
# 找到进程
ps aux | grep python

# 杀掉进程
kill -9 <PID>
```

---

## 🎯 总结

### 最简单的方式
```bash
# 本地打包
bash package_for_server.sh

# 上传
scp VisionTSRAR_deploy.zip user@server:/home/user/

# 服务器端
ssh user@server
unzip VisionTSRAR_deploy.zip
cd VisionTSRAR
bash deploy/setup_server.sh

# 训练
tmux new -s train
cd long_term_tsf
python3 run.py --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_96_96 --model VisionTSRAR --data ETTh1 --features M --seq_len 96 --pred_len 96 --rar_arch rar_l_0.3b --ft_type ln --train_epochs 20 --batch_size 32 --learning_rate 0.0001 --des 'Exp'
# Ctrl+B 然后 D
```

就这么简单！🎉

---

**更新时间**: 2026-04-12  
**版本**: VisionTSRAR v0.4.0 (Server Ready)
