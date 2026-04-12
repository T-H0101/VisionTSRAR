#!/bin/bash

# VisionTSRAR 服务器快速部署脚本（使用内置 PyTorch 环境）
# 服务器配置：pytorch/2.9.0-cuda13.0-py3.11.14-内置nvcc
# 使用方法：bash setup_server.sh

set -e  # 遇到错误立即退出

echo "========================================="
echo "VisionTSRAR 服务器环境配置"
echo "服务器环境：pytorch/2.9.0-cuda13.0-py3.11.14"
echo "========================================="

# 1. 检查并激活 PyTorch 环境
echo "检查 PyTorch 环境..."

# 检查是否有 module 命令（大多数服务器使用 module 管理环境）
if command -v module &> /dev/null; then
    echo "检测到 module 命令，尝试加载 PyTorch 模块..."
    
    # 尝试加载服务器提供的 PyTorch 模块
    if module avail 2>&1 | grep -q "pytorch/2.9.0"; then
        module load pytorch/2.9.0-cuda13.0-py3.11.14
        echo "✓ 已加载 PyTorch 模块"
    else
        echo "⚠️  未找到精确匹配的模块，请手动加载"
        echo "可用的 PyTorch 模块："
        module avail pytorch
        echo ""
        read -p "请输入要加载的模块名（或直接回车跳过）: " module_name
        if [ -n "$module_name" ]; then
            module load $module_name
            echo "✓ 已加载 $module_name"
        fi
    fi
else
    echo "⚠️  未检测到 module 命令，使用当前环境"
fi

# 验证 Python 和 PyTorch
echo ""
echo "验证环境..."
python3 --version
python3 -c "import torch; print(f'✓ PyTorch: {torch.__version__}'); print(f'✓ CUDA: {torch.cuda.is_available()}')"
python3 -c "import torch; import os; os.system('echo ✓ GPU: ' + torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None"

# 2. 配置清华源（加速下载）
echo ""
echo "========================================="
echo "配置 PyPI 清华源..."
echo "========================================="

mkdir -p ~/.pip
cat > ~/.pip/pip.conf << 'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF

echo "✓ 清华源已配置"

# 3. 安装 Python 依赖（使用清华源）
echo ""
echo "========================================="
echo "安装 Python 依赖..."
echo "========================================="

cd long_term_tsf
pip3 install -r requirements.txt

echo "✓ 依赖已安装"

# 4. 下载模型权重
echo ""
echo "========================================="
echo "下载模型权重..."
echo "========================================="

cd ..

if [ -f "ckpt/download_ckpt.py" ]; then
    python3 ckpt/download_ckpt.py
    echo "✓ 模型权重已下载"
else
    echo "警告：未找到下载脚本，请手动下载权重到 ckpt/ 目录"
fi

# 5. 验证安装
echo ""
echo "========================================="
echo "验证安装..."
echo "========================================="

cd long_term_tsf

python3 -c "
import torch
import numpy as np
import pandas as pd
print('✓ 所有依赖正常')
print(f'  - PyTorch: {torch.__version__}')
print(f'  - CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  - GPU: {torch.cuda.get_device_name(0)}')
"

# 6. 完成
echo ""
echo "========================================="
echo "✅ 部署完成！"
echo "========================================="
echo ""
echo "📝 使用前记得激活 PyTorch 环境："
echo "   module load pytorch/2.9.0-cuda13.0-py3.11.14"
echo ""
echo "🚀 使用方法："
echo "1. 激活环境（如果需要）"
echo "2. 进入目录：cd VisionTSRAR/long_term_tsf"
echo "3. 开始训练：python3 run.py --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_96_96 --model VisionTSRAR --data ETTh1 --features M --seq_len 96 --pred_len 96 --rar_arch rar_l_0.3b --train_epochs 20 --batch_size 32 --learning_rate 0.0001"
echo ""
echo "后台运行（推荐）："
echo "tmux new -s train"
echo "cd VisionTSRAR/long_term_tsf"
echo "python3 run.py --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_96_96 --model VisionTSRAR --data ETTh1 --features M --seq_len 96 --pred_len 96 --rar_arch rar_l_0.3b --train_epochs 20 --batch_size 32 --learning_rate 0.0001"
echo "# Ctrl+B 然后 D 退出"
echo ""
echo "查看日志：tail -f logs/ETTh1_96_96/log.txt"
echo "========================================="
