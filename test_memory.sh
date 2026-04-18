#!/bin/bash
# VisionTSRAR 显存优化测试脚本
# 使用方法: 在服务器上运行 bash test_memory.sh

echo "=========================================="
echo "VisionTSRAR 显存优化测试"
echo "=========================================="

# 激活环境
source /opt/miniconda3/bin/activate 1d_tokenizer

# 进入训练目录
cd /Users/tian/Desktop/VisionTS/VisionTSRAR/long_term_tsf

echo "开始训练测试 (batch_size=16, 10个batch)..."
echo ""

# 运行训练
python run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model VisionTSRAR \
    --model_id test_memory_optimization \
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
    --itr 1 \
    --train_epochs 2 \
    --learning_rate 0.0001 \
    --skip_validation 1

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="