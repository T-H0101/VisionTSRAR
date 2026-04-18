#!/bin/bash
# VisionTSRAR 优化训练脚本
# 优化目标：降低 loss + 缩短训练时间

echo "=========================================="
echo "VisionTSRAR 优化训练"
echo "=========================================="

# 激活环境
source /opt/miniconda3/bin/activate 1d_tokenizer

# 进入训练目录
cd /Users/tian/Desktop/VisionTS/VisionTSRAR/long_term_tsf

echo "开始训练 (优化配置)..."
echo ""

# 运行训练
python run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model VisionTSRAR \
    --model_id optimized_training \
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
    --gradient_accumulation_steps 2 \
    --lradj cosine \
    --itr 1 \
    --train_epochs 10 \
    --learning_rate 0.0001 \
    --skip_validation 1

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="