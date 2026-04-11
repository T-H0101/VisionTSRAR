#!/bin/bash
# DLinear基线测试 - ETTh1数据集
# 在Time-Series-Library框架中运行

cd /Users/tian/Desktop/VisionTS/Time-Series-Library

# 创建结果目录
mkdir -p ../VisionTSRAR/result/dlinear

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_96_96 \
    --model DLinear \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 8 \
    --learning_rate 0.005 \
    --train_epochs 1 \
    --use_gpu False \
    --devices 0 \
    2>&1 | tee ../VisionTSRAR/result/dlinear/ETTh1_96_96.log

echo "DLinear测试完成，结果保存在 ../VisionTSRAR/result/dlinear/"
