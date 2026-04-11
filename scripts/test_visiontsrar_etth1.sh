#!/bin/bash
# VisionTSRAR测试 - ETTh1数据集，训练1轮
# 使用long_term_tsf框架

cd /Users/tian/Desktop/VisionTS/VisionTSRAR/long_term_tsf

# 创建结果目录
mkdir -p ../result/visiontsrar

# VisionTSRAR参数说明：
# --model VisionTSRAR: 使用VisionTSRAR模型
# --seq_len 96: 输入序列长度（从96开始，避免MPS内存问题）
# --pred_len 96: 预测长度
# --periodicity 24: ETTh1的周期（24小时）
# --train_epochs 1: 训练1轮
# --num_inference_steps 88: RAR推理步数
# --batch_size 8: Mac Air M4 32G

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_96_96 \
    --model VisionTSRAR \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --periodicity 24 \
    --norm_const 0.4 \
    --align_const 0.4 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 8 \
    --num_workers 0 \
    --train_epochs 1 \
    --learning_rate 0.001 \
    --device mps \
    --rar_arch rar_l_0.3b \
    --num_inference_steps 88 \
    --position_order raster \
    --temperature 1.0 \
    --top_k 0 \
    --top_p 1.0 \
    2>&1 | tee ../result/visiontsrar/ETTh1_96_96.log

echo "VisionTSRAR测试完成，结果保存在 ../result/visiontsrar/"
