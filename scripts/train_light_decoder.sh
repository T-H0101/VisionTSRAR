#!/bin/bash
# VisionTSRAR 轻量Decoder训练脚本
# 使用方法: bash scripts/train_light_decoder.sh

# ============================================================
# 配置区域 - 根据需要修改
# ============================================================
MODEL="VisionTSRAR"
DATA="ETTh1"
SEQ_LEN=96
PRED_LEN=96
BATCH_SIZE=16              # 轻量Decoder支持更大batch
TRAIN_EPOCHS=10            # 训练轮数
LEARNING_RATE=0.0001
FT_TYPE="In"               # 训练Decoder模式
USE_LIGHT_DECODER=1        # 使用轻量Decoder
LIGHT_DECODER_CHANNELS=64  # Decoder通道数

# ============================================================
# 训练命令
# ============================================================
python -m run \
    --task_name long_term_forecast \
    --is_training 1 \
    --model ${MODEL} \
    --model_id light_decoder_${DATA}_sl${SEQ_LEN}_pl${PRED_LEN} \
    --data ${DATA} \
    --root_path ./dataset/ETT-small/ \
    --data_path ${DATA}.csv \
    --features M \
    --seq_len ${SEQ_LEN} \
    --label_len 48 \
    --pred_len ${PRED_LEN} \
    --seasonal_patterns Monthly \
    --use_lightweight_decoder \
    --lightweight_decoder_channels ${LIGHT_DECODER_CHANNELS} \
    --ft_type ${FT_TYPE} \
    --use_amp \
    --batch_size ${BATCH_SIZE} \
    --itr 1 \
    --train_epochs ${TRAIN_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --des light_decoder_ch${LIGHT_DECODER_CHANNELS}_bs${BATCH_SIZE}

echo "训练完成！"
