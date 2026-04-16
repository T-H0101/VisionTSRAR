# VisionTSRAR 轻量Decoder训练命令
# 直接复制粘贴到终端运行即可

# ============================================================
# 快速测试版（1个epoch，batch_size=8）
# ============================================================
python -m run \
    --task_name long_term_forecast \
    --is_training 1 \
    --model VisionTSRAR \
    --model_id test_light_decoder \
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
    --train_epochs 1 \
    --learning_rate 0.0001 \
    --des light_decoder_test

# ============================================================
# 完整训练版（10个epoch，batch_size=16）
# ============================================================
python -m run \
    --task_name long_term_forecast \
    --is_training 1 \
    --model VisionTSRAR \
    --model_id light_decoder_ETTh1_sl96_pl96 \
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
    --batch_size 16 \
    --itr 1 \
    --train_epochs 10 \
    --learning_rate 0.0001 \
    --des light_decoder_ch64_bs16

# ============================================================
# 其他数据集示例
# ============================================================

# ETTh2
python -m run \
    --task_name long_term_forecast \
    --is_training 1 \
    --model VisionTSRAR \
    --model_id light_decoder_ETTh2_sl96_pl96 \
    --data ETTh2 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --seasonal_patterns Monthly \
    --use_lightweight_decoder \
    --lightweight_decoder_channels 64 \
    --ft_type In \
    --use_amp \
    --batch_size 16 \
    --itr 1 \
    --train_epochs 10 \
    --learning_rate 0.0001 \
    --des light_decoder_ch64_bs16

# ETTm1
python -m run \
    --task_name long_term_forecast \
    --is_training 1 \
    --model VisionTSRAR \
    --model_id light_decoder_ETTm1_sl96_pl96 \
    --data ETTm1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --seasonal_patterns Monthly \
    --use_lightweight_decoder \
    --lightweight_decoder_channels 64 \
    --ft_type In \
    --use_amp \
    --batch_size 16 \
    --itr 1 \
    --train_epochs 10 \
    --learning_rate 0.0001 \
    --des light_decoder_ch64_bs16

# Weather (需要更大的batch_size可能需要调整)
python -m run \
    --task_name long_term_forecast \
    --is_training 1 \
    --model VisionTSRAR \
    --model_id light_decoder_Weather_sl96_pl96 \
    --data Weather \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
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
    --train_epochs 10 \
    --learning_rate 0.0001 \
    --des light_decoder_ch64_bs8
