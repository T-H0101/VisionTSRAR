# VisionTSRAR 轻量Decoder训练命令（平衡版）
# 通道数64 + 4层，参数量约12M，checkpoint约50MB

# ============================================================
# 重要说明
# ============================================================
# 1. 只保存可训练参数（轻量Decoder），不保存冻结的RAR GPT
# 2. checkpoint大小约50MB，不会有I/O问题
# 3. 训练完成后会自动打印checkpoint大小

# ============================================================
# 训练命令
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
    --lradj cosine \
    --skip_validation 1 \
    --skip_test 1 \
    --des light_decoder_ch64_4layers

# ============================================================
# 测试命令
# ============================================================
python -m run \
    --task_name long_term_forecast \
    --is_training 0 \
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
    --batch_size 4 \
    --itr 1 \
    --des light_decoder_ch64_4layers

# ============================================================
# 配置说明
# ============================================================
# 通道数: 64
# 层数: 4层 (ch_mult=[1, 2, 4, 4])
# 参数量: 约12M
# checkpoint大小: 约50MB
# 训练显存: 约21GB (batch_size=16)
# 测试显存: 需要batch_size=4
