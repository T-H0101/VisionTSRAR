#!/usr/bin/env python3
"""
训练时间估算工具

使用方法：
    python estimate_training_time.py [数据集] [预测长度] [GPU 型号]
    
示例：
    python estimate_training_time.py ETTh1 96 RTX4090
    python estimate_training_time.py ETTh1 96 A100
"""

import sys

# 训练配置
TRAIN_CONFIG = {
    'train_epochs': 20,
    'batch_size': 32,
    'seq_len': 96,
    'pred_len': 96,
    'model': 'VisionTSRAR-0.3B',
}

# 不同 GPU 的性能估算（每秒处理的 batch 数）
GPU_PERFORMANCE = {
    # NVIDIA 消费级
    'rtx3090': 2.5,      # ~2.5 batch/s
    'rtx4090': 4.0,      # ~4.0 batch/s
    'rtx3080': 1.8,
    'rtx4080': 3.2,
    
    # NVIDIA 专业级
    'a100': 6.0,         # ~6.0 batch/s
    'a100-80g': 6.0,
    'v100': 2.0,
    'a40': 4.5,
    'a10': 2.5,
    
    # 默认值（保守估计）
    'default': 2.0,
}

# 不同数据集的大小
DATASET_SIZE = {
    'etth1': {'train_samples': 8500, 'test_samples': 2500},
    'etth2': {'train_samples': 8500, 'test_samples': 2500},
    'ettm1': {'train_samples': 34000, 'test_samples': 10000},
    'ettm2': {'train_samples': 34000, 'test_samples': 10000},
}


def estimate_training_time(dataset='ETTh1', gpu_name='RTX4090'):
    """
    估算训练时间
    
    Args:
        dataset: 数据集名称
        gpu_name: GPU 型号
    
    Returns:
        (total_time_hours, time_per_epoch)
    """
    # 获取数据集大小
    dataset = dataset.lower()
    if dataset not in DATASET_SIZE:
        print(f"⚠️  未知数据集 {dataset}，使用 ETTh1 作为参考")
        dataset = 'etth1'
    
    train_samples = DATASET_SIZE[dataset]['train_samples']
    
    # 计算总 batch 数
    batch_size = TRAIN_CONFIG['batch_size']
    batches_per_epoch = train_samples // batch_size
    total_batches = batches_per_epoch * TRAIN_CONFIG['train_epochs']
    
    # 获取 GPU 性能
    gpu_name_lower = gpu_name.lower().replace(' ', '')
    if gpu_name_lower not in GPU_PERFORMANCE:
        print(f"⚠️  未知 GPU {gpu_name}，使用默认性能")
        batches_per_second = GPU_PERFORMANCE['default']
    else:
        batches_per_second = GPU_PERFORMANCE[gpu_name_lower]
    
    # 计算时间
    total_seconds = total_batches / batches_per_second
    total_hours = total_seconds / 3600
    time_per_epoch = total_hours / TRAIN_CONFIG['train_epochs']
    
    return total_hours, time_per_epoch, batches_per_epoch


def print_training_info(dataset='ETTh1', gpu_name='RTX4090'):
    """打印训练信息"""
    
    print("=" * 70)
    print("📊 VisionTSRAR 训练时间估算")
    print("=" * 70)
    print()
    
    # 打印配置
    print("📋 训练配置:")
    print(f"  - 模型：{TRAIN_CONFIG['model']}")
    print(f"  - 数据集：{dataset}")
    print(f"  - 训练轮数：{TRAIN_CONFIG['train_epochs']} epochs")
    print(f"  - Batch Size: {TRAIN_CONFIG['batch_size']}")
    print(f"  - 序列长度：{TRAIN_CONFIG['seq_len']}")
    print(f"  - 预测长度：{TRAIN_CONFIG['pred_len']}")
    print()
    
    # 估算时间
    total_hours, time_per_epoch, batches_per_epoch = estimate_training_time(dataset, gpu_name)
    
    print("⏱️  时间估算:")
    print(f"  - 每个 epoch: {time_per_epoch:.2f} 小时 ({time_per_epoch*60:.1f} 分钟)")
    print(f"  - 总训练时间：{total_hours:.2f} 小时")
    
    if total_hours < 1:
        print(f"  - 大约：{total_hours*60:.1f} 分钟")
    elif total_hours < 24:
        print(f"  - 大约：{total_hours:.1f} 小时")
    else:
        print(f"  - 大约：{total_hours/24:.1f} 天")
    print()
    
    # 打印结果位置
    print("📁 训练结果位置:")
    print(f"  - 日志文件：~/VisionTSRAR/long_term_tsf/logs/{dataset}_96_96/log.txt")
    print(f"  - 模型权重：~/VisionTSRAR/long_term_tsf/checkpoints/{dataset}_96_96/")
    print(f"  - 预测结果：~/VisionTSRAR/long_term_tsf/results/{dataset}_96_96/")
    print()
    
    # 查看进度的命令
    print("👀 查看训练进度:")
    print("  # 实时查看日志")
    print(f"  tail -f ~/VisionTSRAR/long_term_tsf/logs/{dataset}_96_96/log.txt")
    print()
    print("  # 监控 GPU 使用情况")
    print("  watch -n 1 nvidia-smi")
    print()
    print("  # 查看已用时间")
    print(f"  cat ~/VisionTSRAR/long_term_tsf/logs/{dataset}_96_96/log.txt | grep 'Epoch'")
    print()
    
    # 下载结果的命令
    print("📥 下载训练结果到本地:")
    print(f"  # 下载日志")
    print(f"  scp your_username@server:~/VisionTSRAR/long_term_tsf/logs/{dataset}_96_96/log.txt ./")
    print()
    print(f"  # 下载模型权重")
    print(f"  scp -r your_username@server:~/VisionTSRAR/long_term_tsf/checkpoints/{dataset}_96_96 ./")
    print()
    
    print("=" * 70)
    print()


if __name__ == '__main__':
    # 解析命令行参数
    if len(sys.argv) >= 2:
        dataset = sys.argv[1]
    else:
        dataset = 'ETTh1'
    
    if len(sys.argv) >= 3:
        gpu_name = sys.argv[2]
    else:
        gpu_name = 'RTX4090'
    
    # 打印信息
    print_training_info(dataset, gpu_name)
    
    # 打印支持的 GPU 列表
    print("💡 支持的 GPU 型号:")
    print("  消费级：RTX3090, RTX4090, RTX3080, RTX4080")
    print("  专业级：A100, A100-80G, V100, A40, A10")
    print()
    print("💡 示例:")
    print("  python estimate_training_time.py ETTh1 96 RTX4090")
    print("  python estimate_training_time.py ETTh1 96 A100")
