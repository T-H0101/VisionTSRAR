#!/usr/bin/env python3
"""
VisionTSRAR 预训练权重自动下载脚本

从 HuggingFace (yucornell/RandAR) 下载 VQ Tokenizer 和 RandAR GPT 的预训练权重。

用法:
    python download_ckpt.py [--ckpt_dir ./ckpt/] [--rar_arch rar_l_0.3b] [--vq_only] [--rar_only]

示例:
    # 下载所有权重（默认）
    python download_ckpt.py

    # 仅下载 VQ Tokenizer 权重
    python download_ckpt.py --vq_only

    # 指定下载目录
    python download_ckpt.py --ckpt_dir /path/to/ckpt/

    # 指定 RAR 架构
    python download_ckpt.py --rar_arch rar_l_0.3b
"""

import argparse
import os
import sys

# 将项目根目录加入搜索路径，以便导入 visiontsrar
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from visiontsrar.util import download_rar_ckpt, download_vq_ckpt


def main():
    parser = argparse.ArgumentParser(description='下载 VisionTSRAR 预训练权重')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt/',
                        help='权重文件存储目录（默认: ./ckpt/）')
    parser.add_argument('--rar_arch', type=str, default='rar_l_0.3b',
                        choices=['rar_l_0.3b'],
                        help='RAR GPT 架构（默认: rar_l_0.3b）')
    parser.add_argument('--vq_only', action='store_true',
                        help='仅下载 VQ Tokenizer 权重')
    parser.add_argument('--rar_only', action='store_true',
                        help='仅下载 RAR GPT 权重')

    args = parser.parse_args()

    # 确保 ckpt_dir 是绝对路径
    ckpt_dir = os.path.abspath(args.ckpt_dir)

    print("=" * 60)
    print("VisionTSRAR 预训练权重下载工具")
    print("=" * 60)
    print(f"权重目录: {ckpt_dir}")
    print(f"HuggingFace 仓库: yucornell/RandAR")
    print()

    if not args.rar_only:
        print("[1/2] 下载 VQ Tokenizer 权重...")
        try:
            vq_path = download_vq_ckpt(ckpt_dir=ckpt_dir)
            print(f"  ✓ VQ Tokenizer: {vq_path}")
        except Exception as e:
            print(f"  ✗ VQ Tokenizer 下载失败: {e}")
            if args.vq_only:
                sys.exit(1)
        print()

    if not args.vq_only:
        print(f"[2/2] 下载 RAR GPT ({args.rar_arch}) 权重...")
        try:
            rar_path = download_rar_ckpt(ckpt_name=args.rar_arch, ckpt_dir=ckpt_dir)
            print(f"  ✓ RAR GPT: {rar_path}")
        except Exception as e:
            print(f"  ✗ RAR GPT 下载失败: {e}")
            sys.exit(1)
        print()

    print("=" * 60)
    print("所有权重下载完成！")
    print("=" * 60)

    # 列出目录中的所有权重文件
    print(f"\n{ckpt_dir} 目录内容：")
    for f in os.listdir(ckpt_dir):
        fpath = os.path.join(ckpt_dir, f)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"  {f} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
