"""
VisionTSRAR 工具函数模块

包含：
- download_file: 通用文件下载（带进度条）
- safe_resize: 兼容不同 torchvision 版本的 Resize
- freq_to_seasonality_list: 时间序列频率→周期性映射
- download_rar_ckpt: 从 HuggingFace 下载 RAR GPT 预训练权重
- download_vq_ckpt: 从 HuggingFace 下载 VQ Tokenizer 预训练权重
"""

import inspect
import os
import requests

import pandas as pd
from torchvision.transforms import Resize
from tqdm import tqdm

from huggingface_hub import hf_hub_download


def download_file(url, local_filename):
    """
    从URL下载文件到本地（带进度条显示）
    
    用于自动下载预训练权重文件
    
    Args:
        url: 远程文件URL
        local_filename: 本地保存路径（自动创建父目录）
    """
    response = requests.get(url, stream=True)
    os.makedirs(os.path.dirname(local_filename), exist_ok=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(local_filename, 'wb') as file:
        with tqdm(
            desc=f"Download: {local_filename}",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
            dynamic_ncols=True
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
                    bar.update(len(chunk))


def safe_resize(size, interpolation):
    """
    安全的Resize构造函数，兼容不同版本的torchvision
    
    torchvision 0.17+ 的Resize新增了antialias参数，
    此函数检测参数签名，自动处理兼容性问题。
    
    Args:
        size: 目标尺寸 (H, W) 或 int
        interpolation: 插值方法（如Image.BILINEAR）
    Returns:
        Resize变换对象
    """
    signature = inspect.signature(Resize)
    params = signature.parameters
    if 'antialias' in params:
        return Resize(size, interpolation, antialias=False)
    else:
        return Resize(size, interpolation)


# 时间序列频率→可能的周期性(季节性)映射表
# 每种频率对应多个可能的周期长度，VisionTSRAR会选择最合适的周期进行2D折叠
# 例如：
#   "H"（小时级数据）可能具有日周期(24)或周周期(168)
#   "D"（日级数据）可能具有周周期(7)、月周期(30)或年周期(365)
#   "M"（月度数据）可能具有季度周期(3)、半年周期(6)或年周期(12)
POSSIBLE_SEASONALITIES = {
    "S": [3600],  # 秒级数据: 1小时=3600秒
    "T": [1440, 10080],  # 分钟级数据: 1天=1440分钟 或 1周=10080分钟
    "H": [24, 168],  # 小时级数据: 1天=24小时 或 1周=168小时
    "D": [7, 30, 365],  # 日级数据: 1周=7天, 1月≈30天 或 1年=365天
    "W": [52, 4],  # 周级数据: 1年=52周 或 1月≈4周
    "M": [12, 6, 3],  # 月度数据: 1年=12月, 半年=6月 或 1季度=3月
    "B": [5],  # 工作日: 1周=5个工作日
    "Q": [4, 2],  # 季度数据: 1年=4季度 或 半年=2季度
}


def norm_freq_str(freq_str: str) -> str:
    """
    标准化频率字符串
    
    处理pandas频率字符串的特殊格式：
    - 去除"-"后缀（如"H-_start"→"H"）
    - 去除末尾的"S"（如"MS"→"M"，表示月初而非月末）
    
    Args:
        freq_str: pandas频率字符串
    Returns:
        标准化后的频率字符
    """
    base_freq = freq_str.split("-")[0]
    if len(base_freq) >= 2 and base_freq.endswith("S"):
        return base_freq[:-1]
    return base_freq


def freq_to_seasonality_list(freq: str, mapping_dict=None) -> int:
    """
    根据时间序列频率返回可能的周期性列表
    
    VisionTSRAR使用此函数自动确定periodicity参数。
    对于给定频率，返回所有可能的整除周期值，最后附加P=1（无周期）。
    
    例如：freq="H" → [24, 168, 1]
    - 24: 日周期（24小时）
    - 168: 周周期（168小时）
    - 1: 无显著周期性
    
    Args:
        freq: 时间序列频率字符串（如"H"、"D"、"M"等，也支持"3H"、"15min"等）
        mapping_dict: 自定义频率-周期映射表（默认使用POSSIBLE_SEASONALITIES）
    Returns:
        seasonality_list: 可能的周期值列表，最后始终包含1（无周期）
    """
    if mapping_dict is None:
        mapping_dict = POSSIBLE_SEASONALITIES
    offset = pd.tseries.frequencies.to_offset(freq)
    base_seasonality_list = mapping_dict.get(norm_freq_str(offset.name), [])
    seasonality_list = []
    for base_seasonality in base_seasonality_list:
        # 处理倍数频率：如"3H"的周期应该是 24/3=8, 168/3=56
        seasonality, remainder = divmod(base_seasonality, offset.n)
        if not remainder:  # 只保留能整除的周期
            seasonality_list.append(seasonality)
    seasonality_list.append(1)  # we append P=1 for those without significant periodicity
    return seasonality_list


# ============================================================
# RAR / VQ 预训练权重下载
# ============================================================

# HuggingFace 仓库配置
# RandAR 预训练权重托管在 HuggingFace 上
RAR_HF_REPO_ID = "ziqipang/RandAR"

# RAR GPT 权重文件名映射表（使用官方 RandAR 仓库的文件名）
RAR_CKPT_FILES = {
    "rar_l_0.3b": "randar_0.3b_llamagen_360k_bs_1024_lr_0.0004.safetensors",
}

# VQ Tokenizer 权重文件名（使用 LlamaGen 仓库的文件名）
VQ_CKPT_FILE = "vq_ds16_c2i.pt"


def download_rar_ckpt(ckpt_name, ckpt_dir='./ckpt/'):
    """
    从 HuggingFace 下载 RAR GPT 预训练权重
    
    Args:
        ckpt_name: RAR架构名称，如 "rar_l_0.3b"
        ckpt_dir: 本地存储目录
    
    Returns:
        ckpt_path: 下载后的本地文件路径
    """
    if ckpt_name not in RAR_CKPT_FILES:
        raise ValueError(f"Unknown RAR checkpoint: {ckpt_name}. Available: {list(RAR_CKPT_FILES.keys())}")
    
    ckpt_filename = RAR_CKPT_FILES[ckpt_name]
    ckpt_path = os.path.join(ckpt_dir, ckpt_filename)
    
    if not os.path.isfile(ckpt_path):
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Downloading RAR checkpoint '{ckpt_name}' from HuggingFace ({RAR_HF_REPO_ID})...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=RAR_HF_REPO_ID,
                filename=ckpt_filename,
                local_dir=ckpt_dir,
            )
            print(f"RAR checkpoint downloaded to: {downloaded_path}")
        except Exception as e:
            print(f"Failed to download from HuggingFace: {e}")
            print(f"Please manually download '{ckpt_filename}' from https://huggingface.co/{RAR_HF_REPO_ID}")
            print(f"and place it at: {ckpt_path}")
            raise
    
    return ckpt_path


def download_vq_ckpt(ckpt_dir='./ckpt/'):
    """
    从 HuggingFace 下载 VQ Tokenizer 预训练权重
    
    Args:
        ckpt_dir: 本地存储目录
    
    Returns:
        ckpt_path: 下载后的本地文件路径
    """
    ckpt_path = os.path.join(ckpt_dir, VQ_CKPT_FILE)
    
    if not os.path.isfile(ckpt_path):
        os.makedirs(ckpt_dir, exist_ok=True)
        # VQ Tokenizer 来自 LlamaGen 仓库
        print(f"Downloading VQ Tokenizer checkpoint from HuggingFace (FoundationVision/LlamaGen)...")
        try:
            downloaded_path = hf_hub_download(
                repo_id="FoundationVision/LlamaGen",
                filename=VQ_CKPT_FILE,
                local_dir=ckpt_dir,
            )
            print(f"VQ Tokenizer checkpoint downloaded to: {downloaded_path}")
        except Exception as e:
            print(f"Failed to download from HuggingFace: {e}")
            print(f"Please manually download '{VQ_CKPT_FILE}' from https://huggingface.co/FoundationVision/LlamaGen")
            print(f"and place it at: {ckpt_path}")
            raise
    
    return ckpt_path
