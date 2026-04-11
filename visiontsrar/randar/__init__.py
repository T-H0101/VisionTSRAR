"""
RandAR 核心组件包

包含从原始 RandAR 项目迁移的所有核心模块：
- randar_gpt: RandAR Transformer 模型（随机顺序自回归生成器）
- llamagen_gpt: LlamaGen 基础 Transformer 组件（LabelEmbedder, RMSNorm, RoPE 等）
- tokenizer: VQ-VAE Tokenizer（图像↔离散token转换）
- generate: 采样与生成逻辑（top-k/top-p 采样等）
- utils: 工具函数（DropPath, interleave_tokens, 并行解码调度等）
"""

from .randar_gpt import RandARTransformer
from .tokenizer import VQModel
from .llamagen_gpt import (
    LabelEmbedder,
    CaptionEmbedder,
    RMSNorm,
    FeedForward,
    KVCache,
    find_multiple,
    apply_rotary_emb,
    precompute_freqs_cis_2d,
)
from .generate import (
    sample,
    top_k_top_p_filtering,
    logits_to_probs,
)
from .utils import (
    DropPath,
    interleave_tokens,
    calculate_num_query_tokens_for_parallel_decoding,
)

__all__ = [
    # 核心模型
    "RandARTransformer",
    "VQModel",
    # 嵌入与归一化
    "LabelEmbedder",
    "CaptionEmbedder",
    "RMSNorm",
    # 前馈与缓存
    "FeedForward",
    "KVCache",
    # 辅助函数
    "find_multiple",
    "apply_rotary_emb",
    "precompute_freqs_cis_2d",
    # 采样函数
    "sample",
    "top_k_top_p_filtering",
    "logits_to_probs",
    # 工具函数
    "DropPath",
    "interleave_tokens",
    "calculate_num_query_tokens_for_parallel_decoding",
]
