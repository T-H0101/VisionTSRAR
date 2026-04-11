"""
RandAR 基础 Transformer 组件模块

本模块包含从 LlamaGen 迁移的 Transformer 基础组件，为 RandAR 提供底层构建块：
- LabelEmbedder: 类别标签嵌入（支持 CFG 的随机丢弃）
- CaptionEmbedder: 文本描述嵌入（支持 CFG 的随机丢弃）
- MLP: 多层感知机（用于特征投影）
- RMSNorm: 均方根归一化（比 LayerNorm 更高效）
- FeedForward: SwiGLU 前馈网络（LLaMA 风格）
- KVCache: 键值缓存（加速自回归推理）
- Attention: 多头注意力（支持 GQA 分组查询注意力）
- TransformerBlock: Transformer 残差块
- Transformer: 完整的 LlamaGen Transformer（作为基线参考）
- 位置编码函数: precompute_freqs_cis, precompute_freqs_cis_2d, apply_rotary_emb

来源:
- LlamaGen: https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/models/gpt.py
- VQGAN: https://github.com/CompVis/taming-transformers
- DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
- nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
- LLaMA: https://github.com/facebookresearch/llama/blob/main/llama/model.py
- gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
- PixArt: https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
"""

from dataclasses import dataclass
from typing import Optional, List


import torch
import torch.nn as nn
from torch.nn import functional as F
from .utils import DropPath
from .generate import (
    top_k_top_p_filtering,
    sample,
    logits_to_probs,
    prefill,
    decode_one_token,
    decode_n_tokens,
)


def find_multiple(n: int, k: int):
    """
    找到大于等于 n 的最小的 k 的倍数。
    
    用于确保张量维度是某个数的倍数（如 8），以优化 GPU 计算效率。
    例如：find_multiple(13, 8) = 16, find_multiple(16, 8) = 16
    
    Args:
        n: 目标值
        k: 倍数的基数
    
    Returns:
        大于等于 n 的最小 k 的倍数
    """
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class ModelArgs:
    """
    LlamaGen Transformer 的模型配置参数。
    
    注意：RandAR 使用自己的参数配置，此类保留用于兼容性。
    """
    dim: int = 4096               # 隐藏维度
    n_layer: int = 32             # Transformer 层数
    n_head: int = 32              # 注意力头数
    n_kv_head: Optional[int] = None  # KV头数（GQA，None表示与n_head相同）
    multiple_of: int = 256        # FFN隐藏维度的倍数基准（使SwiGLU维度对齐2的幂次）
    ffn_dim_multiplier: Optional[float] = None  # FFN维度乘数
    rope_base: float = 10000      # RoPE 基数
    norm_eps: float = 1e-5        # 归一化 epsilon
    initializer_range: float = 0.02  # 参数初始化标准差

    token_dropout_p: float = 0.1   # Token dropout 概率
    attn_dropout_p: float = 0.0    # 注意力 dropout 概率
    resid_dropout_p: float = 0.1   # 残差 dropout 概率
    ffn_dropout_p: float = 0.1     # FFN dropout 概率
    drop_path_rate: float = 0.0    # Drop Path 概率

    num_classes: int = 1000        # 类别数（类条件生成）
    caption_dim: int = 2048        # 文本特征维度（文本条件生成）
    class_dropout_prob: float = 0.1  # 类别标签丢弃概率（用于 CFG）
    model_type: str = "c2i"        # 模型类型：c2i（类条件）或 t2i（文本条件）

    vocab_size: int = 16384        # VQ 码本大小
    cls_token_num: int = 1         # 条件 token 数量
    block_size: int = 256          # 图像 token 序列长度（16x16=256）
    max_batch_size: int = 32       # 最大批次大小
    max_seq_len: int = 2048        # 最大序列长度


#################################################################################
#                      嵌入层：类别标签                          #
#################################################################################
class LabelEmbedder(nn.Module):
    """
    类别标签嵌入器：将类别标签映射为向量表示，并支持 Classifier-Free Guidance (CFG)。
    
    CFG 的工作原理：
    - 训练时：以概率 class_dropout_prob 随机将标签替换为特殊的"无条件"标签
    - 推理时：同时计算有条件和无条件的预测，按 CFG 公式混合
    - 这使模型既能条件生成，也能无条件生成，从而实现引导生成
    
    无条件标签的索引为 num_classes（即码本大小），与正常类别标签不冲突。
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        """
        Args:
            num_classes: 类别数量（不含无条件标签）
            hidden_size: 嵌入维度
            dropout_prob: 标签丢弃概率（>0 时启用 CFG）
        """
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        # 多预留一个位置给无条件标签
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        随机丢弃标签以支持 Classifier-Free Guidance。
        
        Args:
            labels: 类别标签，shape (batch_size,)
            force_drop_ids: 强制丢弃掩码（用于推理时精确控制）
        
        Returns:
            处理后的标签（部分被替换为无条件标签索引）
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        """
        Args:
            labels: 类别标签索引
            train: 是否训练模式
            force_drop_ids: 强制丢弃控制
        
        Returns:
            标签嵌入，shape (batch_size, 1, hidden_size)
        """
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels).unsqueeze(1)
        return embeddings


#################################################################################
#                      嵌入层：文本特征                          #
#################################################################################
class CaptionEmbedder(nn.Module):
    """
    文本描述嵌入器：将文本特征映射为与模型维度匹配的向量序列，支持 CFG。
    
    与 LabelEmbedder 不同，文本条件输入已经是连续特征（来自文本编码器），
    需要通过 MLP 投影到模型维度。CFG 的无条件嵌入是可学习的参数。
    """

    def __init__(self, in_channels, hidden_size, uncond_prob, token_num=120):
        """
        Args:
            in_channels: 输入文本特征维度
            hidden_size: 模型隐藏维度
            uncond_prob: 无条件丢弃概率
            token_num: 文本 token 数量（默认120）
        """
        super().__init__()
        self.cap_proj = MLP(
            in_features=in_channels,
            hidden_features=hidden_size,
            out_features=hidden_size,
        )
        # 无条件嵌入：可学习参数，用于 CFG 推理
        self.register_buffer(
            "uncond_embedding",
            nn.Parameter(torch.randn(token_num, in_channels) / in_channels**0.5),
        )
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        随机丢弃文本特征以支持 CFG。
        
        Args:
            caption: 文本特征，shape (batch_size, token_num, in_channels)
            force_drop_ids: 强制丢弃控制
        
        Returns:
            处理后的文本特征（部分被替换为无条件嵌入）
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None], self.uncond_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        """
        Args:
            caption: 文本特征，shape (batch_size, token_num, in_channels)
            train: 是否训练模式
            force_drop_ids: 强制丢弃控制
        
        Returns:
            投影后的文本嵌入，shape (batch_size, token_num, hidden_size)
        """
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        embeddings = self.cap_proj(caption)
        return embeddings


class MLP(nn.Module):
    """
    简单的两层 MLP，使用 GELU 激活函数（tanh 近似版本）。
    
    主要用于 CaptionEmbedder 中的特征投影。
    """
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


#################################################################################
#                                  GPT 模型核心组件                               #
#################################################################################
class RMSNorm(torch.nn.Module):
    """
    均方根归一化（Root Mean Square Normalization）。
    
    与 LayerNorm 相比，RMSNorm 不需要计算均值，仅除以均方根，
    计算更高效且效果相近。被 LLaMA 和许多现代 Transformer 采用。
    
    公式: output = (x / sqrt(mean(x^2) + eps)) * weight
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        """
        Args:
            dim: 归一化维度
            eps: 防止除零的小常数
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    """
    SwiGLU 前馈网络（LLaMA 风格）。
    
    与标准 FFN 的区别：
    - 使用 SwiGLU 激活（SiLU 门控线性单元）代替 ReLU/GELU
    - 两个上投影矩阵 w1 和 w3，一个下投影矩阵 w2
    - 公式: output = w2(SiLU(w1(x)) * w3(x))
    
    隐藏维度计算遵循 LLaMA 的约定：4*dim → 2/3*dim → 对齐到 multiple_of 的倍数
    """
    def __init__(
        self, dim: int, ffn_dim_multiplier: int, multiple_of: int, ffn_dropout_p: float
    ):
        """
        Args:
            dim: 输入/输出维度
            ffn_dim_multiplier: FFN 维度乘数（可进一步缩放隐藏维度）
            multiple_of: 隐藏维度需对齐到的倍数
            ffn_dropout_p: Dropout 概率
        """
        super().__init__()
        hidden_dim = 4 * dim
        hidden_dim = int(2 * hidden_dim / 3)
        # 自定义维度乘数
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # 上投影（门控路径）
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # 上投影（值路径）
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # 下投影
        self.ffn_dropout = nn.Dropout(ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class KVCache(nn.Module):
    """
    键值缓存（Key-Value Cache），用于加速自回归推理。
    
    在自回归生成中，每一步都需要计算当前 token 与之前所有 token 的注意力。
    KV Cache 将之前计算过的 Key 和 Value 缓存起来，避免重复计算，
    将每步复杂度从 O(seq_len) 降低到 O(1)。
    
    缓存形状: (max_batch_size, n_head, max_seq_length, head_dim)
    """
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        """
        更新 KV Cache 并返回完整的 Key/Value 张量。
        
        Args:
            input_pos: 当前 token 的位置索引，shape [S]
            k_val: 当前步的 Key，shape [B, H, S, D]
            v_val: 当前步的 Value，shape [B, H, S, D]
        
        Returns:
            (keys, values): 更新后的完整 Key 和 Value
        """
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class Attention(nn.Module):
    """
    多头注意力模块，支持分组查询注意力（Grouped Query Attention, GQA）。
    
    GQA 允许 KV 头数少于查询头数，多个查询头共享同一组 KV，
    在保持性能的同时减少 KV Cache 的内存占用和计算量。
    
    例如：n_head=32, n_kv_head=8 时，每4个查询头共享1组 KV。
    
    推理时使用 KV Cache 加速，训练时计算完整注意力。
    """
    def __init__(
        self,
        dim: int,
        n_head: int,
        n_kv_head: int,
        attn_dropout_p: float,
        resid_dropout_p: float,
    ):
        """
        Args:
            dim: 隐藏维度
            n_head: 查询头数
            n_kv_head: KV 头数（None 表示与 n_head 相同，即标准 MHA）
            attn_dropout_p: 注意力 dropout
            resid_dropout_p: 残差 dropout
        """
        super().__init__()
        assert dim % n_head == 0
        self.dim = dim
        self.head_dim = dim // n_head
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        # 合并 QKV 投影以提高效率
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim

        self.wqkv = nn.Linear(dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.kv_cache = None

        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout_p)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor = None,
        input_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x: 输入张量，shape (batch_size, seq_len, dim)
            freqs_cis: 旋转位置编码，shape 随模式变化
            input_pos: 位置索引（推理时用于 KV Cache 更新）
            mask: 因果注意力掩码
        
        Returns:
            注意力输出，shape (batch_size, seq_len, dim)
        """
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)

        # 应用旋转位置编码
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)

        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv
        # GQA: 复制 KV 头以匹配查询头数
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        output = F.scaled_dot_product_attention(
            xq,
            keys,
            values,
            attn_mask=mask,
            is_causal=(
                True if mask is None else False
            ),  # mask=None 时使用内置因果掩码
            dropout_p=self.attn_dropout_p if self.training else 0,
        )

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        output = self.resid_dropout(self.wo(output))
        return output


class TransformerBlock(nn.Module):
    """
    Transformer 残差块：Attention + FFN，带 RMSNorm 和 DropPath。
    
    结构: x + DropPath(Attention(RMSNorm(x)))  →  h + DropPath(FFN(RMSNorm(h)))
    即 Pre-Norm 架构（先归一化再计算），比 Post-Norm 更稳定。
    """
    def __init__(
        self,
        dim=4096,
        n_layer=32,
        n_head=32,
        n_kv_head=None,
        multiple_of=256,
        ffn_dim_multiplier=None,
        rope_base=10000,
        norm_eps=1e-5,
        token_dropout_p=0.1,
        attn_dropout_p=0.0,
        resid_dropout_p=0.1,
        ffn_dropout_p=0.1,
        drop_path=0.0,
    ):
        super().__init__()
        self.attention = Attention(
            dim, n_head, n_kv_head, attn_dropout_p, resid_dropout_p
        )
        self.feed_forward = FeedForward(
            dim, ffn_dim_multiplier, multiple_of, ffn_dropout_p
        )
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int,
        mask: Optional[torch.Tensor] = None,
    ):
        h = x + self.drop_path(
            self.attention(self.attention_norm(x), freqs_cis, start_pos, mask)
        )
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out


class Transformer(nn.Module):
    """
    LlamaGen 风格的完整 Transformer 模型（基线版本）。
    
    这是原始 LlamaGen 的 Transformer 实现，用于类条件或文本条件的图像生成。
    RandAR 使用自己的 RandARTransformer 类（在 randar_gpt.py 中），
    本类保留用于兼容性和基线对比。
    
    核心特性：
    - Pre-Norm + RMSNorm
    - SwiGLU FFN
    - 2D 旋转位置编码（RoPE）
    - GQA 分组查询注意力
    - KV Cache 加速推理
    - 支持 Classifier-Free Guidance
    """
    def __init__(
        self,
        dim=4096,
        n_layer=32,
        n_head=32,
        n_kv_head=None,
        multiple_of=256,
        ffn_dim_multiplier=None,
        rope_base=10000,
        norm_eps=1e-5,
        initializer_range=0.02,
        token_dropout_p=0.1,
        attn_dropout_p=0.0,
        resid_dropout_p=0.1,
        ffn_dropout_p=0.1,
        drop_path_rate=0.0,
        num_classes=1000,
        caption_dim=2048,
        class_dropout_prob=0.1,
        model_type="c2i",
        vocab_size=16384,
        cls_token_num=1,
        block_size=256,
        max_batch_size=32,
        max_seq_len=2048,
    ):
        super().__init__()
        self.dim = dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.rope_base = rope_base
        self.norm_eps = norm_eps
        self.token_dropout_p = token_dropout_p
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout_p = resid_dropout_p
        self.ffn_dropout_p = ffn_dropout_p
        self.drop_path_rate = drop_path_rate
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.num_classes = num_classes
        self.model_type = model_type
        self.cls_token_num = cls_token_num
        
        # 条件嵌入
        if self.model_type == "c2i":
            self.cls_embedding = LabelEmbedder(num_classes, dim, class_dropout_prob)
        elif self.model_type == "t2i":
            self.cls_embedding = CaptionEmbedder(caption_dim, dim, class_dropout_prob)
        else:
            raise Exception("please check model type")
        
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.tok_dropout = nn.Dropout(token_dropout_p)

        # Transformer 层，使用线性递增的 DropPath 概率
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layer)]
        self.layers = torch.nn.ModuleList()
        for layer_id in range(n_layer):
            self.layers.append(
                TransformerBlock(
                    dim=dim,
                    n_layer=n_layer,
                    n_head=n_head,
                    n_kv_head=n_kv_head,
                    multiple_of=multiple_of,
                    ffn_dim_multiplier=ffn_dim_multiplier,
                    rope_base=rope_base,
                    norm_eps=norm_eps,
                    token_dropout_p=token_dropout_p,
                    attn_dropout_p=attn_dropout_p,
                    resid_dropout_p=resid_dropout_p,
                    ffn_dropout_p=ffn_dropout_p,
                    drop_path=dpr[layer_id],
                )
            )

        # 输出层
        self.norm = RMSNorm(dim, eps=norm_eps)
        self.output = nn.Linear(dim, vocab_size, bias=False)

        # 2D 旋转位置编码
        grid_size = int(self.block_size**0.5)
        assert grid_size * grid_size == self.block_size
        self.freqs_cis = precompute_freqs_cis_2d(
            grid_size, self.dim // self.n_head, self.rope_base, self.cls_token_num
        )

        # KV Cache 状态
        self.max_batch_size = -1
        self.max_seq_length = -1

        # 初始化
        self.initializer_range = initializer_range
        self.initialize_weights()

    def initialize_weights(self):
        """初始化所有权重：Linear 和 Embedding 使用正态分布，输出层初始化为0"""
        self.apply(self._init_weights)
        # 输出层零初始化：使模型初始输出均匀分布，训练更稳定
        nn.init.constant_(self.output.weight, 0)

    def _init_weights(self, module):
        std = self.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        """
        设置 KV Cache，为推理做准备。
        
        Args:
            max_batch_size: 最大批次大小
            max_seq_length: 最大序列长度（对齐到8的倍数以优化性能）
            dtype: 缓存数据类型
        """
        head_dim = self.dim // self.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(
                max_batch_size, max_seq_length, self.n_head, head_dim, dtype
            )

        # 创建因果注意力掩码（下三角矩阵）
        causal_mask = torch.tril(
            torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)
        )
        self.causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        
        # 重新计算 2D 位置编码
        grid_size = int(self.block_size**0.5)
        assert grid_size * grid_size == self.block_size
        self.freqs_cis = precompute_freqs_cis_2d(
            grid_size, self.dim // self.n_head, self.rope_base, self.cls_token_num
        )

    def forward(
        self,
        idx: torch.Tensor,
        cond_idx: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
    ):
        """
        前向传播，支持训练和推理两种模式。
        
        训练模式 (idx 和 cond_idx 都不为 None):
            - 拼接条件嵌入和 token 嵌入
            - 计算完整注意力（无 KV Cache）
            - 计算交叉熵损失
        
        推理模式:
            - 预填充阶段：只有 cond_idx，生成第一个 token
            - 解码阶段：只有 idx，利用 KV Cache 逐 token 生成
        
        Args:
            idx: 图像 token 索引（训练时用）或当前 token（推理时用）
            cond_idx: 条件 token 索引
            input_pos: 位置索引（推理时用）
            targets: 目标 token（训练时计算损失）
            mask: 注意力掩码
            valid: 有效 token 掩码（部分 token 不参与损失计算）
        
        Returns:
            (logits, loss, None): 预测 logits、损失（可能为 None）、占位符
        """
        if idx is not None and cond_idx is not None:  # 训练或朴素推理
            cond_embeddings = self.cls_embedding(cond_idx, train=self.training)[
                :, : self.cls_token_num
            ]
            idx = idx[:, :-1]  # 自回归：输入去掉最后一个 token
            token_embeddings = self.tok_embeddings(idx)
            token_embeddings = torch.cat((cond_embeddings, token_embeddings), dim=1)
            h = self.tok_dropout(token_embeddings)
            self.freqs_cis = self.freqs_cis.to(h.device)
        else:
            if cond_idx is not None:  # 推理的预填充阶段
                token_embeddings = self.cls_embedding(cond_idx, train=self.training)[
                    :, : self.cls_token_num
                ]
            else:  # 推理的解码阶段（利用 KV Cache）
                token_embeddings = self.tok_embeddings(idx)

            bs = token_embeddings.shape[0]
            mask = self.causal_mask[:bs, None, input_pos]
            h = self.tok_dropout(token_embeddings)
            self.freqs_cis = self.freqs_cis

        # 选择对应位置的频率编码
        if self.training:
            freqs_cis = self.freqs_cis[: token_embeddings.shape[1]]
        else:
            freqs_cis = self.freqs_cis[input_pos]
        
        # 逐层前向传播
        for layer in self.layers:
            h = layer(h, freqs_cis, input_pos, mask)

        # 输出层
        h = self.norm(h)
        logits = self.output(h).float()

        if self.training:
            # 训练时去掉条件 token 对应的 logits（条件 token 不需要预测）
            logits = logits[:, self.cls_token_num - 1 :].contiguous()

        # 损失计算
        loss = None
        if valid is not None:
            # 带有效掩码的损失：部分 token 不参与损失计算
            loss_all = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none"
            )
            valid_all = valid[:, None].repeat(1, targets.shape[1]).view(-1)
            loss = (loss_all * valid_all).sum() / max(valid_all.sum(), 1)
        elif targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, None

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        """返回 FSDP（Fully Sharded Data Parallel）包装的模块列表"""
        return list(self.layers)

    def configure_optimizer(
        self, lr, weight_decay, beta1, beta2, max_grad_norm, **kwargs
    ):
        """
        配置 AdamW 优化器，对不同维度的参数采用不同的权重衰减策略：
        - 2D 及以上参数（权重矩阵、嵌入矩阵）：应用权重衰减
        - 1D 参数（偏置、归一化参数）：不应用权重衰减
        """
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        
        # 优先使用融合版 AdamW（CUDA 上更快）
        import inspect
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        extra_args = dict(fused=True) if fused_available else dict()

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
            **extra_args
        )
        return optimizer

    def generate(
        self,
        cond,
        max_new_tokens,
        emb_masks=None,
        cfg_scale=1.0,
        cfg_interval=-1,
        **sampling_kwargs
    ):
        """
        LlamaGen 风格的图像生成（顺序自回归，非并行解码）。
        
        详见 generate.py 中的同名函数说明。
        """
        if self.model_type == "c2i":
            if cfg_scale > 1.0:
                cond_null = torch.ones_like(cond) * self.num_classes
                cond_combined = torch.cat([cond, cond_null])
            else:
                cond_combined = cond
            T = 1
        elif self.model_type == "t2i":
            if cfg_scale > 1.0:
                cond_null = torch.zeros_like(cond) + self.cls_embedding.uncond_embedding
                cond_combined = torch.cat([cond, cond_null])
            else:
                cond_combined = cond
            T = cond.shape[1]
        else:
            raise Exception("please check model type")

        T_new = T + max_new_tokens
        max_seq_length = T_new
        max_batch_size = cond.shape[0]

        device = cond.device
        with torch.device(device):
            max_batch_size_cfg = (
                max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
            )
            self.setup_caches(
                max_batch_size=max_batch_size_cfg,
                max_seq_length=max_seq_length,
                dtype=self.tok_embeddings.weight.dtype,
            )

        if emb_masks is not None:
            assert emb_masks.shape[0] == max_batch_size
            assert emb_masks.shape[-1] == T
            if cfg_scale > 1.0:
                self.causal_mask[:, :, :T] = self.causal_mask[:, :, :T] * torch.cat(
                    [emb_masks, emb_masks]
                ).unsqueeze(1)
            else:
                self.causal_mask[:, :, :T] = self.causal_mask[
                    :, :, :T
                ] * emb_masks.unsqueeze(1)

            eye_matrix = torch.eye(
                self.causal_mask.size(1), self.causal_mask.size(2), device=device
            )
            self.causal_mask[:] = self.causal_mask * (1 - eye_matrix) + eye_matrix

        seq = torch.empty((max_batch_size, T_new), dtype=torch.int, device=device)

        input_pos = torch.arange(0, T, device=device)
        next_token = prefill(
            self, cond_combined, input_pos, cfg_scale, **sampling_kwargs
        )
        seq[:, T : T + 1] = next_token

        input_pos = torch.tensor([T], device=device, dtype=torch.int)
        generated_tokens, _ = decode_n_tokens(
            self,
            next_token,
            input_pos,
            max_new_tokens - 1,
            cfg_scale,
            cfg_interval,
            **sampling_kwargs
        )
        seq[:, T + 1 :] = torch.cat(generated_tokens, dim=1)

        return seq[:, T:]


#################################################################################
#                      旋转位置编码函数 (Rotary Positional Embedding)            #
#################################################################################

def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000, cls_token_num=120
):
    """
    预计算 1D 旋转位置编码（RoPE）的频率复数表示。
    
    RoPE 通过在注意力计算中旋转 Query 和 Key 来编码位置信息，
    使模型能自然地感知相对位置关系。
    
    Args:
        seq_len: 序列长度
        n_elem: 每个头的维度（head_dim）
        base: 频率基数（越大，低频分辨率越高）
        cls_token_num: 条件 token 数量（位置编码前补零）
    
    Returns:
        freqs_cis: shape (cls_token_num + seq_len, head_dim // 2, 2)
        条件 token 位置编码为零（不影响注意力计算）
    """
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)  # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack(
        [freqs_cis.real, freqs_cis.imag], dim=-1
    )  # (seq_len, head_dim // 2, 2)
    # 条件 token 的位置编码为零（不影响注意力得分）
    cond_cache = torch.cat(
        [torch.zeros(cls_token_num, n_elem // 2, 2), cache]
    )  # (cls_token_num + seq_len, head_dim // 2, 2)
    return cond_cache


def precompute_freqs_cis_2d(
    grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120
):
    """
    预计算 2D 旋转位置编码（2D RoPE）的频率表示。
    
    对于图像生成，2D RoPE 比 1D RoPE 更合适，因为它能分别编码
    水平和垂直两个方向的位置信息。
    
    实现方式：
    - 将 head_dim 分成两半，一半编码 x 坐标，一半编码 y 坐标
    - 构建 grid_size x grid_size 的 2D 频率网格
    - 展平为 (grid_size^2, head_dim//2, 2) 的张量
    
    Args:
        grid_size: 网格边长（如 16，对应 16x16=256 个 token）
        n_elem: 每个头的维度（head_dim）
        base: 频率基数
        cls_token_num: 条件 token 数量
    
    Returns:
        freqs_cis: shape (cls_token_num + grid_size^2, head_dim // 2, 2)
    """
    # 将维度分为两半，分别用于 x 和 y 坐标
    half_dim = n_elem // 2
    freqs = 1.0 / (
        base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim)
    )
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs)  # (grid_size, head_dim // 4)
    
    # 构建 2D 频率网格：x 方向和 y 方向
    freqs_grid = torch.concat(
        [
            freqs[:, None, :].expand(-1, grid_size, -1),  # x 方向频率
            freqs[None, :, :].expand(grid_size, -1, -1),  # y 方向频率
        ],
        dim=-1,
    )  # (grid_size, grid_size, head_dim // 2)
    
    # 分离实部和虚部
    cache_grid = torch.stack(
        [torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1
    )  # (grid_size, grid_size, head_dim // 2, 2)
    
    # 展平空间维度
    cache = cache_grid.flatten(0, 1)  # (grid_size^2, head_dim // 2, 2)
    
    # 条件 token 位置编码补零
    cond_cache = torch.cat(
        [torch.zeros(cls_token_num, n_elem // 2, 2), cache]
    )  # (cls_token_num + grid_size^2, head_dim // 2, 2)
    return cond_cache


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    """
    应用旋转位置编码（RoPE）到输入张量。
    
    RoPE 的核心思想：将位置信息编码为旋转矩阵，通过旋转 Query 和 Key
    使内积自然包含相对位置信息。具体来说，位置 m 的向量乘以角度 m*θ 的旋转矩阵，
    两个位置的内积只取决于相对位置差 (m-n)。
    
    这里使用复数表示实现旋转：将向量视为复数，乘以频率复数即完成旋转。
    
    Args:
        x: 输入张量，shape (bs, seq_len, n_head, head_dim)
        freqs_cis: 频率编码，shape (seq_len, head_dim // 2, 2)
    
    Returns:
        旋转后的张量，shape 与输入相同
    """
    # 将最后一维拆成 (head_dim//2, 2)，对应复数的实部和虚部
    xshaped = x.float().reshape(
        *x.shape[:-1], -1, 2
    )  # (bs, seq_len, n_head, head_dim//2, 2)
    freqs_cis = freqs_cis.view(
        1, xshaped.size(1), 1, xshaped.size(3), 2
    )  # (1, seq_len, 1, head_dim//2, 2)
    # 复数乘法：(a + bi)(c + di) = (ac - bd) + (ad + bc)i
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        dim=-1,
    )
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)
