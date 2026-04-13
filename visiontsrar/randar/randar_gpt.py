"""
RandAR 核心 GPT 模型 —— 随机顺序自回归 Transformer

本模块实现了 RandAR（Randomized Autoregressive）的核心模型，这是 VisionTSRAR 项目的关键组件。
RandAR 的核心创新在于：不同于传统自回归模型从左到右顺序生成 token，
RandAR 可以在任意随机顺序下生成图像 token，通过"位置指令 token"告知模型
"接下来要生成哪个空间位置的 token"。

这种设计带来了两个重要优势：
1. 并行解码：可以在每步同时预测多个位置的 token，大幅减少推理步数
2. Inpainting 能力：给定部分已知 token，可以生成剩余未知 token

在 VisionTSRAR 中，inpainting 能力尤为重要：
- 时间序列被渲染为图像后，左半部分（历史数据）作为已知 token
- 右半部分（预测目标）作为需要生成的 token
- RandAR 利用已知上下文生成预测部分，实现时间序列预测

核心组件：
- RandARTransformer: 主模型类，支持训练和推理
- Attention: 支持 KV Cache 切片的注意力模块
- batch_apply_rotary_emb: 批量旋转位置编码应用

来源:
- LlamaGen: https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/models/gpt.py
- VQGAN: https://github.com/CompVis/taming-transformers
- DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
- nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
- LLaMA: https://github.com/facebookresearch/llama/blob/main/llama/model.py
- gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
- PixArt: https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
"""

import random
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from dataclasses import dataclass
from typing import Optional, List, Tuple

from .utils import DropPath, interleave_tokens, calculate_num_query_tokens_for_parallel_decoding
from .generate import sample
from .llamagen_gpt import LabelEmbedder, CaptionEmbedder, MLP, RMSNorm, \
    FeedForward, KVCache, find_multiple, apply_rotary_emb, precompute_freqs_cis_2d


def batch_apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    """
    批量应用旋转位置编码（支持不同样本使用不同的位置编码）。
    
    与 llamagen_gpt.apply_rotary_emb 的区别：
    - apply_rotary_emb: freqs_cis shape 为 (seq_len, ...)，所有样本共享同一组位置编码
    - batch_apply_rotary_emb: freqs_cis shape 为 (bs, seq_len, ...)，每个样本可以有不同位置编码
    
    在 RandAR 中必须使用 batch 版本，因为每个样本的 token 顺序是随机排列的，
    对应的位置编码也各不相同。而在传统自回归模型中，所有样本的 token 顺序相同，
    位置编码可以共享。
    
    Args:
        x: 输入张量，shape (bs, seq_len, n_head, head_dim)
        freqs_cis: 批量频率编码，shape (bs, seq_len, head_dim // 2, 2)
    
    Returns:
        旋转后的张量，shape 与输入相同
    """
    bs, seq_len, n_head, head_dim = x.shape
    xshaped = x.float().reshape(
        *x.shape[:-1], head_dim // 2, 2
    )  # (bs, seq_len, n_head, head_dim//2, 2)
    freqs_cis = freqs_cis.view(
        bs, xshaped.size(1), 1, xshaped.size(3), 2
    )  # (bs, seq_len, 1, head_dim//2, 2)
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


class Attention(nn.Module):
    """
    RandAR 自定义注意力模块，支持 KV Cache 切片加速推理。
    
    与 LlamaGen 的 Attention 相比，主要修改在于 KV Cache 读取时：
    - 使用 input_pos 的最大值截取有效的 KV 部分，而非返回整个缓存
    - 这是因为 RandAR 的序列长度在推理过程中动态变化（并行解码时步长不同）
    
    支持分组查询注意力（GQA）和 KV Cache。
    """
    def __init__(
        self,
        dim: int,
        n_head: int,
        n_kv_head: int,
        attn_dropout_p: float,
        resid_dropout_p: float,
    ):
        super().__init__()
        assert dim % n_head == 0
        self.dim = dim
        self.head_dim = dim // n_head
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
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
            x: 输入张量，shape (bsz, seqlen, dim)
            freqs_cis: 旋转位置编码（RandAR 中使用 batch 版本）
            input_pos: 位置索引（推理时用于 KV Cache 更新）
            mask: 因果注意力掩码
        
        Returns:
            注意力输出，shape (bsz, seqlen, dim)
        """
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)

        # RandAR 修改：使用 batch 版本的旋转位置编码
        xq = batch_apply_rotary_emb(xq, freqs_cis)
        xk = batch_apply_rotary_emb(xk, freqs_cis)

        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.kv_cache is not None and input_pos is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)

            max_pos = torch.max(input_pos) + 1
            keys = keys[:, :, :max_pos]
            values = values[:, :, :max_pos]
            if mask is not None:
                mask = mask[:, :, :, :max_pos]
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
            ),
            dropout_p=self.attn_dropout_p if self.training else 0,
        )

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        output = self.resid_dropout(self.wo(output))
        return output


class TransformerBlock(nn.Module):
    """
    Transformer 残差块，与 LlamaGen 相同，但使用 RandAR 自定义的 Attention。
    
    结构: x + DropPath(Attention(RMSNorm(x))) → h + DropPath(FFN(RMSNorm(h)))
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


class RandARTransformer(nn.Module):
    """
    RandAR 核心模型：随机顺序自回归 Transformer。
    
    与传统自回归模型的关键区别：
    1. 位置指令 token（Position Instruction Token）：
       在每个图像 token 前插入一个"位置指令"token，告知模型接下来要预测
       哪个空间位置。这使得模型可以在任意顺序下生成 token。
    
    2. 并行解码（Parallel Decoding）：
       利用余弦调度，在推理时每步同时预测多个 token，大幅减少推理步数。
       例如 256 个 token 的图像可以在 88 步内生成完毕。
    
    3. Inpainting 支持（VisionTSRAR 新增）：
       可以给定部分已知 token，只预测剩余的未知 token。
       这对时间序列预测至关重要：历史数据作为已知 token，
       预测目标作为需要生成的 token。
    
    序列结构（训练时）：
    [cond_token, pos_instr_0, img_tok_0, pos_instr_1, img_tok_1, ...]
    
    其中 pos_instr_i 是位置指令 token（携带旋转位置编码信息），
    img_tok_i 是图像 token（来自 VQ-VAE 的离散编码）。
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
        position_order="random",
        num_inference_steps=88,
        zero_class_qk=True,
        grad_checkpointing=True,
    ):
        """
        Args:
            dim: 隐藏维度
            n_layer: Transformer 层数
            n_head: 注意力头数
            n_kv_head: KV 头数（GQA）
            multiple_of: FFN 维度对齐基数
            ffn_dim_multiplier: FFN 维度乘数
            rope_base: RoPE 基数
            norm_eps: 归一化 epsilon
            initializer_range: 参数初始化标准差
            token_dropout_p: Token dropout 概率
            attn_dropout_p: 注意力 dropout 概率
            resid_dropout_p: 残差 dropout 概率
            ffn_dropout_p: FFN dropout 概率
            drop_path_rate: Drop Path 概率
            num_classes: 类别数
            caption_dim: 文本特征维度
            class_dropout_prob: CFG 标签丢弃概率
            model_type: 模型类型（c2i 或 t2i）
            vocab_size: VQ 码本大小
            cls_token_num: 条件 token 数量
            block_size: 图像 token 序列长度
            max_batch_size: 最大批次大小
            max_seq_len: 最大序列长度
            position_order: 位置顺序（random 或 raster）
            num_inference_steps: 推理步数（-1 表示逐 token 生成）
            zero_class_qk: 是否将条件 token 的 QK 置零
            grad_checkpointing: 是否使用梯度检查点节省显存
        """
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

        # Transformer 层（线性递增的 DropPath）
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

        # ===== RandAR 特有参数 =====
        # 位置指令嵌入：一个可学习向量，与旋转位置编码结合形成位置指令 token
        # 位置指令 token 的作用：告知模型"接下来要生成哪个空间位置的 token"
        self.pos_instruct_embeddings = nn.Parameter(torch.randn(1, self.dim) * self.initializer_range)
        self.position_order = position_order
        self.num_inference_steps = num_inference_steps
        self.zero_class_qk = zero_class_qk
        self.grad_checkpointing = grad_checkpointing

    def initialize_weights(self):
        """初始化所有权重：Linear 和 Embedding 使用正态分布，输出层初始化为0"""
        self.apply(self._init_weights)
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
            max_seq_length: 最大序列长度（对齐到8的倍数）
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

        causal_mask = torch.tril(
            torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)
        )
        self.causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        grid_size = int(self.block_size**0.5)
        assert grid_size * grid_size == self.block_size
        self.freqs_cis = precompute_freqs_cis_2d(
            grid_size, self.dim // self.n_head, self.rope_base, self.cls_token_num
        )
    
    def remove_caches(self):
        """清除 KV Cache，释放显存"""
        for l in self.layers:
            l.attention.kv_cache = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def forward(
        self,
        idx: torch.Tensor,
        cond_idx: torch.Tensor,
        token_order: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
        visible_tokens: Optional[torch.Tensor] = None,
    ):
        """
        前向传播入口，根据输入自动选择训练或推理模式。
        
        Args:
            idx: 图像 token 索引
            cond_idx: 条件 token 索引
            token_order: 位置顺序
            input_pos: 位置索引
            targets: 目标 token
            mask: 注意力掩码
            valid: 有效 token 掩码
            visible_tokens: 【VisionTSRAR 新增】已知 token 索引，用于 inpainting 模式
        """
        if idx is not None and cond_idx is not None:
            return self.forward_train(idx, cond_idx, token_order, input_pos, targets, mask, valid, visible_tokens)
        else:
            raise ValueError("idx and cond_idx cannot be both None")
        
    def forward_train(self,
                      idx: torch.Tensor,
                      cond_idx: torch.Tensor,
                      token_order: Optional[torch.Tensor] = None,
                      input_pos: Optional[torch.Tensor] = None,
                      targets: Optional[torch.Tensor] = None,
                      mask: Optional[torch.Tensor] = None,
                      valid: Optional[torch.Tensor] = None,
                      visible_tokens: Optional[torch.Tensor] = None):
        """
        训练时的前向传播。
        
        流程：
        1. 准备 token 顺序（随机或光栅扫描顺序）
        2. 按 token 顺序重排图像 token 和目标 token
        3. 构建输入序列：[cond, pos_instr_0, img_tok_0, pos_instr_1, img_tok_1, ...]
        4. 逐层 Transformer 前向传播
        5. 计算损失（仅对图像 token 位置计算）
        
        【VisionTSRAR 新增】Inpainting 模式：
        当 visible_tokens 不为 None 时，表示部分 token 已知（如时间序列的历史数据）。
        - 已知 token 的嵌入被添加到序列开头（cond_embeddings 之后）
        - 已知 token 不需要预测，对应位置的 valid mask 设为 0
        - 模型只需预测剩余的未知 token
        
        Args:
            idx: [bsz, seq_len] GT 图像 token 索引（teacher forcing 输入）
            cond_idx: [bsz, cls_token_num] 条件 token 索引
            token_order: [bsz, seq_len] 每个 token 的位置顺序
            input_pos: [seq_len] 位置索引
            targets: [bsz, seq_len] 目标 token（teacher forcing 目标）
            mask: [bsz, seq_len, seq_len] 因果注意力掩码
            valid: [bsz, seq_len] 有效 token 掩码
            visible_tokens: [bsz, num_visible] 已知 token 索引（inpainting 模式）
        
        Returns:
            token_logits: [bsz, seq_len, vocab_size] 预测 logits
            loss: 标量损失值（可能为 None）
            token_order: [bsz, seq_len] 使用的位置顺序
        """
        # ===== 1. 准备 token 顺序 =====
        bs = idx.shape[0]
        if token_order is None:
            if self.position_order == "random":
                # 随机排列：每个样本独立随机打乱 token 顺序
                token_order = torch.arange(self.block_size, device=self.tok_embeddings.weight.device, dtype=torch.long)
                token_order = token_order.unsqueeze(0).repeat(bs, 1)
                for i in range(bs):
                    token_order[i] = token_order[i][torch.randperm(self.block_size)]
                token_order = token_order.contiguous()
            elif self.position_order == "raster":
                # 光栅扫描顺序：从左到右、从上到下
                token_order = torch.arange(self.block_size, device=idx.device)
                token_order = token_order.unsqueeze(0).repeat(bs, 1)
                token_order = token_order.contiguous()
            else:
                raise ValueError(f"Invalid position order: {self.position_order}")
        
        # 按 token 顺序重排图像 token 和目标
        idx = torch.gather(idx.unsqueeze(-1), 1, token_order.unsqueeze(-1)).squeeze(-1).contiguous()
        targets = torch.gather(targets.unsqueeze(-1), 1, token_order.unsqueeze(-1)).squeeze(-1).contiguous()

        # ===== 2. 准备嵌入和位置编码 =====
        self.freqs_cis = self.freqs_cis.to(cond_idx.device)
        cond_embeddings = self.cls_embedding(cond_idx, train=self.training)[
            :, : self.cls_token_num
        ] # [bsz, cls_token_num, dim]

        token_embeddings = self.tok_embeddings(idx)
        token_embeddings = self.tok_dropout(token_embeddings) # [bsz, seq_len, dim]
        position_instruction_tokens = self.get_position_instruction_tokens(token_order) # [bsz, seq_len, dim]

        # ===== 【VisionTSRAR 新增】Inpainting 模式：处理已知 token =====
        if visible_tokens is not None:
            num_visible = visible_tokens.shape[1]
            # 将已知 token 嵌入到序列开头（紧跟 cond_embeddings 之后）
            visible_embeddings = self.tok_embeddings(visible_tokens)  # [bsz, num_visible, dim]
            visible_embeddings = self.tok_dropout(visible_embeddings)
            # 已知 token 也需要位置指令
            # 已知 token 在 token_order 中的位置是前 num_visible 个
            visible_token_order = token_order[:, :num_visible]
            visible_position_instructions = self.get_position_instruction_tokens(visible_token_order)  # [bsz, num_visible, dim]
            
            # 构建完整序列：[cond, visible_pos_instr_0, visible_tok_0, ..., 
            #                query_pos_instr_0, query_tok_0, ...]
            query_token_embeddings = token_embeddings[:, num_visible:]  # 剩余的 query token
            query_position_instructions = position_instruction_tokens[:, num_visible:]
            
            h = torch.cat(
                (cond_embeddings,
                 interleave_tokens(visible_position_instructions, visible_embeddings),
                 interleave_tokens(query_position_instructions, query_token_embeddings)),
                dim=1
            )
            
            # 位置编码：已知 token 和 query token 的位置编码
            visible_freqs_cis = self.freqs_cis[self.cls_token_num:].clone().to(token_order.device)[visible_token_order]
            query_token_order = token_order[:, num_visible:]
            query_freqs_cis = self.freqs_cis[self.cls_token_num:].clone().to(token_order.device)[query_token_order]
            
            freqs_cis = torch.cat(
                (self.freqs_cis[:self.cls_token_num].unsqueeze(0).repeat(bs, 1, 1, 1),
                 interleave_tokens(visible_freqs_cis, visible_freqs_cis),
                 interleave_tokens(query_freqs_cis, query_freqs_cis)),
                dim=1
            )
            
            # 修改 valid mask：已知 token 位置不参与损失计算
            if valid is not None:
                # valid shape: [bsz, block_size]
                # 前 num_visible 个 token 是已知的，不需要预测
                valid_with_visible = valid.clone()
                valid_with_visible[:, :num_visible] = 0
                # 重新排列 valid 以匹配 token_order
                valid_reordered = torch.gather(valid.unsqueeze(-1), 1, token_order.unsqueeze(-1)).squeeze(-1).contiguous()
                valid_reordered[:, :num_visible] = 0  # 已知 token 不参与 loss
                valid = valid_reordered
        else:
            # 标准模式：[cond, pos_instr_0, img_tok_0, pos_instr_1, img_tok_1, ...]
            h = torch.cat(
                (cond_embeddings, interleave_tokens(position_instruction_tokens, token_embeddings)),
                dim=1
            )
            
            token_freqs_cis = self.freqs_cis[self.cls_token_num:].clone().to(token_order.device)[token_order]
            freqs_cis = torch.cat(
                (self.freqs_cis[:self.cls_token_num].unsqueeze(0).repeat(bs, 1, 1, 1), interleave_tokens(token_freqs_cis, token_freqs_cis)),
                dim=1
            )

        # ===== 3. Transformer 前向传播 =====
        for layer in self.layers:
            if self.grad_checkpointing:
                h = checkpoint(layer, h, freqs_cis, input_pos, mask, use_reentrant=False)
            else:
                h = layer(h, freqs_cis, input_pos, mask)
        
        h = self.norm(h)
        logits = self.output(h).float()
        
        # 仅取图像 token 位置的 logits（跳过条件 token 和位置指令 token）
        token_logits = logits[:, self.cls_token_num::2].contiguous()

        # ===== 4. 损失计算 =====
        loss = None
        if visible_tokens is not None:
            # Inpainting 模式的损失计算
            num_visible = visible_tokens.shape[1]
            if valid is not None:
                loss_all = F.cross_entropy(
                    token_logits.view(-1, token_logits.size(-1)), targets.view(-1), reduction="none"
                )
                valid_all = valid[:, None].repeat(1, targets.shape[1]).view(-1)
                loss = (loss_all * valid_all).sum() / max(valid_all.sum(), 1)
            elif targets is not None:
                # 只对 query token（非已知 token）计算损失
                query_targets = targets[:, num_visible:]  # query 部分的目标
                query_logits = token_logits[:, num_visible:]  # query 部分的 logits
                loss = F.cross_entropy(query_logits.reshape(-1, query_logits.size(-1)), query_targets.reshape(-1))
        else:
            # 标准模式损失
            if valid is not None:
                loss_all = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none"
                )
                valid_all = valid[:, None].repeat(1, targets.shape[1]).view(-1)
                loss = (loss_all * valid_all).sum() / max(valid_all.sum(), 1)
            elif targets is not None:
                loss = F.cross_entropy(token_logits.view(-1, token_logits.size(-1)), targets.view(-1))

        return token_logits, loss, token_order
    
    def forward_inference(self, 
                          x: torch.Tensor, 
                          freqs_cis: torch.Tensor, 
                          input_pos: torch.Tensor):
        """
        推理时的前向传播（利用 KV Cache）。
        
        与训练模式不同，推理时：
        - 使用 KV Cache 避免重复计算
        - 使用因果掩码确保自回归特性
        - 不计算损失
        
        Args:
            x: [bs, query_num, dim] 输入 token 嵌入
            freqs_cis: [bs, query_num, n_head, dim // n_head] 频率编码
            input_pos: [query_num] 位置索引
        
        Returns:
            logits: [bs, query_num, vocab_size] 预测 logits
        """
        bs = x.shape[0]
        mask = self.causal_mask[:bs, None, input_pos]
        h = x
        for layer in self.layers:
            h = layer(h, freqs_cis, start_pos=input_pos, mask=mask)
        h = self.norm(h)
        logits = self.output(h).float()
        return logits

    def get_position_instruction_tokens(self, token_order):
        """
        生成位置指令 token：将可学习的位置指令嵌入与旋转位置编码结合。
        
        位置指令 token 的构成：
        1. 基础嵌入：一个可学习向量 pos_instruct_embeddings，所有位置共享
        2. 位置编码：根据 token_order 索引对应的 2D RoPE 频率编码
        3. 两者结合：将基础嵌入 reshape 为多头形式，应用旋转位置编码
        
        这样每个位置指令 token 既包含"我是位置指令"的信息（通过共享的基础嵌入），
        又包含"我要预测哪个位置"的信息（通过位置编码）。
        
        Args:
            token_order: [bsz, num_tokens] 每个 token 的位置顺序
        
        Returns:
            position_instruct_tokens: [bsz, num_tokens, dim] 位置指令 token
        """
        position_instruct_tokens = self.pos_instruct_embeddings.view(1, 1, self.n_head, self.dim // self.n_head)
        position_instruct_tokens = position_instruct_tokens.repeat(token_order.shape[0], token_order.shape[1], 1, 1)
        
        # 获取对应位置的频率编码并应用旋转位置编码
        position_instruct_freqs_cis = self.freqs_cis[self.cls_token_num:].clone().to(token_order.device)[token_order]
        position_instruct_tokens = batch_apply_rotary_emb(position_instruct_tokens, position_instruct_freqs_cis)
        position_instruct_tokens = position_instruct_tokens.view(token_order.shape[0], token_order.shape[1], self.dim).contiguous()
        return position_instruct_tokens

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        """返回 FSDP 包装的模块列表"""
        return list(self.layers)

    def configure_optimizer(
        self, lr, weight_decay, beta1, beta2, max_grad_norm, **kwargs
    ):
        """
        配置 AdamW 优化器（2D+ 参数权重衰减，1D 参数不衰减）。
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
        cond: torch.Tensor,
        token_order: torch.Tensor,
        cfg_scales: Tuple[float, float] = (1.0, 1.0),
        num_inference_steps: int = 88,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        visible_tokens: Optional[torch.Tensor] = None,
    ):
        """
        RandAR 并行解码生成函数。
        
        与传统自回归模型逐 token 顺序生成不同，RandAR 使用并行解码策略：
        每步同时预测多个位置的 token，通过余弦调度控制每步的解码数量。
        例如 256 个 token 的图像可以在 88 步内生成完毕（而非 256 步）。
        
        生成流程：
        1. 准备 token 顺序和位置指令 token
        2. 处理 CFG（分类器自由引导）
        3. 设置 KV Cache
        4. 预填充：输入条件 token 和第一个位置指令 token，生成第一个图像 token
        5. 循环解码：
           a. 将上一步生成的 token 和当前位置指令组成输入
           b. 利用 KV Cache 前向传播
           c. 采样生成新 token
           d. 计算下一步需要解码的 token 数量
        6. 将结果按逆排列恢复到原始光栅顺序
        
        【VisionTSRAR 新增】Inpainting 模式：
        当 visible_tokens 不为 None 时，表示部分 token 已知。
        - visible_tokens 作为已知的 prefix tokens 紧跟在 cond 后面
        - 只生成剩余 block_size - num_visible 个 token
        - 生成的 result_indices 前 num_visible 个位置直接用 visible_tokens 填充
        - 这支持了"给定左半部分图像，生成右半部分"的时序预测场景
        
        Args:
            cond: [bsz, cls_token_num] 条件 token 索引
            token_order: [bsz, block_size] 每个 token 的位置顺序
            cfg_scales: (起始CFG缩放, 结束CFG缩放)，线性插值
            num_inference_steps: 推理步数（-1 表示逐 token 生成）
            temperature: 采样温度
            top_k: Top-k 采样的 k 值
            top_p: Top-p 采样的 p 值
            visible_tokens: [bsz, num_visible] 已知 token 索引（inpainting 模式）
        
        Returns:
            result_indices: [bsz, block_size] 生成的 token 索引（光栅顺序）
        """
        bs = cond.shape[0]
        num_visible = 0  # 已知 token 数量
        
        # ===== Step-1: 生成 token 顺序和结果序列 =====
        if token_order is None:
            token_order = torch.arange(self.block_size, device=cond.device)
            token_order = token_order.unsqueeze(0).repeat(bs, 1)
            token_order = token_order.contiguous()
            if self.position_order == "random":
                for i in range(bs):
                    token_order[i] = token_order[i][torch.randperm(self.block_size)]
            token_order = token_order.contiguous()
        else:
            assert token_order.shape == (bs, self.block_size)
        
        result_indices = torch.zeros((bs, self.block_size), dtype=torch.long, device=cond.device)
        
        # ===== 【VisionTSRAR 新增】处理 inpainting 模式 =====
        if visible_tokens is not None:
            num_visible = visible_tokens.shape[1]
            # 将已知 token 直接填入结果序列的前 num_visible 个位置
            result_indices[:, :num_visible] = visible_tokens
        
        # ===== Step-2: 准备位置指令 token 和频率编码 =====
        position_instruction_tokens = self.get_position_instruction_tokens(token_order)
        img_token_freq_cis = self.freqs_cis[self.cls_token_num:].clone().to(token_order.device)[token_order]

        # ===== Step-3: 准备 CFG =====
        if cfg_scales[-1] > 1.0:
            cond_null = torch.ones_like(cond) * self.num_classes
            cond_combined = torch.cat([cond, cond_null])
            img_token_freq_cis = torch.cat([img_token_freq_cis, img_token_freq_cis])
            position_instruction_tokens = torch.cat([position_instruction_tokens, position_instruction_tokens])
            bs *= 2
        else:
            cond_combined = cond
        cond_combined_tokens = self.cls_embedding(cond_combined, train=False)
    
        # ===== Step-4: KV Cache 设置 =====
        max_seq_len = cond_combined_tokens.shape[1] + self.block_size * 2
        with torch.device(cond.device):
            self.setup_caches(max_batch_size=bs, max_seq_length=max_seq_len, dtype=self.tok_embeddings.weight.dtype)

        # ===== Step-5: 自回归生成（并行解码） =====
        if num_inference_steps == -1:
            num_inference_steps = self.block_size
        
        cur_inference_step = 0
        num_query_token_cur_step = 1
        # 【VisionTSRAR 修改】inpainting 模式下，从 num_visible 位置开始生成
        query_token_idx_cur_step = num_visible

        # ===== Step 5-1: 准备第一步的输入 =====
        # 构建输入序列: [cond_token, (visible_img_tokens + visible_pos_instructions)..., query_pos_instr_0]
        if num_visible > 0:
            # Inpainting 模式：将已知 token 嵌入和位置指令也加入输入
            visible_img_embeddings = self.tok_embeddings(visible_tokens)
            if cfg_scales[-1] > 1.0:
                visible_img_embeddings = torch.cat([visible_img_embeddings, visible_img_embeddings], dim=0)
            
            # 已知 token 的位置指令
            visible_pos_instructions = position_instruction_tokens[:, :num_visible]
            # 已知 token 的频率编码
            visible_freqs_cis = img_token_freq_cis[:, :num_visible]
            
            # 构建预填充输入：[cond, vis_pos_instr_0, vis_img_0, ..., vis_pos_instr_n, vis_img_n, query_pos_instr_0]
            x = torch.cat([
                cond_combined_tokens,
                interleave_tokens(visible_pos_instructions, visible_img_embeddings),
                position_instruction_tokens[:, query_token_idx_cur_step : query_token_idx_cur_step + num_query_token_cur_step]
            ], dim=1)
            
            # 频率编码也需要包含已知 token 的部分
            cur_freqs_cis = torch.cat([
                self.freqs_cis[:self.cls_token_num].unsqueeze(0).repeat(bs, 1, 1, 1),
                interleave_tokens(visible_freqs_cis, visible_freqs_cis),
                img_token_freq_cis[:, query_token_idx_cur_step : query_token_idx_cur_step + num_query_token_cur_step]
            ], dim=1)
        else:
            # 标准模式：[cond_token, query_pos_instr_0]
            x = torch.cat([cond_combined_tokens, 
                           position_instruction_tokens[:, query_token_idx_cur_step : query_token_idx_cur_step + num_query_token_cur_step]], 
                           dim=1)
            cur_freqs_cis = torch.cat([self.freqs_cis[:self.cls_token_num].unsqueeze(0).repeat(bs, 1, 1, 1), 
                                       img_token_freq_cis[:, query_token_idx_cur_step : query_token_idx_cur_step + num_query_token_cur_step]], 
                                       dim=1)
        
        input_pos = torch.arange(0, x.shape[1], device=cond.device)

        # ===== Step 5-2: 开始生成循环 =====
        while query_token_idx_cur_step <= self.block_size - num_query_token_cur_step and query_token_idx_cur_step <= self.block_size - 1:
            # Step 5-3: 解码当前步的 token
            logits = self.forward_inference(x, cur_freqs_cis, input_pos)

            # 应用 CFG
            if cfg_scales[-1] > 1.0:
                cur_cfg_scale = cfg_scales[0] + (cfg_scales[-1] - cfg_scales[0]) * query_token_idx_cur_step / self.block_size
                cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                logits = uncond_logits + cur_cfg_scale * (cond_logits - uncond_logits)

            # 获取当前步 query token 的 logits 和采样结果
            logits = logits[:, -num_query_token_cur_step:]
            indices = torch.zeros(result_indices.shape[0], num_query_token_cur_step, dtype=torch.long, device=cond.device)
            for i in range(num_query_token_cur_step):
                indices[:, i : i + 1] = sample(logits[:, i : i + 1], temperature=temperature, top_k=top_k, top_p=top_p)[0]
            
            # 保存生成的 token
            result_indices[:, query_token_idx_cur_step : query_token_idx_cur_step + num_query_token_cur_step] = indices.clone()
            
            img_tokens = self.tok_embeddings(indices)
            if cfg_scales[-1] > 1.0:
                img_tokens = torch.cat([img_tokens, img_tokens], dim=0)

            # Step 5-4: 准备下一步的输入
            cur_inference_step += 1
            num_query_token_next_step = calculate_num_query_tokens_for_parallel_decoding(
                cur_inference_step, num_inference_steps, self.block_size, 
                query_token_idx_cur_step, num_query_token_cur_step)
            
            ########## 准备 token 序列 ##########
            # 结构: [cur_img_0, cur_query_1, cur_img_1, ..., cur_query_n, cur_img_n, next_query_0, ..., next_query_m]
            x = torch.zeros(bs, 2 * num_query_token_cur_step - 1 + num_query_token_next_step, self.dim, dtype=x.dtype, device=cond.device)
            
            # cur_img_0: 上一步第一个生成的 token
            x[:, :1] = img_tokens[:, :1] 
            
            # [cur_query_1, ..., cur_query_n]: 当前步剩余 query 的位置指令
            cur_query_position_instruction_tokens = position_instruction_tokens[:, query_token_idx_cur_step + 1 : query_token_idx_cur_step + num_query_token_cur_step]
            x[:, 1 : 2 * num_query_token_cur_step - 1][:, ::2] = cur_query_position_instruction_tokens
            
            # [cur_img_1, ..., cur_img_n]: 当前步剩余生成的 token
            x[:, 1 : 2 * num_query_token_cur_step - 1][:, 1::2] = img_tokens[:, 1 : num_query_token_cur_step]
            
            # [next_query_0, ..., next_query_m]: 下一步 query 的位置指令
            query_token_idx_next_step = query_token_idx_cur_step + num_query_token_cur_step
            next_position_instruction_tokens = position_instruction_tokens[:, query_token_idx_next_step : query_token_idx_next_step + num_query_token_next_step]
            x[:, 2 * num_query_token_cur_step - 1 :] = next_position_instruction_tokens

            ########## 准备频率编码 ##########
            cur_freqs_cis = torch.zeros((bs, 2 * num_query_token_cur_step - 1 + num_query_token_next_step, *self.freqs_cis.shape[-2:]), 
                                         dtype=cur_freqs_cis.dtype, device=cond.device)
            
            # cur_img_0 的频率编码
            cur_freqs_cis[:, :1] = img_token_freq_cis[:, query_token_idx_cur_step : query_token_idx_cur_step + 1]

            # 当前步剩余 query 的频率编码
            cur_query_freq_cis = img_token_freq_cis[:, query_token_idx_cur_step + 1 : query_token_idx_cur_step + num_query_token_cur_step]
            cur_freqs_cis[:, 1 : 2 * num_query_token_cur_step - 1][:, ::2] = cur_query_freq_cis

            # 当前步剩余生成 token 的频率编码（与 query 相同位置）
            cur_freqs_cis[:, 1 : 2 * num_query_token_cur_step - 1][:, 1::2] = cur_query_freq_cis

            # 下一步 query 的频率编码
            next_freq_cis = img_token_freq_cis[:, query_token_idx_next_step : query_token_idx_next_step + num_query_token_next_step]
            cur_freqs_cis[:, 2 * num_query_token_cur_step - 1 :] = next_freq_cis

            # Step 5-5: 更新指针
            query_token_idx_cur_step = query_token_idx_next_step
            if query_token_idx_cur_step > self.block_size:
                break
            
            last_input_pos = input_pos[input_pos.shape[0] - num_query_token_cur_step]
            input_pos = torch.arange(2 * num_query_token_cur_step - 1 + num_query_token_next_step, device=cond.device, dtype=torch.long) + last_input_pos + 1
            num_query_token_cur_step = num_query_token_next_step
        
        # ===== Step 6: 将结果按逆排列恢复到光栅顺序 =====
        reverse_permutation = torch.argsort(token_order, dim=-1).long().unsqueeze(-1).expand(-1, -1, 1)
        result_indices = torch.gather(result_indices.unsqueeze(-1), 1, reverse_permutation).squeeze(-1)
        return result_indices
