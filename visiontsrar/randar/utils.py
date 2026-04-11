"""
RandAR 工具函数模块

本模块包含 RandAR 模型中使用的各种工具函数和辅助类：
- DropPath: 随机深度（Stochastic Depth）正则化，用于残差块的主路径
- interleave_tokens: 将两个序列交错合并，用于 RandAR 的"位置指令+图像token"交替排列
- calculate_num_query_tokens_for_parallel_decoding: 计算并行解码时每步需要生成的 token 数量

来源: https://github.com/FoundationVision/LlamaGen/blob/main/utils/drop_path.py
"""

import torch
import math


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """
    随机深度（Drop Path / Stochastic Depth）：以一定概率丢弃整个样本的残差路径。
    
    与 Dropout 丢弃单个神经元不同，Drop Path 丢弃的是整个残差分支，
    这在深层 Transformer 中能有效缓解过拟合，同时保持训练和推理的一致性。
    
    实现原理：
    - 生成一个二值掩码（Bernoulli采样），保留概率为 keep_prob = 1 - drop_prob
    - 若 scale_by_keep=True，则除以 keep_prob 以保持期望值不变（类似 inverted dropout）
    
    Args:
        x: 输入张量，任意形状
        drop_prob: 丢弃概率，0表示不丢弃
        training: 是否在训练模式（仅训练时才执行丢弃）
        scale_by_keep: 是否按保留概率缩放，保持期望不变
    
    Returns:
        丢弃路径后的张量，形状与输入相同
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # shape = (batch_size, 1, 1, ...) —— 对每个样本独立决定是否保留
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(torch.nn.Module):
    """
    Drop Path 模块封装，将 drop_path 函数封装为 nn.Module 便于在模型中使用。
    
    在 Transformer 残差块中用于随机深度正则化：
        h = x + DropPath(Attention(LayerNorm(x)))
        out = h + DropPath(FFN(LayerNorm(h)))
    
    当 drop_prob=0 时退化为恒等映射（nn.Identity），不引入任何额外开销。
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


def interleave_tokens(seq1, seq2):
    """
    将两个序列交错合并。
    
    例如：seq1 = [A, B, C], seq2 = [D, E, F]
    结果 = [A, D, B, E, C, F]
    
    在 RandAR 中，seq1 是位置指令 token（position instruction tokens），
    seq2 是图像 token（image tokens）。交错排列的目的是让每个图像 token
    紧跟在其对应的位置指令 token 之后，模型在预测时能先看到"要预测哪个位置"，
    再生成该位置的 token。
    
    这种设计是 RandAR 的核心创新之一：通过位置指令 token 告知模型
    "接下来要生成哪个空间位置的 token"，从而实现随机顺序的自回归生成。
    
    Args:
        seq1: 第一个序列，shape [bsz, L, ...]
        seq2: 第二个序列，shape [bsz, L, ...]（必须与 seq1 长度相同）
    
    Returns:
        交错合并后的序列，shape [bsz, 2*L, ...]
    """
    result = torch.zeros_like(torch.cat((seq1, seq2), dim=1))
    result[:, ::2] = seq1   # 偶数位放 seq1（位置指令）
    result[:, 1::2] = seq2  # 奇数位放 seq2（图像token）
    return result


def calculate_num_query_tokens_for_parallel_decoding(cur_step, total_step, block_size, 
                                                     query_token_idx_cur_step, num_query_token_cur_step):
    """
    计算并行解码时，下一步需要解码的 token 数量。
    
    RandAR 的并行解码策略采用余弦调度（cosine schedule）：
    - 开始时每次只解码少量 token（确定性高）
    - 逐步增加每步解码的 token 数量（利用已生成 token 的上下文）
    - 这种策略在减少推理步数的同时保持生成质量
    
    调度公式：num_target = (1 - cos(π/2 * (step+1)/total)) * block_size + 1
    这意味着已解码的 token 比例随步数呈余弦曲线增长。
    
    Args:
        cur_step: 当前推理步数（从0开始计数）
        total_step: 总推理步数（num_inference_steps）
        block_size: 总 token 数量（如图像为 16x16=256）
        query_token_idx_cur_step: 当前步第一个 token 的索引位置
        num_query_token_cur_step: 当前步解码的 token 数量
    
    Returns:
        num_query_tokens_next_step: 下一步需要解码的 token 数量
    """
    # 按余弦调度计算到目前为止应该已经解码的 token 总数
    num_target_decoded_tokens = (
        1.0 - math.cos(math.pi / 2.0 * (cur_step + 1) / total_step)
    ) * block_size + 1
    num_target_decoded_tokens = min(
        int(num_target_decoded_tokens), block_size
    )

    # 下一步需要额外解码的 token 数 = 目标总数 - 已处理的 - 当前步的
    num_query_tokens_next_step = (
        num_target_decoded_tokens - query_token_idx_cur_step - num_query_token_cur_step
    )
    # 至少解码1个，最多不超过剩余未解码的 token 数
    num_query_tokens_next_step = max(num_query_tokens_next_step, 1)
    num_query_tokens_next_step = min(
        num_query_tokens_next_step,
        block_size - query_token_idx_cur_step - num_query_token_cur_step,
    )

    return num_query_tokens_next_step
