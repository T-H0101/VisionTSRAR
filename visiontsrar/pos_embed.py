# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# 位置编码工具：2D正弦-余弦位置编码
#
# 在 VisionTSRAR 中，此模块主要用于：
# - 为 RAR GPT 的 2D RoPE 提供参考实现
# - 保留原始 MAE 的位置编码逻辑以备兼容
#
# 注意：RAR GPT 使用 RoPE（旋转位置编码）而非正弦-余弦位置编码，
# 但此模块作为 VisionTS 的基础设施仍然保留。
# --------------------------------------------------------

import numpy as np

import torch


# --------------------------------------------------------
# 2D sine-cosine position embedding
# 2D正弦-余弦位置编码
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    生成2D正弦-余弦位置编码
    
    在 VisionTS/VisionTSRAR 中，图像被分成 grid_size×grid_size 的 patch 网格，
    每个 patch 需要一个位置编码来标识其在图像中的2D位置。
    
    编码方式：
    - 将 embed_dim 平均分成两半，前半编码行(高度)，后半编码列(宽度)
    - 每个维度使用正弦和余弦函数，频率从1到10000递减
    
    Args:
        embed_dim: 位置编码的维度（需为偶数）
        grid_size: 网格的高度和宽度（即patch数，通常为14）
        cls_token: 是否在开头添加CLS令牌的位置编码（全零）
    Returns:
        pos_embed: [grid_size*grid_size, embed_dim] 或 [1+grid_size*grid_size, embed_dim]（含CLS令牌）
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)  # CLS令牌位置编码为全零
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    从2D网格坐标生成位置编码
    
    将 embed_dim 平分为两部分：
    - 前半维度(grid_h方向): 编码行位置
    - 后半维度(grid_w方向): 编码列位置
    
    Args:
        embed_dim: 编码维度（需为偶数）
        grid: [2, 1, H, W] 网格坐标，grid[0]=行坐标，grid[1]=列坐标
    Returns:
        emb: [H*W, embed_dim] 位置编码
    """
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    生成1D正弦-余弦位置编码
    
    编码公式：
    - 对于位置p和维度索引i:
      - omega_i = 1 / 10000^(2i/embed_dim)
      - 编码[p, 2i]   = sin(p * omega_i)
      - 编码[p, 2i+1] = cos(p * omega_i)
    
    这种编码方式的优势：
    - 不同频率的正弦/余弦可以唯一标识每个位置
    - 相对位置关系可以通过编码的线性变换表示
    
    Args:
        embed_dim: 输出编码维度（需为偶数）
        pos: 位置列表，size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,) 频率从1递减到1/10000

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), 外积：每个位置与每个频率相乘

    emb_sin = np.sin(out)  # (M, D/2) 正弦分量
    emb_cos = np.cos(out)  # (M, D/2) 余弦分量

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D) 交替拼接sin和cos
    return emb



# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# 高分辨率位置编码插值
# 当模型分辨率与预训练权重不一致时，通过双三次插值调整位置编码
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    """
    插值位置编码以适应不同分辨率的模型
    
    当加载的预训练权重的图像分辨率与当前模型不一致时，
    需要对位置编码进行2D双三次插值，以匹配新的patch网格大小。
    CLS令牌的位置编码保持不变。
    
    Args:
        model: 当前模型
        checkpoint_model: 预训练权重字典（会被原地修改）
    """
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches  # 额外令牌数（如CLS令牌）
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
