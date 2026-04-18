"""
RandAR 采样与生成逻辑模块

本模块实现了自回归模型的采样策略和生成流程，包括：
- top_k_top_p_filtering: Top-k 和 Top-p（nucleus）采样过滤
- sample: 带温度缩放的采样函数
- logits_to_probs: 将 logits 转换为概率分布
- prefill: 推理时的预填充阶段（处理条件 token）
- decode_one_token: 推理时解码单个 token
- decode_n_tokens: 推理时解码多个 token
- generate: 完整的图像生成流程

来源:
- LlamaGen: https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/models/generate.py
- gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
- DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch._dynamo.config
import torch._inductor.config
import copy


def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """
    Top-k 和 Top-p（Nucleus）采样过滤。
    
    这两种采样策略用于控制生成多样性：
    - Top-k: 只保留概率最高的 k 个 token，其余设为 -inf
    - Top-p: 只保留累积概率达到 p 的最小 token 集合（nucleus sampling）
    
    两者可以同时使用，先应用 top-k 再应用 top-p。
    
    论文参考: Holtzman et al. "The Curious Case of Neural Text Degeneration" (http://arxiv.org/abs/1904.09751)
    
    Args:
        logits: logits 分布，shape (batch_size, vocab_size)
        top_k: 保留的最高概率 token 数，0 表示不过滤
        top_p: 累积概率阈值，1.0 表示不过滤
        filter_value: 被过滤 token 的 logits 值（默认 -inf，softmax后为0）
        min_tokens_to_keep: 每个样本最少保留的 token 数
    
    Returns:
        过滤后的 logits，shape 与输入相同
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # 安全检查
        # 找到 top-k 的最小阈值，低于该值的 token 全部过滤
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # 移除累积概率超过阈值的 token（但保留阈值之上的第一个token）
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # 至少保留 min_tokens_to_keep 个 token
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # 右移一位，确保恰好超过阈值的那个 token 也被保留
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # 将排序后的掩码散射回原始索引
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits


def sample(
    logits,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    sample_logits=True,
):
    """
    从 logits 分布中采样 token。
    
    流程：温度缩放 → top-k/top-p 过滤 → softmax → 采样
    
    注意：torch.multinomial 只支持 1D 或 2D 的概率输入。
    
    Args:
        logits: 模型输出的 logits，shape (batch_size, 1, vocab_size)
        temperature: 温度参数，越高越随机，越低越确定。1e-5以下视为贪婪解码
        top_k: Top-k 采样的 k 值
        top_p: Top-p 采样的 p 值
        sample_logits: True 为随机采样，False 为贪婪取 argmax
    
    Returns:
        idx: 采样得到的 token 索引，shape (batch_size, 1)
        probs: 采样概率分布，shape (batch_size, 1, vocab_size)
    """
    logits = logits[:, -1, :] / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    
    # 数值稳定性处理：替换 NaN 和 Inf
    logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
    logits = torch.where(torch.isinf(logits), torch.zeros_like(logits), logits)
    
    probs = F.softmax(logits, dim=-1)
    
    # 确保 probs 有效（无 NaN，和为 1）
    if torch.isnan(probs).any() or torch.isinf(probs).any():
        # 如果 softmax 后仍有问题，使用均匀分布
        probs = torch.ones_like(probs) / probs.shape[-1]
    
    if sample_logits:
        # 多项式采样
        idx = torch.multinomial(probs, num_samples=1)
    else:
        # 贪婪解码
        _, idx = torch.topk(probs, k=1, dim=-1)
    return idx, probs


def logits_to_probs(
    logits, temperature: float = 1.0, top_p: float = 1.0, top_k: int = None, **kwargs
):
    """
    将 logits 转换为概率分布（不进行采样）。
    
    与 sample() 不同，此函数仅返回概率分布，不进行采样。
    适用于需要获取完整概率分布的场景（如分析、可视化等）。
    
    Args:
        logits: 模型输出的 logits
        temperature: 温度参数
        top_p: Top-p 过滤阈值
        top_k: Top-k 过滤阈值
    
    Returns:
        probs: 概率分布，shape 与 logits 相同
    """
    logits = logits / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def prefill(
    model,
    cond_idx: torch.Tensor,
    input_pos: torch.Tensor,
    cfg_scale: float,
    **sampling_kwargs
):
    """
    推理时的预填充阶段：处理条件 token 并生成第一个图像 token。
    
    在自回归推理中，首先需要将条件信息（类别标签或文本描述）送入模型，
    获取第一个图像 token 的预测。这个过程称为"预填充"（prefill）。
    
    如果启用 CFG（Classifier-Free Guidance），会同时计算有条件和无条件的预测，
    然后按 CFG 公式混合：logits = uncond + scale * (cond - uncond)
    
    Args:
        model: 生成模型（需实现 forward 方法）
        cond_idx: 条件 token 索引
        input_pos: 输入位置索引
        cfg_scale: CFG 缩放因子，>1.0 时启用 CFG
        **sampling_kwargs: 采样参数（temperature, top_k, top_p 等）
    
    Returns:
        第一个生成的 token 索引，shape (batch_size, 1)
    """
    if cfg_scale > 1.0:
        # CFG: 同时计算条件和无条件预测
        logits, _, _ = model(None, cond_idx, input_pos)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(
            logits_combined, len(logits_combined) // 2, dim=0
        )
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
    else:
        logits, _, _ = model(None, cond_idx, input_pos)

    return sample(logits, **sampling_kwargs)[0]


def decode_one_token(
    model,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    cfg_scale: float,
    cfg_flag: bool,
    **sampling_kwargs
):
    """
    推理时解码单个 token（利用 KV Cache）。
    
    在预填充阶段之后，每次只需解码一个 token。此函数利用 KV Cache
    避免重复计算之前的 key/value，显著加速推理。
    
    Args:
        model: 生成模型
        x: 当前 token 的嵌入，shape (batch_size, 1, dim)
        input_pos: 当前 token 的位置索引，shape (1,)
        cfg_scale: CFG 缩放因子
        cfg_flag: 是否应用 CFG（在某些步骤可能关闭以加速）
        **sampling_kwargs: 采样参数
    
    Returns:
        (idx, probs): 采样结果和概率分布
    """
    assert input_pos.shape[-1] == 1
    if cfg_scale > 1.0:
        # CFG: 将输入复制两份，分别作为条件和无条件输入
        x_combined = torch.cat([x, x])
        logits, _, _ = model(x_combined, cond_idx=None, input_pos=input_pos)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(
            logits_combined, len(logits_combined) // 2, dim=0
        )
        if cfg_flag:
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        else:
            logits = cond_logits
    else:
        logits, _, _ = model(x, cond_idx=None, input_pos=input_pos)
    return sample(logits, **sampling_kwargs)


def decode_n_tokens(
    model,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    cfg_scale: float,
    cfg_interval: int,
    **sampling_kwargs
):
    """
    推理时解码多个 token（串行逐个解码，利用 KV Cache）。
    
    逐个 token 解码，每解码一个 token 就更新 KV Cache，
    这样每个 token 只需计算一次 attention，而不是像训练时那样计算完整的注意力矩阵。
    
    Args:
        model: 生成模型
        cur_token: 当前 token 索引，shape (batch_size, 1)
        input_pos: 当前位置索引
        num_new_tokens: 需要解码的 token 数量
        cfg_scale: CFG 缩放因子
        cfg_interval: 在第几个 token 后关闭 CFG（-1 表示始终开启）
        **sampling_kwargs: 采样参数
    
    Returns:
        (new_tokens, new_probs): 解码得到的 token 列表和对应概率列表
    """
    new_tokens, new_probs = [], []
    cfg_flag = True
    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):  # 对 Inductor 代码生成更友好
            if cfg_interval > -1 and i > cfg_interval:
                cfg_flag = False
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, cfg_scale, cfg_flag, **sampling_kwargs
            )
            input_pos += 1
            new_tokens.append(next_token.clone())
            new_probs.append(next_prob.clone())
            cur_token = next_token.view(-1, 1)

    return new_tokens, new_probs


@torch.no_grad()
def generate(
    model,
    cond,
    max_new_tokens,
    emb_masks=None,
    cfg_scale=1.0,
    cfg_interval=-1,
    **sampling_kwargs
):
    """
    LlamaGen 风格的图像生成函数（非 RandAR 版本）。
    
    完整的图像生成流程：
    1. 处理条件 token（类别标签或文本描述）
    2. 设置 KV Cache
    3. 预填充：生成第一个 token
    4. 逐 token 解码：利用 KV Cache 加速
    5. 返回生成的 token 序列
    
    注意：此函数是 LlamaGen 的顺序生成方式（从左到右逐 token 生成），
    RandAR 使用自己的并行解码 generate() 方法（在 RandARTransformer 类中）。
    保留此函数是为了兼容性和可能的基线对比。
    
    Args:
        model: LlamaGen 风格的 Transformer 模型
        cond: 条件信息（类别标签或文本特征）
        max_new_tokens: 最大生成 token 数
        emb_masks: 条件嵌入的注意力掩码
        cfg_scale: CFG 缩放因子
        cfg_interval: 在第几个 token 后关闭 CFG
        **sampling_kwargs: 采样参数
    
    Returns:
        生成的 token 序列，shape (batch_size, max_new_tokens)
    """
    if model.model_type == "c2i":
        # 类条件生成：cond 是类别标签索引
        if cfg_scale > 1.0:
            cond_null = torch.ones_like(cond) * model.num_classes  # 无条件标签
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
        T = 1  # 类条件只有1个 token
    elif model.model_type == "t2i":
        # 文本条件生成：cond 是文本特征序列
        if cfg_scale > 1.0:
            cond_null = torch.zeros_like(cond) + model.cls_embedding.uncond_embedding
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
        T = cond.shape[1]  # 文本条件有多个 token
    else:
        raise Exception("please check model type")

    T_new = T + max_new_tokens
    max_seq_length = T_new
    max_batch_size = cond.shape[0]

    device = cond.device
    with torch.device(device):
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
        model.setup_caches(
            max_batch_size=max_batch_size_cfg,
            max_seq_length=max_seq_length,
            dtype=model.tok_embeddings.weight.dtype,
        )

    if emb_masks is not None:
        # 处理文本条件中的注意力掩码
        assert emb_masks.shape[0] == max_batch_size
        assert emb_masks.shape[-1] == T
        if cfg_scale > 1.0:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * torch.cat(
                [emb_masks, emb_masks]
            ).unsqueeze(1)
        else:
            model.causal_mask[:, :, :T] = model.causal_mask[
                :, :, :T
            ] * emb_masks.unsqueeze(1)

        # 确保自身位置始终可见（对角线为1）
        eye_matrix = torch.eye(
            model.causal_mask.size(1), model.causal_mask.size(2), device=device
        )
        model.causal_mask[:] = model.causal_mask * (1 - eye_matrix) + eye_matrix

    # 创建空序列，逐步填充生成的 token
    seq = torch.empty((max_batch_size, T_new), dtype=torch.int, device=device)

    # 预填充阶段：处理条件 token，生成第一个图像 token
    input_pos = torch.arange(0, T, device=device)
    next_token = prefill(model, cond_combined, input_pos, cfg_scale, **sampling_kwargs)
    seq[:, T : T + 1] = next_token

    # 逐 token 解码阶段
    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    generated_tokens, _ = decode_n_tokens(
        model,
        next_token,
        input_pos,
        max_new_tokens - 1,
        cfg_scale,
        cfg_interval,
        **sampling_kwargs
    )
    seq[:, T + 1 :] = torch.cat(generated_tokens, dim=1)

    return seq[:, T:]
