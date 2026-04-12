"""
RAR 模型封装层 (RARWrapper)

将 VQ Tokenizer + RandAR GPT 封装为统一的"图像→图像"接口，
使其可无缝替换 VisionTS 中的 MAE 模型。

核心流程：
  训练模式: image_input → VQ encode → RAR GPT forward (teacher forcing) → VQ decode → reconstructed_image
  推理模式: image_input → VQ encode (可见区) → RAR GPT generate (自回归) → VQ decode → reconstructed_image

设计要点：
1. VQ Tokenizer 参数始终冻结（与原 VisionTS 冻结 MAE 的思路一致）
2. RAR GPT 参数根据 finetune_type 决定冻结策略（默认仅微调 LayerNorm）
3. 类别条件使用固定值0（时序预测不需要类别条件），通过 LabelEmbedder 的 dropout 机制处理
4. 默认使用 random 顺序（RandAR原始设计），训练时随机打乱token顺序以学习更通用的依赖关系
   推理时可切换为raster顺序以保持空间连续性
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# 导入 RandAR 核心组件（由 randar-migrator 迁移）
from .randar.randar_gpt import RandARTransformer
from .randar.tokenizer import VQModel
from .randar.generate import sample
from .randar.utils import (
    interleave_tokens,
    calculate_num_query_tokens_for_parallel_decoding,
)
from .randar.llamagen_gpt import (
    LabelEmbedder,
    precompute_freqs_cis_2d,
    find_multiple,
)

from .util import download_rar_ckpt, download_vq_ckpt


# ============================================================
# RAR 架构配置表
# ============================================================
# 每种架构对应一组超参数和预训练权重文件名
# 目前仅支持 rar_l_0.3b（Mac Air M4 32G 最合适的规格）
RAR_ARCH_CONFIG = {
    "rar_l_0.3b": {
        # RandARTransformer 构造参数
        "n_layer": 24,
        "n_head": 16,
        "dim": 1024,
        "model_type": "c2i",
        "vocab_size": 16384,
        "block_size": 256,       # 16x16 = 256 tokens
        "num_classes": 1000,
        "cls_token_num": 1,
        "resid_dropout_p": 0.1,
        "ffn_dropout_p": 0.1,
        "drop_path_rate": 0.0,
        "token_dropout_p": 0.1,
        "grad_checkpointing": True,
        "zero_class_qk": True,
        # 推理参数（训练时用 position_order=random，推理时可用raster）
        "num_inference_steps": 88,
        "position_order": "random",
        # 预训练权重文件名
        "rar_ckpt": "rbrar_l_0.3b_c2i.safetensors",
    },
}

# VQ Tokenizer 配置（从 llamagen.yaml）
VQ_CONFIG = {
    "codebook_size": 16384,
    "codebook_embed_dim": 8,
    "codebook_l2_norm": True,
    "codebook_show_usage": True,
    "commit_loss_beta": 0.25,
    "entropy_loss_ratio": 0.0,
    "encoder_ch_mult": [1, 1, 2, 2, 4],
    "decoder_ch_mult": [1, 1, 2, 2, 4],
    "z_channels": 256,
    "dropout_p": 0.0,
}


class RARWrapper(nn.Module):
    """
    RAR模型封装层：将 VQ Tokenizer + RandAR GPT 封装为统一的"图像→图像"接口
    
    核心流程：
    image_input → VQ encode → 提取可见区token → RAR GPT自回归生成 → VQ decode → reconstructed_image
    
    此封装层设计为可直接替换 VisionTS 中的 MAE 模型：
    - 训练时：forward() 返回 (reconstructed_image, loss)
    - 推理时：generate() 返回 reconstructed_image
    
    与 MAE 的关键区别：
    1. MAE 使用固定掩码 + 编码器-解码器架构，一次性预测所有被掩码的 patch
    2. RAR 使用自回归生成，逐步预测每个 token（可见区token作为条件）
    3. MAE 输出连续像素值，RAR 输出离散 token 索引再通过 VQ decode 恢复像素值
    """
    
    def __init__(
        self,
        rar_arch: str = 'rar_l_0.3b',
        finetune_type: str = 'ln',
        ckpt_dir: str = './ckpt/',
        load_ckpt: bool = True,
        num_inference_steps: int = 88,
        position_order: str = 'random',
        device: str = 'auto',
        vq_ckpt_path: str = None,
        rar_ckpt_path: str = None,
    ):
        """
        初始化 RAR 封装层
        
        Args:
            rar_arch: RAR架构选择，目前仅支持 'rar_l_0.3b'
            finetune_type: 微调策略，控制 RAR GPT 中哪些参数可训练
                - 'full': 所有参数可训练
                - 'ln': 仅 RMSNorm 参数可训练（默认，与VisionTS一致）
                - 'bias': 仅偏置参数可训练
                - 'none': 冻结所有参数（零样本推理）
            ckpt_dir: 预训练权重的本地存储目录
            load_ckpt: 是否加载预训练权重
            num_inference_steps: RAR推理步数（并行解码的总步数，越小越快但质量越低）
            position_order: token顺序策略
                - 'random': 随机顺序（RandAR原始设计，训练时随机排列）
                - 'raster': 光栅顺序（从左到右，从上到下），推理时使用，保持空间连续性
            device: 计算设备，'auto' 自动选择 CUDA > MPS > CPU
            vq_ckpt_path: VQ Tokenizer权重文件的完整路径（None则从HuggingFace自动下载）
            rar_ckpt_path: RAR GPT权重文件的完整路径（None则从HuggingFace自动下载）
        """
        super().__init__()
        
        if rar_arch not in RAR_ARCH_CONFIG:
            raise ValueError(f"Unknown RAR arch: {rar_arch}. Should be in {list(RAR_ARCH_CONFIG.keys())}")
        
        self.rar_arch = rar_arch
        self.finetune_type = finetune_type
        self.num_inference_steps = num_inference_steps
        self.position_order = position_order
        
        # 自动选择设备
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        # ============================================================
        # 1. 初始化 VQ Tokenizer（冻结权重）
        # ============================================================
        self.vq_tokenizer = VQModel(**VQ_CONFIG)
        
        # 加载 VQ Tokenizer 预训练权重
        if load_ckpt:
            if vq_ckpt_path is not None:
                self._load_vq_ckpt(vq_ckpt_path)
            else:
                vq_ckpt_path = download_vq_ckpt(ckpt_dir)
                self._load_vq_ckpt(vq_ckpt_path)
        
        # 冻结 VQ Tokenizer 所有参数
        for param in self.vq_tokenizer.parameters():
            param.requires_grad = False
        self.vq_tokenizer.eval()
        
        # ============================================================
        # 2. 初始化 RandAR GPT（可微调部分参数）
        # ============================================================
        arch_config = RAR_ARCH_CONFIG[rar_arch].copy()
        rar_ckpt_filename = arch_config.pop('rar_ckpt')
        
        # 覆盖推理参数
        arch_config['num_inference_steps'] = num_inference_steps
        arch_config['position_order'] = position_order
        
        self.rar_gpt = RandARTransformer(**arch_config)
        
        # 加载 RAR GPT 预训练权重
        if load_ckpt:
            if rar_ckpt_path is not None:
                self._load_rar_ckpt(rar_ckpt_path)
            else:
                rar_ckpt_path = download_rar_ckpt(rar_arch, ckpt_dir)
                self._load_rar_ckpt(rar_ckpt_path)
        
        # 根据 finetune_type 冻结 RAR GPT 参数
        self._apply_finetune_strategy(finetune_type)
        
        # VQ Tokenizer 的下采样率：16（因为 encoder_ch_mult 有5个元素，4次下采样 × 2^2 = 16）
        # 所以 224×224 图像 → 14×14 = 196 个空间位置... 但 block_size=256，对应 16×16
        # 实际上 VQ 的下采样率是 16，224/16=14，所以 14×14=196 ≠ 256
        # 注意：RandAR 使用 256×256 输入，256/16=16，16×16=256 tokens
        # 但 VisionTS 使用 224×224 输入，需要特殊处理
        self.vq_downsample_ratio = 16  # VQ Tokenizer 的空间下采样率
        
        # 记录 block_size（token总数）
        self.block_size = arch_config['block_size']  # 256
    
    def _load_vq_ckpt(self, ckpt_path: str):
        """加载 VQ Tokenizer 预训练权重"""
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            # 处理不同的权重格式
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            self.vq_tokenizer.load_state_dict(state_dict, strict=True)
            print(f"VQ Tokenizer checkpoint loaded from: {ckpt_path}")
        except Exception as e:
            print(f"Failed to load VQ Tokenizer checkpoint: {e}")
            print(f"VQ Tokenizer will use random initialization.")
    
    def _load_rar_ckpt(self, ckpt_path: str):
        """加载 RAR GPT 预训练权重（支持 safetensors 格式）"""
        try:
            if ckpt_path.endswith('.safetensors'):
                from safetensors.torch import load_file
                state_dict = load_file(ckpt_path)
            else:
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            
            # 处理可能的 key 前缀不匹配
            missing, unexpected = self.rar_gpt.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"RAR GPT: Missing keys ({len(missing)}): {missing[:5]}...")
            if unexpected:
                print(f"RAR GPT: Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
            print(f"RAR GPT checkpoint loaded from: {ckpt_path}")
        except Exception as e:
            print(f"Failed to load RAR GPT checkpoint: {e}")
            print(f"RAR GPT will use random initialization.")
    
    def _apply_finetune_strategy(self, finetune_type: str):
        """
        根据微调策略冻结 RAR GPT 参数
        
        与 VisionTS 的策略一致：
        - 'ln': 仅 RMSNorm 参数可训练（对应 VisionTS 的 'ln' 即仅 LayerNorm）
        - 'bias': 仅偏置参数可训练
        - 'none': 冻结所有参数（零样本推理）
        - 'full': 所有参数可训练
        """
        if finetune_type == 'full':
            return  # 不冻结任何参数
        
        for n, param in self.rar_gpt.named_parameters():
            if finetune_type == 'ln':
                # RandAR 使用 RMSNorm，权重名为 'weight'，属于包含 'norm' 的模块
                param.requires_grad = 'norm' in n.lower()
            elif finetune_type == 'bias':
                param.requires_grad = 'bias' in n
            elif finetune_type == 'none':
                param.requires_grad = False
            elif 'mlp' in finetune_type:
                param.requires_grad = '.feed_forward.' in n or 'ffn' in n.lower()
            elif 'attn' in finetune_type:
                param.requires_grad = '.attention.' in n
    
    def encode_image(self, image_input: torch.Tensor) -> torch.Tensor:
        """
        将图像编码为 1D 离散 token 索引序列
        
        使用 VQModel.encode_to_tokens() 而非 encode_indices()：
        - encode_indices() 返回 2D 空间索引 [bs, H/16, W/16]
        - encode_to_tokens() 返回 1D 展平序列 [bs, L]，L = (H/16)*(W/16)
        - RandAR GPT 需要 1D 展平格式
        
        Args:
            image_input: [bs, 3, H, W] 输入图像
        Returns:
            token_indices: [bs, L] 离散 token 索引（1D 展平，按行优先顺序）
        """
        with torch.no_grad():
            token_indices = self.vq_tokenizer.encode_to_tokens(image_input)
        return token_indices
    
    def decode_tokens(self, token_indices: torch.Tensor, image_size: int = 256) -> torch.Tensor:
        """
        将 1D 离散 token 索引序列解码为图像
        
        使用 VQModel.decode_tokens_to_image() 而非 decode_code()：
        - decode_code() 需要 2D 空间索引和手动构造 qz_shape
        - decode_tokens_to_image() 接受 1D 展平序列，自动处理 reshape 和尺寸调整
        
        Args:
            token_indices: [bs, L] 离散 token 索引（1D 展平）
            image_size: 输出图像尺寸（默认256，VQ Tokenizer 原始输出尺寸）
        Returns:
            reconstructed_image: [bs, 3, H, W] 重建图像，值域 [-1, 1]
        """
        with torch.no_grad():
            reconstructed_image = self.vq_tokenizer.decode_tokens_to_image(
                token_indices, image_size=image_size
            )
        
        return reconstructed_image
    
    def forward(
        self,
        image_input: torch.Tensor,
        num_visible_tokens: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        训练模式 forward：使用 teacher forcing + inpainting 模式
        
        与 MAE 的 forward 不同：
        - MAE: 输入图像 + 掩码 → 编码器(可见patch) → 解码器(预测被掩码patch) → 重建图像
        - RAR: 输入图像 → VQ encode(全部token) → RAR GPT(teacher forcing + visible_tokens) → VQ decode → 重建图像
        
        训练时利用 RandARTransformer.forward_train() 的 visible_tokens 参数：
        - 将可见区 token 作为已知条件传入
        - RAR GPT 只对未知区域 token 计算 cross-entropy loss
        - 这比"传入全部 token 再手动替换"更高效且语义正确
        
        Args:
            image_input: [bs, 3, H, W] 完整图像（左半有值，右半零填充）
                         注意：对于 VisionTS，H=W=224，但实际上 RAR 需要 256×256
                         我们在内部处理这个尺寸差异
            num_visible_tokens: int, 可见区域的token数量
                                对应图像左半部分（输入时序渲染的区域）
        
        Returns:
            reconstructed_image: [bs, 3, H, W] 重建图像
            loss: 训练loss（交叉熵），推理时为 None
        """
        bs = image_input.shape[0]
        
        # ============================================================
        # Step 1: 调整图像尺寸以匹配 VQ Tokenizer 的要求
        # ============================================================
        # VQ Tokenizer 将 256×256 图像编码为 16×16=256 个 token
        # VisionTS 使用 224×224 图像，需要先上采样到 256×256
        vq_input_size = 256  # VQ Tokenizer 期望的输入尺寸
        if image_input.shape[-1] != vq_input_size or image_input.shape[-2] != vq_input_size:
            image_resized = F.interpolate(
                image_input, size=(vq_input_size, vq_input_size), mode='bilinear', align_corners=False
            )
        else:
            image_resized = image_input
        
        # ============================================================
        # Step 2: VQ encode - 图像→离散token索引
        # ============================================================
        all_tokens = self.encode_image(image_resized)  # [bs, 256]
        
        # ============================================================
        # Step 3: 准备可见区 token 和目标 token
        # ============================================================
        # 在 raster 顺序下，前 num_visible_tokens 个 token 对应图像左半部分
        visible_tokens = all_tokens[:, :num_visible_tokens]  # [bs, num_visible_tokens]
        target_tokens = all_tokens  # 全部token作为目标（teacher forcing）
        
        # 类别条件：使用固定值0（时序预测不需要类别条件）
        # LabelEmbedder 在训练时会对条件标签做 dropout（class_dropout_prob=0.1），
        # 这实际上充当了正则化效果
        cond_idx = torch.zeros(bs, dtype=torch.long, device=image_input.device)
        
        # ============================================================
        # Step 4: RAR GPT forward（训练模式，teacher forcing + inpainting）
        # ============================================================
        # 利用 randar_gpt.forward_train() 的 visible_tokens 参数
        # 已知 token 被嵌入到序列开头（cond 之后），对应位置的 loss 被屏蔽
        token_logits, loss, token_order = self.rar_gpt(
            idx=all_tokens,              # 全部token用于 teacher forcing
            cond_idx=cond_idx,           # 类别条件（固定为0）
            token_order=None,            # 使用 position_order 决定顺序
            targets=target_tokens,       # 目标token（与输入相同）
            visible_tokens=visible_tokens,  # 传入可见区 token，启用 inpainting 模式
        )
        
        # ============================================================
        # Step 5: 从logits获取预测的token（取argmax）
        # ============================================================
        predicted_tokens = token_logits.argmax(dim=-1)  # [bs, block_size]
        
        # 用可见区的真实token替换预测的前num_visible_tokens个
        # 这确保输入区域的信息被完美保留
        predicted_tokens[:, :num_visible_tokens] = visible_tokens
        
        # ============================================================
        # Step 6: VQ decode - token→重建图像
        # ============================================================
        reconstructed_image = self.decode_tokens(predicted_tokens, image_size=vq_input_size)
        
        # ============================================================
        # Step 7: 恢复到原始图像尺寸
        # ============================================================
        if reconstructed_image.shape[-1] != image_input.shape[-1]:
            reconstructed_image = F.interpolate(
                reconstructed_image,
                size=(image_input.shape[-2], image_input.shape[-1]),
                mode='bilinear',
                align_corners=False,
            )
        
        # 值域对齐：将 VQ 解码输出 [-1, 1] 转换到 image_input 的值域
        # 使用归一化对齐：output = (output - mean(output)) / std(output) * std(image_input) + mean(image_input)
        recon_mean = reconstructed_image.mean(dim=[1, 2, 3], keepdim=True)
        recon_std = reconstructed_image.std(dim=[1, 2, 3], keepdim=True) + 1e-8
        img_mean = image_input.mean(dim=[1, 2, 3], keepdim=True)
        img_std = image_input.std(dim=[1, 2, 3], keepdim=True)
        
        reconstructed_image = (reconstructed_image - recon_mean) / recon_std
        reconstructed_image = reconstructed_image * img_std + img_mean
        
        # 训练模式：返回重建图像，loss 设为 None（不使用 RAR 的交叉熵）
        # 原因：最终目标是时序预测，应该使用时序 MSE loss
        # RAR 的交叉熵是 token 空间的 loss，与时序优化目标不一致
        return reconstructed_image, None
    
    @torch.no_grad()
    def generate(
        self,
        image_input: torch.Tensor,
        num_visible_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """
        推理模式 generate：使用自回归生成 + inpainting
        
        与 MAE 推理的区别：
        - MAE: 编码器处理可见patch → 解码器一次性预测所有被掩码patch
        - RAR: VQ encode可见区token → RAR GPT自回归生成未知区token → VQ decode
        
        推理流程：
        1. VQ encode 完整图像，提取可见区 token
        2. RAR GPT 自回归生成：给定可见 token 作为条件，逐步生成其余 token
           直接利用 randar_gpt.generate() 的 visible_tokens 参数
        3. VQ decode 生成的 token → 重建图像
        
        Args:
            image_input: [bs, 3, H, W] 完整图像（左半有值，右半零填充）
            num_visible_tokens: int, 可见区域的token数量
            temperature: 采样温度，1.0为标准，越低越确定
            top_k: Top-k 采样的 k 值，0 表示不过滤
            top_p: Top-p 采样的 p 值，1.0 表示不过滤
        
        Returns:
            reconstructed_image: [bs, 3, H, W] 重建图像
        """
        bs = image_input.shape[0]
        
        # ============================================================
        # Step 1: 调整图像尺寸
        # ============================================================
        vq_input_size = 256
        if image_input.shape[-1] != vq_input_size or image_input.shape[-2] != vq_input_size:
            image_resized = F.interpolate(
                image_input, size=(vq_input_size, vq_input_size), mode='bilinear', align_corners=False
            )
        else:
            image_resized = image_input
        
        # ============================================================
        # Step 2: VQ encode 提取可见区 token
        # ============================================================
        all_tokens = self.encode_image(image_resized)  # [bs, 256]
        visible_tokens = all_tokens[:, :num_visible_tokens]  # 可见区域token
        
        # ============================================================
        # Step 3: RAR GPT generate（推理模式，自回归 + inpainting）
        # ============================================================
        # 类别条件：使用固定值0（时序预测不需要类别条件）
        cond_idx = torch.zeros(bs, dtype=torch.long, device=image_input.device)
        
        # 直接利用 randar_gpt.generate() 的 visible_tokens 参数
        # 已知 token 作为 prefix，只生成剩余 block_size - num_visible_tokens 个 token
        generated_tokens = self.rar_gpt.generate(
            cond=cond_idx,
            token_order=None,        # 使用 position_order 决定顺序
            cfg_scales=(1.0, 1.0),   # 时序预测不使用 CFG
            num_inference_steps=self.num_inference_steps,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            visible_tokens=visible_tokens,  # 传入可见token，启用 inpainting 模式
        )
        
        # ============================================================
        # Step 4: VQ decode
        # ============================================================
        reconstructed_image = self.decode_tokens(generated_tokens, image_size=vq_input_size)
        
        # ============================================================
        # Step 5: 恢复到原始图像尺寸
        # ============================================================
        if reconstructed_image.shape[-1] != image_input.shape[-1]:
            reconstructed_image = F.interpolate(
                reconstructed_image,
                size=(image_input.shape[-2], image_input.shape[-1]),
                mode='bilinear',
                align_corners=False,
            )
        
        return reconstructed_image
    
    def train(self, mode: bool = True):
        """重写 train 方法，确保 VQ Tokenizer 始终在 eval 模式"""
        super().train(mode)
        self.vq_tokenizer.eval()  # VQ Tokenizer 始终在 eval 模式
        return self
