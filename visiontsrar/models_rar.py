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
from .randar.lightweight_decoder import LightweightDecoder
from .randar.llamagen_gpt import (
    LabelEmbedder,
    precompute_freqs_cis_2d,
    find_multiple,
)

from .util import download_rar_ckpt, download_vq_ckpt


# ============================================================
# Straight-Through Estimator (STE)
# 用于解决 argmax 不可导的问题
# Forward: argmax（离散索引）
# Backward: 梯度直接传给被选中位置的 logits
# ============================================================
class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits):
        ctx.save_for_backward(logits)
        max_indices = logits.argmax(dim=-1)
        return max_indices

    @staticmethod
    def backward(ctx, grad_output):
        logits, = ctx.saved_tensors
        grad_logits = torch.zeros_like(logits)
        max_indices = logits.argmax(dim=-1)
        grad_logits.scatter_(dim=-1, index=max_indices.unsqueeze(-1), src=grad_output.unsqueeze(-1))
        return grad_logits


def ste_hard_topk(logits, k=1, dim=-1):
    """
    STE 版本的多 token 选择（用于选择 top-k）

    Args:
        logits: [B, L, V] 预测 logits
        k: 每个样本选择的 token 数量
        dim: 维度
    Returns:
        selected_tokens: [B, k] 选择的 token 索引
    """
    B, L, V = logits.shape
    if k == 1:
        tokens = StraightThroughEstimator.apply(logits)
        return tokens
    else:
        _, topk_idx = torch.topk(logits.view(B, -1), k=k, dim=-1)
        return topk_idx


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
    # 轻量级模型配置 (~35M 参数)
    "rar_l_35m": {
        "n_layer": 8,
        "n_head": 12,
        "n_kv_head": 6,
        "dim": 768,
        "model_type": "c2i",
        "vocab_size": 16384,
        "block_size": 256,
        "num_classes": 1000,
        "cls_token_num": 1,
        "resid_dropout_p": 0.1,
        "ffn_dropout_p": 0.1,
        "drop_path_rate": 0.0,
        "token_dropout_p": 0.1,
        "grad_checkpointing": True,
        "zero_class_qk": True,
        "num_inference_steps": 88,
        "position_order": "random",
        "rar_ckpt": None,  # 随机初始化，不加载预训练权重
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
        use_lightweight_decoder: bool = False,
        lightweight_decoder_channels: int = 64,
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
            use_lightweight_decoder: 是否使用轻量级 Decoder（替换原 VQ Decoder）
            lightweight_decoder_channels: 轻量 Decoder 的 base_channels
        """
        super().__init__()
        
        if rar_arch not in RAR_ARCH_CONFIG:
            raise ValueError(f"Unknown RAR arch: {rar_arch}. Should be in {list(RAR_ARCH_CONFIG.keys())}")
        
        self.rar_arch = rar_arch
        self.finetune_type = finetune_type
        self.num_inference_steps = num_inference_steps
        self.position_order = position_order
        self.use_lightweight_decoder = use_lightweight_decoder
        self.lightweight_decoder_channels = lightweight_decoder_channels
        
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
        
        # 冻结 VQ Tokenizer 部分参数（端到端训练时只冻结 Encoder 和 Quantize）
        # - Encoder: 冻结（预训练的视觉特征提取器，不需要改变）
        # - Quantize: 冻结（码本，通过 EMA 更新，不支持梯度更新）
        # - Decoder: 解冻（允许学习，实现端到端训练）
        for name, param in self.vq_tokenizer.named_parameters():
            if 'encoder' in name.lower() or 'quantize' in name.lower():
                param.requires_grad = False
            else:
                param.requires_grad = True
        self.vq_tokenizer.eval()

        # 如果使用轻量级 Decoder，替换原 VQ Decoder
        if self.use_lightweight_decoder:
            self._setup_lightweight_decoder()
        
        # ============================================================
        # 2. 初始化 RandAR GPT（可微调部分参数）
        # ============================================================
        arch_config = RAR_ARCH_CONFIG[rar_arch].copy()
        rar_ckpt_filename = arch_config.pop('rar_ckpt')
        
        # 覆盖推理参数
        arch_config['num_inference_steps'] = num_inference_steps
        arch_config['position_order'] = position_order

        # 检查是否是预训练模型
        rar_ckpt_file = arch_config.get('rar_ckpt')
        self.is_pretrained = rar_ckpt_file is not None

        self.rar_gpt = RandARTransformer(**arch_config)

        # 加载 RAR GPT 预训练权重（仅预训练模型）
        if load_ckpt and self.is_pretrained:
            if rar_ckpt_path is not None:
                self._load_rar_ckpt(rar_ckpt_path)
            else:
                rar_ckpt_path = download_rar_ckpt(rar_arch, ckpt_dir)
                self._load_rar_ckpt(rar_ckpt_path)
        elif not self.is_pretrained:
            # 轻量模型：随机初始化，打印参数量
            total_params = sum(p.numel() for p in self.rar_gpt.parameters())
            print(f"[轻量模型 {rar_arch}] 随机初始化，总参数量: {total_params:,}")

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

    def _setup_lightweight_decoder(self):
        """
        设置轻量级 Decoder 替换原 VQ Decoder

        轻量 Decoder 会作为 vq_tokenizer.decoder 替换原 Decoder，
        并且默认解冻参与训练，实现端到端优化。
        """
        print(f"Setting up lightweight decoder (base_channels={self.lightweight_decoder_channels})...")
        self.lightweight_decoder = LightweightDecoder(
            z_channels=256,
            base_channels=self.lightweight_decoder_channels,
            ch_mult=[1, 2, 4],
            num_res_blocks=2,
            norm_type='group',
            dropout=0.0,
            out_channels=3,
            use_attn_last_only=True,
            use_depthwise=False,
        )
        self.vq_tokenizer.decoder = self.lightweight_decoder
        for param in self.lightweight_decoder.parameters():
            param.requires_grad = True
        self.lightweight_decoder.train()
        total_params = sum(p.numel() for p in self.lightweight_decoder.parameters())
        print(f"Lightweight decoder parameters: {total_params:,}")

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
        根据微调策略冻结 RAR GPT 和 VQ Tokenizer 参数

        与 VisionTS 的策略一致：
        - 'ln': 仅 RMSNorm 参数可训练（对应 VisionTS 的 'ln' 即仅 LayerNorm）
        - 'bias': 仅偏置参数可训练
        - 'none': 冻结所有参数（零样本推理）
        - 'full': 所有参数可训练
        - 'In': Inpainting 模式，冻结 RAR GPT，训练 VQ Decoder
        - 'In_light': 轻量级 Inpainting 模式，冻结 RAR GPT 和 VQ Decoder，只训练 post_quant_conv
        """
        aliases = {
            'in': 'In',
            'inlight': 'In_light',
            'in_light': 'In_light',
            'in-light': 'In_light',
            'none': 'none',
            'full': 'full',
            'ln': 'ln',
            'bias': 'bias',
        }
        normalized_ft_type = aliases.get(finetune_type, finetune_type)
        valid_prefixes = ('mlp', 'attn')
        valid_types = {'full', 'ln', 'bias', 'none', 'In', 'In_light'}
        if normalized_ft_type not in valid_types and not normalized_ft_type.startswith(valid_prefixes):
            raise ValueError(
                f"Unknown finetune_type: {finetune_type}. "
                f"Supported values: {sorted(valid_types)} plus prefixes {valid_prefixes}."
            )
        finetune_type = normalized_ft_type

        # ============================================================
        # 1. 处理 VQ Tokenizer 的冻结策略
        # ============================================================
        if finetune_type == 'In_light':
            # 轻量级模式：只训练 post_quant_conv，其他全部冻结
            for name, param in self.vq_tokenizer.named_parameters():
                if 'post_quant_conv' in name.lower():
                    param.requires_grad = True
                    print(f"VQ: {name} - 可训练")
                else:
                    param.requires_grad = False
            self.vq_tokenizer.eval()  # 保持 eval 模式，但 post_quant_conv 会有梯度
        elif finetune_type == 'In':
            # Inpainting 模式：训练 Decoder
            if self.use_lightweight_decoder:
                # 使用轻量Decoder：冻结整个VQ Tokenizer，只训练轻量Decoder
                for param in self.vq_tokenizer.parameters():
                    param.requires_grad = False
                self.vq_tokenizer.eval()
                # 解冻轻量Decoder
                for param in self.lightweight_decoder.parameters():
                    param.requires_grad = True
                print(f"[In模式] 轻量Decoder已解冻，参数量: {sum(p.numel() for p in self.lightweight_decoder.parameters()):,}")
            else:
                # 使用原版VQ Decoder：训练VQ Decoder
                for name, param in self.vq_tokenizer.named_parameters():
                    if 'encoder' in name.lower() or 'quantize' in name.lower():
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                self.vq_tokenizer.eval()
        else:
            # 其他模式：冻结整个 VQ Tokenizer
            for param in self.vq_tokenizer.parameters():
                param.requires_grad = False
            self.vq_tokenizer.eval()
        
        # ============================================================
        # 2. 处理 RAR GPT 的冻结策略
        # ============================================================
        # 【轻量模型】非预训练模型：全部可训练
        if not self.is_pretrained:
            for param in self.rar_gpt.parameters():
                param.requires_grad = True
            print(f"[轻量模型] RAR GPT 全部可训练，参数量: {sum(p.numel() for p in self.rar_gpt.parameters()):,}")
        else:
            # 【预训练模型】强制冻结 RAR GPT
            for param in self.rar_gpt.parameters():
                param.requires_grad = False
            print(f"[优化] RAR GPT 已冻结，参数量: {sum(p.numel() for p in self.rar_gpt.parameters()):,}")
        
        # 根据 finetune_type 处理 VQ Tokenizer
        if finetune_type == 'full':
            # 全参数训练：解冻 VQ Decoder
            for name, param in self.vq_tokenizer.named_parameters():
                if 'encoder' in name.lower() or 'quantize' in name.lower():
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        elif finetune_type in ('In', 'In_light'):
            # Inpainting 模式：训练 VQ Decoder
            if self.use_lightweight_decoder:
                for param in self.lightweight_decoder.parameters():
                    param.requires_grad = True
            else:
                for name, param in self.vq_tokenizer.named_parameters():
                    if 'encoder' in name.lower() or 'quantize' in name.lower():
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
        else:
            # 其他模式：冻结整个 VQ Tokenizer
            for param in self.vq_tokenizer.parameters():
                param.requires_grad = False
    
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

        注意：训练时不能使用 torch.no_grad()，否则会阻断梯度流

        Args:
            token_indices: [bs, L] 离散 token 索引（1D 展平）
            image_size: 输出图像尺寸（默认256，VQ Tokenizer 的输入尺寸）
        Returns:
            reconstructed_image: [bs, 3, H, W] 重建图像，值域 [-1, 1]
        """
        def _mem():
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024**3
            return 0
        
        # print(f"[decode_tokens] 开始: {_mem():.2f}GB")
        spatial_size = int(token_indices.shape[1] ** 0.5)
        # print(f"[decode_tokens] spatial_size={spatial_size}, token_indices shape={token_indices.shape}")
        codes = token_indices.view(token_indices.shape[0], spatial_size, spatial_size)
        # print(f"[decode_tokens] codes shape={codes.shape}")
        qz_shape = (
            token_indices.shape[0],
            self.vq_tokenizer.codebook_embed_dim,
            spatial_size,
            spatial_size
        )
        # print(f"[decode_tokens] qz_shape={qz_shape}")
        quant = self.vq_tokenizer.quantize.get_codebook_entry(codes, qz_shape, channel_first=True)
        # print(f"[decode_tokens] quant shape={quant.shape}, 预计显存={quant.numel() * 4 / 1024**3:.2f}GB")
        # print(f"[decode_tokens] quant之后: {_mem():.2f}GB")
        # print(f"[decode_tokens] vq_tokenizer.decoder type: {type(self.vq_tokenizer.decoder)}")
        reconstructed_image = self.vq_tokenizer.decode(quant)
        # print(f"[decode_tokens] decode之后: {_mem():.2f}GB, recon shape={reconstructed_image.shape}")
        if reconstructed_image.shape[-1] != image_size:
            reconstructed_image = F.interpolate(
                reconstructed_image, size=(image_size, image_size), mode="bicubic"
            )

        return reconstructed_image
    
    def forward(
        self,
        image_input: torch.Tensor,
        num_visible_tokens: int = None,
        current_epoch: int = 0,
        use_teacher_forcing: bool = None,  # 【优化】允许外部控制
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        训练/推理统一接口

        【训练模式】：
        1. VQ encode(整个图像) → all_tokens
        2. 打乱 token 顺序：visible 用 raster，query 用 random
        3. VQ decode(shuffled_tokens) → recon_shuffled
        4. 逆变换恢复原始顺序
        5. Loss = MAE(recon_history, history) + MAE(recon_future, future)
        6. 梯度只传给 VQ Tokenizer

        【推理模式】：
        1. VQ encode(可见部分) → visible_tokens
        2. RAR GPT generate → generated_tokens
        3. VQ decode(generated_tokens) → reconstructed_image

        Args:
            image_input: [bs, 3, H, W] 完整图像（训练时含真实future，推理时右半为零）
            num_visible_tokens: int, 可见区域的token数量
            current_epoch: int, 当前训练epoch（用于Schedule Sampling）

        Returns:
            reconstructed_image: [bs, 3, H, W] 重建图像
            loss: 分开计算的 MAE loss（训练时），推理时为 None
        """
        bs = image_input.shape[0]

        # ============================================================
        # Step 1: 调整图像尺寸以匹配 VQ Tokenizer 的要求
        # ============================================================
        vq_input_size = 256
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
        # Step 3: 训练/推理分支
        # ============================================================
        if self.training:
            # 【训练模式】打乱 token 顺序 + VQ 重建 + 分开 MAE
            return self._forward_train(
                image_resized, all_tokens, image_input, num_visible_tokens, vq_input_size, current_epoch, use_teacher_forcing
            )
        else:
            # 【推理模式】使用 RAR GPT 生成
            if num_visible_tokens is None:
                raise ValueError("num_visible_tokens must be provided in inference mode")

            visible_tokens = all_tokens[:, :num_visible_tokens]

            cond_idx = torch.zeros(bs, dtype=torch.long, device=image_input.device)
            generated_tokens = self.rar_gpt.generate(
                cond=cond_idx,
                token_order=None,
                visible_tokens=visible_tokens,
            )

            reconstructed_image = self.decode_tokens(generated_tokens, image_size=vq_input_size)

            if reconstructed_image.shape[-1] != image_input.shape[-1]:
                reconstructed_image = F.interpolate(
                    reconstructed_image,
                    size=(image_input.shape[-2], image_input.shape[-1]),
                    mode='bilinear',
                    align_corners=False,
                )

            return reconstructed_image, None

    def _forward_train(
        self,
        image_resized: torch.Tensor,
        all_tokens: torch.Tensor,
        image_input: torch.Tensor,
        num_visible_tokens: int,
        vq_input_size: int,
        current_epoch: int = 0,
        use_teacher_forcing: bool = None,  # 【优化】允许外部控制
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        训练模式 forward：使用 RAR GPT 生成 + MAE/MSE loss

        流程：
        1. VQ encode 获取 all_tokens
        2. RAR GPT forward（冻结但参与，计算 token 顺序）
        3. VQ decode
        4. 计算 MAE/MSE loss（历史 + 未来）
        
        Schedule Sampling策略：
        - 0~2 epoch: 100% teacher forcing
        - 3~6 epoch: 线性下降 1.0 → 0.2
        - 7+ epoch: 保持20%
        
        【优化】use_teacher_forcing 参数：
        - None: 使用内部随机决策（向后兼容）
        - True/False: 使用外部决策（用于训练循环控制）
        """
        def _mem():
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024**3
            return 0

        bs = all_tokens.shape[0]
        total_tokens = all_tokens.shape[1]

        # -------------------------------------------------
        # Step 1: 准备 visible tokens 和条件
        # -------------------------------------------------
        visible_tokens = all_tokens[:, :num_visible_tokens]
        cond_idx = torch.zeros(bs, dtype=torch.long, device=all_tokens.device)

        # -------------------------------------------------
        # Step 2: 100% generate（所有模型都用generate，不使用teacher forcing）
        # -------------------------------------------------
        query_len = total_tokens - num_visible_tokens
        max_gen = query_len  # 生成全部 query tokens

        with torch.no_grad():
            generated_tokens = self.rar_gpt.generate(
                cond=cond_idx,
                token_order="random",     # 训练 generate 使用 random 顺序（泛化能力）
                visible_tokens=visible_tokens,
                max_new_tokens=max_gen,   # 生成全部 query tokens
                num_inference_steps=44,   # 训练时使用 44 步加速
                kv_window_size=64,        # KV window 限制（训练激进设置）
            )

        # 全部使用生成的 tokens（100% generate）
        predicted_tokens = generated_tokens

        # -------------------------------------------------
        # Step 4: VQ decode 预测的 tokens
        # -------------------------------------------------
        recon_image = self.decode_tokens(predicted_tokens, image_size=vq_input_size)

        # -------------------------------------------------
        # Step 5: 恢复到原始图像尺寸
        # -------------------------------------------------
        if recon_image.shape[-1] != image_input.shape[-1]:
            recon_image = F.interpolate(
                recon_image,
                size=(image_input.shape[-2], image_input.shape[-1]),
                mode='bilinear',
                align_corners=False,
            )

        # -------------------------------------------------
        # Step 6: 值域对齐
        # -------------------------------------------------
        recon_image = self._normalize_image(recon_image, image_input)

        # -------------------------------------------------
        # Step 7: 计算 MAE/MSE loss（历史 + 未来分开）
        # -------------------------------------------------
        H = image_input.shape[2]
        W = image_input.shape[3]
        mid_w = W // 2

        history_input = image_input[:, :, :, :mid_w]
        future_input = image_input[:, :, :, mid_w:]
        history_recon = recon_image[:, :, :, :mid_w]
        future_recon = recon_image[:, :, :, mid_w:]

        loss_mae = F.l1_loss(history_recon, history_input) + F.l1_loss(future_recon, future_input)
        loss_mse = F.mse_loss(history_recon, history_input) + F.mse_loss(future_recon, future_input)

        loss = loss_mae + loss_mse

        return recon_image, loss

    def _normalize_image(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """值域对齐：将重建图像归一化到目标图像的值域"""
        recon_mean = recon.mean(dim=[1, 2, 3], keepdim=True)
        recon_std = recon.std(dim=[1, 2, 3], keepdim=True) + 1e-8
        img_mean = target.mean(dim=[1, 2, 3], keepdim=True)
        img_std = target.std(dim=[1, 2, 3], keepdim=True) + 1e-8

        recon = (recon - recon_mean) / recon_std
        recon = recon * img_std + img_mean
        return recon
    
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
        # 【优化】推理时使用 raster 顺序 + 88 步保证生成质量
        generated_tokens = self.rar_gpt.generate(
            cond=cond_idx,
            token_order="raster",      # 推理使用 raster 顺序（空间连续，局部一致）
            cfg_scales=(1.0, 1.0),     # 时序预测不使用 CFG
            num_inference_steps=88,    # 推理时使用 88 步（高质量生成）
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            visible_tokens=visible_tokens,  # 传入可见token，启用 inpainting 模式
            kv_window_size=128,        # KV window 限制（推理保守设置）
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
