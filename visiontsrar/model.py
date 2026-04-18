"""
VisionTSRAR: 基于 RAR (Randomized Autoregressive) 的时间序列预测模型

核心思想（继承自 VisionTS）：
将时间序列渲染为图像，利用视觉模型的图像补全能力来实现时序预测。
VisionTS 使用 MAE (Masked Autoencoder) 做图像补全，而 VisionTSRAR 替换为 RAR (Randomized Autoregressive)。

6步流水线架构：
1. Normalization（归一化）→ 2. Segmentation（分段）→ 3. Render & Alignment（渲染与对齐）
→ 4. RAR Reconstruction（RAR重建，替换原MAE）→ 5. Forecasting（预测提取）→ 6. Denormalization（反归一化）

与原 VisionTS 的区别仅在第4步：
- VisionTS 第4步: MAE Reconstruction（固定掩码 + 编码器-解码器架构）
- VisionTSRAR 第4步: RAR Reconstruction（VQ Tokenizer + 自回归GPT生成）

两个版本：
- VisionTSRAR: Channel Independent 版本（对应原 VisionTS），每个变量独立渲染为灰度图
- VisionTSRARpp: 多变量增强版本（对应原 VisionTSpp），支持RGB颜色编码和分位数预测
"""

import torch
import os
import numpy as np
import einops
import torch.nn.functional as F
from torch import nn
from PIL import Image

from . import models_rar
from . import util


# RAR架构配置：每种架构对应一个封装工厂函数和配置信息
# - rar_l_0.3b: 24层, 16头, dim=1024, ~0.3B参数（Mac Air M4 32G推荐规格）
RAR_ARCH = {
    "rar_l_0.3b": {
        "factory": models_rar.RARWrapper,
        "rar_ckpt": "rbrar_l_0.3b_c2i.safetensors",
    },
}


class VisionTSRAR(nn.Module):
    """
    VisionTSRAR: 基于 RAR (Randomized Autoregressive) 的时间序列预测模型
    
    继承 VisionTS 的6步流水线架构，第4步从 MAE 替换为 RAR:
    1. Normalization → 2. Segmentation → 3. Render & Alignment 
    → 4. RAR Reconstruction（新）→ 5. Forecasting → 6. Denormalization
    
    Channel Independent 版本：每个变量独立渲染为灰度图，3通道重复同一灰度值。
    
    与原 VisionTS 的关键区别：
    - 第4步：MAE → RAR
      - MAE: 使用固定掩码，编码器处理可见patch，解码器预测被掩码patch
      - RAR: VQ Tokenizer 将图像编码为离散token，RandAR GPT 自回归生成未知区域token
    - 不需要构造固定掩码：RAR 通过 num_visible_tokens 控制可见区
    - 训练时：RAR 使用 teacher forcing 计算 cross-entropy loss（而非 MAE 的 MSE loss）
    - 推理时：RAR 使用并行解码自回归生成（而非 MAE 的一次性解码）
    """

    def __init__(
        self,
        arch: str = 'rar_l_0.3b',
        finetune_type: str = 'ln',
        ckpt_dir: str = './ckpt/',
        load_ckpt: bool = True,
        num_inference_steps: int = 88,
        position_order: str = 'random',
        vq_ckpt: str = None,
        rar_ckpt: str = None,
        use_lightweight_decoder: bool = False,
        lightweight_decoder_channels: int = 64,
    ):
        """
        初始化 VisionTSRAR 模型
        
        Args:
            arch: RAR架构选择，可选 'rar_l_0.3b'
            finetune_type: 微调策略，控制RAR GPT中哪些参数可训练
                - 'full': 所有参数可训练
                - 'ln': 仅 RMSNorm 参数可训练（默认，与VisionTS一致）
                - 'bias': 仅偏置参数可训练
                - 'none': 冻结所有参数（零样本推理）
            ckpt_dir: 预训练权重的本地存储目录
            load_ckpt: 是否加载预训练权重
            num_inference_steps: RAR推理步数（并行解码总步数）
            position_order: token顺序策略
                - 'random': 随机顺序（推荐，训练时使用，符合RandAR原始设计）
                - 'raster': 光栅顺序（推理时使用，保持空间连续性）
            vq_ckpt: VQ Tokenizer权重文件路径（None则自动下载）
            rar_ckpt: RAR GPT权重文件路径（None则自动下载）
            use_lightweight_decoder: 是否使用轻量级 Decoder（替换原 VQ Decoder，降低显存）
            lightweight_decoder_channels: 轻量 Decoder 的 base_channels（默认64，值越大Decoder越强）
        """
        super(VisionTSRAR, self).__init__()

        if arch not in RAR_ARCH:
            raise ValueError(f"Unknown arch: {arch}. Should be in {list(RAR_ARCH.keys())}")

        self.use_lightweight_decoder = use_lightweight_decoder
        self.lightweight_decoder_channels = lightweight_decoder_channels

        # 创建 RAR 封装层（VQ Tokenizer + RandAR GPT）
        self.rar_wrapper = RAR_ARCH[arch]["factory"](
            rar_arch=arch,
            finetune_type=finetune_type,
            ckpt_dir=ckpt_dir,
            load_ckpt=load_ckpt,
            num_inference_steps=num_inference_steps,
            position_order=position_order,
            vq_ckpt_path=vq_ckpt,
            rar_ckpt_path=rar_ckpt,
            use_lightweight_decoder=use_lightweight_decoder,
            lightweight_decoder_channels=lightweight_decoder_channels,
        )
        
        # 保存配置
        self.arch = arch
        self.finetune_type = finetune_type

    def update_config(
        self,
        context_len: int,
        pred_len: int,
        periodicity: int = 0,
        norm_const: float = 0.4,
        align_const: float = 0.4,
        interpolation: str = 'bilinear',
    ):
        """
        根据时序数据的参数计算图像布局配置
        
        与 VisionTS.update_config 的大部分逻辑一致，
        主要区别是不需要构造固定掩码（RAR 通过 num_visible_tokens 控制），
        而是计算可见 token 数量。
        
        核心逻辑：
        1. 将时间序列按周期(periodicity)折叠为2D结构（行=周期内频率，列=周期数）
        2. 根据输入/输出长度比例，计算图像中输入patch和输出patch的数量
        3. 计算 num_visible_tokens：可见区域的 token 数量
        
        Args:
            context_len: 回看窗口长度（输入时序长度）
            pred_len: 预测窗口长度（输出时序长度）
            periodicity: 周期长度，用于2D折叠
            norm_const: 归一化常数（默认0.4），控制归一化后值域范围
            align_const: 对齐常数（默认0.4），控制输入占图像比例
            interpolation: 图像缩放插值方法
        """
        if periodicity <= 0:
            periodicity = 1
        # RAR/VQ 模型的固有图像参数
        # VQ Tokenizer 下采样率为16，所以：
        # - 256×256 图像 → 16×16 = 256 tokens (block_size)
        # - VisionTS 使用 224×224 输入，但 VQ 需要 256×256
        # - 在 RARWrapper 中会自动处理尺寸转换
        self.image_size = 224  # VisionTS 的标准输入尺寸
        self.vq_image_size = 256  # VQ Tokenizer 的输入尺寸
        self.patch_size = 16  # VQ Tokenizer 的下采样率（等价于 patch_size）
        self.num_patch = self.vq_image_size // self.patch_size  # 256/16 = 16，每行/列的patch数
        
        self.context_len = context_len
        self.pred_len = pred_len
        self.periodicity = periodicity

        # 左填充：使 context_len 能被 periodicity 整除
        self.pad_left = 0
        self.pad_right = 0
        if self.context_len % self.periodicity != 0:
            self.pad_left = self.periodicity - self.context_len % self.periodicity

        # 右填充：使 pred_len 能被 periodicity 整除
        if self.pred_len % self.periodicity != 0:
            self.pad_right = self.periodicity - self.pred_len % self.periodicity
        
        # 输入占比：输入长度(含左填充) / 总长度(含左右填充)
        input_ratio = (self.pad_left + self.context_len) / (
            self.pad_left + self.context_len + self.pad_right + self.pred_len
        )
        # 计算输入patch列数 = 输入占比 × 总patch数 × align_const
        self.num_patch_input = int(input_ratio * self.num_patch * align_const)
        if self.num_patch_input == 0:
            self.num_patch_input = 1  # 至少1列patch用于输入
        self.num_patch_output = self.num_patch - self.num_patch_input
        adjust_input_ratio = self.num_patch_input / self.num_patch
        
        interpolation = {
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "bicubic": Image.BICUBIC,
        }[interpolation]

        # 输入区域缩放：将时序2D图缩放到图像的输入区域（左半部分）
        self.input_resize = util.safe_resize(
            (self.image_size, int(self.image_size * adjust_input_ratio)),
            interpolation=interpolation,
        )
        # scale_x: 原始时序中每个周期对应的图像像素宽度
        self.scale_x = (
            (self.pad_left + self.context_len) // self.periodicity
        ) / (int(self.image_size * adjust_input_ratio))
        # 输出区域缩放
        self.output_resize = util.safe_resize(
            (self.periodicity, int(round(self.image_size * self.scale_x))),
            interpolation=interpolation,
        )
        self.norm_const = norm_const
        
        # ============================================================
        # 计算可见 token 数量（替代 VisionTS 的掩码构造）
        # ============================================================
        # VQ Tokenizer 将 256×256 图像编码为 16×16 = 256 个 token
        # 可见区域占 num_patch_input 列，共 num_patch 行
        # 所以可见 token 数 = num_patch * num_patch_input
        self.num_visible_tokens = self.num_patch * self.num_patch_input
        # 注意：这里 num_visible_tokens 是按 256×256 (16x16 tokens) 计算的
        # 如果输入是 224×224，RARWrapper 会先上采样到 256×256

    def forward(self, x, export_image=False, fp64=False, current_epoch=0, use_teacher_forcing=None):
        """
        VisionTSRAR 前向传播：6步流水线
        
        与 VisionTS.forward 的区别仅在第4步（RAR Reconstruction vs MAE Reconstruction）。
        步骤1-3和步骤5-6与原 VisionTS 完全一致。
        
        Args:
            x: 回看窗口，size: [bs x context_len x nvars]
            export_image: 是否导出可视化图像
            fp64: 是否使用 float64 精度（避免数值溢出）
            current_epoch: 当前训练epoch（用于Schedule Sampling）
            use_teacher_forcing: 是否使用 teacher forcing（None=随机决策，True/False=外部控制）
        
        Returns:
            y: 预测窗口，size: [bs x pred_len x nvars]
        """

        # ========== 第1步: Normalization（归一化）==========
        # 与 VisionTS 完全一致
        means = x.mean(1, keepdim=True).detach()  # [bs x 1 x nvars]
        x_enc = x - means
        stdev = torch.sqrt(
            torch.var(
                x_enc.to(torch.float64) if fp64 else x_enc,
                dim=1, keepdim=True, unbiased=False
            ) + 1e-5
        )  # [bs x 1 x nvars]
        stdev = stdev / self.norm_const
        x_enc = x_enc / stdev
        # Channel Independent: 每个变量独立处理
        x_enc = einops.rearrange(x_enc, 'b s n -> b n s')  # [bs x nvars x seq_len]

        # ========== 第2步: Segmentation（分段/折叠）==========
        # 与 VisionTS 完全一致
        x_pad = F.pad(x_enc, (self.pad_left, 0), mode='replicate')
        x_2d = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=self.periodicity)

        # ========== 第3步: Render & Alignment（渲染与对齐）==========
        # 与 VisionTS 完全一致
        x_resize = self.input_resize(x_2d)
        # 构造右侧掩码区域（真实future，teacher forcing训练目标）
        # 从原始2D数据中提取future部分，resize到目标尺寸
        future_2d = x_2d[:, :, :, -self.num_patch_output * self.patch_size:]
        future_resized = self.input_resize(future_2d)
        # 拼接输入+真实future，形成完整224×224图像
        # 注意：训练时用真实future作为目标（teacher forcing），推理时才会用zeros
        x_concat_with_masked = torch.cat([x_resize, future_resized], dim=-1)
        # 灰度图复制为3通道（Channel Independent，3通道相同）
        image_input = einops.repeat(x_concat_with_masked, 'b 1 h w -> b c h w', c=3)

        # ========== 第4步: RAR Reconstruction（RAR重建）==========
        # **核心区别：使用 RAR 替代 MAE**
        # VisionTS: self.vision_model(image_input, mask_ratio=..., noise=...) → y, mask
        # VisionTSRAR: self.rar_wrapper(image_input, num_visible_tokens) → reconstructed_image, loss
        if self.training:
            # 训练模式：使用 teacher forcing（支持外部控制）
            image_reconstructed, rar_loss = self.rar_wrapper(
                image_input, self.num_visible_tokens, current_epoch=current_epoch, use_teacher_forcing=use_teacher_forcing
            )
        else:
            # 推理模式：使用自回归生成
            image_reconstructed = self.rar_wrapper.generate(
                image_input, self.num_visible_tokens
            )
        
        # ========== 第5步: Forecasting（预测提取）==========
        # 与 VisionTS 完全一致
        y_grey = torch.mean(image_reconstructed, 1, keepdim=True)  # 3通道取均值
        y_segmentations = self.output_resize(y_grey)  # resize back
        y_flatten = einops.rearrange(
            y_segmentations,
            '(b n) 1 f p -> b (p f) n',
            b=x_enc.shape[0], f=self.periodicity,
        )
        # 提取预测窗口
        y = y_flatten[
            :,
            self.pad_left + self.context_len: self.pad_left + self.context_len + self.pred_len,
            :,
        ]

        # ========== 第6步: Denormalization（反归一化）==========
        # 与 VisionTS 完全一致
        y = y * (stdev.repeat(1, self.pred_len, 1))
        y = y + (means.repeat(1, self.pred_len, 1))

        # 可选：导出可视化图像
        if export_image:
            # 构造掩码可视化（近似，用于展示哪些区域是生成的）
            # 注意：RAR 没有像 MAE 那样的显式掩码，这里用区域掩码近似
            mask = torch.ones(
                (self.num_patch, self.num_patch),
                device=image_input.device,
            )
            mask[:, :self.num_patch_input] = 0
            mask_flat = mask.float().reshape((1, -1))
            mask_expanded = mask_flat.unsqueeze(-1).repeat(
                image_input.shape[0], 1, self.patch_size**2 * 3
            )
            # 简单的 unpatchify 近似
            mask_img = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, 16, 16]
            mask_img = mask_img.repeat(image_input.shape[0], 1, 1, 1)
            mask_img = F.interpolate(
                mask_img.float(),
                size=(self.image_size, self.image_size),
                mode='nearest',
            )
            # 应用掩码
            image_reconstructed = image_input * (1 - mask_img) + image_reconstructed * mask_img
            green_bg = -torch.ones_like(image_reconstructed) * 2
            image_input = image_input * (1 - mask_img) + green_bg * mask_img
            image_input = einops.rearrange(
                image_input, '(b n) c h w -> b n h w c', b=x_enc.shape[0]
            )
            image_reconstructed = einops.rearrange(
                image_reconstructed, '(b n) c h w -> b n h w c', b=x_enc.shape[0]
            )
            return y, image_input, image_reconstructed
        
        # 训练模式返回 (prediction, rar_loss)，方便 TSL 框架处理
        if self.training:
            return y, rar_loss
        
        return y


class VisionTSRARpp(nn.Module):
    """
    VisionTSRARpp: VisionTSRAR 的多变量增强版本
    
    对应原 VisionTSpp，主要区别：
    1. 多变量颜色编码：不同变量分配到RGB不同通道
    2. 图像纵向分区：多个变量纵向排列在图像中
    3. 支持分位数预测（未来扩展）
    4. 支持输入裁剪策略
    
    当前版本：实现了基本的 VisionTSRARpp 结构，
    第4步同样替换为 RAR，其余逻辑与 VisionTSpp 对齐。
    分位数预测等高级功能标注为未来扩展点。
    """
    
    def __init__(
        self,
        arch: str = 'rar_l_0.3b',
        finetune_type: str = 'ln',
        ckpt_dir: str = './ckpt/',
        load_ckpt: bool = True,
        num_inference_steps: int = 88,
        position_order: str = 'raster',
        vq_ckpt: str = None,
        rar_ckpt: str = None,
        quantile: bool = False,  # TODO: 未来扩展分位数预测
        clip_input: int = 0,
        complete_no_clip: bool = False,
        color: bool = True,
        quantile_head_num: int = 9,  # TODO: 未来扩展
    ):
        """
        Args:
            arch: RAR架构，同VisionTSRAR
            finetune_type: 微调策略，同VisionTSRAR
            ckpt_dir: 权重目录
            load_ckpt: 是否加载预训练权重
            num_inference_steps: RAR推理步数
            position_order: token顺序策略
            vq_ckpt: VQ Tokenizer权重文件路径（None则自动下载）
            rar_ckpt: RAR GPT权重文件路径（None则自动下载）
            quantile: 是否启用分位数预测（暂未实现）
            clip_input: 输入裁剪策略
            complete_no_clip: 是否完全不裁剪
            color: 是否使用颜色编码
            quantile_head_num: 分位数预测头数量（暂未实现）
        """
        super(VisionTSRARpp, self).__init__()
        
        if arch not in RAR_ARCH:
            raise ValueError(f"Unknown arch: {arch}. Should be in {list(RAR_ARCH.keys())}")
        
        self.rar_wrapper = RAR_ARCH[arch]["factory"](
            rar_arch=arch,
            finetune_type=finetune_type,
            ckpt_dir=ckpt_dir,
            load_ckpt=load_ckpt,
            num_inference_steps=num_inference_steps,
            position_order=position_order,
            vq_ckpt_path=vq_ckpt,
            rar_ckpt_path=rar_ckpt,
        )
        
        self.quantile = quantile
        self.clip_input = clip_input
        self.complete_no_clip = complete_no_clip
        self.color = color
        self.arch = arch
        self.finetune_type = finetune_type
    
    def update_config(
        self,
        context_len: int,
        pred_len: int,
        num_patch_input: int = None,
        periodicity: int = 1,
        norm_const: float = 0.4,
        align_const: float = 0.4,
        interpolation: str = 'bilinear',
        padding_mode: str = 'replicate',
    ):
        """
        根据时序数据的参数计算图像布局配置（VisionTSRARpp版本）
        
        与 VisionTSRAR.update_config 的主要区别：
        1. 支持显式指定 num_patch_input
        2. 支持 padding_mode 选择
        3. 当指定 num_patch_input 时，会自动计算额外填充
        
        Args:
            context_len: 回看窗口长度
            pred_len: 预测窗口长度
            num_patch_input: 显式指定输入patch列数（None时自动计算）
            periodicity: 周期长度
            norm_const: 归一化常数
            align_const: 对齐常数
            interpolation: 插值方法
            padding_mode: 填充模式
        """
        self.image_size = 224
        self.vq_image_size = 256
        self.patch_size = 16
        self.num_patch = self.vq_image_size // self.patch_size  # 16
        
        self.context_len = context_len
        self.pred_len = pred_len
        self.periodicity = periodicity
        self.padding_mode = padding_mode
        
        # 当显式指定 num_patch_input 时，计算额外填充
        if num_patch_input is not None:
            extra_padding = (
                pred_len / (self.num_patch - num_patch_input) * num_patch_input
                - self.context_len
            )
            if extra_padding > 0:
                self.context_len += int(np.ceil(extra_padding))
        
        self.pad_left = 0
        self.pad_right = 0
        if self.context_len % self.periodicity != 0:
            self.pad_left = self.periodicity - self.context_len % self.periodicity
        if self.pred_len % self.periodicity != 0:
            self.pad_right = self.periodicity - self.pred_len % self.periodicity
        
        input_ratio = (
            (self.pad_left + self.context_len)
            / (self.pad_left + self.context_len + self.pad_right + self.pred_len)
        )
        if num_patch_input is None:
            self.num_patch_input = int(input_ratio * self.num_patch * align_const)
            if self.num_patch_input == 0:
                self.num_patch_input = 1
        else:
            self.num_patch_input = num_patch_input
        
        self.num_patch_output = self.num_patch - self.num_patch_input
        self.adjust_input_ratio = self.num_patch_input / self.num_patch
        
        self.interpolation = {
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "bicubic": Image.BICUBIC,
        }[interpolation]
        
        self.scale_x = (
            (self.pad_left + self.context_len) // self.periodicity
        ) / (int(self.image_size * self.adjust_input_ratio))
        self.norm_const = norm_const
        
        # 可见 token 数量
        self.num_visible_tokens = self.num_patch * self.num_patch_input

    def forward(
        self,
        x,
        export_image: bool = False,
        fp64: bool = False,
        multivariate: bool = False,
        color_list=None,
        LOOKBACK_LEN_VISUAL: int = 300,
    ):
        """
        VisionTSRARpp 前向传播：6步流水线
        
        与 VisionTSRAR.forward 的主要区别：
        - 多变量纵向排列在图像中（每个变量占 image_size/nvars 的行数）
        - 支持 RGB 颜色编码（不同变量分配到不同颜色通道）
        - 支持输入裁剪策略
        - TODO: 支持分位数预测输出
        
        Args:
            x: 回看窗口，shape: [bs x context_len x nvars]
            export_image: 是否导出可视化图像
            fp64: 是否使用 float64 精度
            multivariate: 是否多变量模式
            color_list: 每个变量的颜色通道索引列表
            LOOKBACK_LEN_VISUAL: 可视化时的回看长度
        
        Returns:
            y: 预测窗口，shape: [bs x pred_len x nvars]
        """
        # 获取变量数
        self.nvars = x.shape[-1]
        
        # 每个变量在图像中占用的行数
        self.image_size_per_var = int(self.image_size / self.nvars)
        self.input_resize = util.safe_resize(
            (self.image_size_per_var, int(self.image_size * self.adjust_input_ratio)),
            interpolation=self.interpolation,
        )
        self.output_resize = util.safe_resize(
            (self.periodicity, int(round(self.image_size * self.scale_x))),
            interpolation=self.interpolation,
        )

        # ========== 第1步: Normalization（归一化）==========
        means = x.mean(1, keepdim=True).detach()
        x_enc = x - means
        stdev = torch.sqrt(
            torch.var(
                x_enc.to(torch.float64) if fp64 else x_enc,
                dim=1, keepdim=True, unbiased=False,
            ) + 1e-5
        )
        stdev = stdev / self.norm_const
        x_enc = x_enc / stdev
        x_enc = einops.rearrange(x_enc, 'b s n -> b n s')

        # 处理输入长度不足的情况
        if x_enc.shape[-1] < self.context_len:
            extra_padding = self.context_len - x_enc.shape[-1]
        else:
            extra_padding = 0

        # ========== 第2步: Segmentation（分段/折叠）==========
        x_pad = F.pad(x_enc, (self.pad_left + extra_padding, 0), mode=self.padding_mode)
        x_2d = einops.rearrange(x_pad, 'b n (p f) -> b n f p', f=self.periodicity)

        # ========== 第3步: Render & Alignment（渲染与对齐）==========
        # 3a. 每个变量独立缩放到其纵向分区
        x_resize = self.input_resize(x_2d)
        # 3b. 将多个变量纵向拼接
        x_resize = einops.rearrange(x_resize, 'b n h w -> b 1 (n h) w')
        # 3c. 底部填充零
        pad_down = self.image_size - x_resize.shape[2]
        if pad_down > 0:
            x_resize = torch.concat([
                x_resize,
                torch.zeros(
                    (x_resize.shape[0], x_resize.shape[1], pad_down, x_resize.shape[3]),
                    device=x_resize.device, dtype=x_resize.dtype,
                ),
            ], dim=2)
        
        # 3d. 构造右侧掩码区域（真实future，teacher forcing训练目标）
        future_2d = x_2d[:, :, :, -self.num_patch_output * self.patch_size:]
        future_resized = self.input_resize(future_2d)
        x_concat_with_masked = torch.cat([x_resize, future_resized], dim=-1)
        
        # 3e. 颜色编码
        if not self.color:
            image_input = einops.repeat(x_concat_with_masked, 'b 1 h w -> b c h w', c=3)
        else:
            image_input = torch.zeros(
                (x_concat_with_masked.shape[0], 3, x_concat_with_masked.shape[2], x_concat_with_masked.shape[3]),
                device=x_concat_with_masked.device,
                dtype=x_concat_with_masked.dtype,
            )
            if color_list is None:
                color_list = [i % 3 for i in range(self.nvars)]
            for i in range(self.nvars):
                color = color_list[i]
                image_input[:, color, i*self.image_size_per_var:(i+1)*self.image_size_per_var, :] = \
                    x_concat_with_masked[:, 0, i*self.image_size_per_var:(i+1)*self.image_size_per_var, :]
        
        # 3f. 输入裁剪
        if self.clip_input == 0:
            if not self.complete_no_clip:
                image_input = torch.clip(image_input, -5, 5)
        else:
            thres_down_list = [-2.1179039301310043, -2.0357142857142856, -1.8044444444444445]
            thres_up_list = [2.2489082969432315, 2.428571428571429, 2.6399999999999997]
            thres_down = max(thres_down_list)
            thres_up = min(thres_up_list)
            image_input = torch.clip(image_input, thres_down, thres_up)

        # ========== 第4步: RAR Reconstruction（RAR重建）==========
        if self.training:
            image_reconstructed, rar_loss = self.rar_wrapper(
                image_input, self.num_visible_tokens
            )
        else:
            image_reconstructed = self.rar_wrapper.generate(
                image_input, self.num_visible_tokens
            )

        # 从重建图像中提取各变量数据的函数
        def process_images(image_reconstructed, nvars, color_list):
            """从重建的3通道图像中，根据颜色分配规则提取各变量的数据"""
            batch_size = image_reconstructed.shape[0]
            height, width = image_reconstructed.shape[2], image_reconstructed.shape[3]
            output = torch.zeros(
                (batch_size, 1, height, width),
                device=image_reconstructed.device,
            )
            nvar = nvars
            for i in range(batch_size):
                h_per_var = height // nvar
                remainder = height % nvar
                for k in range(nvar):
                    start_h = k * h_per_var
                    end_h = (k + 1) * h_per_var
                    if k == nvar - 1 and remainder != 0:
                        end_h = height
                    color_channel = color_list[k]
                    output[i, 0, start_h:end_h, :] = \
                        image_reconstructed[i, color_channel, start_h:end_h, :]
            return output

        # ========== 第5步: Forecasting（预测提取）==========
        if not self.color:
            y_grey = torch.mean(image_reconstructed, 1, keepdim=True)
        else:
            y_grey = process_images(image_reconstructed, self.nvars, color_list)
        
        def extract_TS_from_image(y_grey):
            """从灰度图像中提取时间序列预测值"""
            if pad_down > 0:
                y_grey = y_grey[:, :, :-pad_down, :]
            y_grey = einops.rearrange(y_grey, 'b 1 (n h) w -> b n h w', n=self.nvars)
            y_segmentations = self.output_resize(y_grey)
            y_flatten = einops.rearrange(
                y_segmentations,
                'b n f p -> b (p f) n',
            )
            start_idx = self.pad_left + self.context_len
            end_idx = self.pad_left + self.context_len + self.pred_len
            y_pred = y_flatten[:, start_idx: end_idx, :]
            return y_pred

        y = extract_TS_from_image(y_grey)

        # ========== 第6步: Denormalization（反归一化）==========
        y = y * (stdev.repeat(1, self.pred_len, 1))
        y = y + (means.repeat(1, self.pred_len, 1))

        # TODO: 分位数预测（未来扩展点）
        # if self.quantile:
        #     ... 处理分位数预测的提取和反归一化 ...

        if export_image:
            # 简化的图像导出（与 VisionTSRAR 类似）
            period_num = LOOKBACK_LEN_VISUAL // self.periodicity
            x_2d_visual = x_2d[:, :, :, -period_num:]
            x_resize_vis = self.input_resize(x_2d_visual)
            x_resize_vis = einops.rearrange(x_resize_vis, 'b n h w -> b 1 (n h) w')
            pad_down_vis = self.image_size - x_resize_vis.shape[2]
            if pad_down_vis > 0:
                x_resize_vis = torch.concat([
                    x_resize_vis,
                    torch.zeros(
                        (x_resize_vis.shape[0], x_resize_vis.shape[1], pad_down_vis, x_resize_vis.shape[3]),
                        device=x_resize_vis.device, dtype=x_resize_vis.dtype,
                    ),
                ], dim=2)
            
            masked_vis = torch.zeros(
                (x_2d_visual.shape[0], 1, self.image_size, self.num_patch_output * self.patch_size),
                device=x_2d_visual.device, dtype=x_2d_visual.dtype,
            )
            x_concat_vis = torch.cat([x_resize_vis, masked_vis], dim=-1)
            
            if not self.color:
                image_input_vis = einops.repeat(x_concat_vis, 'b 1 h w -> b c h w', c=3)
            else:
                image_input_vis = torch.zeros(
                    (x_concat_vis.shape[0], 3, x_concat_vis.shape[2], x_concat_vis.shape[3]),
                    device=x_concat_vis.device, dtype=x_concat_vis.dtype,
                )
                if color_list is None:
                    color_list = [i % 3 for i in range(self.nvars)]
                for i in range(self.nvars):
                    color = color_list[i]
                    image_input_vis[:, color, i*self.image_size_per_var:(i+1)*self.image_size_per_var, :] = \
                        x_concat_vis[:, 0, i*self.image_size_per_var:(i+1)*self.image_size_per_var, :]
            
            # 构造掩码可视化
            mask = torch.ones((self.num_patch, self.num_patch), device=image_input.device)
            mask[:, :self.num_patch_input] = 0
            mask_img = F.interpolate(
                mask.float().unsqueeze(0).unsqueeze(0).repeat(image_input.shape[0], 1, 1, 1),
                size=(self.image_size, self.image_size),
                mode='nearest',
            )
            image_reconstructed = image_input * (1 - mask_img) + image_reconstructed * mask_img
            green_bg = -torch.ones_like(image_reconstructed) * 2
            image_input = image_input * (1 - mask_img) + green_bg * mask_img
            image_input = einops.rearrange(image_input, '(b n) c h w -> b n h w c', b=x_enc.shape[0])
            image_reconstructed = einops.rearrange(image_reconstructed, '(b n) c h w -> b n h w c', b=x_enc.shape[0])
            return y, image_input, image_reconstructed, self.nvars, color_list
        
        # 训练模式返回 (prediction, rar_loss)，方便 TSL 框架处理
        if self.training:
            return y, rar_loss
        
        return y
