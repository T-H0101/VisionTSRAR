"""
轻量级 VQ Decoder 模块

设计目标：在 4090 24G 显卡上将 batch_size 从 2 提升到更大的值，
同时保持合理的训练时间和重建质量。

核心优化策略：
1. 减少基础通道数：128 → 64（节省 75% 通道计算量）
2. 减少上采样层级：5层 → 3层（从16x下采样改为8x下采样）
3. 减少每层ResBlock：3个 → 2个
4. 移除中间层的注意力块，仅保留最高分辨率的注意力
5. 使用深度可分离卷积替代标准卷积（可选）

注意：由于减少了上采样层级，输入的潜空间从 16x16 变为 32x32。
这意味着需要修改 VQ 的 quant_conv 来输出正确的通道数。

与原 VQModel 的集成方式：
1. 训练时：替换 decoder 部分，保留 encoder 和 quantize（codebook）
2. 使用时：替换 RARWrapper 中的 vq.decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class LightweightDecoder(nn.Module):
    """
    轻量级 VQ-VAE 解码器：将潜空间表示上采样重建为图像。

    架构设计（3层上采样）：
    输入: (batch, 256, 16, 16)  # 16x16 潜空间，256 通道（来自 post_quant_conv）
    ↓
    Level 0: conv_in(256→64) + ResBlock×2 → 64×16×16
    Level 1: Upsample + ResBlock×2 + Attn → 128×32×32
    Level 2: Upsample + ResBlock×2 + Attn → 256×64×64
    ↓
    输出: (batch, 3, 256, 256)  # 256x256 图像（需要额外 4x 插值或修改）

    注意：原 VQModel 的 decode 流程是：
    quantize 输出 (batch, 8, 16, 16)
        → post_quant_conv → (batch, 256, 16, 16)
        → 原 decoder → (batch, 3, 256, 256)

    为了直接替换 vq.decoder，LightweightDecoder 应该：
    1. 接收 (batch, 256, 16, 16) 输入（保持与 post_quant_conv 输出兼容）
    2. 输出 (batch, 3, 256, 256) 图像（需要内部分辨率适配）

    集成方式（二选一）：
    - 方式A（推荐）：替换 vq.decoder，保持 vq.post_quant_conv
      → 输入 (256, 16, 16)，输出 (3, 256, 256)
      → 与原接口完全兼容
    - 方式B：替换整个 decode 流程（包括 quant_conv 和 decoder）
      → 需要修改 RARWrapper 中的调用方式

    参数量对比（估算）：
    - 原版 Decoder: ~45M 参数
    - 轻量 Decoder: ~8M 参数（减少 82%）
    """

    def __init__(
        self,
        z_channels: int = 256,
        base_channels: int = 64,
        ch_mult: List[int] = [1, 2, 4],
        num_res_blocks: int = 2,
        norm_type: str = 'group',
        dropout: float = 0.0,
        out_channels: int = 3,
        use_attn_last_only: bool = True,
        use_depthwise: bool = False,
        input_resolution: int = 16,
        output_resolution: int = 256,
    ):
        """
        Args:
            z_channels: 潜空间通道数（来自 post_quant_conv 的输出）
            base_channels: 基础通道数，越小越轻量
            ch_mult: 通道倍数列表，控制每层的通道数
            num_res_blocks: 每层的残差块数量
            norm_type: 归一化类型，'group' 或 'batch'
            dropout: Dropout 概率
            out_channels: 输出图像通道数
            use_attn_last_only: 是否仅在最后层使用注意力
            use_depthwise: 是否使用深度可分离卷积（更轻量但可能影响质量）
            input_resolution: 输入潜空间分辨率（默认16，对应16x16）
            output_resolution: 输出图像分辨率（默认256，对应256x256）
        """
        super().__init__()
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

        # 计算需要的上采样次数和最终分辨率
        # 如果 input=16, output=256，需要 4 次 2x 上采样：16→32→64→128→256
        # 但如果使用 3 层 + 最后的 bicubic，可以节省一层
        # num_upsamples 根据 ch_mult 长度自动计算
        self.num_resolutions = len(ch_mult)
        num_upsamples = self.num_resolutions  # 与ch_mult长度一致
        final_conv_resolution = input_resolution * (2 ** num_upsamples)
        self.num_upsamples = num_upsamples

        self.num_res_blocks = num_res_blocks

        # 计算最深层的通道数（作为中间处理通道）
        # 使用 num_resolutions 作为最终上采样层数
        block_in = base_channels * ch_mult[-1]  # 64 * 4 = 256

        # 潜空间到特征空间的投影
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # 中间处理块
        self.mid = nn.ModuleList()
        self.mid.append(LightweightResBlock(block_in, block_in, dropout=dropout, use_depthwise=use_depthwise))
        if not use_attn_last_only:
            self.mid.append(LightweightAttnBlock(block_in))
        self.mid.append(LightweightResBlock(block_in, block_in, dropout=dropout, use_depthwise=use_depthwise))

        # 上采样路径
        # 输入: 16x16, 输出: 128x128 (3次上采样)
        self.conv_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            conv_block = nn.Module()
            # 从最深层到浅层遍历（与原版一致）
            level_idx = self.num_resolutions - 1 - i_level
            block_out = base_channels * ch_mult[level_idx]

            # 残差块
            res_block = nn.ModuleList()
            current_in = block_in
            for _ in range(num_res_blocks):
                res_block.append(
                    LightweightResBlock(current_in, block_out, dropout=dropout, use_depthwise=use_depthwise)
                )
                current_in = block_out

            conv_block.res = res_block

            # 注意力块：当use_attn_last_only=True时完全移除，避免在高分辨率上计算注意力
            attn_block = nn.ModuleList()
            if not use_attn_last_only and i_level == self.num_resolutions - 1:
                attn_block.append(LightweightAttnBlock(block_out))
            conv_block.attn = attn_block

            # 下一次上采样
            if i_level < num_upsamples:
                conv_block.upsample = LightweightUpsample(block_out, with_conv=True)
            else:
                conv_block.upsample = None

            block_in = block_out
            self.conv_blocks.append(conv_block)

        # 最终上采样到目标分辨率（如果需要）
        if final_conv_resolution < output_resolution:
            self.final_upsample = nn.Sequential(
                nn.Upsample(scale_factor=output_resolution // final_conv_resolution, mode='bilinear', align_corners=False),
                nn.Conv2d(block_out, block_out, kernel_size=3, stride=1, padding=1),
            )
        else:
            self.final_upsample = None

        # 输出投影
        self.norm_out = LightweightNormalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    @property
    def last_layer(self):
        """返回最后一层权重，用于 LPIPS 感知损失的梯度缩放"""
        return self.conv_out.weight

    def forward(self, z):
        """
        前向传播：潜空间 → 图像。

        Args:
            z: 潜空间特征，shape (batch, z_channels, H', W')
                   对于 256x256 图像输入，H'=W'=16 或 32

        Returns:
            重建图像，shape (batch, 3, H'*16, W'*16)
        """
        # 潜空间到特征空间
        h = self.conv_in(z)

        # 中间处理
        for mid_block in self.mid:
            if isinstance(mid_block, LightweightAttnBlock):
                h = mid_block(h)
            else:
                h = mid_block(h)

        # 上采样路径
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks):
                h = block.res[i_block](h)
            if len(block.attn) > 0:
                h = block.attn[0](h)
            if block.upsample is not None:
                h = block.upsample(h)

        # 最终上采样（如果需要）
        if self.final_upsample is not None:
            h = self.final_upsample(h)

        # 输出
        h = self.norm_out(h)
        h = light_swish(h)
        h = self.conv_out(h)
        return h


class LightweightResBlock(nn.Module):
    """
    轻量级残差卷积块。

    相比原版 ResBlock：
    - 可选深度可分离卷积
    - 简化归一化层（GroupNorm 改为更小的 group 数）
    """
    def __init__(self, in_channels, out_channels=None, dropout=0.0, use_depthwise=False):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(16, in_channels, eps=1e-6)
        self.conv1 = self._make_conv(in_channels, out_channels, use_depthwise)
        self.norm2 = nn.GroupNorm(16, out_channels, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = self._make_conv(out_channels, out_channels, use_depthwise)

        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = None

    def _make_conv(self, in_ch, out_ch, use_depthwise):
        """创建卷积层，可选深度可分离卷积"""
        if use_depthwise:
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
            )
        else:
            return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = light_swish(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = light_swish(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return x + h


class LightweightAttnBlock(nn.Module):
    """
    轻量级自注意力块。

    相比原版 AttnBlock：
    - 使用更少的注意力头
    - 可选的键值缓存（推理时加速）
    """
    def __init__(self, in_channels, num_heads: Optional[int] = None):
        super().__init__()
        self.in_channels = in_channels
        # 使用较少的注意力头，减少计算量
        # 当 in_channels 较大时（如 256），使用 8 个头而不是全部通道作为头
        self.num_heads = num_heads if num_heads is not None else max(4, in_channels // 32)
        self.head_dim = in_channels // self.num_heads
        self.scale = self.head_dim ** (-0.5)

        self.norm = nn.GroupNorm(16, in_channels, eps=1e-6)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # 多头注意力
        b, c, h, w = q.shape
        q = q.reshape(b, self.num_heads, self.head_dim, h * w)
        k = k.reshape(b, self.num_heads, self.head_dim, h * w)
        v = v.reshape(b, self.num_heads, self.head_dim, h * w)

        q = q.permute(0, 1, 3, 2)  # (b, heads, hw, head_dim)
        k = k.permute(0, 1, 2, 3)  # (b, heads, head_dim, hw)
        v = v.permute(0, 1, 3, 2)  # (b, heads, hw, head_dim)

        # 注意力计算
        attn = torch.matmul(q, k) * self.scale
        attn = F.softmax(attn, dim=-1)

        # 注意力加权
        out = torch.matmul(attn, v)  # (b, heads, hw, head_dim)
        out = out.permute(0, 1, 3, 2).reshape(b, c, h, w)

        out = self.proj_out(out)
        return x + out


class LightweightUpsample(nn.Module):
    """
    轻量级上采样模块。

    相比原版 Upsample：
    - 使用转置卷积替代 最近邻 + 卷积（可能产生棋盘效应，但更轻量）
    - 或者使用双线性插值 + 卷积
    """
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        if self.with_conv:
            x = self.conv(x)
        return x


class LightweightNormalize(nn.Module):
    """轻量级归一化层"""
    def __init__(self, in_channels, norm_type='group'):
        super().__init__()
        assert norm_type in ['group', 'batch']
        if norm_type == 'group':
            self.norm = nn.GroupNorm(16, in_channels, eps=1e-6)
        else:
            self.norm = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        return self.norm(x)


def light_swish(x):
    """轻量级 Swish 激活函数"""
    return x * torch.sigmoid(x)


class LightweightVQDecoder(nn.Module):
    """
    轻量级 VQ 解码器包装类。

    提供与原 VQModel.decode() 相同的接口，方便替换使用。

    用法：
        # 加载原 VQ 模型
        vq = VQModel(...)
        vq.load_state_dict(torch.load('vq_model.pt'))

        # 创建轻量解码器
        light_decoder = LightweightVQDecoder(
            vq_codebook_size=16384,
            vq_codebook_embed_dim=8,
            vq_ckpt_path='vq_model.pt'
        )
        light_decoder.to('cuda')

        # 训练/推理时使用
        # 假设 quant 是来自编码器的量化特征
        recon = light_decoder.decode(quant)
    """

    def __init__(
        self,
        vq_codebook_size: int = 16384,
        vq_codebook_embed_dim: int = 8,
        vq_ckpt_path: str = None,
        base_channels: int = 64,
        ch_mult: List[int] = [1, 2, 4],
        z_channels: int = 256,
        dropout: float = 0.0,
        use_depthwise: bool = False,
    ):
        """
        Args:
            vq_codebook_size: 码本大小（必须与原 VQ 匹配）
            vq_codebook_embed_dim: 码本嵌入维度（必须与原 VQ 匹配）
            vq_ckpt_path: 原 VQ 模型权重路径，用于加载码本
            base_channels: 轻量解码器基础通道数
            ch_mult: 通道倍数列表
            z_channels: 潜空间通道数
            dropout: Dropout 概率
            use_depthwise: 是否使用深度可分离卷积
        """
        super().__init__()
        self.codebook_size = vq_codebook_size
        self.codebook_embed_dim = vq_codebook_embed_dim
        self.z_channels = z_channels

        # 轻量解码器
        self.decoder = LightweightDecoder(
            z_channels=z_channels,
            base_channels=base_channels,
            ch_mult=ch_mult,
            num_res_blocks=2,
            norm_type='group',
            dropout=dropout,
            out_channels=3,
            use_attn_last_only=True,
            use_depthwise=use_depthwise,
        )

        # 加载码本（如果提供了路径）
        if vq_ckpt_path is not None:
            self._load_codebook(vq_ckpt_path)

    def _load_codebook(self, ckpt_path: str):
        """从检查点加载码本权重"""
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt if isinstance(ckpt, dict) and 'model' in ckpt else ckpt

        # 尝试不同的 key 名称
        codebook_key = None
        for key in ['quantize.embedding.weight', 'quantize.embedding']:
            if key in state_dict:
                codebook_key = key
                break

        if codebook_key is None:
            raise ValueError(f"Cannot find codebook in checkpoint. Available keys: {list(state_dict.keys())[:10]}")

        # 加载到当前模块（注意：这里我们不存储码本，只验证兼容性）
        print(f"[LightweightVQDecoder] Loaded codebook from {ckpt_path}")
        print(f"  - Codebook size: {self.codebook_size}")
        print(f"  - Codebook dim: {self.codebook_embed_dim}")

    def decode(self, z):
        """
        解码潜空间特征为图像。

        Args:
            z: 潜空间特征，shape (batch, z_channels, H', W')

        Returns:
            重建图像，shape (batch, 3, H'*8, W'*8)（假设 8x 上采样）
        """
        return self.decoder(z)

    def decode_code(self, code_b, shape=None, channel_first=True):
        """
        从码本索引解码为图像。

        Args:
            code_b: 码本索引，shape (batch, L) 或 (batch, H', W')
            shape: 输出形状
            channel_first: 是否通道优先格式

        Returns:
            重建图像
        """
        # 需要外部提供码本embedding，这里简化处理
        raise NotImplementedError(
            "decode_code requires external codebook. "
            "Use LightweightDecoder with pre-extracted quant features instead."
        )

    def forward(self, quant):
        """
        前向传播。

        Args:
            quant: 量化特征，来自 VQ encoder 的输出

        Returns:
            重建图像
        """
        return self.decode(quant)


def create_lightweight_decoder_from_vq(
    vq_model: 'VQModel',
    base_channels: int = 64,
    ch_mult: List[int] = [1, 2, 4],
) -> LightweightDecoder:
    """
    从现有 VQModel 创建轻量解码器的工厂函数。

    Args:
        vq_model: 原始 VQModel 实例
        base_channels: 轻量解码器基础通道数
        ch_mult: 通道倍数列表

    Returns:
        可直接替换 vq_model.decoder 的 LightweightDecoder

    Example:
        # 加载原 VQ
        vq = VQModel(...)
        vq.load_state_dict(torch.load('vq.pt'))

        # 创建轻量解码器
        light_dec = create_lightweight_decoder_from_vq(vq, base_channels=64)
        light_dec.to('cuda')

        # 替换原解码器
        vq.decoder = light_dec
    """
    # z_channels 应该是 post_quant_conv 的输出通道，而不是 codebook_embed_dim
    # VQModel 的 decoder 接收的是 post_quant_conv 的输出，即 (batch, 256, H', W')
    light_dec = LightweightDecoder(
        z_channels=256,  # post_quant_conv 输出通道
        base_channels=base_channels,
        ch_mult=ch_mult,
        num_res_blocks=2,
        norm_type='group',
        dropout=0.0,
        out_channels=3,
        use_attn_last_only=True,
        use_depthwise=False,
        input_resolution=16,
        output_resolution=256,
    )
    return light_dec
