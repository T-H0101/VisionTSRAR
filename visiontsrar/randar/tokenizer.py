"""
RandAR VQ 变分自编码器（VQ-VAE）模块

本模块实现了将图像编码为离散 token 序列、以及将 token 序列解码回图像的 VQ-VAE 模型。
这是 RandAR 图像生成流程的基础：先训练 VQ-VAE 将连续图像离散化，
再用自回归 Transformer 生成离散 token 序列，最后用 VQ-VAE 解码器重建图像。

核心组件：
- VQModel: 完整的 VQ-VAE 模型，包含编码器、量化器、解码器
- Encoder: 卷积编码器，将图像压缩为潜空间特征
- Decoder: 卷积解码器，将潜空间特征重建为图像
- VectorQuantizer: 向量量化器，将连续特征映射到离散码本

新增接口（为 VisionTSRAR 设计）：
- encode_to_tokens(x): 图像 → token 索引的简化接口
- decode_tokens_to_image(tokens, image_size): token 索引 → 图像的简化接口

来源:
- taming-transformers: https://github.com/CompVis/taming-transformers
- maskgit: https://github.com/google-research/maskgit
- LlamaGen: https://github.com/FoundationVision/LlamaGen/blob/main/tokenizer/tokenizer_image/vq_model.py
"""

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    """
    VQ-VAE 模型配置参数。
    
    注意：此类与 LlamaGen 的 ModelArgs 不同，专用于 VQ-VAE 配置。
    """
    codebook_size: int = 16384         # 码本大小（即词汇表大小）
    codebook_embed_dim: int = 8        # 码本嵌入维度
    codebook_l2_norm: bool = True      # 是否对码本向量做 L2 归一化
    codebook_show_usage: bool = True   # 是否追踪码本使用率
    commit_loss_beta: float = 0.25     # Commitment loss 权重
    entropy_loss_ratio: float = 0.0    # 熵正则化损失权重
    
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])  # 编码器通道倍数
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])  # 解码器通道倍数
    z_channels: int = 256              # 潜空间通道数
    dropout_p: float = 0.0             # Dropout 概率


class VQModel(nn.Module):
    """
    VQ 变分自编码器（VQ-VAE）：将图像编码为离散 token，再解码重建图像。
    
    整体流程：
    1. Encoder 将输入图像编码为连续潜空间特征 h
    2. quant_conv 将 h 投影到码本嵌入维度
    3. VectorQuantizer 将连续特征量化为离散码本索引
    4. post_quant_conv 将量化特征投影回潜空间维度
    5. Decoder 将潜空间特征解码为重建图像
    
    训练时的损失 = 重建损失 + VQ损失（commitment loss + codebook loss + entropy loss）
    推理时只使用编码-量化-解码的前向路径。
    """
    def __init__(self,
                 codebook_size=16384,
                 codebook_embed_dim=8,
                 codebook_l2_norm=True,
                 codebook_show_usage=True,
                 commit_loss_beta=0.25,
                 entropy_loss_ratio=0.0,
                 encoder_ch_mult=[1, 1, 2, 2, 4],
                 decoder_ch_mult=[1, 1, 2, 2, 4],
                 z_channels=256,
                 dropout_p=0.0):
        super().__init__()
        self.encoder = Encoder(ch_mult=encoder_ch_mult, z_channels=z_channels, dropout=dropout_p)
        self.decoder = Decoder(ch_mult=decoder_ch_mult, z_channels=z_channels, dropout=dropout_p)

        self.quantize = VectorQuantizer(codebook_size, codebook_embed_dim, 
                                        commit_loss_beta, entropy_loss_ratio,
                                        codebook_l2_norm, codebook_show_usage)
        # 量化前后的维度转换卷积
        self.quant_conv = nn.Conv2d(z_channels, codebook_embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(codebook_embed_dim, z_channels, 1)
        self.codebook_embed_dim = codebook_embed_dim

    def encode(self, x):
        """
        编码图像为量化潜空间表示。
        
        Args:
            x: 输入图像，shape (batch, 3, H, W)
        
        Returns:
            quant: 量化后的潜空间特征，shape (batch, codebook_embed_dim, H/16, W/16)
            emb_loss: VQ 损失元组 (vq_loss, commit_loss, entropy_loss, codebook_usage)
            info: 附加信息 (perplexity, min_encodings, min_encoding_indices)
        """
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info
    
    def encode_indices(self, x):
        """
        编码图像为离散 token 索引。
        
        Args:
            x: 输入图像，shape (batch, 3, H, W)
        
        Returns:
            token 索引，shape (batch, H/16, W/16)，每个元素是码本中的索引
        """
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return info[2]  # min_encoding_indices

    def decode(self, quant):
        """
        将量化潜空间特征解码为图像。
        
        Args:
            quant: 量化特征，shape (batch, codebook_embed_dim, H', W')
        
        Returns:
            重建图像，shape (batch, 3, H'*16, W'*16)
        """
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b, shape=None, channel_first=True):
        """
        从码本索引解码为图像。
        
        Args:
            code_b: 码本索引，shape (batch, L) 或 (batch, H', W')
            shape: 输出形状 (batch, channel, height, width)
            channel_first: 是否通道优先格式
        
        Returns:
            重建图像特征（未归一化到 [0, 255]）
        """
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        """
        前向传播：编码 → 量化 → 解码。
        
        Args:
            input: 输入图像
        
        Returns:
            dec: 重建图像
            diff: VQ 损失
        """
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff
    
    def decode_codes_to_img(self, codes, tgt_size):
        """
        将码本索引解码为像素图像（已归一化到 [0, 255] uint8）。
        
        Args:
            codes: 码本索引，shape (batch, L)，L = H' * W'
            tgt_size: 目标图像尺寸（正方形边长）
        
        Returns:
            numpy 图像数组，shape (batch, tgt_size, tgt_size, 3)，dtype uint8
        """
        qz_shape = (
            codes.shape[0],
            self.codebook_embed_dim,
            int(codes.shape[1] ** 0.5),
            int(codes.shape[1] ** 0.5)
        )
        results = self.decode_code(codes, qz_shape)
        if results.shape[-1] != tgt_size:
            results = F.interpolate(results, size=(tgt_size, tgt_size), mode="bicubic")
        # 从 [-1, 1] 归一化空间反变换到 [0, 255] 像素空间
        imgs = results.detach() * 127.5 + 128
        imgs = torch.clamp(imgs, 0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        return imgs

    def encode_to_tokens(self, x):
        """
        【VisionTSRAR 专用接口】将图像编码为 1D token 索引序列。
        
        与 encode_indices 不同，此方法返回展平的 1D 序列而非 2D 空间索引，
        直接与 RandAR Transformer 的输入格式兼容。
        
        在 VisionTSRAR 中，时间序列被渲染为图像后通过此方法编码为 token 序列，
        然后交给 RandAR 进行自回归预测。
        
        Args:
            x: 输入图像，shape (batch, 3, H, W)
        
        Returns:
            token 索引，shape (batch, L)，其中 L = (H/16) * (W/16)
            例如 256x256 图像 → L = 16*16 = 256 个 token
        """
        indices = self.encode_indices(x)  # 返回的是展平的 1D tensor: (batch*H/16*W/16,)
        # 计算每个图像的 token 数量
        spatial_tokens = x.shape[2] // 16 * x.shape[3] // 16  # H/16 * W/16
        # 重新 reshape 为 (batch, H/16*W/16)
        tokens = indices.view(x.shape[0], spatial_tokens)  # (batch, L)
        return tokens

    def decode_tokens_to_image(self, tokens, image_size):
        """
        【VisionTSRAR 专用接口】将 1D token 索引序列解码为图像。
        
        在 VisionTSRAR 中，RandAR Transformer 生成 token 序列后，
        通过此方法将 token 解码回图像，再从图像中提取预测的时间序列值。
        
        Args:
            tokens: token 索引序列，shape (batch, L)
            image_size: 输出图像尺寸（正方形边长，如 256）
        
        Returns:
            重建图像，shape (batch, 3, image_size, image_size)，值域 [-1, 1]
        """
        # 计算潜空间空间尺寸
        spatial_size = int(tokens.shape[1] ** 0.5)
        # 将 1D 序列重塑为 2D 空间形状
        codes = tokens.view(tokens.shape[0], spatial_size, spatial_size)
        # 从码本索引获取量化向量
        qz_shape = (
            tokens.shape[0],
            self.codebook_embed_dim,
            spatial_size,
            spatial_size
        )
        quant = self.quantize.get_codebook_entry(codes, qz_shape, channel_first=True)
        # 解码为图像
        dec = self.decode(quant)
        # 如果尺寸不匹配，双三次插值调整
        if dec.shape[-1] != image_size:
            dec = F.interpolate(dec, size=(image_size, image_size), mode="bicubic")
        return dec


class Encoder(nn.Module):
    """
    VQ-VAE 编码器：将输入图像逐步下采样为紧凑的潜空间表示。
    
    结构：Conv-In → [ResBlock × 2 + Attn + Downsample] × 5 → MidBlock → Conv-Out
    
    每个分辨率级别包含：
    - 2个 ResnetBlock（残差卷积块）
    - 最高分辨率级别额外添加 AttnBlock（自注意力块）
    - 除最后一个级别外，使用 Downsample 下采样 2x
    
    总下采样倍率：2^4 = 16（5个级别，4次下采样）
    即 256x256 输入 → 16x16 潜空间
    """
    def __init__(self, in_channels=3, ch=128, ch_mult=(1,1,2,2,4), num_res_blocks=2, 
                 norm_type='group', dropout=0.0, resamp_with_conv=True, z_channels=256):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        # 下采样路径
        in_ch_mult = (1,) + tuple(ch_mult)
        self.conv_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            conv_block = nn.Module()
            # 残差块和注意力块
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                res_block.append(ResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
                # 仅在最高分辨率级别添加注意力（计算量最大，信息最丰富）
                if i_level == self.num_resolutions - 1:
                    attn_block.append(AttnBlock(block_in, norm_type))
            conv_block.res = res_block
            conv_block.attn = attn_block
            # 下采样层（最后一个级别不需要）
            if i_level != self.num_resolutions-1:
                conv_block.downsample = Downsample(block_in, resamp_with_conv)
            self.conv_blocks.append(conv_block)

        # 中间块（最深层的额外处理）
        self.mid = nn.ModuleList()
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        self.mid.append(AttnBlock(block_in, norm_type))
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))

        # 输出投影
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        h = self.conv_in(x)
        # 下采样路径
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.downsample(h)
        
        # 中间处理
        for mid_block in self.mid:
            h = mid_block(h)
        
        # 输出
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h



class Decoder(nn.Module):
    """
    VQ-VAE 解码器：将潜空间表示逐步上采样重建为图像。
    
    结构：Conv-In → MidBlock → [ResBlock × 3 + Attn + Upsample] × 5 → Conv-Out
    
    与编码器对称，每个分辨率级别包含：
    - 3个 ResnetBlock（比编码器多1个，提升重建质量）
    - 最高分辨率级别额外添加 AttnBlock
    - 除第一个级别外，使用 Upsample 上采样 2x
    
    总上采样倍率：2^4 = 16
    """
    def __init__(self, z_channels=256, ch=128, ch_mult=(1,1,2,2,4), num_res_blocks=2, norm_type="group",
                 dropout=0.0, resamp_with_conv=True, out_channels=3):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        block_in = ch*ch_mult[self.num_resolutions-1]
        # 潜空间到特征空间的投影
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # 中间块
        self.mid = nn.ModuleList()
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        self.mid.append(AttnBlock(block_in, norm_type))
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))

        # 上采样路径（从高分辨率到低分辨率，即反向遍历 ch_mult）
        self.conv_blocks = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            conv_block = nn.Module()
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):  # 解码器多1个残差块
                res_block.append(ResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn_block.append(AttnBlock(block_in, norm_type))
            conv_block.res = res_block
            conv_block.attn = attn_block
            # 上采样层（第一个级别不需要）
            if i_level != 0:
                conv_block.upsample = Upsample(block_in, resamp_with_conv)
            self.conv_blocks.append(conv_block)

        # 输出投影
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    @property
    def last_layer(self):
        """返回最后一层权重，用于 LPIPS/LPIP 感知损失的梯度缩放"""
        return self.conv_out.weight
    
    def forward(self, z):
        # 潜空间到特征空间
        h = self.conv_in(z)

        # 中间处理
        for mid_block in self.mid:
            h = mid_block(h)
        
        # 上采样路径
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks + 1):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.upsample(h)

        # 输出
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class VectorQuantizer(nn.Module):
    """
    向量量化器：将连续潜空间特征映射到离散码本。
    
    工作原理：
    1. 计算输入特征与码本中所有向量的距离
    2. 找到最近的码本向量（最近邻查找）
    3. 用码本向量替换输入特征（straight-through estimator 保持梯度）
    
    损失函数：
    - VQ loss: ||z_q - z||^2（使码本向量靠近编码器输出）
    - Commitment loss: β * ||z_q.detach() - z||^2（使编码器输出靠近码本向量）
    - Entropy loss: 鼓励码本均匀使用，防止码本崩塌
    
    如果启用 L2 归一化，则距离计算基于余弦相似度而非欧氏距离。
    """
    def __init__(self, n_e, e_dim, beta, entropy_loss_ratio, l2_norm, show_usage):
        """
        Args:
            n_e: 码本大小（词汇表大小）
            e_dim: 码本嵌入维度
            beta: Commitment loss 权重
            entropy_loss_ratio: 熵正则化损失权重
            l2_norm: 是否对码本和输入做 L2 归一化
            show_usage: 是否追踪码本使用率
        """
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        if self.l2_norm:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=-1)
        if self.show_usage:
            # 记录最近使用的码本索引，用于计算码本利用率
            # 使用 register_buffer 而不是 Parameter，避免梯度问题
            self.register_buffer("codebook_used", torch.zeros(65536))

    
    def forward(self, z):
        """
        向量量化前向传播。
        
        Args:
            z: 编码器输出特征，shape (batch, channel, height, width)
        
        Returns:
            z_q: 量化后的特征，shape 与输入相同
            emb_loss: 损失元组 (vq_loss, commit_loss, entropy_loss, codebook_usage)
            info: 附加信息 (perplexity, min_encodings, min_encoding_indices)
        """
        # 转换为 (batch, height, width, channel) 格式并展平
        z = torch.einsum('b c h w -> b h w c', z).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        if self.l2_norm:
            z = F.normalize(z, p=2, dim=-1)
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight

        # 计算到所有码本向量的距离: ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z*e^T
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, torch.einsum('n d -> d n', embedding))

        # 最近邻查找
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = embedding[min_encoding_indices].view(z.shape)
        
        # 初始化输出变量
        perplexity = None
        min_encodings = None
        vq_loss = None
        commit_loss = None
        entropy_loss = None
        codebook_usage = 0

        # 追踪码本使用率
        if self.show_usage and self.training:
            with torch.no_grad():
                cur_len = min_encoding_indices.shape[0]
                self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
                self.codebook_used[-cur_len:] = min_encoding_indices
            codebook_usage = len(torch.unique(self.codebook_used)) / self.n_e

        # 计算训练损失
        if self.training:
            vq_loss = torch.mean((z_q - z.detach()) ** 2)          # 码本损失：更新码本
            commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2)  # 承诺损失：更新编码器
            entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-d)  # 熵正则化

        # 直通估计器（Straight-Through Estimator）：
        # 前向使用量化值 z_q，反向梯度直接传给 z，使编码器可训练
        z_q = z + (z_q - z).detach()

        # 转回 (batch, channel, height, width) 格式
        z_q = torch.einsum('b h w c -> b c h w', z_q)

        return z_q, (vq_loss, commit_loss, entropy_loss, codebook_usage), (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        """
        根据码本索引获取对应的码本向量。
        
        Args:
            indices: 码本索引
            shape: 输出形状 (batch, channel, height, width)
            channel_first: 是否通道优先格式
        
        Returns:
            码本向量，shape 由 shape 参数决定
        """
        if self.l2_norm:
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight
        z_q = embedding[indices]  # (b*h*w, c)

        if shape is not None:
            if channel_first:
                z_q = z_q.reshape(shape[0], shape[2], shape[3], shape[1])
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
            else:
                z_q = z_q.view(shape)
        return z_q


class ResnetBlock(nn.Module):
    """
    残差卷积块：Conv-Norm-Activate-Conv + 残差连接。
    
    结构: x → Norm → Swish → Conv → Norm → Swish → Dropout → Conv → +x
    
    如果输入输出通道不同，使用 1x1 卷积（或 3x3 卷积）调整通道数。
    """
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type='group'):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels, norm_type)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x+h


class AttnBlock(nn.Module):
    """
    自注意力块：在空间维度上计算自注意力。
    
    使用 1x1 卷积生成 Q、K、V，然后计算标准的多头注意力，
    最后用 1x1 卷积投影输出。这与 Transformer 的注意力类似，
    但操作在 2D 特征图的空间维度上。
    
    仅在编码器和解码器的最高分辨率级别使用，平衡计算开销和信息融合能力。
    """
    def __init__(self, in_channels, norm_type='group'):
        super().__init__()
        self.norm = Normalize(in_channels, norm_type)
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

        # 计算自注意力
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # (b, hw, c)
        k = k.reshape(b,c,h*w) # (b, c, hw)
        w_ = torch.bmm(q,k)    # (b, hw, hw)
        w_ = w_ * (int(c)**(-0.5))  # 缩放
        w_ = F.softmax(w_, dim=2)

        # 注意力加权
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # (b, hw, hw)
        h_ = torch.bmm(v,w_)     # (b, c, hw)
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


def nonlinearity(x):
    """
    Swish 激活函数：x * sigmoid(x)。
    
    与 ReLU 相比，Swish 平滑且非单调，在 VQ-VAE 中表现更好。
    """
    return x*torch.sigmoid(x)


def Normalize(in_channels, norm_type='group'):
    """
    归一化层工厂函数。
    
    Args:
        in_channels: 输入通道数
        norm_type: 'group' 使用 GroupNorm（默认，单卡训练），'batch' 使用 SyncBatchNorm（多卡训练）
    """
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'batch':
        return nn.SyncBatchNorm(in_channels)


class Upsample(nn.Module):
    """
    上采样模块：最近邻插值 2x + 可选卷积。
    
    卷积层用于平滑插值后的特征，避免棋盘效应。
    """
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    下采样模块：2x 步幅卷积 或 平均池化。
    
    卷积方式使用 asymmetric padding 避免边界信息丢失。
    """
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # PyTorch 卷积不支持非对称 padding，需手动填充
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


def compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01):
    """
    计算熵正则化损失，鼓励码本均匀使用。
    
    码本崩塌（codebook collapse）是 VQ-VAE 的常见问题：
    只有少数码本向量被使用，其余永远不被选中。
    熵正则化通过最大化码本使用的熵来缓解这个问题。
    
    损失 = 样本熵 - 平均熵（鼓励每个样本使用多样的码本向量）
    
    Args:
        affinity: 码本亲和度矩阵（距离的负值）
        loss_type: 损失类型（目前仅支持 "softmax"）
        temperature: 温度参数（越低越尖锐）
    
    Returns:
        熵正则化损失标量
    """
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
    if loss_type == "softmax":
        target_probs = probs
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    avg_probs = torch.mean(target_probs, dim=0)
    avg_entropy = - torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
    sample_entropy = - torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    loss = sample_entropy - avg_entropy
    return loss
