
import sys
from pathlib import Path

import torch
import torch.nn as nn

# 将visiontsrar核心库加入搜索路径
sys.path.append(str(Path(__file__).resolve().parents[2]))
from visiontsrar import VisionTSRAR


class Model(nn.Module):
    """
    VisionTSRAR适配器：将VisionTSRAR核心模型封装为Time-Series-Library的标准接口

    适配器模式：
    - Time-Series-Library要求所有模型遵循统一的接口（forward/forecast等）
    - VisionTSRAR核心模型使用RAR(Randomized Autoregressive)替代MAE做图像补全
    - 此适配器负责：构造VisionTSRAR模型、调用update_config、转发forecast请求

    与VisionTS适配器的区别：
    - 使用RAR GPT生成替代MAE补全
    - 支持RAR专用的采样参数（temperature, top_k, top_p, cfg_scales）
    - 训练时forward返回(prediction, loss)，需要处理loss
    - 测试时使用generate而非forward

    使用方式：
    - 通过run.py中的--model VisionTSRAR参数指定
    """

    def __init__(self, config):
        """
        根据配置初始化VisionTSRAR模型

        Args:
            config: 命令行参数对象，包含以下参数：
                - rar_arch: RAR GPT架构选择（'rar_l_0.3b'）
                - ft_type: 微调策略（'ln'/'bias'/'none'/'full'等）
                - vm_ckpt: 模型权重文件目录
                - vm_pretrained: 是否加载预训练权重
                - vq_ckpt: VQ tokenizer权重路径
                - rar_ckpt: RAR GPT权重路径
                - num_inference_steps: RAR推理步数
                - position_order: Token位置顺序
                - seq_len: 输入序列长度
                - pred_len: 预测序列长度
                - periodicity: 周期长度（0=自动检测）
                - interpolation: 插值方法
                - norm_const: 归一化常数
                - align_const: 对齐常数
        """
        super().__init__()
        self.task_name = config.task_name
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len

        # 创建 VisionTSRAR 核心模型
        self.vm = VisionTSRAR(
            arch=config.rar_arch,
            finetune_type=getattr(config, 'ft_type', 'ln'),
            ckpt_dir=getattr(config, 'vm_ckpt', '../ckpt/'),
            load_ckpt=getattr(config, 'vm_pretrained', 1) == 1,
            vq_ckpt=getattr(config, 'vq_ckpt', None),
            rar_ckpt=getattr(config, 'rar_ckpt', None),
            num_inference_steps=getattr(config, 'num_inference_steps', 88),
            position_order=getattr(config, 'position_order', 'random'),
            use_lightweight_decoder=getattr(config, 'use_lightweight_decoder', False),
            lightweight_decoder_channels=getattr(config, 'lightweight_decoder_channels', 64),
        )

        # 保存RAR采样参数
        self.temperature = getattr(config, 'temperature', 1.0)
        self.top_k = getattr(config, 'top_k', 0)
        self.top_p = getattr(config, 'top_p', 1.0)
        self.cfg_scales = self._parse_cfg_scales(getattr(config, 'cfg_scales', '1.0,1.0'))

        # 根据时序数据参数配置图像布局
        self.vm.update_config(
            context_len=config.seq_len,
            pred_len=config.pred_len,
            periodicity=config.periodicity,
            interpolation=getattr(config, 'interpolation', 'bilinear'),
            norm_const=getattr(config, 'norm_const', 0.4),
            align_const=getattr(config, 'align_const', 0.4),
        )

    def _parse_cfg_scales(self, cfg_scales_str):
        """解析CFG scales字符串为(float, float)元组"""
        try:
            parts = cfg_scales_str.split(',')
            return (float(parts[0]), float(parts[1]))
        except (ValueError, IndexError):
            return (1.0, 1.0)

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        时序预测（VisionTSRAR的核心功能）

        训练和测试均调用 vm.forward()，内部根据 self.training 自动选择：
        - 训练时：使用 teacher forcing，返回 (prediction, loss)
        - 推理时：使用 RAR GPT generate 自回归生成，返回 prediction

        Args:
            x_enc: 编码器输入（回看窗口），[B, L, D]
            x_mark_enc: 时间特征（VisionTSRAR不使用）
            x_dec: 解码器输入（VisionTSRAR不使用）
            x_mark_dec: 解码器时间特征（VisionTSRAR不使用）
        Returns:
            训练时: (prediction, loss) 元组
            测试时: prediction 张量
        """
        result = self.vm.forward(x_enc)
        if self.training and isinstance(result, tuple):
            # 训练模式：forward 内部调用 rar_wrapper.forward()，返回 (reconstructed, loss)
            # 但实际上 VisionTSRAR.forward() 训练时只返回 y（预测值），
            # 因为 model.py 中的第4步返回了 loss 但没有传递到最终输出
            # 这里需要检查实际行为
            return result
        return result

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        """时序填补（VisionTSRAR暂不支持）"""
        raise NotImplementedError()

    def anomaly_detection(self, x_enc):
        """异常检测（VisionTSRAR暂不支持）"""
        raise NotImplementedError()

    def classification(self, x_enc, x_mark_enc):
        """分类（VisionTSRAR暂不支持）"""
        raise NotImplementedError()

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        统一前向接口，根据任务类型分发到对应方法

        Args:
            x_enc: 编码器输入
            x_mark_enc: 编码器时间特征
            x_dec: 解码器输入
            x_mark_dec: 解码器时间特征
            mask: 掩码（仅imputation任务使用）
        Returns:
            任务对应的输出
        """
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            if isinstance(dec_out, tuple):
                # 训练模式返回(prediction, loss)
                pred, loss = dec_out
                return pred[:, -self.pred_len:, :], loss
            return dec_out[:, -self.pred_len:, :]  # [B, L, D] 只返回预测部分
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
