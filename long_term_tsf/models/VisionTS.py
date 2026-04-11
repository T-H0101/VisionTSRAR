
import sys
sys.path.append("../")

from torch import nn
from visionts import VisionTS

class Model(nn.Module):
    """
    VisionTS适配器：将VisionTS核心模型封装为Time-Series-Library的标准接口
    
    适配器模式：
    - Time-Series-Library要求所有模型遵循统一的接口（forward/forecast/imputation等）
    - VisionTS核心模型(visionts.VisionTS)只提供forward方法
    - 此适配器负责：构造VisionTS模型、调用update_config、转发forecast请求
    
    使用方式：
    - 通过run.py中的--model VisionTS参数指定
    - 模型名通过models/model_dict.py映射到此文件
    """

    def __init__(self, config):
        """
        根据配置初始化VisionTS模型
        
        Args:
            config: 命令行参数对象，包含以下VisionTS专用参数：
                - vm_arch: MAE架构选择（'mae_base'/'mae_large'/'mae_huge'）
                - ft_type: 微调策略（'ln'/'bias'/'none'/'full'等）
                - vm_pretrained: 是否加载预训练权重（1=加载，0=不加载）
                - vm_ckpt: MAE权重文件目录
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

        # 创建VisionTS核心模型，传入MAE架构和微调策略
        self.vm = VisionTS(arch=config.vm_arch, finetune_type=config.ft_type, load_ckpt=config.vm_pretrained == 1, ckpt_dir=config.vm_ckpt)

        # 根据时序数据参数配置图像布局
        # 这一步计算了掩码、缩放参数等，是VisionTS适配不同时序数据的关键
        self.vm.update_config(context_len=config.seq_len, pred_len=config.pred_len, periodicity=config.periodicity, interpolation=config.interpolation, norm_const=config.norm_const, align_const=config.align_const)


    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        时序预测（VisionTS的核心功能）
        
        Args:
            x_enc: 编码器输入（回看窗口），[B, L, D]
            x_mark_enc: 时间特征（VisionTS不使用）
            x_dec: 解码器输入（VisionTS不使用）
            x_mark_dec: 解码器时间特征（VisionTS不使用）
        Returns:
            预测结果，[B, L, D]
        """
        return self.vm.forward(x_enc)


    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        """时序填补（VisionTS暂不支持）"""
        raise NotImplementedError()


    def anomaly_detection(self, x_enc):
        """异常检测（VisionTS暂不支持）"""
        raise NotImplementedError()


    def classification(self, x_enc, x_mark_enc):
        """分类（VisionTS暂不支持）"""
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
