"""
VisionTSRAR: 基于 RAR (Randomized Autoregressive) 的时间序列预测模型

将 VisionTS 的 MAE 图像补全替换为 RAR 自回归生成，
继承 VisionTS 的6步流水线架构：
1. Normalization → 2. Segmentation → 3. Render & Alignment
→ 4. RAR Reconstruction（新）→ 5. Forecasting → 6. Denormalization
"""

from .model import VisionTSRAR, VisionTSRARpp
from .util import freq_to_seasonality_list

__version__ = "0.1.0"
__all__ = ["VisionTSRAR", "VisionTSRARpp", "freq_to_seasonality_list"]
