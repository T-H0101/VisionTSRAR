import os
import torch
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, VisionTSRAR, \
    iTransformer, Koopa, TiDE, FreTS

# VisionTS 依赖外部 visionts 包，可选导入
try:
    from models import VisionTS
    _HAS_VISIONTS = True
except ImportError:
    _HAS_VISIONTS = False


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS, 
            'VisionTSRAR': VisionTSRAR,
        }
        if _HAS_VISIONTS:
            self.model_dict['VisionTS'] = VisionTS
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        
        # torch.compile 加速（需要PyTorch 2.0+）
        if getattr(args, 'use_torch_compile', False):
            try:
                print("[torch.compile] 正在编译模型...")
                self.model = torch.compile(self.model)
                print("[torch.compile] 模型编译完成")
            except Exception as e:
                print(f"[torch.compile] 编译失败: {e}")

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
