import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
import random
import numpy as np

if __name__ == '__main__':
    # fix_seed = 2021
    # random.seed(fix_seed)
    # torch.manual_seed(fix_seed)
    # np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--save_dir', type=str, default='.', help='save dir')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--channel_independence', type=int, default=0,
                        help='1: channel dependence 0: channel independence for FreTS model')
    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # VisionTS专用参数
    # 以下参数控制VisionTS模型的行为，是VisionTS论文中的关键超参数
    parser.add_argument('--vm_pretrained', type=int, default=1,
                        help='是否加载MAE预训练权重（1=加载，0=不加载/随机初始化）。'
                             '预训练权重是VisionTS的核心，加载ImageNet预训练的MAE权重是零样本预测的基础')
    parser.add_argument('--vm_ckpt', type=str, default="../ckpt/",
                        help='MAE预训练权重的本地存储目录。'
                             '首次运行时会自动从Facebook下载权重到此目录')
    parser.add_argument('--vm_arch', type=str, default='mae_base',
                        help='MAE架构选择: mae_base(ViT-B/16), mae_large(ViT-L/16), mae_huge(ViT-H/14)。'
                             '更大的架构性能更好但计算开销更大')
    parser.add_argument('--ft_type', type=str, default='ln',
                        help='微调策略: ln(仅LayerNorm), bias(仅偏置), none(完全冻结/零样本), '
                             'full(全部参数), mlp*(仅MLP), attn*(仅注意力)。'
                             'ln是论文推荐的默认策略，仅微调少量参数即可获得良好效果')
    parser.add_argument('--periodicity', type=int, default=0,
                        help='时序周期长度，用于2D折叠。0=自动检测（通过freq参数推断）。'
                             '例如: 24=小时数据的日周期, 12=月度数据的年周期。'
                             '1=无周期性，1D序列直接排列。'
                             '周期性越大，2D折叠后图像越能捕捉周期模式')
    parser.add_argument('--interpolation', type=str, default='bilinear',
                        help='图像缩放插值方法: bilinear(默认), nearest, bicubic。'
                             'bilinear在大多数情况下效果最好')
    parser.add_argument('--norm_const', type=float, default=0.4,
                        help='归一化常数，控制归一化后值域范围（论文默认0.4）。'
                             '值越大→归一化后数值范围越小→图像像素越集中。'
                             '0.4约束值域约在[-2.5, 2.5]，与ImageNet归一化后范围接近，'
                             '使MAE预训练权重能更好地迁移到时序数据')
    parser.add_argument('--align_const', type=float, default=0.4,
                        help='对齐常数，控制输入占图像比例（论文默认0.4）。'
                             '值越大→输入区域占比越大→MAE补全空间越小。'
                             '0.4表示输入只占40%的patch列数，60%留给MAE补全预测。'
                             '较小的align_const给MAE更多"画布"，但可能导致输入信息不足')

    # RAR模型专用参数
    # 以下参数控制VisionTSRAR模型的行为，基于RAR(Randomized Autoregressive)替代MAE
    parser.add_argument('--rar_arch', type=str, default='rar_l_0.3b',
                        choices=['rar_l_0.3b'], help='RAR GPT架构选择')
    parser.add_argument('--vq_ckpt', type=str, default=None,
                        help='VQ tokenizer权重文件路径')
    parser.add_argument('--rar_ckpt', type=str, default=None,
                        help='RAR GPT权重文件路径')
    parser.add_argument('--num_inference_steps', type=int, default=88,
                        help='RAR生成的推理步数（默认88）')
    parser.add_argument('--position_order', type=str, default='raster',
                        choices=['raster', 'random'], help='Token位置顺序')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='RAR采样温度')
    parser.add_argument('--rar_top_k', type=int, default=0,
                        help='RAR采样的top-k值（0=不使用）')
    parser.add_argument('--rar_top_p', type=float, default=1.0,
                        help='RAR采样的top-p值（1.0=不使用）')
    parser.add_argument('--cfg_scales', type=str, default='1.0,1.0',
                        help='RAR生成的CFG scales（起始,结束），如"1.0,1.0"')


    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            if args.save_dir != '.':
                setting = '_'
            else:
                setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                    args.task_name,
                    args.model_id,
                    args.model,
                    args.data,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.factor,
                    args.embed,
                    args.distil,
                args.des, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
