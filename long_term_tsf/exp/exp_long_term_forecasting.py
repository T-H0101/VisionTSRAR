from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import json
from tqdm import tqdm

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    """
    长期时序预测实验类
    
    管理VisionTS/VisionTSRAR的训练、验证和测试流程，包括：
    - 模型构建
    - 数据加载
    - 优化器和损失函数选择
    - 训练循环（含早停）
    - 测试评估（含指标计算和结果保存）
    
    基于Time-Series-Library的实验框架，VisionTS和VisionTSRAR作为模型接入。
    
    VisionTSRAR的特殊处理：
    - 训练时forward返回(prediction, loss)元组，需要拆分处理
    - 测试时使用generate方法进行RAR采样
    - 自动根据freq参数配置periodicity
    """
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self._is_visiontsrar = (args.model == 'VisionTSRAR')
        # VisionTSRAR自动配置periodicity
        if self._is_visiontsrar and getattr(args, 'periodicity', 0) == 0:
            args.periodicity = self._infer_periodicity(args.freq)

    @staticmethod
    def _infer_periodicity(freq):
        """
        根据freq参数自动推断周期长度
        
        Args:
            freq: 时间频率字符串，如'h'(小时),'min'(分钟),'d'(天)等
        Returns:
            推断的周期长度，1表示无周期性
        """
        freq_map = {
            'h': 24,       # 小时数据 → 日周期24
            'min': 96,     # 分钟数据 → 日周期96(15min间隔)
            't': 96,       # minutely → 同min
            'd': 7,        # 日数据 → 周周期7
            'b': 5,        # 工作日 → 周期5
            'w': 52,       # 周数据 → 年周期52
            'm': 12,       # 月数据 → 年周期12
            'q': 4,        # 季度数据 → 年周期4
            '10min': 144,  # 10分钟间隔 → 日周期144
            '15min': 96,   # 15分钟间隔 → 日周期96
            '30min': 48,   # 30分钟间隔 → 日周期48
            's': 86400,    # 秒数据 → 日周期
        }
        # 先尝试完全匹配
        if freq in freq_map:
            return freq_map[freq]
        # 尝试解析带数字前缀的频率，如'3h'→24, '15min'→96
        import re
        match = re.match(r'(\d*)(h|min|t|d|w|m|b|q|s)', freq)
        if match:
            base_freq = match.group(2)
            if base_freq in freq_map:
                return freq_map[base_freq]
        return 1  # 默认无周期性

    def _build_model(self):
        """构建模型，根据配置创建VisionTS适配器实例"""
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        """获取数据集和数据加载器"""
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        """选择优化器：Adam（VisionTS仅微调少量参数，如LayerNorm）"""
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        """选择损失函数：MSE（均方误差）"""
        criterion = nn.MSELoss()
        return criterion

    def _model_forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        """
        统一的前向调用接口，处理不同模型的输出格式

        对于VisionTSRAR：
        - 训练时返回(prediction, loss)元组
        - 推理时返回prediction

        对于VisionTS等其他模型：
        - 始终返回prediction

        Args:
            batch_x, batch_x_mark, dec_inp, batch_y_mark: 标准TSL输入
        Returns:
            outputs: 模型预测输出
            model_loss: 模型自带的loss（VisionTSRAR训练时），无则None
        """
        model_loss = None
        if self.args.output_attention:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        # VisionTSRAR训练时forward返回(prediction, loss)元组
        if self._is_visiontsrar and isinstance(outputs, tuple):
            outputs, model_loss = outputs

        return outputs, model_loss

    def vali(self, vali_data, vali_loader, criterion):
        """
        验证过程：在验证集上计算损失，用于早停判断

        Args:
            vali_data: 验证数据集
            vali_loader: 验证数据加载器
            criterion: 损失函数（MSE）
        Returns:
            平均验证损失
        """
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(vali_loader, desc='vali')):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, model_loss = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs, model_loss = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                if model_loss is not None:
                    loss = model_loss
                else:
                    pred = outputs.detach().cpu()
                    true = batch_y.detach().cpu()
                    loss = criterion(pred, true)

                total_loss.append(loss.item() if isinstance(loss, torch.Tensor) else loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        """
        训练流程
        
        训练过程：
        1. 加载训练/验证/测试数据
        2. 创建优化器、损失函数、早停机制
        3. 每个epoch: 前向→计算loss→反向传播→参数更新
        4. 每个epoch结束后验证，根据验证损失早停
        5. 训练结束后加载最佳模型
        
        VisionTS/VisionTSRAR的训练特点：
        - 默认ft_type='ln'，仅微调LayerNorm参数，训练非常快
        - 主体参数冻结，预训练的视觉知识得到保留
        - 通常只需少量epoch（5-10）即可收敛
        - VisionTSRAR训练时forward返回(prediction, loss)，loss可直接用于反向传播
        
        Args:
            setting: 实验设置字符串（用于保存checkpoint和结果）
        Returns:
            训练后的模型
        """
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.save_dir, self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            pbar = tqdm(train_loader)
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pbar):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, model_loss = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                        if model_loss is not None:
                            loss = model_loss
                        else:
                            loss = criterion(outputs, batch_y)

                        train_loss.append(loss.item() if isinstance(loss, torch.Tensor) else loss)
                else:
                    outputs, model_loss = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    if model_loss is not None:
                        loss = model_loss
                        if (i + 1) % 100 == 0:
                            tqdm.write(f"[Loss] MAE+MSE: {loss.item():.6f}")
                    else:
                        pred = outputs
                        loss = criterion(pred, batch_y)

                    train_loss.append(loss.item() if isinstance(loss, torch.Tensor) else loss)

                pbar.set_description("epoch: {0} | loss: {1:.7f}".format(epoch + 1, loss.item()))
                if (i + 1) % 100 == 0:
                    tqdm.write("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    tqdm.write('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'fixed':
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        if os.path.isfile(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
        else:
            print("Test without train!",best_model_path)

        return self.model

    def test(self, setting, test=0):
        """
        测试流程
        
        在测试集上评估模型性能，计算指标（MSE/MAE/RMSE/MAPE/MSPE），
        并保存预测结果和指标到文件。
        
        Args:
            setting: 实验设置字符串
            test: 是否从checkpoint加载模型（1=加载，0=使用当前模型）
        Returns:
            None（结果保存到文件）
        """
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(f'{self.args.save_dir}/checkpoints/' + setting, 'checkpoint.pth')))

        valid_loss_path = os.path.join(f'{self.args.save_dir}/checkpoints/' + setting, 'valid_loss.json')
        if os.path.isfile(valid_loss_path):
            with open(valid_loss_path) as f:
                valid_loss = json.load(f)
                best_valid_loss = valid_loss['best_valid_loss']
                best_valid_epoch = valid_loss['best_valid_epoch']
        else:
            best_valid_loss = -1
            best_valid_epoch = -1
        

        preds = []
        trues = []
        folder_path = f'{self.args.save_dir}/test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader, desc='test')):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder（VisionTSRAR在eval模式下使用generate）
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, _ = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs, _ = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = f'{self.args.save_dir}/results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, best_valid_loss, best_valid_epoch]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
