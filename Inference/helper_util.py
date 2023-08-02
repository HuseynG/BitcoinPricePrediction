import time
import requests
import pandas as pd
import torch

from collections import namedtuple
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import sys
sys.path.append('../Prediction_Models/other_models_to_compare/LTSF-Linear-main')
from exp.exp_main import Exp_Main
# from exp.exp_main import Exp_Main

def get_btc_data(start_time, end_time, interval):
    total_call_count = 1
    requests_made = 0
    cooldown_period = 100
    results = []
    limit = 1000
    url = 'https://www.bitstamp.net/api/v2/ohlc/btcusd'

    while end_time > start_time:
        if requests_made >= 8000: 
            print("Cooldown for 100 seconds.")
            time.sleep(cooldown_period) 
            requests_made = 0 
        print("Call No.", total_call_count)
        params = {
            'start': start_time,
            'end': end_time,
            'step': interval,
            'limit': limit
        }
        response = requests.get(url, params=params)
        data = response.json()['data']['ohlc']
        results.extend(data)
        end_time -= (interval * limit)
        total_call_count += 1
        requests_made += 1
    results = [d for d in results if int(d['timestamp']) >= start_time]
    results.reverse()
    return results

def prep_btc_data_for_prediction(lookback, pred_len = 1):

    # get btc data
    end_time = int(time.time())
    start_time = end_time - ((int(lookback) + 1 + pred_len) * 60 * 60) # last 504 hours
    interval = 3600 # resolution (1 hour)
    btc_data = get_btc_data(start_time, end_time, interval)

    # preprocess btc data for the model
    temp_df = pd.DataFrame(btc_data)[['timestamp','close']] # 
    temp_df['date'] = pd.to_datetime(temp_df['timestamp'], unit='s')
    temp_df.drop(columns=['timestamp'], inplace=True)
    temp_df = temp_df[['date'] + [col for col in temp_df.columns if col != 'date' and col != 'close'] + ['close']] # last col is target col
    temp_df.to_csv('data.csv', index=False)

class Train_Tester:
    def __init__(self, **kwargs):

        self.is_training = kwargs.get('is_training', 1)
        self.train_only = kwargs.get('train_only', False)
        self.model_id = kwargs.get('model_id', 'test')
        self.model = kwargs.get('model', 'Autoformer')
        self.saved_model_path = kwargs.get('saved_model_path', None)

        self.data = kwargs.get('data', 'ETTm1')
        self.root_path = kwargs.get('root_path', './data/ETT/')
        self.data_path = kwargs.get('data_path', 'ETTh1.csv')
        self.features = kwargs.get('features', 'M')
        self.target = kwargs.get('target', 'close')
        self.freq = kwargs.get('freq', 'h')
        self.checkpoints = kwargs.get('checkpoints', './checkpoints/')

        self.seq_len = kwargs.get('seq_len', 96)
        self.label_len = kwargs.get('label_len', 48)
        self.pred_len = kwargs.get('pred_len', 96)

        self.individual = kwargs.get('individual', False)
        self.embed_type = kwargs.get('embed_type', 0)
        self.enc_in = kwargs.get('enc_in', 7)
        self.dec_in = kwargs.get('dec_in', 7)
        self.c_out = kwargs.get('c_out', 7)
        self.d_model = kwargs.get('d_model', 512)
        self.n_heads = kwargs.get('n_heads', 8)
        self.e_layers = kwargs.get('e_layers', 2)
        self.d_layers = kwargs.get('d_layers', 1)
        self.d_ff = kwargs.get('d_ff', 2048)
        self.moving_avg = kwargs.get('moving_avg', 25)
        self.factor = kwargs.get('factor', 1)
        self.distil = kwargs.get('distil', True)
        self.dropout = kwargs.get('dropout', 0.05)
        self.embed = kwargs.get('embed', 'timeF')
        self.activation = kwargs.get('activation', 'gelu')
        self.output_attention = kwargs.get('output_attention', False)
        self.do_predict = kwargs.get('do_predict', False)
        self.do_custom_predict = kwargs.get('do_custom_predict', False)

        self.num_workers = kwargs.get('num_workers', 10)
        self.itr = kwargs.get('itr', 2)
        self.train_epochs = kwargs.get('train_epochs', 10)
        self.batch_size = kwargs.get('batch_size', 32)
        self.patience = kwargs.get('patience', 3)
        self.learning_rate = kwargs.get('learning_rate', 0.0001)
        self.des = kwargs.get('des', 'test')
        self.loss = kwargs.get('loss', 'mse')
        self.lradj = kwargs.get('lradj', 'type1')
        self.use_amp = kwargs.get('use_amp', False)

        self.use_gpu = kwargs.get('use_gpu', True)
        self.gpu = kwargs.get('gpu', 0)
        self.use_multi_gpu = kwargs.get('use_multi_gpu', False)
        self.devices = kwargs.get('devices', '0,1,2,3')
        self.test_flop = kwargs.get('test_flop', False)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.exp = Exp_Main

    def _plot(self, trues, preds):
        plt.figure(figsize=(12,6))
        plt.plot(trues, label='True', linewidth=0.5)
        plt.plot(preds, label='Predicted', linewidth=0.5)
        plt.title("Model Per. on Test Set")
        plt.legend()
        plt.show()

    def _set_args(self):
        Args = namedtuple('Args', ['is_training', 'train_only', 'model_id', 'model', 'saved_model_path',
                        'data', 'root_path', 'data_path', 'features', 'target', 'freq', 'checkpoints',
                        'seq_len', 'label_len', 'pred_len', 'individual', 'embed_type', 'enc_in',
                        'dec_in', 'c_out', 'd_model', 'n_heads', 'e_layers', 'd_layers', 'd_ff',
                        'moving_avg', 'factor', 'distil', 'dropout', 'embed', 'activation',
                        'output_attention', 'do_predict', 'do_custom_predict', 'num_workers', 'itr',
                        'train_epochs', 'batch_size', 'patience', 'learning_rate', 'des', 'loss',
                        'lradj', 'use_amp', 'use_gpu', 'gpu', 'use_multi_gpu', 'devices', 'test_flop'])

        self.args = Args(is_training=self.is_training, train_only=self.train_only, model_id=self.model_id,
                    model=self.model, saved_model_path=self.saved_model_path, data=self.data,
                    root_path=self.root_path, data_path=self.data_path, features=self.features,
                    target=self.target, freq=self.freq, checkpoints=self.checkpoints, seq_len=self.seq_len,
                    label_len=self.label_len, pred_len=self.pred_len, individual=self.individual,
                    embed_type=self.embed_type, enc_in=self.enc_in, dec_in=self.dec_in, c_out=self.c_out,
                    d_model=self.d_model, n_heads=self.n_heads, e_layers=self.e_layers, d_layers=self.d_layers,
                    d_ff=self.d_ff, moving_avg=self.moving_avg, factor=self.factor, distil=self.distil,
                    dropout=self.dropout, embed=self.embed, activation=self.activation,
                    output_attention=self.output_attention, do_predict=self.do_predict,
                    do_custom_predict=self.do_custom_predict, num_workers=self.num_workers, itr=self.itr,
                    train_epochs=self.train_epochs, batch_size=self.batch_size, patience=self.patience,
                    learning_rate=self.learning_rate, des=self.des, loss=self.loss, lradj=self.lradj,
                    use_amp=self.use_amp, use_gpu=self.use_gpu, gpu=self.gpu, use_multi_gpu=self.use_multi_gpu,
                    devices=self.devices, test_flop=self.test_flop)


        self.exp = Exp_Main(self.args)

    def train(self):
        self._set_args()
        for ii in range(self.itr):
            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                self.model_id,
                self.model,
                self.data,
                self.features,
                self.seq_len,
                self.label_len,
                self.pred_len,
                self.d_model,
                self.n_heads,
                self.e_layers,
                self.d_layers,
                self.d_ff,
                self.factor,
                self.embed,
                self.distil,
                self.des, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            self.exp.train(setting)
            torch.cuda.empty_cache()
        return self.exp.train(setting)
    
    def validate(self, model):
        self._set_args()
        preds, trues, mse, mae = self.exp.custom_predict(model, load_saved=False)
        return mse, mae

    def test(self, model):
        self._set_args()
        preds, trues, mse, mae = self.exp.custom_predict(model, load_saved=True)
        print(f'MSE: {mse} MAE: {mae}')
        preds = preds[:, -1].reshape(-1, 1)
        trues = trues[:, -1].reshape(-1, 1)
        # self._plot(trues, preds)
        
        return preds, trues, mse, mae
