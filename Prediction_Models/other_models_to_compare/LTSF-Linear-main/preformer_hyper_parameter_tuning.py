import torch
from exp.exp_main import Exp_Main
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
from hyperopt import fmin, tpe, Trials, STATUS_OK, hp
import torch
import pickle


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
        preds, trues, mse, mae = self.exp.custom_predict(model, load_saved=False)
        print(f'MSE: {mse} MAE: {mae}')
        preds = preds[:, -1].reshape(-1, 1)
        trues = trues[:, -1].reshape(-1, 1)
        self._plot(trues, preds)
        return preds, trues, mse, mae


def objective(params):
    try:
        args_dict_train = {
            "is_training": 1,
            "train_only": True,
            "model_id": "preformer_btc_1_1_1_close_only",
            "model": "Preformer",
            "data": "custom",
            "root_path": "../../../data/",
            "data_path": "training_data_close_col_only_for_other_models.csv",
            "features": "MS",
            "target": "close",
            "freq": "h",
            "seq_len": 12,
            "label_len": 1,
            "pred_len": 1,
            "enc_in": 1,
            "dec_in": 1,
            "c_out": 1,
            "d_model": params['d_model'],
            "n_heads": params['n_heads'],
            "e_layers": params['e_layers'],
            "d_layers": params['d_layers'],
            "d_ff": params['d_ff'],
            "dropout": params['dropout'],
            "embed": "timeF",
            "activation": "gelu",
            "do_predict": True,
            "train_epochs": 50,
            "batch_size": 64,
            "learning_rate": params['learning_rate'],
            "loss": "mse",
            "use_gpu": 1,
            "gpu": 3,
            "train_epochs": 1
        }
        torch.cuda.set_device(3)  
        model = Train_Tester(**args_dict_train).train()

        args_dict_val = {
            "root_path": "../../../data/",
            "data_path": "validation_data_close_col_only_for_other_models.csv",
            "model_id": "preformer_btc_1_1_1_close_only",
            "model": "Preformer",
            "data": "custom",
            "features": "MS",
            "target": "close",
            "freq": "h",
            "seq_len": 12,
            "label_len": 1,
            "pred_len": 1,
            "enc_in": 1,
            "dec_in": 1,
            "c_out": 1,
            "d_model": params['d_model'],
            "n_heads": params['n_heads'],
            "e_layers": params['e_layers'],
            "d_layers": params['d_layers'],
            "d_ff": params['d_ff'],
            "dropout": params['dropout'],
            "embed": "timeF",
            "activation": "gelu",
            "do_predict": True,
            "use_gpu": 1,
            "gpu": 3,
            "batch_size": 1,
        }

        mse, mae = Train_Tester(**args_dict_val).validate(model)

        return {'loss': mse, 'status': STATUS_OK}
    except Exception as e:
        print(f"Exception occurred with parameters={params}: {str(e)}")
        return {'loss': float('inf'), 'status': STATUS_OK}


def main():
    space = {
    'd_model': hp.choice('d_model', [256, 512, 1024]),
    'n_heads': hp.choice('n_heads', [4, 8, 16]),
    'e_layers': hp.choice('e_layers', [1, 2, 3]),
    'd_layers': hp.choice('d_layers', [1, 2, 3]),
    'd_ff': hp.choice('d_ff', [1024, 2048, 4096]),
    'dropout': hp.uniform('dropout', 0.0, 0.5),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(1)),
    }

    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials
    )

    print(best)

    file_path = 'preformer_best_hyperparams.pkl'
    with open(file_path, 'wb') as file:
        pickle.dump(best, file)

    

if __name__ == "__main__":
    main()