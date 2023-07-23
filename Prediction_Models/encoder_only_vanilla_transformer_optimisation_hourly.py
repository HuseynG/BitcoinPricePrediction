import time
from collections import OrderedDict

import torch
import pandas as pd
import numpy as np
from torch.optim import Adam
from torch.nn import MSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.pyll import scope
from transformer import VanillaTimeSeriesTransformer_EncoderOnly
from utils import Trainer, preprocess_data, seed_everything
import pickle


class Optimizer:
    def __init__(self, data_file, input_seq_len=36, output_seq_len=1, batch_size=256, cuda_='cuda:3'):
        
        self.df = pd.read_csv(data_file)
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.batch_size = batch_size
        self.temp_df = None
        self.cuda_ = cuda_

    def objective(self, params):
        scaled_df, _, scaler_close, train_dataloader, val_dataloader, test_dataloader = \
            preprocess_data(self.temp_df, self.batch_size, self.input_seq_len, self.output_seq_len)
        if (params['num_layers'] <= 0 or params['num_heads'] <= 0 or 
        params['d_model_by_num_heads'] <= 0 or params['dff'] <= 0 or
        params['mlp_size'] <= 0):
            print("Invalid hyperparameters sampled, correcting values...")
            params['num_layers'] = max(1, params['num_layers'])
            params['num_heads'] = max(1, params['num_heads'])
            params['d_model_by_num_heads'] = max(32, params['d_model_by_num_heads'])
            params['dff'] = max(2, params['dff'])
            params['mlp_size'] = max(32, params['mlp_size'])

        start_time = int(time.time())
        device = torch.device(self.cuda_ if torch.cuda.is_available() else 'cpu')
        num_heads = params['num_heads']
        d_model = params['d_model_by_num_heads'] * num_heads

        model = VanillaTimeSeriesTransformer_EncoderOnly(
            num_features=int(len(scaled_df.columns)),
            num_layers=params['num_layers'],
            d_model=d_model,
            num_heads=num_heads,
            dff=params['dff'],
            mlp_size=params['mlp_size'],
            dropout_rate=params['dropout_rate'],
            mlp_dropout_rate=params['mlp_dropout_rate']
        )
        model = model.to(device)

        optimiser = Adam(model.parameters(), lr=params['lr'])
        scheduler = ReduceLROnPlateau(optimiser, 'min', factor=0.9, patience=5)
        criterion = MSELoss()

        model_trainer = Trainer(model=model,
                                train_dataloader=train_dataloader,
                                val_dataloader=val_dataloader,
                                test_dataloader=test_dataloader,
                                criterion=criterion,
                                optimiser=optimiser,
                                scheduler=scheduler,
                                device=device,
                                num_epochs=50,
                                early_stopping_patience_limit=10,
                                is_save_model=True,
                                scaler=scaler_close,
                                file_path=f"models/best_model_{start_time}.pt")

        _, val_losses = model_trainer.train_loop()

        return {'loss': val_losses[-1], 'status': STATUS_OK}

    def optimize(self, space, df_features, max_evals=100):
        self.temp_df = self.df[df_features]
        trials = Trials()
        best = fmin(
            fn=self.objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials
        )
        return best

    def find_top_features(self, best):
        stats = {}
        for cols in self.df.columns:
            if cols != "close":
                print(cols)
                # prepare temp_df
                temp_df = self.df[["close", cols]]

                # preprocesing temp_df
                scaled_df, _, scaler_close, train_dataloader, \
                val_dataloader, test_dataloader = preprocess_data(temp_df,
                    batch_size = 256,
                    input_seq_len=self.input_seq_len,
                    output_seq_len=self.output_seq_len)

                # instantiate model (after hyperparameter tuning)
                num_features = int(len(scaled_df.columns)) # a.k.a, number of cols in df
                num_layers = int(best["num_layers"])
                num_heads = int(best["num_heads"])
                d_model = int(best['d_model_by_num_heads']) * num_heads
                dff = int(best['dff'])
                mlp_size = int(best['mlp_size']) # size of the first MLP layer
                dropout_rate = round(best['dropout_rate'], 3)  # dropout rate for the Transformer layers
                mlp_dropout_rate = round(best['mlp_dropout_rate'], 3) # dropout rate for the MLP layers

                # instantiating model
                model = VanillaTimeSeriesTransformer_EncoderOnly(num_features, num_layers, d_model, num_heads, dff,
                                                                mlp_size, dropout_rate, mlp_dropout_rate)

                # moving the model to the device (GPU if available)
                device = torch.device(self.cuda_ if torch.cuda.is_available() else 'cpu')
                model = model.to(device)


                criterion = MSELoss()
                optimiser = Adam(model.parameters(), lr=round(best['lr'], 6))
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', factor=0.9, patience=5)

                # declaring trainer object
                model_trainer = Trainer(model=model,
                                train_dataloader=train_dataloader,
                                val_dataloader=val_dataloader,
                                test_dataloader=test_dataloader,
                                criterion=criterion,
                                optimiser=optimiser,
                                scheduler=scheduler,
                                device=device,
                                num_epochs=50,
                                early_stopping_patience_limit=10,
                                is_save_model=True,
                                scaler=scaler_close,
                                file_path = "models/best_model.pt")
                # training
                train_losses, val_losses = model_trainer.train_loop()
                # testing
                mse, mae = model_trainer.test_model()

                stats[cols] = {
                    "mse":mse,
                    "mae":mae
                }

        return stats



if __name__ == '__main__':
    seed_everything()

    ##### Vanilla Transformer (Next Step Prediction | Encoder only) ######

    ### Hyperparameter Tuning (Structural Tuning Too) for 2 Features ###
    optimizer = Optimizer("../data/Final_data_hourly.csv")
    space = {
        'num_layers': scope.int(hp.quniform('num_layers', 1, 8, 1)),
        'num_heads': scope.int(hp.quniform('num_heads', 1, 8, 1)),
        'd_model_by_num_heads': scope.int(hp.quniform('d_model_by_num_heads', 32, 64, 2)),
        'dff': scope.int(hp.quniform('dff', 2, 2048, 50)),
        'mlp_size': scope.int(hp.quniform('mlp_size', 32, 64, 2)),
        'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.3),
        'mlp_dropout_rate': hp.uniform('mlp_dropout_rate', 0.1, 0.3),
        'lr': hp.loguniform('lr', np.log(0.0001), np.log(0.1))
    }
    best = optimizer.optimize(space, ["close", "open"])
    print(best)
    with open('../data/stats_on_hyperparam_for_two_cols_vanilla_transformer_hourly_encoder_only.pkl', 'wb') as file:
        pickle.dump(best, file)


    ### Finding Top Features ###
    with open('../data/stats_on_hyperparam_for_two_cols_vanilla_transformer_hourly_encoder_only.pkl', 'rb') as file:
        best = pickle.load(file)
    stats = optimizer.find_top_features(best)
    with open('../data/stats_on_features_vanilla_transformer_hourly.pkl', 'wb') as file:
        pickle.dump(stats, file)
        
    with open('../data/stats_on_features_vanilla_transformer_hourly.pkl', 'rb') as file:
        loaded_stats = pickle.load(file)
    sorted_loaded_stats = OrderedDict(sorted(loaded_stats.items(), key=lambda item: item[1]['mse']))
    count = 0
    top_feattures = []
    for k,v in sorted_loaded_stats.items():
        if count < 10:
            top_feattures.append(k)
            count += 1
    print(top_feattures)
    with open('../data/top_feattures.pkl', 'wb') as file:
        pickle.dump(top_feattures, file)
    
    ### Hyperparameter Tuning (Structural Tuning Too) for Top Features ###
    with open('../data/top_feattures.pkl', 'rb') as file:
        top_feattures = pickle.load(file)
    top_feattures.append("close")
    best = optimizer.optimize(space, top_feattures)
    print(best)
    with open('../data/best_on_hyperparam_for_top_Feature_Combo_vanilla_transformer_hourly.pkl', 'wb') as file:
        pickle.dump(best, file)

    
    
    
    

