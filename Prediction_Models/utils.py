import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from IPython.display import clear_output
from pprint import pprint

import time
import pandas as pd
from torch.optim import Adam
from torch.nn import MSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from hyperopt import fmin, tpe, Trials, STATUS_OK
from transformer import VanillaTimeSeriesTransformer_EncoderOnly, VanillaTimeSeriesTransformer

def seed_everything(seed=123):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def normalize(df, cols_to_scale, close_col):
    # instantiating two separate scalers
    scaler_general = MinMaxScaler()
    scaler_close = MinMaxScaler()

    # fitting and transform the general scaler
    df_scaled = df.copy()  # creating a copy to avoid changing the original dataframe (in case)

    if len(df.columns) > 1:
        df_scaled[cols_to_scale] = scaler_general.fit_transform(df[cols_to_scale])
    df_scaled[close_col] = scaler_close.fit_transform(df[[close_col]])

    return df_scaled, scaler_general, scaler_close

def create_sequence_of_data(data, input_seq_len, output_seq_len=1, y_col='close', output_as_seq=True):
    Xs = []
    Ys = []

    y_col_position = data.columns.get_loc(y_col)

    # converting DataFrame to numpy array
    data = data.values

    # for i in range(len(data)-input_seq_len-output_seq_len+1):
    for i in range(len(data)-input_seq_len-(output_seq_len if not output_as_seq else 0)):
        x = data[i:(i+input_seq_len)]
        if output_as_seq:
            y = data[(i+input_seq_len):(i+input_seq_len+output_seq_len), y_col_position]
        else:
            # extracting the target output and ensure it is a 1D numpy array
            y = np.atleast_1d(data[(i+input_seq_len+output_seq_len), y_col_position])
        Xs.append(x)
        Ys.append(y)

    return np.array(Xs), np.array(Ys)

def split_data(X, y, train_frac=0.9, val_frac=0.05):
    # indices for the split
    train_index = int(len(X) * train_frac)
    val_index = int(len(X) * (train_frac + val_frac))

    # splitting into train, validation and test set
    X_train, X_val, X_test = X[:train_index], X[train_index:val_index], X[val_index:]
    y_train, y_val, y_test = y[:train_index], y[train_index:val_index], y[val_index:]

    return X_train, X_val, X_test, y_train, y_val, y_test

def preprocess_data(df, batch_size = 32, input_seq_len=24, output_seq_len=1, output_as_seq=True):
    scaled_df, scaler_general, scaler_close = normalize(df, list(df.columns.drop(["close"])), "close")
    X, y = create_sequence_of_data(scaled_df, input_seq_len, output_seq_len, output_as_seq=output_as_seq)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    # print(X_train.shape)
    # print(y_train.shape)

    # converting numpy arrays to pytorch tensors and adding extra dimension at the third postion (index 2)
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).unsqueeze(2).float()

    X_val_tensor = torch.from_numpy(X_val).float()
    y_val_tensor = torch.from_numpy(y_val).unsqueeze(2).float()

    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).unsqueeze(2).float()

    # creating TensorDataset for train, validation and test sets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # defining DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # we want order of sequences to be random
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(X_train_tensor.shape)
    print(y_train_tensor.shape)
    return scaled_df, scaler_general, scaler_close, train_dataloader, val_dataloader, test_dataloader


class HyperParamOptimizer:
    def __init__(self, data_file, model_name, input_seq_len=36, output_seq_len=1, batch_size=256, cuda_='cuda:3'):
        
        self.df = pd.read_csv(data_file)
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.batch_size = batch_size
        self.temp_df = None
        self.cuda_ = cuda_
        self.model_name = model_name
        

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
        #### defining the model ####
        model_current_params = {
            'num_features': int(len(scaled_df.columns)),
            'd_model': d_model, 
            'num_layers': params['num_layers'], 
            'dff': params['dff'], 
            'dropout_rate': params['dropout_rate'], 
            'mlp_size': params['mlp_size'], 
            'mlp_dropout_rate': params['mlp_dropout_rate'],
            "num_heads":num_heads}

        model = eval(self.model_name)(**model_current_params)
        ####
        
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

class Trainer:

    def __init__(self, **kwargs):
        self.model = kwargs.get('model')
        self.train_dataloader = kwargs.get('train_dataloader')
        self.val_dataloader = kwargs.get('val_dataloader')
        self.test_dataloader = kwargs.get('test_dataloader')
        self.criterion = kwargs.get('criterion')
        self.optimiser = kwargs.get('optimiser')
        self.scheduler = kwargs.get('scheduler')
        self.device = kwargs.get('device')
        self.num_epochs = kwargs.get('num_epochs', 100)
        self.early_stopping_patience_limit = kwargs.get('early_stopping_patience_limit', 10)
        self.early_stopping_patience_counter = 0
        self.best_val_loss = float('inf')
        self.best_val_mae  = float('inf')
        self.is_save_model = kwargs.get('is_save_model', False)
        self.scaler = kwargs.get('scaler', None)
        self.file_path = kwargs.get('file_path', "best_model.pt")
        self.is_teacher_forcing = kwargs.get('is_teacher_forcing', False)

    def train_model(self):
        self.model.train() # setting to training mode
        running_loss = 0
        for i, batch in enumerate(self.train_dataloader):
            self.optimiser.zero_grad()
            X, y = batch[0].to(self.device), batch[1].to(self.device)
            # output = self.model(X).unsqueeze(-1)
            if self.is_teacher_forcing:
                output = self.model(X, y=y).unsqueeze(-1)
            else:
                output = self.model(X).unsqueeze(-1)
            loss = self.criterion(output, y)
            loss.backward()
            self.optimiser.step()
            running_loss += loss.item() * X.size(0) # accumulating the loss multiplied by the batch size. This is essentially track the running total of the loss while training
        return running_loss / len(self.train_dataloader.dataset) # getting average loss per training. (deviding the accumlated running loss by the the totla number of training examples)


    def validate_model(self):
        self.model.eval() # setting to evaluation mode
        running_mse = 0
        running_mae  = 0
        with torch.no_grad():
            for i, batch in enumerate(self.val_dataloader):
                X, y = batch[0].to(self.device), batch[1].to(self.device)
                output = self.model(X).unsqueeze(-1)
                loss = self.criterion(output, y)
                mae = torch.abs(output - y).mean()
                running_mse += loss.item() * X.size(0)
                running_mae  += mae.item() * X.size(0) # MAE for this batch
        return running_mse / len(self.val_dataloader.dataset), running_mae / len(self.val_dataloader.dataset)
    

    
    def test_model(self):
        self.model.eval()
        y_preds = []
        y_true  = []
        with torch.no_grad():
            for i, batch in enumerate(self.test_dataloader):
                X, y = batch[0].to(self.device), batch[1].to(self.device)
                output = self.model(X)
                y_preds.append(output.cpu().numpy().squeeze())
                y_true.append(y.cpu().numpy().squeeze())

        # flattening the list of predictions and ground truths
        y_preds = [item for sublist in y_preds for item in sublist]
        y_true = [item for sublist in y_true for item in sublist]

        # reshaping 1D numpy arrays to 2D as scaler expects 2D array as input
        y_preds = np.array(y_preds).reshape(-1, 1)
        y_true = np.array(y_true).reshape(-1, 1)

        if self.scaler:
          # using scaler to inverse transform the data
          y_preds = self.scaler.inverse_transform(y_preds)
          y_true = self.scaler.inverse_transform(y_true)

        # calculate the MSE and MAE
        mse = mean_squared_error(y_true, y_preds)
        mae = mean_absolute_error(y_true, y_preds)
        print(f"Test MSE: {mse}, Test MAE: {mae}")

        # plotting
        plt.plot(y_true, label='True', linewidth=0.5)
        plt.plot(y_preds, label='Predicted', linewidth=0.5)
        plt.title("Model Per. on Test Set")
        plt.legend()
        plt.show()

        return mse, mae

    
    def check_early_stopping(self, vals, is_mae=False):
        if is_mae:
            # self.best_val_mae
            print(f"Current MAE {vals}")
            print(f"Best MAE {self.best_val_mae}")
            if vals < self.best_val_mae:
                self.best_val_mae = vals
                # saving model?
                if self.is_save_model:
                    print(f"Model saved. MAE: {self.best_val_mae}")
                    torch.save(self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict(), self.file_path)

                self.early_stopping_patience_counter = 0  # Reset the counter
            else:
                self.early_stopping_patience_counter += 1  # Increase the counter

            if self.early_stopping_patience_counter >= self.early_stopping_patience_limit:
                print("Early stopping triggered")
                return True

            return False
            
        else:
            if vals < self.best_val_loss:
                self.best_val_loss = vals
                # saving model?
                if self.is_save_model:
                    # print(f"Model saved. VAL LOSS: {self.best_val_loss}")
                    torch.save(self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict(), self.file_path)

                self.early_stopping_patience_counter = 0  # Reset the counter
            else:
                self.early_stopping_patience_counter += 1  # Increase the counter

            if self.early_stopping_patience_counter >= self.early_stopping_patience_limit:
                print("Early stopping triggered")
                return True

            return False
        
    def _plot_losses(self):
      clear_output(True)
      plt.plot(self.train_losses, label='Training Loss')
      plt.plot(self.val_losses, label='Val. Loss')
      plt.legend()
      plt.show()

    def train_loop(self, is_plot=False, is_plot_and_plot_test=False):
        self.train_losses  = []
        self.val_losses = []

        if os.path.exists(self.file_path):
            os.remove(self.file_path)

        for epoch in range(self.num_epochs):
            train_loss = self.train_model()
            val_loss, val_mae = self.validate_model()
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.scheduler.step(val_loss)  # updating scheduler based on the validation loss

            if is_plot:
              
              print(f'Epoch {epoch+1},\n Train Loss: {train_loss},\n Validation Loss: {val_loss},\n Learning Rate: {self.optimiser.param_groups[0]["lr"]}, Validation MAE: {val_mae}')
              self._plot_losses()

            if self.check_early_stopping(val_mae, is_mae=True):
                break  # Break the loop


        # Loading the best model as final output.
        if os.path.exists(self.file_path):
            state_dict = torch.load(self.file_path)
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)

        return self.train_losses, self.val_losses

if __name__ == "__main__":
    print("Executing utils.py as the main file.")