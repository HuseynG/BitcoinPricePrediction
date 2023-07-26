# This script mainly follows preprocess_wind.py script. preprocess_wind.py script prepares the wind 
# dataset to be preprocessed by pytorch for Pyraformer Model.
# Similarly, preprocess_full_btc_w_cov.py script prepares the full btc dataset to be preprocessed by 
# pytorch for Pyraformer Model.
# This script is the one with covariate version (i.e., considering time). Whereas, 
# preprocess_full_btc_wo_cov.py does not considers the covariates (time). 


import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
import os

def load_data(datadir):
    df = pd.read_csv(datadir)
    data = (df.values).transpose(1, 0)

    return data

def get_covariates(data_len, start_day):
    """Get covariates"""
    start_timestamp = datetime.timestamp(datetime.strptime(start_day, '%Y-%m-%d %H:%M:%S'))
    timestamps = np.arange(data_len) * 3600 + start_timestamp
    timestamps = [datetime.fromtimestamp(i) for i in timestamps]

    weekdays = stats.zscore(np.array([i.weekday() for i in timestamps]))
    hours = stats.zscore(np.array([i.hour for i in timestamps]))
    months = stats.zscore(np.array([i.month for i in timestamps]))

    covariates = np.stack([weekdays, hours, months], axis=1)

    return covariates

def split_seq(sequences, covariates, seq_length, slide_step, next_ith, save_dir):
    """Divide the training sequence into windows"""
    data_length = len(sequences[0])
    windows = (data_length-seq_length+slide_step) // slide_step
    train_windows = int(0.9 * windows)
    val_windows = int(0.05 * windows)
    test_windows = windows - train_windows - val_windows
    train_data = np.zeros((train_windows*len(sequences), seq_length+next_ith-1, 5), dtype=np.float32)
    val_data = np.zeros((val_windows*len(sequences), seq_length+next_ith-1, 5), dtype=np.float32)
    test_data = np.zeros((test_windows*len(sequences), seq_length+next_ith-1, 5), dtype=np.float32)

    count = 0
    split_start = 0
    seq_ids = np.arange(len(sequences))[:, None]
    end = split_start + seq_length + next_ith - 1
    while end <= data_length:
        if count < train_windows:
            train_data[count*len(sequences):(count+1)*len(sequences), :, 0] = sequences[:, split_start:end]
            train_data[count*len(sequences):(count+1)*len(sequences), :, 1:4] = covariates[split_start:end, :]
            train_data[count*len(sequences):(count+1)*len(sequences), :, -1] = seq_ids
        elif count < train_windows + val_windows:
            val_data[(count-train_windows)*len(sequences):(count-train_windows+1)*len(sequences), :, 0] = sequences[:, split_start:end]
            val_data[(count-train_windows)*len(sequences):(count-train_windows+1)*len(sequences), :, 1:4] = covariates[split_start:end, :]
            val_data[(count-train_windows)*len(sequences):(count-train_windows+1)*len(sequences), :, -1] = seq_ids
        else:
            test_data[(count-train_windows-val_windows)*len(sequences):(count-train_windows-val_windows+1)*len(sequences), :, 0] = sequences[:, split_start:end]
            test_data[(count-train_windows-val_windows)*len(sequences):(count-train_windows-val_windows+1)*len(sequences), :, 1:4] = covariates[split_start:end, :]
            test_data[(count-train_windows-val_windows)*len(sequences):(count-train_windows-val_windows+1)*len(sequences), :, -1] = seq_ids

        count += 1
        split_start += slide_step
        end = split_start + seq_length + next_ith - 1

    os.makedirs(save_dir, exist_ok=True)

    train_data, v = normalize(train_data, seq_length)
    save(train_data, v, save_dir + 'train')
    val_data, v = normalize(val_data, seq_length)
    save(val_data, v, save_dir + 'val')
    test_data, v = normalize(test_data, seq_length)
    save(test_data, v, save_dir + 'test')

def normalize(inputs, seq_length):
    base_seq = inputs[:, :seq_length, 0]
    nonzeros = (base_seq != 0).sum(1)
    inputs = inputs[nonzeros > 0]

    base_seq = inputs[:, :seq_length, 0]
    nonzeros = nonzeros[nonzeros > 0]
    v = np.where(nonzeros != 0, np.abs(base_seq.sum(1)) / nonzeros, 0)
    inputs[:, :, 0] = np.where(v[:, None] != 0, inputs[:, :, 0] / v[:, None], 0)

    return inputs, v



def save(data, v, save_dir):
    np.save(save_dir+'_data_full_btc_w_cov.npy', data)
    np.save(save_dir+'_v_full_btc_w_cov.npy', v)


if __name__ == '__main__':
    datadir = '../../data/Final_data_hourly.csv'
    all_data = load_data(datadir)
    covariates = get_covariates(len(all_data[0]), '2012-01-01 00:00:00') # can check the start date from Data_Preprocessing_hourly.ipynb file
    split_seq(all_data, covariates, 96, 95, 1, 'data/btc_w_cov/')
    print("Done")

    train_data_full_btc_w_cov = np.load('data/btc_w_cov/train_data_full_btc_w_cov.npy')
    train_v_full_btc_w_cov = np.load('data/btc_w_cov/train_v_full_btc_w_cov.npy')

    val_data_full_btc_w_cov = np.load('data/btc_w_cov/val_data_full_btc_w_cov.npy')
    val_v_full_btc_w_cov = np.load('data/btc_w_cov/val_v_full_btc_w_cov.npy')

    test_data_full_btc_w_cov = np.load('data/btc_w_cov/test_data_full_btc_w_cov.npy')
    test_v_full_btc_w_cov = np.load('data/btc_w_cov/test_v_full_btc_w_cov.npy')

    print("train_data_full_btc_w_cov shape:", train_data_full_btc_w_cov.shape)
    print("train_v_full_btc_w_cov shape:", train_v_full_btc_w_cov.shape)
    print("val_data_full_btc_w_cov shape:", val_data_full_btc_w_cov.shape)
    print("val_v_full_btc_w_cov shape", val_v_full_btc_w_cov.shape)
    print("test_data_full_btc_w_cov shape:", test_data_full_btc_w_cov.shape)
    print("test_v_full_btc_w_cov shape:",test_v_full_btc_w_cov.shape)

