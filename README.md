# The Efficacy of NLinear Neural Network Over the Performer, a Transformer Variant: A Study in Bitcoin Price Prediction

<img width="897" alt="Screenshot 2023-08-20 at 21 15 36" src="https://github.com/HuseynG/Bitcoin_Project/assets/64325152/384abe71-6313-4cf2-a2f2-9fbf4968e1a8">

This repository contains the scripts for dissertation project as part of EECS MSc Project (ECS750P) module (MSc AI - QMUL). Data_Collection and Data_Preparation directories includes script to collect and prepare data. For model training and testing please refer to Prediction_Models directory. The sub-directory called "other_models_to_compare" in Prediction_Models contains the scripts and original source code for LTSF_Linear and Preformer. Lastly, you can refer to Inference directory to run the streamlit app predicting Bitcoin's next hour price similar to the figure above.

Direcotries: 
```.
├── Data_Collection
│   ├── BTC_Price_Data_Collection_Bitstamp.ipynb
│   ├── Glassnode_24h_Data-2012-2023.csv
│   ├── Glassnode_Data_Collection.ipynb
│   ├── Indices_Commodities_Inflation_Interest_rates.ipynb
│   ├── Sentiment_Data_Collection_Crypto.ipynb
│   ├── Sentiment_Data_Collection_FX.ipynb
│   ├── Sentiment_Data_Collection_Stock.ipynb
│   ├── btc_data.csv
│   ├── combined_usa_rates_2012-2023.csv
│   ├── daily_btc_data.csv
│   ├── indices_commodoties_2012-2023.csv
│   ├── raw_news_sentiment_BTC_ETH.pkl
│   ├── raw_news_sentiment_all_stock.pkl
│   └── raw_news_sentiment_general_fx.pkl
├── Data_Preparation
│   ├── Data_Cleaning.ipynb
│   ├── Data_Preprocessing.ipynb
│   ├── Data_Preprocessing_hourly.ipynb
│   ├── Final_data_daily.csv
│   ├── Final_data_hourly.csv
│   ├── daily_news_sentiments.csv
│   ├── glassnode_data.csv
│   └── indices_rates_commodoties.csv
├── Inference
│   ├── app.py
│   ├── data.csv
│   └── helper_util.py
├── LICENSE
├── Prediction_Models
│   ├── encoder_decoder_vanilla_transformer_optimisation_hourly.py
│   ├── encoder_only_vanilla_transformer_optimisation_hourly.py
│   ├── models
│   ├── other_models_to_compare
│   ├── transformer.py
│   ├── utils.py
│   └── vanilla_transformer_experiments_hourly.ipynb
├── README.md
├── data
│   ├── Final_data_daily.csv
│   ├── Final_data_hourly.csv
│   ├── _1m_training_data_close_col_only_for_other_models.csv
│   ├── best_on_hyperparam_for_top_Feature_Combo_vanilla_transformer_hourly.pkl
│   ├── best_on_hyperparam_for_top_Feature_Combo_vanilla_transformer_w_decoder_hourly.pkl
│   ├── btc_data.csv
│   ├── stats_on_features_vanilla_transformer_hourly.pkl
│   ├── stats_on_hyperparam_for_two_cols_vanilla_transformer_hourly_encoder_only.pkl
│   ├── testing_data_close_col_only_for_other_models.csv
│   ├── testing_data_top_features_for_other_models.csv
│   ├── top_feattures.pkl
│   ├── training_data_close_col_only_for_other_models.csv
│   ├── training_data_top_features_for_other_models.csv
│   ├── validation_data_close_col_only_for_other_models.csv
│   └── validation_data_top_features_for_other_models.csv
├── environment.yml
└── requirements.txt
```

#### To avoid repetition Preformer model is integrated to LTSF_Linear source code (LTSF-Linear-main). In addtion, the following scripts are added to LTSF-Linear-main directory:
  - preformer_hyper_parameter_tuning.py
  - preformer_models_btc_data.ipynb
  - train_linear_models_btc_data.ipynb
  - test_linear_models_btc_data.ipynb

#### Contents of /Bitcoin_Project_final/Prediction_Models/other_models_to_compare/LTSF-Linear-main:
```.
├── FEDformer
├── LICENSE
├── LTSF-Benchmark.md
├── Pyraformer
├── README.md
├── checkpoints
├── data_provider
├── environment.yml
├── exp
├── layers
├── logs
├── models
├── pics
├── preformer_hyper_parameter_tuning.py
├── preformer_models_btc_data.ipynb
├── requirements.txt
├── results
├── run_longExp.py
├── run_stat.py
├── scripts
├── test_linear_models_btc_data.ipynb
├── test_results
├── train_linear_models_btc_data.ipynb
├── utils
└── weight_plot.py
```
### Results
#### Performance of the NLinear model on predicting Bitcoin's next closing price using a 12/1 I/O setting, trained exclusively on Bitcoin close prices:
<img width="860" alt="Screenshot 2023-08-20 at 21 17 12" src="https://github.com/HuseynG/Bitcoin_Project/assets/64325152/c9e5c71f-9813-4af0-b53e-85f3bb1d465b">

#### Comparsion of vanilla Transformers on various feature combinations (in 96/1 I/O setting):
<img width="661" alt="Screenshot 2023-08-20 at 21 19 55" src="https://github.com/HuseynG/Bitcoin_Project/assets/64325152/972a081c-42c3-41eb-a49d-c26b940b0e02">

#### Comparsion of all models on various input and output settings when only trained on Bitcoin close price:
<img width="1073" alt="Screenshot 2023-08-20 at 21 22 00" src="https://github.com/HuseynG/Bitcoin_Project/assets/64325152/54aeec13-68e6-49f5-84b1-37b19a330261">

#### Comparsion of all models on various input and output settings when trained on top features and Bitcoin close price:
<img width="1076" alt="Screenshot 2023-08-20 at 21 23 45" src="https://github.com/HuseynG/Bitcoin_Project/assets/64325152/e9867153-c96a-4f8c-bb8c-36dc55fc7154">


