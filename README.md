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


## Instructions to run scripts:
You can download full project source code, prepared dataset, trained models, and project paper from the following Link: https://qmulprod-my.sharepoint.com/:f:/r/personal/ec22344_qmul_ac_uk/Documents/Dissertation_Contents_Shared?csf=1&web=1&e=DL0qJ0

### Step 0: Prerequisites & Environment Setup
- NVIDIA GPU is required with minimum VRAM of 8Gb. 
- Anaconda or Mini Conda is available
#### Setting Up Environments
##### Setting Up Main Environment: bitcoinproject1
  1. Navigate to the project root directory where environment.yml file exists
  2. Create a conda environment using the environment.yml file: 
  ```conda env create -f environment.yml```
  3. Activate the newly created environment: 
  ```conda activate bitcoinproject1```

#### Setting Up Main Environment: LTSF_Linear
  1. Navigate to the Bitcoin_Project_final/Prediction_Models/other_models_to_compare/LTSF-Linear-main/ directory where environment.yml exists
  2. Create a conda environment using the environment.yml file: 
  ```conda env create -f environment.yml```
  3. Activate the newly created environment: 
  ```conda activate LTSF_Linear```

Note: You might need to modify prefix in environment.yml files accordingly. 

### Step 1: Data Collection
Navigate to Data_Collection directory:
- To collect Bitcoin OHCL data run the script called "BTC_Price_Data_Collection_Bitstamp.ipynb"
- To collect Glassnode data run the script called "Glassnode_Data_Collection.ipynb"
- To collect sentiment data run the following scripts: Sentiment_Data_Collection_Crypto.ipynb, Sentiment_Data_Collection_FX.ipynb, Sentiment_Data_Collection_Stock.ipynb
- Lastly, to collect commodities and inflation rates run Indices_Commodities_Inflation_Interest_rates.ipynb notebook.

Note: Some of these scripts require API key/token. Make sure you provide and paste your keys/tokens properly into the scripts.

### Step 2: Data Preparation
Run the following scripts located in the Data_Preparation directory for the data cleaning and preparation process:
- Data_Cleaning.ipynb
- Data_Preprocessing.ipynb
- Data_Preprocessing_hourly.ipynb

### Step 3: Model Hyperparameters Tunings
#### Tuning Base Models
Navigate to the Prediction_Models directory and run the following scripts:
- encoder_only_vanilla_transformer_optimisation_hourly.py
- encoder_decoder_vanilla_transformer_optimisation_hourly.py

#### Tuning Preformer
Navigate to Prediction_Models/other_models_to_compare/LTSF-Linear-main directory and run the following script:
- preformer_hyper_parameter_tuning.py

Note: For LTSF_Linear and Preformer models make sure the LTSF_Linear conda environment is always active. Next before running the optimisation script for the Preformer, run the notebook called data_prep.ipynb (located in Prediction_Models/other_models_to_compare) which will prepare Training, Validation and Testing sets for tuning, training and testing.

### Step 4: Model Training and Testing
Note: For the base models please ensure that bitcoinproject1 conda environment is activated. For LTSF_Linear and Preformer models make sure the LTSF_Linear conda environment is active as well.

#### Base Models Training and Testing
Navigate to the Prediction_Models directory and run the following script:
- vanilla_transformer_experiments_hourly.ipynb

#### Preformer Training and Testing
Navigate to Prediction_Models/other_models_to_compare/LTSF-Linear-main directory and run the following script:
- preformer_models_btc_data.ipynb

#### DLinear and NLinear Model Training and Testing
Navigate to Prediction_Models/other_models_to_compare/LTSF-Linear-main directory and run the following script:
- train_linear_models_btc_data.ipynb
- test_linear_models_btc_data.ipynb

### Step 5: Running Streamlit Applications (Predicting next hour's Bitcoin close price)
If you would like to run the demo Streamlit application, activate the LTSF_Linear environment. Next, make sure you have installed the Streamlit python package properly. You can install it using the following command:

  ```pip install streamlit```

Next, navigate to Inference directory and run the following command:

  ```streamlit run app.py```
