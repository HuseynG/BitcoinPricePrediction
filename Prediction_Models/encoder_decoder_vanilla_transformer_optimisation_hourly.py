from collections import OrderedDict
import numpy as np
from hyperopt import hp
from hyperopt.pyll import scope
from utils import seed_everything, HyperParamOptimizer
import pickle

def main():
    seed_everything()
    ##### Vanilla Transformer (Next Step Prediction | Encoder Decoder) ######
    ### Hyperparameter Tuning (Structural Tuning Too) for Top Features ###
    optimizer = HyperParamOptimizer("../data/Final_data_hourly.csv", "VanillaTimeSeriesTransformer")
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

    with open('../data/top_feattures.pkl', 'rb') as file:
        top_feattures = pickle.load(file)
    top_feattures.append("close")
    best = optimizer.optimize(space, top_feattures)
    print(best)
    with open('../data/best_on_hyperparam_for_top_Feature_Combo_vanilla_transformer_w_decoder_hourly.pkl', 'wb') as file:
        pickle.dump(best, file)

if __name__ == '__main__':
    main()

    
    
    
    

