# coding: utf-8

import os 
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='HEALER', help='name of models') 
    parser.add_argument('--dataset', '-d', type=str, default='sports', help='name of datasets')
    
    args, _ = parser.parse_known_args()
    
    config_dict = {
        'learning_rate': 0.001,
        'n_layers': 2,
        'reg_weight': 0.01, 
        'gpu_id': 0,
        'knn_k':5,
        'loss_start_weight': [0.7], 
        'ortho_weight': [0.0001],
        'hyper_parameters': ['loss_start_weight','ortho_weight']
        }

    
    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)


