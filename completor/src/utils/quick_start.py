# utils/quick_start.py
import os
import torch
import pickle
from logging import getLogger
from utils.logger import init_logger
from models.unified_dif_model import UnifiedDifModel
from utils.trainer import Trainer
from utils.featureloader import load_and_process_modal_features
import numpy as np

def load_modal_features(dataset_name, config):
    base_path = config['data_base_path']
    data_dir = os.path.join(base_path, dataset_name)
    device = torch.device(f'cuda:{config["gpu_id"]}' if torch.cuda.is_available() else 'cpu')
    return load_and_process_modal_features(data_dir, device)

def run_pretrain(dataset_name, config_dict, save_dir):
    config_dict['dataset'] = dataset_name
    config_dict['model'] = 'unified_dif'
    config_dict.setdefault('state', 'INFO')

    init_logger(config_dict)
    logger = getLogger()

    v_feat, t_feat = load_modal_features(dataset_name, config_dict)
    model = UnifiedDifModel(config_dict, v_feat, t_feat, mode='pretrain')
    trainer = Trainer(config_dict, model)
    trainer.fit_pretrain()

    weight_path = os.path.join(save_dir, f'{dataset_name}_unet.pth')
    model.save_unet_weights(weight_path)
    logger.info(f'Model weights saved to {weight_path}')

def run_infer(pretrain_dataset, infer_dataset, config_dict, save_dir, output_dir):
    config_dict['dataset'] = infer_dataset
    config_dict['model'] = 'unified_dif'
    config_dict.setdefault('state', 'INFO')

    init_logger(config_dict)
    logger = getLogger()

    v_feat, t_feat = load_modal_features(infer_dataset, config_dict)
    weight_path = os.path.join(save_dir, f'{pretrain_dataset}_unet.pth')

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Pretrained weights not found at {weight_path}")

    model = UnifiedDifModel(config_dict, v_feat, t_feat, mode='infer', unet_weight_path=weight_path)
    trainer = Trainer(config_dict, model)
    predicted_v = trainer.infer()

    output_file = os.path.join(output_dir, f'infer_{infer_dataset}.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(predicted_v.detach().cpu().numpy(), f)
    logger.info(f'Inference result saved to {output_file}')
