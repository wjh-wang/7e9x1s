# main.py
import os
import argparse
from utils.quick_start import run_pretrain, run_infer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['pretrain', 'infer'], required=True)
    parser.add_argument('--pretrain_dataset', type=str, default='sports')
    parser.add_argument('--infer_dataset', type=str, default='sports')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--train_mask_ratio', type=float, default=0.4)
    parser.add_argument('--infer_mask_ratio', type=float, default=0.4)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--output_dir', type=str, default='output')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    config_dict = {
        'gpu_id': args.gpu_id,
        'train_mask_ratio': args.train_mask_ratio,
        'infer_mask_ratio': args.infer_mask_ratio,
        'knn_k': 5,
        'timesteps': 20,
        'beta_start': 0.0001,
        'beta_end': 0.02,
        'beta_sche': 'linear',
        'train_batch_size': 128,
        'eval_batch_size': 128,
        'data_base_path': '../data'
    }

    if args.mode == 'pretrain':
        run_pretrain(args.pretrain_dataset, config_dict, args.save_dir)
    else:
        run_infer(args.pretrain_dataset, args.infer_dataset, config_dict, args.save_dir, args.output_dir)

if __name__ == '__main__':
    main()
