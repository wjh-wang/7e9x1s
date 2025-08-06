import os
import numpy as np
import torch
from sklearn.decomposition import PCA

def load_and_process_modal_features(data_dir, device, v_dim=128, t_dim=128):
    v_feat_path = os.path.join(data_dir, 'image_feat.npy')
    t_feat_path = os.path.join(data_dir, 'text_feat.npy')

    if not os.path.exists(v_feat_path) or not os.path.exists(t_feat_path):
        raise FileNotFoundError(f"Missing feature files in {data_dir}")

    # Load npy
    v_feat_np = np.load(v_feat_path, allow_pickle=True)
    t_feat_np = np.load(t_feat_path, allow_pickle=True)

    # Apply PCA
    v_pca = PCA(n_components=v_dim)
    t_pca = PCA(n_components=t_dim)
    v_feat_reduced = v_pca.fit_transform(v_feat_np)
    t_feat_reduced = t_pca.fit_transform(t_feat_np)

    # Convert to tensor and apply tanh
    v_feat = torch.tanh(torch.tensor(v_feat_reduced, dtype=torch.float32, device=device))
    t_feat = torch.tanh(torch.tensor(t_feat_reduced, dtype=torch.float32, device=device))

    return v_feat, t_feat
