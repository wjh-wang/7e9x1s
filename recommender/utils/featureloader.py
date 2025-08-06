import torch
import pickle
import os

def load_inferred_features(pkl_path, device='cpu', dtype=torch.float32):
    """
    Load predicted features from .pkl and convert to torch.Tensor.

    Args:
        pkl_path (str): Path to .pkl file. Supports relative or absolute path.
        device (str): Device to load the tensor to ('cpu' or 'cuda').
        dtype (torch.dtype): Data type of the output tensor.

    Returns:
        torch.Tensor
    """
    # Convert to absolute path (relative to caller file's location)
    if not os.path.isabs(pkl_path):
        caller_dir = os.path.dirname(os.path.abspath(__file__))  # path of featureloader.py
        # From utils to HEALER-main/models or wherever model is, navigate to pkl
        pkl_path = os.path.abspath(os.path.join(caller_dir, '..', pkl_path))

    with open(pkl_path, 'rb') as f:
        np_array = pickle.load(f)

    tensor = torch.from_numpy(np_array).to(dtype=dtype, device=device)
    return tensor
