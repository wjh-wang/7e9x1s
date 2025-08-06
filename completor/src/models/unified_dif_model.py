import torch
import torch.nn as nn
import os
from models.Udnet import UdNet
from models.diffusion import diffusion

class UnifiedDifModel(nn.Module):
    def __init__(self, config, v_feat, t_feat, mode='pretrain', unet_weight_path=None):
        super().__init__()
        self.config = config
        self.v_feat = v_feat
        self.t_feat = t_feat
        self.knn_k = config['knn_k']
        self.steps = config['timesteps']
        self.mode = mode

        self.device = torch.device(f'cuda:{config["gpu_id"]}' if torch.cuda.is_available() else 'cpu')
        self.v_feat = v_feat.to(self.device)
        self.t_feat = t_feat.to(self.device)

        self.model_cross = UdNet(self.config).to(self.device)
        self.diff = diffusion(self.config)

        self._init_knn_adj()


        

        if mode == 'infer':
            assert unet_weight_path is not None, "Unet weight path must be provided in infer mode."
            self._load_pretrained_unet(unet_weight_path)

    
    
    def _init_knn_adj(self):
        dataset_path = os.path.join(self.config['data_base_path'], self.config['dataset'])
        adj_file = os.path.join(dataset_path, f'mm_adj_{self.knn_k}_t.pt')

        if os.path.exists(adj_file):
            print(f"[Load] Loading cached KNN adj from {adj_file}")
            self.mm_adj_t = torch.load(adj_file).to(self.device).coalesce()
        else:
            print(f"[Build] Building KNN adj on CPU...")

            t_feat_cpu = self.t_feat.cpu()
            context_norm = t_feat_cpu / torch.norm(t_feat_cpu, dim=-1, keepdim=True)
            sim = torch.mm(context_norm, context_norm.T)
            _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
            adj_size = sim.size()
            del sim

            indices0 = torch.arange(knn_ind.shape[0]).unsqueeze(1).expand(-1, self.knn_k)
            indices = torch.stack((indices0.flatten(), knn_ind.flatten()), 0)

            values = torch.ones_like(indices[0], dtype=torch.float32)
            adj = torch.sparse_coo_tensor(indices, values, adj_size)

            row_sum = 1e-7 + torch.sparse.sum(adj, dim=-1).to_dense()
            r_inv_sqrt = torch.pow(row_sum, -0.5)

            coalesced = adj.coalesce()
            idx0, idx1 = coalesced.indices()
            norm_vals = r_inv_sqrt[idx0] * r_inv_sqrt[idx1]
            norm_adj = torch.sparse_coo_tensor(coalesced.indices(), norm_vals, adj_size)

            torch.save(norm_adj, adj_file)
            print(f"[Save] Saved KNN adj to {adj_file}")
            self.mm_adj_t = norm_adj.to(self.device)



    def get_features(self):
        return self.v_feat, self.t_feat

    def _load_pretrained_unet(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.model_cross.load_state_dict(state_dict)
        self.model_cross.eval()
        for param in self.model_cross.parameters():
            param.requires_grad = False

    def save_unet_weights(self, path):
        torch.save(self.model_cross.state_dict(), path)

    def calculate_loss_diff_batch(self, t_feat_online, v_feat_online):
        batch_size = v_feat_online.size(0)
        mask_ratio = self.config['train_mask_ratio'] if self.mode == 'pretrain' else self.config['infer_mask_ratio']

        mask = (torch.rand(batch_size, device=v_feat_online.device) < mask_ratio).float()
        masked_v_feat = v_feat_online * (1 - mask.view(-1, 1))

        t = torch.randint(0, self.steps, (t_feat_online.shape[0] // 2 + 1,), device=self.device)
        t = torch.cat([t, self.steps - t - 1], dim=0)[:t_feat_online.shape[0]]

        # build sampled adj for current batch
        row_idx, col_idx = self.mm_adj_t.indices()
        mask_batch = row_idx < batch_size
        batch_row_idx = row_idx[mask_batch]
        batch_col_idx = col_idx[mask_batch]
        sampled_indices = torch.stack((batch_row_idx, batch_col_idx), 0)
        adj_size = (batch_size, batch_size)
        sampled_adj = torch.sparse_coo_tensor(sampled_indices, torch.ones_like(batch_row_idx).float(), adj_size, device=self.device)

        scenario = t_feat_online
        for _ in range(2):
            scenario = torch.sparse.mm(sampled_adj, scenario)

        diff_loss, predicted_v = self.diff.p_losses(self.model_cross, masked_v_feat, scenario, t_feat_online, t, noise=None, loss_type="l2")

        if self.mode == 'infer':
            predicted_v_masked = predicted_v * mask.view(-1, 1)
            return predicted_v_masked, diff_loss
        else:
            return diff_loss
