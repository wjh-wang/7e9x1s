# coding: utf-8
# 

import os
import numpy as np
import scipy.sparse as sp
import torch    
import pickle
import math
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
import torch_geometric
from tqdm import tqdm
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from utils.featureloader import load_inferred_features

class HEALER(GeneralRecommender):
    def __init__(self, config, dataset):
        super(HEALER, self).__init__(config, dataset)

        num_user = self.n_users
        num_item = self.n_items
        batch_size = config['train_batch_size']         
        dim_x = config['embedding_size']                
        self.n_layers = 2       
        self.knn_k = config['knn_k']                    
        self.mm_image_weight = config['mm_image_weight']
        has_id = True
        self.dropout = nn.Dropout(p=0.3)
        self.batch_size = batch_size
        self.num_user = num_user                       
        self.num_item = num_item                       
        self.k = 40
        self.num_interest = 3
        self.aggr_mode = config['aggr_mode']           
        self.user_aggr_mode = 'softmax'
        self.num_layer = 1
        self.dataset = dataset
        self.construction = 'cat'
        self.reg_weight = config['reg_weight']        
        self.drop_rate = 0.1
        self.v_rep = None
        self.t_rep = None
        self.a_rep = None

        self.dropout_prob = 0.1
        self.dim_latent = 384
        self.mm_adj = None

        self.adapter = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(256, self.dim_latent),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob)
        )
  
        nn.init.xavier_normal_(self.adapter[0].weight)  
        nn.init.xavier_normal_(self.adapter[3].weight)
       

        
        if self.v_feat is not None:
            self.v_feat = nn.Embedding.from_pretrained(self.v_feat, freeze=False).weight         
        if self.t_feat is not None:
            self.t_feat = nn.Embedding.from_pretrained(self.t_feat, freeze=False).weight
        if self.a_feat is not None:
            self.a_feat = nn.Embedding.from_pretrained(self.a_feat, freeze=False).weight

        pkl_relative_path = '../completor/src/output/infer_sports.pkl'
        self.predictedv_embedding = load_inferred_features(pkl_relative_path, device=self.device)
        
        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)                     
        
        self.ui_graph = self.matrix_to_tensor(self.csr_norm(train_interactions, mean_flag=False))
        self.iu_graph = self.matrix_to_tensor(self.csr_norm(train_interactions.T, mean_flag=False))
 
        self.disentangler=DisentangleNet(feat_dim=self.dim_latent)
        self.loss_start_weight=config['loss_start_weight']
        self.ortho_weight=config['ortho_weight']
        self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
            np.random.randn(num_user, self.dim_latent), dtype=torch.float32, requires_grad=True),
            gain=1).to(self.device))
       
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])

        
        mm_adj_file_t = os.path.join(dataset_path, 'mm_adj_{}_t.pt'.format(self.knn_k)) 
        if os.path.exists(mm_adj_file_t):
            self.mm_adj_t = torch.load(mm_adj_file_t)                                           #(7050,7050)
        else:
            if self.t_feat is not None:
                indices, text_adj = self.get_knn_adj_mat(self.t_feat.detach())
                self.mm_adj_t = text_adj
                del text_adj
            torch.save(self.mm_adj_t, mm_adj_file_t)
       
    def cs(self,A,B):
        dot_product=torch.mm(A.half(),B.t().half())
        norm_A=torch.norm(A,dim=1,keepdim=True)
        norm_B=torch.norm(B,dim=1,keepdim=True)
        similarity_matrix=dot_product/(norm_A*norm_B.t())
        return similarity_matrix     
            
    def pca(self, x, k=2):
        x_mean = torch.mean(x, 0)
        x = x - x_mean
        cov_matrix = torch.matmul(x.t(), x) / (x.size(0) - 1)
        eigenvalues, eigenvectors = torch.linalg.eig(cov_matrix)
        sorted_eigenvalues, indices = torch.sort(eigenvalues.real, descending=True)
        components = eigenvectors[:, indices[:k]]
        x_pca = torch.matmul(x, components)

        return x_pca
    
    def matrix_to_tensor(self, cur_matrix):
        if type(cur_matrix) != sp.coo_matrix:
            cur_matrix = cur_matrix.tocoo()  
        ui_indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))  
        iu_indices = torch.from_numpy(np.vstack((cur_matrix.col, cur_matrix.row)).astype(np.int64))
        values = torch.from_numpy(cur_matrix.data)  
        shape = torch.Size(cur_matrix.shape)

        return torch.sparse.FloatTensor(ui_indices, values, shape).to(torch.float32).cuda()

    def csr_norm(self, csr_mat, mean_flag=False):
        rowsum = np.array(csr_mat.sum(1))
        rowsum = np.power(rowsum+1e-8, -0.5).flatten()
        rowsum[np.isinf(rowsum)] = 0.
        rowsum_diag = sp.diags(rowsum)

        colsum = np.array(csr_mat.sum(0))
        colsum = np.power(colsum+1e-8, -0.5).flatten()
        colsum[np.isinf(colsum)] = 0.
        colsum_diag = sp.diags(colsum)

        if mean_flag == False:
            return rowsum_diag*csr_mat*colsum_diag
        else:
            return rowsum_diag*csr_mat
    def mm(self, x, y): 
        return torch.sparse.mm(x, y)
    

        
    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        return indices, self.compute_normalized_laplacian(indices, adj_size)
    
    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    
    def pre_epoch_processing(self):
        self.epoch_user_graph, self.user_weight_matrix = self.topk_sample(self.k)   
        self.user_weight_matrix = self.user_weight_matrix.to(self.device)

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        return np.column_stack((rows, cols))

    
    def _gcn_pp(self, item_embed, user_embed, uig, iug, norm=False):
        if norm == True:
            item_res =  item_embed = F.normalize(item_embed)
            user_res =  user_embed = F.normalize(user_embed)
            for _ in range(2):
                user_agg = self.mm(uig, item_embed)   
                item_agg = self.mm(iug, user_embed)
                item_embed = item_agg
                user_embed = user_agg
                
                item_res = item_res + item_embed
                user_res = user_res + user_embed
        
        else:
            item_res =  item_embed = F.normalize(item_embed)
            user_res =  user_embed = F.normalize(user_embed)
            for _ in range(2):
                user_agg = self.mm(uig, item_embed)
                item_agg = self.mm(iug, user_embed)
                item_embed = F.normalize(item_agg)
                user_embed = F.normalize(user_agg)


                item_res = item_res + item_embed
                user_res = user_res + user_embed
        
        
        return user_res, item_res



    def forward(self):

        t_feat = self.t_feat
        v_feat = self.v_feat
        a_feat = self.a_feat
        
    
        scenario = self.t_feat 
        for i in range(2):
            scenario = torch.sparse.mm(self.mm_adj_t, scenario)  
     
        v_feat_predict=self.predictedv_embedding 
        mask_v = torch.all(v_feat_predict == 0, dim=1)
        expanded_mask = mask_v.view(-1, 1).repeat(1, self.dim_latent)
        v_feat_original= v_feat * expanded_mask
        
        v_feat_predict_adapt=self.adapter(v_feat_predict) 
        subject, background =self.disentangler(v_feat_predict_adapt)
        
        v_feat_all=v_feat_original + subject
        item_features = t_feat + v_feat_all
        user_features = self.preference
        user_embed,item_embed=self._gcn_pp(item_features,user_features,self.ui_graph,self.iu_graph,norm=True)
          
        indices, image_adj_generate = self.get_knn_adj_mat(v_feat_all.detach())      
        mm_adj_update= 0.3 * image_adj_generate + 0.7 * self.mm_adj_t
    
        h = item_embed
        for i in range(self.n_layers):
            h = torch.sparse.mm(mm_adj_update, h)                                          
        item_embed = item_embed + h
      
        return user_embed, item_embed, subject, background, scenario, v_feat_all
             
       

       
    def _sparse_dropout(self, x, rate=0.0):
        noise_shape = x._nnz()                                      

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)       
        dropout_mask = torch.floor(random_tensor).type(torch.bool)   
        i = x._indices()                                            
        v = x._values()                                             

        i = i[:, dropout_mask]                                      
        v = v[dropout_mask]                                        

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)                           
        return out


    def bpr_loss(self, interaction):
        
        user_embed, item_embed,subject,background, item_start, v_feat_all = self.forward()
        
        user_nodes, pos_item_nodes, neg_item_nodes = interaction[0], interaction[1], interaction[2]


        user_embed = user_embed[user_nodes]
        pos_item_embed = item_embed[pos_item_nodes]
        neg_item_embed = item_embed[neg_item_nodes]
        pos_scores = torch.sum(torch.mul(user_embed, pos_item_embed), dim=1)
        neg_scores = torch.sum(torch.mul(user_embed, neg_item_embed), dim=1)


        regularizer = 1./2*(user_embed**2).sum() + 1./2*(pos_item_embed**2).sum() + 1./2*(neg_item_embed**2).sum()        
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = 1e-4 * regularizer
        loss_sem = F.mse_loss(item_start, v_feat_all)
        
        
        ortho_loss=torch.norm(torch.mm(subject.T,background),p='fro')**2/(background.shape[0]**2)
        loss = mf_loss + emb_loss + self.loss_start_weight*loss_sem + self.ortho_weight*ortho_loss
        return loss
    

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_scores, neg_scores = self.forward(interaction)
        loss_value = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
        reg_embedding_loss = (self.preference[user] ** 2).mean() if self.preference is not None else 0.0

        reg_loss = self.reg_weight * reg_embedding_loss
        if self.construction == 'weighted_sum':
            reg_loss += self.reg_weight * (self.weight_u ** 2).mean()
            reg_loss += self.reg_weight * (self.weight_i ** 2).mean()
        elif self.construction == 'cat':
            reg_loss += self.reg_weight * (self.weight_u ** 2).mean()
        elif self.construction == 'cat_mlp':
            reg_loss += self.reg_weight * (self.MLP_user.weight ** 2).mean()
        return loss_value + reg_loss


    def full_sort_predict(self, interaction):
        user_tensor, item_tensor,__,__,__,__= self.forward()
        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix
    

class User_Graph_sample(torch.nn.Module):
    def __init__(self, num_user, aggr_mode,dim_latent):
        super(User_Graph_sample, self).__init__()
        self.num_user = num_user
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode

    def forward(self, features,user_graph,user_matrix):
        index = user_graph
        u_features = features[index]           
        user_matrix = user_matrix.unsqueeze(1) 
        u_pre = torch.matmul(user_matrix,u_features)
        u_pre = u_pre.squeeze()
        return u_pre                    

    
class DisentangleNet(torch.nn.Module):
    def __init__(self, feat_dim=384):
        super().__init__()
 
        self.subject_net = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512, feat_dim)
        )
        self.background_net = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.Tanh(),
            nn.Linear(256, feat_dim)
        )

    def forward(self, feat):
        subject=self.subject_net(feat)
        background=self.background_net(feat)
      
        return subject, background



class Base_gcn(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
        super(Base_gcn, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr                  
        self.in_channels = in_channels    
        self.out_channels = out_channels  

    def forward(self, x, edge_index, size=None):
        if size is None:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = x.unsqueeze(-1) if x.dim() == 1 else x  
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        if self.aggr == 'add':
            row, col = edge_index
            deg = degree(row, size[0], dtype=x_j.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return norm.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)
    
    

