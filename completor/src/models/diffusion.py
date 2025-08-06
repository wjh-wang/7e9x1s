import os
import numpy as np
import pandas as pd
import math
import random
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler


import time as Time
import ipdb


torch.pi = math.pi


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def linear_beta_schedule(timesteps, beta_start, beta_end):
    beta_start = beta_start
    beta_end = beta_end
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def exp_beta_schedule(timesteps, beta_min=0.1, beta_max=10):
    x = torch.linspace(1, 2 * timesteps + 1, timesteps)
    betas = 1 - torch.exp(- beta_min / timesteps - x * 0.5 * (beta_max - beta_min) / (timesteps * timesteps))
    return betas

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)




class diffusion():
    def __init__(self, config):
        self.config = config
        self.timesteps = config['timesteps']
        self.beta_start = config['beta_start']
        self.beta_end = config['beta_end']
        # self.w = w

        if config['beta_sche'] == 'linear':
            self.betas = linear_beta_schedule(timesteps=self.timesteps, beta_start=self.beta_start,
                                              beta_end=self.beta_end)
        elif config['beta_sche'] == 'exp':
            self.betas = exp_beta_schedule(timesteps=self.timesteps)
        elif config['beta_sche'] == 'cosine':
            self.betas = cosine_beta_schedule(timesteps=self.timesteps)
        elif config['beta_sche'] == 'sqrt':
            self.betas = torch.tensor(betas_for_alpha_bar(self.timesteps, lambda t: 1 - np.sqrt(t + 0.0001), )).float()

        # define alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (
                    1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, v_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(v_start)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, v_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, v_start.shape
        )
        return sqrt_alphas_cumprod_t * v_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, denoise_model, v_feat, scenario, t_feat, t, noise=None, loss_type="l2"):

        if noise is None:
            noise_x = torch.randn_like(v_feat)
        v_feat_noisy = self.q_sample(v_start=v_feat, t=t, noise=noise_x)
        predicted_v1, predicted_v2, predicted_v3 = denoise_model(v_feat_noisy, scenario, t_feat, t)

        predicted_v= (predicted_v1 + predicted_v2 + predicted_v3) / 3.0
        if loss_type == 'l1':
            loss = F.l1_loss(v_feat, predicted_v)
        elif loss_type == 'l2':
            loss_x = F.mse_loss(v_feat, predicted_v)
            loss = loss_x 
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(v_feat, predicted_v)
        else:
            raise NotImplementedError()

        return loss, predicted_v

    # def predict_noise_from_start(self, x_t, t, x0):
    #     return (
    #             (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
    #             extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
    #     )

#     @torch.no_grad()
#     def p_sample(self, denoise_model, v_t, p_t, t_t, t, t_index): #x_t→v_t, v_t→p_t
# #         ipdb.set_trace()
#         t = t.cuda()
#         v_start, p_start, t_start= denoise_model(v_t, p_t, t_t, t)
        
#         v_mean=(v_start+ p_start+ t_start)/3.0
#         model_mean_v = (
#                 extract(self.posterior_mean_coef1, t, v_t.shape) * v_mean +
#                 extract(self.posterior_mean_coef2, t, v_t.shape) * v_t
#         )


#         if t_index == 0:
#             return model_mean_v
#         else:
# #             ipdb.set_trace()
#             posterior_variance_t_v = extract(self.posterior_variance, t, v_t.shape)
#             noise_v = torch.randn_like(v_t)

#             return model_mean_v + torch.sqrt(posterior_variance_t_v) * noise_v

#     @torch.no_grad()
#     def sample(self, denoise_model, v_start, p_start, t_start):
# #         ipdb.set_trace()
       
#         noise_v = torch.randn_like(v_start)
# #             noise_t = torch.randn_like(t_start)
# #             noise_v = torch.randn_like(v_start)
#             # noise = torch.randn_like(x_start) / 100
# # 
#         #
# #         ipdb.set_trace()
#         t = torch.tensor([self.timesteps-1] * v_start.shape[0]).to(v_start.device)
# #         x_t = self.q_sample(x_start, t)
#         v_noisy = self.q_sample(v_start=v_start, t=t, noise=noise_v)
#         v_t = v_noisy
#         p_t = p_start
#         t_t = t_start
#         for n in reversed(range(0, self.timesteps)):
#             v_t = self.p_sample(denoise_model, v_t, p_t, t_t,
#                               torch.full((v_t.shape[0],), n, dtype=torch.long), n)
# #         ipdb.set_trace()
#         return v_t