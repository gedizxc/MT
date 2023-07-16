# @Time    : 2023/6/15 9:18 上午
# @Author  : tang
# @File    : Diffusion.py
import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from models.UNet import UNet
import logging

class Diffusion:
    def __init__(self, noise_steps=1, beta_start=1e-4, beta_end=0.02):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.beta = self.prepare_noise_schedule().to(self.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_forward(self, x, t): #正向一步到位
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None,None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None,None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    #
    def sample(self, model, sample_num=1,enc_out = None, condition=None):
        logging.info(  f"Sampling {sample_num} new data....")
        with torch.no_grad():
            sample_shape = enc_out.shape[0]*10
            x = torch.randn((sample_shape,1, enc_out.shape[2],enc_out.shape[3])).to(self.device) #11  从高斯开始采样 #nsample
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0): #扩散步数
                t = (torch.ones(sample_shape) * i).long().to(self.device)
                predicted_noise = model(x, t,condition=condition)
                alpha = self.alpha[t][:, None, None,None]
                alpha_hat = self.alpha_hat[t][:, None, None,None]
                beta = self.beta[t][:, None, None,None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                #逆向最核心的公式，由t推t-1
        x = (x.clamp(-1, 1) + 1) / 2      #压缩（0，1）
        return x

