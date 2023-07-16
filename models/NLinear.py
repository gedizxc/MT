# @Time    : 2023/6/10 11:28 上午
# @Author  : tang
# @File    : NLinear.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.Diffusion import Diffusion
from models.UNet import UNet

class Model(nn.Module):
    """
    Normalization-Linear
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.is_diffusion = configs.is_diffusion
        self.is_diff_condition = configs.is_diff_condition
        self.sample_num = configs.sample_num
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model

        self.UNet = UNet(configs)  # Unet初始化给seq_input_len 的维度

        self.mse = nn.MSELoss()
        self.diffusion = Diffusion(noise_steps=configs.noise_step)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = configs.enc_in
        self.individual = configs.individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

        self.input = nn.Linear(self.enc_in,self.d_model)
        self.out = nn.Linear(self.d_model,self.enc_in)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x= self.input(x)
        origin_x = x
        #diffusion
        x = torch.unsqueeze(x, dim=1)  # 3->4维
        t = self.diffusion.sample_timesteps(x.shape[0]).to(self.device)
        x_t, noise = self.diffusion.noise_forward(x, t)  # x0 -> xt
        predicted_noise = self.UNet(x_t, t, condition=None)
        diff_loss = self.mse(noise, predicted_noise)
        sample_result = self.diffusion.sample(self.UNet, sample_num=self.sample_num, enc_out=x,
                                              condition=None)
        x = sample_result.view(x.shape[0], -1, x.shape[2], x.shape[3])
        x = torch.mean(x, dim=1, keepdim=True)
        # enc_out,_= torch.max(enc_out, dim=1)
        x = torch.squeeze(x, dim=1)
        #x = x +origin_x

        #pre
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last

        x= self.relu(x)
        x = self.out(x)

        return x,diff_loss  # [Batch, Output length, Channel]