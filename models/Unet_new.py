# @Time    : 2023/6/15 9:49 上午
# @Author  : tang
# @File    : UNet.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np




class UNet(nn.Module):
    def __init__(self,configs):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.linear1 =nn.Linear(1,configs.seq_len)
        self.linear2 =nn.Linear(1,configs.d_model)
    def forward(self, x,t,condition):
        t = t.unsqueeze(1)
        t = self.linear1(t.float().unsqueeze(2))
        t = self.linear2(t.unsqueeze(3))
        if condition is not None:
            t += condition
        x = x +t
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class UNet_sample(nn.Module):
    def __init__(self,configs):
        super(UNet_sample, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(configs.sample_num, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16,configs.sample_num, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.linear = nn.Linear(1,configs.sample_num)
        self.linear1 =nn.Linear(1,configs.seq_len)
        self.linear2 =nn.Linear(1,configs.d_model)
    def forward(self, x, t, condition):
        t = self.linear(t.float().unsqueeze(1))
        t = self.linear1(t.unsqueeze(2))
        t = self.linear2(t.unsqueeze(3))
        if condition is not None:
            t += condition
        x = x +t

        x = self.encoder(x)
        x = self.decoder(x)
        return x