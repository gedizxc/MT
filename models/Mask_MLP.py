# @Time    : 2023/6/10 3:51 下午
# @Author  : tang
# @File    : Mask_MLP.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
def Mask_MLP(x):
    # x: [Batch, Input length, Channel]
    mask_x, gt_x = mask_func(x)
    mask_len = 3
    seq_len = x.shape[1]

    Linear = nn.Linear(seq_len,mask_len)

    x = Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
    criterion = nn.MSELoss()
    mask_loss = criterion(x,gt_x)


    return mask_loss  # [Batch, Output length, Channel]



def randint_generation(min, max, mount):
    list = []
    while len(list) != mount:
        unit = random.randint(min, max)
        if unit not in list:
            list.append(unit)
    return list

def mask_func(x):    #每个batchsize mask3行时间戳
    Mask_Matrix = torch.ones_like(x)
    ground_truth = torch.zeros(x.shape[0], 3, x.shape[-1])

    for i in range(Mask_Matrix.size(0)):
        index1, index2, index3 = randint_generation(0,
                                                    Mask_Matrix.size(1) - 1, 3)
        ground_truth[i, 0, :] = x[i, index1, :]
        ground_truth[i, 1, :] = x[i, index2, :]
        ground_truth[i, 2, :] = x[i, index3, :]

        Mask_Matrix[i, index1, :] = -1  # mask处置为-1
        Mask_Matrix[i, index2, :] = -1
        Mask_Matrix[i, index3, :] = -1
    Mask_x = x * Mask_Matrix
    return Mask_x, ground_truth

