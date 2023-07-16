# @Time    : 2023/6/9 4:33 下午
# @Author  : tang
# @File    : Transformer.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer,Decoder, DecoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding,DataEmbedding_wo_pos,DataEmbedding_wo_temp,DataEmbedding_wo_pos_temp
import numpy as np


from models.Diffusion import Diffusion
from models.UNet import UNet



class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.is_diffusion = configs.is_diffusion
        self.is_diff_condition = configs.is_diff_condition
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.sample_num = configs.sample_num

        self.UNet = UNet(configs) # Unet初始化给seq_input_len 的维度

        self.mse = nn.MSELoss()
        self.diffusion = Diffusion(noise_steps=configs.noise_step)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





        # Embedding
        if configs.embed_type == 0:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                            configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        elif configs.embed_type == 1:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 2:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        elif configs.embed_type == 3:
            self.enc_embedding = DataEmbedding_wo_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 4:
            self.enc_embedding = DataEmbedding_wo_pos_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, args=None):
        # 这个enc_self_mask就是batch_y(label+pre)
        enc_input = self.enc_embedding(x_enc, x_mark_enc)
        dec_input = self.dec_embedding(x_dec, x_mark_dec)

        enc_out, attns = self.encoder(enc_input, attn_mask=enc_self_mask)
        if self.is_diffusion:
            if self.is_diff_condition:
                condition = dec_input[:,:self.seq_len,:]
                condition = torch.unsqueeze(condition,dim=1)
            else:
                condition = None

            #diffusion
            enc_out = torch.unsqueeze(enc_out, dim=1)  # 3->4维
            t = self.diffusion.sample_timesteps(enc_out.shape[0]).to(self.device)
            x_t, noise = self.diffusion.noise_forward(enc_out, t)  # x0 -> xt
            predicted_noise = self.UNet(x_t, t, condition=condition)
            diff_loss = self.mse(noise, predicted_noise)

            sample_result = self.diffusion.sample(self.UNet, sample_num= self.sample_num, enc_out=enc_out,condition=condition) #返回10（8x36x512）
            enc_out = sample_result.view(enc_out.shape[0],-1,enc_out.shape[2],enc_out.shape[3])
            enc_out = torch.mean(enc_out,dim=1,keepdim=True)
            #enc_out,_= torch.max(enc_out, dim=1)
            enc_out = torch.squeeze(enc_out, dim=1)
        else:
            #enc_out = enc_out
            diff_loss = 0


        dec_out = self.decoder(dec_input, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :], diff_loss  # [B, L, D]

def Diffusion_Fuc(x):


    return x,loss