import copy
import tqdm
import torch
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from torch import nn
from math import ceil
from pathlib import Path
from torch.nn.init import trunc_normal_ as __call_trunc_normal_

tot = 0

def kaiming_normal_(tensor):
    torch.nn.init.kaiming_normal_(
        tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std)  # , a=-std, b=std)


def calc_sim(x, y, tau):
    return torch.exp(F.cosine_similarity(x, y, dim=0)/tau)


class conv_block(nn.Module):
    def __init__(self, in_c, out_c, layer=1, dropout=0):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential()
        dim = in_c

        lst = []
        if layer == 2:
            if in_c < out_c:
                lst.append(out_c//2)
                lst.append(out_c)
            else:
                lst.append(in_c//2)
                lst.append(out_c)
        else:
            lst.append(out_c)

        for p in range(layer):
            to_dim = lst[p]
            self.conv.add_module("conv-{}".format(p), nn.Sequential(
                nn.Conv1d(dim, to_dim, 1,
                          stride=1, padding=0),
                nn.Dropout(dropout),  
                nn.BatchNorm1d(to_dim),  
                nn.GELU(),
            ))
            dim = to_dim

    def forward(self, x):
        return self.conv(x)


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha=1):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class projection_mlp(nn.Module):
    def __init__(self, in_dim, out_dim, layer=1, dropout=0):
        super(projection_mlp, self).__init__()
        self.mlp = nn.Sequential()
        dim = in_dim

        lst = []
        if layer == 2:
            if in_dim < out_dim:
                lst.append(int(out_dim//2))
                lst.append(out_dim)
            else:
                lst.append(int(in_dim//2))
                lst.append(out_dim)
        else:
            lst.append(out_dim)
        for p in range(0, layer):
            to_dim = lst[p]
            self.mlp.add_module("projection-{}".format(p), nn.Sequential(
                nn.Linear(dim, to_dim),
                nn.Dropout(dropout),
                nn.BatchNorm1d(to_dim),
                nn.GELU(),
            ))
            dim = to_dim

    def forward(self, x):
        return self.mlp(x)


class DASS_encoder(nn.Module):
    def __init__(self, n_channel, conv_hidden_dim, trans_hidden_dim, n_layer, heads,
                 dropout=0, layer_drop=0, init_method='xavier', mlp_layer=1):
        super(DASS_encoder, self).__init__()
        self.dropout = dropout
        self.layer_drop = layer_drop
        self.conv = conv_block(n_channel, conv_hidden_dim,
                               layer=mlp_layer, dropout=dropout)
        self.first_token = nn.Parameter(torch.zeros(
            1, conv_hidden_dim, 1), requires_grad=True)
        self.channel_emb = nn.Parameter(torch.zeros(
            1, conv_hidden_dim, 1), requires_grad=True)
        self.seq_emb = nn.Parameter(
            torch.zeros(1, 1, 62+1), requires_grad=True)
        transformerlayer = nn.TransformerEncoderLayer(d_model=conv_hidden_dim, nhead=heads, dim_feedforward=trans_hidden_dim,
                                                      dropout=dropout, activation='gelu')
        self.transformer_layers = nn.ModuleList(
            [copy.deepcopy(transformerlayer) for _ in range(n_layer)])

        trunc_normal_(self.first_token)
        trunc_normal_(self.channel_emb)
        trunc_normal_(self.seq_emb)

        self.tot = 0

    def forward(self, x, M):  # (batch_size,channel,seq)
        batch_size, channel, seq = x.size()
        x = self.conv(x)  # (batch_size,conv_hidden_dim,seq)

        batch_size, dim, seq = x.size()

        cls_token = self.first_token.expand(
            batch_size, -1, -1)  # (batch_size,conv_hidden_dim,1)
        x = torch.cat((cls_token, x), dim=2)
        assert (x.size()[2] == seq+1)
        channel_emb_used = self.channel_emb[:, :dim, :].expand(
            batch_size, -1, seq)  # no seq+1
        seq_emb_used = self.seq_emb[:, :, :seq+1].expand(batch_size, dim, -1)
        x = x+seq_emb_used
        x[:, :, 1:] += channel_emb_used

        x = x.permute([2, 0, 1])  # (seq+1,batch_size,channel)

        for layer in self.transformer_layers:
            if not self.training or torch.rand(1) > self.layer_drop:
                x = layer(x, src_key_padding_mask=M.bool())

        return x.permute([1, 2, 0])  # (batch_size,channel,seq+1)


class DASS_body(nn.Module):
    def __init__(self, n_sub, n_channel, conv_hidden_dim, trans_hidden_dim, n_layer, heads,
                 gamma=0.1, dropout=0, layer_drop=0, init_std=0.1, init_method='xavier', mlp_layer=1):
        super(DASS_body, self).__init__()
        self.reconstruct_conv = conv_block(
            conv_hidden_dim, n_channel, layer=mlp_layer, dropout=0)  # or dropout=0??
        self.subject_encoder = DASS_encoder(n_channel, conv_hidden_dim, trans_hidden_dim, n_layer, heads,
                                            dropout, layer_drop, init_method, mlp_layer)
        self.general_encoder = DASS_encoder(n_channel, conv_hidden_dim, trans_hidden_dim, n_layer, heads,
                                            dropout, layer_drop, init_method, mlp_layer)
        self.init_std = init_std
        self.init_method = init_method
        
    def forward(self, X, M):
        x_sub = self.subject_encoder(X, M)
        # (batch_size,conv_hidden_dim,seq+1)
        x_gen = self.general_encoder(X, M)
        x_sub = x_sub.permute([1, 0, 2])
        x_gen = x_gen.permute([1, 0, 2])
        x_sub = x_sub*(1-M)
        x_gen = x_gen*(1-M)
        x_sub = x_sub.permute([1, 0, 2])
        x_gen = x_gen.permute([1, 0, 2])

        if self.training:
            rec_fea = self.reconstruct_conv(x_sub[:, :, 1:]+x_gen[:, :, 1:])
            return x_sub, x_gen, rec_fea
        else:
            return x_sub, x_gen

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename, freeze=False, strict=True, device=None):
        state_dict = torch.load(filename, map_location=lambda storage, loc: storage.cuda(
            device) if device is not None else None)
        self.load_state_dict(state_dict, strict=strict)


class DASS_module(nn.Module):
    def __init__(self, n_sub, n_channel, conv_hidden_dim, trans_hidden_dim, n_layer, heads,
                 gamma=0.1, dropout=0, layer_drop=0, init_std=0.1, init_method='xavier', mlp_layer=1, tau=0.2):
        super(DASS_module, self).__init__()

        self.body = DASS_body(n_sub, n_channel, conv_hidden_dim, trans_hidden_dim, n_layer, heads,
                              gamma, dropout, layer_drop, init_std, init_method, mlp_layer)
        self.gamma = gamma
        self.init_std = init_std
        self.init_method = init_method
        self.apply(self._init_para)
        self.tau = tau

    def _init_para(self, m):
        if isinstance(m, nn.Linear):
            kaiming_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            kaiming_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _calc_contrastive_loss(self, emb, S):
        L2_emb = torch.norm(emb, dim=1, p=2, keepdim=True)
        norm_emb = emb/L2_emb
        cross_mat = torch.matmul(norm_emb, norm_emb.t())
        cross_mat = torch.exp(cross_mat/self.tau)
        eye = torch.eye(cross_mat.shape[0]).to(emb.device)
        S_add_dim = torch.unsqueeze(S, dim=0)
        equal = (S_add_dim == S_add_dim.t()).int().to(emb.device)
        pos = torch.sum(cross_mat*(equal-eye))
        neg = torch.sum(
            cross_mat*(torch.ones_like(cross_mat).to(emb.device)-equal))
        return -torch.log(pos/(pos+neg))

    def forward(self, X, S, M, alpha=0):  # (batch_size,channel,seq)
        if self.training:
            x_sub, x_gen, rec_fea = self.body(X, M)
            sub_emb = x_sub[:, :, 0]
            gen_emb = x_gen[:, :, 0]
            loss_sub = self._calc_contrastive_loss(sub_emb, S)
            loss_gen = self._calc_contrastive_loss(gen_emb, S)
            return sub_emb, gen_emb, rec_fea, loss_sub, ReverseLayerF.apply(loss_gen, alpha)
        else:
            x_sub, x_gen = self.body(X, M)
            sub_emb = x_sub[:, :, 0]
            gen_emb = x_gen[:, :, 0]
            return sub_emb, gen_emb

    def save_body(self, filename):
        self.body.save(filename=filename)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename, freeze=False, strict=True, device=None):
        state_dict = torch.load(filename, map_location=lambda storage, loc: storage.cuda(
            device) if device is not None else None)
        self.load_state_dict(state_dict, strict=strict)

    def load_body(self, encoder_file, freeze=False, strict=True, device=None):
        self.body.load(filename=encoder_file, device=device)


class DASS(nn.Module):
    def __init__(self, n_class, n_sub, n_channel, conv_hidden_dim, trans_hidden_dim, n_layer, heads,
                 gamma=0.1, dropout=0, layer_drop=0, init_std=0.1, init_method='xavier', mlp_layer=1, tau=0.2):
        super(DASS, self).__init__()
        self.DASS_m = DASS_module(n_sub, n_channel, conv_hidden_dim, trans_hidden_dim, n_layer, heads,
                                  gamma, dropout, layer_drop, init_std, init_method, mlp_layer, tau)
        self.predict_mlp = projection_mlp(
            2*conv_hidden_dim, n_class, mlp_layer, dropout)
        self.gamma = gamma
        self.init_std = init_std
        self.init_method = init_method
        self.apply(self._init_para)
        self.n_class = n_class
        self.n_sub = n_sub

    def _init_para(self, m):
        if isinstance(m, nn.Linear):
            kaiming_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename, freeze=False, strict=True, device=None):
        state_dict = torch.load(filename, map_location=lambda storage, loc: storage.cuda(
            device) if device is not None else None)
        self.load_state_dict(state_dict, strict=strict)

    def load_DASS_m(self, encoder_file, freeze=False, strict=True, device=None):
        self.DASS_m.load(encoder_file, strict=strict, device=device)

    def load_DASS_body(self, encoder_file, freeze=False, strict=True, device=None):
        self.DASS_m.body.load(encoder_file, freeze, strict, device=device)

    def forward(self, X, S, M, alpha=0):
        if self.training:
            sub_emb, gen_emb, rec_fea, loss_sub, loss_gen = self.DASS_m(
                X, S, M, alpha)
            emotion_predict = self.predict_mlp(
                torch.cat((sub_emb, gen_emb), 1))
            return sub_emb, gen_emb, rec_fea, loss_sub, loss_gen, emotion_predict
        else:
            sub_emb, gen_emb = self.DASS_m(X, S, M)
            emotion_predict = self.predict_mlp(
                torch.cat((sub_emb, gen_emb), 1))
            return sub_emb, gen_emb, emotion_predict
