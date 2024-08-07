# DASS:Disentangled Autoencoder Self-Supervised Learning for EEG Emotion Recognition
import copy
# import parse
import tqdm
import torch
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from torch import nn
from math import ceil
from pathlib import Path
from torch.nn.init import trunc_normal_ as __call_trunc_normal_
from dgcnn import DGCNN

tot=0

def kaiming_normal_(tensor):
    torch.nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std)# , a=-std, b=std)

def calc_sim(x,y,tau):
    return torch.exp(F.cosine_similarity(x, y, dim=0)/tau)

class Linear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        nn.init.xavier_normal_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)

class conv_block(nn.Module):
    def __init__(self,in_c,out_c,layer=1,dropout=0):
        super(conv_block, self).__init__()
        self.conv=nn.Sequential()
        dim=in_c
        
        lst=[]
        if layer==2:
            if in_c<out_c:
                lst.append(out_c//2)
                lst.append(out_c)
            else:
                lst.append(in_c//2)
                lst.append(out_c)
        else:
            lst.append(out_c)
            
        for p in range(layer):
            to_dim=lst[p]
            self.conv.add_module("conv-{}".format(p), nn.Sequential(
                    nn.Conv1d(dim, to_dim, 1,
                            stride=1, padding=0),
                    nn.Dropout(dropout),  # changed from dropout2d to dropout1d
                    #nn.GroupNorm(out_c // 2, out_c), #BatchNorm1d or GroupNorm?
                    nn.BatchNorm1d(to_dim), #BatchNorm1d or GroupNorm?
                    nn.GELU(),
                ))
            dim=to_dim
    def forward(self,x):
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
    def __init__(self,in_dim,out_dim,layer=1,dropout=0):
        super(projection_mlp, self).__init__()
        self.mlp = nn.Sequential()
        dim=in_dim
        
        lst=[]
        if layer==2:
            if in_dim<out_dim:
                lst.append(int(out_dim//2))
                lst.append(out_dim)
            else:
                lst.append(int(in_dim//2))
                lst.append(out_dim)
        else:
            lst.append(out_dim)
        for p in range(0, layer ):
            to_dim=lst[p]
            self.mlp.add_module("projection-{}".format(p), nn.Sequential(
                nn.Linear(dim, to_dim),
                nn.Dropout(dropout),
                #nn.GroupNorm(out_dim // 2, out_dim),
                nn.BatchNorm1d(to_dim),
                nn.GELU(),
            ))
            dim=to_dim
    def forward(self,x):
        # print(x.shape)
        # print(self.mlp)
        return self.mlp(x)

class DASS_encoder(nn.Module):
    def __init__(self,n_channel,conv_hidden_dim,trans_hidden_dim,n_layer,heads,
                 dropout=0,layer_drop=0,init_method='xavier',mlp_layer=1):
        super(DASS_encoder, self).__init__()
        self.dropout=dropout
        self.layer_drop=layer_drop
        self.conv=conv_block(n_channel,conv_hidden_dim,layer=mlp_layer,dropout=dropout)
        self.first_token=nn.Parameter(torch.zeros(1, conv_hidden_dim,1), requires_grad=True)
        self.channel_emb=nn.Parameter(torch.zeros(1, conv_hidden_dim,1), requires_grad=True)
        self.seq_emb=nn.Parameter(torch.zeros(1, 1, 62+1), requires_grad=True)
        
        self.gnn=DGCNN(conv_hidden_dim,62,2,conv_hidden_dim)
        # self.transformer_layers = nn.ModuleList(
        #     [copy.deepcopy(transformerlayer) for _ in range(n_layer)])

        # if init_method=='xavier':
        #     nn.init.xavier_uniform_(self.first_token)
        #     nn.init.xavier_uniform_(self.channel_emb)
        #     nn.init.xavier_uniform_(self.seq_emb)
        # else:
        trunc_normal_(self.first_token)
        trunc_normal_(self.channel_emb)
        trunc_normal_(self.seq_emb)
        
        self.fc1 = Linear(62 * conv_hidden_dim, conv_hidden_dim)
        # self.fc2 = Linear(64, num_classes)
        
        self.tot=0


    def forward(self,x,M): #(batch_size,channel,seq)
        batch_size,channel,seq=x.size()
        # if self.tot<=1:
        #     print("x before conv",x)
        x=self.conv(x) #(batch_size,conv_hidden_dim,seq)
        
        # if self.tot<=1:
        #     print("x after conv ",x)
            
        
        batch_size,dim,seq=x.size()
        
        # cls_token=self.first_token.expand(batch_size,-1,-1) #(batch_size,conv_hidden_dim,1)
        # x=torch.cat((cls_token,x),dim=2)
        # assert(x.size()[2]==seq+1)
        # channel_emb_used=self.channel_emb[:,:dim,:].expand(batch_size,-1,seq)  #no seq+1
        # seq_emb_used=self.seq_emb[:,:,:seq+1].expand(batch_size,dim,-1)
        # x=x+seq_emb_used
        # x[:,:,1:]+=channel_emb_used
        
        # x=x.permute([2,0,1]) #(seq+1,batch_size,channel)
        x=x.permute([0,2,1]) #batch_size,seq(electrode),dim
        
        # if self.tot<=1:
        #     print("x and M",x,M)
        #     self.tot+=1
            
        # for layer in self.transformer_layers:
        #     if not self.training or torch.rand(1) > self.layer_drop:
        #         x = layer(x,src_key_padding_mask=M.bool())
        x=self.gnn(x)
        
        return x.permute([0,2,1]),self.fc1(x.reshape(x.shape[0], -1))  #(batch_size,channel,seq+1)

class DASS_body(nn.Module):
    def __init__(self,n_sub,n_channel,conv_hidden_dim,trans_hidden_dim,n_layer,heads,
                 gamma=0.1,dropout=0,layer_drop=0,init_std=0.1,init_method='xavier',mlp_layer=1):
        super(DASS_body, self).__init__()
        #self.common_conv=conv_block(n_channel,conv_hidden_dim,dropout,layer=1)
        #print(mlp_layer,dropout)
        self.reconstruct_conv=conv_block(conv_hidden_dim,n_channel,layer=mlp_layer,dropout=0) # or dropout=0??
        self.subject_encoder=DASS_encoder(n_channel,conv_hidden_dim,trans_hidden_dim,n_layer,heads,
                 dropout,layer_drop,init_method,mlp_layer)
        self.general_encoder=DASS_encoder(n_channel,conv_hidden_dim,trans_hidden_dim,n_layer,heads,
                 dropout,layer_drop,init_method,mlp_layer)
        self.init_std=init_std
        self.init_method=init_method
        
        # self._init_para()
        #self.apply(self._init_para)
    
    def forward(self,X,M):
        #x=self.common_conv(x)
        x_sub,sub_emb=self.subject_encoder(X,M)
        x_gen,gen_emb=self.general_encoder(X,M) #(batch_size,conv_hidden_dim,seq+1)
        #print("before permute ",x_sub.shape)
        x_sub=x_sub.permute([1,0,2])
        x_gen=x_gen.permute([1,0,2])
        #print("XX",x_sub.shape,"MM",M.shape)
        x_sub=x_sub*(1-M)
        x_gen=x_gen*(1-M)
        x_sub=x_sub.permute([1,0,2])
        x_gen=x_gen.permute([1,0,2])
        
        if self.training:
            # rec_fea=self.reconstruct_conv(x_sub[:,:,1:]+x_gen[:,:,1:])
            rec_fea=self.reconstruct_conv(x_sub+x_gen)
            return x_sub,x_gen,sub_emb,gen_emb,rec_fea
        else:
            return x_sub,x_gen,sub_emb,gen_emb

    def save(self,filename):
        torch.save(self.state_dict(), filename)
        
    def load(self, filename, freeze=False, strict=True,device=None):
        state_dict = torch.load(filename,map_location=lambda storage, loc: storage.cuda(device) if device is not None else None)
        self.load_state_dict(state_dict, strict=strict)
        
        
class DASS_module(nn.Module):
    def __init__(self,n_sub,n_channel,conv_hidden_dim,trans_hidden_dim,n_layer,heads,
                 gamma=0.1,dropout=0,layer_drop=0,init_std=0.1,init_method='xavier',mlp_layer=1,tau=0.2):
        super(DASS_module, self).__init__()
        
        self.body=DASS_body(n_sub,n_channel,conv_hidden_dim,trans_hidden_dim,n_layer,heads,
                 gamma,dropout,layer_drop,init_std,init_method,mlp_layer)
        
        #self.sub_predict=projection_mlp(conv_hidden_dim,n_sub,mlp_layer,dropout)
        #self.gen_predict=projection_mlp(conv_hidden_dim,n_sub,mlp_layer,dropout)
        self.gamma=gamma
        self.init_std=init_std
        self.init_method=init_method
        # self._init_para()
        self.apply(self._init_para)
        self.tau=tau
        
        
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

    def _calc_contrastive_loss(self,emb,S):
        # print("emb",emb)
        L2_emb=torch.norm(emb,dim=1,p=2,keepdim=True)
        # print("L2_emb",L2_emb)
        norm_emb=emb/L2_emb
        cross_mat=torch.matmul(norm_emb,norm_emb.t())
        # print("cross_mat_before_exp",cross_mat)
        cross_mat=torch.exp(cross_mat/self.tau)
        eye=torch.eye(cross_mat.shape[0]).to(emb.device)
        S_add_dim=torch.unsqueeze(S,dim=0)
        equal=(S_add_dim==S_add_dim.t()).int().to(emb.device)
        #print(equal.shape,eye.shape,cross_mat.shape)
        # print("cross_mat",cross_mat)
        pos=torch.sum(cross_mat*(equal-eye))
        neg=torch.sum(cross_mat*(torch.ones_like(cross_mat).to(emb.device)-equal))
        # print("pos and neg",pos,neg)
        return -torch.log(pos/(pos+neg))
        
    def forward(self,X,S,M,alpha=0):  #(batch_size,channel,seq)
        
        # print("S",S)
        # print("M",M)
        if self.training:
            x_sub,x_gen,sub_emb,gen_emb,rec_fea=self.body(X,M)
            # print("x_sub",x_sub)
            # sub_emb=x_sub[:,:,0]
            # gen_emb=x_gen[:,:,0]
            
            # sub_emb=x_sub.mean(dim=-1)
            # gen_emb=x_gen.mean(dim=-1)
            
            #rec_fea=self.reconstruct_conv(x_sub[:,:,1:]*self.gamma+x_gen[:,:,1:]*(1-self.gamma))
            #rec_fea=self.reconstruct_conv(x_sub[:,:,1:]+x_gen[:,:,1:])
            
        
            # print("st-------")
            
            loss_sub=self._calc_contrastive_loss(sub_emb,S)
            loss_gen=self._calc_contrastive_loss(gen_emb,S)
            
            # for i in range(X.shape[0]):
            #     for j in range(X.shape[0]):
            #         if i==j:
            #             continue
            #         if S[i]==S[j]:
            #             loss_sub_pos+=calc_sim(sub_emb[i],sub_emb[j],self.tau)
            #             loss_gen_pos+=calc_sim(gen_emb[i],gen_emb[j],self.tau)
            #         else:
            #             loss_sub_neg+=calc_sim(sub_emb[i],sub_emb[j],self.tau)
            #             loss_gen_neg+=calc_sim(gen_emb[i],gen_emb[j],self.tau)
            # print("nd---------")
            #sub_pre=self.sub_predict(sub_emb)
            #gen_pre=self.gen_predict(ReverseLayerF.apply(gen_emb, alpha))
            return sub_emb,gen_emb,rec_fea,loss_sub,ReverseLayerF.apply(loss_gen,alpha)
        else:
            x_sub,x_gen,sub_emb,gen_emb=self.body(X,M)
            # sub_emb=x_sub[:,:,0]
            # gen_emb=x_gen[:,:,0]
            # sub_emb=x_sub.mean(dim=-1)
            # gen_emb=x_gen.mean(dim=-1)
            
            return sub_emb,gen_emb
    
    def save_body(self,filename):
        self.body.save(filename=filename)
    
    def save(self,filename):
        torch.save(self.state_dict(), filename)
        
    def load(self, filename, freeze=False, strict=True,device=None):
        state_dict = torch.load(filename,map_location=lambda storage, loc: storage.cuda(device) if device is not None else None)
        self.load_state_dict(state_dict, strict=strict)
        
    def load_body(self, encoder_file, freeze=False, strict=True,device=None):
        self.body.load(filename=encoder_file,device=device)
        
class DASS(nn.Module):
    def __init__(self,n_class,n_sub,n_channel,conv_hidden_dim,trans_hidden_dim,n_layer,heads,
                 gamma=0.1,dropout=0,layer_drop=0,init_std=0.1,init_method='xavier',mlp_layer=1,tau=0.2):
        super(DASS, self).__init__()
        self.DASS_m=DASS_module(n_sub,n_channel,conv_hidden_dim,trans_hidden_dim,n_layer,heads,
                 gamma,dropout,layer_drop,init_std,init_method,mlp_layer,tau)
        self.predict_mlp=projection_mlp(2*conv_hidden_dim,n_class,mlp_layer,dropout)
        self.gamma=gamma
        self.init_std=init_std
        self.init_method=init_method
        self.apply(self._init_para)
        self.n_class=n_class
        self.n_sub=n_sub
        
    def _init_para(self, m):
        if isinstance(m, nn.Linear):
            kaiming_normal_(m.weight)
            # if self.init_method=='xavier':
            #     nn.init.xavier_uniform_(m.weight)
            # else:
            #     trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def save(self,filename):
        torch.save(self.state_dict(), filename)
        
    def load(self, filename, freeze=False, strict=True,device=None):
        # self.DASS_m.load(encoder_file, strict=strict,device=device)
        # self.DASS_m.freeze_features(unfreeze=not freeze)
        state_dict = torch.load(filename,map_location=lambda storage, loc: storage.cuda(device) if device is not None else None)
        self.load_state_dict(state_dict, strict=strict)
    
    def load_DASS_m(self, encoder_file, freeze=False, strict=True,device=None):
        self.DASS_m.load(encoder_file, strict=strict,device=device)
        #self.encoder.freeze_features(unfreeze=not freeze)

    def load_DASS_body(self, encoder_file, freeze=False, strict=True,device=None):
        self.DASS_m.body.load(encoder_file, freeze, strict,device=device)
    
    def forward(self,X,S,M,alpha=0):
        if self.training:
            sub_emb,gen_emb,rec_fea,loss_sub,loss_gen=self.DASS_m(X,S,M,alpha)
            #emotion_predict=self.predict_mlp(sub_emb*self.gamma+gen_emb*(1-self.gamma))
            emotion_predict=self.predict_mlp(torch.cat((sub_emb,gen_emb),1))
            return sub_emb,gen_emb,rec_fea,loss_sub,loss_gen,emotion_predict
        else:
            sub_emb,gen_emb=self.DASS_m(X,S,M)
            #emotion_predict=self.predict_mlp(sub_emb*self.gamma+gen_emb*(1-self.gamma))
            emotion_predict=self.predict_mlp(torch.cat((sub_emb,gen_emb),1))
            return sub_emb,gen_emb,emotion_predict
