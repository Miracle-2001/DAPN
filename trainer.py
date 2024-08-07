import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
import os
import time
import tqdm
import inspect
from model import DASS, DASS_encoder, DASS_module
from torch.utils.data import DataLoader
from pandas import DataFrame

def L2_norm(x):
    return torch.sum(torch.square(x))

def L1_norm(x):
    return torch.sum(torch.abs(x))

def green_print(x,end='\n'):
    print("\033[91m"+str(x)+"\033[0m",end=end)

def calc_acc(predictions,target,nd=False):
    class_predict = predictions.argmax(axis=-1)
    class_predict = class_predict.cpu().detach().numpy()
    target=target.cpu().detach().numpy()
    num_correct_predict = np.sum(class_predict == target)
    
    if nd:
        return num_correct_predict,class_predict,class_predict==target
    else:
        return num_correct_predict,class_predict

class DASS_processer(object):
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def _do_epoch(self, data_loader, mode='eval', epoch_num=1):
        if mode == 'train':
            self.model = self.model.train()
        else:
            self.model = self.model.eval()

        epoch_loss = {'Predict_loss': 0, 'Rec_loss': 0,
                      'Gen_loss': 0, 'Sub_loss': 0, 'Orth_loss': 0}
        epoch_metrics = {}

        pbar = tqdm.trange(1, len(data_loader) + 1, desc="Iter",
                           initial=1, total=len(data_loader),ncols=110)
        data_iterator = iter(data_loader)

        predict_loss_f = nn.CrossEntropyLoss()
        rec_loss_f = nn.MSELoss()
        # gen_loss_f = nn.CrossEntropyLoss()
        # sub_loss_f = nn.CrossEntropyLoss()

        total_sample=0
        num_correct_predict=0
        # num_correct_sub_p=0
        # num_correct_gen_p=0
        total_class_predictions= []
        
        n_valid_sub=14
        acc_sub=np.zeros(n_valid_sub)
        
        if self.num_epoch != 1:
            p = epoch_num/(self.num_epoch-1)
        else:
            p = 1
        beta = 2/(1+math.exp(-10*p))-1
        
        for iteration in pbar:
            X, S, Y, M= next(data_iterator)
            num_sample=X.shape[0]
            
            if mode == 'train':
                sub_emb, gen_emb, rec_fea, loss_sub, loss_gen, emotion_predict = self.model(X,S,M,beta)
                # print(sub_emb.shape,gen_emb.shape,rec_fea.shape,sub_pre.shape,gen_pre.shape,emotion_predict.shape)
                
                predict_loss = predict_loss_f(
                    emotion_predict.float(),
                    torch.Tensor(Y.float()).long().to(self.device))
                rec_loss=rec_loss_f(rec_fea,X)
                norm_sub_emb=torch.norm(sub_emb,dim=1,p=2,keepdim=True)
                norm_gen_emb=torch.norm(gen_emb,dim=1,p=2,keepdim=True)
                orth_loss=L1_norm(torch.mul(sub_emb/norm_sub_emb,gen_emb/norm_gen_emb))/X.shape[0]
                Loss=self.co_p*predict_loss+self.co_r*rec_loss+\
                     self.co_o*orth_loss+self.co_g*loss_gen+self.co_s*loss_sub
            
                self.optimizer.zero_grad()
                Loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=40)  # change to 20
                self.optimizer.step()
                
                ncp,predict_class=calc_acc(emotion_predict,Y)
                
            else:
                sub_emb, gen_emb, emotion_predict = self.model(X,S,M)
                predict_loss = predict_loss_f(
                    emotion_predict.float(),
                    torch.Tensor(Y.float()).long().to(self.device))
                ncp,predict_class,check_list=calc_acc(emotion_predict,Y,nd=True)
                assert(len(check_list)==S.shape[0])
                for i in range(len(check_list)):
                    acc_sub[S[i]]+=check_list[i]
                    
            total_class_predictions += [
                    item for item in predict_class]
            num_correct_predict+=ncp
            
            with torch.no_grad():
                total_sample+=num_sample
                epoch_loss['Predict_loss']+=predict_loss.item()
                if mode == 'train':
                    epoch_loss['Orth_loss']+=orth_loss.item()
                    epoch_loss['Rec_loss']+=rec_loss.item()
                    epoch_loss['Gen_loss']+=loss_gen.item()
                    epoch_loss['Sub_loss']+=loss_sub.item()
            
            pbar.set_postfix({'acc':num_correct_predict/total_sample,'GL':epoch_loss['Gen_loss']/(total_sample/self.batch_size),'SL':epoch_loss['Sub_loss']/(total_sample/self.batch_size),
                              'OL':epoch_loss['Orth_loss']/(total_sample/self.batch_size),'RL':epoch_loss['Rec_loss']/(total_sample/self.batch_size),
                              })
            
        epoch_metrics['predict_acc'] = num_correct_predict/total_sample   
        epoch_metrics['Gen_loss']=epoch_loss['Gen_loss']/(total_sample/self.batch_size)
        epoch_metrics['Sub_loss']=epoch_loss['Sub_loss']/(total_sample/self.batch_size)
        
        if mode == 'eval':
            acc_sub=acc_sub/(total_sample/n_valid_sub)
            for i in range(n_valid_sub):
                epoch_metrics[str(i+2)+'_acc']=acc_sub[i]
        
        for k in epoch_loss.keys():
            epoch_loss[k]/=(total_sample/self.batch_size)
            
        return epoch_metrics,total_class_predictions,epoch_loss
                
    def train(self, train_dataset, valid_dataset, num_epoch, batch_size, lr, 
              weight_decay=0, des=None,
              co_p=1,co_o=0,co_r=0,co_g=0,co_s=0,
              early_stop=None,betas=[0.9,0.98],opt='Adam',
              save=False,save_fold=None,save_name=None
              ):
        self.model.to(self.device)
        self.co_p=co_p
        self.co_o=co_o
        self.co_r=co_r
        self.co_g=co_g
        self.co_s=co_s
        self.batch_size=batch_size
        self.num_epoch=num_epoch
        
        if opt=='SGD':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        best_acc = 0
        if early_stop is None:
            early_stop = num_epoch
        early_stop_num = 0
        best_epoch = -1

        train_log = list()
        valid_log = list()

        epoch_bar = tqdm.trange(1, num_epoch + 1, desc="Epo",
                                initial=1, total=num_epoch,ncols=80)
        
        n_sub=14
        best_acc_per_sub=np.zeros(n_sub)
        epoch_acc_per_sub=np.zeros(n_sub)
        sample_per_sub=int(len(valid_dataset)/n_sub)
        best_predict_per_sub=np.zeros((n_sub,sample_per_sub))
        
        for epoch in epoch_bar:
            train_dataset.shuffle()
            train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True)
            
            valid_loader = DataLoader(
            dataset=valid_dataset, batch_size=batch_size, shuffle=False)
        
        
            train_metric, _,train_loss= self._do_epoch(train_loader, 'train', epoch)
            eval_metric, predictions,eval_loss= self._do_epoch(valid_loader, 'eval',epoch)
            
            # train_metric['epoch']=epoch
            # eval_metric['epoch']=epoch
            
            train_tmp_metric=train_metric.copy()
            train_tmp_metric.update(train_loss)
            eval_tmp_metric=eval_metric.copy()
            eval_tmp_metric.update(eval_loss)
            train_log.append(train_tmp_metric)
            valid_log.append(eval_tmp_metric)
            
            print()
            green_print('After Epoch '+str(epoch)+" in "+des)
            green_print('train: ',end=' ')
            for k,v in train_tmp_metric.items():
                if k=='predict_acc' or k=='Predict_loss':
                    green_print(str(k)+str(' %.3f' %v),end=' | ')
                else:
                    print(k,'%.3f' %v,end=' | ')
            print()
            green_print('eval: ',end=' ')
            for k,v in eval_tmp_metric.items():
                if k=='predict_acc' or k=='Predict_loss' or k.split('_')[0].isdigit():
                    green_print(str(k)+str(' %.3f' %v),end=' | ')
            print()    
            
            for i in range(n_sub):
                now_acc=eval_metric[str(i+2)+'_acc']
                if now_acc>best_acc_per_sub[i]:
                    best_acc_per_sub[i]=now_acc
                    epoch_acc_per_sub[i]=epoch
                    #print(len(predictions))
                    #print(i*sample_per_sub,(i+1)*sample_per_sub)
                    best_predict_per_sub[i][:]=np.array(predictions[i*sample_per_sub:(i+1)*sample_per_sub])
                    self.model.save(os.path.join(save_fold,'best_model_param_for_'+str(i+2)+'.pt'))
            
            # if eval_metric['predict_acc'] > best_acc:
            #     best_acc = eval_metric['predict_acc']
            #     best_epoch = epoch
            #     best_total_class_predictions=predictions
                
            #     if save==True:
            #         assert(save_fold is not None)
            #         assert(save_name is not None)
            #         self.model.save(os.path.join(save_fold,save_name))
            
            # if epoch>=0.8*num_epoch:
            #     green_print("change opt to SGD")    
            #     self.optimizer = torch.optim.SGD(
            #     self.model.parameters(), lr=lr, weight_decay=weight_decay)
                
        self.train_log = train_log
        self.valid_log = valid_log
        return DataFrame(train_log), DataFrame(valid_log),best_acc_per_sub,epoch_acc_per_sub,best_predict_per_sub
    
    def evaluate(self,):
        pass



class DASS_pretrain_processer(object):
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def _do_epoch(self, data_loader, epoch_num=-1):
        self.model = self.model.train()

        epoch_loss = { 'Rec_loss': 0,
                      'Gen_loss': 0, 'Sub_loss': 0, 'Orth_loss': 0}
        epoch_metrics = {}

        pbar = tqdm.trange(1, len(data_loader) + 1, desc="Iter",
                           initial=1, total=len(data_loader),ncols=100)
        data_iterator = iter(data_loader)

        rec_loss_f = nn.MSELoss()
        # gen_loss_f = nn.CrossEntropyLoss()
        # sub_loss_f = nn.CrossEntropyLoss()

        total_sample=0
        
        if self.num_epoch != 1:
            p = epoch_num/(self.num_epoch-1)
        else:
            p = 1
        beta = 2/(1+math.exp(-10*p))-1
        
        
        for iteration in pbar:
            X, S, M= next(data_iterator)
            
            # print("X",X[0])
            
            num_sample=X.shape[0]
            sub_emb, gen_emb, rec_fea, loss_sub, loss_gen = self.model(X,S,M,alpha=beta)
            # print(sub_emb.shape,gen_emb.shape,rec_fea.shape,sub_pre.shape,gen_pre.shape,emotion_predict.shape)
            rec_loss=rec_loss_f(rec_fea,X)
            norm_sub_emb=torch.norm(sub_emb,dim=1,p=2,keepdim=True)
            norm_gen_emb=torch.norm(gen_emb,dim=1,p=2,keepdim=True)
            orth_loss=L1_norm(torch.mul(sub_emb/norm_sub_emb,gen_emb/norm_gen_emb))/X.shape[0]
            # gen_loss=gen_loss_f(gen_pre.float(),torch.Tensor(S.float()).long().to(self.device))
            # sub_loss=sub_loss_f(sub_pre.float(),torch.Tensor(S.float()).long().to(self.device))
            Loss=self.co_r*rec_loss+\
                    self.co_o*orth_loss+self.co_g*loss_gen+self.co_s*loss_sub
            
            self.optimizer.zero_grad()
            Loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=40)  # change to 20
            self.optimizer.step()
            
            # ncsp,_=calc_acc(sub_pre,S)
            # ncgp,_=calc_acc(gen_pre,S)
            # num_correct_sub_p+=ncsp
            # num_correct_gen_p+=ncgp
            
            with torch.no_grad():
                total_sample+=num_sample
                epoch_loss['Orth_loss']+=orth_loss.item()
                epoch_loss['Rec_loss']+=rec_loss.item()
                epoch_loss['Gen_loss']+=loss_gen.item()
                epoch_loss['Sub_loss']+=loss_sub.item()
            
            # print("totsample,batchsize",total_sample,self.batch_size)
            pbar.set_postfix({'GL':epoch_loss['Gen_loss']/(total_sample/self.batch_size),'SL':epoch_loss['Sub_loss']/(total_sample/self.batch_size),
                              'OL':epoch_loss['Orth_loss']/(total_sample/self.batch_size),'RL':epoch_loss['Rec_loss']/(total_sample/self.batch_size),
                              })
            
        epoch_metrics['Gen_loss']=epoch_loss['Gen_loss']/(total_sample/self.batch_size)
        epoch_metrics['Sub_loss']=epoch_loss['Sub_loss']/(total_sample/self.batch_size)
        
        for k in epoch_loss.keys():
            epoch_loss[k]/=(total_sample/self.batch_size)
            
        return epoch_metrics,epoch_loss
                
    def train(self, train_dataset, num_epoch, batch_size, lr, 
              weight_decay=0, des=None,
              co_o=0,co_r=0,co_g=0,co_s=0,
              early_stop=None,betas=[0.9,0.98],opt='Adam',
              save=False,save_fold=None,save_name=None
              ):
        self.model.to(self.device)
        self.co_o=co_o
        self.co_r=co_r
        self.co_g=co_g
        self.co_s=co_s
        self.batch_size=batch_size
        self.num_epoch=num_epoch
        
        if opt=='SGD':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        
        best_acc = 0
        if early_stop is None:
            early_stop = num_epoch
        early_stop_num = 0
        best_epoch = -1

        train_log = list()

        epoch_bar = tqdm.trange(1, num_epoch + 1, desc="Epo",
                                 initial=1, total=num_epoch,ncols=60)
        for epoch in epoch_bar:
            
            train_dataset.shuffle()
            train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
            train_metric, train_loss= self._do_epoch(train_loader, epoch)
            train_tmp_metric=train_metric.copy()
            train_tmp_metric.update(train_loss)
            
            train_log.append(train_tmp_metric)
            
            print()
            green_print('After Epoch '+str(epoch)+" in "+des)
            green_print('train: ',end=' ')
            for k,v in train_tmp_metric.items():
                if k=='predict_acc' or k=='Predict_loss':
                    green_print(str(k)+str(' %.3f' %v),end=' | ')
                else:
                    print(k,'%.3f' %v,end=' | ')
            print()
            best_epoch = epoch
            if save==True:
                assert(save_fold is not None)
                assert(save_name is not None)
                self.model.save_body(os.path.join(save_fold,save_name+'_'+str(epoch)+'.pt'))
                
        self.train_log = train_log
        return DataFrame(train_log), best_acc,best_epoch