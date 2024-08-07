import torch
import tqdm
import argparse
import numpy as np
import random
from manager_torch import GPUManager
import time
import os
import yaml
from torch.utils.data import ConcatDataset
import mne
import json
from utils import data_load, data_prepare,ExperimentConfig,PretrainDataset
from utils import NormalDataset,PretrainDataset
from model import DASS,DASS_module
from trainer import DASS_processer,DASS_pretrain_processer
import shutil  
  
def green_print(x,end='\n'):
    print("\033[91m"+str(x)+"\033[0m",end=end)

mne.set_log_level(False)

def parse_args():
    parser = argparse.ArgumentParser(description="Pretrains a BENDER model.")
    parser.add_argument('--ds-config', default=None,
                        help="The DN3 config file to use.")
    parser.add_argument('--pretrain',default='False',
                        help="Pretrain or not")
    parser.add_argument('--randSeed',default=-1,help="The Random Seed.")
    parser.add_argument('--pre-load',default=-1,help="Whether use pretrain weights or not.")
    parser.add_argument('--d', default=None,
                    help="Description.")
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()
    gm = GPUManager()
    cuda_index =gm.auto_choice(mode=0)
    #cuda_index=0
    
    args.device_index = "cuda:"+str(cuda_index)
    print("Using device ", args.device_index)
    args.device = torch.device('cuda', cuda_index)
    print("args loaded.")
    
    experiment = ExperimentConfig(args.ds_config)
    print("experiment config loaded.")
    
    if args.d is not None:
        green_print(args.d)
        experiment.experiment['data_params'].update({'des':args.d})
    
    if args.randSeed==-1:
        args.randSeed=experiment.data_params['randSeed']
    else:
        args.randSeed=int(args.randSeed)
        experiment.experiment['data_params']['randSeed']=args.randSeed
    if args.pre_load==-1:
        args.pre_load=experiment.data_params['pre_load']
    else:
        if args.pre_load[0]=='T' or args.pre_load[0]=='t':
            args.pre_load=True
        else:
            args.pre_load=False
        experiment.experiment['data_params']['pre_load']=args.pre_load
    
    torch.manual_seed(args.randSeed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(args.randSeed)  # 为当前GPU设置随机种子
    # if you are using multi-GPU，为所有GPU设置随机种子
    torch.cuda.manual_seed_all(args.randSeed)
    np.random.seed(args.randSeed)  # Numpy module.
    random.seed(args.randSeed)  # Python random module.
    torch.manual_seed(args.randSeed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    args.dataset = experiment.data_params["dataset"]
    #args.hidden_size = experiment.model_params["hidden_size"]
    args.data_root_dir = experiment.data_root_dir
    args.data_name = experiment.data_params["data_name"]
    args.num_sub = experiment.data_params['num_sub']
    args.num_channels = experiment.data_params['num_channels']
    args.input_dim=experiment.data_params['input_dim']
    print(args.data_name)
    
    now_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    
    if args.pretrain[0]=='T' or args.pretrain[0]=='t':
        args.pretrain=True
    else:
        args.pretrain=False
        
    if args.pretrain:
        result_save_folder = os.path.join(
            'checkpoints_pretrain',str(args.dataset), now_time)
    else:
        result_save_folder = os.path.join(
            'results', str(args.dataset), experiment.data_params['split_type']+str(experiment.data_params['num_class']), now_time)
    
    if os.path.exists(result_save_folder) == False:
        os.makedirs(result_save_folder)
    
    # save yaml file
    with open(os.path.join(result_save_folder, '0yaml_config'), 'w') as file:
        file.write(yaml.dump(experiment.experiment, allow_unicode=True))
    
    file=os.listdir()
    for f in file:
        if os.path.isfile(f):
            src_file = os.path.join('./',f)
            dst_folder=os.path.join(result_save_folder,'pyfiles')
            if os.path.exists(dst_folder) == False:
                os.makedirs(dst_folder)
            dst_file = os.path.join(dst_folder,f)
            shutil.copy(src_file, dst_file)
            #print(src_file,dst_file)
    
    if args.pretrain:
        green_print("Pretraining...")
        data_lst=[]
        tot_subs=0
        tot_samples=0
        for name in args.dataset: 
            data_now, sub_list,num_subs,mask,mask_list=data_load(args,True,tot_subs,name)
            tot_subs+=num_subs
            tot_samples+=data_now.shape[0]
            data_lst.append((data_now, sub_list,num_subs,mask,mask_list))
        
        original_data=np.zeros((tot_samples,args.num_channels,args.input_dim))
        sub_list=None
        mask_list=None
        tot_num_sub=0
        cnt=0
        for tp in data_lst:
            original_data[cnt:cnt+tp[0].shape[0],:,:]=tp[0]
            if sub_list is None:
                sub_list=tp[1]
            else:    
                sub_list=np.concatenate((sub_list,tp[1]),axis=0)
            
            if mask_list is None:
                mask_list=tp[4]
            else:    
                mask_list=np.concatenate((mask_list,tp[4]),axis=0)
                
            tot_num_sub+=tp[2]
            cnt+=tp[0].shape[0]
        
        print("training data loaded.")
        # for i in range(100):
        #     k=random.randint(0,original_data.shape[0])
        #     print(original_data[k])
            
        pretrain_dataset = PretrainDataset(original_data, sub_list, mask_list,tot_num_sub,device=args.device)
        print(len(pretrain_dataset))
        
        data_params=experiment.data_params
        training_params=experiment.training_params
        model_params=experiment.model_params
        
        model=DASS_module(
                    n_sub=num_subs,
                    n_channel=data_params['num_channels'],
                    **model_params
                    )
        #print(model)
        
        trainer=DASS_pretrain_processer(model,device=args.device)
        train_log, best_acc,best_epoch =trainer.train(
            train_dataset=pretrain_dataset,
            save=experiment.data_params.get('save',False),
            save_fold=result_save_folder,
            save_name='DASS_pretrain_weight',
            des=args.d+" "+now_time,
            **training_params
        )
        train_log.to_csv(os.path.join(
            result_save_folder, 'pretrain_result'), encoding="utf-8-sig", header=True)
            
    else:
        green_print("Fintuning...")
        original_data, sub_list,num_subs,mask,mask_list= data_load(args,pretrain=False,name=args.dataset)
        
        data_num=0
        total_acc=0
        # for fold, (training, validation, test) in enumerate(tqdm.tqdm(utils.get_lmoso_iterator(ds_name, ds))):
        '''
        fewshot时, 这里val和train直接反过来,之后容易处理。注意！
        '''
        data_val, label_val, val_sub,val_sub_list,val_mask_list,\
        data_train, label_train, train_sub,train_sub_list,train_mask_list= data_prepare(
            original_data, sub_list, mask_list, args,experiment, fold=0,few_shot=True)
        print(type(data_train),type(label_train),type(data_val),type(label_val))
        print(data_train.shape, label_train.shape,
            data_val.shape, label_val.shape)
        
        # ---- make dataset and train and test
        # print(train_sub_list)
        # print(label_train)

        train_dataset = NormalDataset(data_train, label_train,train_sub_list,train_mask_list,num_sub=num_subs, device=args.device,is_train=True)
        valid_dataset = NormalDataset(data_val,label_val, val_sub_list,val_mask_list,num_sub=num_subs, device=args.device,is_train=False)
        
        # tqdm.tqdm.write(torch.cuda.memory_summary())
        # print(data_train.shape,label_train.shape,data_val.shape,label_val.shape)
        print(len(train_dataset), len(valid_dataset))
        data_params=experiment.data_params
        training_params=experiment.training_params
        model_params=experiment.model_params
        model=DASS(n_class=data_params['num_class'],
                n_sub=len(train_sub),
                n_channel=data_params['num_channels'],
                    **model_params
                )
        if args.pre_load==True:
            green_print("Preloading Weights.")
            model.load_DASS_body(experiment.pretrain_DASS_m_weight,device=args.device)
            
        trainer=DASS_processer(model,device=args.device)
        train_log, valid_log,best_acc_per_sub,epoch_acc_per_sub,best_total_class_predictions =trainer.train(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            save=experiment.data_params.get('save',False),
            save_fold=result_save_folder,
            save_name='downstream_DASS_best.pt',
            des=args.d+" "+now_time,
            **training_params
        )
        
        print("**************** Result:")
        print(best_acc_per_sub)
        print(epoch_acc_per_sub)
        print(best_acc_per_sub.mean())
        
        train_log.to_csv(os.path.join(
            result_save_folder, 'train_log'), encoding="utf-8-sig", header=True)
        valid_log.to_csv(os.path.join(
            result_save_folder, 'valid_log'), encoding="utf-8-sig", header=True)
        np.save(os.path.join(
            result_save_folder, 'best_total_class_predictions'),best_total_class_predictions)

        result_fold_dict = {'best_acc_per_sub': best_acc_per_sub.tolist(), 'epoch_acc_per_sub': epoch_acc_per_sub.tolist(),'avg':best_acc_per_sub.mean()}
        result_fold_name = 'result_log'

        with open(os.path.join(result_save_folder, result_fold_name), 'w') as file:
            json.dump(result_fold_dict, file)

