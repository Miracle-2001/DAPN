import torch
import yaml
import numpy as np
import os
import scipy.io as sio
import random
import hdf5storage as hdf5
from torch.utils.data import Dataset as TorchDataset


class ExperimentConfig:
    """
    Parses DN3 configuration files. Checking the DN3 token for listed datasets.
    """
    def __init__(self, config_filename: str, adopt_auxiliaries=True):
        """
        Parses DN3 configuration files. Checking the DN3 token for listed datasets.

        Parameters
        ----------
        config_filename : str
                          String for path to yaml formatted configuration file
        adopt_auxiliaries : bool
                             For any additional tokens aside from DN3 and specified datasets, integrate them into this
                             object for later use. Defaults to True. This will propagate for the detected datasets.
        """
        with open(config_filename, 'r') as fio:
            self._original_config = yaml.load(fio, Loader=yaml.FullLoader)

        self.experiment = self._original_config #working_config.pop('Configuratron')
        print(self.experiment)
        
        self.data_params=self.experiment.get("data_params",None)
        self.data_root_dir=self.data_params.get("data_root_dir",None)
        self.data_name=self.data_params.get("data_name",None)
        self.model_params=self.experiment.get('model_params',None)

        self.pretrain_DASS_m_weight=self.experiment.get('pretrain_DASS_m_weight',None)
        
        self.configuration=self.experiment.get('Configuratron',None)
        if self.configuration:
            self.global_samples = self.configuration.get('samples', None)
            self.global_sfreq = self.configuration.get('sfreq', None)
        self.bending_college_args=self.experiment.get('bending_college_args',None)
        self.optimizer_params=self.experiment.get('optimizer_params',None)
        self.augmentation_params=self.experiment.get("augmentation_params",None)
        self.training_params=self.experiment.get("training_params",None)
        self.bendr_classification_args=self.experiment.get("bendr_classification_args",None)

def get_data_info(args,name):
    all_channel=args.input_dim #62
    if name=='FACED':
        data_root_dir=os.path.join(args.data_root_dir,name)
        data_name='de_lds_fold0.mat'
        all_subs=123
        reorder=[0, 1, 5, 3, 2, 4, 6, 9, 7, 8, 10, 14, 12, 11, 13, 15, 18, 16, 17, 19, 23, 21, 20, 22, 24, 25, 26, 28, 27, 29]
        mask=[False, True, False, True, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, True, False, True, False, True, True, True, False, False, False, True]
    elif name=='DEAP':
        data_root_dir=os.path.join(args.data_root_dir,name)
        data_name='de_lds_5band.mat'
        all_subs=32
        reorder=[0, 16, 1, 17, 3, 2, 18, 19, 20, 4, 5, 22, 21, 7, 6, 23, 24, 25, 8, 9, 27, 26, 11, 10, 15, 28, 29, 12, 30, 13, 14, 31]
        mask=[False, True, False, False, False, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, True, False, True, False, True, True, True, False, False, False, True]
    elif name=='SEED' or name=='SEED_IV':
        data_root_dir=os.path.join(args.data_root_dir,name)
        data_name='de_lds.mat'
        all_subs=15
        reorder=[i for i in range(all_channel)]
        mask=[False for i in range(all_channel)]
    return data_root_dir,data_name,all_subs,reorder,mask

def data_load(args, pretrain=False,start_sub=0,name=None):
    print("loading_dataset:",name)
    all_channel=args.input_dim #62
    data_root_dir,data_name,all_subs,reorder,mask=get_data_info(args,name)
    data_dir = os.path.join(data_root_dir, data_name)
    data = hdf5.loadmat(data_dir)['de_lds']
    
    data = data_norm(
        data, True)
    # label shape: 720 or 840   720=24*30. 840=28*30   24/28 videos and 30 samples are generated by each video
    # data shape: (123, 720, 5*30)  720=24*30  120=4*30  30:channel num  4: band num
    
    
    # if pretrain:
    # 这里面的num_channels不是指脑电电极数量，指的是脑电的band数量，也就是进入模型里面后每个token的通道数量（每个token对应脑电一个电极）
    # feature shape对应的才是脑电的电极数量30，32或者62
    feature_shape = int(data.shape[-1]/args.num_channels)
    print(feature_shape,args.num_channels,all_channel) #(30/32/62,5,62)
    data = data.reshape(-1, args.num_channels,feature_shape)  #.transpose([0, 2, 1])
    data=data[:,:,reorder]
    data_pad=np.zeros((data.shape[0],args.num_channels,all_channel))
    
    cnt=0
    for i in range(all_channel):
        if mask[i]==False:
            data_pad[:,:,i]=data[:,:,cnt]
            cnt+=1
    # print(data_pad)
    assert(cnt==feature_shape)
    # if args.num_sub!=all_subs:
    #     total=data.shape[0]
    #     single=int(total/all_subs)
    #     data=data[:single*args.num_sub,:,:]
    
    sample_per_sub=int(data.shape[0]/all_subs)
    sub_list=[]
    for id in range(start_sub,start_sub+all_subs):
        sub_list=sub_list+[id]*sample_per_sub
        
    
    #mask=[False]+mask #add a position for cls token
    
    mask_list=np.zeros((data.shape[0],len(mask))).astype(int)
    mask_list[:,:]=mask
    
    sub_list=np.array(sub_list).astype(int)
    print("shape of data loaded after reshape.", data.shape,sub_list.shape)
    
    return data_pad,sub_list,all_subs,mask,mask_list
    # else:
    #     # if args.num_sub!=all_subs:
    #     #     data=data[-args.num_sub:,:,:]
    #     print("data shape after loaded.", data.shape)
        
    #     return data


def data_norm(data, channel_norm):  # 这个相当于是pretrain的时候额外加的数据处理
    # 这里(之后要对BENDR加一个通道，保留一个固定值，其他的)直接标准化就行了
    # 不加通道了,preprocess简单一些.
    # isFilt: False  filten:1   channel_norm: True
    # Normalization for each sub
    if channel_norm:
        for i in range(data.shape[0]):
            data[i, :, :] = (data[i, :, :] - np.mean(data[i, :, :],
                             axis=0)) / np.std(data[i, :, :], axis=0)
    return data


def data_prepare(data,sub_list, mask_list, args, experiment, fold,session=None,few_shot=True):

    data_param_dict = experiment.data_params
    n_channel=experiment.data_params['num_channels']
    name=data_param_dict['dataset']
    n_subs = data_param_dict['num_sub']
    n_per = n_subs//data_param_dict['num_fold']
    n_fold = data_param_dict['num_fold']
    
    data_root_dir,data_name,_,_,_=get_data_info(args,name)
    label_path=os.path.join(data_root_dir,data_name)
    labels=hdf5.loadmat(label_path)['label'] #(32,1200,2)
    print(labels.shape)
    # if name=='SEED':
        
    # elif name=='SEED_IV':
    #     label_path='../../dataset/SEED_IV/de_lds.mat'
    #     labels=hdf5.loadmat(label_path)['label'] #(32,1200,2)
    
    feature_shape = data.shape[-1]
    # label shape: 720 or 840   720=24*30. 840=28*30   24/28 videos and 30 samples are generated by each video, 视频在前面samples在后面
    # data shape: (123, 720, 120 or 150 or 255*30)  720=24*30  120=4*30  30:channel num  4: band num

    val_sub = None
    val_list = None
    # print(n_per)
    # cross-subject / subject-independent
    if fold < n_fold - 1:
        val_sub = np.arange(n_per * fold, n_per * (fold + 1))
    else:
        val_sub = np.arange(n_per * fold, n_subs)
    train_sub = np.array(list(set(np.arange(n_subs)) - set(val_sub)))
    val_sub = [int(_) for _ in val_sub]

    
    sample_per_sub=int(data.shape[0]/n_subs)
    
    data=data.reshape(n_subs,sample_per_sub,-1)
    mask_list=mask_list.reshape(n_subs,sample_per_sub,-1)
    
    data_train = data[list(train_sub), :, :].reshape(-1,
                                                        n_channel,feature_shape) #.transpose([0, 2, 1])
    data_val = data[list(val_sub), :, :].reshape(-1,
                                                    n_channel,feature_shape) #.transpose([0, 2, 1])
    mask_train=mask_list[list(train_sub), :, :].reshape(-1,mask_list.shape[-1])
    mask_val=mask_list[list(val_sub), :, :].reshape(-1,mask_list.shape[-1])
    
    
    train_sub_list=[]
    # 注意这里train_sub的id是从0开始直接编号，和原来的编号无关
    for id in range(train_sub.shape[0]):
        train_sub_list=train_sub_list+[id]*sample_per_sub
    train_sub_list=np.array(train_sub_list)
    
    val_sub_list=[]
    # 注意这里val_sub的id是从0开始直接编号，和原来的编号无关
    for id in range(len(val_sub)):
        val_sub_list=val_sub_list+[id]*sample_per_sub
    val_sub_list=np.array(val_sub_list)
    
    assert(train_sub_list.shape[0]==data_train.shape[0])
    assert(train_sub_list.shape[0]==mask_train.shape[0])
    
    print(train_sub)
    print(val_sub)
    label_train = labels[list(train_sub), :].reshape(-1)
    label_val = labels[list(val_sub), :].reshape(-1)
    
    
    return data_train, label_train, train_sub,train_sub_list,mask_train,\
            data_val, label_val, val_sub,val_sub_list,mask_val 

def label_gen(data, label_type, istest):
    # isFilt: False  filten:1   channel_norm: True
    if istest:
        print("\033[91m" + "############# IS TEST ##############" + "\033[0m")
        n_vids = int(data.shape[1]/2)
        label = [0] * n_vids
        label.extend([1] * n_vids)
        return label

    if label_type == 2:
        n_vids = 24
    elif label_type == 9:
        n_vids = 28
    n_samples = np.ones(n_vids).astype(np.int32) * 30  # (30,30,...,30)

    if label_type == 2:
        label = [0] * 12
        label.extend([1] * 12)
    elif label_type == 9:
        label = [0] * 3
        for i in range(1, 4):
            label.extend([i] * 3)
        label.extend([4] * 4)
        for i in range(5, 9):
            label.extend([i] * 3)

    label_repeat = []
    for i in range(len(label)):
        label_repeat = label_repeat + [label[i]]*n_samples[i]

    # print(len(label_repeat))
    return label_repeat

class NormalDataset(TorchDataset):  # Here, it is inherited from DN3ataset
    def __init__(self, data,label,sub_list,mask_list,num_sub, device='cpu',is_train=True):
        super(NormalDataset, self).__init__()
        self.data = data
        self.sub_list=sub_list
        self.mask_list=mask_list
        self.device = device
        self.num_sub=num_sub
        self.label=label
        self.is_train=is_train
        # print("datasetLen",data.shape,sub_list.shape,mask_list.shape)
        # print("sublist",self.sub_list,np.mean(self.sub_list))
        # print()
        # self.data= torch.from_numpy(data).to(self.device, dtype=torch.float32)

    def shuffle(self,):
        #print(self.data.shape)
        self.order=[i for i in range(self.data.shape[0])]
        random.shuffle(self.order)
        #print(self.order)
        self.list_each_sub=[[] for i in range(self.num_sub)]
        for i in self.order:
            self.list_each_sub[self.sub_list[i]].append(i)
        self.cnt_each_sub=[0 for i in range(self.num_sub)]
        #self.out=[0 for i in range(self.data.shape[0])]
        self.start=0
        self.last_sub=-1
        
        tmp_list=[]
        cnt=0
        now=-1
        for id in self.sub_list:
            if id==now:
                cnt+=1
            else:
                cnt=1
                now=id
            if cnt%2==1:
                tmp_list.append(id)
        self.sub_order=tmp_list
        random.shuffle(self.sub_order)
    
    def __getitem__(self, ind):
        if self.is_train:
            if self.last_sub==-1:
                now_sub=self.sub_order[self.start]
                ind=self.list_each_sub[now_sub][self.cnt_each_sub[now_sub]]
                self.cnt_each_sub[now_sub]+=1
                self.start+=1
                
                if self.cnt_each_sub[now_sub]==len(self.list_each_sub[now_sub]):
                    self.last_sub=-1
                else:
                    self.last_sub=now_sub
            else:
                now_sub=self.last_sub
                ind=self.list_each_sub[now_sub][self.cnt_each_sub[now_sub]]
                self.cnt_each_sub[now_sub]+=1
                self.last_sub=-1
            assert(now_sub==self.sub_list[ind])
        
        X = np.array(self.data[ind])  
        S = np.array(self.sub_list[ind])
        Y = np.array(self.label[ind]) 
        M = np.array(self.mask_list[ind]) 
        return torch.from_numpy(X).to(self.device, dtype=torch.float32), torch.from_numpy(S).to(self.device, dtype=torch.int32), torch.from_numpy(Y).to(self.device, dtype=torch.int32),torch.from_numpy(M).to(self.device, dtype=torch.int32)

    def __len__(self,):
        return self.data.shape[0]


class PretrainDataset(TorchDataset):
    def __init__(self, data,sub_list,mask_list,num_sub, device):
        super(PretrainDataset, self).__init__()
        self.data = data
        self.sub_list=sub_list
        self.mask_list=mask_list
        self.device = device
        self.num_sub=num_sub
        
        print("datasetLen",data.shape,sub_list.shape,mask_list.shape)
        # self.data= torch.from_numpy(data).to(self.device, dtype=torch.float32)

    def shuffle(self,):
        #print(self.data.shape)
        self.order=[i for i in range(self.data.shape[0])]
        random.shuffle(self.order)
        #print(self.order)
        self.list_each_sub=[[] for i in range(self.num_sub)]
        for i in self.order:
            self.list_each_sub[self.sub_list[i]].append(i)
        self.cnt_each_sub=[0 for i in range(self.num_sub)]
        #self.out=[0 for i in range(self.data.shape[0])]
        self.start=0
        self.last_sub=-1
        
        tmp_list=[]
        cnt=0
        now=-1
        for id in self.sub_list:
            if id==now:
                cnt+=1
            else:
                cnt=1
                now=id
            if cnt%2==1:
                tmp_list.append(id)
        self.sub_order=tmp_list
        random.shuffle(self.sub_order)
        
    
    def __getitem__(self, ind):
        # print("ind",ind)
        if self.last_sub==-1:
            now_sub=self.sub_order[self.start]
            ind=self.list_each_sub[now_sub][self.cnt_each_sub[now_sub]]
            self.cnt_each_sub[now_sub]+=1
            self.start+=1
            
            if self.cnt_each_sub[now_sub]==len(self.list_each_sub[now_sub]):
                self.last_sub=-1
            else:
                self.last_sub=now_sub
        else:
            now_sub=self.last_sub
            ind=self.list_each_sub[now_sub][self.cnt_each_sub[now_sub]]
            self.cnt_each_sub[now_sub]+=1
            self.last_sub=-1
        
        assert(now_sub==self.sub_list[ind])
        
        # print("now_ind",ind)
        X = np.array(self.data[ind])
        # print("X with ind",X)  
        S = np.array(self.sub_list[ind])
        # print("S with ind",S)
        M = np.array(self.mask_list[ind])#.astype(float)
        #print(M)
        return torch.from_numpy(X).to(self.device, dtype=torch.float32), torch.from_numpy(S).to(self.device, dtype=torch.int32),torch.from_numpy(M).to(self.device, dtype=torch.int32)

    def __len__(self,):
        return self.data.shape[0]
