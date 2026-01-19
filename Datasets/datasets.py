import torch, os
from torch.utils.data import Dataset, Subset, DataLoader, TensorDataset
import scipy.io as sio
import json
import random
import numpy as np
import pickle


def return_data(args, datasets_name, fewShot):
    if datasets_name == 'awgn':
        json_path = os.path.join(os.getcwd(),'Datasets', 'sig_awgn.json')
    elif datasets_name == 'rayleigh':
        json_path = os.path.join(os.getcwd(),'Datasets', 'sig_rayleigh.json')
    elif datasets_name == 'rician':
        json_path = os.path.join(os.getcwd(),'Datasets', 'sig_rician.json')
    elif datasets_name == 'rayleigh_128':
        json_path = os.path.join(os.getcwd(),'Datasets', 'sig_rayleigh_128.json')
    
    if datasets_name == 'RadioML':
        dataset = RadioMLDataset()
    else:
        dataset = MyDataset(json_path)
    
    train_dataset, eval_dataset, test_dataset = DataSplit(args, dataset)
    
    dataloader_args = {"batch_size": args.batch_size, "shuffle": True, "num_workers": 8, "drop_last":True}
    # semi-supervised setting
    if fewShot : 
        train_dataset_label, train_dataset_unlabel = DataSplitFewshot(args, train_dataset)
        train_data_loader_label = DataLoader(train_dataset_label, batch_size=args.batch_size_target_label, shuffle=True, num_workers=8, drop_last=True)
        train_data_loader_unlabel = DataLoader(train_dataset_unlabel, **dataloader_args)
    else :
        train_dataset_label = train_dataset 
        train_data_loader_label = DataLoader(train_dataset_label, **dataloader_args)
        train_data_loader_unlabel = TensorDataset()

    eval_data_loader = DataLoader(eval_dataset, **dataloader_args)
    test_data_loader = DataLoader(test_dataset, **dataloader_args)
    return train_data_loader_label, train_data_loader_unlabel, eval_data_loader, test_data_loader


class MyDataset(Dataset):
    def __init__(self, json_path):
        super().__init__()
        with open(json_path, 'r') as file:
            json_data = json.load(file)
        self.numFrame = json_data['numFrame']
        self.numSample = json_data['numSample']
        self.data = json_data['data']
        self.sig = []
        self.label = []
        self.snr = []
        mapping = {'BPSK': 0, 'QPSK': 1, '8PSK': 2, '16QAM': 3, '64QAM': 4, 'PAM4': 5, 'GFSK': 6, 'CPFSK': 7, 'B-FM': 8, 'DSB-AM': 9, 'SSB-AM': 10}
        for sample in self.data:
            data_path = os.path.join(os.getcwd(),'Datasets', sample['data_path'].replace('\\', '/'))
            label = mapping[sample['label']]
            snr = sample['snr']
            sig = np.array(sio.loadmat(data_path)['data'], dtype='float32')
            for sub_array in sig:
                self.sig.append(sub_array)
            [self.label.append(label) for _ in range(self.numFrame )]
            [self.snr.append(snr) for _ in range(self.numFrame )]
        print('loading json done')
    
    def __getitem__(self, idx):
        sig = self.sig[idx]
        label = self.label[idx]
        snr = self.snr[idx]
        return sig, label, snr
    
    def __len__(self):
        return len(self.data)* self.numFrame
    




class RadioMLDataset(Dataset):
    def __init__(self):
        super().__init__()
        dataset_pkl = open('Datasets/Sig_RadioML2016/RML2016.10a_dict.pkl','rb')
        RML_dataset_location = pickle.load(dataset_pkl, encoding='bytes')
        self.numFrame = 1000
        self.numSample = 128
        self.sig = []
        self.label = []
        self.snr = []
        mapping = {'BPSK': 0, 'QPSK': 1, '8PSK': 2, 'QAM16': 3, 'QAM64': 4, 'PAM4': 5, 'GFSK': 6, 'CPFSK': 7, 'WBFM': 8, 'AM-DSB': 9, 'AM-SSB': 10}
        for sample in RML_dataset_location:  
            label =  mapping[str(sample[0], encoding = "gbk")]
            snr = sample[1]
            if snr in np.arange(-6, 20, 2):
                sig = RML_dataset_location[sample]
                sig = sig/np.sqrt(2*np.mean(np.square(sig)))
                for sub_array in sig:
                    self.sig.append(sub_array)
                [self.label.append(label) for _ in range(self.numFrame )]
                [self.snr.append(snr) for _ in range(self.numFrame )]
        self.data_len = len(self.sig)
        print('loading pkl done')    
    
    def __getitem__(self, idx):
        sig = self.sig[idx]
        label = self.label[idx]
        snr = self.snr[idx]
        return sig, label, snr
    
    def __len__(self):
        return self.data_len



def DataSplit(args, dataset):
    train_size = int(0.7 * len(dataset))
    eval_size  = int(0.15 * len(dataset))
    test_size  = int(0.15 * len(dataset))
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    train_dataset, eval_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size, test_size])
    check_result = [check_dataset(dataset) for dataset in (train_dataset, eval_dataset, test_dataset)]
    print()
    print('[dataset split result]')
    print('train_dataset'+str(check_result[0]))
    print('eval_dataset'+str(check_result[1]))
    print('test_dataset'+str(check_result[2]))
    print()
    return train_dataset, eval_dataset, test_dataset


# check if dataset divided uniformly
def check_dataset(dataset):
    modulationTypes = np.arange(0, 11, 1)
    snrRange =  np.arange(-6, 20, 2)
    mod_count = []
    snr_count = []
    for mod in modulationTypes:
        mod_count.append(list(Subset(dataset.dataset.label, dataset.indices)).count(mod))
    for snr in snrRange:
        snr_count.append(list(Subset(dataset.dataset.snr, dataset.indices)).count(snr))
    return mod_count, snr_count


def DataSplitFewshot(args, dataset):
    modulationTypes = np.arange(0, 11, 1)
    selected_idx = []
    for mod in modulationTypes:
        modlist = list(Subset(dataset.dataset.label, dataset.indices))
        indices = [i for i, x in enumerate(modlist) if x == mod]
        random_indices = random.sample(indices, k=min(args.k_shot, len(indices)))
        selected_idx.extend([dataset.indices[i] for i in random_indices])
    train_dataset_label = Subset(dataset.dataset, selected_idx)
    unselected_idx = [x for x in dataset.indices if x not in selected_idx]
    train_dataset_unlabel = Subset(dataset.dataset, unselected_idx)
    return train_dataset_label, train_dataset_unlabel


def return_data_plot(args, datasets_name):
    if datasets_name == 'awgn':
        json_path = os.path.join(os.getcwd(),'Datasets', 'sig_awgn.json')
    elif datasets_name == 'rayleigh':
        json_path = os.path.join(os.getcwd(),'Datasets', 'sig_rayleigh.json')
    elif datasets_name == 'rician':
        json_path = os.path.join(os.getcwd(),'Datasets', 'sig_rician.json')
    elif datasets_name == 'rayleigh_128':
        json_path = os.path.join(os.getcwd(),'Datasets', 'sig_rayleigh_128.json')
    
    if datasets_name == 'RadioML':
        dataset = RadioMLDataset()
    else:
        dataset = MyDatasetPlot(json_path)
    modulationTypes = np.arange(0, 11, 1)
    data_loader_list = []
    for mod in modulationTypes:
        mod_ind = np.where(np.array(dataset.label)==mod)
        mod_subset = Subset(dataset, list(mod_ind[0]))
        dataloader_args = {"batch_size": len(mod_subset), "shuffle": True, "num_workers": 8, "drop_last":True}
        mod_data_loader = DataLoader(mod_subset, **dataloader_args)
        data_loader_list.append(mod_data_loader)
   
    return data_loader_list


class MyDatasetPlot(Dataset):
    def __init__(self, json_path):
        super().__init__()
        with open(json_path, 'r') as file:
            json_data = json.load(file)
        self.numFrame = json_data['numFrame']
        self.numSample = json_data['numSample']
        self.data = json_data['data']
        self.sig = []
        self.label = []
        self.snr = []
        mapping = {'BPSK': 0, 'QPSK': 1, '8PSK': 2, '16QAM': 3, '64QAM': 4, 'PAM4': 5, 'GFSK': 6, 'CPFSK': 7, 'B-FM': 8, 'DSB-AM': 9, 'SSB-AM': 10}
        for sample in self.data:
            data_path = os.path.join(os.getcwd(),'Datasets', sample['data_path'].replace('\\', '/'))
            snr = sample['snr']
            if snr == 18:
                label = mapping[sample['label']]
                sig = np.array(sio.loadmat(data_path)['data'], dtype='float32')
                for sub_array in sig:
                    self.sig.append(sub_array)
                [self.label.append(label) for _ in range(self.numFrame )]
                [self.snr.append(snr) for _ in range(self.numFrame )]
        print('loading json done')
    
    def __getitem__(self, idx):
        sig = self.sig[idx]
        label = self.label[idx]
        snr = self.snr[idx]
        return sig, label, snr
    
    def __len__(self):
        return len(self.data)* self.numFrame

    
    


