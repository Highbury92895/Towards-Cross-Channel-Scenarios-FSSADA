import torch
import argparse
import numpy as np
from solver import Solver
from utils import str2bool



def main(args):
    
   
    torch.cuda.set_device(args.cuda_device_ind)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    #np.set_printoptions(precision=4)
    #torch.set_printoptions(precision=4)

    print()
    print('[ARGUMENTS]')
    print(args)
    print()

    net = Solver(args)

    if args.mode == 'train' : net.train()
    elif args.mode == 'train_source' : net.train_source()
    elif args.mode == 'test' : net.test()
    elif args.mode == 'train_dann' : net.train_dann()
    elif args.mode == 'plot' : net.plot()
    else : return 0

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='TOY')
    parser.add_argument('--mode',default='train', type=str, help='train or test or train_source or train_dann')
    parser.add_argument('--batch_size', default = 1024, type=int, help='batch size')
    parser.add_argument('--batch_size_target_label', default = 64, type=int, help='batch size')#64 11
    parser.add_argument('--source_name', default = 'awgn', type=str)  #RadioML rayleigh_128 awgn
    parser.add_argument('--target_name', default = 'rayleigh', type=str)  
    parser.add_argument('--seed', default = 31, type=int, help='random seed')
    parser.add_argument('--epoch', default = 1000, type=int, help='epoch size')
    parser.add_argument('--lr', default = 0.0001, type=float, help='learning rate')
    parser.add_argument('--cuda',default=True,type=str2bool, help='enable cuda')
    parser.add_argument('--cuda_device_ind',default=4, type=int)
    parser.add_argument('--load_ckpt',default='', type=str, help='checkpoint name')#pretrain test
    parser.add_argument('--k_shot',default=90, type=int, help='num k shot') #280--3% 94 12 1200 1
    parser.add_argument('--lamda1',default=0.1, type=float)
    parser.add_argument('--lamda2',default=0.1, type=float)
    parser.add_argument('--datetimeSel',default='2024-06-07_10-45-24', type=str)
    parser.add_argument('--net',default='resnet', type=str) #simple resnet
    parser.add_argument('--MMEenable',default=True, type=str2bool, help='Minimax Entropy') 
    parser.add_argument('--WDGRLenable',default=True, type=str2bool, help='Wasserstein Distance Guided Representation Learning') 
    parser.add_argument('--MMEMenable',default=False, type=str2bool, help='Minimax Modified Entropy') 

    parser.add_argument('--log_dir',default='ResFinal/AWGNToRayleigh/Fusion/log', type=str) 
    parser.add_argument('--model_dir',default='ResFinal/AWGNToRayleigh/Fusion/model', type=str) 
    parser.add_argument('--image_dir',default='ResFinal/AWGNToRayleigh/Fusion/images', type=str) 
    parser.add_argument('--dir',default='Final', type=str) #Final/date

    


    
    args = parser.parse_args()
    
    main(args)
