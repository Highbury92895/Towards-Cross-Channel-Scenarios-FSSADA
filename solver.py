import numpy as np
import torch

import argparse
import os
import datetime
import random
import math

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')


from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from utils import cuda, data_aug_rotation, rotation_2d, NTXentLoss
from Datasets.datasets import return_data, return_data_plot
from Model.Net.net import Feedforward, Classifier, WDGRL, ResNet18Feature, DANN
from pathlib import Path
from torch.utils.data import TensorDataset
from itertools import cycle
from sklearn.manifold import TSNE
from torch.autograd import grad
from tqdm import tqdm

class Solver(object):
    def __init__(self, args):
        if args.mode != 'plot':
            # Datasets
            self.source_train, _ , self.source_eval, self.source_test = return_data(args, args.source_name, fewShot = False)
            self.target_train_label, self.target_train_unlabel, self.target_eval, self.target_test = return_data(args, args.target_name, fewShot=True)

        # basic parameters
        self.args = args
        self.cuda = (args.cuda and torch.cuda.is_available())
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.global_epochIdx = 0
        self.global_iterIdx = 0
        self.acc_target_best = 0
        
        # Network & Optimizer
        if args.net == 'simple':
            self.netF = cuda(Feedforward(), self.cuda)
        elif args.net == 'resnet':
            self.netF = cuda(ResNet18Feature(), self.cuda)
        
        self.netD = cuda(Classifier(), self.cuda)
        self.netW = cuda(WDGRL(), self.cuda)
        self.optimF = optim.Adam(self.netF.parameters(), lr=self.lr)
        self.optimD = optim.Adam(self.netD.parameters(), lr=self.lr)
        self.optimW = optim.Adam(self.netW.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        

        
        # PathFolder
        self.load_ckpt = args.load_ckpt

        if self.args.dir == 'date':
            pathPreBase = os.path.join(os.getcwd(),'Model', 'Pretrain')
            pathResBase = os.path.join(os.getcwd(),'Model', 'Res')
            pathPre = os.path.join(os.getcwd(),'Model', 'Pretrain', datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            pathRes = os.path.join(os.getcwd(),'Model', 'Res', datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            pathLog = os.path.join(os.getcwd(),'log', datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            if not os.path.exists(pathLog):  os.makedirs(pathLog)
            self.pathPre, self.pathRes, self.pathLog = pathPre, pathRes, pathLog
            
            if self.args.mode != 'test':
                if not os.path.exists(pathPre):  os.makedirs(pathPre)
                if not os.path.exists(pathRes):  os.makedirs(pathRes)
                
            
            self.datetimeSel = args.datetimeSel
            if self.load_ckpt == 'pretrain': 
                print('loading pre_trained model')
                self.netF.load_state_dict(torch.load(os.path.join(pathPreBase,  self.datetimeSel, 'netF.pth')))
                self.netD.load_state_dict(torch.load(os.path.join(pathPreBase,  self.datetimeSel, 'netD.pth')))
            elif self.load_ckpt == 'test': 
                print('loading trained model begin to test')
                self.netF.load_state_dict(torch.load(os.path.join(pathResBase,  self.datetimeSel, 'netF.pth')))
                self.netD.load_state_dict(torch.load(os.path.join(pathResBase,  self.datetimeSel, 'netD.pth')))
       
        elif self.args.dir == 'Final':
            self.pathRes, self.pathLog = self.args.model_dir, self.args.log_dir
            if self.load_ckpt == 'test': 
                print('loading trained model begin to test')
                self.netF.load_state_dict(torch.load(os.path.join(self.pathRes, 'netF.pth')))
                self.netD.load_state_dict(torch.load(os.path.join(self.pathRes, 'netD.pth')))
        



        # tensorboard
        self.writer = SummaryWriter(self.pathLog)
        self.writer.add_text(tag = 'argument', text_string=str(args))
        self.writer.add_text(tag = 'info', text_string='remain unlabel training')

    def train(self):
        # set mode
        self.netF.train()
        self.netD.train()
        
        for epochInd in range(self.epoch) :
            self.global_epochIdx = self.global_epochIdx + 1
            
            
            for bacthInd, (data_source, data_target_label, data_target_unlabel) in enumerate(zip(self.source_train, cycle(self.target_train_label), cycle(self.target_train_unlabel))):
               
                self.global_iterIdx = self.global_iterIdx + 1
                
                
                # labeled data training
                self.optimF.zero_grad()
                self.optimD.zero_grad()
                data_cat = cuda(torch.cat((data_source[0].unsqueeze(1), data_target_label[0].unsqueeze(1)), 0), self.cuda)
                data_cat_label = cuda(torch.cat((data_source[1], data_target_label[1]), 0), self.cuda)
                
                """
                if epochInd < 20:
                    data_cat = cuda(data_source[0].unsqueeze(1), self.cuda)
                    data_cat_label = cuda(data_source[1], self.cuda)
                """
                
                hidden_feature_label = self.netF(data_cat)

            
 
                output = self.netD(hidden_feature_label, False, 1)
                class_loss = self.criterion(output, data_cat_label)
                loss = class_loss 
                loss.backward(retain_graph=True)
                self.optimF.step()
                self.optimD.step()

                
                if ((bacthInd + 1) % 10 == 0):
                    print("Epoch {} Step [{}/{}]:   L_1:{:.5f}"
                    .format(epochInd , bacthInd + 1 , len(self.source_train), loss.item()))
                    self.writer.add_scalar("loss1",loss.item(),   self.global_iterIdx)

                  
                
                
                temp1 = data_target_unlabel[1]
                temp2 = data_target_unlabel[2]
                data_target_unlabel = cuda(data_target_unlabel[0].unsqueeze(1), self.cuda)
                if self.args.MMEenable:
                    # unlabeled data training
                    self.optimF.zero_grad()
                    self.optimD.zero_grad()
                    hidden_feature_unlabel = self.netF(data_target_unlabel)
                    

                    if self.args.MMEMenable:
                        output = self.netD(hidden_feature_unlabel, False, 1)
                        out_t  = F.softmax(output, dim=1)
                        loss_t = -self.args.lamda1 * torch.mean(torch.sum(out_t *(torch.log(out_t + 1e-5)), 1))
                        loss_t.backward()
                        self.optimF.step()
                        self.optimD.step()
                    else:
                        output = self.netD(hidden_feature_unlabel, True, 1)
                        out_t  = F.softmax(output, dim=1)
                        loss_t = self.args.lamda1 * torch.mean(torch.sum(out_t *(torch.log(out_t + 1e-5)), 1))
                        loss_t.backward()
                        self.optimF.step()
                        self.optimD.step()

                    
                    
                    
                        

                    
                    
                    
                    
                    if ((bacthInd + 1) % 10 == 0):
                        print("Epoch {} Step [{}/{}]:   L_2:{:.5f}"
                        .format(epochInd , bacthInd + 1 , len(self.source_train), loss_t.item()))
                        self.writer.add_scalar("loss2",loss_t.item(), self.global_iterIdx)
                
                

                if self.args.WDGRLenable:
                    self.optimW.zero_grad()
                    self.optimF.zero_grad()
                    hidden_feature_label = self.netF(cuda(data_source[0].unsqueeze(1), self.cuda))
                    hidden_feature_unlabel = self.netF(data_target_unlabel)
                    label_out = self.netW(hidden_feature_label, True, 1)
                    unlabel_out = self.netW(hidden_feature_unlabel, True, 1)
                    #indices = cuda(torch.tensor(random.choices(range(0,len(hidden_feature_label)), k=len(hidden_feature_unlabel))),self.cuda)
                    #hidden_feature_label_sel = torch.index_select(hidden_feature_label, 0, indices)
                    # 1-Lipschitz penelization
                    alpha_loss = torch.rand(hidden_feature_label.size(0), 1)
                    alpha_loss = alpha_loss.expand(hidden_feature_label.size()).type_as(hidden_feature_label)
                    interpolations = alpha_loss * hidden_feature_label+ (1. - alpha_loss) *  hidden_feature_unlabel
                    interpolations = torch.cat((interpolations, hidden_feature_label, hidden_feature_unlabel), dim=0).requires_grad_()
                    preds = self.netW(interpolations, True, 1)
                    gradients = grad(
                        preds,
                        interpolations,
                        grad_outputs=torch.ones_like(preds),
                        retain_graph=True,
                        create_graph=True,
                    )[0]
                    gradient_norm = gradients.norm(2, dim=1)
                    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
                    wasserstein_distance = label_out.mean() -  unlabel_out.mean()
                    divergence = self.args.lamda2*(-wasserstein_distance + 10*gradient_penalty)

                    self.writer.add_scalar("wasserstein_distance", wasserstein_distance.item(), self.global_iterIdx)
                    self.writer.add_scalar("gradient_penalty", gradient_penalty.item(), self.global_iterIdx)

                    
                    
                    divergence.backward()
                    self.optimW.step()
                    self.optimW.step()
                    self.optimW.step()
                    self.optimW.step()
                    self.optimW.step()
                    self.optimF.step()

                    if ((bacthInd + 1) % 10 == 0):
                        print("Epoch {} Step [{}/{}]:   wasserstein_distance:{:.5f}  gradient_penalty:{:.5f}  divergence:{:.5f}"
                        .format(epochInd , bacthInd + 1 , len(self.source_train), wasserstein_distance.item(), gradient_penalty.item(), divergence.item()))
                
                        
                
                
                    
                
                
                
           
            self.eval()
    
    
    def eval(self):
        # set mode
        self.netF.eval()
        self.netD.eval()

        acc_source = self.acc_cal(self.source_eval)
        acc_target = self.acc_cal(self.target_eval)
        print("acc_source:{:.5f}  acc_target:{:.5f}"
        .format(acc_source.item() , acc_target.item()))
        self.writer.add_scalar("acc_source",acc_source.item(), self.global_epochIdx)
        self.writer.add_scalar("acc_target",acc_target.item(), self.global_epochIdx)

        if self.args.mode == 'train_source': 
            if acc_source.item() > self.acc_target_best:
                self.acc_target_best = acc_source.item()
                torch.save(self.netF.state_dict(), os.path.join(self.pathPre, 'netF.pth'))
                torch.save(self.netD.state_dict(), os.path.join(self.pathPre, 'netD.pth'))
                print('save current checkpoint')
        elif self.args.mode == 'train': 
            if acc_target.item() > self.acc_target_best:
                self.acc_target_best = acc_target.item()
                torch.save(self.netF.state_dict(), os.path.join(self.pathRes, 'netF.pth'))
                torch.save(self.netD.state_dict(), os.path.join(self.pathRes, 'netD.pth'))
                print('save current checkpoint')
        elif self.args.mode == 'train_dann': 
            if acc_target.item() > self.acc_target_best:
                self.acc_target_best = acc_target.item()
                torch.save(self.netF.state_dict(), os.path.join(self.pathRes, 'netF.pth'))
                torch.save(self.netD.state_dict(), os.path.join(self.pathRes, 'netD.pth'))
                print('save current checkpoint')
        elif self.args.mode == 'semiAMC': 
            torch.save(self.netF.state_dict(), os.path.join(self.pathRes, 'netF.pth'))
            torch.save(self.netD.state_dict(), os.path.join(self.pathRes, 'netD.pth'))
            print('save current checkpoint')
        elif self.args.mode == 'semiAMCFT': 
            if acc_target.item() > self.acc_target_best:
                self.acc_target_best = acc_target.item()
                torch.save(self.netF.state_dict(), os.path.join(self.pathRes, 'netF.pth'))
                torch.save(self.netD.state_dict(), os.path.join(self.pathRes, 'netD.pth'))
                print('save current checkpoint')
        
        


        
    def test(self):
        self.netF.eval()
        self.netD.eval()
        

        #self.acc_plt(self.source_train, False, 'source_train')
        #self.acc_plt(self.source_eval, False, 'source_eval.jpg')
        #self.acc_plt(self.source_test, False, 'source_test.jpg')
        
        
        #self.acc_plt(self.target_train_unlabel ,False, 'target_train_unlabel')
        #self.acc_plt(self.target_train_label, True, 'target_train_label')
        #self.acc_plt(self.target_eval, False, 'target_eval.jpg')
        #self.acc_plt(self.target_test, False, 'target_test.jpg')
        
        
        
        
        
        
        
        #self.cal_acc_snr(self.source_test, 'source.jpg')
        acc_source = self.acc_cal(self.source_test)
        acc_target = self.acc_cal(self.target_test)
        self.writer.add_scalar("acc_all_snr_source", acc_source)
        self.writer.add_scalar("acc_all_snr_target", acc_target)
        self.cal_acc_snr(self.target_test, 'target.jpg')
        print("acc_source:{:.5f}  acc_target:{:.5f}"
        .format(acc_source.item() , acc_target.item()))
    
    
    
    def acc_plt(self, dataloader, contat, name):
        if contat:
            hid = np.zeros([self.args.batch_size_target_label,128])
            for bacthInd, data in enumerate(dataloader):
                index = []
                hidden_feature = self.netF(cuda(data[0].unsqueeze(1),self.cuda))
                temp = hidden_feature.cpu().detach().numpy()
                hid  = np.concatenate((hid, temp))
            
            hid = hid[self.args.batch_size_target_label:1088,:]
            tsne = TSNE(n_components=2)
            x_tsne = tsne.fit_transform(hid)
            plt.figure()
            plt.scatter(x_tsne[:,0],x_tsne[:,1])
            plt.savefig(os.path.join(os.getcwd(),'images', name),format='svg')
            plt.savefig(os.path.join(os.getcwd(),'images', name))
        else:
            ind = random.randint(0,len(dataloader)-1)
            plt.figure()
            for bacthInd, data in enumerate(dataloader): 
                if bacthInd == ind:
                    hidden_feature = self.netF(cuda(data[0][index].unsqueeze(1),self.cuda))
                    temp = hidden_feature.cpu().detach().numpy()
                    tsne = TSNE(n_components=2)
                    x_tsne = tsne.fit_transform(temp)
            plt.scatter(x_tsne[:,0],x_tsne[:,1])
            plt.savefig(os.path.join(os.getcwd(),'images', name),format='svg')
            plt.savefig(os.path.join(os.getcwd(),'images', name))
                  

    
    def acc_cal(self, dataloader):
        acc_count = []
        for bacthInd, data in enumerate(dataloader):
            data_label = cuda(data[1],self.cuda)
            hidden_feature = self.netF(cuda(data[0].unsqueeze(1),self.cuda))
            output = self.netD(hidden_feature, False, 1)
            output_label = torch.max(output, 1, keepdim=False)[1]
            acc_count.append((torch.eq(output_label, data_label)==True).sum())
        return  sum(acc_count) / (len(output_label)*len(dataloader))   
                

    
    def cal_acc_snr(self, dataloader, name):
        acc_all  = []
        snrs = list(range(-6,20,2))

        def filter_snr_samples(self, dataloader, target_snr):
            right_sample = 0
            num_sample = 0
            for data in dataloader:
                target_indices = torch.nonzero(data[2] == target_snr, as_tuple=False).squeeze()
                input = data[0]
                input_label = data[1]
                if target_indices.numel() > 0:
                    data_input = cuda(input[target_indices],self.cuda)
                    data_label = cuda(input_label[target_indices],self.cuda)
                    hidden_feature = self.netF(cuda(data_input.unsqueeze(1),self.cuda))
                    output = self.netD(hidden_feature, False, 1)
                    output_label = torch.max(output, 1, keepdim=False)[1]
                    right_sample = right_sample + (torch.eq(output_label, data_label)==True).sum().item()
                    num_sample   = num_sample + output_label.shape[0]
                    
            return right_sample / num_sample

        
        for target_snr in snrs:
            acc = filter_snr_samples(self, dataloader, target_snr)
            self.writer.add_scalar("acc_snr", acc, target_snr)
            acc_all.append(acc)
        
        plt.figure()
        plt.plot(snrs, acc_all, 'steelblue', marker='.', markersize=15, linestyle='-')
        plt.xticks(snrs)
        plt.yticks(np.arange(0, 1.01, 0.1))
        plt.ylabel('Accuracy (%)')
        plt.xlabel('SNR (dB)')
        plt.savefig(os.path.join(self.args.image_dir,name))

        

   
        
    
    
    
    def train_source(self):
        # set mode
        self.netF.train()
        self.netD.train()  
        for epochInd in range(self.epoch) :
            self.global_epochIdx = self.global_epochIdx + 1
            for bacthInd, data_source in enumerate(self.source_train):
               
                self.global_iterIdx = self.global_iterIdx + 1
                # labeled data training
                self.optimF.zero_grad()
                self.optimD.zero_grad()
                data = cuda(data_source[0].unsqueeze(1),self.cuda)
                data_label = cuda(data_source[1].unsqueeze(1),self.cuda)
                
                """
                if epochInd < 20:
                    data_cat = cuda(data_source[0].unsqueeze(1), self.cuda)
                    data_cat_label = cuda(data_source[1], self.cuda)
                """
                
                hidden_feature_label = self.netF(data)
                output = self.netD(hidden_feature_label, False, 1)
                loss = self.criterion(output, data_label.flatten())
                loss.backward()
                self.optimF.step()
                self.optimD.step() 
                if ((bacthInd + 1) % 10 == 0):
                    print("Epoch {} Step [{}/{}]:   L_1:{:.5f}"
                    .format(epochInd , bacthInd + 1 , len(self.source_train), loss.item()))
                    self.writer.add_scalar("loss1",loss.item(), self.global_iterIdx)

                
           
            self.eval()        
    

    def train_dann(self):
        # set mode
        self.netF.train()
        self.netD.train()  
        self.netW = cuda(DANN(), self.cuda)
        self.netW.train()
        optimizer = optim.Adam(list(self.netF.parameters())+list(self.netD.parameters()), lr= 0.001)
        optimizer_d = optim.Adam(self.netW.parameters(), lr= 0.001)

        """  
        print('pre-training')
        for epochInd in range(1) :
            for bacthInd, (data_source, data_target_label) in enumerate(zip(self.source_train, cycle(self.target_train_label))):
                 
                 
                 data_cat = cuda(torch.cat((data_source[0].unsqueeze(1), data_target_label[0].unsqueeze(1)), 0), self.cuda)
                 data_cat_label = cuda(torch.cat((data_source[1], data_target_label[1]), 0), self.cuda)
                 
                 #data_cat = cuda(data_source[0].unsqueeze(1), self.cuda)
                 #data_cat_label = cuda(data_source[1], self.cuda)
                 
                 
                 hidden_feature_label = self.netF(data_cat)
                 output_label = self.netD(hidden_feature_label, False, 1)
                 loss1 = nn.CrossEntropyLoss()(output_label, data_cat_label)
                 optimizer.zero_grad()
                 loss1.backward()
                 optimizer.step()
                 if ((bacthInd + 1) % 10 == 0):
                    print("Epoch {} Step [{}/{}]:   L_1:{:.5f}"
                    .format(epochInd , bacthInd + 1 , len(self.source_train), loss1.item()))
            self.eval()       
        """  
        
        print('DANN begin')
        for epochInd in range(self.epoch) :
            self.global_epochIdx = self.global_epochIdx + 1
            for bacthInd, (data_source, data_target_label, data_target_unlabel) in enumerate(zip(self.source_train, cycle(self.target_train_label), self.target_train_unlabel)):
                

                p = float(((bacthInd+1) + (epochInd) * len(self.source_train)) /5000 / len(self.source_train))
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                #alpha = 0.0001
                
                
                self.global_iterIdx = self.global_iterIdx + 1
                # labeled data training
                optimizer.zero_grad()
                optimizer_d.zero_grad()
                
                
                data_cat = cuda(torch.cat((data_source[0].unsqueeze(1), data_target_label[0].unsqueeze(1)), 0), self.cuda)
                data_cat_label = cuda(torch.cat((data_source[1], data_target_label[1]), 0), self.cuda)
                #data_cat = cuda(data_source[0].unsqueeze(1), self.cuda)
                #data_cat_label = cuda(data_source[1], self.cuda)
                
                hidden_feature_label = self.netF(data_cat)
                output_label = self.netD(hidden_feature_label, False, 1)
                labeled_output_domain_hat = self.netW(hidden_feature_label, True, alpha)


                hidden_feature_unlabel = self.netF(cuda(data_target_unlabel[0].unsqueeze(1), self.cuda))
                unlabeled_output_domain_hat = self.netW(hidden_feature_unlabel, True, alpha)

                temp1 = cuda(torch.zeros(len(data_cat)), self.cuda)
                temp2 = cuda(torch.ones(len(data_target_unlabel[0].unsqueeze(1))),self.cuda)
                loss1 = nn.CrossEntropyLoss()(output_label, data_cat_label)
                loss2 = nn.CrossEntropyLoss()(labeled_output_domain_hat, temp1.long())+ nn.CrossEntropyLoss()(unlabeled_output_domain_hat, temp2.long())
                
                loss = loss1 + loss2
                loss.backward()
                optimizer.step()
                optimizer_d.step()
                

                if ((bacthInd + 1) % 10 == 0):
                    print("Epoch {} Step [{}/{}]:   L_1:{:.5f}  L_2:{:.5f}  alpha:{:.5f}"
                    .format(epochInd , bacthInd + 1 , len(self.source_train), loss1.item(), loss2.item(), alpha))
                    self.writer.add_scalar("loss1",loss1.item(),   self.global_iterIdx)
                    self.writer.add_scalar("loss2",loss2.item(),   self.global_iterIdx)

           
            self.eval()   

    
    def plot(self):
        self.netF.eval()
        self.netD.eval()
        source_data = return_data_plot(self.args, self.args.source_name)
        target_data = return_data_plot(self.args, self.args.target_name)

        acc_source = self.acc_cal(source_data[0])
        acc_target = self.acc_cal(target_data[0])
        print("acc_source:{:.5f}  acc_target:{:.5f}"
        .format(acc_source.item() , acc_target.item()))




        modulationTypes = np.arange(0, 11, 1)
        mapping = {'BPSK': 0, 'QPSK': 1, '8PSK': 2, '16QAM': 3, '64QAM': 4, 'PAM4': 5, 'GFSK': 6, 'CPFSK': 7, 'B-FM': 8, 'DSB-AM': 9, 'SSB-AM': 10}
        mapping = {v: k for k, v in mapping.items()}        

        plt.figure()
        for mod in modulationTypes: 
            print(mod)
            for data in enumerate(source_data[mod]):
                hidden_feature = self.netF(cuda(data[1][0][0:40].unsqueeze(1),self.cuda))
                temp = hidden_feature.cpu().detach().numpy()
                tsne = TSNE(n_components=2)
                x_tsne = tsne.fit_transform(temp)
                plt.scatter(x_tsne[:,0],x_tsne[:,1],s=6,label=mapping[mod])
        #plt.legend()
        plt.xticks([])
        plt.yticks([])
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.savefig(os.path.join(self.args.image_dir, 'source'),format='svg')
        plt.savefig(os.path.join(self.args.image_dir, 'source'))


        plt.figure()
        for mod in modulationTypes: 
            print(mod)
            for data in enumerate(target_data[mod]):
                hidden_feature = self.netF(cuda(data[1][0][0:40].unsqueeze(1),self.cuda))
                temp = hidden_feature.cpu().detach().numpy()
                tsne = TSNE(n_components=2)
                x_tsne = tsne.fit_transform(temp)
                plt.scatter(x_tsne[:,0],x_tsne[:,1],s=6,label=mapping[mod])
        #plt.legend()
        plt.xticks([])
        plt.yticks([])
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.savefig(os.path.join(self.args.image_dir, 'target'),format='svg')
        plt.savefig(os.path.join(self.args.image_dir, 'target'))
    
    
    
    def semiAMC(self):
        self.netF.train()
        self.netD.train() 
        print('semiAMC begin')
        optimizer1 = optim.Adam(list(self.netF.parameters())+list(self.netD.parameters()), lr= 0.0001)
        criterion = NTXentLoss() 
        for epochInd in range(200) :
            self.global_epochIdx = self.global_epochIdx + 1
            for bacthInd, data_target_unlabel in enumerate(self.target_train_unlabel):
                
                self.global_iterIdx = self.global_iterIdx + 1
                X_1, X_2 = data_aug_rotation(data_target_unlabel[0])
                X_1 = torch.stack(X_1)
                X_2 = torch.stack(X_2)

                X_1_hidden = self.netF(cuda(X_1.unsqueeze(1), self.cuda))
                X_1_output = self.netD(X_1_hidden, False, 1)

                X_2_hidden = self.netF(cuda(X_2.unsqueeze(1), self.cuda))
                X_2_output = self.netD(X_2_hidden, False, 1)

                loss1 = criterion(X_1_output, X_2_output)
                optimizer1.zero_grad()
                loss1.backward()
                optimizer1.step()
                

                if ((bacthInd + 1) % 1 == 0):
                    print("Epoch {} Step [{}/{}]:   L_1:{:.5f}"
                    .format(epochInd , bacthInd + 1 , len(self.source_train), loss1.item()))
                    self.writer.add_scalar("loss1",loss1.item(),   self.global_iterIdx)

           
            #self.eval()  
        #torch.save(self.netF.state_dict(), os.path.join(self.pathRes, 'netF.pth'))
        #torch.save(self.netD.state_dict(), os.path.join(self.pathRes, 'netD.pth')) 
    
    def semiAMCFT(self):
        self.semiAMC()
        print('semiAMC finetune begin')
        self.netD.train()
        optimizer2 = optim.Adam(self.netD.parameters(), lr=0.0001)  
        optimizer1 = optim.Adam(list(self.netF.parameters())+list(self.netD.parameters()), lr= 0.0001)
        for epochInd in range(self.epoch) :
            self.global_epochIdx = self.global_epochIdx + 1
            for bacthInd, data_target in enumerate(self.target_train_label):
               
                self.global_iterIdx = self.global_iterIdx + 1
                # labeled data training
                optimizer1.zero_grad()
               
                data = cuda(data_target[0].unsqueeze(1),self.cuda)
                data_label = cuda(data_target[1].unsqueeze(1),self.cuda)
                
                hidden_feature_label = self.netF(data)
                output = self.netD(hidden_feature_label, False, 1)
                loss = self.criterion(output, data_label.flatten())
                loss.backward()
                optimizer1.step()

                if ((bacthInd + 1) % 10 == 0):
                    print("Epoch {} Step [{}/{}]:   L_1:{:.5f}"
                    .format(epochInd , bacthInd + 1 , len(self.target_train_label), loss.item()))
                    self.writer.add_scalar("loss1",loss.item(), self.global_iterIdx)

                
           
            self.eval()        

    def testMatrix(self):
        self.netF.eval()
        self.netD.eval()

        acc_all  = []
        mapping = {'BPSK': 0, 'QPSK': 1, '8PSK': 2, 'QAM16': 3, 'QAM64': 4, 'PAM4': 5, 'GFSK': 6, 'CPFSK': 7, 'WBFM': 8, 'AM-DSB': 9, 'AM-SSB': 10}
        

        dataloader = self.target_test
        confusion_matrix = torch.zeros(11, 11, dtype=torch.int64)
        #for target_class in range(11):
        for target_class in tqdm(range(11), desc="Processing classes"):
            for data in dataloader:
                condition = (data[1] == target_class) & (data[2] == 18)
                target_indices = torch.nonzero(condition, as_tuple=False).squeeze()
                input = data[0]
                input_label = data[1]
                if target_indices.numel() > 0:
                    data_input = cuda(input[target_indices],self.cuda)
                    data_label = cuda(input_label[target_indices],self.cuda)
                    hidden_feature = self.netF(cuda(data_input.unsqueeze(1),self.cuda))
                    output = self.netD(hidden_feature, False, 1)
                    output_label = torch.max(output, 1, keepdim=False)[1]
                    for t, p in zip(data_label.view(-1), output_label.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1
        
        np.save(os.path.join(self.args.image_dir, 'target'),confusion_matrix.numpy())
        
        
            


        