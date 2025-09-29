import torch.nn as nn
from torch.autograd import Function
import torch
from torchvision import models

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight,mode = 'fan_out',nonlinearity='relu') 
        if m.bias is not None:
            nn.init.constant_(m.bias,0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight,mode = 'fan_out',nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias,0) 
   

class Feedforward(nn.Module):
    def __init__(self):
        super(Feedforward, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,256,[1,4],[1,4]),nn.ReLU(),nn.Dropout(0.5),
            nn.Conv2d(256,256,[2,2],[1,2]),nn.ReLU(),nn.Dropout(0.5),
            nn.Conv2d(256,80,[1,3],1),nn.ReLU(),nn.Dropout(0.5),
            nn.Conv2d(80,80,[1,3],1),nn.ReLU(),nn.Dropout(0.5)
                )
        self.linear1 = nn.Sequential(nn.Linear(80*124, 512),nn.ReLU(),nn.Dropout(0.5))  
        self.linear2 = nn.Sequential(nn.Linear(512, 128),nn.ReLU(),nn.Dropout(0.5))   
        #self.linear1 = nn.Sequential(nn.Linear(960, 512),nn.ReLU(),nn.Dropout(0.5))                  
        
        self.apply(init_weights)
        
    def forward(self,x):
        y = self.conv(x)
        feature = y.view(y.size(0), -1)
        feature = self.linear1(feature)  
        feature = self.linear2(feature)  
        return feature


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.linear = nn.Sequential(
            #nn.Linear(80*124, 512),nn.ReLU(),nn.Dropout(0.5),
            #nn.Linear(512, 128),nn.ReLU(),nn.Dropout(0.5), 
            nn.Linear(128,11)
            )
    
        self.apply(init_weights)
        
    def forward(self, f1, reverse, alpha):    
        if reverse:
            f1 = GRL.apply(f1, alpha)
        y = self.linear(f1) 
        return y   







class WDGRL(nn.Module):
    def __init__(self):
        super(WDGRL,self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(128, 512),nn.ReLU(),
            nn.Linear(512, 128),nn.ReLU(),
            nn.Linear(128,1)
                )
        self.apply(init_weights)
    def forward(self,feature, reverse, alpha):
        if reverse:
            feature = GRL.apply(feature, alpha)
        y = self.linear(feature)
        return y

class DANN(nn.Module):
    def __init__(self):
        super(DANN,self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(128, 512),nn.ReLU(),
            nn.Linear(512, 128),nn.ReLU(),
            nn.Linear(128,2)
                )
        self.apply(init_weights)
    def forward(self,feature, reverse, alpha):
        if reverse:
            feature = GRL.apply(feature, alpha)
        y = self.linear(feature)
        return y
    

class GRL(Function):
    @staticmethod
    def forward(ctx, x, alpha):  
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class ResNet18Feature(nn.Module):
    def __init__(self):
        super(ResNet18Feature, self).__init__()
        model_resnet18 = models.resnet18()
        self.conv1 = nn.Conv2d(1,64,[1,4],[1,4])
        self.bn1 = model_resnet18.bn1
        self.relu = model_resnet18.relu
        self.maxpool = model_resnet18.maxpool
        self.layer1 = model_resnet18.layer1
        self.layer2 = model_resnet18.layer2
        self.layer3 = model_resnet18.layer3
        self.layer4 = model_resnet18.layer4
        self.avgpool = model_resnet18.avgpool
        self._out_features = model_resnet18.fc.in_features
        self.linear = nn.Linear(512, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
    



    def output_size(self):
        return self._out_features