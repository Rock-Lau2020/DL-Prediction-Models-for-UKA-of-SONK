# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:44:51 2024
sonk models
@author: HASEE
"""

import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F

from torchsummary import summary



class Contrastive_Loss(nn.Module):

    def __init__(self, margin=1.0):
        
        super(Contrastive_Loss,self).__init__()

        self.margin = margin
        

    def forward(self, x1, x2, label):
        euclidean_distance = F.pairwise_distance(x1, x2, keepdim=True)

        euclidean_distance_2 = torch.pow(euclidean_distance, 2)
        euclidean_distance_cl = torch.clamp(self.margin - euclidean_distance, min=0.0)
        euclidean_distance_cl2 = torch.pow(euclidean_distance_cl,2)

        loss_contrastive = torch.mean(((1 - label) * euclidean_distance_2) +
                                      ((label) * euclidean_distance_cl2))

        return loss_contrastive


class Sonk_model(nn.Module):
    

    
    def __init__(self):
        
        super(Sonk_model,self).__init__()
        

        res_18 = torchvision.models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(res_18.children())[:-1])
        self.backbone.requires_grad_(False) 

        self.flatten_1 = nn.Flatten()

        self.con2d_1 = nn.Conv2d(512, 64, kernel_size=3, padding=(1, 1))
        self.act_1 = nn.LeakyReLU()
        self.con2d_2 = nn.Conv2d(64, 64, kernel_size=3, padding=(1, 1))
        self.act_2 = nn.LeakyReLU()
        self.con2d_3 = nn.Conv2d(64, 64, kernel_size=3, padding=(1, 1))
        self.act_3 = nn.LeakyReLU()
        
        
    def forward(self, x1, x2):


        s4 = self.backbone(x1)
        
        
        s4_1 = self.con2d_1(s4)
        s4_1 = self.act_1(s4_1)
        
        s4_2 = self.con2d_2(s4_1)
        s4_2 = self.act_2(s4_2)
        
        s4_3 = self.con2d_3(s4_2)
        s4_3 = self.act_3(s4_3)

        y1 = self.flatten_1(s4_3)
        
        
        
        

        s5 = self.backbone(x2)
        
        s5_1 = self.con2d_1(s5)
        s5_1 = self.act_1(s5_1)
        
        s5_2 = self.con2d_2(s5_1)
        s5_2 = self.act_1(s5_2)
        
        s5_3 = self.con2d_3(s5_2)
        s5_3 = self.act_1(s5_3)
        
        y2 = self.flatten_1(s5_3)
        
        
        return y1, y2

class Res_model(nn.Module):

        def __init__(self):
            super(Res_model, self).__init__()


            res_18 = torchvision.models.resnet18(pretrained=False)
            self.backbone = nn.Sequential(*list(res_18.children())[:-1])


            self.flatten_1 = nn.Flatten()
            self.linear = nn.Linear(512,2)
            self.softmax = nn.Softmax()

        def forward(self,x1):

            x1 = self.backbone(x1)
            x1 = self.flatten_1(x1)
            x1 = self.linear(x1)
            y = self.softmax(x1)

            return y

class Vgg_model(nn.Module):
        def __init__(self):
            super(Vgg_model, self).__init__()


            res_18 = torchvision.models.vgg16(pretrained=False)
            self.backbone = nn.Sequential(*list(res_18.children())[:-1])

            self.conv = nn.Conv2d(512, 64, kernel_size=7, padding=(0, 0))
            self.flatten_1 = nn.Flatten()
            self.linear = nn.Linear(64, 2)
            self.softmax = nn.Softmax()

        def forward(self, x1):

            x1 = self.backbone(x1)
            x1 = self.conv(x1)
            x1 = self.flatten_1(x1)
            x1 = self.linear(x1)
            y = self.softmax(x1)

            return y




if __name__ == "__main__":


    model_1 = Sonk_model()
    model_1.cuda()
    
    loss_fn = Contrastive_Loss()
    
    model_1.cuda()
    summary(model_1, (3,224,224))
    



