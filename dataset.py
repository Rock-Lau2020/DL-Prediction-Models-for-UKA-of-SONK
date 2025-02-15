# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:49:05 2024
dataset for sonk model
@author: user
"""

import glob
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from PIL import Image



class Sonk_Dataset(Dataset):
    
    def __init__(self,image_dir=None,is_test=False,
                class_names=None,image_size=(224,224),shuffle=True):
        
        super(Sonk_Dataset,self).__init__()
        
        self.shuffle = shuffle


        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.58116895, 0.58116895, 0.58116895],
                                     std=[0.12170879, 0.12170879, 0.12170879])])
    
        if not class_names:
            self.class_names = ['no_sonk','sonk']
        else:
            self.class_names = class_names
        
        if not image_size:
            self.image_size = (224,224)
        else:
            self.image_size = image_size
       
        if not is_test:
            self.mode = "train"
            
            
            if not image_dir:
                self.image_dir = r".\train_set"
            else:
                self.image_dir = image_dir
                
            self.no_sonk_path = sorted(glob.glob(self.image_dir+'\\'+ "no_sonk" +'\\'+'/*.JPG'))
            self.sonk_path = sorted(glob.glob(self.image_dir+'\\'+ "sonk" +'\\'+'/*.JPG'))
            

            self.diff_sample_list = []
            for i in range(len(self.no_sonk_path)):
                for j in range(len(self.sonk_path)):
                    diff_sample = [1, self.no_sonk_path[i], self.sonk_path[j]]
                    self.diff_sample_list.append(diff_sample)
                    
            self.same_sample_list = []
            for i in range(len(self.no_sonk_path)-1):
                for j in range(i+1,len(self.no_sonk_path)):
                    same_sample = [0, self.no_sonk_path[i], self.no_sonk_path[j]]
                    self.same_sample_list.append(same_sample)
                
            for i in range(len(self.sonk_path)-1):
                for j in range(i+1,len(self.sonk_path)):
                    same_sample = [0, self.sonk_path[i], self.sonk_path[j]]
                    self.same_sample_list.append(same_sample)
            
            
            self.sample_list = self.diff_sample_list + self.same_sample_list
            
            self.datalen = len(self.sample_list)
                    
                
            
        else:
            self.mode = "test"
            
            if not image_dir:
                self.image_dir = r".\test_set"
            else:
                self.image_dir = image_dir
                
            self.no_sonk_path = sorted(glob.glob(self.image_dir+'\\'+ "no_sonk" +'\\'+'/*.JPG'))
            self.sonk_path = sorted(glob.glob(self.image_dir+'\\'+ "sonk" +'\\'+'/*.JPG'))
            
            
            self.sample_list = []
            
            for p in self.no_sonk_path:
                self.sample_list.append([0,p])
                
            for p in self.sonk_path:
                self.sample_list.append([1,p])
            
            self.datalen = len(self.sample_list)
            
            
    def auto_resize(self,img):
        
        width = img.size[0]
        height = img.size[1]
        
        if width >= height:
            new_height = int(height/(width/self.image_size[0]))
            img = img.resize((self.image_size[0], new_height),Image.ANTIALIAS)
            img = np.array(img)
            above_height_add = int(0.5*(self.image_size[1]-new_height))
            under_height_add = self.image_size[1]-new_height-above_height_add
            img = np.pad(img,((above_height_add,under_height_add),(0,0),(0,0)),'constant')
            img = Image.fromarray(img)
            
        else:
            new_width = int(width/(height/self.image_size[1]))
            img = img.resize((new_width, self.image_size[1]),Image.ANTIALIAS)
            img = np.array(img)
            left_width_add = int(0.5*(self.image_size[0]-new_width))
            right_width_add = self.image_size[0]-new_width-left_width_add
            img = np.pad(img,((0,0),(left_width_add,right_width_add),(0,0)),'constant')
            img = Image.fromarray(img)
            
        return img
        
            
        
    def __len__(self):
        return int(self.datalen)
    
    def __getitem__(self,idx):
        
        if self.mode == "train":
            """
            img_path = self.filepath[idx]
            label = self.class_names.index(img_path.split("\\")[-2])
            #label_name = img_path.split("\\")[-2]
            img = Image.open(img_path).convert('RGB')
            img_ = self.auto_resize(img)
            img = self.transform(img_)
            """
            sample = self.sample_list[idx]
            label = sample[0]
            path_1 = sample[1]
            path_2 = sample[2]
            
            img_1 = Image.open(path_1).convert('RGB')
            img_1 = self.auto_resize(img_1)
            
            img_2 = Image.open(path_2).convert('RGB')
            img_2 = self.auto_resize(img_2)
            
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)
            
            
            return label,img_1,img_2
        
        else:
            
            sample = self.sample_list[idx]
            label = sample[0]
            path = sample[1]
            img = Image.open(path).convert('RGB')
            img = self.auto_resize(img)
            img = self.transform(img)
            
            return label,img
            #return img


class Sonk_Dataset_val(Dataset):
    
    def __init__(self,image_dir=None, image_size=(224,224),shuffle=True):
        
        super(Sonk_Dataset_val,self).__init__()
        
        self.shuffle = shuffle

        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.58116895, 0.58116895, 0.58116895],
                                     std=[0.12170879, 0.12170879, 0.12170879])])
        
        if not image_size:
            self.image_size = (224,224)
        else:
            self.image_size = image_size
             
        
        if not image_dir:
            self.image_dir = r".\train_set"
        else:
            self.image_dir = image_dir
            
        self.no_sonk_path = sorted(glob.glob(self.image_dir+'\\'+ "no_sonk_e" +'\\'+'/*.JPG'))
        self.sonk_path = sorted(glob.glob(self.image_dir+'\\'+ "sonk_e" +'\\'+'/*.JPG'))
        
        
        self.sample_list = []
        
        for p in self.no_sonk_path:
            self.sample_list.append([0,p])
                
        for p in self.sonk_path:
            self.sample_list.append([1,p])
            
        self.datalen = len(self.sample_list)
            
            
    def auto_resize(self,img):
        
        width = img.size[0]
        height = img.size[1]
        
        if width >= height:
            new_height = int(height/(width/self.image_size[0]))
            img = img.resize((self.image_size[0], new_height),Image.ANTIALIAS)
            img = np.array(img)
            above_height_add = int(0.5*(self.image_size[1]-new_height))
            under_height_add = self.image_size[1]-new_height-above_height_add
            img = np.pad(img,((above_height_add,under_height_add),(0,0),(0,0)),'constant')
            img = Image.fromarray(img)
            
        else:
            new_width = int(width/(height/self.image_size[1]))
            img = img.resize((new_width, self.image_size[1]),Image.ANTIALIAS)
            img = np.array(img)
            left_width_add = int(0.5*(self.image_size[0]-new_width))
            right_width_add = self.image_size[0]-new_width-left_width_add
            img = np.pad(img,((0,0),(left_width_add,right_width_add),(0,0)),'constant')
            img = Image.fromarray(img)
            
        return img
        
            
        
    def __len__(self):
        return int(self.datalen)
    
    def __getitem__(self,idx):
            
        sample = self.sample_list[idx]
        label = sample[0]
        path = sample[1]
        img = Image.open(path).convert('RGB')
        img = self.auto_resize(img)
        img = self.transform(img)
        
        return label,img


class Other_dataset(Dataset):
    
    def __init__(self,image_dir=None,is_test=False,
                class_names=None,image_size=(224,224),shuffle=True):
        
        super(Other_dataset,self).__init__()
        
        self.shuffle = shuffle
        #self.image_dir = r"..\data\msta_train" image_dir

        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.58116895, 0.58116895, 0.58116895],
                                     std=[0.12170879, 0.12170879, 0.12170879])])
        
        if not image_size:
            self.image_size = (224,224)
        else:
            self.image_size = image_size
       
        if not is_test:
            self.mode = "train"
            
            
            if not image_dir:
                self.image_dir = r".\train_set"
            else:
                self.image_dir = image_dir
                
            self.no_sonk_path = sorted(glob.glob(self.image_dir+'\\'+ "no_sonk_e" +'\\'+'/*.JPG'))
            self.sonk_path = sorted(glob.glob(self.image_dir+'\\'+ "sonk_e" +'\\'+'/*.JPG'))
            
            
            self.sample_list = []
            
            for p in self.no_sonk_path:
                self.sample_list.append([0,p])
                
            for p in self.sonk_path:
                self.sample_list.append([1,p])
            
            self.datalen = len(self.sample_list)
                
            
        else:
            self.mode = "test"
            
            if not image_dir:
                self.image_dir = r".\test_set"
            else:
                self.image_dir = image_dir
                
            self.no_sonk_path = sorted(glob.glob(self.image_dir+'\\'+ "no_sonk" +'\\'+'/*.JPG'))
            self.sonk_path = sorted(glob.glob(self.image_dir+'\\'+ "sonk" +'\\'+'/*.JPG'))
            
            
            self.sample_list = []
            
            for p in self.no_sonk_path:
                self.sample_list.append([0,p])
                
            for p in self.sonk_path:
                self.sample_list.append([1,p])
            
            self.datalen = len(self.sample_list)
            
            
    def auto_resize(self,img):
        
        width = img.size[0]
        height = img.size[1]
        
        if width >= height:
            new_height = int(height/(width/self.image_size[0]))
            img = img.resize((self.image_size[0], new_height),Image.ANTIALIAS)
            img = np.array(img)
            above_height_add = int(0.5*(self.image_size[1]-new_height))
            under_height_add = self.image_size[1]-new_height-above_height_add
            img = np.pad(img,((above_height_add,under_height_add),(0,0),(0,0)),'constant')
            img = Image.fromarray(img)
            
        else:
            new_width = int(width/(height/self.image_size[1]))
            img = img.resize((new_width, self.image_size[1]),Image.ANTIALIAS)
            img = np.array(img)
            left_width_add = int(0.5*(self.image_size[0]-new_width))
            right_width_add = self.image_size[0]-new_width-left_width_add
            img = np.pad(img,((0,0),(left_width_add,right_width_add),(0,0)),'constant')
            img = Image.fromarray(img)
            
        return img
        
            
        
    def __len__(self):
        return int(self.datalen)
    
    def __getitem__(self,idx):
            
        sample = self.sample_list[idx]
        label = sample[0]
        path = sample[1]
        img = Image.open(path).convert('RGB')
        img = self.auto_resize(img)
        img = self.transform(img)
        
        return label,img
    
    
    


def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 
    :return: (mean, std)
    '''
    #train_data = Imgdataset(jpg_path)
    
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].float().mean()
            std[d] += X[:, d, :, :].float().std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    
    print("mean: ", list(mean.numpy()))
    print("std: ", list(std.numpy()))
    
    return mean.numpy(), std.numpy()


if __name__ == "__main__":
    
    dataset = Sonk_Dataset(is_test=False)
    #m, std = getStat(dataset)
    
    
    label, img, img_ = dataset[0]
    #img = np.array(img)
    img_1 = np.array(img_)



    
    
    
    
    
    
    
    
    
    
    
    
    
