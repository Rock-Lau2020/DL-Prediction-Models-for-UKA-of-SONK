import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import Sonk_model,Contrastive_Loss, Res_model, Vgg_model
from dataset import Sonk_Dataset, Sonk_Dataset_val, Other_dataset


def get_train_vector():
    dataset_train = Sonk_Dataset_val()
    train_dataloader = DataLoader(dataset_train, batch_size=126, shuffle=False, num_workers=0)

    model = torch.load(".\\model\\model_best.pt")
    model.eval()

    sonk_tensor_list = []
    no_sonk_tensor_list = []

    for i, data in enumerate(train_dataloader):

        label = Variable(data[0].cuda())
        img_batch = Variable(data[1].cuda())

        y1, y2 = model(img_batch, img_batch)

        for i in range(len(img_batch)):
            if label[i] > 0.5:
                sonk_tensor_list.append(y1[i].unsqueeze(0))
            else:
                no_sonk_tensor_list.append(y1[i].unsqueeze(0))

    no_sonk_tensor = torch.cat(no_sonk_tensor_list, axis=0)
    sonk_tensor = torch.cat(sonk_tensor_list, axis=0)

    no_sonk_vector = torch.mean(no_sonk_tensor, dim=0).detach().cpu().numpy()
    sonk_vector = torch.mean(sonk_tensor, dim=0).detach().cpu().numpy()

    return no_sonk_tensor.detach().cpu().numpy(), sonk_tensor.detach().cpu().numpy(), no_sonk_vector, sonk_vector
    
def get_test_vector():
    
    dataset_test = Sonk_Dataset(is_test=True)
    test_dataloader = DataLoader(dataset_test, batch_size=126, shuffle=False, num_workers=0)
    
    model = torch.load(".\\model\\model_best.pt")
    model.eval()
    
    sonk_tensor_list = []
    no_sonk_tensor_list = []
    
    for i, data in enumerate(test_dataloader):
        
        label = Variable(data[0].cuda())
        img_batch = Variable(data[1].cuda())
        
        y1,y2 = model(img_batch, img_batch)
        
        for i in range(len(img_batch)):
            if label[i]>0.5:
                sonk_tensor_list.append(y1[i].unsqueeze(0))
            else:
                no_sonk_tensor_list.append(y1[i].unsqueeze(0))
    
    no_sonk_tensor = torch.cat(no_sonk_tensor_list,axis=0)
    sonk_tensor = torch.cat(sonk_tensor_list,axis=0)
    
    no_sonk_vector = torch.mean(no_sonk_tensor, dim=0).detach().cpu().numpy()
    sonk_vector = torch.mean(sonk_tensor, dim=0).detach().cpu().numpy()
    
    return no_sonk_tensor.detach().cpu().numpy(), sonk_tensor.detach().cpu().numpy(), no_sonk_vector, sonk_vector


if __name__ == "__main__":

    dataset_train = Other_dataset(is_test=False)
    train_dataloader = DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=0)
    dataset_test = Other_dataset(is_test=True)
    test_dataloader = DataLoader(dataset_test, batch_size=16, shuffle=True, num_workers=0)


    no_sonk_tensor_test, sonk_tensor_test, no_sonk_vector_test, sonk_vector_test = get_test_vector()
    no_sonk_tensor_train, sonk_tensor_train, no_sonk_vector_train, sonk_vector_train = get_train_vector()



    check_list = []
    pre_pm = []

    loss_fn = Contrastive_Loss()

    for i in range(len(no_sonk_tensor_test)):

        dis_1 = np.linalg.norm(no_sonk_tensor_test[i] - no_sonk_vector_train)
        dis_2 = np.linalg.norm(no_sonk_tensor_test[i] - sonk_vector_train)

        conf = dis_2/(dis_1+dis_2)

        #pre_pm.append(1-dis_2)
        pre_pm.append(1-conf)

        if dis_1 < dis_2:
            check_list.append(0)
        else:
            check_list.append(1)

    for i in range(len(sonk_tensor_test)):

        dis_1 = np.linalg.norm(sonk_tensor_test[i] - no_sonk_vector_train)
        dis_2 = np.linalg.norm(sonk_tensor_test[i] - sonk_vector_train)

        #pre_pm.append(1-dis_2)
        conf = dis_1/(dis_1+dis_2)
        pre_pm.append(conf)

        if dis_1 < dis_2:
            check_list.append(0)
        else:
            check_list.append(1)
