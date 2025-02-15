import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import visdom
from model import Sonk_model,Contrastive_Loss, Res_model, Vgg_model
from dataset import Sonk_Dataset, Sonk_Dataset_val, Other_dataset


class Visualizer(object):
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}

    def plot_many_stack(self, d):
        '''
        self.plot('loss',1.00)
        '''
        name = list(d.keys())
        name_total = " ".join(name)
        x = self.index.get(name_total, 0)
        val = list(d.values())
        if len(val) == 1:
            y = np.array(val)
        else:
            y = np.array(val).reshape(-1, len(val))
        # print(x)
        self.vis.line(Y=y, X=np.ones(y.shape) * x,
                      win=str(name_total),  # unicode
                      opts=dict(legend=name,
                                title=name_total),
                      update=None if x == 0 else 'append'
                      )
        self.index[name_total] = x + 1

    def plot_heatmap(self, tensor):
        self.vis.images(
            tensor,
            nrow=8,
            win=str('heatmap'),
            opts={'title': 'heatmap'}
        )


class Sonk_Trainer(object):

    def __init__(self, model, train_loader, val_loader, optimizer=None, log_dir=r'.\log'):

        if model == None:
            self.model = Sonk_model()
        else:
            self.model = model

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.log_dir = log_dir
        self.loss_fn = Contrastive_Loss()
        self.vis = Visualizer(env="sonk")

        if optimizer == None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=0.001)
        else:
            self.optimizer = optimizer

        self.model.cuda()

    def update_valid_loss(self, epoch):

        self.model.eval()

        val_loss_sum = 0
        l2_loss_sum = 0

        for i, data in enumerate(self.val_loader):

            label_target = Variable(data[0].cuda())
            input_img_batch = Variable(data[1].cuda())

            if label_target[0] == label_target[1]:
                label = 0
            else:
                label = 1

            # vector_target = Variable(data[2].cuda().double())
            # label_pre,vector_pre = self.model(input_img_batch)
            x1, x2, y1, y2 = self.model(input_img_batch)

            loss_fn = Contrastive_Loss(y1, y2, label)
            val_loss, l2_val = loss_fn()
            val_loss_sum += val_loss.cpu().data.numpy()
            l2_loss_sum += l2_val.cpu().data.numpy()

            pred1 = torch.max(x1, 1)[1]
            pred2 = torch.max(x2, 1)[1]
            if pred1 == pred2:
                label_pre = 0
            else:
                label_pre = 1
            results = label_pre == label
            correct_points = torch.sum(results.long())
            all_correct_points += correct_points
            all_points += results.size()[0]

        acc = all_correct_points.float() / all_points
        val_overall_acc = acc.cpu().data.numpy()
        val_loss = val_loss_sum / len(self.val_loader)
        l2_loss_val = l2_loss_sum / len(self.val_loader)
        print('l2_loss_val : ', l2_loss_val)
        print('val overall acc. : ', val_overall_acc)
        print('val loss : ', val_loss)

        self.model.train()

        return val_loss, l2_loss_val, val_overall_acc

    def run(self, n_epochs=400):

        best_val_loss = 1000000
        best_train_loss = 1000000

        self.model.train()

        for epoch in range(n_epochs):

            i_acc = 0
            loss_sum = 0.

            for i, data in enumerate(self.train_loader):

                label = Variable(data[0].cuda())
                img_1_batch = Variable(data[1].cuda())
                img_2_batch = Variable(data[2].cuda())

                self.optimizer.zero_grad()

                y1, y2 = self.model(img_1_batch, img_2_batch)

                loss = self.loss_fn(y1, y2, label)

                # loss = loss.double()
                loss.backward()
                self.optimizer.step()
                loss_ = loss.cpu().item()

                log_str = 'epoch %d, step %d: train_loss %.3f;' % (epoch + 1, i + 1, loss_)
                self.vis.plot_many_stack({"train_loss (timestep)": loss_})

                if (i + 1) % 1 == 0:
                    print(log_str)

                i_acc += 1
                loss_sum += loss_

            loss_per_epoch = loss_sum / i_acc
            self.vis.plot_many_stack({"train_loss (epoch)": loss_per_epoch})

            n = 1
            torch.save(self.model, ".\\model\\model_last.pt")

            if loss_per_epoch < best_train_loss:
                best_train_loss = loss_per_epoch
                torch.save(self.model, ".\\model\\model_best.pt")

        print("----------train end-----------")
        print("best train loss: ", best_train_loss)

        # update learning rate
        if epoch > 0 and (epoch + 1) % 10 == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.5


if __name__ == "__main__":
    dataset_train = Sonk_Dataset(is_test=False)
    train_dataloader = DataLoader(dataset_train, batch_size=126, shuffle=True, num_workers=0)
    dataset_test = Sonk_Dataset(is_test=True)
    test_dataloader = DataLoader(dataset_test, batch_size=126, shuffle=True, num_workers=0)
    model = Sonk_model()
    # loss = Contrastive_Loss
    trainer = Sonk_Trainer(model, train_dataloader, test_dataloader)
    trainer.run(n_epochs=300)
