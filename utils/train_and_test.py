import os

print("当前运行路径: ", os.getcwd())

import numpy as np
import torch
from torch import nn, optim
import visdom
import matplotlib.pyplot as plt

from utils.datasets import mnist, cifar10


class TrainTest:
    def __init__(self, model, dataset="mnist"):
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 有 cuda 则使用 GPU
        print("gpu available:", torch.cuda.is_available(), "use:", self.device)

        self.model.to(self.device)
        self.train_loader, self.test_loader = \
            mnist(batch_size=64) if dataset.lower() == "mnist" else cifar10(batch_size=64)

    def train(self,
              epochs=10,
              optimize="Adam",
              save=True,
              save_dst="./models",
              save_name="resnet",
              visloss=False):
        global acc
        if visloss:
            vis = visdom.Visdom()  # 用 visdom 实时可视化loss曲线
            counter = 1
            print("visdom实时可视化loss已开启, 使用: python -m visdom.server")

        criterion = nn.CrossEntropyLoss()  # 包含了 softmax 层
        if optimize.lower() == "adam":
            optimizer = optim.Adam(self.model.parameters())
        else:
            optimizer = optim.SGD(self.model.parameters(), lr=0.01,
                                  momentum=0.05)  # SGD 对 batch-size 很敏感，64 是最好的；lr=0.01, momentum=0.5

        optim_name = optimizer.__str__().split('(')[0].strip().lower()
        print("optimizer name:", optim_name)

        acc_lst = []
        loss_lst_ = []

        for epoch in range(epochs):
            TP = 0
            loss_lst = []
            total = 0
            for i, (imgs, labels) in enumerate(self.train_loader):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                y_pred = self.model(imgs)
                # print("x:", x.shape, "y:", labels.shape, "y_pred:", y_pred.shape)
                total += labels.size(0)

                loss = criterion(y_pred, labels)
                loss_lst.append(loss.item())

                if self.device != 'cpu':
                    TP += torch.sum((labels.flatten() == torch.argmax(y_pred.data, dim=1))).cpu().numpy()
                else:
                    TP += torch.sum((labels.flatten() == torch.argmax(y_pred.data, dim=1))).numpy()
                # visdom 实时绘制 loss 曲线
                if i % 50 == 49 and visloss:
                    vis.line(Y=[np.mean(loss_lst[i-49:])], X=[counter*50], win='loss', update='append')
                    vis.line(Y=[TP/total], X=[counter*50], win='acc', update='append')
                    counter += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # TP = TP.cpu().data.numpy() if self.device != "cpu" else TP.numpy()
            acc = TP / total
            acc_lst.append(acc)
            loss_lst_.extend(loss_lst)
            print("epoch:", epoch, "loss:", np.mean(loss_lst), "acc:", round(acc, 3),
                  f"TP: {TP} / {total}")

        # 保存模型
        if save:
            torch.save(self.model.state_dict(), os.path.join(save_dst, f"{save_name}_{optim_name}_acc_{round(acc, 2)}.pth"))
            print(f"model saved in {save_dst}")
            np.savez(os.path.join(save_dst, f"{save_name}_{optim_name}_accloss.npz"), acc=acc_lst, loss=loss_lst_)
            print(f"npz saved in {save_dst}")

        plt.plot(np.arange(len(acc_lst)), acc_lst)
        plt.tight_layout()
        plt.grid()
        plt.show()

    def test(self, load_dst="./models/acc_0.99.pth"):
        TP = 0
        total = 0
        self.model.load_state_dict(torch.load(load_dst, map_location=self.device))

        for i, (imgs, labels) in enumerate(self.test_loader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            total += labels.size(0)
            with torch.no_grad():
                y_pred = self.model(imgs)

            TP += torch.sum(labels.flatten() == torch.argmax(y_pred.data, dim=1))  # .sum().item()

        TP = TP.cpu().data.numpy() if self.device != "cpu" else TP.data.numpy()
        acc = TP / total
        print("acc:", round(acc, 4), f"TP: {TP} / {total}")

    def draw(self, device="cpu", load_dst="./models/google_adam_acc_0.99.pth"):
        self.model.load_state_dict(torch.load(load_dst, map_location=self.device))
        self.model.to(device)

        examples = enumerate(self.test_loader)
        _, (imgs, labels) = next(examples)
        imgs, labels = imgs.to(device), labels.to(device)

        with torch.no_grad():
            y_pred = self.model(imgs)

        for i in range(12):
            plt.subplot(3, 4, i + 1)
            plt.tight_layout()
            plt.imshow(imgs[i][0], cmap='gray', interpolation='none')
            plt.title("pred:{},label:{}".format(
                y_pred.data.max(1, keepdim=True)[1][i].item(), labels[i]))
            plt.xticks([])
            plt.yticks([])
        plt.show()
