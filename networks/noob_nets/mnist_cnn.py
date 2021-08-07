import os
import copy

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt


trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_set = datasets.MNIST(root="../datasets/mnist",
                           train=True,
                           transform=trans,  # 原始是 PIL Image 格式
                           download=False)
test_set = datasets.MNIST(root="../datasets/mnist",
                          train=False,
                          transform=trans,
                          download=False)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=True)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 24, 1)
        self.conv2 = nn.Conv2d(24, 16, 3)
        self.conv3 = nn.Conv2d(16, 8, 3)
        self.linear1 = nn.Linear(32, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 10)
        self.mp = nn.MaxPool2d(2)
        self.activate = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)  # batch-size 所在维度不变，其它维展开成320，输入线性层
        x = self.activate(self.mp(self.conv1(x)))
        x = self.activate(self.mp(self.conv2(x)))
        x = self.activate(self.mp(self.conv3(x)))  # out: N, 8, 2, 2
        x = x.view(batch_size, -1)
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.linear3(x)
        return x


model = Model()

def train(model, train_loader, save_dst="./models"):
    global acc

    criterion = nn.CrossEntropyLoss()  # 包含了 softmax 层
    optimizer = optim.Adam(model.parameters())  # SGD 对 batch-size 很敏感，64 是最好的；lr=0.01, momentum=0.5
    optim_name = optimizer.__str__().split('(')[0].strip()
    print("optimizer name:", optim_name)

    acc_lst = []

    for epoch in range(10):
        TP = 0
        loss_lst = []
        for i, (imgs, labels) in enumerate(train_loader):
            y_pred = model(imgs)
            # print("x:", x.shape, "y:", labels.shape, "y_pred:", y_pred.shape)

            loss = criterion(y_pred, labels)
            loss_lst.append(loss.item())

            y_hat = copy.copy(y_pred)
            TP += torch.sum(labels.flatten() == torch.argmax(y_hat, dim=1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = TP.data.numpy() / len(train_set)
        acc_lst.append(acc)
        print("epoch:", epoch, "loss:", np.mean(loss_lst), "acc:", round(acc, 3), f"TP: {TP} / {len(train_set)}")

    # 保存模型
    torch.save(model.state_dict(), os.path.join(save_dst, f"{optim_name}_macc_{round(acc, 2)}.pth"))
    print(f"model saved in {save_dst}")

    plt.plot(np.arange(len(acc_lst)), acc_lst)
    plt.tight_layout()
    plt.grid()
    plt.show()


def test(model, test_loader, load_dst="./models/SGD_acc_0.99.pth"):
    TP = 0
    model.load_state_dict(torch.load(load_dst))

    for i, (imgs, labels) in enumerate(test_loader):
        with torch.no_grad():
            y_pred = model(imgs)
        # print("x:", x.shape, "y:", labels.shape, "y_pred:", y_pred.shape)

        y_hat = copy.copy(y_pred)
        TP += torch.sum(labels.flatten() == torch.argmax(y_hat, dim=1))  # .sum().item()
    acc = TP.data.numpy() / len(test_set)
    print("acc:", round(acc, 4), f"TP: {TP} / {len(test_set)}")


def draw(model, test_loader, load_dst="./models/SGD_acc_0.99.pth"):
    model.load_state_dict(torch.load(load_dst))

    examples = enumerate(test_loader)
    _, (imgs, labels) = next(examples)

    with torch.no_grad():
        y_pred = model(imgs)

    for i in range(30):
        plt.subplot(5, 6, i + 1)
        plt.tight_layout()
        plt.imshow(imgs[i][0], cmap='gray', interpolation='none')
        plt.title("p: {}".format(
            y_pred.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()


if __name__ == '__main__':
    # train(model, train_loader)

    load_dst = "./models/Adam_macc_0.98.pth"
    test(model, test_loader, load_dst=load_dst)
    draw(model, test_loader, load_dst=load_dst)
