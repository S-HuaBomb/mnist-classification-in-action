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
                           download=True)
test_set = datasets.MNIST(root="../datasets/mnist",
                          train=False,
                          transform=trans,
                          download=True)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=True)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(784, 512)  # MNIST 每个图像大小为 28*28=784
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 64)
        self.linear5 = nn.Linear(64, 10)
        self.activate = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.activate(self.linear3(x))
        x = self.activate(self.linear4(x))
        x = self.linear5(x)
        return x


model = Model()

def train(model, train_loader, save_dst="./models"):
    global acc

    criterion = nn.CrossEntropyLoss()  # 包含了 softmax 层
    optimizer = optim.Adam(model.parameters())  # SGD 对 batch-size 很敏感，64 是最好的；lr=0.01, momentum=0.5
    optim_name = optimizer.__str__().split('(')[0].strip()
    print("optimizer name:", optim_name)

    for epoch in range(5):
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
        print("epoch:", epoch, "loss:", np.mean(loss_lst), "acc:", round(acc, 3), f"TP: {TP} / {len(train_set)}")

    # 保存模型
    torch.save(model.state_dict(), os.path.join(save_dst, f"{optim_name}_acc_{round(acc, 2)}.pth"))
    print(f"model saved in {save_dst}")


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
    train(model, train_loader)
    # test(model, test_loader, load_dst="./models/SGD_acc_0.99.pth")
    # draw(model, test_loader, load_dst="./models/SGD_acc_0.99.pth")
