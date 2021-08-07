import time
import sys
sys.path.append(r"pytorch_src/mnist-classification-in-action")

import torch
from torch import nn
from torch.nn import functional as F

from utils.train_and_test import TrainTest


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.bn = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)  # 特征图大小没变
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu(self.bn(self.conv1(x)))
        y = self.bn(self.conv1(y))

        return F.relu(x + y)  # 不是拼接，所以返回通道数还是 channel


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.rblock1 = ResidualBlock(channels=16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.rblock2 = ResidualBlock(channels=32)

        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.mp(F.relu(self.conv1(x)))  # N,16,12,12
        x = self.rblock1(x)  # N,16,12,12

        x = self.mp(F.relu(self.conv2(x)))  # N,32,4,4
        x = self.rblock2(x)  # N,32,4,4

        x = x.view(batch_size, -1)
        x = self.fc(x)

        return x


model = ResNet()

if __name__ == '__main__':
    # 模拟数据输入网络以便得到linear层的输入通道数
    # input = torch.randn(1, 1, 28, 28)
    # output = model(input)
    # print(output.shape)

    train_test = TrainTest(model, dataset='mnist')

    s = time.time()

    # train_test.train(epochs=1, optimize="Adam", save=True, save_dst="./models", save_name="resnet")

    # load_dst = "./models/resnet_adam_acc_1.0.pth"
    # train_test.test(load_dst=load_dst)
    # train_test.draw(load_dst=load_dst)

    print("time cost:", time.time() - s)
    # gpu time cost: 179.4
    # cpu time cost: 493.4 in 5 epochs
