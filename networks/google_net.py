import sys
import time

sys.path.append(r"pytorch_src/mnist-classification-in-action")

import torch
from torch import nn
from torch.nn import functional as F

from utils.train_and_test import TrainTest


class Inception(nn.Module):
    """
    由于特征图要拼接，所以需要通过设置padding、stride来保证卷积过后特征图大小不变；
    由于在Inception块之前还有卷积层，所以输入的通道数不是一样的，需要把输入通道数作为一个参数。

    :return:
      输出的通道数为 16+24×3=88，故返回 N,88,*,* 的特征图
    """

    def __init__(self, in_channels):
        super(Inception, self).__init__()

        self.pool_conv1x1 = nn.Conv2d(in_channels, 24, kernel_size=1)  # 池化+一个1×1卷积，输出24通道

        self.conv1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)  # 三个同样的1×1卷积，输出16通道

        self.conv3x3_16 = nn.Conv2d(16, 24, kernel_size=3, padding=1)  # 输入为16通道的3×3卷积，输出24通道
        self.conv3x3_24 = nn.Conv2d(24, 24, kernel_size=3, padding=1)  # 输入为24通道的3×3卷积，输出24通道

        self.conv5x5 = nn.Conv2d(16, 24, kernel_size=5, padding=2)  # 一个5×5卷积，输出24通道

    def forward(self, x):
        out1 = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)  # N,*,,
        out1 = self.pool_conv1x1(out1)  # N,24,,

        out2 = self.conv1x1(x)  # N,16,,

        out3 = self.conv1x1(x)  # N,16,,
        out3 = self.conv5x5(out3)  # N,24,,

        out4 = self.conv1x1(x)  # N,16,,
        out4 = self.conv3x3_16(out4)  # N,24,,
        out4 = self.conv3x3_24(out4)  # N,24,,

        outputs = (out1, out2, out3, out4)

        return torch.cat(outputs, dim=1)  # N,88,,


class GoogleNet(nn.Module):
    def __init__(self):
        super(GoogleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # return N,10,,
        self.incep1 = Inception(in_channels=10)  # return N,88,,

        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)
        self.incep2 = Inception(in_channels=20)  # return N,88,,

        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(1408, 10)

    def forward(self, x):
        batch_size = x.size(0)

        x = F.relu(self.mp(self.conv1(x)))  # N,10,12,12
        x = self.incep1(x)  # N,88,12,12
        x = F.relu(self.mp(self.conv2(x)))  # N,20,4,4
        x = self.incep2(x)  # N,88,4,4

        x = x.view(batch_size, -1)
        x = self.fc(x)

        return x


model = GoogleNet()

if __name__ == '__main__':
    # 模拟数据输入网络以便得到linear层的输入通道数
    # input = torch.randn(1, 1, 28, 28)
    # output = model(input)
    # print(output.shape)

    train_test = TrainTest(model, dataset='mnist')

    s = time.time()

    # train_test.train(epochs=10, optimize="Adam", save_dst="./models", save_name="google")

    load_dst = "models/google_adam_acc_0.99.pth"
    # train_test.test(load_dst=load_dst)
    train_test.draw(load_dst=load_dst)

    print("time cost:", time.time() - s)
    # gpu time cost: 260.8 s
    # cpu time cost: 357.85
