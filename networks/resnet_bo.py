import time
import sys
sys.path.append(r"pytorch_src/mnist-classification-in-action")

import torch
from torch import nn
from torch.nn import functional as F

from utils.train_and_test import TrainTest


class BottleNeck(nn.Module):
    """
    bottlenect 对于高维的训练数据比 residual block 更高效
    """

    def __init__(self, channels):
        super(BottleNeck, self).__init__()

        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=1)  # 1×1卷积用于降维
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # 前后的1×1卷积减少了3×3卷积的计算量
        self.conv3 = nn.Conv2d(64, channels, kernel_size=1)  # 1×1卷积用于升维

    def forward(self, x):
        y = self.conv1(F.relu(self.bn1(x)))  # N,64,,
        y = self.conv2(F.relu(self.bn2(y)))  # N,64,,
        y = self.conv3(F.relu(self.bn2(y)))  # N,channels,,

        return x + y


class ResNet(nn.Module):
    """
    以为数据集的图像大小是 224×224
    """
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=3)
        self.bneck1 = BottleNeck(channels=256)

        self.conv2 = nn.Conv2d(256, 256, kernel_size=3)
        self.bneck2 = BottleNeck(channels=256)

        self.conv3 = nn.Conv2d(256, 64, kernel_size=1)

        self.mp1 = nn.MaxPool2d(2)
        self.ap = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(61504, 100)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.mp1(F.relu(self.conv1(x)))  # N,256,127,127
        x = self.bneck1(x)  # N,256,127,127

        x = self.mp1(F.relu(self.conv2(x)))  # N,256,62,62
        x = self.bneck2(x)  # N,256,62,62

        x = self.ap(self.conv3(x))  #  N,64,31,31

        x = x.view(batch_size, -1)  # N,61504
        x = self.fc1(x)

        return x


model = ResNet()

if __name__ == '__main__':
    # # 模拟数据输入网络以便得到linear层的输入通道数
    # input = torch.randn(1, 3, 256, 256)
    # output = model(input)
    # print(output.shape)

    train_test = TrainTest(model)

    s = time.time()

    # train_test.train(epochs=5, optimize="Adam", save=True, save_dst="./models", save_name="resnet")

    load_dst = "./models/resnet_adam_acc_1.0.pth"
    train_test.test(load_dst=load_dst)
    train_test.draw(load_dst=load_dst)

    print("time cost:", time.time() - s)
    # gpu time cost: 179.4
    # cpu time cost: 308.2 in 5 epochs
