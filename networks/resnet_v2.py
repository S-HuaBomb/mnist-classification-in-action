import time
import sys
sys.path.append(r"pytorch_src/mnist-classification-in-action")

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


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.bn = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)  # 特征图大小没变
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = self.conv1(F.relu(self.bn(x)))
        y = self.conv2(F.relu(self.bn(y)))

        return x + y  # 不是拼接，所以返回通道数还是 channel


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

    print(model)

    train_test = TrainTest(model, dataset='mnist')

    s = time.time()

    # train_test.train(epochs=8, optimize="Adam", save=True, save_dst="./models", save_name="resnet")

    load_dst = "./models/resnet_adam_acc_0.96.pth"
    train_test.test(load_dst=load_dst)
    train_test.draw(load_dst=load_dst)

    print("time cost:", time.time() - s)
    # gpu time cost: 179.4
    # cpu time cost: 308.2 in 5 epochs
