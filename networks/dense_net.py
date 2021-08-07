import time
import sys
sys.path.append(r"pytorch_src/mnist-classification-in-action")
from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F

from utils.train_and_test import TrainTest


def transition_layer(input, in_channels, out_channels):
    """
    过渡层，在 Dense Block 和 Dense Block 之间，把前一个的输出通道和长宽减半;
    注意这里直接调用了网络来计算，所以要单独把参数转移到cuda.
    :param in_channels: 前一个的输出通道
    :param out_channels: 输出通道减半
    :return:
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 有 cuda 则使用 GPU
    blk = nn.Sequential(OrderedDict([
        ('bn', nn.BatchNorm2d(in_channels)),
        ('relu', nn.ReLU()),
        ('conv1x1', nn.Conv2d(in_channels, out_channels, kernel_size=1)),
        ('avgpool', nn.AvgPool2d(2))
    ]))
    blk.to(device)
    return blk(input)


def conv_block(in_channels, out_channels, bo=False):
    """
    Dense Block 的基本组件
    :param in_channels:
    :param out_channels: growth rate k
    :return:
    """
    in_channels_ = out_channels * 2 if bo else in_channels

    bottlenect_layers = nn.Sequential(OrderedDict([
        ('bn0', nn.BatchNorm2d(in_channels)),
        ('relu0', nn.ReLU()),
        ('conv1x1', nn.Conv2d(in_channels, in_channels_, kernel_size=1)),
    ]))

    basic_blk = nn.Sequential(OrderedDict([
        ('bn1', nn.BatchNorm2d(in_channels_)),
        ('relu1', nn.ReLU()),
        ('conv3x3', nn.Conv2d(in_channels_, out_channels, kernel_size=3, padding=1)),
    ]))

    blk = nn.Sequential()
    if bo:
        blk.add_module('bottleneck', bottlenect_layers)
    blk.add_module('basic_blk', basic_blk)
    return blk


class DenseBlock(nn.Module):
    """
    由 conv_blk 组成
    :param in_channels: 动态的，由上一个 Dense Block 和过渡层决定
    :param out_channels: growth rate k
    :param dense_blk_num: 由多少 conv_blk 组成一个 Dense Block
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_blk_num=4,
                 bo=False,
                 transition=True):
        super(DenseBlock, self).__init__()
        self.transition = transition
        self.net = nn.Sequential()
        for i in range(conv_blk_num):
            in_channels_ = in_channels + i * out_channels
            self.net.add_module(f'conv_blk_{i}',
                                conv_block(in_channels_, out_channels, bo))

    def forward(self, x):
        for blk in self.net:
            y = blk(x)
            x = torch.cat((x, y), dim=1)
        if self.transition:
            out_channels = x.shape[1]
            x = transition_layer(x, out_channels, out_channels // 2)
        # print(x.shape)
        return x


class DenseNet(nn.Module):
    """
    由 Dense Block 组成
    :param in_channels: 3-cifar-10, 1-mnist
    :param out_channels: growth rate k
    :param dense_blk_num: 由多少 Dense Block 组成
    """
    def __init__(self,
                 in_channels=1,
                 out_channels=12,
                 conv_blk_num=4,
                 dense_blk_num=3,
                 bo=False,
                 transition=True):
        super(DenseNet, self).__init__()
        self.bn0 = nn.BatchNorm2d(in_channels)
        self.conv0 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)

        self.dense_net = nn.Sequential()
        in_channels_ = 16  # 等于初始卷积的输出通道数
        for i in range(dense_blk_num):
            # 计算通道数，有点绕，想搞清楚可以手动遍历
            in_channels_ += conv_blk_num * out_channels if i > 0 else 0
            in_channels_ = in_channels_ // 2 if i > 0 else in_channels_
            print("in_channels_:", in_channels_)

            if transition:
                self.is_transition = dense_blk_num - 1 - i  # 最后一个 dense Block 不用接过渡层

            self.dense_net.add_module(f"dense_blk_{i}",
                                      DenseBlock(in_channels_,
                                                 out_channels,
                                                 conv_blk_num=conv_blk_num,
                                                 bo=bo,
                                                 transition=self.is_transition))
        self.aap = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(1408, 10)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv0(F.relu(self.bn0(x)))  # N,16,,
        x = self.dense_net(x)
        x = self.aap(x)
        x = x.view(batch_size, -1)
        x = F.softmax(self.fc(x))

        return x


in_channels, k = 1, 12
model = DenseNet(in_channels=in_channels,
                 out_channels=k,
                 conv_blk_num=4,
                 dense_blk_num=3,
                 bo=True,
                 transition=True
                 )
# print(model)

if __name__ == '__main__':
    # 模拟数据输入网络以便得到linear层的输入通道数

    # input = torch.randn(1, 1, 28, 28)  # mnist图像大小
    # output = model(input)
    # print("final x shape:", output.shape)

    train_test = TrainTest(model, dataset="mnist")

    s = time.time()

    # train_test.train(epochs=8, optimize="Adam", save=True, save_dst="./models", save_name="densenet")

    load_dst = "./models/densenet_adam_acc_0.96.pth"
    train_test.test(load_dst=load_dst)
    train_test.draw(load_dst=load_dst)
    #
    print("time cost:", time.time() - s)
    # gpu time cost: 938.34 in 10 epochs
    # cpu time cost: 308.2 in 5 epochs
