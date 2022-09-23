import numpy as np
from torch import nn


class Model(nn.Module):
    """
    :params: 16402
    """
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


def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


if __name__ == '__main__':
    # model = Model()
    # param1 = next(model.parameters())
    # print(param1.size())

    from networks.google_net import model as google
    from networks.resnet_v2 import model as resnetv2
    from networks.resnet_bo import model as resnetbo
    from networks.dense_net import model as densenet

    v2params = count_parameters(resnetv2)
    boparams = count_parameters(resnetbo)
    goparams = count_parameters(google)
    deparams = count_parameters(densenet)
    print("resnetv2 params:", v2params)  # 41610
    print("resnetbo params:", boparams)  # 6905508
    print("goparams params:", goparams)  # 96274
    print("deparams params:", deparams)  # 61132
