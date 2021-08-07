import os

import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid


data_root = r"pytorch_src/mnist-classification-in-action/datasets"

data_transforms = {
    'mnist': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]),
    'cifar10': transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),  # 居中裁剪成 224×224的小图
        transforms.ToTensor(),  # 转成 Tensor 格式
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  # 以均值0.5，标准差0.5，分别在3个通道上进行归一化
    ])
}

def mnist(batch_size=64):
    """
    img: torch.Size([1, 1, 28, 28])
    label: tensor([5])
    :param batch_size:
    :return:
      train_loader, test_loader
    """

    train_set = datasets.MNIST(root=os.path.join(data_root, "mnist"),
                                train=True,
                                transform=data_transforms["mnist"],  # 原始是 PIL Image 格式
                                download=True)
    test_set = datasets.MNIST(root=os.path.join(data_root, "mnist"),
                               train=False,
                               transform=data_transforms["mnist"],
                               download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def cifar10(batch_size=64):
    """
    img: torch.Size([1, 3, 32, 32])
    label: tensor([5])
    :param batch_size:
    :return:
      train_loader,
      test_loader,
      classes: ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    """

    train_set = datasets.CIFAR10(root=os.path.join(data_root, "cifar-10"),
                                 train=True,
                                 transform=data_transforms["cifar10"],  # 原始是 PIL Image 格式
                                 download=True)
    test_set = datasets.CIFAR10(root=os.path.join(data_root, "cifar-10"),
                                train=False,
                                transform=data_transforms["cifar10"],
                                download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    classes = np.asarray(('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck'))

    return train_loader, test_loader, classes


if __name__ == '__main__':
    from utils.imshow import img_show, img_grid_show

    batch_size = 3  # 通过batch_size来设置一次展示的图片个数
    train_loader, test_loader = mnist(batch_size=batch_size)
    imgs, labels = next(iter(train_loader))
    print(imgs.shape, labels)

    img_show(imgs, labels, batch_size)

    img_grid = make_grid(imgs)
    print(img_grid.shape)
    img_grid_show(img_grid)
