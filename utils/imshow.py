import math
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["savefig.bbox"] = 'tight'


def img_grid_show(img_grid):
    """
    展示 imgs_grid 图像

    :param imgs: imgs_grid, torch.Size(C, H, W)
    :return:
    """
    # print(np.unique(img_grid))
    imgs = img_grid / 2 + 0.5  # 逆归一化，像素值从[-1, 1]回到[0, 1]
    # print("float in [0., 1.]?", ((0. <= imgs) & (imgs <= 1.)).all())  # False
    # imgs = F.to_pil_image(img_grid)  # 图像从(C, H, W)转回(H, W, C), 以及 uint8 in [0, 255]
    # print("uint8 in [0, 255]?", ((0 <= imgs) & (imgs <= 255)).all())  # True
    imgs = imgs.numpy().transpose(1, 2, 0)  # 图像从(C, H, W)转回(H, W, C)的numpy矩阵
    plt.imshow(imgs)
    plt.show()


def img_show(imgs, labels=None, batch_size=1):
    """
    展示图像及其标签

    :param imgs: torch.Size(N, C, H, W)
    :param labels: e.g. tensor([3, 8, 8, 2, 7]
    :param batch_size: N
    :return:
    """
    channel = imgs.shape[1]
    row = int(math.sqrt(batch_size))  # 下取整
    col = math.ceil(batch_size / row)  # 上取整

    imgs = imgs.numpy().transpose((0, 2, 3, 1))  # (N, H, W, C)
    imgs = imgs / 2.5 + 0.5

    for i in range(batch_size):
        plt.subplot(row, col, i + 1)
        plt.tight_layout()
        plt.imshow(imgs[i] , cmap='gray' if channel==1 else None)  # interpolation='none'

        if labels is not None:
            plt.title("label:{}".format(labels[i]))

        plt.xticks([])
        plt.yticks([])
    plt.show()
