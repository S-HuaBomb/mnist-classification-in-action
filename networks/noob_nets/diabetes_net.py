import copy

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


class DiabetesDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt("./datasets/diabetes/diabetes.csv.gz", delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        """很明显 Index 是行坐标"""
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset()
data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
print(len(dataset))


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(8, 6)
        self.linear2 = nn.Linear(6, 4)
        self.linear3 = nn.Linear(4, 1)
        self.activate = nn.Sigmoid()

    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.activate(self.linear3(x))
        return x


model = Model()

criterion = nn.BCELoss()  # 二分类问题损失函数
optimizer = optim.AdamW(model.parameters(), lr=0.003)

if __name__ == '__main__':
    for epoch in range(50):
        TP = 0
        loss_lst = []
        for i, (x, y) in enumerate(data_loader):
            y_pred = model(x)
            loss = criterion(y_pred, y)

            loss_lst.append(loss.item())
            y_hat = copy.copy(y_pred.data.numpy())
            y_hat[y_hat >= 0.5] = 1.0
            y_hat[y_hat < 0.5] = 0.0
            TP += np.sum(y.data.numpy().flatten() == y_hat.flatten())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = TP / len(dataset)
        print("epoch:", epoch, "loss:", np.mean(loss_lst), "acc:", acc, "TP:", TP)
