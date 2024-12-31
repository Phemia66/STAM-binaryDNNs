import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import scipy.io as io
import time
import NNstructure
from torch import nn
from torch.nn import init

import torch.nn.functional as F
from collections import OrderedDict

train = torchvision.datasets.MNIST(root='~/Datasets',
                                   train=True, download=True, transform=transforms.ToTensor())
test = torchvision.datasets.MNIST(root='~/Datasets',
                                  train=False, download=True, transform=transforms.ToTensor())
# train, val = torch.utils.data.random_split(train, [55000, 5000])

batch_size = 128
train_iter = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
test_iter = torch.utils.data.DataLoader(test, batch_size=100, shuffle=False, num_workers=0)

num_epochs = 200
para_weightdecay = 0
lr = 0.001

tr_loss = np.zeros(num_epochs)
ts_loss = np.zeros(num_epochs)
vl_loss = np.zeros(num_epochs)
tr_acc = np.zeros(num_epochs)
ts_acc = np.zeros(num_epochs)
vl_acc = np.zeros(num_epochs)
bp_time = np.zeros(num_epochs)


def Quant_Proj(x):
    s = x.abs().sum() / x.numel()
    q = torch.ones(x.size()).cuda()
    q[x < 0] = -1
    return s * q


def Quant_Proj_OriginBinary(x):
    q = torch.ones(x.size()).cuda()
    q[x < 0] = -1
    return q


def evaluate_accuracy_loss(data_iter, net):
    acc_sum, loss_sum, n = 0.0, 0.0, 0
    for X, y in data_iter:
        with torch.no_grad():
            X = X.cuda()
            y = y.cuda()
            y_hat = net(X)
            acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
            loss_sum += loss(y_hat, y)
            n += y.shape[0]
        torch.cuda.empty_cache()
    return acc_sum / n, loss_sum / n


input_size = [1, 28, 28]
net = NNstructure.MLP_mnist().cuda()
params = list(net.parameters())
g_list = []
lambda_list = []
w_list = []

for m in net.modules():
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight, std=1e-3)  # 1e-3
        if m.bias is not None:
            # init.normal_(m.bias, std=1e-3)
            init.constant_(m.bias, 0)

for param in params:
    g_param = param.clone().detach()
    w_list.append(g_param)

list_len = len(params)
loss = torch.nn.CrossEntropyLoss(reduction="sum")
# loss = torch.nn.MSELoss()
rho = 1   # relax parameter

for epoch in range(num_epochs):
    cal_time = 0
    train_loss, train_acc, n = 0.0, 0.0, 0,
    for X, y in train_iter:
        X = X.cuda()
        y = y.cuda()
        start = time.time()
        for i in range(list_len):
            params[i].data = Quant_Proj(w_list[i])
            # params[i].data = (Quant_Proj(w_list[i]) * rho + w_list[i])/(1 + rho)
        y_hat = net(X)
        l = loss(y_hat, y)
        l.backward()
        for i in range(list_len):
            w_list[i].data = (1 - para_weightdecay) * w_list[i].data - params[i].grad * lr
            #print(params[i].grad)
            params[i].grad.zero_()
        end = time.time()
        cal_time = cal_time + end - start
        train_loss += l.item()
        train_acc += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.shape[0]
        # print(train_acc / n, train_loss / n)
    # rho = min(rho * 1.05, 200)
    for i in range(list_len):
        params[i].data = Quant_Proj(w_list[i])
        print(params[i].data)
    bp_time[epoch] = cal_time
    tr_acc[epoch], tr_loss[epoch] = evaluate_accuracy_loss(train_iter, net)
    ts_acc[epoch], ts_loss[epoch] = evaluate_accuracy_loss(test_iter, net)
    print(tr_acc[epoch], tr_loss[epoch], ts_acc[epoch], ts_loss[epoch], max(ts_acc))
    # print(tr_loss[epoch], epoch + 1)
dataSGD = 'E://PythonProject//BinaryProx_v1//br_1.mat'
io.savemat(dataSGD, {'tr_acc': tr_acc, 'tr_loss': tr_loss, 'ts_acc': ts_acc, 'ts_loss': ts_loss, 'vl_acc': vl_acc,
                     'vl_loss': vl_loss, 'epochs': num_epochs,
                     'para_weightdecay': para_weightdecay,
                     'batch': batch_size, 'bp_time': bp_time})
