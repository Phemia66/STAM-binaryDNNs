import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import scipy.io as io
import time
import NNstructure
import copy
from torch import nn
from torch.utils.data import Dataset

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train = torchvision.datasets.CIFAR10(root='/home/bianfm/Desktop/CIFAR10/Datasets',
                                     train=True, download=True, transform=transform_train)
test = torchvision.datasets.CIFAR10(root='/home/bianfm/Desktop/CIFAR10/Datasets',
                                    train=False, download=True, transform=transform_test)
loss=torch.nn.CrossEntropyLoss(reduction="sum")
batchsize=128
train_iter = torch.utils.data.DataLoader(train, batch_size=batchsize, shuffle=True)
test_iter = torch.utils.data.DataLoader(test, batch_size=2000, shuffle=False)

######## gen net ##########
torch.cuda.set_device(2)
Net = NNstructure.VGG('VGG11').cuda()
params = list(Net.parameters())
list_len = len(params)

def Quant_Proj(x):
    s = x.abs().sum() / x.numel()
    q = torch.ones(x.size(), device='cuda')
    q[x < 0] = -1
    return s * q

def reg_loss(beta,w,u):
    regular_loss=0
    for i in range(list_len):
        regular_loss += torch.sum(torch.square(params[i]-w[i]-u[i]/beta).cuda()).cuda()
    return regular_loss

###### start algorithm #######
def evaluate_accuracy_loss(data_iter, Net):
    acc_sum, loss_sum, n = 0.0, 0.0, 0
    for X, y in data_iter:
        with torch.no_grad():
            X = X.cuda()
            y = y.cuda()
            y_hat = Net(X)
            loss_sum += loss(y_hat, y)
            acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
        torch.cuda.empty_cache()
    return acc_sum/n, loss_sum/n

def disable_bn(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

def enable_bn(model):
    model.train()

nb=50000
num_epochs=400
wdecay=0
alpha= 1e3
lamb=20
gamma=18

####################################
tr_loss=np.zeros(num_epochs+1)
ts_loss=np.zeros(num_epochs+1)
tr_acc=np.zeros(num_epochs+1)
ts_acc=np.zeros(num_epochs+1)
bp_time =np.zeros(num_epochs+1)
tr_acc[0], tr_loss[0] = evaluate_accuracy_loss(train_iter,Net)
ts_acc[0], ts_loss[0] = evaluate_accuracy_loss(test_iter,Net)
bp_time[0] = 0
print( 'epoch',0,'trloss:', tr_loss[0], 'tsloss', ts_loss[0],'tracc:',tr_acc[0], 'tsacc',ts_acc[0])
######################################
if (nb%batchsize)==0:
    batch_num = round(nb/batchsize)
elif (nb%batchsize)!=0:
    batch_num = round(nb/batchsize) + 1
######################################
iter_num=0
wtilde_list=[]
w_list=[]
u_list=[]
v_list=[]

for param in params:
    param.data = Quant_Proj(param.data)
    wtilde_param = param.clone().detach()
    #s_param.data = s_param.sign()
    wtilde_list.append(wtilde_param)
    w_list.append(wtilde_param)
    u_list.append(wtilde_param)
    v_list.append(wtilde_param)

cal_time = 0
for epoch in range(num_epochs):
    
    if(epoch>5):
        gamma=max(0.995*gamma,0.01)
        lamb=max(0.95*lamb,4)


    if (epoch>5):
        alpha = max(0.9999*alpha,5e2)


    for X, y in train_iter:
        X = X.cuda()
        y = y.cuda()
        start = time.time()
        y_hat = Net(X)
        l = loss(y_hat, y)             #+0.5*beta*reg_loss(beta,w_list,u_list)
        l.backward()

        for i in range(list_len):
            w_list[i] = ((alpha-lamb) * w_list[i] + lamb * wtilde_list[i] - params[i].grad) / alpha
            params[i].grad.zero_()
            wtilde_list[i]=(gamma*lamb*w_list[i]+u_list[i])/(gamma*lamb+1)
            v_list[i] = Quant_Proj(2*wtilde_list[i]-u_list[i])
            u_list[i] = u_list[i]+(v_list[i]-wtilde_list[i])
        end = time.time()
        for i in range(list_len):
            params[i].data = copy.deepcopy(w_list[i])
        cal_time = cal_time  + end - start



    for i in range(list_len):
        params[i].data = copy.deepcopy(v_list[i].data)


    bp_time[epoch+1] = cal_time
    tr_acc[epoch+1], tr_loss[epoch+1] = evaluate_accuracy_loss(train_iter,Net)
    ts_acc[epoch+1], ts_loss[epoch+1] = evaluate_accuracy_loss(test_iter,Net)

    print( 'v', 'epoch',epoch+1,'trloss:', tr_loss[epoch+1], 'tsloss', ts_loss[epoch+1],'tracc:',tr_acc[epoch+1],
           'tsacc',ts_acc[epoch+1], 'max', max(ts_acc),'time', bp_time[epoch+1])

    for i in range(list_len):
        params[i].data = copy.deepcopy(w_list[i].data)

results = '/home/bianfm/Desktop/CIFAR10/VGG11/cifar10.mat'
io.savemat(results, {'SGD_trloss': tr_loss, 'SGD_tsloss': ts_loss, 'SGD_time': bp_time,
           'SGD_tsacc':ts_acc,'SGD_tracc':tr_acc})
