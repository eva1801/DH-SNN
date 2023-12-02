import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR,MultiStepLR
import matplotlib.pyplot as plt
import math
from sklearn.utils import shuffle
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
from sklearn.metrics import confusion_matrix
import scipy.io
import random
import os
from shd_dataset import my_Dataset
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# device =  torch.device("cpu")
print('device: ',device)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(0)

dt = 1  #

batch_size = 100
train_dir = '/data1/SHD/train_1ms/'
train_files = [train_dir+i for i in os.listdir(train_dir)]

test_dir = '/data1/SHD/test_1ms/'
test_files = [test_dir+i for i in os.listdir(test_dir)]
train_dataset = my_Dataset(train_files)
test_dataset = my_Dataset(test_files)

train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)#,num_workers=10)

test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True)#,num_workers=5)



from SNN_layers.spike_dense import *
from SNN_layers.spike_neuron import *
from SNN_layers.spike_rnn import *
thr_func = ActFun_adp.apply  
is_bias=True

class rnn_test(nn.Module):
    def __init__(self,):
        super(rnn_test, self).__init__()
        n = 64
        # vanilla SRNN layer
        self.rnn_1 = spike_rnn_test_origin(700,n,vth= 1,dt = 1,device=device)

  
        self.dense_2 = readout_integrator_test(n,20,dt = 1,device=device)

        torch.nn.init.xavier_normal_(self.dense_2.dense.weight)
 
        if is_bias:

            torch.nn.init.constant_(self.dense_2.dense.bias,0)

    def forward(self,input):
        input.to(device)
        b,seq_length,input_dim = input.shape
        self.rnn_1.set_neuron_state(b)
        self.dense_2.set_neuron_state(b)
        output = 0
        for i in range(seq_length):

            input_x = input[:,i,:].reshape(b,input_dim)
            mem_layer1,spike_layer1 = self.rnn_1.forward(input_x)

            mem_layer2 = self.dense_2.forward(spike_layer1)

            if i>10:
                output += F.softmax(mem_layer2,dim=1)

        return output

model = rnn_test()

criterion = nn.CrossEntropyLoss()

print("device:",device)
model.to(device)

def test():
    test_acc = 0.
    sum_sample = 0.

    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):

            images = images.to(device)
        
            labels = labels.view((-1)).long().to(device)
            predictions = model(images)

            _, predicted = torch.max(predictions.data, 1)
            labels = labels.cpu()
            predicted = predicted.cpu().t()
            
            test_acc += (predicted == labels).sum()
            sum_sample+=predicted.numel()

    return test_acc.data.cpu().numpy()/sum_sample


def train(epochs,criterion,optimizer,scheduler=None):
    acc_list = []
    best_acc = 0
    prun_step_size = 20
    decay_rate = 0.5
    path = 'model/'  # .pth'
    name = 'rnn_origin_1ms_final_64neuron_MG_bs100_initzeros_seed0'
    for epoch in range(epochs):
        train_acc = 0
        sum_sample = 0
        train_loss_sum = 0 
        model.train()

        for i, (images, labels) in enumerate(train_loader):
            # if i ==0: 

            images = images.to(device)
 
            labels = labels.view((-1)).long().to(device)
            optimizer.zero_grad()

            predictions = model(images)
            _, predicted = torch.max(predictions.data, 1)
            train_loss = criterion(predictions,labels)
            

            train_loss.backward()

            #nn.utils.clip_grad_norm_(model.parameters(),20)
            train_loss_sum += train_loss.item()
            optimizer.step()


            labels = labels.cpu()
            predicted = predicted.cpu().t()

            train_acc += (predicted ==labels).sum()
            sum_sample+=predicted.numel()

        if scheduler:
            scheduler.step()
        train_acc = train_acc.data.cpu().numpy()/sum_sample
        valid_acc= test()
        train_loss_sum+= train_loss

        acc_list.append(train_acc)
        print('lr: ',optimizer.param_groups[0]["lr"])
        if valid_acc>best_acc:
            best_acc = valid_acc
            torch.save(model, path+name+str(best_acc)[:7]+'-srnn-shd.pth')
        print('epoch: {:3d}, Train Loss: {:.4f}, Train Acc: {:.4f},Valid Acc: {:.4f}'.format(epoch,
                                                                           train_loss_sum/len(train_loader),
                                                                           train_acc,valid_acc), flush=True)
  
    return acc_list





learning_rate = 1e-2 


base_params = [                    

                    model.dense_2.dense.weight,
                    model.dense_2.dense.bias,


                    model.rnn_1.dense.weight,
                    model.rnn_1.dense.bias,

  

                ]


optimizer = torch.optim.Adam([
                              {'params': base_params, 'lr': learning_rate},
                              {'params': model.dense_2.tau_m, 'lr': learning_rate*2},  
                              {'params': model.rnn_1.tau_m, 'lr': learning_rate*2},  

                              ],
                        lr=learning_rate)

scheduler = StepLR(optimizer, step_size=20, gamma=.5) # 20
epochs =100
acc_list = train(epochs,criterion,optimizer,scheduler)

test_acc = test()
print(test_acc)


