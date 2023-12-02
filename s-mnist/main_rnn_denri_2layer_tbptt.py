import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.optim.lr_scheduler import LinearLR,StepLR,MultiStepLR
import math
#import keras
from torch.utils import data
import matplotlib.pyplot as plt
from datetime import datetime
import os

torch.manual_seed(0)




from SNN_layers.spike_dense import *
from SNN_layers.spike_neuron import *
from SNN_layers.spike_rnn import *
thr_func = ActFun_adp.apply  
is_bias=True
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
print("device:",device)

#create the network
class rnn_test(nn.Module):
    def __init__(self, input_size, stride, hidden_dims, output_size):
        super(rnn_test, self).__init__()

        self.stride = stride
        self.input_size = input_size
        self.output_size = output_size

        self.rnn_1 = spike_rnn_test_denri_wotanh_R(input_size,hidden_dims[0],
                                    tau_ninitializer = 'uniform',low_n = 0,high_n = 4,vth= 1,branch=4,dt = 1,device=device,bias=is_bias)
        self.rnn_2 = spike_rnn_test_denri_wotanh_R(hidden_dims[0],hidden_dims[1],
                                    tau_ninitializer = 'uniform',low_n = 0,high_n = 4,vth= 1,branch=4,dt = 1,device=device,bias=is_bias)
  
        self.dense_2 = spike_dense_test_origin(hidden_dims[1],10,
                                    vth= 0.5,dt = 1,device=device,bias=is_bias)

        self.criterion = nn.CrossEntropyLoss()

    def compute_input_steps(self,seq_num):
        return int(seq_num/self.stride)
    
    #detach
    def detach_network(self):
        self.rnn_1.d_input = self.rnn_1.d_input.detach()
        self.rnn_1.mem = self.rnn_1.mem.detach()
        self.rnn_1.spike = self.rnn_1.spike.detach()
        self.rnn_2.d_input = self.rnn_2.d_input.detach()
        self.rnn_2.mem = self.rnn_2.mem.detach()
        self.rnn_2.spike = self.rnn_2.spike.detach()
        self.dense_2.spike = self.dense_2.spike.detach()
        self.dense_2.mem = self.dense_2.mem.detach()
        
    def forward(self,input,labels,tbptt_steps=50,Training=True,optimizer=None):
        batch_size, seq_num, input_dim = input.shape

        self.rnn_1.set_neuron_state(batch_size)
        self.rnn_2.set_neuron_state(batch_size)
        self.dense_2.set_neuron_state(batch_size)

        input_steps  = self.compute_input_steps(seq_num)
        r1_spikes = []
        r2_spikes = []
        d1_spikes = []
        d2_spikes = []
        output = 0
        for i in range(input_steps):
            start_idx = i*self.stride
            if start_idx < (seq_num - self.input_size):
                input_x = input[:, start_idx:start_idx+self.input_size, :].reshape(-1,self.input_size)
            else:
                input_x = input[:, -self.input_size:, :].reshape(-1,self.input_size)

            mem_layer1,spike_layer1 = self.rnn_1.forward(input_x)
            mem_layer2,spike_layer2 = self.rnn_2.forward(spike_layer1)
            mem_layer3,spike_layer3 = self.dense_2.forward(spike_layer2)


            output += spike_layer3
            
            # gradient truncation(tnptt_steps)
            if Training and i % tbptt_steps==0:
                optimizer.zero_grad()
                loss = self.criterion(output,labels)
                loss.backward()
                optimizer.step()

                output = output.detach()
                self.detach_network()
                

        return output


def test(model, dataloader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.view(-1, seq_dim, input_dim).to(device)
            if is_perm:
                images = images[:,perm,:] #perm
            outputs= model(images,labels,tbptt_steps=300,Training=False,optimizer=optimizer)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            model.rnn_1.apply_mask()
            model.rnn_2.apply_mask()
            if torch.cuda.is_available():
                correct += (predicted.cpu() == labels.long().cpu()).sum()
            else:
                correct += (predicted == labels).sum()

        accuracy = 100. * correct.numpy() / total
    return accuracy
    
def train(epochs,criterion,optimizer,scheduler=None):
    acc_list = []
    best_acc = 0
    path = 'model/'  # .pth'
    name = 'perm_rnn_denri_branch4_64_256neuron_MG_bs128_final_nl0h4_twolayer_initzeros_woclip_seed0_tbptt300'
    decay_mode = 'exp'
    for epoch in range(epochs):
        model.train()
        train_acc = 0
        sum_sample = 0
        train_loss_sum = 0 
        model.train()
        model.rnn_1.apply_mask()
        model.rnn_2.apply_mask()
        for i, (images, labels) in enumerate(train_loader):
            # transfer images to squences
            images = images.view(-1, seq_dim, input_dim).to(device)
            if is_perm:
                images = images[:,perm,:] #perm
 
            labels = labels.view(-1).long().to(device)
            optimizer.zero_grad()

            predictions = model(images,labels,tbptt_steps=300,Training=True,optimizer=optimizer)
            _, predicted = torch.max(predictions.data, 1)
            train_loss = criterion(predictions,labels)
            
            # print(predictions,predicted)

            train_loss.backward()
            #nn.utils.clip_grad_norm_(model.parameters(),20)
            train_loss_sum += train_loss.item()
            optimizer.step()
            model.rnn_1.apply_mask()
            model.rnn_2.apply_mask()
            labels = labels.cpu()
            predicted = predicted.cpu().t()

            train_acc += (predicted ==labels).sum()
            sum_sample+=predicted.numel()

        if scheduler:
            scheduler.step()
        train_acc = train_acc.data.cpu().numpy()/sum_sample
        valid_acc= test(model,test_loader)
        train_loss_sum+= train_loss

        acc_list.append(train_acc)
        print('lr: ',optimizer.param_groups[0]["lr"])
        if valid_acc>best_acc and train_acc>0.900:
            best_acc = valid_acc
            torch.save(model, path+name+str(best_acc)[:7]+'-srnn-psmnist.pth')
        print('epoch: {:3d}, Train Loss: {:.4f}, Train Acc: {:.4f},Valid Acc: {:.4f}'.format(epoch,
                                                                           train_loss_sum/len(train_loader),
                                                                           train_acc,valid_acc), flush=True)

    return acc_list
batch_size = 128
task = 'psmnist'


train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('mnist_data',train=True,download=True,
                               transform=torchvision.transforms.Compose(
                                   [   
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.1307,),(0.3081,))
                                   ]
                               )
                               ),
    batch_size=batch_size,shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/',train=False,download=True,
                               transform=torchvision.transforms.Compose(
                                   [   
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.1307,),(0.3081,))
                                   ]
                               )
                               ),
    batch_size=batch_size,shuffle=False
)


input_dim = 1
input_size=1
stride = 1
hidden_dims = [64,256] #network_size
output_dim = 10
seq_dim = int(784 / input_dim)  # Number of steps to unroll

#perm 
perm = torch.randperm(seq_dim)

model =rnn_test(input_size, stride,hidden_dims, output_dim)


model.to(device)
criterion = nn.CrossEntropyLoss()
learning_rate = 1e-2  # 1e-2

#ps-mnist or s-mnist, True for ps-mnist,  False for s-mnist
is_perm = True

base_params = [                    
                    model.rnn_1.dense.weight,
                    model.rnn_1.dense.bias,
                    model.rnn_2.dense.weight,
                    model.rnn_2.dense.bias,
                    model.dense_2.dense.weight,
                    model.dense_2.dense.bias,

  

                ]



optimizer = torch.optim.Adam([
    {'params': base_params},

    {'params': model.dense_2.tau_m, 'lr': learning_rate},  
    {'params': model.rnn_1.tau_n, 'lr': learning_rate},
    {'params': model.rnn_1.tau_m, 'lr': learning_rate}, 
    {'params': model.rnn_2.tau_n, 'lr': learning_rate},
    {'params': model.rnn_2.tau_m, 'lr': learning_rate}, 
    ],
    lr=learning_rate)

scheduler = StepLR(optimizer, step_size=50, gamma=.1)

epochs =150

acc_list = train(epochs,criterion,optimizer,scheduler)

accuracy = test(model,test_loader)
print(' Accuracy: ', accuracy)
print(' Accuracy: ', accuracy)