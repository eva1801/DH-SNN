import os
import numpy as np
import torch
import tables
from torch.utils.data import DataLoader, TensorDataset

from scipy.sparse import coo_matrix


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR,MultiStepLR
import math
#import keras
from torch.utils import data
import matplotlib.pyplot as plt
from datetime import datetime
import os

torch.manual_seed(42)

time_steps = 100 # total timesteps
channel = 2 #signal1 and signal 2
channel_rate = [0.2,0.6] #spiking rates of high or low
noise_rate = 0.01
channel_size = 20
#lasting time of signal 1 and signal 2
coding_time =10
remain_time =5
start_time = 10

#init xor label
label = torch.zeros(len(channel_rate),len(channel_rate))
label[1][0] = 1
label[0][1] = 1

def get_batch():
    """Generate the mutlitimescale spiking xor problem dataset"""
    # Build the first sequence
    values = torch.rand(batch_size,time_steps,channel_size*2,requires_grad=False) <= noise_rate
    targets = torch.zeros(time_steps,batch_size,requires_grad=False).int()
    #build the signal 1
    init_pattern = torch.randint(len(channel_rate),size=(batch_size,))
    #generate spikes
    prob_matrix = torch.ones(start_time,channel_size,batch_size)*torch.tensor(channel_rate)[init_pattern]
    add_patterns = torch.bernoulli(prob_matrix).permute(2,0,1).bool()
    values[:,:start_time,:channel_size] = values[:,:start_time,:channel_size] | add_patterns
    
    #build the signal 2
    for i in range((time_steps-start_time) //(coding_time+remain_time)):
        pattern = torch.randint(len(channel_rate),size=(batch_size,))
        label_t = label[init_pattern,pattern].int()
        #generate spikes
        prob = torch.tensor(channel_rate)[pattern]
        prob_matrix = torch.ones(coding_time,channel_size,batch_size)*prob
        add_patterns = torch.bernoulli(prob_matrix).permute(2,0,1).bool()

        values[:,start_time+i*(coding_time+remain_time)+remain_time:start_time+(i+1)*(coding_time+remain_time),channel_size:] = values[:,start_time+i*(coding_time+remain_time)+remain_time:start_time+(i+1)*(coding_time+remain_time),channel_size:] | add_patterns
        targets[start_time+i*(coding_time+remain_time):start_time+(i+1)*(coding_time+remain_time)] = label_t
    return values, targets.transpose(0,1).contiguous()

from SNN_layers.spike_dense import *
from SNN_layers.spike_neuron import *
from SNN_layers.spike_rnn import *
thr_func = ActFun_adp.apply  

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
print("device:",device)

#create networks
class Dense_test_1layer(nn.Module):  # DH-SFNNs with 1 layer
    def __init__(self, input_size, hidden_dims, output_dim):
        super(Dense_test_1layer, self).__init__()

        self.input_size = input_size
        self.dense_1 = spike_dense_test_denri_wotanh_R_xor(input_size,hidden_dims,
                                    tau_ninitializer = 'uniform',low_n = 2,high_n = 6,vth= 1,dt = 1,branch = 2,device=device,bias =False)
        self.dense_2 = nn.Linear(hidden_dims,output_dim)
        self.criterion = nn.CrossEntropyLoss()

    def init(self):
        self.dense_1.set_neuron_state(batch_size)
    def forward(self,input,target):
        batch_size,seq_num,input_size= input.shape
        
        d2_output = torch.zeros(batch_size,seq_num,2)
        output = 0
        loss = 0
        total = 0
        correct = 0
        for i in range(seq_num):

            input_x = input[:,i,:]
            mem_layer1,spike_layer1 = self.dense_1.forward(input_x)
            l2_output= self.dense_2(spike_layer1)
            d2_output[:,i,:] = l2_output.cpu()
            if (((i-start_time) % (coding_time+remain_time))> remain_time)and(i>start_time):
                output = F.softmax(l2_output,dim=1)
                loss += self.criterion(output,target[:,i].long())
                _, predicted = torch.max(output.data, 1)
                labels = target[:,i].cpu()
                predicted = predicted.cpu().t()
                correct += (predicted ==labels).sum()
                total += labels.size()[0]

        return loss,d2_output,correct,total
# DH-SFNNs with 2 layer    
class Dense_test_2layer(nn.Module):
    def __init__(self, input_size, hidden_dims, output_dim):
        super(Dense_test_2layer, self).__init__()

        self.input_size = input_size

        self.dense_1 = spike_dense_test_denri_wotanh_R_xor(input_size,hidden_dims,
                                    tau_ninitializer = 'uniform',low_n = 2,high_n = 6,vth= 1,dt = 1,branch = 1,device=device,bias = False)
        self.dense_2 = spike_dense_test_denri_wotanh_R_xor(hidden_dims,hidden_dims,
                                    tau_ninitializer = 'uniform',low_n = 2,high_n = 6,vth= 1,dt = 1,branch = 1,device=device,bias = False)
        self.dense_3 = nn.Linear(hidden_dims,output_dim)

        self.criterion = nn.CrossEntropyLoss()

    def init(self):
        self.dense_1.set_neuron_state(batch_size)
        self.dense_2.set_neuron_state(batch_size)
    def forward(self,input,target):
        batch_size,seq_num,input_size= input.shape
        
        d2_output = torch.zeros(batch_size,seq_num,2)

        output = 0
        loss = 0
        total = 0
        correct = 0
        for i in range(seq_num):

            input_x = input[:,i,:]
            mem_layer1,spike_layer1 = self.dense_1.forward(input_x)
            mem_layer2,spike_layer2 = self.dense_2.forward(spike_layer1)
            l2_output= self.dense_3(spike_layer2)
            d2_output[:,i,:] = l2_output.cpu()
            if (((i-start_time) % (coding_time+remain_time))> remain_time)and(i>start_time):
                output = F.softmax(l2_output,dim=1)
                loss += self.criterion(output,target[:,i].long())
                _, predicted = torch.max(output.data, 1)
                labels = target[:,i].cpu()
                predicted = predicted.cpu().t()

                correct += (predicted ==labels).sum()
                total += labels.size()[0]

        return loss,d2_output,correct,total
    
# DH-SRNNs with 1 layer
class RNN_test_1layer(nn.Module):
    def __init__(self, input_size, hidden_dims, output_dim):
        super(RNN_test_1layer, self).__init__()



        self.input_size = input_size


        self.rnn_1 = spike_rnn_test_denri_wotanh_R(input_size,hidden_dims,branch = 1,
                                   vth= 1,dt = 1,device=device,bias=False)
        self.dense_2 = nn.Linear(hidden_dims,output_dim)

        self.criterion = nn.CrossEntropyLoss()

    def init(self):
        self.rnn_1.set_neuron_state(batch_size)
    def forward(self,input,target):
        batch_size,seq_num,input_size= input.shape
        
        #input = input/255.
        d2_output = torch.zeros(batch_size,seq_num,2)
        output = 0
        loss = 0
        total = 0
        correct = 0
        for i in range(seq_num):

            input_x = input[:,i,:]
            mem_layer1,spike_layer1 = self.rnn_1.forward(input_x)
            l2_output= self.dense_2(spike_layer1)
            d2_output[:,i,:] = l2_output.cpu()
            if (((i-start_time) % (coding_time+remain_time))> remain_time)and(i>start_time):
                output = F.softmax(l2_output,dim=1)
                loss += self.criterion(output,target[:,i].long())
                _, predicted = torch.max(output.data, 1)
                labels = target[:,i].cpu()
                predicted = predicted.cpu().t()
                correct += (predicted ==labels).sum()
                total += labels.size()[0]

        return loss,d2_output,correct,total

def train(epochs,optimizer,scheduler=None):
    acc_list = []
    best_loss = 150
    log_internel = 100
    path = 'model/'  # .pth'
    name = 'dense_denri_2branch_16neruon_channel2_size20_remain5_coding10_start10_time100_mem_nl2h6_'
    for epoch in range(epochs):
        train_acc = 0
        sum_sample = 0
        train_loss_sum = 0 
        sum_correct = 0
        model.train()
        model.dense_1.apply_mask()
        
        for _ in range(log_internel):
            # if i ==0: 
            model.init()
            model.dense_1.apply_mask()

            data, target = get_batch()
            data = data.detach().to(device)
 
            target= target.detach().to(device)
            optimizer.zero_grad()

            loss,output,correct,total = model(data,target)

        
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),20)
            train_loss_sum += loss.item()
            optimizer.step()
            sum_correct += correct.item()
            sum_sample+= total

        if scheduler:
            scheduler.step()
        acc_list.append(train_acc)
        print('lr: ',optimizer.param_groups[0]["lr"])
    
        if train_loss_sum<best_loss*log_internel :
            best_loss = train_loss_sum
            torch.save(model, path+name+str(best_loss/log_internel)[:7]+'-sfnn_multitime.pth')

        print('log_internel:{:3d}, epoch: {:3d}, Train Loss: {:.4f}, Acc: {:.3f}'.format(log_internel,epoch,train_loss_sum/log_internel,sum_correct/sum_sample), flush=True)
  
    return acc_list
batch_size = 500

#network size
hidden_dims = 16
output_dim = 2


model = Dense_test_1layer(channel_size*2,hidden_dims,output_dim)


model.to(device)

learning_rate = 1e-2  
is_perm = True

base_params = [                    

                    model.dense_2.weight,
                    model.dense_2.bias,

                    model.dense_1.dense.weight,
  

                ]


optimizer = torch.optim.Adam([
    {'params': base_params},

    {'params': model.dense_1.tau_m, 'lr': learning_rate},  
    {'params': model.dense_1.tau_n, 'lr': learning_rate}, 


    ],
    lr=learning_rate)

scheduler = StepLR(optimizer, step_size=50, gamma=.1)


epochs =150

acc_list = train(epochs,optimizer,scheduler)


