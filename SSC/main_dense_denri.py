import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR,MultiStepLR
import math
import torch.nn.functional as F
from torch.utils import data
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from ssc_dataset_f import my_Dataset
import os
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
batch_size = 100

#dataset and dataloader
train_dir = '/data1/SSC/train_1ms/'
train_files = [train_dir+i for i in os.listdir(train_dir)]
valid_dir = '/data1/SSC/valid_1ms/'
valid_files = [valid_dir+i for i in os.listdir(valid_dir)]
test_dir = '/data1/SSC/test_1ms/'
test_files = [test_dir+i for i in os.listdir(test_dir)]
train_dataset = my_Dataset(train_files)
valid_dataset = my_Dataset(valid_files)
test_dataset = my_Dataset(test_files)

train_loader = data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)#,num_workers=10)
valid_loader = data.DataLoader(valid_dataset,batch_size=batch_size,shuffle=True)#,num_workers=5)
test_loader = data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True)#,num_workers=5)

num_epochs = 150  

from SNN_layers.spike_dense import *
from SNN_layers.spike_neuron import *
from SNN_layers.spike_rnn import *
thr_func = ActFun_adp.apply  
is_bias=True
#DH-SFNN with 1 layer
class Dense_test_1layer(nn.Module):
    def __init__(self,):
        super(Dense_test_1layer, self).__init__()
        n = 200 #network size
        self.dense_1 = spike_dense_test_denri_wotanh_R(700,n,vth= 1,dt = 1,branch =4,tau_ninitializer = 'uniform', low_n=2,high_n=6,device=device,bias=is_bias,test_sparsity=False)
        self.dense_2 = readout_integrator_test(n,35,
                                    dt = 1,device=device,bias=is_bias)


        torch.nn.init.xavier_normal_(self.dense_2.dense.weight)
      
        if is_bias:
            torch.nn.init.constant_(self.dense_2.dense.bias,0)

    def forward(self,input):
        input.to(device)
        b,seq_length,input_dim = input.shape
        self.dense_2.set_neuron_state(b)
        self.dense_1.set_neuron_state(b)
        
        output = torch.zeros(b, 35).to(device)
 
        for i in range(seq_length):

            input_x = input[:,i,:].reshape(b,input_dim)
            mem_layer1,spike_layer1 = self.dense_1.forward(input_x)
            mem_layer2 = self.dense_2.forward(spike_layer1)
            if i>0:
                output += mem_layer2

        output = F.log_softmax(output/seq_length, dim=1)
        return output 
    
#DH-SFNN with 2 layer
class Dense_test_2layer(nn.Module):
    def __init__(self,):
        super(Dense_test_2layer, self).__init__()
        n = 200
        # is_bias=False
        #self.dense_1 = spike_dense(700,n,
        #                            tauM = 20,tauM_inital_std=5,device=device,bias=is_bias)
        self.dense_1 = spike_dense_test_denri_wotanh_R(700,n,vth= 1,dt = 1,branch =4,tau_ninitializer = 'uniform', low_n=2,high_n=6,device=device,bias=is_bias)
        self.dense_2 = spike_dense_test_denri_wotanh_R(n,n,vth= 1,dt = 1,branch =4,tau_ninitializer = 'uniform', low_n=2,high_n=6,device=device,bias=is_bias)
        self.dense_3 = readout_integrator_test(n,35,
                                    dt = 1,device=device,bias=is_bias)


        torch.nn.init.xavier_normal_(self.dense_3.dense.weight)
      
        if is_bias:
            torch.nn.init.constant_(self.dense_3.dense.bias,0)

    def forward(self,input):
        input.to(device)
        b,seq_length,input_dim = input.shape
        self.dense_3.set_neuron_state(b)
        self.dense_2.set_neuron_state(b)
        self.dense_1.set_neuron_state(b)
        
        output = torch.zeros(b, 35).to(device)
 
        for i in range(seq_length):

            input_x = input[:,i,:].reshape(b,input_dim)
            mem_layer1,spike_layer1 = self.dense_1.forward(input_x)
            mem_layer2,spike_layer2 = self.dense_2.forward(spike_layer1)
            mem_layer3 = self.dense_3.forward(spike_layer2)
            if i>0:
                output += mem_layer3

        output = F.log_softmax(output/seq_length, dim=1)
        return output 

model = Dense_test_1layer()

criterion = nn.CrossEntropyLoss()

print("device:",device)
model.to(device)

def test():
    test_acc = 0.
    sum_sample = 0.

    model.eval()
    with torch.no_grad():
        model.dense_1.apply_mask()
        for i, (images, labels) in enumerate(test_loader):
            #images = images.view(-1,250, 700).to(device)
            images = images.to(device)
            labels = labels.view((-1)).long().to(device)
            predictions = model(images)

            _, predicted = torch.max(predictions.data, 1)
            labels = labels.cpu()
            predicted = predicted.cpu().t()
            model.dense_1.apply_mask()
            test_acc += (predicted == labels).sum()
            sum_sample+=predicted.numel()

    return test_acc.data.cpu().numpy()/sum_sample


def train(epochs,criterion,optimizer,scheduler=None):
    acc_list = []
    best_acc = 0
  
    path = 'model/'  # .pth'
    name = 'dense_branch4_neuron200_1ms_MG_bs100_nl2h6_woclipnorm_final_initzeros_seed0'
    for epoch in range(epochs):
        train_acc = 0
        sum_sample = 0
        train_loss_sum = 0 
        model.train()
        model.dense_1.apply_mask()
        for i, (images, labels) in enumerate(train_loader):
            # if i ==0: 
           
            images = images.to(device)
 
            labels = labels.view((-1)).long().to(device)
            optimizer.zero_grad()

            predictions = model(images)
            _, predicted = torch.max(predictions.data, 1)
            train_loss = criterion(predictions,labels)
            
            # print(predictions,predicted)

            train_loss.backward()
            #nn.utils.clip_grad_norm_(model.parameters(),20)
            train_loss_sum += train_loss.item()
            optimizer.step()
            labels = labels.cpu()
            predicted = predicted.cpu().t()
            model.dense_1.apply_mask()
            train_acc += (predicted ==labels).sum()
            sum_sample+=predicted.numel()

        if scheduler:
            scheduler.step()
        train_acc = train_acc.data.cpu().numpy()/sum_sample
        valid_acc= test()
        train_loss_sum+= train_loss

        acc_list.append(train_acc)
        print('lr: ',optimizer.param_groups[0]["lr"])
        if valid_acc>best_acc and train_acc>0.500:
            best_acc = valid_acc
            torch.save(model, path+name+str(best_acc)[:7]+'-sfnn-ssc.pth')
        print('epoch: {:3d}, Train Loss: {:.4f}, Train Acc: {:.4f},Valid Acc: {:.4f}'.format(epoch,
                                                                           train_loss_sum/len(train_loader),
                                                                           train_acc,valid_acc), flush=True)

    return acc_list



# create network

learning_rate = 1e-2#1.2e-2


base_params = [                    
 
                    model.dense_2.dense.weight,
                    model.dense_2.dense.bias,


                    model.dense_1.dense.weight,
                    model.dense_1.dense.bias,


                ]


optimizer = torch.optim.Adam([
                              {'params': base_params, 'lr': learning_rate},
                              {'params': model.dense_2.tau_m, 'lr': learning_rate },  
                              {'params': model.dense_1.tau_m, 'lr': learning_rate},  
                              {'params': model.dense_1.tau_n, 'lr': learning_rate}, 

                              ],
                        lr=learning_rate)

scheduler = StepLR(optimizer, step_size=25, gamma=.1) 
# epoch=0
epochs =100

acc_list = train(epochs,criterion,optimizer,scheduler)

test_acc = test()
print(test_acc)


