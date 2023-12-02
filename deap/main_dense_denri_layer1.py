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

from preprocessing import dataset_prepare, dataset_prepare_for_KF
from sklearn.model_selection import KFold
torch.manual_seed(0)
batch_size = 200

#preprocess, label_type[0] is 0 for valence 1 for arousal, label_type[1] represent the number of classes in the task, 2 for two-class, 3 for three-class
train_set, test_set = dataset_prepare(label_type = [0,3])

train_loader = data.DataLoader(dataset = train_set, batch_size = batch_size, shuffle = True) 
test_loader = data.DataLoader(dataset = test_set, batch_size = batch_size, shuffle = False)

import os
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")


num_epochs = 150  # 150  # n_iters / (len(train_dataset) / batch_size)

output_dim = 3 #three-class


from SNN_layers.spike_dense import *
from SNN_layers.spike_neuron import *
from SNN_layers.spike_rnn import *
thr_func = ActFun_adp.apply  
is_bias=True
# create network
#DH-SFNN
class Dense_test(nn.Module):
    def __init__(self,):
        super(Dense_test, self).__init__()
        n = 200

        self.dense_1 = spike_dense_test_denri_wotanh_R(32,n,vth= 1,dt = 1,branch =1,tau_ninitializer = 'uniform', low_n=2,high_n=6,device=device,bias=is_bias)

        self.dense_2 = readout_integrator_test(n,output_dim,
                                    dt = 1,device=device,bias=is_bias)


    def forward(self,input):
        input.to(device)
        b,seq_length,input_dim = input.shape
        #self.dense_3.set_neuron_state(b)
        self.dense_2.set_neuron_state(b)
        self.dense_1.set_neuron_state(b)
        
        output = torch.zeros(b, output_dim).to(device)
 
        for i in range(seq_length):

            input_x = input[:,i,:].reshape(b,input_dim)
            mem_layer1,spike_layer1 = self.dense_1.forward(input_x)
            #mem_layer2,spike_layer2 = self.dense_2.forward(spike_layer1)
            mem_layer2 = self.dense_2.forward(spike_layer1)
            if i>0:
                output += mem_layer2

        output = output/seq_length
        return output 


model = Dense_test()

criterion = nn.CrossEntropyLoss()#nn.NLLLoss()

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
    name = 'dense_denri_branch1_neuron200_MG_bs200_nl2h6_final_initzeros_seed0_epoch100_renew_label3_layer1'
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
        if valid_acc>best_acc and train_acc>0.700:
            best_acc = valid_acc
            torch.save(model, path+name+str(best_acc)[:7]+'-sfnn-deap.pth')
        print('epoch: {:3d}, Train Loss: {:.4f}, Train Acc: {:.4f},Valid Acc: {:.4f}'.format(epoch,
                                                                           train_loss_sum/len(train_loader),
                                                                           train_acc,valid_acc), flush=True)
 
    return acc_list


learning_rate = 1e-2

base_params = [                    
  
                    model.dense_2.dense.weight,
                    model.dense_2.dense.bias,

                    model.dense_1.dense.weight,
                    model.dense_1.dense.bias,



  

                ]



optimizer = torch.optim.Adam([
                              {'params': base_params, 'lr': learning_rate},
   
                              {'params': model.dense_2.tau_m, 'lr': learning_rate*2},  
   
                              {'params': model.dense_1.tau_m, 'lr': learning_rate*2},  
                              {'params': model.dense_1.tau_n, 'lr': learning_rate*2}, 

   
                              ],
                        lr=learning_rate)

scheduler = StepLR(optimizer, step_size=100, gamma=.1) # 20
# epoch=0
epochs =200

acc_list = train(epochs,criterion,optimizer,scheduler)

test_acc = test()
print(test_acc)


