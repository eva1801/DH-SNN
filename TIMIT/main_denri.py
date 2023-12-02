# bi-directional srnn within pkg

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR,MultiStepLR
import math
import torch.nn.functional as F
from torch.utils import data

from SNN_layers.spike_dense import *#spike_dense,readout_integrator
from SNN_layers.spike_neuron import *#output_Neuron
from SNN_layers.spike_rnn import *# spike_rnn

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print('device: ',device)

def normalize(data_set,Vmax,Vmin):
    return (data_set-Vmin)/(Vmax-Vmin)

train_data = np.load('data/train_f40_t100.npy')
test_data = np.load('data/test_f40_t100.npy')
valid_data = np.load('data/valid_f40_t100.npy')


num_channels = 39
use_channels = 39
Vmax = np.max(train_data[:,:,:use_channels],axis=(0,1))
Vmin = np.min(train_data[:,:,:use_channels],axis=(0,1))

train_x = normalize(train_data[:,:,:use_channels],Vmax,Vmin)
train_y = train_data[:,:,num_channels:]

test_x = normalize(test_data[:,:,:num_channels],Vmax,Vmin)
test_y = test_data[:,:,num_channels:]

valid_x = normalize(valid_data[:,:,:num_channels],Vmax,Vmin)
valid_y = valid_data[:,:,num_channels:]

print('input dataset shap: ',train_x.shape)
print('output dataset shap: ',train_y.shape)
_,seq_length,input_dim = train_x.shape
_,_,output_dim = train_y.shape

batch_size =64


torch.manual_seed(0)
def get_DataLoader(train_x,train_y,batch_size=200):
    train_dataset = data.TensorDataset(torch.Tensor(train_x), torch.Tensor(np.argmax(train_y,axis=-1)))
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

train_loader = get_DataLoader(train_x,train_y,batch_size=batch_size)
test_loader = get_DataLoader(test_x,test_y,batch_size=batch_size)
valid_loader = get_DataLoader(valid_x,valid_y,batch_size=batch_size)
#Bi-DHSRNN
class rnn_test(nn.Module):
    def __init__(self,criterion,device,delay=0):
        super(rnn_test, self).__init__()
        self.criterion = criterion
        self.delay = delay
        is_bias = True
        #network size
        self.network = [39,256,256,output_dim]

        #DH-SRNN layer forward
        self.rnn_fw1 = spike_rnn_test_denri_wotanh_R(self.network[0],self.network[1],tau_ninitializer = 'uniform',low_n = 2,high_n = 6,vth= 1,dt = 1,branch =8,device=device,bias=is_bias)
        #DH-SRNN layer backward
        self.rnn_bw1 = spike_rnn_test_denri_wotanh_R(self.network[0],self.network[2],tau_ninitializer = 'uniform',low_n = 2,high_n = 6,vth= 1,dt = 1,branch =8,device=device,bias=is_bias)

        self.dense_mean = readout_integrator_test(self.network[2]+self.network[1],self.network[3],
                                    low_m = -4,high_m = 0,device=device)
        

    def forward(self, input,labels=None):
        b,s,c = input.shape
        self.rnn_fw1.set_neuron_state(b)
        self.rnn_bw1.set_neuron_state(b)

        self.dense_mean.set_neuron_state(b)
        
        loss = 0
        predictions = []
        fw_spikes = []
        bw_spikes = []
        mean_tensor = 0

        for l in range(s*stride):
            # forward
            input_fw=input[:,l//stride,:].float()
            # backward
            input_bw=input[:,-l//stride,:].float()

            mem_layer1, spike_layer1 = self.rnn_fw1.forward(input_fw)
            mem_layer2, spike_layer2 = self.rnn_bw1.forward(input_bw)
            fw_spikes.append(spike_layer1)
            bw_spikes.insert(0,spike_layer2)
        #merge forward and backward
        for k in range(s*stride):
            bw_idx = int(k//stride)*stride + (stride - int(k%stride))
            second_tensor = bw_spikes[k]#[bw_idx]
            merge_spikes = torch.cat((fw_spikes[k], second_tensor), -1)
            mean_tensor += merge_spikes
            if k %stride ==(stride-1):
                mem_layer3  = self.dense_mean(mean_tensor/float(stride))# mean or accumulate
            
                output = F.log_softmax(mem_layer3,dim=-1)#
                predictions.append(output.data.cpu().numpy())
                if labels is not None:
                    loss += self.criterion(output, labels[:, k//stride])
                mean_tensor = 0
    
        predictions = torch.tensor(predictions)
        return predictions, loss

def test(data_loader,after_num_frames=0):
    test_acc = 0.
    sum_samples = 0
    fr = []
    for i, (images, labels) in enumerate(data_loader):
        model.rnn_fw1.apply_mask()
        model.rnn_bw1.apply_mask()      
        images = images.view(-1, seq_length, input_dim).to(device)
        labels = labels.view((-1,seq_length)).long().to(device)
        predictions, _ = model(images)
        _, predicted = torch.max(predictions.data, 2)
        labels = labels.cpu()
        predicted = predicted.cpu().t()

        
        test_acc += (predicted == labels).sum()
        
        sum_samples = sum_samples + predicted.numel()

    return test_acc.data.cpu().numpy() / sum_samples



def train(model,loader,optimizer,scheduler=None,num_epochs=10):
    best_acc = 0
    path = 'model/rnn_denri_branch8_neuron256_bs64_woclip_1e-2_t100-40_strde10_new'  
    acc_list=[]
    for epoch in range(num_epochs):
        train_acc = 0
        train_loss_sum = 0
        sum_samples = 0
        for i, (images, labels) in enumerate(loader):
            model.rnn_fw1.apply_mask()
            model.rnn_bw1.apply_mask()
            images = images.view(-1, seq_length, input_dim).requires_grad_().to(device)
            labels = labels.view((-1,seq_length)).long().to(device)
            optimizer.zero_grad()
    
            predictions, train_loss = model(images, labels)
            _, predicted = torch.max(predictions.data, 2)
            
            train_loss.backward()
            train_loss_sum += train_loss
            optimizer.step()

            labels = labels.cpu()
            predicted = predicted.cpu().t()
            
            train_acc += (predicted == labels).sum()
            sum_samples = sum_samples + predicted.numel()
            torch.cuda.empty_cache()
        if scheduler is not None:
            scheduler.step()
            
        train_acc = train_acc.data.cpu().numpy() / sum_samples
        valid_acc = test(valid_loader)
        test_acc = test(test_loader)
        if test_acc>best_acc and train_acc>0.30:
            best_acc = test_acc
            torch.save(model, path+str(best_acc)[:7]+'-bi-srnn-timit.pth')

        acc_list.append(train_acc)
        print('epoch: {:3d}, Train Loss: {:.4f}, Train Acc: {:.4f},Valid Acc: {:.4f},Test Acc: {:.4f}'.format(epoch,
                                                                           train_loss_sum.item()/len(loader)/(seq_length),
                                                                           train_acc,valid_acc,test_acc), flush=True)
    return acc_list


num_epochs = 150
criterion = nn.NLLLoss()
model = rnn_test(criterion=criterion,device=device)
stride = 10

model.to(device)


learning_rate =1e-2
base_params = [
               model.rnn_fw1.dense.weight,model.rnn_fw1.dense.bias,
               model.rnn_bw1.dense.weight,model.rnn_bw1.dense.bias,
               model.dense_mean.dense.weight,model.dense_mean.dense.bias]

optimizer = torch.optim.Adam([
    {'params': base_params},
    {'params': model.rnn_fw1.tau_m, 'lr': learning_rate * 2},
    {'params': model.rnn_fw1.tau_n, 'lr': learning_rate * 2},
    {'params': model.rnn_bw1.tau_m, 'lr': learning_rate * 2},
    {'params': model.rnn_bw1.tau_n, 'lr': learning_rate * 2},
    {'params': model.dense_mean.tau_m, 'lr': learning_rate * 2}],
    lr=learning_rate,eps=1e-5)   


scheduler = StepLR(optimizer, step_size=60, gamma=.1) # LIF

# training network


train_acc_list = train(model,train_loader,optimizer,scheduler,num_epochs=num_epochs)
test_acc = test(test_loader)
print(test_acc)
