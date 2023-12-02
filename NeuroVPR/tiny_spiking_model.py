import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
#
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
thresh = 0.1
lens = 0.5
probs = 0.5
decay = 0.6




from SNN_layers.spike_dense import *
from SNN_layers.spike_neuron import *
from SNN_layers.spike_rnn import *

is_bias=True
#DH-SNN
class Dense_test_4layer(nn.Module):
    def __init__(self,branch=4):
        super(Dense_test_4layer, self).__init__()

        self.dense_1 = spike_dense_test_denri_wotanh_R(32*43*2,512,tau_ninitializer = 'uniform',low_n = 0,high_n=4,vth= 1,dt = 1,branch = branch,device=device)
        self.dense_2 = spike_dense_test_denri_wotanh_R(512,512,tau_ninitializer = 'uniform',low_n = 0,high_n=4,vth= 1,dt = 1,branch = branch,device=device)
        self.dense_3 = spike_dense_test_denri_wotanh_R(512,256,tau_ninitializer = 'uniform',low_n = 0,high_n=4,vth= 1,dt = 1,branch = branch,device=device)
        self.dense_4 = nn.Linear(256,100)
        #self.dense_4 = readout_integrator_test(n,20,dt = 1,device=device)


    def forward(self,dvs_inp, out_mode = 'rate'):
        dvs_inp.to(device)
        batch_size, seq_len, channel, w, h = dvs_inp.size()
        dvs_inp = dvs_inp.permute([1,0,2,3,4])
        self.dense_1.set_neuron_state(batch_size)
        self.dense_2.set_neuron_state(batch_size)
        self.dense_3.set_neuron_state(batch_size)
        output = 0
        for i in range(seq_len):

            spike_inp = dvs_inp[i].reshape(batch_size,-1)
            mem_layer1,spike_layer1 = self.dense_1.forward(spike_inp)
            mem_layer2,spike_layer2 = self.dense_2.forward(spike_layer1)
            mem_layer3,spike_layer3 = self.dense_3.forward(spike_layer2)
            x = self.dense_4(spike_layer3)

            output += x

        return output / seq_len
#vanilla SNN
class Dense_test_origin_4layer(nn.Module):
    def __init__(self,):
        super(Dense_test_origin_4layer, self).__init__()

        self.dense_1 = spike_dense_test_origin(32*43*2,512,vth= 1,dt = 1,device=device)
        self.dense_2 = spike_dense_test_origin(512,512,vth= 1,dt = 1,device=device)
        self.dense_3 = spike_dense_test_origin(512,256,vth= 1,dt = 1,device=device)
        self.dense_4 = nn.Linear(256,100)
        #self.dense_4 = readout_integrator_test(n,20,dt = 1,device=device)


    def forward(self,dvs_inp, out_mode = 'rate'):
        dvs_inp.to(device)
        batch_size, seq_len, channel, w, h = dvs_inp.size()
        dvs_inp = dvs_inp.permute([1,0,2,3,4])
        self.dense_1.set_neuron_state(batch_size)
        self.dense_2.set_neuron_state(batch_size)
        self.dense_3.set_neuron_state(batch_size)
        output = 0
        for i in range(seq_len):

            spike_inp = dvs_inp[i].reshape(batch_size,-1)
            mem_layer1,spike_layer1 = self.dense_1.forward(spike_inp)
            mem_layer2,spike_layer2 = self.dense_2.forward(spike_layer1)
            mem_layer3,spike_layer3 = self.dense_3.forward(spike_layer2)
            x = self.dense_4(spike_layer3)

            output += x

        return output / seq_len

#DH-SRNN
class rnn_test_3layer(nn.Module):
    def __init__(self,branch=4):
        super(rnn_test_3layer, self).__init__()

        self.dense_1 = spike_dense_test_origin(32*43*2,512,vth= 1,dt = 1,device=device)
        self.dense_2 = spike_rnn_test_denri_wotanh_R(512,512,tau_ninitializer = 'uniform',low_n = 0,high_n=4,vth= 1,dt = 1,branch = branch,device=device)
        #self.dense_3 = spike_rnn_test_denri_wotanh_R(300,256,tau_ninitializer = 'uniform',low_n = 0,high_n=4,vth= 1,dt = 1,branch = branch,device=device)
        self.dense_4 = nn.Linear(512,100)
        #self.dense_4 = readout_integrator_test(n,20,dt = 1,device=device)


    def forward(self,dvs_inp, out_mode = 'rate'):
        dvs_inp.to(device)
        batch_size, seq_len, channel, w, h = dvs_inp.size()
        dvs_inp = dvs_inp.permute([1,0,2,3,4])
        self.dense_1.set_neuron_state(batch_size)
        self.dense_2.set_neuron_state(batch_size)
        #self.dense_3.set_neuron_state(batch_size)
        output = 0
        for i in range(seq_len):

            spike_inp = dvs_inp[i].reshape(batch_size,-1)
            mem_layer1,spike_layer1 = self.dense_1.forward(spike_inp)
            mem_layer2,spike_layer2 = self.dense_2.forward(spike_layer1)
            #mem_layer3,spike_layer3 = self.dense_3.forward(spike_layer2)
            x = self.dense_4(spike_layer2)

            output += x

        return output / seq_len