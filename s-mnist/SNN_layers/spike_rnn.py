import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from SNN_layers.spike_neuron import *
from SNN_layers.spike_dense import *


#Vanilla SRNN layer
class spike_rnn_test_origin(nn.Module):
    def __init__(self,input_dim,output_dim,
                 tau_minitializer = 'uniform',low_m = 0,high_m = 4,vth = 1,dt = 1,device='cpu',bias=True):
        """
        Args:
            input_dim(int): input dimension.
            output_dim(int): the number of readout neurons
            tau_minitializer(str): the method of initialization of tau_m
            low_m(float): the low limit of the init values of tau_m
            high_m(float): the upper limit of the init values of tau_m
            vth(float): threshold
        """
        super(spike_rnn_test_origin, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.vth = vth
        self.dt = dt

        self.dense = nn.Linear(input_dim+output_dim,output_dim)
        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))

        if tau_minitializer == 'uniform':
            nn.init.uniform_(self.tau_m,low_m,high_m)
        elif tau_minitializer == 'constant':
            nn.init.constant_(self.tau_m,low_m)


    def set_neuron_state(self,batch_size):
        self.mem = Variable(torch.rand(batch_size,self.output_dim)).to(self.device)
        self.spike = Variable(torch.rand(batch_size,self.output_dim)).to(self.device)
        self.v_th = Variable(torch.ones(batch_size,self.output_dim)*self.vth).to(self.device)
    def forward(self,input_spike):

        #concat the the forward inputs with the recurrent inputs 
        k_input = torch.cat((input_spike.float(),self.spike),1)
        #synaptic inputs
        d_input = self.dense(k_input)
        self.mem,self.spike = mem_update_pra(d_input,self.mem,self.spike,self.v_th,self.tau_m,self.dt,device=self.device)
        
        return self.mem,self.spike

#vanilla SRNN with the noreset LIF neuron
class spike_rnn_test_origin_noreset(nn.Module):
    def __init__(self,input_dim,output_dim,
                 tau_minitializer = 'uniform',low_m = 0,high_m = 4,vth = 0.5,dt = 1,device='cpu',bias=True):
        """
        Args:
            input_dim(int): input dimension.
            output_dim(int): the number of readout neurons
            tau_minitializer(str): the method of initialization of tau_m
            low_m(float): the low limit of the init values of tau_m
            high_m(float): the upper limit of the init values of tau_m
            vth(float): threshold
        """
        super(spike_rnn_test_origin_noreset, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        #self.is_adaptive = is_adaptive
        self.device = device
        self.vth = vth
        self.dt = dt

        self.dense = nn.Linear(input_dim+output_dim,output_dim)
        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))

        if tau_minitializer == 'uniform':
            nn.init.uniform_(self.tau_m,low_m,high_m)
        elif tau_minitializer == 'constant':
            nn.init.constant_(self.tau_m,low_m)


    def set_neuron_state(self,batch_size):
        self.mem = Variable(torch.rand(batch_size,self.output_dim)).to(self.device)
        self.spike = Variable(torch.rand(batch_size,self.output_dim)).to(self.device)
        self.v_th = Variable(torch.ones(batch_size,self.output_dim)*self.vth).to(self.device)
    def forward(self,input_spike):

        #concat the the forward inputs with the recurrent inputs 
        k_input = torch.cat((input_spike.float(),self.spike),1)
        #synaptic inputs
        d_input = self.dense(k_input)
        self.mem,self.spike = mem_update_pra_noreset(d_input,self.mem,self.spike,self.v_th,self.tau_m,self.dt,device=self.device)
        
        return self.mem,self.spike





#DH-SRNN layer
class spike_rnn_test_denri_wotanh_R(nn.Module):
    def __init__(self,input_dim,output_dim,tau_minitializer = 'uniform',low_m = 0,high_m = 4,
                 tau_ninitializer = 'uniform',low_n = 0,high_n = 4,vth = 0.5,dt = 4,branch = 4,device='cpu',bias=True):
        """
        Args:
            input_dim(int): input dimension.
            output_dim(int): the number of readout neurons
            tau_minitializer(str): the method of initialization of tau_m
            low_m(float): the low limit of the init values of tau_m
            high_m(float): the upper limit of the init values of tau_m
            tau_ninitializer(str): the method of initialization of tau_n
            low_n(float): the low limit of the init values of tau_n
            high_n(float): the upper limit of the init values of tau_n
            vth(float): threshold
            branch(int): the number of dendritic branches
        """
        super(spike_rnn_test_denri_wotanh_R, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.vth = vth
        self.dt = dt
        mask_rate = 1/branch
        self.pad = ((input_dim+output_dim)//branch*branch+branch-(input_dim+output_dim)) % branch
        self.dense = nn.Linear(input_dim+output_dim+self.pad,output_dim*branch)

        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))
        self.tau_n = nn.Parameter(torch.Tensor(self.output_dim,branch))
        #the number of dendritic branch
        self.branch = branch

        self.create_mask()
        
        # timing factor of membrane potential
        if tau_minitializer == 'uniform':
            nn.init.uniform_(self.tau_m,low_m,high_m)
        elif tau_minitializer == 'constant':
            nn.init.constant_(self.tau_m,low_m)
        # timing factor of dendritic branches       
        if tau_ninitializer == 'uniform':
            nn.init.uniform_(self.tau_n,low_n,high_n)
        elif tau_ninitializer == 'constant':
            nn.init.constant_(self.tau_n,low_n)

    
    def parameters(self):
        return [self.dense.weight,self.dense.bias,self.tau_m,self.tau_n]
    
    #init
    def set_neuron_state(self,batch_size):

        self.mem = Variable(torch.rand(batch_size,self.output_dim)).to(self.device)
        self.spike = Variable(torch.rand(batch_size,self.output_dim)).to(self.device)
        self.d_input = Variable(torch.zeros(batch_size,self.output_dim,self.branch)).to(self.device)

        self.v_th = Variable(torch.ones(batch_size,self.output_dim)*self.vth).to(self.device)

    #create connection pattern
    def create_mask(self):
        input_size = self.input_dim+self.output_dim+self.pad
        self.mask = torch.zeros(self.output_dim*self.branch,input_size).to(self.device)
        for i in range(self.output_dim):
            seq = torch.randperm(input_size)
            for j in range(self.branch):
                self.mask[i*self.branch+j,seq[j*input_size // self.branch:(j+1)*input_size // self.branch]] = 1
    def apply_mask(self):
        self.dense.weight.data = self.dense.weight.data*self.mask
    def forward(self,input_spike):
        # timing factor of dendritic branches
        beta = torch.sigmoid(self.tau_n)
        padding = torch.zeros(input_spike.size(0),self.pad).to(self.device)
        k_input = torch.cat((input_spike.float(),self.spike,padding),1)
        #update dendritic currents 
        self.d_input = beta*self.d_input+(1-beta)*self.dense(k_input).reshape(-1,self.output_dim,self.branch)
        #summation of dendritic currents
        l_input = (self.d_input).sum(dim=2,keepdim=False)
        
        #update membrane potential and generate spikes
        self.mem,self.spike = mem_update_pra(l_input,self.mem,self.spike,self.v_th,self.tau_m,self.dt,device=self.device)
        
        return self.mem,self.spike


