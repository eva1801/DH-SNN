import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

surrograte_type = 'MG'
print('gradient type: ', surrograte_type)


gamma = 0.5
lens = 0.5
R_m = 1


beta_value = 1.8
b_j0_value = 0.01



def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma

# define approximate firing function

class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):  # input = membrane potential- threshold
        ctx.save_for_backward(input)
        return input.gt(0).float()  # is firing ???

    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # temp = abs(input) < lens
        scale = 6.0
        hight = .15
        if surrograte_type == 'G':
            temp = torch.exp(-(input**2)/(2*lens**2))/torch.sqrt(2*torch.tensor(math.pi))/lens
        #multi gaussian
        elif surrograte_type == 'MG':
            temp = gaussian(input, mu=0., sigma=lens) * (1. + hight) \
                - gaussian(input, mu=lens, sigma=scale * lens) * hight \
                - gaussian(input, mu=-lens, sigma=scale * lens) * hight
        elif surrograte_type =='linear':
            temp = F.relu(1-input.abs())
        elif surrograte_type == 'slayer':
            temp = torch.exp(-5*input.abs())
        elif surrograte_type == 'rect':
            temp = input.abs() < 0.5
        return grad_input * temp.float()*gamma
    

    
act_fun_adp = ActFun_adp.apply    



def mem_update_pra(inputs, mem, spike, v_th, tau_m, dt=1,device=None):
    """
    neural model with soft reset
    """   
    alpha = torch.sigmoid(tau_m)
    mem = mem * alpha  + (1 - alpha) * R_m * inputs-v_th*spike
    inputs_ = mem - v_th

    spike = act_fun_adp(inputs_)  
    return mem, spike


def mem_update_pra_noreset(inputs, mem, spike, v_th, tau_m, dt=1,device=None):
    """
    neural model without reset
    Args:
        input(float): soma input.
        mem(float): soma membrane potential
        spike(int): spike or not spike
        vth(float): threshold
        tau_m(float): time factors of soma
    """   
    alpha = torch.sigmoid(tau_m)
    #without reset
    mem = mem * alpha  + (1 - alpha) * R_m * inputs#-v_th*spike
    inputs_ = mem - v_th

    spike = act_fun_adp(inputs_)  
    return mem, spike
def mem_update_pra_hardreset(inputs, mem, spike, v_th, tau_m, dt=1,device=None):
    """
    neural model with hard reset
    Args:
        input(float): soma input.
        mem(float): soma membrane potential
        spike(int): spike or not spike
        vth(float): threshold
        tau_m(float): time factors of soma
    """   
    alpha = torch.sigmoid(tau_m)
    #hard reset
    mem = mem * alpha*(1-spike)  + (1 - alpha) * R_m * inputs#-v_th*spike
    inputs_ = mem - v_th

    spike = act_fun_adp(inputs_)  
    return mem, spike


def output_Neuron_pra(inputs, mem, tau_m, dt=1,device=None):
    """
    The read out neuron is leaky integrator without spike
    Args:
        input(float): soma input.
        mem(float): soma membrane potential
        tau_m(float): time factors of soma
    """
    alpha = torch.sigmoid(tau_m).to(device)
    mem = mem *alpha +  (1-alpha)*inputs
    return mem