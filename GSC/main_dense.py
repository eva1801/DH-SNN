import os
import sys
sys.path.append("..")
import time
import numpy as np
import scipy.io.wavfile as wav
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader
import torchvision
from torch.optim.lr_scheduler import StepLR,MultiStepLR,LambdaLR,ExponentialLR
from data import SpeechCommandsDataset,Pad, MelSpectrogram, Rescale,Normalize
from utils import generate_random_silence_files
import warnings
warnings.filterwarnings("ignore")
dtype = torch.float
torch.manual_seed(0) 
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
#data dir
train_data_root = "/data/speech_commands"
test_data_root = "/data/speech_commands"

training_words = os.listdir(train_data_root)
training_words = [x for x in training_words if os.path.isdir(os.path.join(train_data_root,x))]
training_words = [x for x in training_words if os.path.isdir(os.path.join(train_data_root,x)) if x[0] != "_" ]
print("{} training words:".format(len(training_words)))
print(training_words)

# generate the 12 labels
testing_words =["yes", "no", 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
"""
testing_words = [x for x in testing_words if os.path.isdir(os.path.join(train_data_root,x))]
testing_words = [x for x in testing_words if os.path.isdir(os.path.join(train_data_root,x)) 
                 if x[0] != "_"]
"""
print("{} testing words:".format(len(testing_words)))
print(testing_words)

label_dct = {k:i for i,k in enumerate(testing_words+ ["_silence_", "_unknown_"])}

for w in training_words:
    label = label_dct.get(w)
    if label is None:
        label_dct[w] = label_dct["_unknown_"]

print("label_dct:")
print(label_dct)

sr = 16000
size = 16000

noise_path = os.path.join(train_data_root, "_background_noise_")
noise_files = []
for f in os.listdir(noise_path):
    if f.endswith(".wav"):
        full_name = os.path.join(noise_path, f)
        noise_files.append(full_name)
print("noise files:")
print(noise_files)

# generate silence training and validation data

silence_folder = os.path.join(train_data_root, "_silence_")
if not os.path.exists(silence_folder):
    os.makedirs(silence_folder)
    # 260 validation / 2300 training
    generate_random_silence_files(2560, noise_files, size, os.path.join(silence_folder, "rd_silence"))

    # save 260 files for validation
    silence_files = [fname for fname in os.listdir(silence_folder)]
    with open(os.path.join(train_data_root, "silence_validation_list.txt"),"w") as f:
        f.writelines("_silence_/"+ fname + "\n" for fname in silence_files[:260])

n_fft = int(30e-3*sr)
hop_length = int(10e-3*sr)
n_mels = 40
fmax = 4000
fmin = 20
delta_order = 2
stack = True
#MFCC
melspec = MelSpectrogram(sr, n_fft, hop_length, n_mels, fmin, fmax, delta_order, stack=stack)
pad = Pad(size)
rescale = Rescale()
normalize = Normalize()

transform = torchvision.transforms.Compose([pad,melspec,rescale])


def collate_fn(data):
    
    X_batch = np.array([d[0] for d in data])
    std = X_batch.std(axis=(0,2), keepdims=True)
    X_batch = torch.tensor(X_batch/std)
    y_batch = torch.tensor([d[1] for d in data])
    
    return X_batch, y_batch 

batch_size = 200
#generate dataset
train_dataset = SpeechCommandsDataset(train_data_root, label_dct, transform = transform, mode="train", max_nb_per_class=None)
train_sampler = torch.utils.data.WeightedRandomSampler(train_dataset.weights,len(train_dataset.weights))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8,sampler=train_sampler, collate_fn=collate_fn)

valid_dataset = SpeechCommandsDataset(train_data_root, label_dct, transform = transform, mode="valid", max_nb_per_class=None)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn)

test_dataset = SpeechCommandsDataset(test_data_root, label_dct, transform = transform, mode="test")
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn)

#####################################################################################################################3
# create network

from SNN_layers.spike_dense import *
from SNN_layers.spike_neuron import *
from SNN_layers.spike_rnn import *
thr_func = ActFun_adp.apply  
is_bias=True
#DH-SFNNs with 3-layer
class Dense_test(nn.Module):
    def __init__(self,):
        super(Dense_test, self).__init__()
        n = 200
        # is_bias=False
        #self.dense_1 = spike_dense(700,n,
        #                            tauM = 20,tauM_inital_std=5,device=device,bias=is_bias)
        self.dense_1 = spike_dense_test_denri_wotanh_R(40*3,n,vth= 1,dt = 1,branch = 8,device=device,bias=is_bias)
        self.dense_2 = spike_dense_test_denri_wotanh_R(n,n,vth= 1,dt = 1,branch = 8,device=device,bias=is_bias)
        self.dense_3 = spike_dense_test_denri_wotanh_R(n,n,vth= 1,dt = 1,branch = 8,device=device,bias=is_bias)
        
        self.dense_4 = readout_integrator_test(n,12,dt = 1,device=device,bias=is_bias)





    def forward(self,input):
        input.to(device)
        b,channel,seq_length,input_dim = input.shape
        #self.dense_1.set_neuron_state(b)
        self.dense_2.set_neuron_state(b)
        self.dense_1.set_neuron_state(b)
        self.dense_3.set_neuron_state(b)
        self.dense_4.set_neuron_state(b)
        output = 0
        input_s = input

        for i in range(seq_length):

            input_x = input_s[:,:,i,:].reshape(b,channel*input_dim)

            mem_layer1,spike_layer1 = self.dense_1.forward(input_x)
            
            mem_layer2,spike_layer2 = self.dense_2.forward(spike_layer1)
            mem_layer3,spike_layer3 = self.dense_3.forward(spike_layer2)
            mem_layer4= self.dense_4.forward(spike_layer3)

            output += mem_layer4


        output = F.log_softmax(output/seq_length,dim=1)
        return output




model = Dense_test()
criterion = nn.CrossEntropyLoss()


print("device:",device)
model.to(device)

def test(data_loader,is_show = 0):
    test_acc = 0.
    sum_sample = 0.
    fr_ = []
    for i, (images, labels) in enumerate(data_loader):
        #apply the connection pattern
        model.dense_1.apply_mask()
        model.dense_2.apply_mask()
        model.dense_3.apply_mask()
        images = images.view(-1,3,101, 40).to(device)
        
        labels = labels.view((-1)).long().to(device)
        predictions= model(images)
        #fr_.append(fr)
        _, predicted = torch.max(predictions.data, 1)
        labels = labels.cpu()
        predicted = predicted.cpu().t()

        test_acc += (predicted ==labels).sum()
        sum_sample+=predicted.numel()

    return test_acc.data.cpu().numpy()/sum_sample


def train(epochs,criterion,optimizer,scheduler=None):
    acc_list = []
    best_acc = 0
    path = 'model/dense_layer3_200neuron_denri_branch8_initzero_MG'  # .pth'
    for epoch in range(epochs):
        train_acc = 0
        sum_sample = 0
        train_loss_sum = 0
        for i, (images, labels) in enumerate(train_dataloader):
            #apply the connection pattern
            model.dense_1.apply_mask()
            model.dense_2.apply_mask()
            model.dense_3.apply_mask()
            # if i ==0:
            images = images.view(-1,3,101, 40).to(device)
 
            labels = labels.view((-1)).long().to(device)
            optimizer.zero_grad()

            predictions= model(images)
            _, predicted = torch.max(predictions.data, 1)
            train_loss = criterion(predictions,labels)
            
            # print(predictions,predicted)

            train_loss.backward()
            train_loss_sum += train_loss.item()
            optimizer.step()

            labels = labels.cpu()
            predicted = predicted.cpu().t()

            train_acc += (predicted ==labels).sum()
            sum_sample+=predicted.numel()

        if scheduler:
            scheduler.step()
        train_acc = train_acc.data.cpu().numpy()/sum_sample
        valid_acc = test(test_dataloader,1)
        train_loss_sum+= train_loss

        acc_list.append(train_acc)
        print('lr: ',optimizer.param_groups[0]["lr"])
        if valid_acc>best_acc and train_acc>0.890:
            best_acc = valid_acc
            torch.save(model, path+str(best_acc)[:7]+'-srnn.pth')

        print('epoch: {:3d}, Train Loss: {:.4f}, Train Acc: {:.4f},Valid Acc: {:.4f}'.format(epoch,
                                                                           train_loss_sum/len(train_dataloader),
                                                                           train_acc,valid_acc), flush=True)
    return acc_list


learning_rate = 1e-2


test_acc = test(test_dataloader)
print(test_acc)
if is_bias:
    base_params = [


                    model.dense_1.dense.weight,
                    model.dense_1.dense.bias,

                    model.dense_2.dense.weight,
                    model.dense_2.dense.bias,

                    model.dense_3.dense.weight,
                    model.dense_3.dense.bias,
                    model.dense_4.dense.weight,
                    model.dense_4.dense.bias,
                    ]


optimizer = torch.optim.Adam([
                              {'params': base_params, 'lr': learning_rate},

                              {'params': model.dense_4.tau_m, 'lr': learning_rate * 2},  
                              {'params': model.dense_1.tau_m, 'lr': learning_rate * 2},  
                              {'params': model.dense_1.tau_n, 'lr': learning_rate * 2},  
                              {'params': model.dense_2.tau_m, 'lr': learning_rate * 2},  
                              {'params': model.dense_2.tau_n, 'lr': learning_rate * 2},  
                              {'params': model.dense_3.tau_m, 'lr': learning_rate * 2},  
                              {'params': model.dense_3.tau_n, 'lr': learning_rate * 2}, 
                              ],
                        lr=learning_rate)


scheduler = StepLR(optimizer, step_size=25, gamma=.5) # 20
# epoch=0
epochs =150

acc_list = train(epochs,criterion,optimizer,scheduler)

test_acc = test(test_dataloader)
print(test_acc)