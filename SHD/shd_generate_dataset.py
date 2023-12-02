'''
the preprocessing code refers to the work(Yin, B., Corradi, F. & Bohté, S. M. Accurate and efficient time-domain classification with adaptive spiking recurrent neural
networks. Nat. Mach. Intell. 3, 905–913 (2021)).
'''
import os
import urllib.request
import gzip, shutil
# from keras.utils import get_file
import matplotlib.pyplot as plt
"""
The dataset is 48kHZ with 24bits precision
* 700 channels
* longest 1.17s
* shortest 0.316s
"""



files = ['/data/SHD/shd_test.h5','/data/SHD/shd_train.h5']

import tables
import numpy as np
fileh = tables.open_file(files[1], mode='r')
units = fileh.root.spikes.units
times = fileh.root.spikes.times
labels = fileh.root.labels

# This is how we access spikes and labels
index = 0
print("Times (ms):", times[index],max(times[index]))
print("Unit IDs:", units[index])
print("Label:", labels[index])

def poisson_spikes_gen(nb_steps,nb_units,  rate):

    spike_trains = (np.random.uniform(0, 1, ( nb_steps,nb_units)) <= rate ).astype(int)

    return spike_trains

def binary_image_readout(times,units,dt = 1e-3):
    img = []
    N = int(1/dt)
    for i in range(N):
        idxs = np.argwhere(times<=i*dt).flatten()
        vals = units[idxs]
        vals = vals[vals > 0]
        vector = np.zeros(700)
        vector[700-vals] = 1
        times = np.delete(times,idxs)
        units = np.delete(units,idxs)
        img.append(vector)
    return np.array(img)

def binary_image_readout_random(times,units,dt = 1e-3,max_timestep = 1000):
    img = []
    N = int(1/dt)
    for i in range(N):
        idxs = np.argwhere(times<=i*dt).flatten()
        vals = units[idxs]
        vals = vals[vals > 0]
        vector = np.zeros(700)
        vector[700-vals] = 1
        times = np.delete(times,idxs)
        units = np.delete(units,idxs)
        img.append(vector)
    if N<max_timestep :
        img = np.array(img)
        len = np.random.randint(0,max_timestep-N)
        head = poisson_spikes_gen(len,700,0.01)
        tail =   poisson_spikes_gen(max_timestep-N-len,700,0.01)  
        out = np.concatenate([head,img,tail])
        return np.vstack([head,np.array(img),tail])
    else:
        return np.array(img)


def generate_dataset(file_name,output_dir,dt=1e-3):
    fileh = tables.open_file(file_name, mode='r')
    units = fileh.root.spikes.units
    times = fileh.root.spikes.times
    labels = fileh.root.labels
    os.mkdir(output_dir)
    # This is how we access spikes and labels
    index = 0
    print("Number of samples: ",len(times))
    for i in range(len(times)):
        x_tmp = binary_image_readout(times[i], units[i],dt=dt)
        y_tmp = labels[i]
        output_file_name = output_dir+'ID:'+str(i)+'_'+str(y_tmp)+'.npy'
        np.save(output_file_name, x_tmp)
    print('done..')
    return 0


generate_dataset(files[0],output_dir='/data/SHD/test_1ms/',dt=1e-3)

generate_dataset(files[1],output_dir='/data/SHD/train_1ms/',dt=1e-3)
