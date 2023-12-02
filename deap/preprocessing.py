import numpy as np
from utils import load_DEAP
from sklearn.model_selection  import train_test_split
from torch.utils.data import TensorDataset
import torch
from sklearn.utils import shuffle
import os
import matplotlib.pyplot as plt
DATA_DIR = "/data/DEAP/data_preprocessed_python/"



def baseline_removal(data):
    """ 
    calculate the baseline signal per second 
    then subtract that baseline from the signal
    """
    # duration of the baseline
    baseline_dur = 3 
    # signal's sampling rate
    sampling_rate = 128
    preprocessed = []
    # loop through the data array (n_instance, n_channels, n_samples)
    for ins in range(data.shape[0]):
        preprocessed_ins = []
        for c in range(data.shape[1]):
            signal = data[ins, c]
            # get all 3 second baseline segment and split in to 3 1-second segments
            all_baseline = np.split(signal[:baseline_dur*sampling_rate], 3)
            signal = signal[baseline_dur*sampling_rate:]
            # calculate the per second mean baseline
            baseline_per_second = np.mean(all_baseline, axis = 0)
            # print(baseline_per_second.shape)
            baseline_to_remove = np.tile(baseline_per_second, int(len(signal)/sampling_rate))
            signal_baseline_removed = signal - baseline_to_remove
    
            signal_split = signal_baseline_removed.reshape(-1, 3*128)
            
            preprocessed_ins.append(signal_split)
        
        preprocessed.append(preprocessed_ins)
        

    return np.array(preprocessed).transpose(0, 2, 1, 3)


def dataset_prepare(segment_duration = 3, n_subjects = 32, load_all = True, single_subject = False, sampling_rate = 128,label_type = [0,2],subject = 1):
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for i in range(n_subjects):  
        data = load_DEAP(DATA_DIR, n_subjects = i, single_subject=True,load_all = False,label_type=label_type)
         
        s1, s1_labels, s1_names = data
        s1_labels = np.repeat(s1_labels.reshape(-1, 1), 20)

        s1_preprocessed = baseline_removal(s1)
        b, s, c, n = s1_preprocessed.shape
        s1_preprocessed = s1_preprocessed.reshape(b*s, c, -1)
        #print("preprocesed data shape: ", s1_preprocessed.shape)
        
        X_train, X_test, y_train, y_test = train_test_split(s1_preprocessed, s1_labels, test_size = 0.1, stratify = s1_labels, shuffle = True, random_state = 29)
        train_x.append(X_train)
        train_y.append(y_train)
        test_x.append(X_test)
        test_y.append(y_test)
    if single_subject:
        train_x = torch.Tensor(np.array(train_x[subject])).reshape(-1,32,segment_duration*sampling_rate).permute(0,2,1) # transform to torch tensor
        train_y = torch.Tensor(np.array(train_y[subject])).reshape(-1,1)
        test_x = torch.Tensor(np.array(test_x[subject])).reshape(-1,32,segment_duration*sampling_rate).permute(0,2,1)   # transform to torch tensor
        test_y = torch.Tensor(np.array(test_y[subject])).reshape(-1,1)  
    else:     
        train_x = torch.Tensor(np.array(train_x)).reshape(-1,32,segment_duration*sampling_rate).permute(0,2,1) # transform to torch tensor
        train_y = torch.Tensor(np.array(train_y)).reshape(-1,1)
        test_x = torch.Tensor(np.array(test_x)).reshape(-1,32,segment_duration*sampling_rate).permute(0,2,1)   # transform to torch tensor
        test_y = torch.Tensor(np.array(test_y)).reshape(-1,1)

    train_dataset = TensorDataset(train_x, train_y.long()) # create your datset
    test_dataset = TensorDataset(test_x, test_y.long())

    return train_dataset, test_dataset

def dataset_plot(segment_duration = 3, n_subjects = 32, load_all = True, single_subject = False, sampling_rate = 128,label_type = [0,2]):
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    data = load_DEAP(DATA_DIR, n_subjects = 1, single_subject=True,load_all = False,label_type=label_type)
    baseline_dur = 3 
    # signal's sampling rate
    sampling_rate = 128
    s1, s1_labels, s1_names = data        
    plt.rc('font',family='Arial')
    fig = plt.figure(figsize = (20,100))
    s1_labels = np.repeat(s1_labels.reshape(-1, 1), 20)
    data = s1
    for c in range(data.shape[1]):
        signal = data[0, c]

        ax = fig.add_subplot(32,1,c+1)
        ax.plot(signal)
        ax.spines['bottom'].set_linewidth(False)
        ax.spines['left'].set_linewidth(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_axis_off()
        # get all 3 second baseline segment and split in to 3 1-second segments
        '''
        all_baseline = np.split(signal[:baseline_dur*sampling_rate], 3)
        signal = signal[baseline_dur*sampling_rate:]
        # calculate the per second mean baseline
        baseline_per_second = np.mean(all_baseline, axis = 0)
        # print(baseline_per_second.shape)
        baseline_to_remove = np.tile(baseline_per_second, int(len(signal)/sampling_rate))
        signal_baseline_removed = signal - baseline_to_remove
    
        signal_split = signal_baseline_removed.reshape(-1, 3*128)
        '''
 
    plt.savefig('images/orgin_signal.png', format='png')
def dataset_prepare_for_KF(n_subjects = 1):
    s1, s1_labels, s1_names = load_DEAP(DATA_DIR, n_subjects = n_subjects, single_subject=True)    
    s1_labels = np.repeat(s1_labels.reshape(-1, 1), 20)

    s1_preprocessed = baseline_removal(s1)
    s1_preprocessed = s1_preprocessed.reshape(800, 32, 3, 128).transpose(0, 2, 1, 3)

    return s1_preprocessed, s1_labels
    
if __name__ == "__main__":
    dataset_prepare()

