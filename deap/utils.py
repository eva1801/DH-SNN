import os
import torch
import numpy as np
import pandas as pd
import pickle as cPickle
import random
from sklearn.utils import shuffle

def load_DEAP(data_dir, n_subjects = 26, single_subject = False, load_all = False, only_phys = False, only_EEG = True, label_type = [0, 2]):
    ## label_type [arousal: 0, valence: 1, n_classes]
    # get all files name to a list
    filenames = os.listdir(data_dir)
    filepaths = []
    for i in filenames:
        filepath = data_dir + "/" + i
        filepaths.append(filepath)

    if single_subject:

        train_paths = [filepaths[n_subjects-1]]
        train_names = [filenames[n_subjects-1]]
        #print(train_paths, "\n", train_names)
        train_data, train_labels = load_with_path(train_paths, label_type = label_type)

        return train_data, train_labels, train_names

    if load_all:

        train_paths = filepaths
        train_names = filenames
        # print(train_paths, "\n", train_names)
        train_data, train_labels = load_with_path(train_paths, label_type = label_type)

        return train_data, train_labels, train_names
        
    filepaths, filenames = shuffle(filepaths, filenames, random_state = 29)

    train_paths = filepaths[:n_subjects]
    test_paths = filepaths[n_subjects:]
    train_names = filenames[:n_subjects]
    test_names = filenames[n_subjects:]
    print(train_names)
    print(test_names)
    train_data, train_labels = load_with_path(train_paths, label_type = label_type)
    test_data, test_labels = load_with_path(test_paths, label_type = label_type)

        # keep_index = np.concatenate((low_index, high_index, mid_index))
        # all_labels = all_labels[keep_index].astype(np.uint8)
        # all_data = all_data[keep_index]
    print("train shape: ", train_data.shape)
    print("test shape: ", test_data.shape)

    return train_data, train_labels, train_names, test_data, test_labels, test_names


def load_with_path(filepaths, label_type = [0, 2], only_phys = False, only_EEG = True):
    all_data = []
    all_labels = []

    print(len(filepaths))
    for filepath in filepaths:
        loaddata = cPickle.load(open(filepath, 'rb'), encoding="latin1",)
        labels = loaddata['labels']
        new_data = loaddata['data'].astype(np.float32)
        if only_phys:
            new_data = new_data[:, 32:, :]
        elif only_EEG:
            new_data = new_data[:, :32, :]
        all_labels.append(labels)
        all_data.append(new_data)
    all_labels = np.array(all_labels)

    all_data = np.array(all_data)

    all_labels = all_labels.reshape(-1, all_labels.shape[-1])
    all_data = all_data.reshape(-1, all_data.shape[-2], all_data.shape[-1])
    #print("data shape:", all_data.shape)

    # print("all label before categorizing", all_labels)

    if label_type[1] == 1:
        all_labels = labels_quantization(all_labels, 4)    
        return all_data, all_labels
    elif label_type[1] == 2:
        # low_index = np.asarray(all_labels < 5).nonzero()[0]
        # high_index = np.asarray(all_labels > 5).nonzero()[0]

        # keep_index = np.concatenate((low_index, high_index))
        # all_labels = all_labels[keep_index]
        # all_labels = np.where(all_labels > 5, 1, 0)
        # all_data = all_data[keep_index]
        all_labels = labels_quantization(all_labels, 2)
    else:
        all_labels = labels_quantization(all_labels, 3)
    all_labels = all_labels[label_type[0]].squeeze()

    # print("all label after categorizing", all_labels)

    return all_data, all_labels


def labels_quantization(labels, num_classes):
    new_labels = labels
    if num_classes == 2:

        median_val = 5
        median_arousal = 5

        labels_val = np.zeros(new_labels.shape[0])
        labels_arousal = np.zeros(new_labels.shape[0])

        labels_val[(1 <= new_labels[:, 0]) & (new_labels[:, 0] <= median_val)] = 0
        labels_val[(median_val < new_labels[:, 0]) & (new_labels[:, 0] <= 9)] = 1

        labels_arousal[(1 <= new_labels[:, 1]) & (new_labels[:, 1] <= median_arousal)] = 0
        labels_arousal[(median_arousal < new_labels[:, 1]) & (new_labels[:, 1] <= 9)] = 1

    elif num_classes == 3:
        low_value = 3
        high_value = 6

        labels_val = np.zeros(new_labels.shape[0])
        labels_arousal = np.zeros(new_labels.shape[0])

        labels_val[(1 <= new_labels[:, 0]) & (new_labels[:, 0] <= low_value)] = 0
        labels_val[(low_value < new_labels[:, 0]) & (new_labels[:, 0] <= high_value)] = 1
        labels_val[(high_value < new_labels[:, 0]) & (new_labels[:, 0] <= 9)] = 2

        labels_arousal[(1 <= new_labels[:, 1]) & (new_labels[:, 1] <= low_value)] = 0
        labels_arousal[(low_value < new_labels[:, 1]) & (new_labels[:, 1] <= high_value)] = 1
        labels_arousal[(high_value < new_labels[:, 1]) & (new_labels[:, 1] <= 9)] = 2
    else:
        median_val = 5
        median_arousal = 5

        labels_all = np.zeros(new_labels.shape[0])


        labels_all[(1 <= new_labels[:, 0]) & (new_labels[:, 0] <= median_val) &(1 <= new_labels[:, 1]) & (new_labels[:, 1] <= median_arousal)] = 0
        labels_all[(median_val < new_labels[:, 0]) & (new_labels[:, 0] <= 9)&(1 <= new_labels[:, 1]) & (new_labels[:, 1] <= median_arousal)] = 1

        labels_all[(1 <= new_labels[:, 0]) & (new_labels[:, 0] <= median_val)&(median_arousal < new_labels[:, 1]) & (new_labels[:, 1] <= 9)] = 2
        labels_all[(median_val < new_labels[:, 0]) & (new_labels[:, 0] <= 9)&(median_arousal < new_labels[:, 1]) & (new_labels[:, 1] <= 9)] = 3
        return labels_all
    output_labels = [labels_val, labels_arousal]

    return np.array(output_labels)


# labels_2_classes = labels_quantization(labels,2)
# labels_3_classes = labels_quantization(labels, 3)
# print(labels_2_classes)
# print(labels_3_classes)

# val_labels_2 = labels_2_classes[0]
# ar_labels_2 = labels_2_classes[1]

# val_labels_3 = labels_3_classes[0]
# ar_labels_3 = labels_3_classes[1]

