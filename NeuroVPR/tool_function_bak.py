import os, argparse, subprocess, shlex, io, time, glob, pickle, pprint
import sys
import os

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import tqdm, fire
from PIL import Image

Image.warnings.simplefilter('ignore')

import torch
import torch.utils.model_zoo
from torch.utils.data import Dataset, DataLoader
import torchvision
import math
import numpy as np
import random
from sklearn.metrics import precision_recall_curve
# math.gcd: greatest common divisor
# lcm: least common multiple
def least_common_multiple(num):
    """
    num: a set of numbers
    """
    mini = 1
    for i in num:
        mini = int(i) * int(mini) /math.gcd(int(i), mini)
        mini = int(mini)
    return mini

# sequential dataset processing
class SequentialDataset(Dataset):
    """Sequence-based dataset."""
    def __init__(self, data_dir, exp_idx, transform, nclass=None, seq_len_aps=None, seq_len_dvs = None,
                 seq_len_gps=None,dvs_expand=None):
        """
        Args:
            data_dir (string): Directory with all data.
            exp_idx  (int): experience data index
            transform (callable, optional): Optional transform to be applied on a sample.
            nclass:
            seq_len_aps: length of sequencial image frames
            seq_len_dvs: length of sequencial  envet-based vision frames (voxel grid)
            seq_len_gps: length of sequencial postions
        """
        self.data_dir = data_dir
        self.transform = transform
        self.num_exp = len(exp_idx)
        self.exp_idx = exp_idx

        self.total_aps = [np.sort(os.listdir(data_dir + str(idx) + '/dvs_frames')) for idx in exp_idx]
        self.total_dvs = [np.sort(os.listdir(data_dir + str(idx) + '/dvs_5ms_5seq')) for idx in exp_idx]
        self.num_imgs = [len(x) for x in self.total_aps]
        self.raw_pos = [np.loadtxt(data_dir + str(idx) + '/position.txt', delimiter=' ') for idx in exp_idx]

        self.t_pos = [x[:, 0] for x in self.raw_pos]
        self.t_aps = [[float(x[:-4]) for x in y] for y in self.total_aps]
        self.t_dvs = [[float(x[:-4]) for x in y] for y in self.total_dvs]

        self.data_pos = [idx[:, 0:3] for idx in self.raw_pos]

        self.seq_len_aps = seq_len_aps
        self.seq_len_gps = seq_len_gps
        self.seq_len_dvs = seq_len_dvs

        self.seq_len = max(seq_len_gps, seq_len_aps)
        self.nclass = nclass

        self.lens = len(self.total_aps) - self.seq_len
        self.dvs_data = None
        self.duration = [x[-1] - x[0] for x in self.t_dvs]

        nums = 1e5
        for x in self.total_dvs:
            if len(x) < nums: nums = len(x)
        for x in self.total_dvs:
            if len(x) < nums: nums = len(x)
        for x in self.raw_pos:
            if len(x) < nums: nums = len(x)

        self.lens = nums

    def __len__(self):
        return self.lens - self.seq_len * 2

    def __getitem__(self, idx):
        # obtain images
        exp_index = np.random.randint(self.num_exp)
        idx = max(min(idx, self.num_imgs[exp_index] - self.seq_len * 2), self.seq_len_dvs * 3)
        # print('debug1')

        img_seq = []
        for i in range(self.seq_len_aps):
            img_loc = self.data_dir + str(self.exp_idx[exp_index]) + '/dvs_frames/' + \
                      self.total_aps[exp_index][idx - self.seq_len_aps + i]
            img_seq += [Image.open(img_loc).convert('RGB')]
        img_seq_pt = []

        if self.transform:
            for images in img_seq:
                img_seq_pt += [torch.unsqueeze(self.transform(images), 0)]

        img_seq = torch.cat(img_seq_pt, dim=0)
        # obtain position by matching time stamps
        t_stamps = self.raw_pos[exp_index][:, 0]
        t_target = self.t_aps[exp_index][idx]

        idx_pos = max(np.searchsorted(t_stamps, t_target), self.seq_len_aps)
        pos_seq = self.data_pos[exp_index][idx_pos - self.seq_len_gps:idx_pos, :]
        pos_seq = torch.from_numpy(pos_seq.astype('float32'))

        idx_dvs = np.searchsorted(self.t_dvs[exp_index], t_target, sorter=None) - 1
        t_stamps = self.t_dvs[exp_index][idx_dvs]
        dvs_seq = torch.zeros(self.seq_len_dvs * 5, 2, 32, 43)

        for i in range(self.seq_len_dvs):
            dvs_path = self.data_dir + str(self.exp_idx[exp_index]) + '/dvs_5ms_5seq/' \
                       + self.total_dvs[exp_index][idx_dvs - self.seq_len_dvs + i + 1]
            dvs_buf = torch.load(dvs_path)
            dvs_buf = dvs_buf.permute([1, 0, 2, 3])
            dvs_seq[i * 5: (i + 1) * 5] = torch.nn.functional.avg_pool2d(dvs_buf, 8)

        ids = int((t_stamps - self.t_dvs[exp_index][0]) / self.duration[exp_index] * self.nclass)
        ids = np.clip(ids, a_min=0, a_max= self.nclass - 1) 
        ids = np.array(ids)
        ids = torch.from_numpy(ids).type(torch.long)

        return (img_seq, pos_seq, dvs_seq), ids


def Data(data_path=None, batch_size=None, exp_idx=None, is_shuffle=True, normalize = None, nclass=None, seq_len_aps=None,
                 seq_len_dvs=None, seq_len_gps=None,dvs_expand=None):
    """
    Args:
        data_path (string): path with all data.
        batch_size  (int): batch size
        exp_idx :
        is_shuffle:
        normalize:
        nclass:
        seq_len_aps: length of sequencial image frames
        seq_len_dvs: length of sequencial  envet-based vision frames (voxel grid)
        seq_len_gps: length of sequencial postions
    """

    dataset = SequentialDataset(data_dir=data_path,
                                exp_idx=exp_idx,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.Resize((240, 320)),  # 640 480
                                    torchvision.transforms.ToTensor(),
                                    normalize,                                
                                ]), 
                                nclass=nclass, 
                                seq_len_aps=seq_len_aps, 
                                seq_len_dvs=seq_len_dvs, 
                                seq_len_gps=seq_len_gps,dvs_expand=dvs_expand)

    data_loader = torch.utils.data.DataLoader(dataset,
                                              shuffle=is_shuffle,
                                              drop_last=True,
                                              batch_size=batch_size,
                                              pin_memory=True,
                                              num_workers=8)
    return data_loader

def Data_mask(data_path=None, batch_size=None, exp_idx=None, is_shuffle=True, normalize = None, nclass=None, seq_len_aps=None,
                 seq_len_dvs=None, seq_len_gps=None):
    """
    Args:
        data_path (string): path with all data.
        batch_size  (int): batch size
        exp_idx :
        is_shuffle:
        normalize:
        nclass:
        seq_len_aps: length of sequencial image frames
        seq_len_dvs: length of sequencial  envet-based vision frames (voxel grid)
        seq_len_gps: length of sequencial postions
    """

    dataset = SequentialDataset(data_dir=data_path,
                                exp_idx=exp_idx,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.Resize((240, 320)),  # 640 480
                                    torchvision.transforms.ToTensor(),
                                    normalize,
                                    torchvision.transforms.RandomCrop(size=128, padding=128),
                                ]),
                                nclass=nclass,
                                seq_len_aps=seq_len_aps,
                                seq_len_dvs=seq_len_dvs,
                                seq_len_gps=seq_len_gps)

    data_loader = torch.utils.data.DataLoader(dataset,
                                              shuffle=is_shuffle,
                                              drop_last=True,
                                              batch_size=batch_size,
                                              pin_memory=True,
                                              num_workers=8)
    return data_loader

def Data_brightness(data_path=None, batch_size=None, exp_idx=None, is_shuffle=True, normalize = None, nclass=None, seq_len_aps=None,
                 seq_len_dvs=None, seq_len_gps=None):
    """
    Args:
        data_path (string): path with all data.
        batch_size  (int): batch size
        exp_idx :
        is_shuffle:
        normalize:
        nclass:
        seq_len_aps: length of sequencial image frames
        seq_len_dvs: length of sequencial  envet-based vision frames (voxel grid)
        seq_len_gps: length of sequencial postions
    """

    dataset = SequentialDataset(data_dir=data_path,
                                exp_idx=exp_idx,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.Resize((240, 320)),  # 640 480
                                    torchvision.transforms.ToTensor(),
                                    normalize,
                                    torchvision.transforms.ColorJitter(brightness=0.5),
                                ]),
                                nclass=nclass,
                                seq_len_aps=seq_len_aps,
                                seq_len_dvs=seq_len_dvs,
                                seq_len_gps=seq_len_gps)

    data_loader = torch.utils.data.DataLoader(dataset,
                                              shuffle=is_shuffle,
                                              drop_last=True,
                                              batch_size=batch_size,
                                              pin_memory=True,
                                              num_workers=8)
    return data_loader


def int_u(u, t, w, Iext, tau, I_inh, k, dt):
    # membrane potential dynamics
    # parameter configuration mainly followed by ref. (wu et al.2008)
    """
    Args:
        u:
        t:
        w:
        Iext:
        tau:
        I_inh:
        k:
        dt:
    """

    u = u.reshape(2, -1)

    r1 = np.square(u)
    r2 = 1.0 + k * np.sum(r1)
    r = r1 / r2

    Irec = np.dot(r, w)

    du = (-u + Irec + Iext + I_inh) * dt / tau

    return du.flatten()

def setup_seed(seed):
    """
    for network initialization, enable results can be repeated
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
    
def set_gpus(n=1):
    """
    Finds all GPUs on the system and restricts to n of them that have the most
    free memory.
    """
    gpus = subprocess.run(shlex.split(
        'nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,nounits'), check=True,
        stdout=subprocess.PIPE).stdout
    gpus = pd.read_csv(io.BytesIO(gpus), sep=', ', engine='python')
    gpus = gpus[gpus['memory.total [MiB]'] > 10000]  # only above 10 GB
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
        visible = [int(i)
                   for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        gpus = gpus[gpus['index'].isin(visible)]
    gpus = gpus.sort_values(by='memory.free [MiB]', ascending=False)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # making sure GPUs are numbered the same way as in nvidia_smi
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
        [str(i) for i in gpus['index'].iloc[:n]])


def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    """
    with torch.no_grad():
        _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = [correct[:k].sum().item() for k in topk]
        return res

def compute_matches(retrieved_all, ground_truth_info):
    matches=[]
    itr=0
    for retr in retrieved_all:
        if (retr == ground_truth_info[itr]):
            matches.append(1)
        else:
            matches.append(0)
        itr=itr+1
    return matches

def compute_precision_recall(matches,scores_all):
    precision, recall, _ = precision_recall_curve(matches, scores_all)
    return precision, recall