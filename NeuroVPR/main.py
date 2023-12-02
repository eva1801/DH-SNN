import os
import torch
import matplotlib.pyplot as plt

# torch.multiprocessing.set_start_method('spawn')
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
print('todo 0502: benchmark for pretrained_models')
# to maintain the same parameter configuration
saving_name = 'tiny_snn_room_epoch100_origin_exp4_seq5_time5ms_3layer_testv4'
from PIL import Image
from tiny_spiking_model import *


Image.warnings.simplefilter('ignore')
import torch.utils.model_zoo
from tool_function_bak import *

import torchvision
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
setup_seed(0)
# weight component of three parts
w_fps = 0.
w_gps = 0
w_dvs = 1
dvs_seq = 3
seq_len_aps = 3
seq_len_gps = 3
seq_len_dvs = 4  # for each seq_len_dvs, it will expand to three times due to the high resolution
dvs_expand = 3
expand_len = least_common_multiple([seq_len_aps, seq_len_dvs * dvs_expand, seq_len_gps])
branch=4
sparse_lambdas = 2
r = 0.1
train_exp_idx = [1, 2, 3, 5, 6, 7]#,9,10,11]
test_exp_idx = [4]#,8]
# path set

data_path = '/data/room/room_v'

# SNN_structure


batch_size = 20
n_class = 100
learning_rate = 1e-4

num_epoch = 100
num_iter = 100  # for each epoch, sample num_iter times from multiple exp.

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

normalize = torchvision.transforms.Normalize(mean=[0.3537, 0.3537, 0.3537],
                                             std=[0.3466, 0.3466, 0.3466])

train_loader = Data(data_path, batch_size=batch_size, exp_idx=train_exp_idx, is_shuffle=True,
                    normalize=normalize, nclass=n_class,
                    seq_len_aps=seq_len_aps, seq_len_dvs=seq_len_dvs, seq_len_gps=seq_len_gps,dvs_expand=dvs_expand)
test_loader = Data(data_path, batch_size=batch_size, exp_idx=test_exp_idx, is_shuffle=True,
                   normalize=normalize, nclass=n_class,
                   seq_len_aps=seq_len_aps, seq_len_dvs=seq_len_dvs, seq_len_gps=seq_len_gps,dvs_expand=dvs_expand)

img_size = None

#model
class DeepSeqSLAM(nn.Module):
    def __init__(self, num_classes=n_class, cann_num=cann_num):
        super(DeepSeqSLAM, self).__init__()
        #
        self.snn = Dense_test_4layer(branch).to(device)
        self.num_classes = num_classes

    def forward(self, inp, epoch=100):
        self.snn.dense_1.apply_mask()
        self.snn.dense_2.apply_mask()
        self.snn.dense_3.apply_mask()
        # get the dvs sensory inputs
        dvs_inp = inp[2].to(device)
        output = self.snn(dvs_inp)

        return output

snn = DeepSeqSLAM()
snn.to(device)
optimizer = torch.optim.Adam(snn.parameters(), lr=1e-3)  

lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
criterion = nn.CrossEntropyLoss()
record = {}
record['loss'], record['top1'], record['top5'], record['top10'] = [], [], [], []
best_test_acc1, best_test_acc5, best_recall, best_test_acc10 = 0., 0., 0, 0

train_iters = iter(train_loader)
iters = 0
start_time = time.time()
print(device)

import torchmetrics

best_recall = 0.
test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_class)
# test_auc = torchmetrics.AUC(average = 'macro', num_classes=n_class)
test_recall = torchmetrics.Recall(task="multiclass",average='none', num_classes=n_class)
test_precision = torchmetrics.Precision(task="multiclass",average='none', num_classes=n_class)
#training
for epoch in range(num_epoch):
    ## for training
    running_loss = 0.
    counts = 1.
    acc1_record, acc5_record, acc10_record = 0., 0., 0.
    while iters < num_iter:
        # print(iters)
        snn.train()
        optimizer.zero_grad()
        try:
            inputs, target = next(train_iters)
        except StopIteration:
            train_iters = iter(train_loader)
            inputs, target = next(train_iters)

        outputs = snn(inputs, epoch=epoch)
        class_loss = criterion(outputs, target.to(device))

        loss = class_loss
        loss.backward()
        optimizer.step()
        running_loss += loss.cpu().item()
        acc1, acc5, acc10 = accuracy(outputs.cpu(), target, topk=(1, 5, 10))
        acc1, acc5, acc10 = acc1 / len(outputs), acc5 / len(outputs), acc10 / len(outputs)
        acc1_record += acc1
        acc5_record += acc5
        acc10_record += acc10
        counts += 1
        iters += 1.
    iters = 0
    sparse_loss = 0
    record['loss'].append(loss.item())
    print('\n\nTime elaspe:', time.time() - start_time, 's')
    print(
        'Training epoch %.1d, training loss :%.4f, sparse loss :%.4f, training Top1 acc: %.4f, training Top5 acc: %.4f' %
        (epoch, running_loss / (num_iter), sparse_lambdas * sparse_loss, acc1_record / counts, acc5_record / counts))

    lr_schedule.step()
    start_time = time.time()

    ## for testing
    running_loss = 0.
    snn.eval()
    with torch.no_grad():
        acc1_record, acc5_record, acc10_record = 0., 0., 0.
        counts = 1.
        for batch_idx, (inputs, target) in enumerate(test_loader):
            outputs = snn(inputs, epoch=epoch)
            loss = criterion(outputs.cpu(), target)
            running_loss += loss.item()
            acc1, acc5, acc10 = accuracy(outputs.cpu(), target, topk=(1, 5, 10))
            acc1, acc5, acc10 = acc1 / len(outputs), acc5 / len(outputs), acc10 / len(outputs)
            acc1_record += acc1
            acc5_record += acc5
            acc10_record += acc10
            counts += 1
            outputs = outputs.cpu()
            test_acc(outputs.argmax(1), target)
            test_recall(outputs.argmax(1), target)
            test_precision(outputs.argmax(1), target)
    #metrics
    total_acc = test_acc.compute().mean()
    total_recall = test_recall.compute().mean()
    total_precison = test_precision.compute().mean()

    # print('Test Accuracy : %.4f, Test recall : %.4f, Test Precision : %.4f'%(total_acc, total_recall,total_precison))

    test_precision.reset()
    test_recall.reset()
    test_acc.reset()

    acc1_record = acc1_record / counts
    acc5_record = acc5_record / counts
    acc10_record = acc10_record / counts

    record['top1'].append(acc1_record)
    record['top5'].append(acc5_record)
    record['top10'].append(acc10_record)

    print(
        'Testing epoch %.1d,  loss :%.4f,  Top1 acc: %.4f,  Top5 acc: %.4f,   Top10 acc: %.4f, recall: %.4f, precision: %.4f,  best Acc1 : %.4f, best Acc5 %.4f, best recall %.4f' % (
            epoch, running_loss / (batch_idx + 1), acc1_record, acc5_record, acc10_record, total_recall, total_precison,
            best_test_acc1, best_test_acc5, best_recall))

    print('Current best Top1, ', best_test_acc1, 'Best Top5, ...', best_test_acc5)
    #record
    if epoch > 1:
        if best_test_acc1 < acc1_record:
            best_test_acc1 = acc1_record
            print('Achiving the best Top1, saving...', best_test_acc1)

        if best_test_acc5 < acc5_record:
            # best_test_acc1 = acc1_record
            best_test_acc5 = acc5_record
            print('Achiving the best Top5, saving...', best_test_acc5)

        if best_recall < total_recall:
            # best_test_acc1 = acc1_record
            best_recall = total_recall
            print('Achiving the best recall, saving...', best_recall)

        if best_test_acc10 < acc10_record:
            best_test_acc10 = acc10_record

        state = {
            'net': snn.state_dict(),
            'snn': snn.snn.state_dict(),
            'record': record,
            'best_recall': best_recall,
            'best_acc1': best_test_acc1,
            'best_acc5': best_test_acc5,
            'best_acc10': best_test_acc10
        }
        if not os.path.isdir('./checkpoint'):
            os.mkdir('./checkpoint')
        torch.save(state, './checkpoint/' + saving_name + '.t7')











