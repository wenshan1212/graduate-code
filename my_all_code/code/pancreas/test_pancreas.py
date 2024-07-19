from asyncore import write
from audioop import avg
from cgi import test
import imp
from multiprocessing import reduction
from turtle import pd
from unittest import loader, result

from yaml import load
import torch
import os
import pdb
import torch.nn as nn

from tqdm import tqdm as tqdm_load
from pancreas_utils_all import *
from test_util import *
from losses import DiceLoss, softmax_mse_loss, mix_loss
from dataloaders_all import get_ema_model_and_dataloader



"""Global Variables"""
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
seed_test = 1337
seed_reproducer(seed = seed_test)

data_root, split_name = '/workspace/BCP/code/pancreas/data_lists/pancreas/data', 'pancreas'
result_dir = 'result/cutmix_2.02/'
mkdir(result_dir)
batch_size, lr = 2, 1e-3
pretraining_epochs, self_training_epochs = 60, 200
pretrain_save_step, st_save_step, pred_step = 20, 20, 5
alpha, consistency, consistency_rampup = 0.99, 0.1, 40
label_percent = 20
u_weight = 1.5
connect_mode = 2
try_second = 1
sec_t = 0.5
self_train_name = 'self_train'

sub_batch = int(batch_size/2)
consistency_criterion = softmax_mse_loss
CE = nn.CrossEntropyLoss()
CE_r = nn.CrossEntropyLoss(reduction='none')
DICE = DiceLoss(nclass=2)
patch_size = 64

logger = None
overall_log = 'cutmix_log.txt'


def test_model(net, test_loader):
    load_path = Path(result_dir) / self_train_name
    print(load_path)
    load_net(net, load_path / 'best_ema_20_self.pth')
    print('Successful Loaded')
    avg_metric, m_list = test_calculate_metric(net, test_loader.dataset, s_xy=16, s_z=4)
    test_dice = avg_metric[0]
    return avg_metric, m_list


if __name__ == '__main__':
    try:
        net, static_teacher_net, dynamic_teacher_net, optimizer, lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b, test_loader = get_ema_model_and_dataloader(data_root, split_name, batch_size, lr, labelp=label_percent)
        avg_metric, m_list = test_model(net, test_loader)
        
        print(avg_metric)
    except Exception as e:
        logger.exception("BUG FOUNDED ! ! !")


### python test_pancreas.py