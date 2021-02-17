import os
from datetime import datetime
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import io
from torchvision import transforms as trans
from data.data_pipe import de_preprocess
import torch
import torch.nn as nn
from model.arcface import l2_norm
import pdb
import cv2
import logging
import math
#from torch.optim.lr_scheduler import _LRScheduler

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def separate_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])
    return paras_only_bn, paras_wo_bn

hflip = trans.Compose([
            de_preprocess,
            trans.ToPILImage(),
            trans.functional.hflip,
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

def hflip_batch(imgs_tensor):
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)
    return hfliped_imgs

def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)
    plot = plt.plot(fpr, tpr, linewidth=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

# class CosineAnnealingWarmUpRestarts(_LRScheduler):
#     def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
#         if T_0 <= 0 or not isinstance(T_0, int):
#             raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
#         if T_mult < 1 or not isinstance(T_mult, int):
#             raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
#         if T_up < 0 or not isinstance(T_up, int):
#             raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
#         self.T_0 = T_0
#         self.T_mult = T_mult
#         self.base_eta_max = eta_max
#         self.eta_max = eta_max
#         self.T_up = T_up
#         self.T_i = T_0
#         self.gamma = gamma
#         self.cycle = 0
#         self.T_cur = last_epoch
#         super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
#         #self.T_cur = last_epoch
#
#     def get_lr(self):
#         if self.T_cur == -1:
#             return self.base_lrs
#         elif self.T_cur < self.T_up:
#             return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
#         else:
#             return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
#                     for base_lr in self.base_lrs]
#
#     def step(self, epoch=None):
#         if epoch is None:
#             epoch = self.last_epoch + 1
#             self.T_cur = self.T_cur + 1
#             if self.T_cur >= self.T_i:
#                 self.cycle += 1
#                 self.T_cur = self.T_cur - self.T_i
#                 self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
#         else:
#             if epoch >= self.T_0:
#                 if self.T_mult == 1:
#                     self.T_cur = epoch % self.T_0
#                     self.cycle = epoch // self.T_0
#                 else:
#                     n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
#                     self.cycle = n
#                     self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
#                     self.T_i = self.T_0 * self.T_mult ** (n)
#             else:
#                 self.T_i = self.T_0
#                 self.T_cur = epoch
#
#         self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
#         self.last_epoch = math.floor(epoch)
#         for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
#             param_group['lr'] = lr
