import pandas as pd
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
from cifar_net import Our_Net
from benchmarks import Float_Net, Quant_Net, Quant_NN_Net
from cifar_train import train, test
from utils import init_seeds
import os
import argparse
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
from plotting_percentage import comp_percentage_order, comp_percentage_snr
# from thop import profile


def denorm(x, channels=None, w=None, h=None, resize=False):
    x = 0.5 * (x + 1)
    # x = x.clamp(0, 1)
    if resize:
        if channels is None or w is None or h is None:
            print('Number of channels, width and height must be provided for resize.')
        x = x.view(x.size(0), channels, w, h)
    return x


def image_recovery(config, test_loader):
    init_seeds(3)

    # get fixed input
    it = iter(test_loader)
    sample_inputs, s = next(it)
    fixed_input = sample_inputs[0:32, :, :, :]

    device = torch.device(config.gpu1 if torch.cuda.is_available() else "cpu")

    model_path2 = '/our_net/{}/'.format(config.mod_method)
    model_name = 'CIFAR_our_net_SNR{:.3f}_Trans{:d}_{}.pth.tar'. \
        format(config.snr_train, config.trans_bit, config.mod_method)
    net = Our_Net(config, device).to(device)
    net.load_state_dict(torch.load(config.model_path + model_path2 + model_name))
    code, code_prob, z, z_hat, pred, rec = net(fixed_input.to(device), 0)
    output_image_jcm = denorm(rec.cpu().detach())

    model_path2 = '/float_net/4qam/'
    model_name = 'CIFAR_float_net_SNR{:.3f}_Trans{}_4qam.pth.tar'. \
        format(config.snr_train, config.trans_bit)
    net = Float_Net(config, device).to(device)
    net.load_state_dict(torch.load(config.model_path + model_path2 + model_name))
    z, z_hat, pred, rec = net(fixed_input.to(device))
    output_image_float = denorm(rec.cpu().detach())

    model_path2 = '/quant_net/{}_{}bits/'.format(config.mod_method, config.quant_num)
    model_name = 'CIFAR_quant_net_SNR{:.3f}_Trans{:d}_{}_{}bits.pth.tar'. \
        format(config.snr_train, config.trans_bit, config.mod_method, config.quant_num)
    net = Quant_Net(config, device).to(device)
    net.load_state_dict(torch.load(config.model_path + model_path2 + model_name))
    z, z_quant, z_hat, pred, rec = net(fixed_input.to(device))
    output_image_uniform = denorm(rec.cpu().detach())

    model_path2 = '/quant_nn/{}_{}bits/'.format(config.mod_method, config.quant_num)
    model_name = 'CIFAR_quant_nn_SNR{:.3f}_Trans{:d}_{}_{}bits_final.pth.tar'. \
        format(config.snr_train, config.trans_bit, config.mod_method, config.quant_num)
    net = Quant_NN_Net(config, device).to(device)
    net.load_state_dict(torch.load(config.model_path + model_path2 + model_name))
    z, z_hat, pred, rec = net(fixed_input.to(device))
    output_image_nn = denorm(rec.cpu().detach())

    for i in range(32):
        file_name = config.rec_path + '/raw'
        if not os.path.exists(file_name):
            os.makedirs(file_name)
        save_image(denorm(fixed_input[i, :, :, :]), file_name + '/{}.png'.format(i))
        file_name = config.rec_path + '/jcm'
        if not os.path.exists(file_name):
            os.makedirs(file_name)
        save_image(output_image_jcm[i, :, :, :], file_name + '/{}.png'.format(i))
        file_name = config.rec_path + '/float'
        if not os.path.exists(file_name):
            os.makedirs(file_name)
        save_image(output_image_float[i, :, :, :], file_name + '/{}.png'.format(i))
        file_name = config.rec_path + '/uniform'
        if not os.path.exists(file_name):
            os.makedirs(file_name)
        save_image(output_image_uniform[i, :, :, :], file_name + '/{}.png'.format(i))
        file_name = config.rec_path + '/nn'
        if not os.path.exists(file_name):
            os.makedirs(file_name)
        save_image(output_image_nn[i, :, :, :], file_name + '/{}.png'.format(i))

