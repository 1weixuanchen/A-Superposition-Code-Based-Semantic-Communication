import torch
import torchvision
from torchvision import transforms as transforms
from torch.utils.data import DataLoader as DataLoader
from dataset import Cifar100
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as comp_psnr
from skimage.metrics import structural_similarity as comp_ssim
from skimage.metrics import mean_squared_error as comp_mse
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
import random


def data_loader(data_root, batch_size, transform_train, transform_test):
    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)

    cifar100_training = torchvision.datasets.CIFAR100(
        root=data_root,
        train=True,
        transform=None,
        download=True,
    )
    train_set = Cifar100(cifar100_training, transform_train)
    train_loader = DataLoader(train_set, batch_size, True)

    cifar100_testing = torchvision.datasets.CIFAR100(
        root=data_root,
        train=False,
        transform=None,
        download=True,
    )
    test_set = Cifar100(cifar100_testing, transform_test)
    test_loader = DataLoader(test_set, batch_size, False)

    return train_loader, test_loader


def init_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if seed == 42:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def denorm(x, channels=None, w=None, h=None, resize=False):
    x = (x * 0.5 + 0.5).clamp(0, 1)
    if resize:
        if channels is None or w is None or h is None:
            print('Number of channels, width and height must be provided for resize.')
        x = x.view(x.size(0), channels, w, h)
    return x

def PSNR(tensor_org, tensor_trans):
    total_psnr = 0
    tensor_org = (tensor_org + 1) / 2
    tensor_trans = (tensor_trans + 1) / 2
    origin = tensor_org.cpu().numpy()
    trans = tensor_trans.cpu().numpy()
    for i in range(np.size(trans, 0)):
        psnr = 0
        for j in range(np.size(trans, 1)):
            psnr_temp = comp_psnr(origin[i, j, :, :], trans[i, j, :, :])
            psnr = psnr + psnr_temp
        psnr /= 3
        total_psnr += psnr

    return total_psnr


def SSIM(tensor_org, tensor_trans):
    total_ssim = 0
    tensor_org = (tensor_org + 1) / 2
    tensor_trans = (tensor_trans + 1) / 2
    origin = tensor_org.cpu().numpy()
    trans = tensor_trans.cpu().numpy()
    for i in range(np.size(trans, 0)):
        ssim = 0
        for j in range(np.size(trans, 1)):
            ssim_temp = comp_ssim(origin[i, j, :, :], trans[i, j, :, :])
            ssim = ssim + ssim_temp
        ssim /= 3
        total_ssim += ssim

    return total_ssim


def MSE(tensor_org, tensor_trans):
    total_mse = 0
    tensor_org = (tensor_org + 1) / 2
    tensor_trans = (tensor_trans + 1) / 2
    origin = tensor_org.cpu().numpy()
    trans = tensor_trans.cpu().numpy()
    for i in range(np.size(trans, 0)):
        mse = 0
        for j in range(np.size(trans, 1)):
            mse_temp = comp_mse(origin[i, j, :, :], trans[i, j, :, :])
            mse = mse + mse_temp
        mse /= 3
        total_mse += mse

    return total_mse


# def count_percentage(code, config, device):
#     result = torch.zeros(16)
#     code = code.reshape(-1, 2)
#     map = torch.tensor([[1, 1], [1, -1], [-1, 1], [-1, -1],
#                         [1, 1 / 3], [1, -1 / 3], [-1, 1 / 3], [-1, -1 / 3],
#                         [1 / 3, 1], [-1 / 3, 1], [1 / 3, -1], [-1 / 3, -1],
#                         [1 / 3, 1 / 3], [1 / 3, -1 / 3], [-1 / 3, 1 / 3], [-1 / 3, -1 / 3]]).to(device)
#     for j in range(16):
#         ref = torch.repeat_interleave(map[j, :].unsqueeze(0), code.shape[0], dim=0)
#         temp = torch.abs(ref - code)
#         temp = torch.sum(temp, dim=1)
#         num = (temp == torch.tensor([0]).to(device)).sum()
#         per = num / code.shape[0]
#         result[j] = per
#
#     return result


def save_constellation(z, z_hat, config):
    fig1 = plt.figure()
    z = z.view(-1, 2).detach().cpu()[0:500, :]
    z_hat = z_hat.view(-1, 2).detach().cpu()[0:500, :]
    plt.scatter(z_hat[:, 0], z_hat[:, 1], s=4, color='b')
    plt.scatter(z[:, 0], z[:, 1], s=16, color='r')

    if config.train_phase == '1':
        fig_name = '/constellation_phase1.png'
    elif config.train_phase == '2':
        fig_name = '/constellation_phase2.png'
    plt.savefig(config.data_path + fig_name)
    plt.close()


def calculate_entropy(probs):
    probs = probs.reshape(probs.shape[0], -1)
    log_prob = torch.log(probs)
    entropy = torch.sum(- probs * log_prob, dim=1) / torch.log(torch.tensor(2))
    mean_entropy = torch.mean(entropy)
    return mean_entropy


def plot_training(data_dic, config):
    epochs = data_dic['epoch']
    train_acc = data_dic['train_acc']
    test_acc = data_dic['acc']
    loss = data_dic['loss']

    fig1 = plt.figure()
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.grid()
    plt.plot(epochs, train_acc, color='b', label='train')
    plt.plot(epochs, test_acc, color='r', label='test')
    plt.legend()
    plt.title('Training epochs vs. accuracy.')
    fig_name = '/acc_{}_SNR{}_Trans{}_lr{}_iters{}.png'.format(config.net, config.snr_train_bad, config.channel_use,
                                                               config.lr, config.train_iters)
    plt.savefig(config.data_path + fig_name)
    plt.close()

    fig3 = plt.figure()
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.grid()
    plt.plot(epochs, loss, color='r')
    plt.title('Training epochs vs. loss.')
    fig_name = '/loss_{}_SNR{}_Trans{}_lr{}_iters{}.png'.format(config.net, config.snr_train_bad, config.channel_use,
                                                                config.lr, config.train_iters)
    plt.savefig(config.data_path + fig_name)
    plt.close()


def plot_training_2(data_dic_bad, data_dic_good, resi_track, cons_per, config):
    epochs = data_dic_bad['epoch']
    acc_1 = data_dic_bad['acc']
    loss = data_dic_bad['loss']
    acc_2_coarse = data_dic_good['acc_coarse']
    acc_2_fine = data_dic_good['acc_fine']
    acc_2_fine_train = data_dic_good['acc_fine_train']
    corner_per = cons_per['corner']
    edge_per = cons_per['edge']
    inner_per = cons_per['inner']
    resi = resi_track['residual']

    fig1 = plt.figure()
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.grid()
    plt.plot(epochs, acc_1, color='black', label='receiver 1')
    plt.plot(epochs, acc_2_coarse, color='blue', label='receiver 2: coarse')
    plt.plot(epochs, acc_2_fine, color='green', label='receiver 2: fine')
    plt.plot(epochs, acc_2_fine_train, color='red', label='receiver 2: fine (training)')
    plt.legend()
    plt.title('Training epochs vs. accuracy.')
    if config.train_mode == 'phase_2':
        fig_name = '/phase2_acc_SNR{}_{}_Trans{}_lr{}_iters{}.png'.format(config.snr_train_good, config.snr_train_bad,
                                                                          config.channel_use, config.lr,
                                                                          config.train_iters)
    elif config.train_mode == 'phase_3':
        fig_name = '/phase3_acc_SNR{}_{}_Trans{}_lr{}_iters{}.png'.format(config.snr_train_good, config.snr_train_bad,
                                                                          config.channel_use, config.lr,
                                                                          config.train_iters)
    plt.savefig(config.data_path + fig_name)
    plt.close()

    fig3 = plt.figure()
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.grid()
    plt.plot(epochs, loss, color='r')
    plt.title('Training epochs vs. loss.')
    if config.train_mode == 'phase_2':
        fig_name = '/phase2_loss_SNR{}_{}_Trans{}_lr{}_iters{}.png'.format(config.snr_train_good, config.snr_train_bad,
                                                                           config.channel_use, config.lr,
                                                                           config.train_iters)
    elif config.train_mode == 'phase_3':
        fig_name = '/phase3_loss_SNR{}_{}_Trans{}_lr{}_iters{}.png'.format(config.snr_train_good, config.snr_train_bad,
                                                                           config.channel_use, config.lr,
                                                                           config.train_iters)
    plt.savefig(config.data_path + fig_name)
    plt.close()

    fig4 = plt.figure()
    plt.ylabel('percentage')
    plt.xlabel('Epochs')
    plt.grid()
    plt.plot(epochs, corner_per, color='r', label='corner point')
    plt.plot(epochs, edge_per, color='b', label='edge point')
    plt.plot(epochs, inner_per, color='g', label='inner point')
    plt.legend()
    plt.title('Training epochs vs. constellation percentage.')
    if config.train_mode == 'phase_2':
        fig_name = '/phase2_constellation_SNR{}_{}_Trans{}_lr{}_iters{}.png'.format(config.snr_train_good,
                                                                                    config.snr_train_bad,
                                                                                    config.channel_use, config.lr,
                                                                                    config.train_iters)
    elif config.train_mode == 'phase_3':
        fig_name = '/phase3_constellation_SNR{}_{}_Trans{}_lr{}_iters{}.png'.format(config.snr_train_good,
                                                                                    config.snr_train_bad,
                                                                                    config.channel_use, config.lr,
                                                                                    config.train_iters)
    plt.savefig(config.data_path + fig_name)
    plt.close()

    fig5 = plt.figure()
    plt.ylabel('Residual')
    plt.xlabel('Epochs')
    plt.grid()
    plt.plot(epochs, resi, color='r')
    plt.title('Training epochs vs. residual.')
    if config.train_mode == 'phase_2':
        fig_name = '/phase2_residual_SNR{}_{}_Trans{}_lr{}_iters{}.png'.format(config.snr_train_good,
                                                                               config.snr_train_bad, config.channel_use,
                                                                               config.lr, config.train_iters)
    elif config.train_mode == 'phase_3':
        fig_name = '/phase3_residual_SNR{}_{}_Trans{}_lr{}_iters{}.png'.format(config.snr_train_good,
                                                                               config.snr_train_bad, config.channel_use,
                                                                               config.lr, config.train_iters)
    plt.savefig(config.data_path + fig_name)
    plt.close()


def count_percentage(code, mod, epoch, snr, channel_use, tradeoff_h, name):
    code = code.reshape(-1)
    index = [i for i in range(len(code))]
    random.shuffle(index)
    code = code[index]
    code = code.reshape(-1, 2).cpu()

    I_point = torch.unique(code.reshape(-1))

    if mod == 16:
        # I_point = torch.tensor([-3, -1, 1, 3])
        order = 16
    elif mod == 64:
        # I_point = torch.tensor([-7, -5, -3, -1, 1, 3, 5, 7])
        order = 64
        # elif mod == '256qam':
        #     I_point = torch.tensor([-15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15])
        #     order = 256
        # elif mod == '1024qam':
        #     I_point = torch.tensor([-31, -29, -27, -25, -23, -21, -19, -17, -15, -13, -11, -9, -7, -5, -3, -1,
        #          1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31])
        #     order = 1024

    I, Q = torch.meshgrid(I_point, I_point)
    map = torch.cat((I.unsqueeze(-1), Q.unsqueeze(-1)), dim=2).reshape(order, 2)
    per_s = []
    fig = plt.figure(dpi=300)
    ax = Axes3D(fig)
    fig.add_axes(ax)
    for i in range(order):
        temp = torch.sum(torch.abs(code - map[i, :]), dim=1)
        num = code.shape[0] - torch.count_nonzero(temp).item()
        per = num / code.shape[0]
        per_s.append(per)
        # plt.plot(map[i, 0], map[i, 1], marker="o", markersize=per * 100, color='b')
    # plt.show()
    per_s = torch.tensor(per_s).cpu()
    height = np.zeros_like(per_s)
    width = depth = 0.05
    surf = ax.bar3d(I.ravel(), Q.ravel(), height, width, depth, per_s, zsort='average', color='palegreen', alpha=0.6)
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d
    file_name = './cons_fig/' + '{}_{}_{}_{}_{}'.format(name, mod, snr, channel_use, tradeoff_h)
    if not os.path.exists(file_name):
        os.makedirs(file_name)
    fig.savefig(file_name + '/{}'.format(epoch))
    # plt.show()
    plt.close()

    fig = plt.figure(dpi=300)
    for k in range(order):
        plt.scatter(map[k, 0], map[k, 1], s=1000 * per_s[k], color='palegreen')
    fig.savefig(file_name + '/scatter_{}'.format(epoch))

    plt.close()

def compute_entropy(code, config):
    code = code.reshape(1, -1)
    if config.order == 16:
        order = 4
    elif config.order == 64:
        order = 8

    I = torch.unique(code.reshape(-1))
    h = 0
    for i in range(order):
        mask = torch.where(code == I[i], 1, 0)
        per = torch.sum(mask * code) / (I[i] * code.shape[-1]) + 1e-9
        h = h - per * torch.log2(per)
        # print(per)
    return h


def count_percentage_super(code, mod, epoch, snr, channel_use, tradeoff_h, name, phase, a, tradeoff):
    code = code.reshape(-1)
    index = [i for i in range(len(code))]
    random.shuffle(index)
    code = code[index]
    code = code.reshape(-1, 2).cpu()

    I_point = torch.unique(code.reshape(-1))

    if mod == '4and4':
        order = 16
    elif mod == '4and16':
        order = 64
    # if mod == 16:
    #     I_point = torch.tensor([-3, -1, 1, 3])
    #     order = 16
    # elif mod == 64:
    #     I_point = torch.tensor([-7, -5, -3, -1, 1, 3, 5, 7])
    #     order = 64
    # elif mod == '256qam':
    #     I_point = torch.tensor([-15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15])
    #     order = 256
    # elif mod == '1024qam':
    #     I_point = torch.tensor([-31, -29, -27, -25, -23, -21, -19, -17, -15, -13, -11, -9, -7, -5, -3, -1,
    #          1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31])
    #     order = 1024

    I, Q = torch.meshgrid(I_point, I_point)
    map = torch.cat((I.unsqueeze(-1), Q.unsqueeze(-1)), dim=2).reshape(order, 2)
    per_s = []
    fig = plt.figure(dpi=300)
    ax = Axes3D(fig)
    fig.add_axes(ax)
    for i in range(order):
        temp = torch.sum(torch.abs(code - map[i, :]), dim=1)
        num = code.shape[0] - torch.count_nonzero(temp).item()
        per = num / code.shape[0]
        per_s.append(per)
        # plt.plot(map[i, 0], map[i, 1], marker="o", markersize=per * 100, color='b')
    # plt.show()
    per_s = torch.tensor(per_s).cpu()
    height = np.zeros_like(per_s)
    width = depth = 0.05
    surf = ax.bar3d(I.ravel(), Q.ravel(), height, width, depth, per_s, zsort='average', alpha=0.6, color='lightcoral')
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d
    if phase == '3':
        file_name = './cons_fig/' + '{}_{}_{}_{}_{}_phase{}_{}_{}'.format(name, mod, snr, channel_use, tradeoff_h,
                                                                          phase, a, tradeoff)
    else:
        file_name = './cons_fig/' + '{}_{}_{}_{}_{}_phase{}_{}'.format(name, mod, snr, channel_use, tradeoff_h, phase,
                                                                       a)
    if not os.path.exists(file_name):
        os.makedirs(file_name)
    fig.savefig(file_name + '/{}'.format(epoch))
    # plt.show()
    plt.close()
    # h_single = torch.mean(torch.sum(-code_prob * torch.log(code_prob + 1e-9), dim=-1))
    # print(h_single)

    fig = plt.figure(dpi=300)
    for k in range(order):
        plt.scatter(map[k, 0], map[k, 1], s=1000 * per_s[k], color='lightcoral')
    fig.savefig(file_name + '/scatter_{}'.format(epoch))
    # plt.show()
    plt.close()

