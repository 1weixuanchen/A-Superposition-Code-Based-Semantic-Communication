from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from cifar_net import Our_Net
from benchmarks import Float_Net
import random
from scipy import stats
from matplotlib import cm


def comp_percentage_snr(data_loader, device, config):

    if config.mod_method == '16qam':
        I_point = torch.tensor([-3, -1, 1, 3])
        order = 16
    elif config.mod_method == '64qam':
        I_point = torch.tensor([-7, -5, -3, -1, 1, 3, 5, 7])
        order = 64
    I, Q = torch.meshgrid(I_point, I_point)
    map = torch.cat((I.unsqueeze(-1), Q.unsqueeze(-1)), dim=2).reshape(order, 2)

    snrs = [18, 0, -12]
    percentages = torch.zeros((len(snrs), order))
    I_point_ave = torch.zeros((len(snrs), int(order ** 0.5)))

    for j in range(len(snrs)):
        config.snr_train = snrs[j]
        config.snr_test = snrs[j]
        net = Our_Net(config, device).to(device)
        model_path2 = '/{}/{}/'.format(config.mod, config.mod_method)
        model_name = 'CIFAR_{}_SNR{:.3f}_Trans{:d}_{}.pth.tar'.\
            format(config.mod, config.snr_train, config.trans_bit, config.mod_method)

        file_name = config.model_path + model_path2
        net.load_state_dict(torch.load(file_name + model_name, map_location='cpu'), strict=False)

        z_total = torch.zeros((20 * config.batch_size, config.trans_bit * 2))
        for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            code, code_prob, z, z_hat, pred, rec = net(data, 0)
            if batch_idx <= 19:
                z_total[batch_idx * config.batch_size:(batch_idx + 1) * config.batch_size, :] = code

        ave_power = torch.mean(z_total ** 2)  # the average power of I channel or Q channel
        I_point_ave[j, :] = (0.5 / ave_power) ** 0.5 * I_point
        z_total = z_total.reshape(-1)
        index = [i for i in range(len(z_total))]
        random.shuffle(index)
        z_total = z_total[index]
        z_total = z_total.reshape(-1, 2).cpu()

        for i in range(order):
            temp = torch.sum(torch.abs(z_total - map[i, :]), dim=1)
            num = z_total.shape[0] - torch.count_nonzero(temp).item()
            per = num / z_total.shape[0]
            percentages[j, i] = per
        # pers = torch.tensor(percentages)
        # h = torch.sum(1 / pers * torch.log2(1 / pers), dim=1)
        # print(h)

        # code = z_total.reshape(1, -1)
        # h = 0
        # for i in range(int(order ** 0.5)):
        #     mask = torch.where(code == I_point[i], 1, 0)
        #     per = torch.sum(mask * code) / (I_point[i] * code.shape[-1]) + 1e-9
        #     h = h - per * torch.log2(per)
        # print(h)

    pers = torch.tensor(percentages) + 1e-9
    h = torch.sum(pers * torch.log2(1 / pers), dim=1)
    print(h)


    color_list = ['pink', 'cornflowerblue', 'lightgreen', 'orange', 'purple', 'red', 'black']

    # fig = plt.figure(dpi=300)
    # ax = Axes3D(fig)
    fontdict = {'fontsize': 12,
                'fontfamily': 'Times New Roman'}
    legenddict = {'size': 10,
                  'family': 'Times New Roman'}
    delta = [-1, 0, 1]
    labels = ['SNR = 18 dB', 'SNR = 0 dB', 'SNR = -12 dB']

    for i in range(len(snrs)):
        fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=100)

        ax.set_xlabel('I', fontdict)
        ax.set_ylabel('Q', fontdict)
        # ax.set_zlabel('Density', fontdict)

        names_x = [-3, -2, -1, 0,  1, 2, 3]
        names_y = names_x
        # names_z = [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
        ax.set_xticks(names_x)
        ax.set_yticks(names_y)
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        # ax.set_zticks(names_z)

        xpos, ypos = torch.meshgrid(I_point_ave[i, :], I_point_ave[i, :])
        map = torch.cat((xpos.unsqueeze(-1), ypos.unsqueeze(-1)), dim=2).reshape(order, 2)

        for k in range(order):
            plt.scatter(map[k, 0].item(), map[k, 1].item(), s=1000 * percentages[i, k].item(), color='purple')

        # fig = plt.figure(dpi=300)
        # ax = Axes3D(fig)
        # fig.add_axes(ax)
        # ax.set_xlabel('I', fontdict)
        # ax.set_ylabel('Q', fontdict)
        # ax.set_zlabel('Density', fontdict)
        #
        # names_x = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]
        # names_y = names_x
        # names_z = [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
        # ax.set_xticks(names_x)
        # ax.set_yticks(names_y)
        # ax.set_zticks(names_z)
        #
        # xpos, ypos = np.meshgrid(np.array(I_point_ave[i, :].detach()), np.array(I_point_ave[i, :].detach()), indexing="ij")
        # xpos = xpos.ravel()
        # ypos = ypos.ravel()
        # data_1 = percentages[0, :]
        # height = np.zeros_like(data_1)
        # width = depth = 0.05
        # surf = ax.bar3d(xpos, ypos, height, width, depth, percentages[i, :], color=color_list[i], zsort='average', label=labels[i],
        #                 alpha=0.8)
        # surf._facecolors2d = surf._facecolor3d
        # surf._edgecolors2d = surf._edgecolor3d

        # names = ['-1', r"$-\frac{1}{3}$", r"$\frac{1}{3}$", '1']
        # plt.xticks([-1.0, -1 / 3, 1 / 3, 1.0], names)
        # plt.yticks([-1.0, -1 / 3, 1 / 3, 1.0], names)
        # ax.legend(loc='best', prop=legenddict)

        # plt.tick_params(labelsize=12)
        # x1_label = ax.get_xticklabels()
        # [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
        # y1_label = ax.get_yticklabels()
        # [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]
        # z1_label = ax.get_zticklabels()
        # [z1_label_temp.set_fontname('Times New Roman') for z1_label_temp in z1_label]
        # ax.legend()
        # plt.tight_layout()
        # plt.show()
        plt.tick_params(labelsize=12)
        x1_label = ax.get_xticklabels()
        [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
        y1_label = ax.get_yticklabels()
        [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

        plt.grid()
        # ax.legend()
        plt.tight_layout()
        plt.show()


def comp_percentage_order(data_loader, device, config):
    # qam = ['4qam', '16qam', '64qam']
    # snr = 12
    #
    # color_list = ['pink', 'cornflowerblue', 'lightgreen']
    #
    # fig = plt.figure(dpi=300)
    # ax = Axes3D(fig)
    # fig.add_axes(ax)
    #
    # fontdict = {'fontsize': 12,
    #             'fontfamily': 'Times New Roman'}
    # legenddict = {'size': 10,
    #               'family': 'Times New Roman'}
    # delta = [-1, 0, 1]
    # labels = ['4QAM', '16QAM', '64QAM']
    #
    # ax.set_xlabel('I', fontdict)
    # ax.set_ylabel('Q', fontdict)
    # ax.set_zlabel('Density', fontdict)
    #
    # names_x = [-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5]
    # names_y = names_x
    # names_z = [0, 0.04, 0.08, 0.12, 0.16, 0.2, 0.24]
    # ax.set_xticks(names_x)
    # ax.set_yticks(names_y)
    # ax.set_zticks(names_z)
    #
    # for j in range(len(qam)):
    #
    #     config.mod_method = qam[j]
    #     if config.mod_method == '4qam':
    #         I_point = torch.tensor([-1, 1])
    #         order = 4
    #     elif config.mod_method == '16qam':
    #         I_point = torch.tensor([-3, -1, 1, 3])
    #         order = 16
    #     elif config.mod_method == '64qam':
    #         I_point = torch.tensor([-7, -5, -3, -1, 1, 3, 5, 7])
    #         order = 64
    #     I, Q = torch.meshgrid(I_point, I_point)
    #     map = torch.cat((I.unsqueeze(-1), Q.unsqueeze(-1)), dim=2).reshape(order, 2)
    #
    #     config.snr_train = snr
    #     config.snr_test = snr
    #     net = Our_Net(config, device).to(device)
    #     model_path2 = '/{}/{}/'.format(config.mod, config.mod_method)
    #     model_name = 'CIFAR_{}_SNR{:.3f}_Trans{:d}_{}.pth.tar'.\
    #         format(config.mod, config.snr_train, config.trans_bit, config.mod_method)
    #
    #     file_name = config.model_path + model_path2
    #     net.load_state_dict(torch.load(file_name + model_name), strict=False)
    #
    #     z_total = torch.zeros((50 * config.batch_size, config.trans_bit * 2))
    #     for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
    #         data, target = data.to(device), target.to(device)
    #         code, code_prob, z, z_hat, pred, rec = net(data, 0)
    #         if batch_idx <= 49:
    #             z_total[batch_idx * config.batch_size:(batch_idx + 1) * config.batch_size, :] = code
    #
    #     ave_power = torch.mean(z_total ** 2)  # the average power of I channel or Q channel
    #     I_point_ave = (0.5 / ave_power) ** 0.5 * I_point
    #     z_total = z_total.reshape(-1)
    #     index = [i for i in range(len(z_total))]
    #     random.shuffle(index)
    #     z_total = z_total[index]
    #     z_total = z_total.reshape(-1, 2).cpu()
    #
    #     percentages = torch.zeros(order)
    #     for i in range(order):
    #         temp = torch.sum(torch.abs(z_total - map[i, :]), dim=1)
    #         num = z_total.shape[0] - torch.count_nonzero(temp).item()
    #         per = num / z_total.shape[0]
    #         percentages[i] = per
    #
    #     xpos, ypos = np.meshgrid(np.array(I_point_ave.detach()), np.array(I_point_ave.detach()), indexing="ij")
    #     xpos = xpos.ravel()
    #     ypos = ypos.ravel()
    #     data_1 = percentages
    #     height = np.zeros_like(data_1)
    #     width = depth = 0.08
    #     surf = ax.bar3d(xpos, ypos, height, width, depth, percentages, color=color_list[j], zsort='average',
    #                     label=labels[j], alpha=0.8)
    #     surf._facecolors2d = surf._facecolor3d
    #     surf._edgecolors2d = surf._edgecolor3d
    #
    #     # names = ['-1', r"$-\frac{1}{3}$", r"$\frac{1}{3}$", '1']
    #     # plt.xticks([-1.0, -1 / 3, 1 / 3, 1.0], names)
    #     # plt.yticks([-1.0, -1 / 3, 1 / 3, 1.0], names)
    #     # ax.legend(loc='best', prop=legenddict)
    #
    #     plt.tick_params(labelsize=12)
    #     x1_label = ax.get_xticklabels()
    #     [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
    #     y1_label = ax.get_yticklabels()
    #     [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]
    #     z1_label = ax.get_zticklabels()
    #     [z1_label_temp.set_fontname('Times New Roman') for z1_label_temp in z1_label]
    #     ax.legend()

    snrs = [-18, 0, 12]
    for i in range(len(snrs)):
        fig = plt.figure(dpi=300)
        ax = Axes3D(fig)
        fig.add_axes(ax)

        config.snr_train = snrs[i]
        net = Float_Net(config, device).to(device)
        model_path2 = '/float_net/4qam/'
        model_name = 'CIFAR_float_net_SNR{:.3f}_Trans{:d}_4qam.pth.tar'. \
            format(config.snr_train, config.trans_bit)

        file_name = config.model_path + model_path2
        net.load_state_dict(torch.load(file_name + model_name), strict=False)

        z_total = torch.zeros((30 * config.batch_size, config.trans_bit * 2))
        for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            z, z_hat, pred, rec = net(data)
            if batch_idx <= 29:
                z_total[batch_idx * config.batch_size:(batch_idx + 1) * config.batch_size, :] = z

        z_total = z_total * 0.5 ** 0.5
        z_total = z_total.reshape(2, -1).detach()
        kernel = stats.gaussian_kde(z_total)

        X, Y = np.mgrid[-2:2:100j, -2:2:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(kernel(positions).T, X.shape)
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.6)
        fig.colorbar(surf, shrink=0.5)

        plt.tight_layout()
        plt.show()

