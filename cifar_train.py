import os
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import optim, nn
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as comp_psnr
from skimage.metrics import structural_similarity as comp_ssim
from skimage.metrics import mean_squared_error as comp_mse
import pandas as pd
from tqdm import tqdm
from torchvision.utils import save_image
# from utils import plot_code, plot_quantized_code, count_percentage, compute_entropy
# from torch.utils.tensorboard import SummaryWriter
import lpips

def denorm(x, channels=None, w=None, h=None, resize=False):
    # x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    if resize:
        if channels is None or w is None or h is None:
            print('Number of channels, width and height must be provided for resize.')
        x = x.view(x.size(0), channels, w, h)
    return x


def PSNR(tensor_org, tensor_trans):
    total_psnr = 0
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
    origin = tensor_org.cpu().numpy()
    trans = tensor_trans.cpu().numpy()
    for i in range(np.size(trans, 0)):
        ssim = 0
        for j in range(np.size(trans, 1)):
            ssim_temp = comp_ssim(origin[i, j, :, :], trans[i, j, :, :], data_range=1.0)
            ssim = ssim + ssim_temp
        ssim /= 3
        total_ssim += ssim

    return total_ssim


def MSE(tensor_org, tensor_trans):
    total_mse = 0
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


def EVAL_lpips(model, data_loader, device, config, step=0, epoch=0):
    model.eval()
    acc_total = 0
    mse_total = 0
    psnr_total = 0
    psnr_bad_total = 0
    ssim_total = 0
    loss_total = 0
    lpips_total = 0.0
    lpips_bad_total = 0.0
    total = 0

    loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    z_total = torch.zeros((10000, config.trans_bit * 2), device=device)

    with torch.no_grad():
        sample_data, _ = next(iter(data_loader))
        sample_data = sample_data.to(device)
        min_val = sample_data.min().item()
        max_val = sample_data.max().item()
        auto_normalize = not (0.0 <= min_val and max_val <= 1.0)
        if auto_normalize:
            print(f"[EVAL] Auto detected data range [{min_val:.2f}, {max_val:.2f}], applying normalization")

        for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            batch_size = len(target)
            total += batch_size

            if config.mod == 'quant_nn':
                z, z_hat, pred, rec = model(data, Test)
                rec_bad = None
            elif config.mod == 'quant_net':
                z, z_quant, z_hat, pred, rec = model(data)
                rec_bad = None
            elif config.mod == 'our_net':
                code, code_prob, z, z_hat, pred, rec, rec_bad = model(data, epoch)
            else:
                z, z_hat, pred, rec = model(data)
                rec_bad = None

            if auto_normalize:
                data_normal = (data - data.min()) / (data.max() - data.min() + 1e-8)
                rec_normal = (rec - rec.min()) / (rec.max() - rec.min() + 1e-8)
                if rec_bad is not None:
                    rec_bad_normal = (rec_bad - rec_bad.min()) / (rec_bad.max() - rec_bad.min() + 1e-8)
            else:
                data_normal = data
                rec_normal = rec
                if rec_bad is not None:
                    rec_bad_normal = rec_bad

            lpips_val = loss_fn_alex(data_normal, rec_normal).mean()
            lpips_total += lpips_val.item() * batch_size

            if rec_bad is not None:
                lpips_bad_val = loss_fn_alex(data_normal, rec_bad_normal).mean()
                lpips_bad_total += lpips_bad_val.item() * batch_size

            acc = (pred.data.max(1)[1] == target.data).float().sum()
            mse = MSE(data, rec)
            psnr = PSNR(data, rec)
            psnr_bad = PSNR(data, rec_bad) if rec_bad is not None else 0
            ssim = SSIM(data, rec)
            loss = nn.MSELoss()(z, z_hat) * batch_size

            acc_total += acc.item()
            mse_total += mse.item()
            psnr_total += psnr.item()
            psnr_bad_total += psnr_bad if rec_bad is not None else 0
            ssim_total += ssim.item()
            loss_total += loss.item()

    avg_lpips = lpips_total / total
    avg_lpips_bad = lpips_bad_total / total if lpips_bad_total != 0 else 0
    avg_acc = acc_total / total
    avg_mse = mse_total / total
    avg_psnr = psnr_total / total
    avg_psnr_bad = psnr_bad_total / total if psnr_bad_total != 0 else 0
    avg_ssim = ssim_total / total
    avg_loss = loss_total / total

    print(f"\n[EVAL] LPIPS: {avg_lpips:.4f} | "
          f"LPIPS_bad: {avg_lpips_bad:.4f} | "  
          f"PSNR: {avg_psnr:.2f}dB | PSNR_bad: {avg_psnr_bad:.2f}dB | "
          f"SSIM: {avg_ssim:.4f} | MSE: {avg_mse:.4f} | Acc: {avg_acc:.2%}")

    return acc_total, mse_total, psnr_total, ssim_total, loss_total , psnr_bad_total

def EVAL(model, data_loader, device, config, step=0, epoch=0):
    model.eval()
    acc_total = 0
    mse_total = 0
    psnr_total = 0
    psnr_bad_total = 0
    ssim_total = 0
    loss_total = 0
    total = 0
    Test = 1

    z_total = torch.zeros((10000, config.trans_bit * 2))
    for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
        data, target = data.to(device), target.to(device)
        total += len(target)

        with torch.no_grad():
            if config.mod == 'quant_nn':
                z, z_hat, pred, rec = model(data, Test)
            elif config.mod == 'quant_net':
                z, z_quant, z_hat, pred, rec = model(data)
                # if batch_idx == 0:
                    # plot_quantized_code(z, z_quant, z_hat)
            elif config.mod == 'our_net':
                code, code_prob, z, z_hat, pred, rec,rec_bad = model(data, epoch)
                if config.mode == 'test' and (config.mod_method == '16qam' or config.mod_method == '64qam'):
                    if batch_idx <= int(10000/config.batch_size)-1:
                        z_total[batch_idx * config.batch_size:(batch_idx + 1) * config.batch_size, :] = code
                    elif batch_idx == int(10000/config.batch_size):
                        z_total[int(10000/config.batch_size) * config.batch_size:, :] = code
                        # count_percentage(z_total, code_prob, config.mod_method, -1, config.snr_train, config.trans_bit, config.tradeoff_h)
                # else:
                #     if batch_idx == 0:
                        # count_percentage(code, code_prob, config.mod_method, epoch, config.snr_train, config.trans_bit, config.tradeoff_h)
            else:
                z, z_hat, pred, rec = model(data)

        acc = (pred.data.max(1)[1] == target.data).float().sum()
        mse = MSE(data, rec)
        psnr = PSNR(data, rec)
        psnr_bad = PSNR(data, rec_bad)
        ssim = SSIM(data, rec)
        loss = nn.MSELoss()(z, z_hat) * z.shape[0]
        acc_total += acc
        mse_total += mse
        psnr_total += psnr
        psnr_bad_total += psnr_bad
        ssim_total += ssim
        loss_total += loss
    acc_total /= total
    mse_total /= total
    psnr_total /= total
    psnr_bad_total /= total
    ssim_total /= total
    loss_total /= total

    return acc_total, mse_total, psnr_total, ssim_total, loss_total , psnr_bad_total


def test(config, net, test_iter, device):
    if config.mod == 'quant_net':
        # if config.quant_num == 1 and config.mod_method == 'bpsk':
        #     model_path2 = '/float_net/bpsk/'
        #     model_name = 'CIFAR_float_net_SNR{:.3f}_Trans{:d}_bpsk.pth.tar'. \
        #         format(config.snr_train, config.trans_bit)
        # elif config.quant_num == 2 and config.mod_method == '4qam':
        #     if config.snr_train == 18 or config.snr_train == 12:
        #         model_path2 = '/float_net/4qam/'
        #         model_name = 'CIFAR_float_net_SNR6.000_Trans{:d}_4qam.pth.tar'. \
        #             format(config.trans_bit)
        #     else:
        #         model_path2 = '/float_net/4qam/'
        #         model_name = 'CIFAR_float_net_SNR{:.3f}_Trans{:d}_4qam.pth.tar'. \
        #             format(config.snr_train, config.trans_bit)
        # elif config.quant_num == 4 and config.mod_method == '16qam':
        #     if config.snr_train == 18 or config.snr_train == 12:
        #         model_path2 = '/float_net/4qam/'
        #         model_name = 'CIFAR_float_net_SNR6.000_Trans{:d}_4qam.pth.tar'. \
        #             format(config.trans_bit)
        #     else:
        #         model_path2 = '/float_net/4qam/'
        #         model_name = 'CIFAR_float_net_SNR{:.3f}_Trans{:d}_4qam.pth.tar'. \
        #             format(config.snr_train, config.trans_bit)
        # else:
        #     model_path2 = '/{}/{}_{}bits/'.format(config.mod, config.mod_method, config.quant_num)
        #     model_name = 'CIFAR_{}_SNR{:.3f}_Trans{:d}_{}_{}bits.pth.tar'.\
        #         format(config.mod, config.snr_train, config.trans_bit, config.mod_method, config.quant_num)
        model_path2 = '/{}/{}_{}bits/'.format(config.mod, config.mod_method, config.quant_num)
        model_name = 'CIFAR_{}_SNR{:.3f}_Trans{:d}_{}_{}bits.pth.tar'. \
            format(config.mod, config.snr_train, config.trans_bit, config.mod_method, config.quant_num)
    elif config.mod == 'quant_nn':
        model_path2 = '/{}/{}_{}bits/'.format(config.mod, config.mod_method, config.quant_num)
        model_name = 'CIFAR_{}_SNR{:.3f}_Trans{:d}_{}_{}bits_final.pth.tar'. \
            format(config.mod, config.snr_train, config.trans_bit, config.mod_method, config.quant_num)
    else:
        model_path2 = '/{}/{}/'.format(config.mod, config.mod_method)
        model_name = 'CIFAR_{}_a{:.3f}_SNR{:.3f}_BADSNR{:.3f}_Trans{:d}_{}_image_recovery.pth.tar'.\
            format(config.mod, config.a, config.snr_train, config.snr_train_bad, config.trans_bit, config.mod_method)

    file_name = config.model_path + model_path2
    net.load_state_dict(torch.load(file_name + model_name), strict=False)

    acc, mse, psnr, ssim, float_mse , psnr_bad = EVAL(net, test_iter, device, config)

    log = 'CIFAR_{}_a{:.3f}_SNRtrain{:.3f}_BADSNRtrain{:.3f}_SNRTest{:.3f}_Trans{:d}_{}_acc_{:.3f}'.\
        format(config.mod, config.a, config.snr_train, config.snr_train_bad, config.snr_test, config.trans_bit, config.mod_method, acc)
    print(log)
    return acc, mse, psnr, ssim, float_mse ,psnr_bad

def train(config, net, train_iter, test_iter, device, fixed_input):

    learning_rate = config.lr
    epochs = config.train_iters

    if config.mod == 'our_net':
        ignored_params = list(map(id, net.prob_convs.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
        optimizer = optim.Adam([
            {'params': base_params},
            {'params': net.prob_convs.parameters(), 'lr': learning_rate/2}], learning_rate)
    loss_f2 = nn.MSELoss()
    results = {'epoch': [], 'acc': [], 'mse': [], 'psnr': [], 'psnr_bad': [],'ssim': [], 'loss': []}
    if config.mod == 'our_net':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=151, T_mult=1, eta_min=1e-6, last_epoch=-1)
    elif config.mod == 'float_net':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[10, 30, 100, 150, 250],
                                                         gamma=0.7)

    best_acc = 0
    iters = len(train_iter)

    for epoch in range(epochs):
        net.train()
        epoch_loss = []
        psnr_total = 0
        psnr_bad_total = 0

        for i, (X, Y) in enumerate(tqdm(train_iter)):
            step = epoch * (int(50000 / config.batch_size) + 1) + i
            X, Y = X.to(device), Y.to(device)

            optimizer.zero_grad()
            if config.mod == 'our_net':
                code, _, _, _, y_class, y_recon , y_recon_bad = net(X, epoch)
            else:
                _, _, y_class, y_recon = net(X)

            loss_good = loss_f2(y_recon, X)
            loss_bad = loss_f2(y_recon_bad, X)
            if config.training_phase == '1' :

                loss = loss_good
            if config.training_phase == '2' :

                loss = loss_bad

            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.cpu().item())
            psnr = PSNR(X, y_recon.detach())
            psnr_total += psnr
            psnr_bad = PSNR(X, y_recon_bad.detach())
            psnr_bad_total += psnr_bad

        scheduler.step()

        loss = sum(epoch_loss) / len(epoch_loss)

        if epoch % 20 == 0:
            acc, mse, psnr, ssim, _, psnr_bad = EVAL(net, test_iter, device, config, step, epoch)
            print(
                'epoch: {:d}, loss: {:.6f}, acc: {:.3f}, mse: {:.6f}, psnr: {:.3f}, psnr_bad: {:.3f},ssim: {:.3f}, lr: {:.6f}'.format
                (epoch, loss, acc, mse, psnr, psnr_bad, ssim, optimizer.state_dict()['param_groups'][0]['lr']))
            print('train acc: {:.3f}'.format(0))
            results['epoch'].append(epoch)
            results['loss'].append(loss)
            results['acc'].append(0)
            results['mse'].append(mse)
            results['psnr'].append(psnr)
            results['psnr_bad'].append(psnr_bad)
            results['ssim'].append(0)

        if (epochs - epoch) <= 10 :

            model_path2 = '/{}/{}/'.format(config.mod, config.mod_method)
            model_name = 'CIFAR_{}_a{:.3f}_SNR{:.3f}_BADSNR{:.3f}_Trans{:d}_{}_image_recovery.pth.tar'.format(
                            config.mod, config.a,config.snr_train, config.snr_train_bad, config.trans_bit, config.mod_method)
                # if config.task == 'classification':
                #     model_name = 'CIFAR_{}_SNR{:.3f}_Trans{:d}_{}_classification.pth.tar'.format(
                #         config.mod, config.snr_train, config.trans_bit, config.mod_method)
                # elif config.task == 'image_recovery':
                #     model_name = 'CIFAR_{}_SNR{:.3f}_Trans{:d}_{}_image_recovery.pth.tar'.format(
                #         config.mod, config.snr_train, config.trans_bit, config.mod_method)
                # elif config.task == 'both':
                #     model_name = 'CIFAR_{}_SNR{:.3f}_Trans{:d}_{}.pth.tar'.format(
                #        config.mod, config.snr_train, config.trans_bit, config.mod_method)
            file_name = config.model_path + model_path2
            if not os.path.exists(file_name):
                os.makedirs(file_name)
            torch.save(net.state_dict(), file_name + model_name)
            print("Save model!")

    # in the end save all the results
    if config.mod == 'quant_net':
        result_path2 = '/{}/{}_{}bits/'.format(config.mod, config.mod_method, config.quant_num)
        result_name = 'CIFAR_{}_SNR{:.3f}_Trans{:d}_{}_{}bits.csv'.format(
            config.mod, config.snr_train, config.trans_bit, config.mod_method, config.quant_num)
    else:
        result_path2 = '/{}/{}/'.format(config.mod, config.mod_method)
        if config.task == 'classification':
            result_name = 'CIFAR_{}_SNR{:.3f}_BADSNR{:.3f}_Trans{:d}_{}_classification.csv'.format(
                config.mod, config.snr_train, config.snr_train_bad, config.trans_bit, config.mod_method)
        elif config.task == 'image_recovery':
            result_name = 'CIFAR_{}_a{:.3f}_SNR{:.3f}_BADSNR{:.3f}_Trans{:d}_{}_image_recovery.csv'.format(
                config.mod, config.a,config.snr_train, config.snr_train_bad, config.trans_bit, config.mod_method)
        elif config.task == 'both':
            result_name = 'CIFAR_{}_SNR{:.3f}_BADSNR{:.3f}_Trans{:d}_{}.csv'.format(
                config.mod, config.snr_train, config.snr_train_bad, config.trans_bit, config.mod_method)

    data = pd.DataFrame(results)
    file_name = config.result_path + result_path2
    if not os.path.exists(file_name):
        os.makedirs(file_name)
    data.to_csv(file_name + result_name, index=False, header=False)


