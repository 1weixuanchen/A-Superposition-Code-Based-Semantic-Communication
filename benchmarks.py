from torch import nn
import torch
from torch.nn.functional import gumbel_softmax
# from quantization import quantization, dequantization
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from modules import Encoder, Decoder_Recon, Decoder_Class, awgn, normalize


class Float_Net(nn.Module):
    def __init__(self, config, device):
        super(Float_Net, self).__init__()
        self.config = config
        self.device = device

        self.encoder = Encoder(self.config)

        if self.config.mod_method == 'bpsk':
            self.dimension = 1
        else:
            self.dimension = 2

        self.avepool = nn.AvgPool2d(4)
        self.decoder_class = Decoder_Class(int(config.trans_bit / 2 * self.dimension), int(config.trans_bit / 8 * self.dimension))
        self.decoder_recon = Decoder_Recon(config)

    def forward(self, x):
        z = self.avepool(self.encoder(x)).reshape(x.shape[0], -1)
        power, z = normalize(z)

        if self.config.mode == 'train':
            z_hat = awgn(self.config.snr_train, z, self.device)
        if self.config.mode == 'test':
            z_hat = awgn(self.config.snr_test, z, self.device)

        y_class = self.decoder_class(z_hat)
        y_recon = self.decoder_recon(z_hat)
        return z, z_hat, y_class, y_recon


class Quant_Net(nn.Module):
    def __init__(self, config, device):
        super(Quant_Net, self).__init__()
        self.config = config
        self.device = device
        self.quant_num = self.config.quant_num

        if self.config.mod_method == 'bpsk':
            self.dimension = 1
        if self.config.mod_method == '4qam' or self.config.mod_method == '16qam':
            self.dimension = 2

        self.encoder = Encoder(self.config)
        self.avepool = nn.AvgPool2d(4)

        self.decoder_class = Decoder_Class(int(config.trans_bit * self.dimension / 2), int(config.trans_bit * self.dimension / 8))
        self.decoder_recon = Decoder_Recon(self.config)

    def forward(self, x):
        z = self.avepool(self.encoder(x)).reshape(x.shape[0], -1)
        power, z = normalize(z)

        z_quant, z_hat = quant_dequant(z, self.device, self.config)

        y_class = self.decoder_class(z_hat)
        y_recon = self.decoder_recon(z_hat)
        return z, z_quant, z_hat, y_class, y_recon


def quant_dequant(z_float, device, config):
    z_float = z_float.clamp(-2, 2)
    """
    1 bit + bpsk / 2 bit + 4qam:
    [-2, 0]: -1 => -1; [0, 2]: 1 => 1;
    4 bit + 16qam:
    [-2, -1]: -1.5 => -1; [-1, 0]: -0.5 => -1/3; [0, 1]: 0.5 => 1/3; [1, 2]: 1.5 => 1;
    """
    if config.mod_method == 'bpsk' or config.mod_method == '4qam':
        x = torch.where(z_float >= 0, 1.0, -1.0)
        normalized_x = (1 / torch.mean(x ** 2) ** 0.5) * x
        if config.mode == 'train':
            z_hat = awgn(config.snr_train, normalized_x, device)
        elif config.mode == 'test':
            z_hat = awgn(config.snr_test, normalized_x, device)
        de_normalized_x = z_hat * (torch.mean(x ** 2) ** 0.5)
        hard_d = torch.where(de_normalized_x >= 0, 1.0, -1.0)
    elif config.mod_method == '16qam':
        x = torch.where(z_float <= -1, -1.5, z_float)
        x = torch.where((z_float > -1) & (z_float <= 0), -0.5, x)
        x = torch.where((z_float > 0) & (z_float <= 1), 0.5, x)
        x = torch.where((z_float > 1) & (z_float <= 2), 1.5, x)
        normalized_x = (1 / torch.mean(x ** 2) ** 0.5) * x
        if config.mode == 'train':
            z_hat = awgn(config.snr_train, normalized_x, device)
        elif config.mode == 'test':
            z_hat = awgn(config.snr_test, normalized_x, device)
        de_normalized_x = z_hat * (torch.mean(x ** 2) ** 0.5)
        hard_d = torch.where(de_normalized_x < -1, -1.5, de_normalized_x)
        hard_d = torch.where((de_normalized_x >= -1) & (de_normalized_x < 0), -0.5, hard_d)
        hard_d = torch.where((de_normalized_x >= 0) & (de_normalized_x < 1), 0.5, hard_d)
        hard_d = torch.where(de_normalized_x >= 1, 1.5, hard_d)

    return x, de_normalized_x


class LBSign_16(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        # if mod == '16qam':
        out = torch.where(input >= 1, 1.5, input)
        out = torch.where((input >= 0) & (input < 1), 0.5, out)
        out = torch.where((input >= -1) & (input < 0), -0.5, out)
        out = torch.where(input <= -1, -1.5, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clamp_(-0.2, 0.2)


class LBSign_4(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class NN_Quant(nn.Module):
    def __init__(self, config, device):
        super(NN_Quant, self).__init__()
        self.config = config
        self.device = device
        self.snr = config.snr_test
        self.quant_num = self.config.quant_num

        if self.config.mod_method == 'bpsk':
            self.num = 1
        if self.config.mod_method == '4qam' or self.config.mod_method == '16qam':
            self.num = 2

        if config.mod_method == '4qam' or self.config.mod_method == 'bpsk':
            if config.snr_train == 18 or config.snr_train == 12 or config.snr_train == 6:
                dropout_factor = 0.1
            elif config.snr_train == 0:
                dropout_factor = 0.35
            elif config.snr_train == -6 or config.snr_train == -12 or config.snr_train == -18 or config.snr_train == -15:
                dropout_factor = 0.7
        else:
            if config.snr_train == 18 or config.snr_train == 12:
                dropout_factor = 0.05
            elif config.snr_train == 6 or config.snr_train == 0:
                dropout_factor = 0.1
            elif config.snr_train == -6 or config.snr_train == -12 or config.snr_train == -18 or config.snr_train == -15:
                dropout_factor = 0.5
        self.quant = nn.Sequential(nn.Dropout(dropout_factor),
            nn.Linear(int(self.config.trans_bit * self.num), int(self.config.trans_bit * self.num)))
        self.de_quant = nn.Linear(int(self.config.trans_bit * self.num), int(self.config.trans_bit * self.num))
        self.tanh = nn.Tanh()

    def forward(self, x, Test=0):
        if self.config.mod_method == '16qam':
            sign = LBSign_16.apply
            symbols = sign(self.tanh(self.quant(x)) * 2)
        elif self.config.mod_method == '4qam' or self.config.mod_method == 'bpsk':
            sign = LBSign_4.apply
            symbols = sign(self.tanh(self.quant(x)))

        power, symbols = normalize(symbols)
        if Test == 0:
            output = self.de_quant(symbols)
        else:
            if self.config.mode == 'train':
                z_hat = awgn(self.config.snr_train, symbols, self.device)
                output = self.de_quant(z_hat)
            elif self.config.mode == 'test':
                z_hat = awgn(self.config.snr_test, symbols, self.device)
                output = self.de_quant(z_hat)

        return output


class Quant_NN_Net(nn.Module):
    def __init__(self, config, device):
        super(Quant_NN_Net, self).__init__()

        self.config = config
        self.device = device

        if self.config.mod_method == 'bpsk':
            self.num = 1
        if self.config.mod_method == '4qam' or self.config.mod_method == '16qam':
            self.num = 2

        self.encoder = Encoder(self.config)
        self.avepool = nn.AvgPool2d(4)

        self.decoder_class = Decoder_Class(int(config.trans_bit * self.num / 2), int(config.trans_bit * self.num / 8))
        self.decoder_recon = Decoder_Recon(self.config)
        self.quant = NN_Quant(self.config, self.device)

    def forward(self, x, Test=0):
        z = self.avepool(self.encoder(x)).reshape(x.shape[0], -1)
        power, z = normalize(z)

        z_hat = self.quant(z, Test)

        y_class = self.decoder_class(z_hat)
        y_recon = self.decoder_recon(z_hat)
        return z, z_hat, y_class, y_recon