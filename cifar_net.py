from torch import nn
import torch
from torch.nn.functional import gumbel_softmax
# from quantization import quantization, dequantization
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from modules import Encoder, Decoder_Recon, Decoder_Class, awgn, normalize, ResidualBlock
# from utils import count_percentage
import math

def modulation(logits, device, epoch, mod_method='bpsk'):
    eps = 1e-10
    if mod_method == 'bpsk':
        num_cate = 2
    if mod_method == '4qam':
        num_cate = 2
    if mod_method == '16qam':
        num_cate = 4
    if mod_method == '64qam':
        num_cate = 8
    if mod_method == '256qam':
        num_cate = 16
    if mod_method == '1024qam':
        num_cate = 32

    prob_z = gumbel_softmax(logits, hard=False)
    discrete_code = gumbel_softmax(logits, hard=True, tau=1.5)
    if mod_method == 'bpsk':
        output = discrete_code[:, :, 0] * (-1) + discrete_code[:, :, 1] * 1
    if mod_method == '4qam':
        const = [1, -1]
        const = torch.tensor(const).to(device)
        temp = discrete_code * const
        output = torch.sum(temp, dim=2)
    if mod_method == '16qam':
        const = [-3, -1, 1, 3]
        const = torch.tensor(const).to(device)
        temp = discrete_code * const
        output = torch.sum(temp, dim=2)
    if mod_method == '64qam':
        const = [-7, -5, -3, -1, 1, 3, 5, 7]
        const = torch.tensor(const).to(device)
        temp = discrete_code * const
        output = torch.sum(temp, dim=2)
    if mod_method == '256qam':
        const = [-15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15]
        const = torch.tensor(const).to(device)
        temp = discrete_code * const
        output = torch.sum(temp, dim=2)
    if mod_method == '1024qam':
        const = [-31, -29, -27, -25, -23, -21, -19, -17, -15, -13, -11, -9, -7, -5, -3, -1,
                 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
        const = torch.tensor(const).to(device)
        temp = discrete_code * const
        output = torch.sum(temp, dim=2)
    return output, prob_z

def quadrant_adjustment(tensor, a, device):

    processed_a = math.sqrt((1 - a) / 2)
    adjust_vectors = torch.tensor([[processed_a, processed_a],
                                   [-processed_a, processed_a],
                                   [processed_a, -processed_a],
                                   [-processed_a, -processed_a]],
                                  ).to(device)

    xx = (tensor[..., 0:1] > 0).long()
    yy = (tensor[..., 1:2] > 0).long()

    quadrant = 2 * yy + xx

    adjustment = adjust_vectors[quadrant.squeeze()]

    tensor += adjustment

    return tensor


class Our_Net(nn.Module):
    def __init__(self, config, device):
        super(Our_Net, self).__init__()
        self.config = config
        self.device = device
        self.a = config.a

        if self.config.mod_method == 'bpsk':
            self.num_category = 2
        elif self.config.mod_method == '4qam':
            self.num_category = 2
        elif self.config.mod_method == '16qam':
            self.num_category = 4
        elif self.config.mod_method == '64qam':
            self.num_category = 8
        elif self.config.mod_method == '256qam':
            self.num_category = 16
        elif self.config.mod_method == '1024qam':
            self.num_category = 32

        self.encoder = Encoder(self.config)
        if config.mod_method == 'bpsk':
            self.prob_convs = nn.Sequential(
                nn.Linear(config.trans_bit * 4 * 4, config.trans_bit * self.num_category),
                nn.ReLU(),
            )
        else:
            self.prob_convs = nn.Sequential(
                nn.Linear(config.trans_bit * 2 * 4 * 4, config.trans_bit * 2 * self.num_category),
                nn.ReLU(),
            )

        self.decoder_recon_good = Decoder_Recon(self.config)

        self.decoder_recon_bad = Decoder_Recon(self.config)

        if self.config.mod_method == 'bpsk':
            self.decoder_class = Decoder_Class(int(config.trans_bit / 2), int(config.trans_bit / 8))
        else:
            self.decoder_class = Decoder_Class(int(config.trans_bit * 2 / 2), int(config.trans_bit * 2 / 8))

        self.initialize_weights()

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight, gain=1)
            elif isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, gain=1)

    def reparameterize(self, probs, step):
        mod_method = self.config.mod_method
        code, prob_code = modulation(probs, self.device, step, mod_method)
        return code, prob_code

    def forward(self, x, epoch):
        x_f = self.encoder(x).reshape(x.shape[0], -1)
        z = self.prob_convs(x_f).reshape(x.shape[0], -1, self.num_category)
        z_v, prob_code = self.reparameterize(z, epoch)

        zv_shape = z_v.shape

        z_u = (torch.randint(2, size=zv_shape).to(self.device) * 2 - 1).float()

        _, z_u_norm = normalize(z_u)
        _, z_v_norm = normalize(z_v)

        z = (self.a ** 0.5) * z_v_norm + ((1 - self.a) ** 0.5) * z_u_norm

        if self.config.mode == 'train':

            z_hat_good = awgn(self.config.snr_train, z, self.device)
            z_hat_bad = awgn(self.config.snr_train_bad, z, self.device)

        if self.config.mode == 'test':
            z_hat_good = awgn(self.config.snr_test, z, self.device)
            z_hat_bad = awgn(self.config.snr_test_bad, z, self.device)

        z_hat_reshape1 = z_hat_good.reshape(z_hat_good.shape[0], -1, 2)

        z_hat_reshape2 = quadrant_adjustment(z_hat_reshape1, self.config.a, self.device)

        recon_good = self.decoder_recon_good(z_hat_reshape2)

        z_hat_reshape3 = z_hat_bad.reshape(z_hat_bad.shape[0], -1, 2)

        z_hat_reshape4 = quadrant_adjustment(z_hat_reshape3, self.config.a, self.device)

        recon_bad = self.decoder_recon_bad(z_hat_reshape4)

        r_class = self.decoder_class(z_hat_good)

        return z_v, prob_code, z, z_hat_good, r_class, recon_good , recon_bad




