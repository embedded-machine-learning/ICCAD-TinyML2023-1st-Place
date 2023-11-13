from typing import Union

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from ..Quantizer import LinQuantExpScale

from ..logger import logger_init, logger_forward

from ..DataWrapper import DataWrapper

from .batchnorm2d import BatchNorm2d

from .. import (
    __DEBUG__,
    __HIGH_PRES__,
    __HIGH_PRES_USE_RUNNING__,
)

from .. import __TESTING_FLAGS__


class BatchNorm2d_playground(BatchNorm2d):
    def __init__(self, num_features: int, eps: float = 0.00001, momentum: float = 0.1, affine: bool = True, track_running_stats: bool = True, device=None, dtype=None, fixed_n: bool = False, out_quant=None, out_quant_args=None, out_quant_kargs=..., shift_alpha_function=None):
        super().__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype, fixed_n, out_quant, out_quant_args, out_quant_kargs, shift_alpha_function)
        global NAME_INDEX
        self.NAME_INDEX = NAME_INDEX
        NAME_INDEX += 1
        self.FILE_NAME = './bn_values/' + str(self.NAME_INDEX)
        self.counter_max = 1000
        self.counter = self.counter_max
        self.STAFFEL = self.NAME_INDEX * 300
        self.register_buffer('mul_norm', torch.ones_like(self.running_var))

    @logger_forward
    def forward(self, input: DataWrapper, activation: Union[None, nn.Module] = None, conv=None) -> DataWrapper:

        if not self.training:
            return self.forward_eval(input, activation)
        # return self.forward_train_fast(input, activation)
        return self.forward_train_test(input, activation)

    @logger_forward
    def pre_forward(self, conv=None) -> None:
        if __TESTING_FLAGS__['FUZE_BN']:
            if self.training:
                # if self.counter == self.counter_max:
                if self.counter % 100 == 0:
                    if conv is not None:
                        print('fuzing bn')
                        mod_fac = self.weight.data.view(-1).div(self.running_var.data.add(self.eps).sqrt().view(-1))
                        mod_fac = mod_fac.view(-1, 1, 1, 1)
                        self.mul_norm *= self.weight.data.square()
                        self.running_var.data = torch.ones_like(self.running_var)
                        conv.weight.data = conv.weight.data * mod_fac
                        self.weight.data = torch.ones_like(self.weight)
                self.counter -= 1
    
    @logger_forward
    def forward_train_test(self, input: DataWrapper, activation: Union[None, nn.Module] = None):
        x, rexp = input.get()

        if activation != None:
            self.out_quant.copy(activation)
            quant = activation
        else:
            quant = self.out_quant

        ###### SAVE VALUES #####
        var_u = torch.var(x.detach(), [0, 2, 3], unbiased=False, keepdim=True)
        mu = torch.mean(x.detach(), [0, 2, 3], keepdim=True)
        with open(self.FILE_NAME + '_var.csv', 'a+') as f:
            np.savetxt(f, var_u.detach().cpu().numpy().reshape(1, -1))
        with open(self.FILE_NAME + '_mu.csv', 'a+') as f:
            np.savetxt(f, mu.detach().cpu().numpy().reshape(1, -1))

        with open(self.FILE_NAME + '_bias.csv', 'a+') as f:
            np.savetxt(f, self.bias.detach().cpu().numpy().reshape(1, -1))
        with open(self.FILE_NAME + '_weight.csv', 'a+') as f:
            np.savetxt(f, self.weight.detach().cpu().numpy().reshape(1, -1))

        ###### CALCULATE #######
        if not __TESTING_FLAGS__['FREEZE_BN']:  # or self.STAFFEL > 0:
            var_u = torch.var(x, [0, 2, 3], unbiased=False, keepdim=True).div(self.mul_norm.view(1, -1, 1, 1))
            mu = torch.mean(x, [0, 2, 3], keepdim=True)
            mul = self.weight.view(1, -1, 1, 1).div(var_u.add(self.eps).sqrt().view(1, -1, 1, 1))
            with torch.no_grad():
                self.running_var = self.running_var * (1 - self.momentum) + var_u.view(-1) * self.momentum
                self.running_mean = self.running_mean * (1 - self.momentum) + mu.view(-1) * self.momentum
            # x = super(BatchNorm2d, self).forward(x)

            with torch.no_grad():
                N = list(x.shape)
                del N[1]
                N = np.prod(N)
                # val = (self.weight.view(1,-1,1,1)/(torch.sqrt(var.view(1,-1,1,1)+self.eps))).view(1,-1,1,1)
                # print(val.shape)
                val = (1 - 1 / N - 1 / (N - 1) * ((x - mu.view(1, -1, 1, 1)) /
                       torch.sqrt(var_u + self.eps).view(1, -1, 1, 1)).square())
                val = val.detach().cpu().numpy().reshape(-1)
                arr = [np.min(val), np.mean(val), np.max(val)]
                arr = np.array(arr)
                with open(self.FILE_NAME + '_backprop.csv', 'a+') as f:
                    np.savetxt(f, arr.reshape(1, -1))

            x = x.sub(mu.view(1, -1, 1, 1)).mul(mul).add(self.bias.view(1, -1, 1, 1))

        else:
            if self.counter == 0:
                print('bn frozen')
            # self.weight.requires_grad_(False)
            var = torch.var(x, [0, 2, 3], unbiased=False,)
            mu = torch.mean(x, [0, 2, 3],)

            mom = 0.01 if self.counter < 0 else (self.counter / self.counter_max) * (1 - 0.01) + 0.01
            # mu.data = self.running_mean.data * (1 - mom) + mu * mom
            # var_u = var.clone()
            var_u = self.running_var.data * (1 - mom) + var.div(self.mul_norm.view(-1)) * mom
            # mu_u = self.running_mean.data * (1 - mom) + mu.div(self.mul_norm.view(-1)) * mom
            # running_mean_tmp = mu
            # running_var_tmp = var
            mul = self.weight.view(1, -1, 1, 1).div(var_u.add(self.eps).sqrt().view(1, -1, 1, 1))

            with torch.no_grad():
                N = list(x.shape)
                del N[1]
                N = np.prod(N)
                # val = (self.weight.view(1,-1,1,1)/(torch.sqrt(var.view(1,-1,1,1)+self.eps))).view(1,-1,1,1)
                # print(val.shape)
                val = (1 - 1 / N - 1 / (N - 1) * ((x - mu.view(1, -1, 1, 1)) /
                       torch.sqrt(var_u + self.eps).view(1, -1, 1, 1)).square())
                val = val.detach().cpu().numpy().reshape(-1)
                arr = [np.min(val), np.mean(val), np.max(val)]
                arr = np.array(arr)
                with open(self.FILE_NAME + '_backprop.csv', 'a+') as f:
                    np.savetxt(f, arr.reshape(1, -1))

            x = x.sub(mu.view(1, -1, 1, 1)).mul(mul).add(self.bias.view(1, -1, 1, 1))
            if mom <= 0.1:
                # self.running_mean.data = mu
                self.running_mean.data = self.running_mean.data * (1 - 0.1) + mu * 0.1
                self.running_var.data = var_u
            else:
                self.running_mean.data = self.running_mean.data * (1 - 0.1) + mu * 0.1
                self.running_var.data = self.running_var.data * (1 - 0.1) + var.div(self.mul_norm.view(-1)) * 0.1
            self.counter -= 1

        ###### SAVE VALUES #####
        with open(self.FILE_NAME + '_running_var.csv', 'a+') as f:
            np.savetxt(f, self.running_var.detach().cpu().numpy().reshape(1, -1))
        with open(self.FILE_NAME + '_running_mu.csv', 'a+') as f:
            np.savetxt(f, self.running_mean.detach().cpu().numpy().reshape(1, -1))

        x = quant(x, False, input)

        rexp = torch.log2(quant.delta_out)
        return input.set(x, rexp)
