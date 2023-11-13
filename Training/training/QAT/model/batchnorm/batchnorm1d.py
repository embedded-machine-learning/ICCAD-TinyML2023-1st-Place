from typing import Union

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from ..Quantizer import LinQuantExpScale

from ..logger import logger_init, logger_forward

from ..DataWrapper import DataWrapper

from .. import (
    __DEBUG__,
    __HIGH_PRES__,
    __HIGH_PRES_USE_RUNNING__,
)

from .. import __TESTING_FLAGS__


from .functions import calculate_n_a, calculate_n_a_fixed, calculate_t

NAME_INDEX = 0


class BatchNorm1d(torch.nn.BatchNorm1d):
    """
    BatchNorm1d Modified nn.BatchNorm1d

    Modified to create a convolution weight adaptation factor and calculate the eval BN as a addition and shift

    :param num_features: Number of channels
    :type num_features: int
    :param eps: A factor to make div 0 impossible, defaults to 0.00001
    :type eps: float, optional
    :param momentum: The momentum of th BN, defaults to 0.1
    :type momentum: float, optional
    :param affine: BN Affine, defaults to True
    :type affine: bool, optional
    :param track_running_stats: BN running stats, defaults to True
    :type track_running_stats: bool, optional

    :param fixed_n: Set the shift to a layer-wise value rather than channel-wise, defaults to False
    :type fixed_n: bool, optional

    :param out_quant:  A callable object which overrides the default output quantization, gets called with (values) , defaults to None
    :type out_quant: _type_, optional
    :param out_quant_bits: Number of bits for the output quantization, defaults to 8
    :type out_quant_bits: int, optional
    :param out_quant_channel_wise: Channel-wise output quantization, defaults to False
    :type out_quant_channel_wise: bool, optional
    :param out_quant_args:  Overrides arguments for the out quantization initializer with custom ones, defaults to None
    :type out_quant_args: _type_, optional
    :param out_quant_kargs: Passes named arguments to the initializer of the out quantization class, defaults to {}
    :type out_quant_kargs: dict, optional
    """

    @logger_init
    def __init__(
        self,
        num_features: int,
        eps: float = 0.00001,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
        fixed_n: bool = False,
        out_quant=None,
        out_quant_args=None,
        out_quant_kargs={},
        shift_alpha_function=None,
    ):
        """
        Please read the class help
        """

        # affine is always True as we require the parameters to be present, however they are set to be untrainable
        super(BatchNorm1d, self).__init__(num_features, eps, momentum, True, track_running_stats, device, dtype)
        
        self.affine = affine
        if not affine:
            self.weight.requires_grad_(False)
            self.bias.requires_grad_(False)


        self.register_buffer("n", torch.zeros(1, num_features))
        self.register_buffer("t", torch.zeros(1, num_features))
        self.register_buffer("alpha", 1.0 / np.sqrt(2.0) * torch.ones(num_features))

        self.func_t = calculate_t
        self.fixed_n = fixed_n
        if fixed_n:
            self.func_n_a = calculate_n_a_fixed
        else:
            self.func_n_a = calculate_n_a

        if shift_alpha_function is not None:
            self.func_n_a = shift_alpha_function

        if out_quant_args == None:
            out_quant_args = (
                8,
                (1, num_features),
            )

        if out_quant == None:
            self.out_quant = LinQuantExpScale(*out_quant_args, **out_quant_kargs)
        else:
            self.out_quant = out_quant(*out_quant_args, **out_quant_kargs)

        # TODO: Delete
        global NAME_INDEX
        # self.NAME_INDEX = NAME_INDEX
        # NAME_INDEX += 1
        # self.FILE_NAME = './bn_values/' + str(self.NAME_INDEX)
        # self.counter_max = 1000
        # self.counter = self.counter_max
        # self.STAFFEL = self.NAME_INDEX * 300
        # self.register_buffer('mul_norm', torch.ones_like(self.running_var))

    def get_weight_factor(self):
        """
        get_weight_factor Returns a function to calculate alpha with a singe value
        """

        def ret_fun(rexp):
            _, self.alpha = self.func_n_a(
                weight=self.weight.view(-1).detach(),
                mean=self.running_mean.view(-1).detach(),
                var=self.running_var.view(-1).detach(),
                out_quant=self.out_quant.delta_out.view(-1).detach(),
                rexp=rexp.view(-1).detach(),
            )
            return self.alpha[:, None, None, None]

        return ret_fun

    @logger_forward
    def forward(self, input: DataWrapper, activation: Union[None, nn.Module] = None, conv=None) -> DataWrapper:

        if not self.training:
            return self.forward_eval(input, activation)
        return self.forward_train_fast(input, activation)

    @logger_forward
    def forward_train_fast(self, input: DataWrapper, activation: Union[None, nn.Module] = None):
        x, rexp = input.get()

        if activation != None:
            self.out_quant=activation
            quant = activation
        else:
            quant = self.out_quant

        x = super(BatchNorm1d, self).forward(x)
        x = quant(x, False, input)

        rexp = torch.log2(quant.delta_out)
        return input.set(x, rexp)

    @logger_forward
    def forward_eval(self, input: DataWrapper, activation: Union[None, nn.Module] = None):
        x, rexp = input.get()

        if activation != None:
            self.out_quant=activation
            quant = activation
        else:
            quant = self.out_quant

        with torch.no_grad():
            n,_ = self.func_n_a(
                weight=self.weight.view(-1),
                mean=self.running_mean.view(-1),
                var=self.running_var.view(-1),
                out_quant=quant.delta_out.view(-1),
                rexp=rexp.view(-1),
            )

            self.n = n.detach().view(1, -1)

            t = self.func_t(
                weight=self.weight.view(-1),
                bias=self.bias.view(-1),
                mean=self.running_mean.view(-1),
                var=self.running_var.view(-1),
                out_quant=quant.delta_out.view(-1),
                rexp=rexp.view(-1),
                n=self.n.view(-1),
            ).detach()

            # tmp = torch.exp2(self.n.view(1, -1))

            def mul_pow2(a:torch.Tensor,exp:torch.Tensor):
                return a*torch.exp2(exp)
                mantissa, exponent = torch.frexp(a)
                exponent += exp.type(torch.int)
                return torch.ldexp(mantissa,exponent)


            # self.t = (t.view(1, -1, 1, 1)).div(tmp).floor()
            self.t = mul_pow2(t.view(1,-1),-self.n.view(1,-1)).floor()
            x = x + self.t
            x = mul_pow2(x,self.n.view(1,-1))
            # x = x.mul(tmp)

            x = x.floor()
            x = x.clamp(quant.min, quant.max)
            if __DEBUG__:
                x = torch.nan_to_num(x)

            rexp = torch.log2(quant.delta_out)
            return input.set(x, rexp)
