# Generic Type imports
from types import FunctionType
from typing import Union

# Torch imports
import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t, Tensor

# module imports
from ..logger import logger_init, logger_forward
from ..Quantizer import LinQuantExpScale, FakeQuant
from ..DataWrapper import DataWrapper
from ..utils import DownScaler

# current module imports
from .weight_quantization import LinQuantWeight

from .. import __DEBUG__


class Conv2d(nn.Conv2d):
    """
    Conv2dQuant The Convolution Class for a quantized convolution

    **IMPORTANT** Acts as an independent class if no function is passed to the forward method (if independent it quantizes the output by shifting)

    It is a 2d convolution from `torch.nn.Conv2d` with modifications to allow for weight shaping

    :param in_channels: Number of input channels
    :type in_channels: int
    :param out_channels: Number of output channels
    :type out_channels: int
    :param kernel_size: The kernal size
    :type kernel_size: _size_2_t
    :param stride: The stride, defaults to 1
    :type stride: _size_2_t, optional
    :param padding: The padding either a number od a string describing it, defaults to 0
    :type padding: Union[str, _size_2_t], optional
    :param dilation: Dilation identical to `torch.nn.Conv2d`, defaults to 1
    :type dilation: _size_2_t, optional
    :param groups: Groups, defaults to 1
    :type groups: int, optional
    :param bias: Adds a trainable bias if True, defaults to True
    :type bias: bool, optional
    :param padding_mode: padding mode identical to `torch.nn.Conv2d`, defaults to "zeros"
    :type padding_mode: str, optional

    :param weight_quant: A callable object which overrides the default weight quantization, gets called with (weight,rexp_mean,rexp_diff,alpha_func(Tensor)->Tensor) , defaults to None
    :type weight_quant: class or function, optional
    :param weight_quant_bits: Number of bits , defaults to 8
    :type weight_quant_bits: int, optional
    :param weight_quant_channel_wise: If True makes a channel-wise quantization, defaults to False
    :type weight_quant_channel_wise: bool, optional
    :param weight_quant_args: Overrides arguments for the weight quantization initializer with custom ones, defaults to None
    :type weight_quant_args: _type_, optional
    :param weight_quant_kargs: Passes named arguments to the initializer of the weight quantization class, defaults to {}
    :type weight_quant_kargs: dict, optional
    :param out_quant: A callable object which overrides the default output quantization, gets called with (values) , defaults to None
    :type out_quant: _type_, optional
    :param out_quant_bits: Number of bits, defaults to 8
    :type out_quant_bits: int, optional
    :param out_quant_channel_wise: If True makes a channel-wise quantization, defaults to False
    :type out_quant_channel_wise: bool, optional
    :param out_quant_args: Overrides arguments for the out quantization initializer with custom ones, defaults to None
    :type out_quant_args: _type_, optional
    :param out_quant_kargs: Passes named arguments to the initializer of the out quantization class, defaults to {}
    :type out_quant_kargs: dict, optional
    """

    @logger_init
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        weight_quant=None,
        weight_quant_bits=8,
        weight_quant_channel_wise=False,
        weight_quant_args=None,
        weight_quant_kargs={},
        out_quant=None,
        out_quant_args=None,
        out_quant_kargs={},
    ) -> None:
        super(Conv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

        assert groups == 1 or (
            groups == in_channels and groups == out_channels)

        # Weight Quant
        if weight_quant_args == None:
            weight_quant_args = (
                weight_quant_bits,
                (-1,) if not weight_quant_channel_wise else (out_channels, 1, 1, 1),
                "round",
                groups == in_channels
            )

        self.layer_wise = groups == in_channels
        # print(self.layer_wise,groups,in_channels)

        if weight_quant == None:
            self.weight_quant = LinQuantWeight(
                *weight_quant_args, **weight_quant_kargs)
        else:
            self.weight_quant = weight_quant(
                *weight_quant_args, **weight_quant_kargs)

        self.register_buffer("quant_weight", torch.zeros_like(self.weight))

        # Out Quant
        # only used if factor_fun in forward is None
        if out_quant_args == None:
            out_quant_args = (
                8,
                (1, out_channels, 1, 1),
                'floor'
            )

        self.down_scaler = DownScaler((1, out_channels, 1, 1), out_quant=out_quant,
                                      out_quant_args=out_quant_args, out_quant_kargs=out_quant_kargs)

        self.register_buffer("rexp_diff", torch.zeros((1, in_channels, 1, 1)))

        self.test = {}

    @logger_forward
    def forward(
        self, input: DataWrapper, factor_fun: FunctionType = None
    ) -> DataWrapper:
        """
        forward Computes the convolution with quantized weights

        **IMPORTANT** Acts as an independent class if no function is passed to the forward method (if independent it quantizes the output by shifting)

        :param invals: The values of the previous layer
        :type invals: Tuple[torch.Tensor, torch.Tensor]
        :param factor_fun: A function for additional weight scaling , defaults to None
        :type factor_fun: FunctionType, optional
        :return: Returns the computed values and the exponents
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        if factor_fun == None:
            return self.forward_stand_alone(input)
        else:
            return self.forward_integrated(input, factor_fun)

    @logger_forward
    def forward_stand_alone(self, input: DataWrapper) -> DataWrapper:
        value, rexp = input.get()

        with torch.no_grad():
            if self.layer_wise:
                rexp_mean = rexp
                self.rexp_diff = rexp - rexp_mean
            else:
                rexp_mean = torch.mean(rexp)
                self.rexp_diff = rexp - rexp_mean

        weight_function = self.down_scaler.get_weight_function()

        weight = self.weight_quant(
            self.weight,
            rexp_mean.exp2(),
            self.rexp_diff.exp2(),
            weight_function,
        )

        self.test['weight'] = weight

        if not self.training:
            self.quant_weight = weight.clone().detach()

        out = self._conv_forward(value, weight, None)

        if self.bias is not None:
            bias = self.bias.view(1,-1,1,1)
        else:
            bias = None

        return self.down_scaler(input.set(out, rexp_mean + (self.weight_quant.delta_out.log2().view(1, -1, 1, 1))), bias)

    @logger_forward
    def forward_integrated(self, input: DataWrapper, factor_fun: FunctionType = None) -> DataWrapper:

        value, rexp = input.get()

        if self.layer_wise:
            rexp_mean = rexp.clone().detach()
            self.rexp_diff = rexp - rexp_mean
        else:
            rexp_mean = torch.mean(rexp)
            self.rexp_diff = rexp - rexp_mean

        weight = self.weight_quant(
            self.weight,
            rexp_mean.exp2(),
            self.rexp_diff.exp2(),
            factor_fun,
        )

        self.test['weight'] = weight

        if not self.training:
            self.quant_weight = weight.clone().detach()

        out = self._conv_forward(value, weight, None)

        return input.set(out, rexp_mean + (self.weight_quant.delta_out.log2().detach().view(1, -1, 1, 1)))
