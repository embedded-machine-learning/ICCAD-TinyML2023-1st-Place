from types import FunctionType
from numpy import outer

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.common_types import Tensor

from .Linear_int import Linear_int

from ..logger import logger_forward, logger_init

from ..Quantizer import LinQuantExpScale, FakeQuant
from ..DataWrapper import DataWrapper
from .. import __DEBUG__
from ..utils import DownScaler

from .weight_quantization import LinQuantWeight


class Linear(nn.Linear):
    """
    Linear The Linear Class for a Quantized Dense Layer

    **IMPORTANT** Acts as an independent class if no function is passed to the forward method (if independent it quantizes the output by shifting)

    It is a linear Layer from `torch.nn.Linear` with modifications to allow for weight shaping

    :param in_features: Number of input features
    :type in_features: int
    :param out_features: Number of output features
    :type out_features: int
    :param bias: If True use a bias, defaults to True
    :type bias: bool, optional

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
        in_features: int,
        out_features: int,
        bias: bool = True,
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

        super(Linear, self).__init__(in_features, out_features, bias, device, dtype)

        if weight_quant_args == None:
            weight_quant_args = (
                weight_quant_bits,
                (-1,) if not weight_quant_channel_wise else (out_features, 1),
                "round",
            )

        if weight_quant == None:
            self.weight_quant = LinQuantWeight(*weight_quant_args, **weight_quant_kargs)
        else:
            self.weight_quant = weight_quant(*weight_quant_args, **weight_quant_kargs)

        self.register_buffer("quant_weight", torch.zeros_like(self.weight))
        # only used if factor_fun in forward is None
        if out_quant_args == None:
            out_quant_args = (
                8,
                (1, out_features),
                'floor'
            )

        self.down_scaler = DownScaler((1,out_features),out_quant=out_quant,out_quant_args=out_quant_args,out_quant_kargs=out_quant_kargs)

        self.test = {}

        self.register_buffer('rexp_diff',torch.zeros((1,in_features)))

        
    def int_extract(self, accumulation_type = torch.int32, small_signed_type = torch.int8, small_unsigned_type=torch.uint8) -> Linear_int:
        return Linear_int(
            self.in_features,
            self.out_features,
            self.quant_weight,
            self.down_scaler.right_shift,
            self.down_scaler.integer_bias if self.down_scaler.integer_bias is not None else None,
            self.down_scaler.out_quant.min,
            self.down_scaler.out_quant.max,
            accumulation_type = accumulation_type,
            small_signed_type = small_signed_type,
            small_unsigned_type = small_unsigned_type,
        )


    @logger_forward
    def forward(self, input: DataWrapper, factor_fun: FunctionType = None) -> torch.Tensor:
        """
        forward Computes the Linear layer with quantization

        **IMPORTANT** Acts as an independent class if no function is passed to the forward method (if independent it quantizes the output by shifting)

        :param input: The values of the previous layer
        :type input: Tuple[torch.Tensor, torch.Tensor]
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
    def forward_stand_alone(self, input: DataWrapper) -> torch.Tensor:
        value, rexp = input.get()

        rexp_mean = torch.mean(rexp)
        rexp_diff = rexp - rexp_mean

        self.rexp_diff = rexp_diff.clone()
        
        weight_function = self.down_scaler.get_weight_function()

        weight = self.weight_quant(
            self.weight,
            rexp_mean.exp2(),
            rexp_diff.exp2(),
            weight_function,
        )

        self.test['weight'] = weight
        
        if not self.training:
            self.quant_weight = weight.detach().clone()

        out = F.linear(value, weight, None)

        if self.bias is not None:
            bias = self.bias.view(1,-1)
        else:
            bias = None

        return self.down_scaler(input.set(out, rexp_mean + self.weight_quant.delta_out.log2().view(1, -1)),bias.view(1,-1) if bias is not None else None)
        
    @logger_forward
    def forward_integrated(self, input: DataWrapper, factor_fun: FunctionType = None) -> torch.Tensor:
        x, rexp = input.get()

        rexp_mean = torch.mean(rexp)
        rexp_diff = rexp - rexp_mean

        weight = self.weight_quant(
            self.weight,
            rexp_mean.exp2(),
            rexp_diff.exp2(),
            factor_fun,
        )

        self.test['weight'] = weight

        if not self.training:
            self.quant_weight = weight.detach().clone()
            
        out = F.linear(x, weight, None)

        return input.set(out, rexp_mean + self.weight_quant.delta_out.log2().view(1, -1))
