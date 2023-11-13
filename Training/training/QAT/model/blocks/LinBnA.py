from typing import Union, Optional

import torch
from torch import nn
from torch.nn.common_types import _size_2_t


from ..DataWrapper import DataWrapper
from ..logger import logger_forward, logger_init

from ..Quantizer import Quant
from ..linear import Linear
from ..batchnorm import BatchNorm1d
from ..activations import ReLU

from .LinBnA_int import LinBnA_int


class LinBnA(nn.Module):
    """
    LinBnA A module with a Linear Batch-Norm and activation

    Per default the activation function is a ReLu

    Convolution:
    :param in_features: Number of input features
    :type in_features: int
    :param out_features: Number of output features
    :type out_features: int
    :param weight_quant: Overrides the default weight quantization for the Convolution, defaults to None
    :type weight_quant: _type_, optional
    :param weight_quant_bits: Number of bits for the Convolution Weight quantization, defaults to 8
    :type weight_quant_bits: int, optional
    :param weight_quant_channel_wise: If the Convolution Weight quantization should be done Layer-wise, defaults to False
    :type weight_quant_channel_wise: bool, optional
    :param weight_quant_args: Overrides the args for the Convolution Weight quantization, defaults to None
    :type weight_quant_args: list, optional
    :param weight_quant_kargs: Additional Named Arguments for the Convolution Weight quantization, defaults to {}
    :type weight_quant_kargs: dict, optional

    Batch-Norm:
    :param eps: EPS for the Batch-Norm , defaults to 1e-5
    :type eps: float, optional
    :param momentum: Momentum for the Batch-Norm, defaults to 0.1
    :type momentum: float, optional
    :param affine: Affine for the Batch-Norm, defaults to True
    :type affine: bool, optional
    :param track_running_stats: Trach running stats for the Batch-Norm, defaults to True
    :type track_running_stats: bool, optional
    :param fixed_n: If the batch-Norm should a single shift factor per layer, defaults to False
    :type fixed_n: bool, optional

    Activation:
    :param activation: Overrides the default activation function, e.g. nn.Sequential() for no activation, defaults to None
    :type activation: Optional[Quant], optional
    :param activation_args: Overrides the Arguments provided to the activation function, defaults to None
    :type activation_args: list, optional
    :param activation_kargs: Additional Named parameters for the activation function, defaults to {}
    :type activation_kargs: dict, optional

    Module Stuff:
    :param device: _description_, defaults to None
    :type device: _type_, optional
    :param dtype: _description_, defaults to None
    :type dtype: _type_, optional
    """

    @logger_init
    def __init__(
        self,
        # Linear
        in_features: int,
        out_features: int,
        weight_quant=None,
        weight_quant_bits=8,
        weight_quant_channel_wise=True,
        weight_quant_args=None,
        weight_quant_kargs={},
        # Batch-Norm
        bn_class=None,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        fixed_n: bool = False,
        BN_shift_alpha_function=None,
        # Activation
        activation: Optional[Quant] = None,
        activation_args=None,
        activation_kargs={},
        # General stuff
        device=None,
        dtype=None,
    ) -> None:
        """
        Please see class documentation
        """
        super(LinBnA, self).__init__()
        self.lin = Linear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
            device=device,
            dtype=dtype,
            weight_quant=weight_quant,
            weight_quant_bits=weight_quant_bits,
            weight_quant_channel_wise=weight_quant_channel_wise,
            weight_quant_args=weight_quant_args,
            weight_quant_kargs=weight_quant_kargs,
            out_quant=None,
            out_quant_args=None,
            out_quant_kargs={},
        )

        if bn_class == None:
            bn_class = BatchNorm1d

        self.bn = bn_class(
            num_features=out_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
            fixed_n=fixed_n,
            out_quant=None,
            out_quant_args=None,
            out_quant_kargs={},
            shift_alpha_function=BN_shift_alpha_function,
        )

        if activation_args is None:
            activation_args = [8, (1, out_features)]
        if activation == None:
            self.activation = ReLU(*activation_args, **activation_kargs)
        else:
            self.activation = activation(*activation_args, **activation_kargs)

    def int_extract(
        self, accumulation_type=torch.int32, small_signed_type=torch.int8, small_unsigned_type=torch.uint8
    ):
        return LinBnA_int(
            self.lin.in_features,
            self.lin.out_features,
            self.lin.quant_weight,
            self.bn.n,
            self.bn.t,
            self.activation.min,
            self.activation.max,
            accumulation_type=accumulation_type,
            small_signed_type=small_signed_type,
            small_unsigned_type=small_unsigned_type,
        )
    


    @logger_forward
    def forward(self, x: DataWrapper) -> DataWrapper:

        fact = self.bn.get_weight_factor()
        x = self.lin(x, fact)
        x = self.bn(x, self.activation)

        return x
