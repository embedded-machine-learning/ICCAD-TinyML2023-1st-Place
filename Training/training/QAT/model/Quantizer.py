# Generic Type imports
from typing import Optional, Tuple

# Torch imports
import torch
from torch import nn
from torch.nn.common_types import Tensor

from .DataWrapper import DataWrapper
from .logger import logger_init, logger_forward
from . import __TESTING_FLAGS__


def FakeQuant(
    x: Tensor,
    delta_in: Tensor,
    delta_out: Tensor,
    training: bool,
    min_quant: Tensor,
    max_quant: Tensor,
    rounding_mode: str = "floor",
    clamp: bool = True,
    random: bool = False,
) -> Tensor:
    """
    FakeQuant fake quantizes a Tensor

    fake quantization is quantization, rounding, dequantization

    :param x: The to fake quantize value
    :type x: Tensor
    :param delta_in: The quantization factor
    :type delta_in: Tensor
    :param delta_out: The de-quantization factor
    :type delta_out: Tensor
    :param training: is it training?
    :type training: bool
    :param min_quant: Lower bound of the quantized value
    :type min_quant: Tensor
    :param max_quant: Upper bound of the quantized value
    :type max_quant: Tensor
    :param rounding_mode: The rounding mode, defaults to "floor"
    :type rounding_mode: str, optional
    :return: The fake quantized value, of if training is false the quantized value
    :rtype: Tensor
    """
    # with torch.no_grad():
    #     if training:
    #         if clamp:
    #             x.data.div_(delta_in, rounding_mode=rounding_mode).clamp_(min_quant, max_quant).mul_(delta_out)
    #         else:
    #             x.data.div_(delta_in, rounding_mode=rounding_mode).mul_(delta_out)
    #     else:
    #         x = x.data.div(delta_in, rounding_mode=rounding_mode).clamp_(min_quant, max_quant)
    # return x
    with torch.no_grad():
        if rounding_mode!='floor_div':
            if training:
                x.data.div_(delta_in)
            else:
                x = x.data.div(delta_in)

            if rounding_mode=='floor':
                x.data.floor_()
            elif rounding_mode=='trunc':
                x.data.trunc_()
            else:#rounding_mode=='round':
                x.data.round_()
                # x.data = (x.data + torch.rand_like(x.data)).floor()
        else:
            if training:
                x.data.div_(delta_in,rounding_mode='floor')
            else:
                x = x.data.div(delta_in,rounding_mode='floor')

        if clamp:
            x.data.clamp_(min_quant, max_quant)
        
        if training:
            x.data.mul_(delta_out)
    return x


class Quant(nn.Module):
    """
    Quant Quantization base module

    :param bits: Number of bits that should be used
    :type bits: int
    :param size: The shape of the quantization factors, defaults to (-1,)
    :type size: Tuple[int], optional
    :param rounding_mode: The desired rounding mode, defaults to "floor"
    :type rounding_mode: str, optional
    :param use_enforced_quant_level: Should the metadata quant level function be used?, defaults to False
    :type use_enforced_quant_level: bool, optional
    """

    @logger_init
    def __init__(
        self, bits: int, size: Tuple[int] = (-1,), rounding_mode: str = "floor", use_enforced_quant_level: bool = False
    ) -> None:
        super(Quant, self).__init__()
        self.simple = False
        self.bits = bits
        self.use_enforced_quant_level = use_enforced_quant_level
        if size == (-1,):
            self.register_buffer("delta_in", torch.ones((1,)))
            self.register_buffer("delta_out", torch.ones((1,)))
            self.size = (1,)
            self.simple = True
        else:
            self.register_buffer("delta_in", torch.ones(size))
            self.register_buffer("delta_out", torch.ones(size))
            self.size = size

        self.permute_list = []
        self.reduce_list = []
        self.number_of_dims = 0
        for i, val in enumerate(size):
            # print(size[i])
            if val != 1:
                self.permute_list.insert(0, i)
                self.number_of_dims += 1
            else:
                self.permute_list.append(i)
                self.reduce_list.append(i)

        self.permute_list = tuple(self.permute_list)
        self.rounding_mode = rounding_mode

        self.register_buffer("max", torch.ones(self.size) * (2 ** (self.bits - 1) - 1))
        self.register_buffer("min", torch.ones(self.size) * (-(2 ** (self.bits - 1))))

    # @logger_forward
    # def copy(self, other: "Quant"):
    #     """
    #     copy copies the internal content from other to self

    #     :param other: Another Quant object
    #     :type other: Quant
    #     """
    #     self.delta_in = other.delta_in.clone().detach()
    #     self.delta_out = other.delta_out.clone().detach()
    #     self.min = other.min.clone().detach()
    #     self.max = other.max.clone().detach()
    #     self.rounding_mode = other.rounding_mode

    @logger_forward
    def use_quant(self, metadata: DataWrapper):
        """
        use_quant Uses the factor quantization function of the meta data

        :param metadata: The meta data
        :type metadata: DataWrapper
        """
        self.delta_in = metadata.use_quant(self.delta_in)
        self.delta_out = metadata.use_quant(self.delta_out)

    @logger_forward
    def get_abs(self, x: Tensor) -> Tensor:
        """
        get_abs get the absolute maximum depending on the dimension to be reduced

        :param x: _description_
        :type x: torch.Tensor
        :return: _description_
        :rtype: torch.Tensor
        """
        if self.simple:
            abs_val = x.abs().max()
        else:
            x_reorderd = x.permute(self.permute_list).contiguous()
            x_reorderd = x_reorderd.view((*x_reorderd.shape[: self.number_of_dims], -1))
            abs_val = x_reorderd.abs().max(dim=(self.number_of_dims), keepdim=True).values.view(self.size)
        return abs_val

    @logger_forward
    def forward(self, x: Tensor, fake: bool = False) -> Tensor:
        """
        forward Fake quantizes the value

        if Fake = True do nothing

        :param x: The to quantize value
        :type x: Tensor
        :param fake: Should anything be done?, defaults to False
        :type fake: bool, optional
        :return: The value
        :rtype: Tensor
        """
        if fake:
            return x
        return FakeQuant(
            x=x,
            delta_in=self.delta_in,
            delta_out=self.delta_out,
            training=self.training,
            min_quant=self.min,
            max_quant=self.max,
            rounding_mode=self.rounding_mode,
        )


class LinQuantExpScale(Quant):
    """
    LinQuantExpScale Uses a quantization scheme which uses the max of abs

    :param bits: Number of bits that should be used
    :type bits: int
    :param size: The shape of the quantization factors, defaults to (-1,)
    :type size: Tuple[int], optional
    :param rounding_mode: The desired rounding mode, defaults to "floor"
    :type rounding_mode: str, optional
    :param use_enforced_quant_level: Should the metadata quant level function be used?, defaults to False
    :type use_enforced_quant_level: bool, optional
    :param mom1: The momentum used to update the internal running mean, defaults to 0.1
    :type mom1: float, optional
    """

    @logger_init
    def __init__(
        self,
        bits: int,
        size: Tuple[int] = (-1,),
        rounding_mode: str = "floor",
        use_enforced_quant_level: bool = False,
        mom1: float = 0.1,
    ) -> None:
        super(LinQuantExpScale, self).__init__(bits, size, rounding_mode, use_enforced_quant_level)
        if size == (-1,):
            self.register_buffer("abs", torch.ones(1))
        else:
            self.register_buffer("abs", torch.ones(size))
        self.take_new = True
        self.mom1 = mom1
        assert self.bits > 0
        self.register_buffer("delta_in_factor", torch.tensor(2.0 / (2.0**self.bits - 1)))
        self.register_buffer("delta_out_factor", torch.tensor(2.0 / (2.0**self.bits - 1)))

    @logger_forward
    def forward(self, x: torch.Tensor, fake: bool = False, metadata: Optional[DataWrapper] = None,*args,**kargs):
        if self.training and not __TESTING_FLAGS__['FREEZE_QUANT']:
            with torch.no_grad():
                abs_value = self.get_abs(x)
                # print(abs)
                self.abs = ((1 - self.mom1) * self.abs + self.mom1 * abs_value).detach()

                abs_value = self.abs.log2().ceil().exp2()
                self.delta_in = abs_value.mul(self.delta_in_factor).detach()  # .log2().ceil().exp2()
                self.delta_out = abs_value.mul(self.delta_out_factor).detach()  # .log2().ceil().exp2()
        if self.use_enforced_quant_level and metadata is not None:
            self.use_quant(metadata)
        if self.use_enforced_quant_level and metadata is None:
            raise ValueError("Quantization function desired but metadata not passed")

        return super(LinQuantExpScale, self).forward(x, fake)
