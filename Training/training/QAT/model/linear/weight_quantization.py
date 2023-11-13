from types import FunctionType
from typing import Tuple

import torch
from torch.nn.common_types import Tensor


from ..Quantizer import Quant,FakeQuant
from ..logger import logger_init,logger_forward

from .. import __TESTING_FLAGS__


class LinQuantWeight(Quant):
    """
    LinQuantWeight Specialized Quantization for Linear Weights

    The special difference to the normal quantization methods is
        that it takes the exponent of the input and a function for
        the weight scaling, which does not modify the quantization
        factor.

    :param bits: Number of bits for quantization, defaults to 8
    :type bits: int, optional
    :param size: The shape of the quantization factor (-1,) would be 1 number (<channels>,1) would be a channel-wise quantization, defaults to (-1,)
    :type size: tuple, optional
    :param rounding_mode: Sets how the values are rounded `https://pytorch.org/docs/stable/generated/torch.div.html`, defaults to "round"
    :type rounding_mode: str, optional
    """
    @logger_init
    def __init__(self, bits: int = 8, size: tuple = (-1,), rounding_mode: str = "round") -> None:
        """
        Please see class documentation `LinQuantWeight`
        """
        super(LinQuantWeight, self).__init__(bits, size, rounding_mode)

        self.bits = bits

        self.rexp_view=(1,-1)

        if size == (-1,):
            self.register_buffer("abs", torch.ones(1))
        else:
            self.register_buffer("abs", torch.ones(size))


        assert self.bits > 0
        self.register_buffer("delta_in_factor", torch.tensor(2.0 / (2.0**self.bits)))
        self.register_buffer("delta_out_factor", torch.tensor(2.0 / (2.0**self.bits - 2)))

        self.register_buffer("max", torch.tensor(2 ** (self.bits - 1) - 1))
        self.register_buffer("min", torch.tensor(-(2 ** (self.bits - 1) - 1)))

    @logger_forward
    def forward(self, x: Tensor, rexp_mean: Tensor, rexp_diff: Tensor, fact_fun: FunctionType) -> Tensor:
        """
        forward Does the quantization, if :cvar:`self.training` returns floats else ints

        Calculates the quantization factors and a scaling factor defined by the passed function.

        :param x: The weights to quantize
        :type x: Tensor
        :param rexp_mean: Mean of the exponent
        :type rexp_mean: Tensor
        :param rexp_diff: Difference of the individual exponents compared to the mean
        :type rexp_diff: Tensor
        :param fact_fun: A function taking one value to calculate a scaling factor to force a following shift operation to a whole number
        :type fact_fun: FunctionType
        :return: Returns the Quantized weights and the scaling factor for debug purposes
        :rtype: tuple[Tensor,Tensor]
        """
        with torch.no_grad():
            if not __TESTING_FLAGS__['FREEZE_QUANT']:
                abs_val = self.get_abs(x * (rexp_diff.view(1, -1)))

                self.abs = abs_val.detach()
            self.delta_in = self.abs.mul(self.delta_in_factor).detach()
            self.delta_out = self.abs.mul(self.delta_out_factor).detach()

            fact = fact_fun((self.delta_out.view(1,-1) * rexp_mean).log2()).view(-1, 1)

        return FakeQuant(
                x=x.clone(),
                delta_in=self.delta_in / ((rexp_diff.view(1, -1) * fact)),
                delta_out=self.delta_out / ((rexp_diff.view(1, -1) * fact)),
                training=self.training,
                min_quant=self.min,
                max_quant=self.max,
                rounding_mode=self.rounding_mode,
            )
