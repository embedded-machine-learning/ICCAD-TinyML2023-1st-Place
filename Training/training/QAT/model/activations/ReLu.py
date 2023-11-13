from typing import Tuple
from typing import Optional

import torch
from torch import nn
from torch.nn.common_types import Tensor

from ..Quantizer import Quant
from ..DataWrapper import DataWrapper


class ReLU(Quant):
    """
    ReLU The implementation of the ReLU activation function fused into the quantization

    :param size: The shape for alpha, defaults to (1,)
    :type size: tuple, optional
    """

    def __init__(self, bits, size=(-1,), rounding_mode: str = "floor", use_enforced_quant_level: bool = False, mom1: int  = 0.1) -> None:
        super(ReLU, self).__init__(bits, size, rounding_mode, use_enforced_quant_level)
        self.bits = bits
        if size == (-1,):
            self.register_buffer("sigma", torch.ones(1))
        else:
            self.register_buffer("sigma", torch.ones(size))
        self.take_new = True
        self.mom1 = mom1
        assert self.bits > 0
        # as defined by f8net
        self.register_buffer("delta_in_factor", torch.tensor(1.0 / 70.0))
        self.register_buffer("delta_out_factor", torch.tensor(1.0 / 70.0))

        nn.init.constant_(self.min, 0)
        nn.init.constant_(self.max, 2**bits - 1)

    def forward(self, x: torch.Tensor, fake: bool = False, metadata: Optional[DataWrapper] = None,*args,**kargs):
        if self.training:
            with torch.no_grad():
                if len(self.size) != len(x.shape):
                    self.size = self.size + [1 for x in range(len(x.shape) - len(self.size))]
                    super(ReLU, self).__init__(self.bits, self.size, self.rounding_mode)
                    print("mismatch in input and definition found, loading wont be possible until fixed")

                sigma = torch.var(x, self.reduce_list, unbiased=False, keepdim=True).add(1e-5).sqrt()
                if self.take_new:
                    self.take_new = False
                    self.sigma = sigma
                else:
                    self.sigma = (1 - self.mom1) * self.sigma + self.mom1 * sigma

                self.delta_in = sigma.mul(self.delta_in_factor).detach()
                self.delta_out = sigma.mul(self.delta_in_factor).detach()

                if self.use_enforced_quant_level and metadata is not None:
                    self.use_quant(metadata)
                if self.use_enforced_quant_level and metadata is None:
                    raise ValueError("Quantization function desired but metadata not passed")

            x = RELU_back_function.apply(x)
        return super(ReLU,self).forward(x, fake)


class RELU_back_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, val: Tensor) -> Tensor:
        ctx.save_for_backward(val >= 0)
        return val.clone()

    @staticmethod
    def backward(ctx, grad_outputs: Tensor) -> Tuple[Tensor, Tensor]:
        (zero_cmp,) = ctx.saved_tensors
        val_gard = grad_outputs * zero_cmp
        return val_gard
