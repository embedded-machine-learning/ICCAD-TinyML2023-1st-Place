from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn.common_types import Tensor

from ..DataWrapper import DataWrapper

from ..logger import logger_init, logger_forward
from ..Quantizer import Quant

from .. import __TESTING_FLAGS__

class PACT(Quant):
    """
    PACT The implementation of the PACT activation function

    This is the implementation of the PACT activation function from `https://openreview.net/forum?id=By5ugjyCb`

    :param size: The shape for alpha, defaults to (1,)
    :type size: tuple, optional
    """

    @logger_init
    def __init__(self, bits, size=(-1,), rounding_mode: str = "floor", use_enforced_quant_level: bool = False) -> None:
        super(PACT, self).__init__(bits, size, rounding_mode, use_enforced_quant_level)
        self.bits = bits
        assert self.bits > 0
        self.register_buffer("delta_in_factor", torch.tensor(1.0 / (2.0**self.bits - 1)))
        self.register_buffer("delta_out_factor", torch.tensor(1.0 / (2.0**self.bits - 1)))

        self.register_parameter("alpha", torch.nn.Parameter(6 * torch.ones(size)))

        nn.init.constant_(self.min, 0)
        nn.init.constant_(self.max, 2**bits - 1)

        self.register_buffer("max_helper", (2**bits - 1) * torch.ones_like(self.max))

    @logger_forward
    def forward(self, x: torch.Tensor, fake: bool = False, metadata: Optional[DataWrapper] = None,*args,**kargs):
        if self.training:
            with torch.no_grad():
                self.alpha.data.clamp_(min=1e-3)    # block 2 small and negative alpha
                
                if __TESTING_FLAGS__['FREEZE_QUANT'] or __TESTING_FLAGS__['FREEZE_ACT_QUANT']:
                    self.alpha.requires_grad_(False)
                # abs = self.alpha_used.log2().ceil().exp2()
                # self.delta_in = self.alpha.mul(self.delta_in_factor).detach()  # .log2().ceil().exp2()
                # self.delta_out = self.alpha.mul(self.delta_out_factor).detach()  # .log2().ceil().exp2()
                self.delta_in = self.alpha.mul(self.delta_in_factor).detach()  # .log2().ceil().exp2()
                self.delta_out = self.alpha.mul(self.delta_out_factor).detach()  # .log2().ceil().exp2()
                if self.use_enforced_quant_level and metadata is not None:
                    self.use_quant(metadata)
                if self.use_enforced_quant_level and metadata is None:
                    raise ValueError("Quantization function desired but metadata not passed")
                

                self.max = self.alpha.div(self.delta_in).round().clamp(self.min,self.max_helper)

            x = PACT_back_function.apply(x, self.alpha)
        return super(PACT,self).forward(x, fake)


class PACT_back_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, val: Tensor, alpha: Tensor) -> Tensor:
        ctx.save_for_backward(val >= alpha, val >= 0)
        return val.clone()

    @staticmethod
    def backward(ctx, grad_outputs: Tensor) -> Tuple[Tensor, Tensor]:
        alpha_cmp, zero_cmp = ctx.saved_tensors
        val_gard = grad_outputs * torch.logical_and(zero_cmp,~alpha_cmp)
        alpha_grad = grad_outputs * alpha_cmp
        return val_gard, alpha_grad
