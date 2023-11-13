from typing import Tuple
from typing import Optional

import torch
from torch import nn
from torch.nn.common_types import Tensor

from ..Quantizer import Quant
from ..DataWrapper import DataWrapper
from ..logger import logger_init, logger_forward

class FixPoint(Quant):
    """
    FixPoint The implementation of a fixpoint conversion

    :param size: The shape for the quantization, defaults to (1,)
    :type size: tuple, optional
    """
    @logger_init
    def __init__(self, bits, size=(-1,), rounding_mode: str = "floor", use_enforced_quant_level: bool = False, fixpoint=4) -> None:
        super(FixPoint, self).__init__(bits, size, rounding_mode, use_enforced_quant_level)
        self.bits = bits
        self.take_new = True
        self.fixpoint = fixpoint
        assert self.bits > 0
        assert use_enforced_quant_level is False        

        nn.init.constant_(self.min, -2**(bits-1))
        nn.init.constant_(self.max, 2**(bits-1) - 1)
        ra = 2**(bits-fixpoint-1)
        self.delta_in = ra/(2**(bits-1)-1)*torch.ones_like(self.delta_in)
        self.delta_out = ra/(2**(bits-1)-1)*torch.ones_like(self.delta_in)


        self.register_buffer('min_float',self.min*2**(-fixpoint))
        self.register_buffer('max_float',self.max*2**(-fixpoint))

    @logger_forward
    def forward(self, x: torch.Tensor, fake: bool = False, metadata: Optional[DataWrapper] = None):
        if self.training and fake:
            x = x.clamp(self.min_float,self.max_float)

        return super(FixPoint,self).forward(x, fake)