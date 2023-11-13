import torch
import torch.nn as nn
from torch.nn.common_types import Tensor

from types import FunctionType
from typing import Tuple

import numpy as np

from ...Quantizer import FakeQuant
from ...linear.weight_quantization import LinQuantWeight

from ...logger import logger_init, logger_forward

from ... import __TESTING_FLAGS__


class LinQuantWeight_mod_MinMSE(LinQuantWeight):
    @logger_init
    def __init__(self, bits: int = 8, size: tuple = (-1,), rounding_mode: str = "round") -> None:
        super(LinQuantWeight_mod_MinMSE,self).__init__(bits, size, rounding_mode)
        self.register_buffer("delta_in_factor", torch.tensor(3.347*np.exp(-0.5739*bits)))
        self.register_buffer("delta_out_factor", torch.tensor(3.347*np.exp(-0.5739*bits)))

        if size == (-1,):
            self.register_buffer("sigma", torch.ones(1))
        else:
            self.register_buffer("sigma", torch.ones(size))
        
    @logger_forward
    def forward(self, x: Tensor, rexp_mean: Tensor, rexp_diff: Tensor, fact_fun: FunctionType) -> Tensor:
        with torch.no_grad():
            if not __TESTING_FLAGS__['FREEZE_QUANT']:
                sigma = (
                    torch.var(x * (rexp_diff.view(1, -1)), self.reduce_list, unbiased=False, keepdim=True).add(1e-5).sqrt()
                )
                self.sigma = sigma

            self.delta_in = self.sigma.mul(self.delta_in_factor)
            self.delta_out = self.sigma.mul(self.delta_in_factor)

            fact = fact_fun((self.delta_out.view(1,-1) * rexp_mean).log2()).view(-1, 1)
            self.delta_for_quant = self.delta_in.div(rexp_diff.view(*self.rexp_view)).div_(fact)

            x.data.clamp_(self.delta_for_quant*(self.min-0.5),
                        self.delta_for_quant*(self.max+0.5))

        return FakeQuant(
                x=x.clone(),
                delta_in=self.delta_for_quant ,
                delta_out=self.delta_for_quant ,
                training=self.training,
                min_quant=self.min,
                max_quant=self.max,
                rounding_mode=self.rounding_mode,
            )