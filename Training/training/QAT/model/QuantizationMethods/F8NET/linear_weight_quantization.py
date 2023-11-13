# https://arxiv.org/pdf/2202.05239.pdf
# modified

import torch
import torch.nn as nn
from torch.nn.common_types import Tensor

from types import FunctionType
from typing import Tuple

from ...Quantizer import FakeQuant
from ...linear.weight_quantization import LinQuantWeight

from ...logger import logger_init, logger_forward



class LinQuantWeight_mod_F8NET(LinQuantWeight):
    @logger_init
    def __init__(self, bits: int = 8, size: tuple = (-1,), rounding_mode: str = "round") -> None:
        super(LinQuantWeight_mod_F8NET,self).__init__(bits, size, rounding_mode)
        self.register_buffer("delta_in_factor", torch.tensor(1.0 / 40.0))
        self.register_buffer("delta_out_factor", torch.tensor(1.0 / 40.0))
        
    @logger_forward
    def forward(self, x: Tensor, rexp_mean: Tensor, rexp_diff: Tensor, fact_fun: FunctionType) -> Tensor:
        with torch.no_grad():
            sigma = (
                torch.var(x * (rexp_diff.view(1, -1)), self.reduce_list, unbiased=False, keepdim=True).add(1e-5).sqrt()
            )

            self.delta_in = sigma.mul(self.delta_in_factor)
            self.delta_out = sigma.mul(self.delta_in_factor)

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
