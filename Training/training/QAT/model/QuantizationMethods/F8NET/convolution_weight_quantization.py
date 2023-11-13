# https://arxiv.org/pdf/2202.05239.pdf
# modified

import torch
from torch.nn.common_types import Tensor

from types import FunctionType
from typing import Tuple

from ...Quantizer import FakeQuant


from ...convolution.weight_quantization import LinQuantWeight 

from ...logger import logger_init, logger_forward
from ... import __TESTING_FLAGS__


class LinQuantWeight_mod_F8NET(LinQuantWeight):
    @logger_init
    def __init__(self, bits: int = 8, size: tuple = (-1,), rounding_mode: str = "round", layer_wise=False) -> None:
        super().__init__(bits, size, rounding_mode,layer_wise)
        self.register_buffer("delta_in_factor", torch.tensor(1.0 / 40.0))
        self.register_buffer("delta_out_factor", torch.tensor(1.0 / 40.0))

        if size == (-1,):
            self.register_buffer("sigma", torch.ones(1))
        else:
            self.register_buffer("sigma", torch.ones(size))

    @logger_forward
    def forward(self, x: Tensor, rexp_mean: Tensor, rexp_diff: Tensor, fact_fun: FunctionType) -> Tensor:
        with torch.no_grad():
            if not __TESTING_FLAGS__['FREEZE_QUANT']:
                sigma = (
                    torch.var(x * (rexp_diff.view(*self.rexp_view)), self.reduce_list, unbiased=False, keepdim=True)
                    .add_(1e-20)
                    .sqrt_()
                )
                self.sigma.data = sigma.clone()

            self.delta_in = self.sigma.mul_(self.delta_in_factor)  # delta in and delta out identical
            self.delta_out.data = self.delta_in

            fact = fact_fun((self.delta_out.view(1,-1,1,1) * rexp_mean).log2()).view(-1, 1, 1, 1)

            self.delta_for_quant = self.delta_in.div(rexp_diff.view(*self.rexp_view)).div_(fact)
       
            # clipping the weights, improves performance
            x.data.clamp_(self.delta_for_quant*(self.min-0.5),
                          self.delta_for_quant*(self.max+0.5))
          
        return FakeQuant(
                x=x.clone(),
                delta_in=self.delta_for_quant,
                delta_out=self.delta_for_quant,
                training=self.training,
                min_quant=self.min,
                max_quant=self.max,
                rounding_mode=self.rounding_mode,
            )