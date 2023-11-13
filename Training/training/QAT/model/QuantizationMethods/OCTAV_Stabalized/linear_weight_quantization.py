# https://proceedings.mlr.press/v162/sakr22a/sakr22a.pdf
# is the base, added an EMA filter on top to stabilize,
#  also increased number of iteration 

import torch
import torch.nn as nn
from torch.nn.common_types import Tensor

from types import FunctionType
from typing import Tuple

from ...Quantizer import FakeQuant
from ...linear.weight_quantization import LinQuantWeight
from ...logger import logger_init, logger_forward



class LinQuantWeight_mod_OCTAV_Stabalized(LinQuantWeight):
    @logger_init
    def __init__(self, bits: int = 8, size: tuple = (-1,), rounding_mode: str = "round") -> None:
        super(LinQuantWeight_mod_OCTAV_Stabalized,self).__init__(bits, size, rounding_mode)
        self.register_buffer('s', torch.ones(size))

    @logger_forward
    def s_it(self, x):
        with torch.no_grad():
            x_a = x.abs()
            gr_s = x_a > self.s
            gr_z = x_a > 0

            return ((x_a*gr_s).sum(self.reduce_list, keepdim=True))/((4**(-self.bits)/3)*(gr_z & ~gr_s).sum(self.reduce_list, keepdim=True) + gr_s.sum(self.reduce_list, keepdim=True))

    @logger_forward
    def forward(self, x: Tensor, rexp_mean: Tensor, rexp_diff: Tensor, fact_fun: FunctionType) -> Tensor:
        with torch.no_grad():
            x_d = x * (rexp_diff.view(*self.rexp_view))
            new_s = self.s_it(x_d)
            counter = 0
            mom = 0.9
            if self.training:
                while ((new_s-self.s).abs()/(new_s.abs())>1e-3).any():     # iterate until relative distance is less than 1e-5
                    # print(self.s.view(-1)[:5])
                    self.s = (1-mom)*new_s+mom*self.s
                    new_s =self.s_it(x_d)
                    counter += 1
                    if counter > 1000:
                        print("OCTAV counter overflow exiting Conv", new_s[(new_s-self.s).abs()/(self.s.abs())>1e-5].view(-1)) 
                        break
            # self.s = new_s
            # print(counter)
            
            self.delta_in = self.s/(2**(self.bits-1))
            self.delta_out.data = self.delta_in

            fact = fact_fun((self.delta_out.view(1,-1) * rexp_mean).log2()).view(-1, 1)
            self.delta_for_quant = self.delta_in.div(rexp_diff.view(*self.rexp_view)).div_(fact)

        return FakeQuant(
                x=x.clone(),
                delta_in=self.delta_for_quant ,
                delta_out=self.delta_for_quant ,
                training=self.training,
                min_quant=self.min,
                max_quant=self.max,
                rounding_mode=self.rounding_mode,
            )
