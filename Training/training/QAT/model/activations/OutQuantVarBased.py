from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn.common_types import Tensor

from ..DataWrapper import DataWrapper

from ..logger import logger_init, logger_forward
from ..Quantizer import Quant

from .. import __TESTING_FLAGS__



class OutQant_var_based(Quant):
        @logger_init
        def __init__( 
            self,
            bits: int,
            size = (-1,),
            rounding_mode: str = "floor",
            use_enforced_quant_level: bool = False,
            mom1: float = 0.1,
        ) -> None:
            super(OutQant_var_based,self).__init__(bits, size, rounding_mode, use_enforced_quant_level)
            self.bits = bits
            if size == (-1,):
                self.register_buffer("sigma", torch.ones(1))
            else:
                self.register_buffer("sigma", torch.ones(size))
            self.take_new = True
            self.mom1 = mom1
            self.register_buffer("delta_in_factor", torch.tensor(1/40))
            self.register_buffer("delta_out_factor", torch.tensor(1/40))
            
        @logger_forward
        def forward(self, x, fake: bool = False, metadata = None, * , var = None) :
            if self.training:
                with torch.no_grad():
                    if not __TESTING_FLAGS__['FREEZE_QUANT']:
                        if var is None:
                            sigma = torch.var(x, self.reduce_list, unbiased=False, keepdim=True).add(1e-10).sqrt()
                            if self.take_new:
                                self.take_new = False
                                self.sigma = sigma
                            else:
                                self.sigma.data = (1 - self.mom1) * self.sigma + self.mom1 * sigma
                        else:
                            self.sigma.data = var.view(self.size).abs().add(1e-10).clone()

                    self.delta_in = self.sigma.mul(self.delta_in_factor).detach()
                    self.delta_out = self.sigma.mul(self.delta_in_factor).detach()
        
            with torch.no_grad():
                if self.use_enforced_quant_level and metadata is not None:
                    self.use_quant(metadata)
                if self.use_enforced_quant_level and metadata is None:
                    raise ValueError("Quantization function desired but metadata not passed")

            x = Clip_back_function.apply(x,self.min*self.delta_out,self.max*self.delta_out)

            return super(OutQant_var_based,self).forward(x, fake)




class Clip_back_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, val: Tensor, min: Tensor, max:Tensor) -> Tensor:
        ctx.save_for_backward(torch.logical_and(val>=min,val<=max))
        return val

    @staticmethod
    def backward(ctx, grad_outputs: Tensor):
        cmp, = ctx.saved_tensors
        return grad_outputs*cmp, None, None