import torch
import torch.nn as nn
from torch.nn.common_types import Tensor

from typing import Optional


from ..DataWrapper import DataWrapper
from ..Quantizer import Quant, LinQuantExpScale
from ..logger import logger_forward, logger_init


def mul_pow2(a:torch.Tensor,exp:torch.Tensor):
    mantissa, exponent = torch.frexp(a)
    exponent += exp.type(torch.int)
    return torch.ldexp(mantissa,exponent)


class DownScaler(nn.Module):
    @logger_init
    def __init__(self,
                 size: tuple = (1,),
                 out_quant=None,
                 out_quant_args=None,
                 out_quant_kargs={},) -> None:
        super(DownScaler, self).__init__()

        self.register_buffer('right_shift', torch.zeros(size))
        self.register_buffer('integer_bias', torch.zeros(size))
        self.register_buffer('weight_factor', torch.ones(size))

        if out_quant_args == None:
            out_quant_args = (
                8,
                size,
            )

        if out_quant == None:
            self.out_quant = LinQuantExpScale(*out_quant_args, **out_quant_kargs)
        else:
            self.out_quant = out_quant(*out_quant_args, **out_quant_kargs)

    @logger_forward
    def get_weight_function(self):
        """
        get_weight_factor Returns a function to calculate alpha with a singe value
        """

        def ret_fun(rexp):
            with torch.no_grad():
                # print(delta_I,delta_O,delta_W)
                n = rexp - torch.log2(self.out_quant.delta_out)
                nr = torch.ceil(n)
                self.weight_factor = torch.exp2(n - nr)
                return self.weight_factor

        return ret_fun

    @logger_forward
    def calculate_right_shift(self, delta_input: Tensor, delta_output: Tensor) -> Tensor:
        """
        calculate_right_shift calculates the scaling shift

        :param delta_I: Input scaling factor
        :type delta_I: Tensor
        :param delta_O: Output scaling factor
        :type delta_O: Tensor
        :return: The shift value
        :rtype: Tensor
        """
        with torch.no_grad():
            n = delta_input/ delta_output
            n = torch.log2(n)
            nr = torch.ceil(n)
        return nr

    @logger_forward
    def forward(self, input: DataWrapper, bias:Optional[Tensor] = None, out_quant: Optional[Quant] = None) -> DataWrapper:
        if self.training:
            return self.forward_train(input, bias, out_quant)
        else:
            return self.forward_eval(input, bias, out_quant)
        

    @logger_forward
    def forward_train(self, input: DataWrapper, bias:Optional[Tensor], out_quant: Optional[Quant] = None) -> DataWrapper:
        value, rexp = input.get()
        
        if out_quant != None:
            self.out_quant=out_quant
        
        if bias is not None:
            value += bias
        value = self.out_quant(value,False,input)

        return input.set(value, torch.log2(self.out_quant.delta_out))
    
    @logger_forward
    def forward_eval(self, input: DataWrapper, bias:Optional[Tensor], out_quant: Optional[Quant] = None) -> DataWrapper:
        value, rexp = input.get()
        
        if out_quant != None:
            self.out_quant=out_quant

        with torch.no_grad():
            self.right_shift = self.calculate_right_shift(rexp.exp2(),self.out_quant.delta_out)
            if bias is not None:
                self.integer_bias = torch.round(bias/(self.out_quant.delta_out*self.right_shift.exp2()))

                value = value + self.integer_bias
            value = value*torch.exp2(self.right_shift)
            value = value.floor()
            value = value.clamp(self.out_quant.min, self.out_quant.max)

        return input.set(value, torch.log2(self.out_quant.delta_out))


