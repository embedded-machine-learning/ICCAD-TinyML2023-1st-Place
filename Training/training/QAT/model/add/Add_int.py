import torch
from torch import nn
from torch.nn.common_types import Tensor

from ..logger import logger_init, logger_forward
from .. import __FLAGS__

class Add_int(nn.Module):
    @logger_init
    def __init__(self, a_shift:Tensor, b_shift:Tensor, Act_min:Tensor, Act_max:Tensor) -> None:
        super(Add_int, self).__init__()
        self.register_buffer("a_shift", a_shift)
        self.register_buffer("a_shift_eq_mult", a_shift.exp2())
        self.register_buffer("b_shift", b_shift)
        self.register_buffer("b_shift_eq_mult", b_shift.exp2())

        self.register_buffer("min", Act_min)
        self.register_buffer("max", Act_max)

    @logger_forward
    def forward(self,a:Tensor,b:Tensor)-> Tensor:
        if __FLAGS__['ONNX_EXPORT']:
            a = a.to(torch.float).mul(self.a_shift_eq_mult).floor().to(self.a_shift.dtype)
            b = b.to(torch.float).mul(self.b_shift_eq_mult).floor().to(self.a_shift.dtype)
        else:
            a = torch.bitwise_right_shift(a,-self.a_shift)
            b = torch.bitwise_right_shift(b,-self.b_shift)

        ret = a+b
        ret = ret.clamp(self.min,self.max)
        return ret