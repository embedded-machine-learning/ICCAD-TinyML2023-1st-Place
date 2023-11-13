from typing import Union, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn

from ..DataWrapper import DataWrapper

from ..logger import logger_forward, logger_init
from ..Quantizer import LinQuantExpScale, Quant

from .. import __HIGH_PRES__


class Add(nn.Module):
    """
    AddQAT Adds 2 numbers

    there is an internal scaling and the required shift operations are being calculated

    :param num_features: number of features
    :type num_features: int
    :param out_quant:  A callable object which overrides the default output quantization, gets called with (values) , defaults to None
    :type out_quant: _type_, optional
    :param out_quant_args:  Overrides arguments for the out quantization initializer with custom ones, defaults to None
    :type out_quant_args: _type_, optional
    :param out_quant_kargs: Passes named arguments to the initializer of the out quantization class, defaults to {}
    :type out_quant_kargs: dict, optional
    """

    @logger_init
    def __init__(
        self,
        size=(1,),
        out_quant=None,
        out_quant_args=None,
        out_quant_kargs={},
    ) -> None:
        super(Add, self).__init__()

        self.register_buffer("a_shift", torch.zeros(size))
        self.register_buffer("b_shift", torch.zeros(size))

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
    def forward(self, in_a: DataWrapper, in_b: DataWrapper, activation: Union[None, nn.Module] = None) -> DataWrapper:
        a = in_a.get()
        b = in_b.get()

        if activation != None:
            self.out_quant.copy(activation)
            quant = activation
        else:
            quant = self.out_quant

        if a[0].shape != b[0].shape:
            raise torch.ErrorReport("ADD: input shapes not identical", a[0].shape, b[0].shape)
        if self.training:
            out = a[0] + b[0]
            out = quant(out, False, in_a)
            rexp = quant.delta_out.log2()
        else:
            rexp = quant.delta_out.log2()
            self.a_shift = (a[1] - rexp).detach().round()
            self.b_shift = (b[1] - rexp).detach().round()
            va = a[0].mul(self.a_shift.exp2())
            vb = b[0].mul(self.b_shift.exp2())
            out = va + vb
            out = out.floor().clamp(quant.min, quant.max)

        return in_a.set(out, rexp)










class Hidden_ReLU(Quant):
    """
    ReLU The implementation of the ReLU activation function fused into the quantization

    :param size: The shape for alpha, defaults to (1,)
    :type size: tuple, optional
    """

    def __init__(self, bits, size=(-1,), rounding_mode: str = "floor", use_enforced_quant_level: bool = False, in_min_shifts = [0,0]) -> None:
        super(Hidden_ReLU, self).__init__(bits, size, rounding_mode, use_enforced_quant_level)
        self.bits = bits

        self.in_min_shifts = in_min_shifts

        nn.init.constant_(self.min, 0)
        nn.init.constant_(self.max, 2**bits - 1)


    def set_quant(self, value1, value2):
        with torch.no_grad():
            exp1,exp2 = value1.exp2(), value2.exp2()
            exp = exp1/(2-self.in_min_shifts[0]) + exp2/(2-self.in_min_shifts[1])
            exp = exp.div(exp1).log2().clip(min=self.in_min_shifts[0]).round().exp2().mul(exp1)
            exp = exp.div(exp2).log2().clip(min=self.in_min_shifts[1]).round().exp2().mul(exp2)
            self.delta_in = exp
            self.delta_out = exp

    def forward(self, x: torch.Tensor, fake: bool = False, metadata: Optional[DataWrapper] = None):
        x = RELUM_back_function.apply(x,self.max*self.delta_out)
        return super(Hidden_ReLU,self).forward(x, fake)


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


class RELUM_back_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, val: Tensor, max: Tensor) -> Tensor:
        ctx.save_for_backward(torch.logical_and((val >= 0),~(val >= max)))
        return val.clone()

    @staticmethod
    def backward(ctx, grad_outputs: Tensor) -> Tuple[Tensor, Tensor]:
        cmp, = ctx.saved_tensors
        val_gard = grad_outputs * cmp
        # alpha_grad = grad_outputs * alpha_cmp
        return val_gard, None


class AddRELU(nn.Module):
    """
    AddQAT Adds 2 numbers

    there is an internal scaling and the required shift operations are being calculated

    :param num_features: number of features
    :type num_features: int
    :param out_quant:  A callable object which overrides the default output quantization, gets called with (values) , defaults to None
    :type out_quant: _type_, optional
    :param out_quant_args:  Overrides arguments for the out quantization initializer with custom ones, defaults to None
    :type out_quant_args: _type_, optional
    :param out_quant_kargs: Passes named arguments to the initializer of the out quantization class, defaults to {}
    :type out_quant_kargs: dict, optional
    """

    @logger_init
    def __init__(
        self,
        size=(1,),
        out_quant=None,
        out_quant_args=None,
        out_quant_kargs={},
        in_min_shifts=[0,0],
    ) -> None:
        super(AddRELU, self).__init__()

        self.register_buffer("a_shift", torch.zeros(size))
        self.register_buffer("b_shift", torch.zeros(size))

        if out_quant_args is None:
            out_quant_args = (
                8,
                size,
                'floor'
            )

        self.out_quant = Hidden_ReLU(*out_quant_args,in_min_shifts=in_min_shifts, **out_quant_kargs)
        
    @logger_forward
    def forward(self, in_a: DataWrapper, in_b: DataWrapper, activation: Union[None, nn.Module] = None) -> DataWrapper:
        a = in_a.get()
        b = in_b.get()

        quant = self.out_quant

        if a[0].shape != b[0].shape:
            raise torch.ErrorReport("ADD: input shapes not identical", a[0].shape, b[0].shape)
        if self.training:
            out = a[0] + b[0]
            quant.set_quant(a[1], b[1])
            out = quant(out, False, in_a)

            rexp = quant.delta_out.log2()
            with torch.no_grad():
                self.a_shift = (a[1] - rexp).detach().round()
                self.b_shift = (b[1] - rexp).detach().round()
        else:
            rexp = quant.delta_out.log2()
            self.a_shift = (a[1] - rexp).detach().round()
            self.b_shift = (b[1] - rexp).detach().round()
            va = a[0].mul(self.a_shift.exp2())
            vb = b[0].mul(self.b_shift.exp2())
            out = va + vb
            out = out.floor().clamp(quant.min, quant.max)

        return in_a.set(out, rexp)
