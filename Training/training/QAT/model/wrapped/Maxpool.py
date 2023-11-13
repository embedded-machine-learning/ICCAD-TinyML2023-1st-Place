from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.common_types import _size_any_t, _size_any_opt_t


from ..logger import logger_forward, logger_init
from ..DataWrapper import DataWrapper


class MaxPool2d_int(nn.MaxPool2d):
    """
    MaxPool2d Wrappes torches nn.MaxPool2d

    _extended_summary_

    :param kernel_size: The shape of kernel
    :type kernel_size: _size_any_t
    :param stride: The Stride, defaults to None
    :type stride: Optional[_size_any_t], optional
    :param padding: The padding size, defaults to 0
    :type padding: _size_any_t, optional
    :param dilation: The dilatation, defaults to 1
    :type dilation: _size_any_t, optional
    :param ceil_mode: Should it ceil, defaults to False
    :type ceil_mode: bool, optional
    """

    @logger_init
    def __init__(
        self,
        kernel_size: _size_any_t,
        stride: Optional[_size_any_t] = None,
        padding: _size_any_t = 0,
        dilation: _size_any_t = 1,
        ceil_mode: bool = False,
    ) -> None:
        super(MaxPool2d_int, self).__init__(kernel_size, stride, padding, dilation, False, ceil_mode)
    @logger_forward
    def forward(self, input: Tensor):
        return super(MaxPool2d_int, self).forward(input.to(torch.float)).to(input.dtype)

    def onnx_export(self, node_list, input_name, input_zero_point, idx=0):
        import onnx
        import onnx.numpy_helper
        import onnx.onnx_ml_pb2
        import onnx.helper as hel

        node = onnx.helper.make_node(
            "MaxPool",
            inputs=[input_name],
            outputs=[f'output_{idx}'],
            kernel_shape=[self.kernel_size, self.kernel_size],
            strides=[self.stride, self.stride],
            pads=[self.padding,self.padding,self.padding,self.padding]
        )

        node_list.append(node)
        return node_list, f'output_{idx}', input_zero_point, idx+1


class MaxPool2d(nn.MaxPool2d):
    """
    MaxPool2d Wrappes torches nn.MaxPool2d

    Maxpool stays quantized.

    :param kernel_size: The shape of kernel
    :type kernel_size: _size_any_t
    :param stride: The Stride, defaults to None
    :type stride: Optional[_size_any_t], optional
    :param padding: The padding size, defaults to 0
    :type padding: _size_any_t, optional
    :param dilation: The dilatation, defaults to 1
    :type dilation: _size_any_t, optional
    :param ceil_mode: Should it ceil, defaults to False
    :type ceil_mode: bool, optional
    """

    @logger_init
    def __init__(
        self,
        kernel_size: _size_any_t,
        stride: Optional[_size_any_t] = None,
        padding: _size_any_t = 0,
        dilation: _size_any_t = 1,
        ceil_mode: bool = False,
    ) -> None:
        super(MaxPool2d, self).__init__(kernel_size, stride, padding, dilation, False, ceil_mode)

    def int_extract(self, accumulation_type=torch.int32, small_signed_type=torch.int8, small_unsigned_type=torch.uint8) -> MaxPool2d_int:
        return MaxPool2d_int(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
        )

    @logger_forward
    def forward(self, input: DataWrapper):
        val, rexp = input.get()
        return input.set(
            F.max_pool2d(
                val, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, self.return_indices
            ),
            rexp,
        )


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    """
    AdaptiveAvgPool2d Wrappes nn.AdaptiveAvgPool2d

    **IMPORTANT** it also quantizes it to the current quantization level,
        if implement in HW make sure that this is implemented correctly, or make a custom class

    :param output_size: The desired output shape
    :type output_size: _size_any_opt_t
    """

    @logger_init
    def __init__(self, output_size: _size_any_opt_t) -> None:
        super(AdaptiveAvgPool2d, self).__init__(output_size)

    def forward(self, input: DataWrapper) -> DataWrapper:
        # does nopt modify the channels so simple wrapping and floor should be enough
        val, rexp = input.get()

        val = super(AdaptiveAvgPool2d, self).forward(val)

        if self.training:
            with torch.no_grad():
                val.data = val.data.div_(torch.exp2(rexp), rounding_mode="floor")
                val.data = val.data.mul_(torch.exp2(rexp))

        else:
            val.floor_()

        return input.set(val, rexp)
