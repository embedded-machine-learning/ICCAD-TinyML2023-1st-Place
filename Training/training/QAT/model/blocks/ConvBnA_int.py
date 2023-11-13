from typing import Union

import torch
from torch import nn
from torch.nn.common_types import Tensor, _size_2_t

import numpy as np

from ..logger import logger_forward, logger_init
from .. import __FLAGS__


class ConvBnA_int(nn.Module):
    """
    ConvBnA_int A integer converted ConvBnA block

    Is not meant for training only for validation of the trained results

    :param Conv_weight: _description_
    :type Conv_weight: _type_
    :param BN_shift: _description_
    :type BN_shift: _type_
    :param BN_add: _description_
    :type BN_add: _type_
    :param Act_min: _description_
    :type Act_min: _type_
    :param Act_max: _description_
    :type Act_max: _type_
    """

    @logger_init
    def __init__(
        self,
        # Convolution definition
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t,
        padding: Union[str, _size_2_t],
        dilation: _size_2_t,
        groups: int,
        # transferred Convolution parameter
        Conv_weight: Tensor,
        BN_shift: Tensor,
        BN_add: Tensor,
        Act_min: Tensor,
        Act_max: Tensor,
        # type information
        accumulation_type=torch.int32,
        small_signed_type=torch.int8,
        small_unsigned_type=torch.uint8,
    ) -> None:
        super(ConvBnA_int, self).__init__()
        self.Conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.Conv.weight.requires_grad_(False)
        self.Conv.weight.data = Conv_weight.to(small_signed_type)

        self.register_buffer("n", BN_shift.int())
        self.register_buffer("n_eq_mult", BN_shift.exp2())

        self.register_buffer("t", BN_add.to(accumulation_type))
        self.register_buffer("min", Act_min.to(accumulation_type))
        self.register_buffer("max", Act_max.to(accumulation_type))

        self.pure_positive = torch.all(Act_min >= 0).cpu().numpy()

        self.accumulation_type = accumulation_type
        self.small_signed_type = small_signed_type
        self.small_unsigned_type = small_unsigned_type

    @logger_forward
    def forward(self, x: Tensor) -> Tensor:
        if __FLAGS__["ONNX_EXPORT"]:
            x = self.Conv._conv_forward(
                x.float(), self.Conv.weight.float(), self.Conv.bias).to(self.accumulation_type)
        else:
            x = self.Conv._conv_forward(x.to(self.accumulation_type), self.Conv.weight.to(
                self.accumulation_type), self.Conv.bias)

        x = x + self.t

        if __FLAGS__["ONNX_EXPORT"]:
            x = x.to(torch.float).mul(self.n_eq_mult).floor().to(self.n.dtype)
        else:
            x = torch.bitwise_right_shift(x, -self.n)

        x = x.clamp(self.min, self.max)
        if self.pure_positive:
            x = x.to(self.small_unsigned_type)
        else:
            x = x.to(self.small_signed_type)
        return x

    def onnx_export_old(self, node_list, input_name, input_zero_point, idx=0):
        import onnx
        import onnx.numpy_helper
        import onnx.onnx_ml_pb2
        import onnx.helper as hel

        eq_mul = self.n_eq_mult.cpu().detach().view(-1).numpy()
        y_scale = (np.max(1/eq_mul)).astype(np.float32)
        w_scale = (eq_mul*y_scale).astype(np.float32)
        w_zero_point = (np.int8(0)*np.ones_like(w_scale)).astype(np.int8)

        B = (self.t.cpu().detach().view(-1).numpy())/w_scale

        conversions = {
            f'w_{idx}': (self.Conv.weight.cpu().detach().numpy()).astype(np.int8),
            f'x_scale_{idx}': np.float32(1),
            f'w_scale_{idx}': w_scale,
            f'y_scale_{idx}': y_scale,
            f'x_zero_point_{idx}': input_zero_point,
            f'w_zero_point_{idx}': w_zero_point,
            f'y_zero_point_{idx}': np.uint8(0) if self.pure_positive else np.uint8(128),
            f'B_{idx}': B.astype(np.int32).reshape(-1),
            f'min_{idx}': (self.min.cpu().detach().numpy() + (np.uint8(0) if self.pure_positive else np.uint8(128))).astype(np.uint8),
            f'max_{idx}': (self.max.cpu().detach().numpy() + (np.uint8(0) if self.pure_positive else np.uint8(128))).astype(np.uint8),
        }

        this_node_list = []       # just here to debug

        for name, value in conversions.items():
            value2 = onnx.numpy_helper.from_array(value)
            value2: onnx.onnx_ml_pb2.TensorProto
            # print(value2)
            this_node_list.append(
                hel.make_node('Constant', inputs=[], outputs=[name], name=(name + '_const'),
                              value=hel.make_tensor(name=name + '_1',
                                                    data_type=value2.data_type,
                                                    dims=value.shape,
                                                    vals=value.flatten()))
            )
        pads = [0,0,0,0]
        if self.Conv.padding.lower() == 'same':
            pads = [self.Conv.kernel_size[_%2]//2 for _ in range(4)]

        node = hel.make_node(
            "QLinearConv",
            name=f'QLinearConv_{idx}',
            pads=pads,
            strides=list(self.Conv.stride),
            inputs=[
                input_name,
                f'x_scale_{idx}',
                f'x_zero_point_{idx}',
                f'w_{idx}',
                f'w_scale_{idx}',
                f'w_zero_point_{idx}',
                f'y_scale_{idx}',
                f'y_zero_point_{idx}',
                f'B_{idx}',
            ],
            # outputs=[f'output_{idx}'],
            outputs=[f'internal_output_2_{idx}'],
            # outputs=[f'internal_output_1_{idx}'],
        )

        this_node_list.append(node)

        # node = onnx.helper.make_node(
        #     "Max",
        #     name=f'Max_{idx}',
        #     inputs=[f'internal_output_1_{idx}', f'min_{idx}'],
        #     outputs=[f'internal_output_2_{idx}'],
        # )
        # this_node_list.append(node)
        
        node = hel.make_node(
            "Min",
            name=f'Min_{idx}',
            inputs=[f'max_{idx}', f'internal_output_2_{idx}'],
            outputs=[f'output_{idx}'],
        )
        this_node_list.append(node)

        node_list = node_list + this_node_list
        return node_list, f'output_{idx}', np.uint8(0) if self.pure_positive else np.uint8(128), idx+1
    

    def onnx_export(self, node_list, input_name, input_zero_point, idx=0):
        import onnx
        import onnx.numpy_helper
        import onnx.onnx_ml_pb2
        import onnx.helper as hel

        eq_mul = self.n_eq_mult.cpu().detach().view(-1).numpy()
        # y_scale = (np.max(1/eq_mul)).astype(np.float32)
        y_scale = ((1/eq_mul)).astype(np.float32)
        # w_scale = (eq_mul*y_scale).astype(np.float32)
        w_scale = np.float32(1)
        w_zero_point = (np.int8(0)*np.ones_like(w_scale)).astype(np.int8)

        B = (self.t.cpu().detach().view(-1).numpy().astype(np.float32))/w_scale

        conversions = {
            f'w_{idx}': (self.Conv.weight.cpu().detach().numpy()).astype(np.int8),
            f'x_scale_{idx}': np.float32(1),
            f'w_scale_{idx}': w_scale,
            f'y_scale_{idx}': y_scale,
            f'x_zero_point_{idx}': input_zero_point,
            f'w_zero_point_{idx}': w_zero_point,
            f'y_zero_point_{idx}': np.zeros_like(y_scale).astype(np.int8 if self.pure_positive else np.int8),
            f'B_{idx}': B.astype(np.float32),
            f'B_scale_{idx}': (y_scale/w_scale).astype(np.float32),
            f'B_zero_point_{idx}': np.zeros_like(w_scale).astype(np.int8),
            f'min_{idx}': (self.min.cpu().detach().numpy()*y_scale).astype(np.float32),
            f'max_{idx}': (self.max.cpu().detach().numpy()*y_scale).astype(np.float32),
            f'max_scale_{idx}': np.float32(1),
            f'max_zero_point_{idx}': np.uint8(0).astype(np.uint8 if self.pure_positive else np.int8),
        }

        this_node_list = []       # just here to debug

        for name, value in conversions.items():
            value2 = onnx.numpy_helper.from_array(value)
            value2: onnx.onnx_ml_pb2.TensorProto
            # print(value2)
            this_node_list.append(
                hel.make_node('Constant', inputs=[], outputs=[name], name=(name + '_const'),
                              value=hel.make_tensor(name=name + '_1',
                                                    data_type=value2.data_type,
                                                    dims=value.shape,
                                                    vals=value.flatten()))
            )
        pads = [0,0,0,0]
        if self.Conv.padding.lower() == 'same':
            pads = [self.Conv.kernel_size[_%2]//2 for _ in range(4)]


        
        node = hel.make_node(
            "DequantizeLinear",
            inputs=[input_name, f'x_scale_{idx}', f'x_zero_point_{idx}'],
            axis=1,
            outputs=[f'input_{idx}_f'],
        )
        this_node_list.append(node)



        node = hel.make_node(
            "DequantizeLinear",
            inputs=[f'w_{idx}', f'w_scale_{idx}', f'w_zero_point_{idx}'],
            axis=0,
            outputs=[f'w_{idx}_f'],
        )
        this_node_list.append(node)

        # node = hel.make_node(
        #     "DequantizeLinear",
        #     inputs=[f'B_{idx}', f'B_scale_{idx}', f'B_zero_point_{idx}'],
        #     axis=1,
        #     outputs=[f'B_{idx}_f'],
        # )
        # this_node_list.append(node)

        node = hel.make_node(
            "Conv",
            inputs=[f'input_{idx}_f', f'w_{idx}_f', f'B_{idx}'],
            # outputs=[f'internal_output_1_{idx}_f'],
            outputs=[f'output_{idx}_f'],
            pads=pads,
            strides=list(self.Conv.stride),
            kernel_shape=list(self.Conv.kernel_size),
            # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
        )

        this_node_list.append(node)


        # node = onnx.helper.make_node(
        #     "Max",
        #     name=f'Max_{idx}',
        #     inputs=[f'internal_output_1_{idx}_f', f'min_{idx}'],
        #     outputs=[f'internal_output_2_{idx}_f'],
        # )
        # this_node_list.append(node)

        # node = hel.make_node(
        #     "Min",
        #     name=f'Min_{idx}',
        #     inputs=[f'internal_output_2_{idx}_f',f'max_{idx}'],
        #     outputs=[f'output_{idx}_f'],
        # )
        # this_node_list.append(node)


        node = hel.make_node(
            "QuantizeLinear",
            inputs=[f'output_{idx}_f', f'y_scale_{idx}', f'y_zero_point_{idx}'],
            # inputs=[f'internal_output_1_{idx}_f', f'y_scale_{idx}', f'y_zero_point_{idx}'],
            axis=1,
            outputs=[f'output_{idx}'],
            # outputs=[f'internal_output_1_{idx}'],
        )
        this_node_list.append(node)

        # node = hel.make_node(
        #     "DequantizeLinear",
        #     inputs=[f'internal_output_1_{idx}', f'max_scale_{idx}', f'max_zero_point_{idx}'],
        #     outputs=[f'internal_output_1_{idx}_f_2'],
        # )
        # this_node_list.append(node)

        # # node = hel.make_node(
        # #     "DequantizeLinear",
        # #     inputs=[f'min_{idx}', f'max_scale_{idx}', f'max_zero_point_{idx}'],
        # #     outputs=[f'min_{idx}_f'],
        # # )
        # # this_node_list.append(node)

        # node = hel.make_node(
        #     "QuantizeLinear",
        #     inputs=[f'internal_output_2_{idx}_f',  f'max_scale_{idx}',f'y_zero_point_{idx}'],
        #     # outputs=[f'output_{idx}'],
        #     outputs=[f'internal_output_2_{idx}'],
        # )
        # this_node_list.append(node)



        # node = hel.make_node(
        #     "DequantizeLinear",
        #     inputs=[f'internal_output_2_{idx}', f'max_scale_{idx}', f'max_zero_point_{idx}'],
        #     outputs=[f'internal_output_2_{idx}_f_2'],
        # )
        # this_node_list.append(node)

        # # node = hel.make_node(
        # #     "DequantizeLinear",
        # #     inputs=[f'max_{idx}', f'max_scale_{idx}', f'max_zero_point_{idx}'],
        # #     outputs=[f'max_{idx}_f'],
        # # )
        # # this_node_list.append(node)
        


        # node = hel.make_node(
        #     "QuantizeLinear",
        #     inputs=[f'output_{idx}_f',  f'max_scale_{idx}',f'y_zero_point_{idx}'],
        #     # outputs=[f'output_{idx}'],
        #     outputs=[f'output_{idx}'],
        # )
        # this_node_list.append(node)


        node_list = node_list + this_node_list
        return node_list, f'output_{idx}', np.int8(0) if self.pure_positive else np.int8(0), idx+1
