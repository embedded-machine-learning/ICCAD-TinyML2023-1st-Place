from typing import Union

import torch
from torch import nn
from torch.nn.common_types import Tensor, _size_2_t
import torch.nn.functional as F

import numpy as np

from ..logger import logger_forward, logger_init
from .. import __FLAGS__


class LinBnA_int(nn.Module):
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
        # linear
        in_features: int,
        out_features: int,
        # transferred Convolution parameter
        Lin_weight: Tensor,
        BN_shift: Tensor,
        BN_add: Tensor,
        Act_min: Tensor,
        Act_max: Tensor,
        # type information
        accumulation_type=torch.int32,
        small_signed_type=torch.int8,
        small_unsigned_type=torch.uint8,
    ) -> None:
        super(LinBnA_int, self).__init__()
        self.Lin = nn.Linear(
            in_features, 
            out_features, 
            False,
        )
        self.Lin.weight.requires_grad_(False)
        self.Lin.weight.data = Lin_weight.to(small_signed_type)

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
            x =F.Linear(
                x.float(), self.Lin.weight.float(), self.Lin.bias).to(self.accumulation_type)
        else:
            x = F.Linear(x.to(self.accumulation_type), self.Lin.weight.to(
                self.accumulation_type), self.Lin.bias)

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

        b_zero_point = (128*np.ones_like(w_scale)).astype(np.uint8)

        B = (self.t.cpu().detach().view(-1).numpy())/w_scale

        conversions = {
            f'b_{idx}': (self.Lin.weight.cpu().detach().numpy()+128).astype(np.uint8).T,
            f'a_scale_{idx}': np.float32(1),
            f'b_scale_{idx}': w_scale,
            f'y_scale_{idx}': y_scale,
            f'a_zero_point_{idx}': input_zero_point,
            f'b_zero_point_{idx}': b_zero_point,
            f'y_zero_point_{idx}': np.uint8(0) if self.pure_positive else np.int8(0),
            f'B_{idx}': B.astype(np.int32).reshape(-1),
            # f'min_{idx}': (self.min.cpu().detach().numpy() + (np.uint8(0) if self.pure_positive else np.int8(128))).astype(np.uint8),
            # f'max_{idx}': (self.max.cpu().detach().numpy() + (np.uint8(0) if self.pure_positive else np.int8(128))).astype(np.uint8),
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

        node = hel.make_node(
            "QLinearMatMul",
            name=f'QLinearMatMul_{idx}',
            inputs=[
                input_name,
                f'a_scale_{idx}',
                f'a_zero_point_{idx}',
                f'b_{idx}',
                f'b_scale_{idx}',
                f'b_zero_point_{idx}',
                f'y_scale_{idx}',
                f'y_zero_point_{idx}',
            ],
            outputs=[f'output_{idx}'],
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
        # node = onnx.helper.make_node(
        #     "Min",
        #     name=f'Min_{idx}',
        #     inputs=[f'internal_output_2_{idx}', f'max_{idx}'],
        #     outputs=[f'output_{idx}'],
        # )

        # this_node_list.append(node)

        node_list = node_list + this_node_list
        return node_list, f'output_{idx}', np.uint8(0) if self.pure_positive else np.int8(0), idx+1
    
    def onnx_export(self, node_list, input_name, input_zero_point, idx=0):
        import onnx
        import onnx.numpy_helper
        import onnx.onnx_ml_pb2
        import onnx.helper as hel

        eq_mul = self.n_eq_mult.cpu().detach().view(-1).numpy().astype(np.float32)
        y_scale = (np.max(1/eq_mul)).astype(np.float32)
        w_scale = (eq_mul*y_scale).astype(np.float32)

        b_zero_point = (0*np.ones_like(w_scale)).astype(np.int8)

        B = (self.t.cpu().detach().view(-1).numpy())/w_scale

        conversions = {
            f'b_{idx}': (self.Lin.weight.cpu().detach().numpy()).astype(np.int8),
            f'a_scale_{idx}': np.float32(1),
            f'b_scale_{idx}': w_scale,
            f'y_scale_{idx}': y_scale,
            f'a_zero_point_{idx}': input_zero_point,
            f'b_zero_point_{idx}': b_zero_point,
            f'y_zero_point_{idx}': np.uint8(0) if self.pure_positive else np.int8(0),
            f'B_{idx}': B.astype(np.float32).reshape(1,-1),
            # f'min_{idx}': (self.min.cpu().detach().numpy() + (np.uint8(0) if self.pure_positive else np.uint8(128))).astype(np.uint8),
            # f'max_{idx}': (self.max.cpu().detach().numpy() + (np.uint8(0) if self.pure_positive else np.uint8(128))).astype(np.uint8),
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


        node = hel.make_node(
            "DequantizeLinear",
            inputs=[input_name, f'a_scale_{idx}', f'a_zero_point_{idx}'],
            axis=1,
            outputs=[f'{input_name}_f'],
        )
        this_node_list.append(node)



        node = hel.make_node(
            "DequantizeLinear",
            inputs=[f'b_{idx}', f'b_scale_{idx}', f'b_zero_point_{idx}'],
            axis=1,
            outputs=[f'b_{idx}_f'],
        )
        this_node_list.append(node)


        node = hel.make_node(
            "Gemm",
            name=f'Gemm_{idx}',
            transB=0,
            transA=0,
            inputs=[
                f'{input_name}_f',
                f'b_{idx}_f',
                f'B_{idx}',
            ],
            outputs=[f'output_{idx}_f'],
            # outputs=[f'internal_output_1_{idx}'],
        )

        this_node_list.append(node)


        node = hel.make_node(
            "QuantizeLinear",
            inputs=[f'output_{idx}_f', f'y_scale_{idx}', f'y_zero_point_{idx}'],
            axis=1,
            outputs=[f'output_{idx}'],
        )
        this_node_list.append(node)

        # node = onnx.helper.make_node(
        #     "Max",
        #     name=f'Max_{idx}',
        #     inputs=[f'internal_output_1_{idx}', f'min_{idx}'],
        #     outputs=[f'internal_output_2_{idx}'],
        # )

        # this_node_list.append(node)
        # node = onnx.helper.make_node(
        #     "Min",
        #     name=f'Min_{idx}',
        #     inputs=[f'internal_output_2_{idx}', f'max_{idx}'],
        #     outputs=[f'output_{idx}'],
        # )

        # this_node_list.append(node)

        node_list = node_list + this_node_list
        return node_list, f'output_{idx}', np.uint8(0) if self.pure_positive else np.int8(0), idx+1
