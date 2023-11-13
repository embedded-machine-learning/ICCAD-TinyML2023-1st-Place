import torch
from torch import nn
from torch.nn.common_types import Tensor

from ..DataWrapper import DataWrapper
from ..logger import logger_forward, logger_init


def Flatten(input: DataWrapper, dim: int) -> DataWrapper:
    """
    Flatten encapsulation of torch.flatten
    """
    val, rexp = input.get()
    orexp = rexp.detach() * torch.ones_like(val[0, ...])  # creates a
    return input.set(val.flatten(dim), orexp.flatten(dim))


class FlattenM_int(nn.Module):
    """
    FlattenM_int A simple module to use flatten as a layer

    :param dim: The flatten dimension
    :type dim: int
    """

    @logger_init
    def __init__(self, dim: int) -> None:
        super(FlattenM_int, self).__init__()
        self.dim = dim

    @logger_forward
    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(self.dim)
    
    def onnx_export(self, node_list, input_name, input_zero_point, idx=0):
        import onnx
        import onnx.numpy_helper
        import onnx.onnx_ml_pb2
        import onnx.helper as hel

        node = onnx.helper.make_node(
            "Flatten",
            axis=1,
            inputs=[input_name],
            outputs=[f'output_{idx}'],  # Default value for axis: axis=1
        )

        node_list.append(node)
        return node_list, f'output_{idx}', input_zero_point, idx+1



class FlattenM(nn.Module):
    """
    FlattenM A simple module to use flatten as a layer

    :param dim: The flatten dimension
    :type dim: int
    """

    @logger_init
    def __init__(self, dim: int) -> None:
        super(FlattenM, self).__init__()
        self.dim = dim

    def int_extract(self, accumulation_type=torch.int32, small_signed_type=torch.int8, small_unsigned_type=torch.uint8) -> FlattenM_int:
        return FlattenM_int(self.dim)

    @logger_forward
    def forward(self, input: DataWrapper) -> DataWrapper:
        return Flatten(input, self.dim)
