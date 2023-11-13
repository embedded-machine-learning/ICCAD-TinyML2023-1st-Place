import torch
from torch import nn


class Sequential_int(nn.Sequential):
    def onnx_export(self, node_list, input_name, input_zero_point, idx=0):
        for module in self:
            if callable(getattr(module, 'onnx_export', None)):
                print('layer eported')
                node_list, input_name, input_zero_point, idx = module.onnx_export(node_list, input_name, input_zero_point, idx)
            else:
                print('layers scipped')
        return node_list, input_name, input_zero_point, idx