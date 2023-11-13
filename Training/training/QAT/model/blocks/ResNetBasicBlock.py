# Modified from: https://github.com/lancopku/label-embedding-network/blob/master/ComputerVision/models/resnet8.py

import torch
import torch.nn as nn


from . import ConvBnA
from ..add import AddRELU
from ..activations import OutQant_var_based, PACT
from ..QuantizationMethods.MinMSE import MinMSE_convolution_weight_quantization

class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_planes, planes, stride=1,
                     weight_bits = 8, 
                     weight_quant = MinMSE_convolution_weight_quantization,
                     activation_bits = 8,
                     bypass_add_quant=OutQant_var_based, activation=PACT):
            super(BasicBlock, self).__init__()

            self.downsample = stride != 1 or in_planes != self.expansion*planes

            def out_quant_args(chan): return (activation_bits, (1, chan, 1, 1), 'floor')
           
            self.conv_bn_a_1 = ConvBnA(in_planes, planes, kernel_size=3, stride=stride, padding=1, weight_quant=weight_quant,
                                    weight_quant_bits=weight_bits, activation=activation, activation_args=out_quant_args(planes), activation_kargs={'use_enforced_quant_level': False})
            self.conv_bn_a_2 = ConvBnA(planes, planes, kernel_size=3, stride=1, padding=1, weight_quant=weight_quant,
                                    weight_quant_bits=weight_bits, activation=bypass_add_quant, activation_args=out_quant_args(planes), activation_kargs={'use_enforced_quant_level': not self.downsample})
            
            nn.init.kaiming_normal_(self.conv_bn_a_1.conv.weight, mode="fan_out", nonlinearity="relu")
            nn.init.kaiming_normal_(self.conv_bn_a_2.conv.weight, mode="fan_out", nonlinearity="relu")
            # nn.init.constant_(self.conv_bn_a_2.bn.weight, 0)  # type: ignore[arg-type]

            if self.downsample:
                self.shortcut = ConvBnA(in_planes, self.expansion*planes, kernel_size=1, stride=stride, padding=0, weight_quant=weight_quant,
                                    weight_quant_bits=weight_bits, activation=bypass_add_quant, activation_args=out_quant_args(self.expansion*planes), activation_kargs={'use_enforced_quant_level': self.downsample})
                nn.init.kaiming_normal_(self.shortcut.conv.weight, mode="fan_out", nonlinearity="relu")

            # self.add = Add((1,planes,1,1),RELU_Var,out_quant_args=out_quant_args(planes), out_quant_kargs={'use_enforced_quant_level': True})
            self.add = AddRELU((1,planes,1,1),None,out_quant_args=out_quant_args(planes),in_min_shifts=[0,0 if self.downsample else 1])
            # self.add = AddRELU((1,planes,1,1),None,out_quant_args=out_quant_args(planes),in_min_shifts=[1,1])

        def forward(self, x):
            x.set_quant()
            out = self.conv_bn_a_1(x)
            out = self.conv_bn_a_2(out)
            out.set_quant()
            if self.downsample:
                x.set_quant(out)
                x = self.shortcut(x)
            out = self.add(out,x)
            
            return out