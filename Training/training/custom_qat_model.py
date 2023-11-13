import torch
from torch import Tensor
import torch.nn as nn

import numpy as np

from QAT.model.sequential import Sequential
from QAT.model.wrapped import Dropout, FlattenM
from QAT.model.linear import Linear
from QAT.model.activations import ReLU,FixPoint, PACT
from QAT.model.activations.ReLu import RELU_back_function

from QAT.model.Conversion import Stop, Start, DataWrapper
from QAT.model.QuantizationMethods.MinMSE import MinMSE_linear_weight_quantization
from QAT.model.QuantizationMethods.OCTAV_Stabalized import OCTAV_Stabalized_linear_weight_quantization
from QAT.model.linear.weight_quantization import LinQuantWeight, FakeQuant, Quant



class FakeBN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self,x):
        x = x-(-8.728156899451278e-7)
        x = x/np.sqrt(0.021395351737737656 + 1e-5)
        x = x*3.715294599533081
        x = x+0.1172400712966919
        return x
    

class Abs(nn.Module):
    def forward(self,x):
        return x.abs()
    

class Custom_Start(Quant):
    def __init__(self, bits, size=(-1,), rounding_mode: str = "floor", use_enforced_quant_level: bool = False, mom1: int  = 0.05) -> None:
        super(Custom_Start, self).__init__(bits, size, rounding_mode, use_enforced_quant_level)
        self.bits = bits
        if size == (-1,):
            self.register_buffer("sigma", torch.ones(1))
        else:
            self.register_buffer("sigma", torch.ones(size))
        self.take_new = True
        self.mom1 = mom1
        assert self.bits > 0
        self.register_buffer("delta_in_factor", torch.tensor(1./(2**(bits-1))))
        self.register_buffer("delta_out_factor", torch.tensor(1./(2**(bits-1))))

        nn.init.constant_(self.min, 0)
        nn.init.constant_(self.max, 2**(bits-1) - 1)
        self.max_list = []
        self.index = 0

    def train(self, mode: bool = True):
        if not mode:
            self.index=0
        return super(Custom_Start,self).train(mode)

    def forward(self, x: torch.Tensor, fake: bool = False, metadata = None,*args,**kargs):
        if self.training:
            with torch.no_grad():
                sigma = torch.amax(x, self.reduce_list, keepdim=True)+1e-10
                # # print(sigma)
                # if self.take_new:
                #     self.take_new = False
                #     self.sigma = sigma
                # else:
                #     self.sigma = (1 - self.mom1) * self.sigma + self.mom1 * sigma
                #     sigma = self.sigma

                self.delta_in = sigma.mul(self.delta_in_factor).detach()
                self.delta_out = sigma.mul(self.delta_in_factor).detach()

                if self.use_enforced_quant_level and metadata is not None:
                    self.use_quant(metadata)
                if self.use_enforced_quant_level and metadata is None:
                    raise ValueError("Quantization function desired but metadata not passed")

        return DataWrapper(
            FakeQuant(
                x = x,
                delta_in = self.delta_in,
                delta_out = self.delta_out,
                training = self.training,
                min_quant =  self.min,
                max_quant = self.max,
                rounding_mode = 'floor',
                clamp = True,
            ),
            torch.log2(self.delta_out),
        )


class Max_Quant(LinQuantWeight):
    def __init__(self, bits: int = 8, size: tuple = (-1,), rounding_mode: str = "round") -> None:
        super(Max_Quant,self).__init__(bits, size, rounding_mode)
        
        self.register_buffer("delta_in_factor", torch.tensor(1./(2**(bits-1))))
        self.register_buffer("delta_out_factor", torch.tensor(1./(2**(bits-1))))
        if size == (-1,):
            self.register_buffer("sigma", torch.ones(1))
        else:
            self.register_buffer("sigma", torch.ones(size))
        
    def forward(self, x: Tensor, rexp_mean: Tensor, rexp_diff: Tensor, fact_fun) -> Tensor:
        with torch.no_grad():
            # print(self.reduce_list)
            sigma = (
                torch.amax(x.abs() * (rexp_diff.view(1, -1)), self.reduce_list, keepdim=True)
            )
            self.sigma = sigma

            self.delta_in = self.sigma.mul(self.delta_in_factor)
            self.delta_out = self.sigma.mul(self.delta_in_factor)

            fact = fact_fun((self.delta_out.view(1,-1) * rexp_mean).log2()).view(-1, 1)
            self.delta_for_quant = self.delta_in.div(rexp_diff.view(*self.rexp_view)).div_(fact)

        return FakeQuant(
                x=x.clone(),
                delta_in=self.delta_for_quant ,
                delta_out=self.delta_for_quant ,
                training=self.training,
                min_quant=self.min,
                max_quant=self.max,
                rounding_mode=self.rounding_mode,
            )


class Max_ReLU(Quant):
    """
    ReLU The implementation of the ReLU activation function fused into the quantization

    :param size: The shape for alpha, defaults to (1,)
    :type size: tuple, optional
    """

    def __init__(self, bits, size=(-1,), rounding_mode: str = "floor", use_enforced_quant_level: bool = False, mom1: int  = 0.1) -> None:
        super(Max_ReLU, self).__init__(bits, size, rounding_mode, use_enforced_quant_level)
        self.bits = bits
        if size == (-1,):
            self.register_buffer("sigma", torch.ones(1))
        else:
            self.register_buffer("sigma", torch.ones(size))
        self.take_new = True
        self.mom1 = mom1
        assert self.bits > 0
        self.register_buffer("delta_in_factor", torch.tensor(1./(2**(bits-1))))
        self.register_buffer("delta_out_factor", torch.tensor(1./(2**(bits-1))))

        nn.init.constant_(self.min, 0)
        nn.init.constant_(self.max, 2**(bits-1) - 1)
        self.max_list = []
        self.index = 0

    def train(self, mode: bool = True):
        if not mode:
            self.index=0
        return super(Max_ReLU,self).train(mode)

    def forward(self, x: torch.Tensor, fake: bool = False, metadata = None,*args,**kargs):
        if self.training:
            with torch.no_grad():
                sigma = torch.amax(x.clip(0), self.reduce_list, keepdim=True)+1e-5
                # # print(sigma)
                # if self.take_new:
                #     self.take_new = False
                #     self.sigma = sigma
                # else:
                #     self.sigma = (1 - self.mom1) * self.sigma + self.mom1 * sigma
                #     sigma = self.sigma

                self.delta_in = sigma.mul(self.delta_in_factor).detach()
                self.delta_out = sigma.mul(self.delta_in_factor).detach()

                if self.use_enforced_quant_level and metadata is not None:
                    self.use_quant(metadata)
                if self.use_enforced_quant_level and metadata is None:
                    raise ValueError("Quantization function desired but metadata not passed")

            x = RELU_back_function.apply(x)
        return super(Max_ReLU,self).forward(x, fake)


class Max_Out_quant(Quant):
    def __init__(self, bits, size=(-1,), rounding_mode: str = "floor", use_enforced_quant_level: bool = False, mom1: int  = 0.1) -> None:
        super(Max_Out_quant, self).__init__(bits, size, rounding_mode, use_enforced_quant_level)
        self.bits = bits
        if size == (-1,):
            self.register_buffer("sigma", torch.ones(1))
        else:
            self.register_buffer("sigma", torch.ones(size))
        self.take_new = True
        self.mom1 = mom1
        assert self.bits > 0
        self.register_buffer("delta_in_factor", torch.tensor(1./(2**(bits-1))))
        self.register_buffer("delta_out_factor", torch.tensor(1./(2**(bits-1))))

        nn.init.constant_(self.min, -2**(bits-1))
        nn.init.constant_(self.max, 2**(bits-1) - 1)

        self.max_list = []
        self.index = 0

    def train(self, mode: bool = True):
        if not mode:
            self.index=0
        return super(Max_Out_quant,self).train(mode)

    def forward(self, x: torch.Tensor, fake: bool = False, metadata = None,*args,**kargs):
        if self.training:
            with torch.no_grad():
                sigma = torch.amax(x.abs(), self.reduce_list, keepdim=True)+1e-5
                # # print(sigma)
                # if self.take_new:
                #     self.take_new = False
                #     self.sigma = sigma
                # else:
                #     self.sigma = (1 - self.mom1) * self.sigma + self.mom1 * sigma
                #     sigma = self.sigma

                self.delta_in = sigma.mul(self.delta_in_factor).detach()
                self.delta_out = sigma.mul(self.delta_in_factor).detach()

                if self.use_enforced_quant_level and metadata is not None:
                    self.use_quant(metadata)
                if self.use_enforced_quant_level and metadata is None:
                    raise ValueError("Quantization function desired but metadata not passed")

        return super(Max_Out_quant,self).forward(x, fake)






class Gatech_net_QAT(nn.Module):
    def __init__(self) -> None:
        super(Gatech_net_QAT, self).__init__()
        self.BN = FakeBN()
        self.conv = nn.Conv1d(1, 3, kernel_size=85, stride=32, padding=0, bias=True)
        self.act = Abs()
        self.flatten = FlattenM(1)
        self.start = Custom_Start(16,(1,3,1))
        self.stop = Stop()
        self.Tlayers = Sequential(
            Dropout(0.3),
            Linear(111, 20,weight_quant_channel_wise=True,out_quant=Max_ReLU,weight_quant=Max_Quant,weight_quant_bits=8,out_quant_args=(16,(1,20),'floor')),
            Linear(20, 2,weight_quant_channel_wise=True,out_quant=Max_Out_quant,weight_quant=Max_Quant,weight_quant_bits=8,out_quant_args=(16,(1,2),'floor')),
        )
        
        self.layers=[0,0,0,0,0,0,0,0,0,0,0]
        self.layers[1] =  self.conv
        self.layers[5] = self.Tlayers[1]
        self.layers[8] = self.Tlayers[2]
        
    def forward(self, x):
        # print(x.shape)
        x = x[:,:,:,0]
        x = self.BN(x)
        x = self.conv(x)
        x = self.act(x)
        x = self.start(x)
        x = self.flatten(x)
        x = self.Tlayers(x)
        x = self.stop(x)
        return x      
        