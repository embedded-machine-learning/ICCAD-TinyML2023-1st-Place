import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from ...blocks import ConvBnA
from ...convolution import Conv2d
from ...Conversion import Start, Stop
from ...DataWrapper import DataWrapper
from ...activations import FixPoint

in_channels = 200
out_channels = 200

class CompareBlock(nn.Module):
    def __init__(self, in_channels , out_channels) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,3,bias=False)
        self.bn = nn.BatchNorm2d(out_channels,momentum=1)
        self.act = nn.ReLU()
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x




def test_ConvBnA():
    DUT = ConvBnA(in_channels,out_channels,3,momentum=1,activation_kargs={'mom1':1})
    comp = CompareBlock(in_channels,out_channels)
    
    start = Start(8,(1,in_channels,1,1),auto_runs=10000)
    stop = Stop((1,out_channels,1,1))

    for i in range(100):
        print('=============== train ======================')
        print(f'run number: {i}')
        DUT.train()
        start.train()
        stop.train()
        comp.train()

        x = torch.rand((10,in_channels,123,44))
        x_dw = start(x)
        x_dw : DataWrapper
        out = DUT(x_dw)
        out = DUT(x_dw)
        out_val = stop(out)


        comp.conv.weight.data = DUT.conv.test['weight'].clone()

        should_be = comp(x)
        dist = out_val-should_be

        mask = torch.isclose(out_val,should_be,atol=1.5*float(DUT.bn.out_quant.delta_out.max().numpy()))
        print('out_val:',out_val[~mask])
        print('should_be:',should_be[~mask])
        print('delta_out:',DUT.bn.out_quant.delta_out.view(-1))
        print('min/max:',dist.min(),dist.max())
        print('error%:',100*torch.sum(~mask)/float(np.prod(mask.shape)))

        assert torch.sum(~mask)/float(np.prod(mask.shape)) < 5/100
        
        print('=============== eval =======================')
        print(f'run number: {i}')
        old_out_val = out_val.clone().detach()
        DUT.eval()
        start.eval()
        stop.eval()
        comp.eval()

        # x = torch.rand((10,in_channels,123,44))
        x_dw = start(x)
        x_dw : DataWrapper
        out = DUT(x_dw)
        out_val = stop(out)

        should_be = comp(x)
        dist = out_val-should_be
        dist2 = out_val-old_out_val

        mask = torch.isclose(out_val,should_be,atol=1.5*float(DUT.bn.out_quant.delta_out.max().numpy()))
        print('out_val:',out_val[~mask])
        print('should_be:',should_be[~mask])
        print('dist:',dist[~mask])
        print('delta_out:',DUT.bn.out_quant.delta_out.view(-1))
        print('min/max:',dist.min(),dist.max())
        print('error%:',100*torch.sum(~mask)/float(np.prod(mask.shape)))

        mask2 = torch.isclose(out_val,old_out_val,atol=1.1*float(DUT.bn.out_quant.delta_out.max().numpy()))
        print('out_val:',out_val[~mask2])
        print('should_be:',should_be[~mask2])
        print('dist:',dist2[~mask2])
        print('delta_out:',DUT.bn.out_quant.delta_out.view(-1))
        print('min/max:',dist2.min(),dist2.max())
        print('error%:',100*torch.sum(~mask2)/float(np.prod(mask2.shape)))

        assert torch.sum(~mask)/float(np.prod(mask.shape)) < 5/100
        assert torch.sum(~mask2)/float(np.prod(mask2.shape)) < 1/100


if __name__ == "__main__":
    test_ConvBnA()
