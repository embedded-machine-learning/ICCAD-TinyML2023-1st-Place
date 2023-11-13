import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from ...linear import Linear
from ...Conversion import Start, Stop
from ...DataWrapper import DataWrapper
from ...activations import FixPoint

in_features = 20000
out_features = 20

def test_linear():
    DUT = Linear(in_features,out_features,bias=False,out_quant_kargs={'mom1':1})
    
    start = Start(8,(1,in_features),auto_runs=10000)
    stop = Stop((1,out_features))

    for i in range(100):
        print('=============== train ======================')
        print(f'run number: {i}')
        DUT.train()
        start.train()
        stop.train()

        x = torch.rand((10,in_features))
        x_dw = start(x)
        x_dw : DataWrapper
        out = DUT(x_dw)
        # out = DUT(x_dw)
        out_val = stop(out)

        should_be = F.linear(x_dw['value'],DUT.test['weight'],DUT.bias)
        # should_be = F.conv2d(x,DUT.weight,DUT.bias,DUT.stride,DUT.padding,DUT.dilation,DUT.groups)
        dist = out_val-should_be

        mask = torch.isclose(out_val,should_be,atol=float(DUT.down_scaler.out_quant.delta_out.max().numpy()))
        print(out_val[~mask])
        print(should_be[~mask])
        print(DUT.down_scaler.out_quant.delta_out.view(-1))
        print(dist.min(),dist.max())
        print(torch.sum(~mask)/float(np.prod(mask.shape)))

        assert torch.sum(~mask)/float(np.prod(mask.shape)) < 1e-4
        
        print('=============== eval =======================')
        print(f'run number: {i}')
        weight = DUT.test['weight'].clone()
        old_out_val = out_val.clone().detach()
        DUT.eval()
        start.eval()
        stop.eval()

        # x = torch.rand((10,in_channels,123,44))
        x_dw = start(x)
        x_dw : DataWrapper
        out = DUT(x_dw)
        out_val = stop(out)

        should_be = F.linear(x_dw['value'],DUT.test['weight'],DUT.bias)*torch.exp2(DUT.down_scaler.right_shift)*out['rexp'].exp2()
        possibly = F.linear(x,DUT.weight,DUT.bias)
        dist = out_val-should_be
        dist2 = out_val-old_out_val

        mask = torch.isclose(out_val,should_be,atol=float(DUT.down_scaler.out_quant.delta_out.max().numpy()))
        print('out_val:',out_val[~mask])
        print('should_be:',should_be[~mask])
        print('delta_out:',DUT.down_scaler.out_quant.delta_out.view(-1))
        print('min/max:',dist.min(),dist.max())
        print('error%:',100*torch.sum(~mask)/float(np.prod(mask.shape)))

        mask2 = torch.isclose(out_val,old_out_val,atol=float(DUT.down_scaler.out_quant.delta_out.max().numpy()))
        print('out_val:',out_val[~mask2])
        print('should_be:',should_be[~mask2])
        print('delta_out:',DUT.down_scaler.out_quant.delta_out.view(-1))
        print('min/max:',dist2.min(),dist2.max())
        print('error%:',100*torch.sum(~mask2)/float(np.prod(mask2.shape)))

        assert torch.sum(~mask)/float(np.prod(mask.shape)) < 1e-4
        assert torch.sum(~mask2)/float(np.prod(mask2.shape)) < 1e-4