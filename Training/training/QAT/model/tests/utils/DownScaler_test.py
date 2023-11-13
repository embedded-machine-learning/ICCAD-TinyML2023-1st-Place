import torch
import torch.nn as nn
import numpy as np

from ...utils import DownScaler
from ...activations import FixPoint
from ...Conversion import Start, Stop
from ...DataWrapper import DataWrapper

channels = 101
def test_DownScaler():
    DUT = DownScaler((1,channels,1,1),out_quant_kargs={'mom1':1})

    start = Start(32,auto_runs=10000,inline=False)
    stop = Stop((1,channels,1,1))

    for i in range(100):
        print('=============== train ======================')
        print(f'run number: {i}')
        DUT.train()
        start.train()
        stop.train()

        x = torch.rand((12,channels,12,41))-0.5
        x_dw = start(x)

        bias = torch.rand((1,channels,1,1))-0.5

        out = DUT(x_dw,bias)
        out_val = stop(out)

        should_be = (x+bias).div(DUT.out_quant.delta_out).floor().clip(min=DUT.out_quant.min, max=DUT.out_quant.max).mul(DUT.out_quant.delta_out)

        mask = torch.isclose(out_val,should_be,atol=DUT.out_quant.delta_out.max()/2)

        print(out_val[~mask])
        print(should_be[~mask])
        print((should_be-out_val)[~mask])
        print(DUT.out_quant.delta_out.view(-1))

        assert torch.isclose(out_val,should_be).all()


        print('=============== eval =======================')
        print(f'run number: {i}')
        old_out_val = out_val.clone().detach()
        DUT.eval()
        start.eval()
        stop.eval()

        # x = torch.rand((12,channels,12,41))
        x_dw = start(x)
        x_dw : DataWrapper
        fun = DUT.get_weight_function()
        weight_factor = fun(x_dw['rexp'])
        x_dw = x_dw.set(x_dw['value']*(weight_factor),x_dw['rexp'])
        # bias = torch.rand((1,channels,1,1))-0.5

        out = DUT(x_dw,bias)
        out_val = stop(out)
        
        should_be = (x+bias).div(DUT.out_quant.delta_out).floor().mul(DUT.out_quant.delta_out)

        mask = torch.isclose(out_val,should_be,atol=DUT.out_quant.delta_out.max()/2)
        print(weight_factor)
        print(out_val[~mask])
        print((x+bias)[~mask])
        print(should_be[~mask])
        print(DUT.out_quant.delta_out.view(-1))

        assert torch.all(DUT.right_shift < 0)
        assert torch.sum(~mask)/np.prod(mask.shape) < 1e-4
        assert torch.all(torch.remainder(DUT.integer_bias,1)==0)

        mask_train = torch.isclose(old_out_val,out_val,atol=DUT.out_quant.delta_out.max()/2)
        print(out_val[~mask_train])
        print((x+bias)[~mask_train])
        print(should_be[~mask_train])
        print(DUT.out_quant.delta_out.view(-1))
        print(torch.sum(~mask_train)/np.prod(mask_train.shape))
        assert torch.sum(~mask_train)/np.prod(mask_train.shape) < 1e-4



if __name__ == "__main__":
    test_DownScaler()

