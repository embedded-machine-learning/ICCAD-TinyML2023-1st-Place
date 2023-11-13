import torch

from ..Conversion import Start, Stop
from ..DataWrapper import DataWrapper

runs = 1

def test_conversion():
    DUT_start = Start(8,(1,100,1,1),auto_runs=100000,inline=False)
    DUT_stop = Stop((1,100,1,1))
    for i in range(runs):
        x = torch.rand((22,100,231,241))
        x_cl = x.clone()
        y = DUT_start(x)

        assert torch.isclose(x,x_cl).all()
        assert (y['rexp'].exp2().log2()-y['rexp']==0).all()
        y : DataWrapper

        x_quant = x_cl.div(DUT_start.delta_in).floor_().mul_(DUT_start.delta_in)

        print(DUT_start.in_min,DUT_start.in_max,DUT_start.only_positive)
        print(DUT_start.min.view(-1),DUT_start.max.view(-1))
        mask = torch.isclose(y['value'],x_quant)
        print(x_quant[~mask],y['value'][~mask])
        print(y['rexp'].exp2().view(-1))

        assert torch.isclose(y['value'],x_quant).all()

        out = DUT_stop(y)

        assert torch.isclose(out,x_quant).all()
        x = torch.rand((22,100,231,241)) - 0.5
        x_cl = x.clone()
        y = DUT_start(x)

        assert torch.isclose(x,x_cl).all()
        assert (y['rexp'].exp2().log2()-y['rexp']==0).all()
        y : DataWrapper

        x_quant = x_cl.div(DUT_start.delta_in).floor_().mul_(DUT_start.delta_in)

        print(DUT_start.in_min,DUT_start.in_max,DUT_start.only_positive)
        print(DUT_start.min.view(-1),DUT_start.max.view(-1))
        mask = torch.isclose(y['value'],x_quant)
        print(x_quant[~mask],y['value'][~mask])
        print(y['rexp'].exp2().view(-1))

        assert torch.isclose(y['value'],x_quant).all()

        out = DUT_stop(y)

        assert torch.isclose(out,x_quant).all()
