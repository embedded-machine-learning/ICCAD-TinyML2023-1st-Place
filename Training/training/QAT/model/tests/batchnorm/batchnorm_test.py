import torch
from torch.nn.common_types import Tensor
import numpy as np

from ...batchnorm import BatchNorm2d
from ...activations import PACT
from ...Conversion import Start, Stop
from ...DataWrapper import DataWrapper

channels = 100

def test_batchnorm():
    DUT = BatchNorm2d(num_features=channels)
    batchnorm = torch.nn.BatchNorm2d(num_features=channels)
    start = Start(24,auto_runs=10000)
    stop  = Stop()
    act = PACT(8,(1,channels,1,1))

    for i in range(100):
        print("=====================train============================================")
        start.train()
        stop.train()
        batchnorm.train()
        DUT.train()


        x = torch.rand((10,channels,123,45))
        # x = x-x.mean(dim=[0,2,3],keepdim=True)
        x_wrapped = start(x.clone().detach())
        x = stop(x_wrapped)
        DUT_y = DUT(x_wrapped,act)
        DUT_y_extracted = stop(DUT_y)
        should_be = act(batchnorm(x))
        
        
        mask = torch.isclose(DUT_y_extracted,should_be)
        print(DUT_y_extracted[~mask])
        print(should_be[~mask])


        assert torch.isclose(DUT.weight,batchnorm.weight).all()
        assert torch.isclose(DUT.running_mean,batchnorm.running_mean).all()
        assert torch.isclose(DUT.running_var,batchnorm.running_var).all()
        assert torch.isclose(DUT.bias,batchnorm.bias).all()
        assert torch.isclose(DUT_y_extracted,should_be).all()
       
        print("=====================eval============================================")
        start.eval()
        stop.eval()
        batchnorm.eval()
        DUT.eval()
       
        x = torch.rand((10,channels,123,45))
        # x = x-x.mean(dim=[0,2,3],keepdim=True)
        fun = DUT.get_weight_factor()
        x_wrapped : DataWrapper
        x_wrapped = start(x.clone().detach())
        x = stop(x_wrapped).clone().detach()
        alpha = fun(x_wrapped['rexp']).view(1,-1,1,1)
        x_wrapped = x_wrapped.set(x_wrapped['value']*alpha,x_wrapped['rexp'])
        DUT_y = DUT(x_wrapped,act)
        DUT_y_extracted = stop(DUT_y)
        should_be = act(batchnorm(x))*act.delta_out
        atol = float(act.delta_out.abs().max().detach().numpy()*1.1)
        mask = torch.isclose(DUT_y_extracted,should_be)
        print('atol', atol)
        print(DUT_y_extracted[~mask])
        print(should_be[~mask])
        diff =should_be[~mask]-DUT_y_extracted[~mask] 
        print(diff)
        if len(diff) != 0:
            print(diff.min(), diff.max())
        print(torch.sum(~mask)/np.prod(x.shape))
        print(fun(x_wrapped['rexp']).view(-1))
        print(DUT.n.view(-1))
        print(DUT.t.view(-1))
        print(act.min.view(-1),act.max.view(-1))
        print(act.delta_in.view(-1))

        assert torch.all(DUT.n<0).all()
        assert torch.isclose(DUT.weight,batchnorm.weight).all()
        assert torch.isclose(DUT.running_mean,batchnorm.running_mean).all()
        assert torch.isclose(DUT.running_var,batchnorm.running_var).all()
        assert torch.isclose(DUT.bias,batchnorm.bias).all()
        assert torch.isclose(DUT_y_extracted,should_be,atol=atol,rtol=0).all()
        assert torch.sum(~mask)/np.prod(x.shape) < 0.001    # 1 promill error allowed

if __name__ == "__main__":
    test_batchnorm()