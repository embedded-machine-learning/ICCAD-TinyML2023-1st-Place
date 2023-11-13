import torch
from torch.nn.common_types import Tensor

from ...batchnorm.functions import calculate_n_a, calculate_n_a_fixed, calculate_t


runs = 5

def test_calculate_n():
    """
    weight: Tensor,
    bias: Tensor,
    mean: Tensor,
    var: Tensor,
    out_quant: Tensor,
    rexp: Tensor,
    """    
    for i in range(runs):
        weight      = torch.rand((1,100,1,1))
        var         = torch.rand((1,100,1,1))
        bias        = torch.rand((1,100,1,1))
        mean        = torch.rand((1,100,1,1))
        out_quant   = torch.rand((1,100,1,1))
        rexp        = torch.rand((1,100,1,1))

        n,a = calculate_n_a(weight=weight,mean=mean,var=var,out_quant=out_quant,rexp=rexp)
        n_should_be = (torch.log2(weight.abs() / (out_quant * torch.sqrt(var + 1e-5))))
        n_should_be = torch.nan_to_num(n_should_be,nan=0,posinf=0,neginf=-32).add(rexp.view(-1)).clip(min=-32,max=0).ceil()


        assert torch.isclose(n,n_should_be).all()


def test_calculate_n_fixed():
    for i in range(runs):
        weight      = torch.rand((1,100,1,1))
        var         = torch.rand((1,100,1,1))
        bias        = torch.rand((1,100,1,1))
        mean        = torch.rand((1,100,1,1))
        out_quant   = torch.rand((1,100,1,1))
        rexp        = torch.rand((1,100,1,1))

        n,a = calculate_n_a_fixed(weight=weight,mean=mean,var=var,out_quant=out_quant,rexp=rexp)
        n_should_be = torch.log2(weight.abs() / (out_quant * torch.sqrt(var + 1e-5)))
        n_should_be = torch.nan_to_num(n_should_be,nan=0,posinf=0,neginf=-32).add(rexp.view(-1)).clip(min=-32,max=0)

        n_should_be = n_should_be.median() * torch.ones_like(n_should_be)
        n_should_be = n_should_be.ceil()

        assert torch.isclose(n,n_should_be).all()


def test_calculate_t():
    for i in range(runs):
        weight      = torch.rand((1,100,1,1))
        var         = torch.rand((1,100,1,1))
        bias        = torch.rand((1,100,1,1))
        mean        = torch.rand((1,100,1,1))
        out_quant   = torch.rand((1,100,1,1))
        rexp        = torch.rand((1,100,1,1))

        t = calculate_t(weight=weight,bias=bias,mean=mean,var=var,out_quant=out_quant,rexp=rexp,n=None)
        t_should_be = -mean * (weight / (out_quant * torch.sqrt(var + 1e-5))) + bias / out_quant

        assert torch.isclose(t,t_should_be).all()


def test_calculate_alpha():
    for i in range(runs):
        weight      = torch.rand((1,100,1,1))
        var         = torch.rand((1,100,1,1))
        mean        = torch.rand((1,100,1,1))
        out_quant   = torch.rand((1,100,1,1))
        rexp        = torch.rand((1,100,1,1))

        n,alpha = calculate_n_a(weight=weight,mean=mean,var=var,out_quant=out_quant,rexp=rexp)
        n_should_be = torch.log2(weight.abs() / (out_quant * torch.sqrt(var + 1e-5)))
        n_should_be = torch.nan_to_num(n_should_be,nan=0,posinf=0,neginf=-32).add(rexp.view(-1)).clip(min=-32,max=0)

        nr = torch.ceil(n_should_be)
        alpha_should_be = torch.sign(weight) * torch.exp2(n_should_be - nr)

        assert torch.isclose(alpha,alpha_should_be).all()


def test_calculate_alpha_fixed():
    for i in range(runs):
        weight      = torch.rand((1,100,1,1))
        var         = torch.rand((1,100,1,1))
        mean        = torch.rand((1,100,1,1))
        out_quant   = torch.rand((1,100,1,1))
        rexp        = torch.rand((1,100,1,1))

        n,alpha = calculate_n_a_fixed(weight=weight,mean=mean,var=var,out_quant=out_quant,rexp=rexp)
        n_should_be = torch.log2(weight.abs() / (out_quant * torch.sqrt(var + 1e-5)))
        n_should_be = torch.nan_to_num(n_should_be,nan=0,posinf=0,neginf=-32).add(rexp.view(-1)).clip(min=-32,max=0)
        
        nr = n_should_be.median() * torch.ones_like(n_should_be)
        nr = torch.ceil(nr)
        alpha_should_be = torch.sign(weight) * torch.exp2(n_should_be - nr)

        assert torch.isclose(alpha,alpha_should_be).all()
