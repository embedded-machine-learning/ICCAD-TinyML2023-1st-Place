import torch

from ...activations import PACT
from ...activations.PACT import PACT_back_function

def unit(un:PACT,gen_size, size,bits):
    x = torch.rand(gen_size)
    alpha = un.alpha.clone()

    delta = (alpha*(1/(2**bits-1)))
    res = x.clone().div(delta).floor().clamp(0,2**bits-1).mul(delta)

    out = un(x)   # just run once
    print('================================================')
    close = torch.isclose(out,res)
    close_delta = torch.isclose(un.delta_in,delta,rtol=1e-8)
    print('calc delta:',delta[~close_delta])
    print('un   delta:',un.delta_in[~close_delta])
    print('calc res:',res[~close])
    print('un   res:',out[~close])
    print('diff:',(out-res)[~close])
    print('un max vs bit max:',un.max.view(-1),2**bits-1)

    assert (un.max == 2**bits-1).all()
    assert torch.isclose(un.delta_in,delta).all()
    assert torch.isclose(un.delta_out,delta).all()
    assert torch.isclose(out,res).all()
    assert id(x) != id(out)


def test():
    size=(1,100,1,1)
    size_gen=(123,100,165,18)
    for bits in range(1,8):
        un = PACT(bits,size,'floor',False)
        un.alpha.data = 1+torch.rand(size)*5
        unit(un,size_gen,size,bits)

def test_backprop():
    size=(1,100,1,1)
    size_gen=(123,100,165,18)
    
    alpha = (1+torch.rand(size)*5).clone().detach().requires_grad_(True)
    x = (torch.rand(size_gen)*10).clone().detach().requires_grad_(True)

    x_pos = x.detach()>=0
    x_greater_alpha = x.detach()>=alpha.detach()

    x_back_should_be = torch.logical_and(x_pos,~x_greater_alpha)*torch.ones(size_gen)
    alpha_back_should_be = (x_greater_alpha*torch.ones(size_gen)).sum((0,2,3),keepdim=True)

    out = torch.sum(PACT_back_function.apply(x,alpha))
    out.backward()

    x_back = x.grad
    assert torch.isclose(x_back,x_back_should_be).all()


    alpha_back = alpha.grad
    assert torch.isclose(alpha_back,alpha_back_should_be).all()

