import torch

from ...activations import ReLU
from ...activations.ReLu import RELU_back_function

def unit(un:ReLU,gen_size, size,bits):
    x = torch.rand(gen_size)
    x_clone = x.clone()

    sigma = x.clone().var(un.reduce_list, unbiased=False, keepdim=True).add(1e-5).sqrt()
    delta = sigma*(1/70)
    res = x.clone().div(delta).floor().clamp(0,2**bits-1).mul(delta)

    out = un(x)   # just run once
    print('================================================')
    close = torch.isclose(out,res)
    close_delta = torch.isclose(un.delta_in,delta,rtol=1e-8)
    close_sigma = torch.isclose(un.sigma,sigma,rtol=1e-8)
    print('sigma:')
    print(sigma[~close_sigma])
    print(un.sigma[~close_sigma])
    print('delta:')
    print(delta[~close_delta])
    print(un.delta_in[~close_delta])
    print('res:')
    print(res[~close])
    print(out[~close])

    assert (un.max == 2**bits-1).all()
    assert torch.isclose(un.delta_in,delta).all()
    assert torch.isclose(un.delta_out,delta).all()
    assert torch.isclose(out,res).all()
    assert id(x) != id(out)
    assert torch.isclose(x,x_clone).all()



    # x = torch.rand(gen_size)

    # sigma = x.clone().var(un.reduce_list, unbiased=False, keepdim=True).add(1e-5).sqrt()
    # delta = sigma*(1/70)
    # res = x.clone().div(delta).floor().clamp(0,2**bits-1).mul(delta)

    out = un(x)   # run again to use momentum
    print('================================================')
    close_sigma = torch.isclose(un.sigma,sigma,rtol=1e-8)
    close_delta = torch.isclose(un.delta_in,delta,rtol=1e-8)
    close = torch.isclose(out,res)
    print('sigma:')
    print(sigma[~close_sigma])
    print(un.sigma[~close_sigma])
    print('delta:')
    print(delta[~close_delta])
    print(un.delta_in[~close_delta])
    print('res:')
    print(res[~close])
    print(out[~close])

    assert (un.max == 2**bits-1).all()
    assert torch.isclose(un.delta_in,delta).all()
    assert torch.isclose(un.delta_out,delta).all()
    assert torch.isclose(out,res).all()
    assert id(x) != id(out)


def test():
    size=(1,100,1,1)
    size_gen=(123,100,165,18)
    for bits in range(1,8):
        un = ReLU(bits,size,'floor',False,1)
        unit(un,size_gen,size,bits)

def test_backprop():
    size=(1,100,1,1)
    size_gen=(123,100,165,18)
    
    x = (torch.rand(size_gen)*10).clone().detach().requires_grad_(True)

    x_pos = x.detach()>=0

    x_back_should_be = x_pos*torch.ones(size_gen)
    
    out = torch.sum(RELU_back_function.apply(x))
    out.backward()

    x_back = x.grad
    assert torch.isclose(x_back,x_back_should_be).all()

