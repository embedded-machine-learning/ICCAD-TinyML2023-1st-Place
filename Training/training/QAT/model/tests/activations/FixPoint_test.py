import torch

from ...activations import FixPoint

def unit(fp,gen_size, size,bits,fixpoint):
    x = torch.rand(gen_size)
    delta = 2**(bits-fixpoint-1)/(2**(bits-1)-1)
    res = x.clone().div(delta).floor().clamp(-2**(bits-1),2**(bits-1)-1).mul(delta)
    assert (fp(x) == res).all()
    assert id(fp(x)) == id(x)


def test():
    size=(1,23,1,1)
    size_gen=(1123,23,15,17)
    for bits in range(2,8):
        for fixpoint in range(1,8):
            fp = FixPoint(bits,size,'floor',False,fixpoint)
            unit(fp,size_gen,size,bits,fixpoint)