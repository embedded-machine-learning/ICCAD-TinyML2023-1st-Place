import numpy as np
import torch
from typing import Tuple, Optional

from ...add import AddRELU
from ...Quantizer import Quant
from ...DataWrapper import DataWrapper
from ...Conversion import Start,Stop

def test_basic():
    DUT = AddRELU((1,),out_quant_args=(8,(1,),'floor'),in_min_shifts=[1,1])
    DUT.train()
    assert DUT.a_shift.shape == (1,)
    assert DUT.b_shift.shape == (1,)
    starta = Start(8)
    startb = Start(8)
    stopa = Stop(8)
    stopb = Stop(8)

    ar = torch.rand((12,30,30,30))
    br = 2*torch.rand((12,30,30,30))

    a = starta(ar)
    b = startb(br)

    a_val = stopa(a)
    b_val = stopa(b)
    
    print(a['rexp'],b['rexp'])
    print(a.clone()['rexp'],b.clone()['rexp'])

    out = DUT(a.clone(),b.clone())

    assert id(a) != id(b)
    assert id(a) != id(out)
    assert id(b) != id(out)

    # print("a_shift:",DUT.a_shift)
    # print("b_shift:",DUT.b_shift)
    # print(a['value'],b['value'])
    # print(out['value'])

    
    mask = torch.isclose(torch.clip(a_val+b_val,min=0),out['value'],atol=2*startb.delta_out.item())
    print("diff:",(torch.clip(a_val+b_val,min=0)-out['value'])[~mask])
    print("a+b:",(a_val+b_val)[~mask])
    print("out:",out['value'][~mask])
    print(torch.sum(~mask))

    assert torch.isclose(torch.clip(a_val+b_val,min=0),out['value'],atol=2*startb.delta_out.item()).all()


    starta.eval()
    startb.eval()
    stopa.eval()
    stopb.eval()
    DUT.eval()

    stopout = Stop()
    stopout.eval()

    a = starta(ar)
    b = startb(br)

    a_val = stopa(a)
    b_val = stopa(b)

    out = DUT(a.clone(),b.clone())

    out_val = stopout(out)
    
    assert id(a) != id(b)
    assert id(a) != id(out)
    assert id(b) != id(out)

    print("a_shift:",DUT.a_shift)
    print("b_shift:",DUT.b_shift)
    print(a['value'],b['value'])
    print(out['value'])

    
    mask = torch.isclose(a_val+b_val,out_val,atol=2*startb.delta_out.item())
    print("diff:",(torch.clip(a_val+b_val,min=0)-out_val)[~mask])

    assert torch.isclose(a_val+b_val,out_val,atol=2*startb.delta_out.item()).all()



