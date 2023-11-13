import torch
import torch.nn as nn

from ..DataWrapper import DataWrapper



value = torch.rand((10,10))
rexp = torch.rand((10,10))


value2 = torch.rand((10,10))
rexp2 = torch.rand((10,10))


DUT = DataWrapper(value.clone(),rexp.clone())
DUT.set_quant()

def test_str():
    str(DUT)


def test_get():
    assert (value==DUT.get()[0]).all()
    assert (rexp==DUT.get()[1]).all()

def test_brackets():
    assert (value==DUT['value']).all()
    assert (rexp==DUT['rexp']).all()
    assert (rexp.exp2() == DUT['quant_val']).all()


def test_clone():
    clone = DUT.clone()
    assert id(clone)!=id(DUT)
    assert id(clone['value'])!=id(DUT['value'])
    assert (clone['value']==DUT['value']).all()
    assert id(clone['rexp'])!=id(DUT['rexp'])
    assert (clone['rexp']==DUT['rexp']).all()
    assert id(clone['quant_val'])!=id(DUT['quant_val'])
    assert (clone['quant_val']==DUT['quant_val']).all()

def test_set():
    DUT2 = DUT.set(value2,rexp2)

    for s,data_orig, data_new in zip(['value','rexp'],[value,rexp],[value2,rexp2]):
        assert (data_orig == DUT[s]).all()
        assert (data_new == DUT2[s]).all()
        assert id(DUT[s]) != id(DUT2[s])

def test_quant():
    x = torch.rand(10,10)
    out = DUT.use_quant(x)

    assert (rexp.exp2() == DUT['quant_val']).all()
    assert (out == x.div(rexp.exp2()).log2().round().exp2().mul(rexp.exp2())).all()