import copy

import torch

from ..Quantizer import FakeQuant, Quant, LinQuantExpScale

##########################################################################
########################### FAKEQUANT ####################################
##########################################################################
# Desired behavior:
# training:
#   if training id should stay the same ony a value change, original value should be modified
#   if eval new value should be created
# rounding mode:
#   floor on floor, default
#   trunc at trunc
#   else round
# clamp:
#   clamp the integer domain if true
#   min,max_quant is the range
# delta:
#   the quantization factor
#       out = x.div(delta_in).<round>.<clamp>.mul(delta_out)

x = torch.rand((100, 100))
delta_in = torch.rand((1, 100))
delta_out = torch.rand((1, 100))
min_quant = torch.rand((1, 100))
max_quant = torch.rand((1, 100))


def regen_x():
    global x
    x = torch.rand((100, 100))


def test_train():
    # rounding mode floor , clamp false
    regen_x()
    out_s = x.div(delta_in).floor().mul(delta_out)
    res = FakeQuant(x, delta_in, delta_out, training=True, min_quant=None,
                    max_quant=None, rounding_mode='floor', clamp=False)
    assert (res == out_s).all()
    assert (res == x).all()
    assert id(res) == id(x)

    # rounding mode floor , clamp True
    regen_x()
    out_s = x.div(delta_in).floor().clamp(
        min=min_quant, max=max_quant).mul(delta_out)
    res = FakeQuant(x, delta_in, delta_out, training=True, min_quant=min_quant,
                    max_quant=max_quant, rounding_mode='floor', clamp=True)
    assert (res == out_s).all()
    assert (res == x).all()
    assert id(res) == id(x)

    # rounding mode trunc , clamp false
    regen_x()
    out_s = x.div(delta_in).trunc().mul(delta_out)
    res = FakeQuant(x, delta_in, delta_out, training=True, min_quant=None,
                    max_quant=None, rounding_mode='trunc', clamp=False)
    assert (res == out_s).all()
    assert (res == x).all()
    assert id(res) == id(x)

    # rounding mode trunc , clamp True
    regen_x()
    out_s = x.div(delta_in).trunc().clamp(
        min=min_quant, max=max_quant).mul(delta_out)
    res = FakeQuant(x, delta_in, delta_out, training=True, min_quant=min_quant,
                    max_quant=max_quant, rounding_mode='trunc', clamp=True)
    assert (res == out_s).all()
    assert (res == x).all()
    assert id(res) == id(x)

    # rounding mode round , clamp false
    regen_x()
    out_s = x.div(delta_in).round().mul(delta_out)
    res = FakeQuant(x, delta_in, delta_out, training=True, min_quant=None,
                    max_quant=None, rounding_mode='round', clamp=False)
    assert (res == out_s).all()
    assert (res == x).all()
    assert id(res) == id(x)

    # rounding mode round , clamp True
    regen_x()
    out_s = x.div(delta_in).round().clamp(
        min=min_quant, max=max_quant).mul(delta_out)
    res = FakeQuant(x, delta_in, delta_out, training=True, min_quant=min_quant,
                    max_quant=max_quant, rounding_mode='round', clamp=True)
    assert (res == out_s).all()
    assert (res == x).all()
    assert id(res) == id(x)


def test_eval():
    # rounding mode floor , clamp false
    regen_x()
    out_s = x.div(delta_in).floor()
    res = FakeQuant(x, delta_in, delta_out, training=False, min_quant=None,
                    max_quant=None, rounding_mode='floor', clamp=False)
    assert (res == out_s).all()
    assert not (res == x).all()
    assert id(res) != id(x)

    # rounding mode floor , clamp True
    regen_x()
    out_s = x.div(delta_in).floor().clamp(min=min_quant, max=max_quant)
    res = FakeQuant(x, delta_in, delta_out, training=False, min_quant=min_quant,
                    max_quant=max_quant, rounding_mode='floor', clamp=True)
    assert (res == out_s).all()
    assert not (res == x).all()
    assert id(res) != id(x)

    # rounding mode trunc , clamp false
    regen_x()
    out_s = x.div(delta_in).trunc()
    res = FakeQuant(x, delta_in, delta_out, training=False, min_quant=None,
                    max_quant=None, rounding_mode='trunc', clamp=False)
    assert (res == out_s).all()
    assert not (res == x).all()
    assert id(res) != id(x)

    # rounding mode trunc , clamp True
    regen_x()
    out_s = x.div(delta_in).trunc().clamp(min=min_quant, max=max_quant)
    res = FakeQuant(x, delta_in, delta_out, training=False, min_quant=min_quant,
                    max_quant=max_quant, rounding_mode='trunc', clamp=True)
    assert (res == out_s).all()
    assert not (res == x).all()
    assert id(res) != id(x)

    # rounding mode round , clamp false
    regen_x()
    out_s = x.div(delta_in).round()
    res = FakeQuant(x, delta_in, delta_out, training=False, min_quant=None,
                    max_quant=None, rounding_mode='round', clamp=False)
    assert (res == out_s).all()
    assert not (res == x).all()
    assert id(res) != id(x)

    # rounding mode round , clamp True
    regen_x()
    out_s = x.div(delta_in).round().clamp(min=min_quant, max=max_quant)
    res = FakeQuant(x, delta_in, delta_out, training=False, min_quant=min_quant,
                    max_quant=max_quant, rounding_mode='round', clamp=True)
    assert (res == out_s).all()
    assert not (res == x).all()
    assert id(res) != id(x)


##########################################################################
########################### QUANT ########################################
##########################################################################
bits = 3
size = (1, 100, 1, 1)
rounding_mode = 'floor'
use_enforced_quant_level = False


compllex_DUT = Quant(bits=bits, size=size, rounding_mode=rounding_mode,
                     use_enforced_quant_level=use_enforced_quant_level)

simple_DUT = Quant(bits=bits, size=(-1,), rounding_mode=rounding_mode,
                   use_enforced_quant_level=use_enforced_quant_level)


def test_init():
    assert compllex_DUT.simple == False
    assert compllex_DUT.bits == bits
    assert compllex_DUT.use_enforced_quant_level == use_enforced_quant_level
    assert compllex_DUT.size == size
    assert compllex_DUT.delta_in.shape == size
    assert compllex_DUT.delta_out.shape == size
    assert compllex_DUT.rounding_mode == rounding_mode
    assert (compllex_DUT.min == torch.ones(size)*(-(2**(bits-1)))).all()
    assert (compllex_DUT.max == torch.ones(size)*((2**(bits-1)-1))).all()
    assert compllex_DUT.permute_list == tuple([1, 0, 2, 3])
    assert compllex_DUT.reduce_list == [0, 2, 3]

    assert simple_DUT.simple == True
    assert simple_DUT.bits == bits
    assert simple_DUT.use_enforced_quant_level == use_enforced_quant_level
    assert simple_DUT.size == (1,)
    assert simple_DUT.delta_in.shape == (1,)
    assert simple_DUT.delta_out.shape == (1,)
    assert simple_DUT.rounding_mode == rounding_mode
    assert (simple_DUT.min == torch.ones((1,))*(-(2**(bits-1)))).all()
    assert (simple_DUT.max == torch.ones((1,))*((2**(bits-1)-1))).all()
    assert simple_DUT.permute_list == tuple([0,])
    assert simple_DUT.reduce_list == []


def test_forward():
    x = torch.rand((10, 100, 12, 34))
    x_clone = x.clone()

    assert (compllex_DUT(x,True) == x_clone).all()
    assert id(compllex_DUT(x,True)) == id(x)


    res = FakeQuant(x.clone(), compllex_DUT.delta_in, compllex_DUT.delta_out,
                    compllex_DUT.training, compllex_DUT.min, compllex_DUT.max, compllex_DUT.rounding_mode)
    assert (res == compllex_DUT(x)).all()
    assert id(x) == id(compllex_DUT(x))

    compllex_DUT.eval()
    res = FakeQuant(x.clone(), compllex_DUT.delta_in, compllex_DUT.delta_out,
                    compllex_DUT.training, compllex_DUT.min, compllex_DUT.max, compllex_DUT.rounding_mode)
    assert (res == compllex_DUT(x)).all()
    assert id(x) != id(compllex_DUT(x))

    m = x.abs().amax(dim=0, keepdim=True).amax(
        dim=2, keepdim=True).amax(dim=3, keepdim=True)
    assert (m == compllex_DUT.get_abs(x)).all()

    res = FakeQuant(x.clone(), simple_DUT.delta_in, simple_DUT.delta_out,
                    simple_DUT.training, simple_DUT.min, simple_DUT.max, simple_DUT.rounding_mode)
    assert (res == simple_DUT(x)).all()
    assert id(x) == id(simple_DUT(x))

    simple_DUT.eval()
    res = FakeQuant(x.clone(), simple_DUT.delta_in, simple_DUT.delta_out,
                    simple_DUT.training, simple_DUT.min, simple_DUT.max, simple_DUT.rounding_mode)
    assert (res == simple_DUT(x)).all()
    assert id(x) != id(simple_DUT(x))

    m = x.abs().amax(dim=0, keepdim=True).amax(
        dim=2, keepdim=True).amax(dim=3, keepdim=True)
    assert (m == simple_DUT.get_abs(x)).all()




##########################################################################
########################### LinQuantExpScale #############################
##########################################################################

mom = 1
LIN_DUT = LinQuantExpScale(bits,size,rounding_mode,use_enforced_quant_level,mom1=mom)


def test_LIN_init():
    assert LIN_DUT.simple == False
    assert LIN_DUT.bits == bits
    assert LIN_DUT.use_enforced_quant_level == use_enforced_quant_level
    assert LIN_DUT.size == size
    assert LIN_DUT.delta_in.shape == size
    assert LIN_DUT.delta_out.shape == size
    assert LIN_DUT.rounding_mode == rounding_mode
    assert (LIN_DUT.min == torch.ones(size)*(-(2**(bits-1)))).all()
    assert (LIN_DUT.max == torch.ones(size)*((2**(bits-1)-1))).all()
    assert LIN_DUT.permute_list == tuple([1, 0, 2, 3])
    assert LIN_DUT.reduce_list == [0, 2, 3]
    assert LIN_DUT.mom1 == mom

    assert LIN_DUT.delta_in_factor == 2/(2**bits-1)
    assert LIN_DUT.delta_out_factor == 2/(2**bits-1)



def test_LIN_forward():
    x = torch.rand((10, 100, 12, 34))
    x_clone = x.clone()

    ma = x.abs().amax(dim=0,keepdim=True).amax(dim=2,keepdim=True).amax(dim=3,keepdim=True)
    delta = ma.log2().ceil().exp2().mul(2/(2**bits-1))
    res = x.clone().div(delta).floor().clamp(LIN_DUT.min,LIN_DUT.max).mul(delta)
    assert (LIN_DUT(x) == res).all()
    assert id(LIN_DUT(x)) == id(x)
