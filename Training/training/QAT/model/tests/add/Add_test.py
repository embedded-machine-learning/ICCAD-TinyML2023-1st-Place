import torch
from typing import Tuple, Optional

from ...add import Add
from ...Quantizer import Quant
from ...DataWrapper import DataWrapper

def test_basic():
    class fake(Quant):
        def __init__(self, bits: int, size: Tuple[int] = ..., rounding_mode: str = "floor", use_enforced_quant_level: bool = False) -> None:
            super().__init__(bits, size, rounding_mode, use_enforced_quant_level)
            self.delta_in.data = 4*torch.ones_like(self.delta_in)
            self.delta_out.data = 4*torch.ones_like(self.delta_in)

        def forward(self, x: torch.Tensor, fake: bool = False, metadata: Optional[DataWrapper] = None) -> torch.Tensor:
            return x.clone()

    DUT = Add((1,),out_quant=fake)
    assert DUT.a_shift.shape == (1,)
    assert DUT.b_shift.shape == (1,)

    a = DataWrapper(torch.rand((12,4,56,23)),torch.ones((1,)))
    b = DataWrapper(torch.rand((12,4,56,23)),2*torch.ones((1,)))

    out = DUT(a.clone(),b.clone())

    assert id(a) != id(b)
    assert id(a) != id(out)
    assert id(b) != id(out)

    assert torch.isclose(a['value']+b['value'],out['value']).all()
    assert torch.isclose(out['rexp'],2*torch.ones((1,))).all()

    DUT.eval()

    out_ev = DUT(a.clone(),b.clone())
    
    assert id(a) != id(b)
    assert id(a) != id(out)
    assert id(b) != id(out)

    assert torch.isclose((a['value']/2+b['value']).floor(),out_ev['value']).all()
    assert torch.isclose(out_ev['rexp'],2*torch.ones((1,))).all()



