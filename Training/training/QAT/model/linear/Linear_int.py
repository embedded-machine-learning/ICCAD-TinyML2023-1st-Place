import torch
from torch import nn
from torch.nn.common_types import Tensor
import torch.nn.functional as F

from .. import __FLAGS__


class Linear_int(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        quant_weights: Tensor,
        shift: Tensor,
        bias: Tensor,
        Act_min: Tensor,
        Act_max: Tensor,
        accumulation_type=torch.int32,
        small_signed_type=torch.int8,
        small_unsigned_type=torch.uint8,
    ) -> None:
        super().__init__(in_features, out_features, False)
        self.weight.requires_grad_(False)
        self.weight.data = quant_weights.to(small_signed_type)
        self.register_buffer("n", shift.to(accumulation_type))
        self.register_buffer("n_eq_mult", shift.exp2())
        if bias is not None:
            self.register_buffer("t", bias.to(accumulation_type))
        else:
            self.t = None
        self.register_buffer("min", Act_min.to(accumulation_type))
        self.register_buffer("max", Act_max.to(accumulation_type))

        self.pure_positive = torch.all(Act_min>=0).cpu().numpy()

        self.accumulation_type = accumulation_type
        self.small_signed_type = small_signed_type
        self.small_unsigned_type = small_unsigned_type

    def forward(self, x: Tensor) -> Tensor:

        if __FLAGS__["ONNX_EXPORT"]:
            out = F.linear(x.float(), self.weight.float(), None).to(self.accumulation_type)
        else:
            out = F.linear(x.to(self.accumulation_type), self.weight.to(self.accumulation_type), None)


        if __FLAGS__["ONNX_EXPORT"]:
            out = out.to(torch.float).mul(self.n_eq_mult).floor().to(self.accumulation_type)
        else:
            out = torch.bitwise_right_shift(out, -self.n)

        if self.t is not None:
            out = out + self.t
        
        out = out.clamp(self.min, self.max)

        if self.pure_positive:
            out = out.to(self.small_unsigned_type)
        else:
            out = out.to(self.small_signed_type)

        return out
