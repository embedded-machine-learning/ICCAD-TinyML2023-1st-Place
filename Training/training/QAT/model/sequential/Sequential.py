import torch
from torch import nn

from .Sequential_int import Sequential_int


class Sequential(nn.Sequential):
    def int_extract(self, accumulation_type=torch.int32, small_signed_type=torch.int8, small_unsigned_type=torch.uint8) -> nn.Sequential:
        out = Sequential_int()
        for module in self:
            if callable(getattr(module, 'int_extract', None)):
                tmp = module.int_extract(accumulation_type=accumulation_type, small_signed_type=small_signed_type, small_unsigned_type=small_unsigned_type)
                if tmp is not None:
                    out.append(tmp)
        return out
