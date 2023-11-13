from typing import Optional

import torch.nn as nn
from torch.nn.common_types import _size_any_t,_ratio_any_t

from ..logger import logger_forward,logger_init
from ..DataWrapper import DataWrapper

class Upsample(nn.Upsample):
    @logger_init
    def __init__(self, size: Optional[_size_any_t] = None, scale_factor: Optional[_ratio_any_t] = None, mode: str = 'nearest', align_corners: Optional[bool] = None, recompute_scale_factor: Optional[bool] = None) -> None:
        super(Upsample,self).__init__(size, scale_factor, mode, align_corners, recompute_scale_factor)
        assert mode == 'nearest'

    @logger_forward
    def forward(self, input: DataWrapper) -> DataWrapper:
        return input.set(super(Upsample,self).forward(input.get()[0]),input.get()[1])