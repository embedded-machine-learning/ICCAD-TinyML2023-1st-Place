import torch.nn as nn
from torch.nn.common_types import _size_4_t

from ..logger import logger_forward,logger_init
from ..DataWrapper import DataWrapper

class ZeroPad2d(nn.ZeroPad2d):#
    @logger_init
    def __init__(self, padding: _size_4_t) -> None:
        super(ZeroPad2d,self).__init__(padding)

    @logger_forward
    def forward(self, input: DataWrapper) -> DataWrapper:
        return input.set(super(ZeroPad2d,self).forward(input.get()[0]),input.get()[1])