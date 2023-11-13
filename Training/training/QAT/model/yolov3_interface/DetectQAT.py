import torch.nn as nn
import torch

from typing import Tuple, Optional

from ..logger import logger_forward, logger_init
from ..DataWrapper import DataWrapper
from ..Conversion import Stop
from ..convolution import Conv2d
from ..blocks import ConvBn,ConvBnA
from ..Quantizer import Quant, FakeQuant
from ..F8NET import F8NET_convolution_weight_quantization
from ..activations import FixPoint

import pkg_resources as pkg


class Detect_LinQuantExpScale(Quant):
    @logger_init
    def __init__(
        self,
        bits: int,
        size: Tuple[int] = (-1,),
        rounding_mode: str = "floor",
        use_enforced_quant_level: bool = False,
        mom1: float = 0.1,
    ) -> None:
        super(Detect_LinQuantExpScale, self).__init__(bits, size, rounding_mode, use_enforced_quant_level)
        if size == (-1,):
            self.register_buffer("abs", torch.ones(1))
        else:
            self.register_buffer("abs", 4*torch.ones(size))
        self.take_new = True
        self.mom1 = mom1
        assert self.bits > 0
        self.register_buffer("delta_in_factor", torch.tensor(2.0 / (2.0**self.bits - 1)))
        self.register_buffer("delta_out_factor", torch.tensor(2.0 / (2.0**self.bits - 1)))

    @logger_forward
    def forward(self, x: torch.Tensor, fake: bool = False, metadata: Optional[DataWrapper] = None):
        if self.training:
            with torch.no_grad():
                # abs_value = self.get_abs(x)
                # # print(abs)
                # self.abs = ((1 - self.mom1) * self.abs + self.mom1 * abs_value).detach()
                # self.abs = self.abs.clamp(-4, 4)

                abs_value = self.abs.log2().ceil().exp2()
                self.delta_in = abs_value.mul(self.delta_in_factor).detach()  # .log2().ceil().exp2()
                self.delta_out = abs_value.mul(self.delta_out_factor).detach()  # .log2().ceil().exp2()
                if self.use_enforced_quant_level and metadata is not None:
                    self.use_quant(metadata)
                if self.use_enforced_quant_level and metadata is None:
                    raise ValueError("Quantization function desired but metadata not passed")

        if fake:
            return x
        return FakeQuant(
            x=x,
            delta_in=self.delta_in,
            delta_out=self.delta_out,
            training=self.training,
            min_quant=self.min,
            max_quant=self.max,
            rounding_mode=self.rounding_mode,
            clamp=False,
        )


def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False):
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    if hard:  # assert min requirements met
        assert result, f'{name}{minimum} required by YOLOv3, but {name}{current} is currently installed'
    else:
        return result


class DetectQAT(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(Conv2d(x,
                                      self.no * self.na,
                                      1,
                                      weight_quant_channel_wise=True,
                                      out_quant_args=(8, (1, self.no * self.na, 1, 1)),
                                      out_quant=FixPoint,
                                      ) for x in ch)  # output conv
        # self.m = nn.ModuleList(ConvBnA(x,
        #                               self.no * self.na,
        #                               1,
        #                             #   weight_quant=F8NET_convolution_weight_quantization,
        #                             #   activation=Detect_LinQuantExpScale,
        #                               activation=FixPoint,
        #                               ) for x in ch)  # output conv
        # self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.stops = nn.ModuleList(Stop((1, self.no * self.na, 1, 1)) for x in ch)  # output conv
        # self.stops = nn.ModuleList(Stop((1, x, 1, 1)) for x in ch)  # output conv

        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            x[i] = self.stops[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for  on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid
