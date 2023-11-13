# Generic Type imports
from typing import Tuple, Union, Optional

# Torch imports
import torch
from torch import nn
from torch.nn.common_types import Tensor

# Self init
from .logger import logger_init, logger_forward
from .Quantizer import FakeQuant
from .DataWrapper import DataWrapper


class Start_int(nn.Module):
    @logger_init
    def __init__(
        self,
        delta_in: Tensor,
        min: Tensor,
        max: Tensor,
        accumulation_type=torch.int32,
        small_signed_type=torch.int8,
        small_unsigned_type=torch.uint8,
    ) -> None:
        super(Start_int, self).__init__()
        self.register_buffer("delta_in", delta_in)
        self.register_buffer("min", min.to(accumulation_type))
        self.register_buffer("max", max.to(accumulation_type))
        self.accumulation_type = accumulation_type
        self.small_signed_type = small_signed_type
        self.small_unsigned_type = small_unsigned_type

    def forward(self, x: Tensor) -> Tensor:
        x = x.div(self.delta_in, rounding_mode="floor")
        x = x.to(self.max.dtype)
        x = x.clamp(self.min, self.max).to(self.small_signed_type)
        return x


class Start(nn.Module):
    """
    Start Transforms passed values into the quantized/fake quantized domain

    mode "auto" detects input range, assumed is symmetric, offset 0, or pure positive.

    :param bits: The number of desired bits, defaults to 8
    :type bits: int, optional
    :param size: The shape of the quantization, defaults to (1,)
    :type size: Tuple[int], optional
    :param mode: Defines if the quantization range should be extracted during runtime or set to [-.5,.5], defaults to "auto"
    :type mode: Union[str, NoneType], optional
    :param auto_runs: Number of epochs to fixate the auto range, defaults to 2
    :type auto_runs: int, optional
    """

    @logger_init
    def __init__(
        self, bits: int = 8, size: Tuple[int] = (1,), mode: Optional[str] = "auto", auto_runs: int = 2, inline:bool=True
    ) -> None:
        """
        Please read Class help
        """
        super(Start, self).__init__()
        self.size = size
        self.inline=inline
        self.register_buffer(
            "bits", (bits) * torch.ones(size, dtype=torch.float))
        self.register_buffer("delta_in", torch.clone(
            (1.0 / (2.0 ** (self.bits) - 1))).detach())
        self.register_buffer("delta_out", torch.clone(
            (1.0 / (2.0 ** (self.bits) - 1))).detach())
        self.register_buffer("max", 2 ** (-self.bits - 1) - 1)
        self.register_buffer("min", -(2 ** (-self.bits - 1)))

        self.only_positive = False

        self.mode = mode
        self.auto_runs = auto_runs
        self.last_run_train = True
        self.register_buffer("in_max", torch.Tensor([-10000.0]))
        self.register_buffer("in_min", torch.Tensor([10000.0]))

    def _out_pure_positive(self):
        self.min = torch.zeros_like(self.min)
        self.max = 2 ** (self.bits) - 1
        if self.only_positive == False:
            print("switching input to only positive values")
        self.only_positive = True

    def _out_pos_and_neg(self):
        self.min = - (2 ** (self.bits - 1))
        self.max = 2 ** (self.bits - 1) - 1
        if self.only_positive == True:
            print("switching input to mixed values")
        self.only_positive = False

    def int_extract(
        self, accumulation_type=torch.int32, small_signed_type=torch.int8, small_unsigned_type=torch.uint8
    ) -> Start_int:
        return Start_int(
            self.delta_in,
            self.min,
            self.max,
            accumulation_type=accumulation_type,
            small_signed_type=small_signed_type,
            small_unsigned_type=small_unsigned_type,
        )

    def train(self, mode: bool = True):
        if mode == False and self.last_run_train and self.auto_runs > 0:
            self.last_run_train = False
            self.auto_runs -= 1
            print("reduce autorun by 1:", self.auto_runs,
                  'min/max', self.in_min.item(), '/', self.in_max.item())
        return super().train(mode)

    @logger_forward
    def forward(self, x: Tensor) -> DataWrapper:
        with torch.no_grad():
            if self.training:
                if self.mode == "auto" and self.auto_runs > 0:
                    self.last_run_train = True
                    tmp_in_min = torch.min(torch.min(x), self.in_min)
                    tmp_in_max = torch.max(torch.max(x), self.in_max)

                    if torch.all(tmp_in_min == 0) and torch.all(tmp_in_max == 0):
                        print('error state detected')
                        self.auto_runs += 2
                        tmp_in_min = tmp_in_min-1
                        tmp_in_max = tmp_in_max+1
                        rang = 2 * \
                            (torch.max(torch.abs(tmp_in_min), torch.abs(tmp_in_max)))
                    else:
                        self.in_min = tmp_in_min
                        self.in_max = tmp_in_max
                        rang = 2 * \
                            (torch.max(torch.abs(self.in_min), torch.abs(self.in_max)))

                    if self.in_min >= 0 and self.in_max > 0:
                        rang = self.in_max
                        self._out_pure_positive()
                    else:
                        self._out_pos_and_neg()

                    tmp = rang / (2.0 ** (self.bits) - 1)
                    tmp = torch.exp2(torch.log2(tmp))
                    self.delta_in = tmp
                    self.delta_out = tmp
                    # self.delta_in = rang / (2.0 ** (self.bits) - 1)
                    # self.delta_out = rang / (2.0 ** (self.bits) - 1)
            elif self.in_min > self.in_max:
                print(
                    "running undefined Start block using expensive computation at runtime")
                x_min = x.min()
                x_max = x.max()
                rang = 2 * (torch.max(torch.abs(x_min), torch.abs(x_max)))
                delta_in = rang / (2.0 ** (self.bits) - 1)
                return DataWrapper(
                    FakeQuant(
                        x.clone(),
                        delta_in,
                        delta_in,
                        self.training,
                        x_min,
                        x_max,
                        'floor',
                    ),
                    torch.log2(delta_in),
                )

        return DataWrapper(
            FakeQuant(
                x = x if self.inline else x.clone(),
                delta_in = self.delta_in,
                delta_out = self.delta_out,
                training = self.training,
                min_quant =  self.min,
                max_quant = self.max,
                rounding_mode = 'floor',
                clamp = True,
            ),
            torch.log2(self.delta_out),
        )


class Stop_int(nn.Module):
    def __init__(self, rexp) -> None:
        super(Stop_int, self).__init__()
        self.register_buffer("rexp", rexp)
        self.register_buffer("mult_factor", rexp.exp2())

    def forward(self, x: Tensor) -> Tensor:
        # print(x)
        # print(self.rexp)
        return x.to(torch.float32).mul(self.mult_factor)


class Stop(nn.Module):
    """
    Stop Return a Tensor pair from the fake-quantized/quantized domain

    :param size: The shape of the output, defaults to (1,)
    :type size: Tuple[int], optional
    """

    @logger_init
    def __init__(self, size: Tuple[int] = (1,)) -> None:
        """
        Please read Class help
        """
        super(Stop, self).__init__()
        self.size = size
        self.register_buffer("exp", torch.zeros(self.size))
        # Only required to know the current datatype
        self.register_buffer("for_dtype", torch.zeros(1))

    def int_extract(self, accumulation_type=torch.int32, small_signed_type=torch.int8, small_unsigned_type=torch.uint8) -> Stop_int:
        return Stop_int(self.exp)

    @logger_forward
    def forward(self, invals: DataWrapper) -> Tensor:
        x, rexp = invals.get()
        self.exp = rexp.detach().clone()
        with torch.no_grad():
            if not self.training:
                shape = [1 for _ in range(len(x.shape))]
                shape[1] = -1
                x = x.to(self.for_dtype.dtype).mul(rexp.exp2().view(*shape))
        return x
