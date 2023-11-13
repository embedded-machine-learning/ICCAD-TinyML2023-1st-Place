import torch
from torch.nn.common_types import Tensor

from ..logger import logger_forward

@logger_forward
def calculate_n_a(
    weight: Tensor,
    mean: Tensor,
    var: Tensor,
    out_quant: Tensor,
    rexp: Tensor,
):
    """
    calculate_n calculates the shift factor and alpha value

    :param weight: The absolute of the weight vector
    :type weight: Tensor
    :param mean: The mean of the BN
    :type mean: Tensor
    :param var: The variance of the BN
    :type var: Tensor
    :param out_quant: The out quantization factor
    :type out_quant: Tensor
    :param rexp: The input exponent
    :type rexp: Tensor
    :return: Return the shift factor
    :rtype: Tensor
    """
    with torch.no_grad():
        n = torch.log2(weight.abs() / (out_quant * torch.sqrt(var + 1e-5)))
        # n = torch.log2(weight.abs() / (out_quant * torch.sqrt(var)))
        n = torch.nan_to_num(n,nan=0,posinf=0,neginf=-32).add(rexp.view(-1)).clip(min=-32,max=0)
        # nr = n
        # alpha = (torch.sign(weight)+1e-5).sign() * torch.exp2(n - nr)
        # return nr, alpha
        nr = n.ceil()
        alpha = (torch.sign(weight)+1e-5).sign() * torch.exp2(n - nr)
        return nr, alpha


@logger_forward
def calculate_n_a_fixed(
    weight: Tensor,
    mean: Tensor,
    var: Tensor,
    out_quant: Tensor,
    rexp: Tensor,
):
    """
    calculate_n calculates the shift factor and alpha value for a whole layer

    :param weight: The absolute of the weight vector
    :type weight: Tensor
    :param mean: The mean of the BN
    :type mean: Tensor
    :param var: The variance of the BN
    :type var: Tensor
    :param out_quant: The out quantization factor
    :type out_quant: Tensor
    :param rexp: The input exponent
    :type rexp: Tensor
    :return: Return the shift factor
    :rtype: Tensor
    """
    with torch.no_grad():
        n = torch.log2(weight.abs() / (out_quant * torch.sqrt(var + 1e-5)))
        n = torch.nan_to_num(n,nan=0,posinf=0,neginf=-32).add(rexp.view(-1)).clip(min=-32,max=0)
        
        # nr = n.max() * torch.ones_like(n)
        nr = n.median() * torch.ones_like(n)
        nr = torch.ceil(nr)
        alpha = (torch.sign(weight)+1e-5).sign() * torch.exp2(n - nr)
        return nr, alpha

@logger_forward
def calculate_t(
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    out_quant: torch.Tensor,
    rexp: torch.Tensor,
    n: torch.Tensor,
) -> torch.Tensor:
    """
    calculate_t calculates the additive value

    :param weight: The absolute of the weight vector
    :type weight: Tensor
    :param bias: The bias of the BN
    :type bias: Tensor
    :param mean: The mean of the BN
    :type mean: Tensor
    :param var: The variance of the BN
    :type var: Tensor
    :param out_quant: The out quantization factor
    :type out_quant: Tensor
    :param rexp: The input exponent
    :type rexp: Tensor
    :return: Return the additive value
    :rtype: Tensor
    """
    with torch.no_grad():
        t = -mean * (weight / (out_quant * torch.sqrt(var + 1e-5))) + bias / out_quant
        return t

