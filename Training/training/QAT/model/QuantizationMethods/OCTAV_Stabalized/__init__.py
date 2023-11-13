# https://proceedings.mlr.press/v162/sakr22a/sakr22a.pdf
# is the base, added an EMA filter on top to stabilize,
#  also increased number of iteration 
from .convolution_weight_quantization import LinQuantWeight_mod_OCTAV_Stabalized as OCTAV_Stabalized_convolution_weight_quantization
from .linear_weight_quantization import LinQuantWeight_mod_OCTAV_Stabalized as OCTAV_Stabalized_linear_weight_quantization