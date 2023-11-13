__HIGH_PRES__ = False

"""
Enables high precision mode, network will take longer and will need more memory
"""


__HIGH_PRES_USE_RUNNING__ = False

"""
In High precision mode use the running stats of the bn in training
"""


__DEBUG__ = False
"""
Enables DEBUG code segments
"""


__ONNX_EXPORT__ = False
"""
If True changes the shift to a float multiplication, on the int exported model
"""

__FLAGS__ = {
    'ONNX_EXPORT': False,
}
__TESTING_FLAGS__ = {
    'FREEZE_BN': False,
    'FUZE_BN': False,
    'FREEZE_WEIGHT_QUANT': False,
    'FREEZE_ACT_QUANT': False,
    'FREEZE_QUANT':False,
}
