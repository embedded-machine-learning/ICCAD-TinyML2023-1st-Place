from .logger import log_class, print_all, print_custom, print_right_of
from .logger import print_left_of, print_between

ENABLE_FORWARD = False
ENABLE_INIT = False
LOGGER_FIILE_PATH = "./logger_redefinitions.py"



logger = log_class(LOGGER_FIILE_PATH)

logger_forward = logger.log(
    {
        " = "    : print_left_of("="),
        " elif"  : print_between("elif", ":", pre=True, additional_indent= 1),                                  
        " if"    : print_between("if", ":"),
        " else"  : print_custom("else Taken", pre=True, additional_indent= 1),
        " with"  : print_between("with", ":", pre=False, result=False),
        " return": print_right_of("return", pre=False),
        "+="     : print_left_of("+="),
        "-="     : print_left_of("-="),
    }
) if ENABLE_FORWARD else lambda x:x

logger_init = logger.log(
    {
        'self.register_buffer("'    : print_between('self.register_buffer("'    , '",', prefix="self.", pre=True),
        'self.register_parameter("' : print_between('self.register_parameter("' , '",', prefix="self.", pre=True),
    }
) if ENABLE_INIT else lambda x: x
