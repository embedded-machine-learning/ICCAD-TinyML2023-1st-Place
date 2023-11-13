import inspect
import re
import os

name_length = 50
result_length = 30


def print_line(i, result=None, additional_indent: int = 0) -> str:
    line = re.split("(\s+)", i)
    name = line[1] + " ".join(i.split()).replace('"', '\\"')
    found_instances = i[:name_length].count('"')
    if len(name) >= name_length + found_instances:
        name = name[: name_length + found_instances - 5] + " ... "

    if result == None:
        out = (
            line[1]
            + " " * (4 * additional_indent)
            + f'print(f"{name[:name_length+found_instances]:<{name_length+found_instances}}")\n'
        )
    else:
        out = (
            line[1]
            + " " * (4 * additional_indent)
            + f"print(\"{name[:name_length+found_instances]:<{name_length+found_instances}}-> {{0}}\".format(''.join(str({result}).split())[:{result_length}]))\n"
        )
    return out


def print_all():
    def func(i) -> list:
        outstr = []
        outstr.append(i)
        outstr.append(print_line(i))
        return outstr

    return func


def print_custom(string, pre: bool = False, additional_indent=0):
    def func(i) -> list:
        outstr = []
        line = re.split("(\s+)", i)
        if pre:
            outstr.append(i)
        outstr.append(
            line[1] + " " * (4 * additional_indent) + f"print(\"{line[1]}{string}\")\n"
        )
        if not pre:
            outstr.append(i)
        return outstr

    return func


def print_left_of(string, pre: bool = True):
    def func(i: str) -> list:
        outstr = []
        if pre:
            outstr.append(i)
        pos = i.find(string)
        if "(" in i[:pos]:
            return [i,]  # early exit to stop priint in brackets
        if "," in i[:pos]:
            outstr.append(print_line(i))
            for val in re.split("(?:,)", i[:pos]):
                outstr.append(print_line(i + ":" + val, val))
        else:
            outstr.append(print_line(i, i[:pos]))
        if not pre:
            outstr.append(i)
        return outstr

    return func


def print_right_of(string, pre: bool = True):
    def func(i: str) -> list:
        outstr = []
        if pre:
            outstr.append(i)
        pos = i.find(string) + len(string)
        outstr.append(print_line(i, " ".join(i[pos:].split())))
        if not pre:
            outstr.append(i)
        return outstr

    return func


def print_between(
    start: str, stop: str, prefix=None, pre: bool = False, result: bool = True, additional_indent: int = 0
):
    def func(i: str) -> list:
        outstr = []
        pos1 = i.find(start)
        pos2 = i.rfind(stop)
        if pre:
            outstr.append(i)
        if prefix == None:
            outstr.append(
                print_line(i, i[pos1 + len(start) : pos2] if result else None, additional_indent=additional_indent)
            )
        else:
            outstr.append(
                print_line(
                    i, prefix + i[pos1 + len(start) : pos2] if result else None, additional_indent=additional_indent
                )
            )
        if not pre:
            outstr.append(i)
        # print(outstr, pre)
        return outstr

    return func


class log_class:
    index = 0
    log_level = 0
    LOGGER_FIILE_PATH = ""

    def __init__(self, LOGGER_FIILE_PATH: str = "") -> None:
        self.LOGGER_FIILE_PATH = LOGGER_FIILE_PATH
        if LOGGER_FIILE_PATH != "":
            if os.path.exists(self.LOGGER_FIILE_PATH):
                os.remove(self.LOGGER_FIILE_PATH)

    def log(self, rules={}):
        def logger(fnc):
            code, cline = inspect.getsourcelines(fnc)
            fnc_name = str()
            outstr = []
            indent_pos = 0
            file = inspect.getfile(fnc)
            old_line = ""
            def_triggered = False
            super_triggered = False
            for ip in range(len(code)):
                i = old_line + code[ip][indent_pos:]
                i = i[: i.rfind("#")]
                if i.count("(") != i.count(")") or i.count("{") != i.count("}") or i.count("[") != i.count("]"):
                    old_line = i
                    continue
                else:
                    old_line = ""
                    i = (4 * " ").join(i.split("\t"))
                    i = " " * (len(i) - len(i.lstrip())) + " ".join(i.split()) + "\n"
                found_in_rules = False
                if "@logger" in i:
                    indent_pos = i.find("@")
                    found_in_rules = True
                if "def" in i and def_triggered == False:
                    def_triggered = True
                    found_in_rules = True
                    outstr.append("#def was triggered\n")
                    pos = i.find("(")
                    outstr.append(i[:pos] + str(self.index) + i[pos:])
                    fnc_name = fnc.__name__ + str(self.index)
                    self.index += 1
                    outstr.append('    print("FUNCTON:' + fnc.__qualname__ + ":" + f"{cline+ip}" + f"\t{file}" + '")\n')

                    super_exists = False
                    for t in range(len(code)):
                        if "super" in code[t]:
                            super_exists = True
                            break

                    if not super_exists:
                        sig = inspect.signature(fnc)
                        # print(sig)
                        for s in sig.parameters:
                            name = "    " + str(sig.parameters[s])
                            addid = name.count('{')+name.count('}')
                            name = name.replace('{}','{{}}')
                            outstr.append(f'    print("{name:<{name_length+addid}}-> {{0}}".format(" ".join(str({s}).split())[:{result_length}]))\n')
                        outstr.append(f'    print(f"  CONTENT:")\n')
                        found_in_rules = True
                    continue
                if "super" in i and "__init__" in i and super_triggered == False:
                    super_triggered = True
                    outstr.append(i)
                    sig = inspect.signature(fnc)
                    # print(sig)
                    outstr.append("#super was triggered\n")
                    for s in sig.parameters:
                        name = "    " + str(sig.parameters[s])
                        addid = name.count('{')+name.count('}')
                        name = name.replace('{}','{{}}')
                        outstr.append(f'    print("{name:<{name_length+addid}}-> {{0}}".format(" ".join(str({s}).split())[:{result_length}]))\n')
                    outstr.append(f'    print(f"  CONTENT:")\n')
                    found_in_rules = True
                    continue
                for case in rules:
                    if case in i:
                        # print(case,i)
                        outstr.extend(rules[case](i))
                        found_in_rules = True
                        break
                if not found_in_rules:
                    outstr.append(i)
            outstr.append(f'    print("  Done with {fnc.__qualname__}")\n')
            if self.LOGGER_FIILE_PATH != "":
                with open(self.LOGGER_FIILE_PATH, "a+") as file:
                    file.write("".join(outstr) + "\n")
            # print("".join(outstr))
            exec("".join(outstr), fnc.__globals__)
            return eval(fnc_name, fnc.__globals__)

        return logger
