import sys
assert sys.version_info[0] >= 3 and sys.version_info[1] >= 9, "Must be run with Python >=3.9!"

from prettytable import PrettyTable
from typing import TypeVar, Union, Any
from types import FrameType
import importlib.util
import signal
import code
import copy
import time
import os

from architectures.architectures import *
from computations import *
from settings import *
from factors import *
from engine import *
from levels import *
from prints import *
from utils import *

#from architectures.solutions_db import *
#from comparisons.ZigZag.zigzag_archs import *
#from comparisons.CoSA.cosa_archs import *


# CLI MANAGEMENT

in_interactive_mode = False

def signal_handler(signal: int, frame: Optional[FrameType]) -> None:
    global in_interactive_mode
    if in_interactive_mode:
        print('EXITING...')
        sys.exit(0)
    else:
        print('\nHANDLING TERMINATION...\n')
        time.sleep(0.2)
        print('\nTERMINATION RECEIVED - SWITCHING TO INTERACTIVE MODE\n[type "exit()" or press "ctrl+c" again to terminate the program]\n')
        in_interactive_mode = True
        code.interact(local=globals())
        in_interactive_mode = False


T = TypeVar('T')

"""
Searchs and removes flags from sys.argv.
If 'with_value' is False, the return values is either True or False depending on the presence or absence of the option.
If 'with_value' is True, the return value is the value assigned with the option, if present, otherwise it is False if
the option is not present and None if no valid argument was provided.
Optionally, 'value_type' can be used to parse the desired value when 'with_value' is True.
"""
def args_match_and_remove(flag : str, with_value : bool = False, value_type : type[T] = str) -> Union[bool, T]:
    try:
        idx = sys.argv.index(flag)
        sys.argv.pop(idx)
        if with_value:
            try:
                value = value_type(sys.argv[idx])
                sys.argv.pop(idx)
                return value
            except:
                return None
        else:
            return True
    except:
        return False
    
def parse_options() -> dict[str, Any]:
    options = {
        "help": args_match_and_remove("-h") or args_match_and_remove("--help"),
        "quiet": args_match_and_remove("-q") or args_match_and_remove("--quiet"),
        "bias": args_match_and_remove("-b") or args_match_and_remove("--bias"),
        "threads": v if (v := args_match_and_remove("-t", True, int)) != False else args_match_and_remove("--threads", True, int),
        "accelergy-data": args_match_and_remove("-ad") or args_match_and_remove("--accelergy-data"),
        "tryall": v if (v := args_match_and_remove("-ta", True)) != False else args_match_and_remove("--tryall", True),
        "gen-tests": args_match_and_remove("-gt") or args_match_and_remove("--gen-tests")
    }
    return options

def help_options() -> None:
    print("Supported options:")
    print("-h, --help\t\tDisplay this help menu.")
    print("-q --quiet\t\tReduces CLI output to final results and errors.")
    print("-b --bias\t\tIf set, the bias is considered present in the GEMM, otherwise it is assumed absent.")
    print("-t --threads\t\tSets the number of concurrent threads to use (yes, threads, remember to use Python >=3.13t!).")
    print("-ad --accelergy-data\tQuery Accelergy instead of using hardcoded component level estimates, effective only if arg. 1 is an architecture name.")
    print("-ta --tryall <name>\tOverrides normal execution, runs FF for all known architectures and all kernels in the <name> group. The list of valid group names is printed when no name is provided.")
    print("-gt --gen-tests\t\tOverrides normal execution, runs FF and generates tests to enforce the obtained results.")
    # TODO: add option to print MOPs per instance!

def help_arch(supported_archs : dict[str, Arch]) -> None:
    print("The first argument should be a valid architecture name or a path to file specifying the architecture.\nValid architecture names are the following:")
    for arch_name in supported_archs.keys():
        print(f"- {arch_name}")
    print("Alternatively, provide a \"path/to/a/file.py\" where a variable \"arch\" is defined. An example of such a file is provided under \"architectures/example_arch.py\".")
    print("Be aware that each architecture is built for a certain coupling, and supports only computations relying on a valid subcoupling of that one.")

def help_comp(supported_comp_groups : dict[str, dict[str, Shape]]) -> None:
    print("The second argument should be a valid computation name, otherwise arguments two-to-four, two-to-seven, or two-to-eleven can be a triple, sextuple, or decuple of positive integers specifying the three, six, or ten dimensions of a GEMM (in order: M, K, N), convolution (in order: M, P, Q, C, R, S), or strided convolution (in order: M, P, Q, C, R, S, Pstride, Qstride, Rdilation, Sdilation) respectively.\nValid computation names are the following:")
    for group_name, group in supported_comp_groups.items():
        for comp_name, comp in group.items():
            print(f"- {group_name}-{comp_name} ->", ", ".join(f"{k}: {v}" for k, v in comp.items()))
    print("Alternatively, provide a \"path/to/a/file.py\" where the variables \"coupling\" and \"comp\" are defined. An example of such a file is provided under \"architectures/example_comp.py\".")
    print("Be wary that the coupling must be compatible with the one used for the selected architecture.")

def printopt(*argv, **kwargs) -> None:
    if not options["quiet"]:
        print(*argv, **kwargs)


# DEFAULT SPECIFICATION:

comp = comp_BERT_large['KQV']
coupling = gemm_coupling
bias_read = False # True if bias is not 0 - outputs are read even the first time

arch = arch_eyeriss


## MAIN:

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    options = parse_options()
    
    supported_archs = {"gemmini": arch_gemmini, "eyeriss": arch_eyeriss, "simba": arch_simba, "tpu": arch_tpu, "gemmini-conv": arch_gemmini_conv, "eyeriss-conv": arch_eyeriss_conv, "simba-conv": arch_simba_conv, "tpu-conv": arch_tpu_conv}
    if options["accelergy-data"]:
        from architectures.architectures_hw_data import *
        supported_archs_accelergy = {"gemmini": get_arch_gemmini_hw_data, "eyeriss": get_arch_eyeriss_hw_data, "simba": get_arch_simba_hw_data, "tpu": get_arch_tpu_hw_data}
    supported_comp_groups = {'BERT': comp_BERT_large, 'MB': comp_maestro_blas, 'VGG16': comp_vgg_16, 'ResNet18': comp_resnet_18}
    supported_couplings = {'BERT': gemm_coupling, 'MB': gemm_coupling, 'VGG16': conv_coupling, 'ResNet18': conv_coupling_with_stride}
    
    if options["help"]:
        print("------------ HELP ------------")
        help_options()
        print("-------- architecture --------")
        help_arch(supported_archs)
        print("-------- computation  --------")
        help_comp(supported_comp_groups)
        print("------------------------------")
        sys.exit(1)
    
    if options["tryall"] == False:
        printopt("-------- architecture --------")
        if not (len(sys.argv) >= 2 and (sys.argv[1] in supported_archs or os.path.exists(sys.argv[1]) and sys.argv[1][-3:] == '.py')):
            if not options["quiet"]:
                help_arch(supported_archs)
            #sys.exit(1)
            print("WARNING: no valid architecture provided, defaulting to \"eyeriss\"...")
        if len(sys.argv) >= 2:
            if sys.argv[1] in supported_archs:
                if options["accelergy-data"]:
                    arch = supported_archs_accelergy[sys.argv[1]]()
                    printopt()
                else:
                    arch = supported_archs[sys.argv[1]]
                    printopt("Architecture:", sys.argv[1])
            elif sys.argv[1][-3:] == '.py':
                arch_file = importlib.util.spec_from_file_location("user_arch", sys.argv[1])
                arch_module = importlib.util.module_from_spec(arch_file)
                sys.modules["user_arch"] = arch_module
                arch_file.loader.exec_module(arch_module)
                arch = getattr(arch_module, "arch")
                printopt("Architecture:", sys.argv[1], "-> arch")
            sys.argv.pop(1)
        
        printopt("-------- computation  --------")
        if not ((len(sys.argv) >= 2 and ((len(split_comp_name := sys.argv[1].split('-')) == 2 and split_comp_name[0] in supported_comp_groups and split_comp_name[1] in supported_comp_groups[split_comp_name[0]]) or os.path.exists(sys.argv[1]) and sys.argv[1][-3:] == '.py')) or (len(sys.argv) == 4 and all([d.isdigit() and d[0] != '-' for d in sys.argv[1:4]])) or (len(sys.argv) >= 7 and all([d.isdigit() and d[0] != '-' for d in sys.argv[1:7]]))):
            if not options["quiet"]:
                help_comp(supported_comp_groups)
            #sys.exit(1)
            print("WARNING: no valid computation provided, defaulting to GEMM: \"BERT-KQV\"...")
        if len(sys.argv) == 2:
            if len(split_comp_name := sys.argv[1].split('-')) == 2 and split_comp_name[0] in supported_comp_groups and split_comp_name[1] in supported_comp_groups[split_comp_name[0]]:
                comp = supported_comp_groups[split_comp_name[0]][split_comp_name[1]]
                coupling = supported_couplings[split_comp_name[0]]
                printopt("Computation:", sys.argv[1]+":", comp)
                printopt("Coupling:", coupling.compactStr())
            elif sys.argv[1][-3:] == '.py':
                comp_file = importlib.util.spec_from_file_location("user_comp", sys.argv[1])
                comp_module = importlib.util.module_from_spec(comp_file)
                sys.modules["user_comp"] = comp_module
                comp_file.loader.exec_module(comp_module)
                comp = getattr(comp_module, "comp")
                coupling = getattr(comp_module, "coupling")
                printopt("Computation:", sys.argv[1], "-> comp:", comp)
                printopt("Coupling:", sys.argv[1], "-> coupling:", coupling.compactStr())
        elif len(sys.argv) > 2:
            if len(sys.argv) == 4:
                comp = Shape(M = int(sys.argv[1]), K = int(sys.argv[2]), N = int(sys.argv[3]))
                coupling = gemm_coupling
            elif len(sys.argv) == 7:
                comp = Shape(M = int(sys.argv[1]), P = int(sys.argv[2]), Q = int(sys.argv[3]), C = int(sys.argv[4]), R = int(sys.argv[5]), S = int(sys.argv[6]))
                coupling = conv_coupling
            elif len(sys.argv) >= 11:
                comp = Shape(M = int(sys.argv[1]), P = int(sys.argv[2]), Q = int(sys.argv[3]), C = int(sys.argv[4]), R = int(sys.argv[5]), S = int(sys.argv[6]), Pstride = int(sys.argv[7]), Qstride = int(sys.argv[8]), Rdilation = int(sys.argv[9]), Sdilation = int(sys.argv[10]))
                coupling = conv_coupling_with_stride
            printopt("Computation:", comp)
            printopt("Coupling:", coupling.compactStr())
    else:
        printopt("-------- try all mode --------")
        if options["tryall"] not in supported_comp_groups:
            print(f"Invalid <name> argument provided for the 'tryall' option ({options['tryall']}). Valid group names are:")
            for group_name in supported_comp_groups.keys():
                print(f"- {group_name}")
            sys.exit(1)
        else:
            printopt("Comp group:", options["tryall"])
    
    bias_read = options["bias"]
    printopt("Bias present:", bias_read)
    
    printopt("------------------------------")
    printopt("----- running FactorFlow -----")
    printopt("------------------------------\n")

    if options["threads"]:
        Settings.THREADS_COUNT = options["threads"]
        if options["threads"] == 1:
            Settings.MULTITHREADED = False
    printopt("Threads:", Settings.THREADS_COUNT)
    
    if options["quiet"]:
        Settings.VERBOSE = False
    
    # EXECUTION
    
    if options["tryall"]:
        extra_constant_columns_names = []#["N"]
        extra_constant_columns_values = []#["4"]
        
        Settings.VERBOSE = False
        comp_BERT_large.pop("Out")
        comp_BERT_large.pop("FF2")
        table = PrettyTable(["Arch", "Comp", "EDP[J*cycle]", "MOPs", "Latency[cc]", "Energy[uJ]", "Utilization[/]", "Runtime"] + extra_constant_columns_names)
        for arch_name, current_arch in zip(["Gemmini", "Eyeriss", "Simba", "TPUv1"], [arch_gemmini, arch_eyeriss, arch_simba, arch_tpu] if supported_couplings[options["tryall"]] is gemm_coupling else [arch_gemmini_conv, arch_eyeriss_conv, arch_simba_conv, arch_tpu_conv]):
            for comp_name, current_comp in zip(supported_comp_groups[options["tryall"]].keys(), supported_comp_groups[options["tryall"]].values()):
                current_arch_copy = copy.deepcopy(current_arch)
                printopt(f"Now running FactorFlow on arch: {arch_name} and comp: {comp_name}...")
                if not current_arch_copy.fitConstraintsToComp(current_comp, comp_name):
                    continue
                edp, mops, energy, latency, utilization, end_time, _ = run_engine(current_arch_copy, current_comp, gemm_coupling, bias_read, verbose = False)
                table.add_row([arch_name, comp_name, f"{edp:.3e}", f"{mops[0]+mops[1]:.0f}", f"{latency:.3e}", f"{energy:.3e}", f"{utilization:.3e}", f"{end_time:.3f}"] + extra_constant_columns_values)
        print(table)
    
    elif options["gen-tests"]:
        arch.checkCouplingCompatibility(coupling, comp, verbose = not options["quiet"])
        arch.fitConstraintsToComp(comp, enforce = True)
        edp, mops, energy, latency, utilization, _, arch = run_engine(arch, comp, coupling, bias_read, verbose = not options["quiet"])
        from test import generateTestMOPs, generateTestLatency
        print("\nGenerated tests:")
        generateTestMOPs(arch)
        generateTestLatency(arch)
    
    else:
        arch.checkCouplingCompatibility(coupling, comp, verbose = not options["quiet"])
        arch.fitConstraintsToComp(comp, enforce = True)
        run_engine(arch, comp, coupling, bias_read, verbose = True)
