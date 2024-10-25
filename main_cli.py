from prettytable import PrettyTable
import importlib.util
import signal
import code
import copy
import time
import sys
import os

from architectures.architectures import *
from computations import *
from settings import *
from factors import *
from engine import *
from levels import *
from prints import *
from utils import *

#from comparisons.ZigZag.zigzag_archs import *
#from comparisons.CoSA.cosa_archs import *
#from comparisons.MAESTRO.maestro_archs import *


# CLI MANAGEMENT

in_interactive_mode = False

def signal_handler(sig, frame):
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

signal.signal(signal.SIGINT, signal_handler)

def args_match_and_remove(flag, with_value = False):
    try:
        idx = sys.argv.index(flag)
        sys.argv.pop(idx)
        if with_value:
            return sys.argv.pop(idx)
        else:
            return True
        return
    except:
        return False
    
def parse_options():
    options = {
        "help": args_match_and_remove("-h") or args_match_and_remove("--help"),
        "bias": args_match_and_remove("-b") or args_match_and_remove("--bias"),
        "processes": args_match_and_remove("-p", True) or args_match_and_remove("--processes", True),
        "tryall": args_match_and_remove("-ta") or args_match_and_remove("--tryall"),
        "gen-tests": args_match_and_remove("-gt") or args_match_and_remove("--gen-tests")
    }
    return options

def help_options():
    print("Supported options:")
    print("-h, --help\t\tDisplay this help menu.")
    print("-b --bias\t\tIf set, the bias is considered present in the GEMM, otherwise it is assumed absent.")
    print("-p --processes\t\tSets the number of concurrent processes to use.")
    print("-ta --tryall\t\tOverrides normal execution, runs FF for all known architectures and GEMMs.")
    print("-gt --gen-tests\t\tOverrides normal execution, runs FF and generates tests to enforce the obtained results.")

def help_arch(supported_archs):
    print("The first argument should be a valid architecture name or a path to file specifying the architecture.\nValid architecture names are the following:")
    for arch_name in supported_archs.keys():
        print(f"- {arch_name}")
    print("Alternatively, provide a \"path/to/a/file.py\" where a variable \"arch\" is defined. An example of such a file is provided under \"architectures/example_arch.py\".")

def help_comp(supported_comps):
    print("The second argument should be a valid computation name or a triplet of integers specifying the three dimension of a GEMM.\nValid computation names are the following:")
    for name, comp in supported_comps.items():
        print(f"- {name} -> ", f"M: {comp.M}, K: {comp.K}, N: {comp.N}")
    print("Alternatively, arguments two to four can be positive integers representing the three dimensions of a GEMM, in order: M, K, and N.")


# DEFAULT SPECIFICATION:

comp = comp_BERT_large['KQV']
bias_read = False # True if bias is not 0 - outputs are read even the first time

arch = arch_eyeriss


## MAIN:

if __name__ == "__main__":
    options = parse_options()
    
    supported_archs = {"gemmini": arch_gemmini, "eyeriss": arch_eyeriss, "simba": arch_simba, "tpu": arch_tpu}
    supported_comps = comp_BERT_large | comp_maestro_blas
    
    if options["help"]:
        print("------------ HELP ------------")
        help_options()
        print("-------- architecture --------")
        help_arch(supported_archs)
        print("-------- computation  --------")
        help_comp(supported_comps)
        print("------------------------------")
        sys.exit(1)
        
    if not (len(sys.argv) >= 2 and (sys.argv[1] in supported_archs or os.path.exists(sys.argv[1]))):
        help_arch(supported_archs)
        #sys.exit(1)
        print("WARNING: no architecture provided, defaulting to \"eyeriss\"...\n")
    if len(sys.argv) >= 2:
        if sys.argv[1] in supported_archs:
            arch = supported_archs[sys.argv[1]]
            print("Architecture:", sys.argv[1])
        else:
            arch_file = importlib.util.spec_from_file_location("user_arch", sys.argv[1])
            arch_module = importlib.util.module_from_spec(arch_file)
            sys.modules["user_arch"] = arch_module
            arch_file.loader.exec_module(arch_module)
            arch = getattr(arch_module, "arch")
            print("Architecture:", sys.argv[1], "-> arch")
        sys.argv.pop(1)
    
    if not ((len(sys.argv) >= 2 and sys.argv[1] in supported_comps) or (len(sys.argv) >= 4 and all([d.isdigit() and d[0] != '-' for d in sys.argv[1:4]]))):
        help_comp(supported_comps)
        print("WARNING: no computation provided, defaulting to \"KQV\"...\n")
        #sys.exit(1)
    if len(sys.argv) == 2:
        comp = supported_comps[sys.argv[1]]
        print("Computation:", sys.argv[1])
    elif len(sys.argv) > 2:
        comp = Shape(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
        print("Computation:", comp)
    
    bias_read = options["bias"]
    
    if options["processes"]:
        Settings.THREADS_COUNT = options["processes"]
    
    # EXECUTION
    
    if options["tryall"]:
        extra_constant_columns_names = []#["N"]
        extra_constant_columns_values = []#["4"]
        
        Settings.VERBOSE = False
        comp_BERT_large.pop("Out")
        comp_BERT_large.pop("FF2")
        table = PrettyTable(["Arch", "Comp", "EDP[J*cycle]", "MOPs", "Latency[cc]", "Energy[uJ]", "Utilization[/]", "Runtime"] + extra_constant_columns_names)
        for arch_name, current_arch in zip(["Gemmini", "Eyeriss", "Simba", "TPUv1"], [arch_gemmini, arch_eyeriss, arch_simba, arch_tpu]):
            for comp_name, current_comp in zip(list(comp_BERT_large.keys()) + list(comp_maestro_blas.keys()), list(comp_BERT_large.values()) + list(comp_maestro_blas.values())):
            #for comp_name, current_comp in zip(list(comp_BERT_large.keys())[:1] + list(comp_maestro_blas.keys())[:1], list(comp_BERT_large.values())[:1] + list(comp_maestro_blas.values())[:1]):
            #for comp_name, current_comp in zip(list(comp_BERT_large.keys())[:4], list(comp_BERT_large.values())[:4]):
            #for comp_name, current_comp in zip(list(comp_maestro_blas.keys())[:2], list(comp_maestro_blas.values())[:2]):
                current_arch_copy = copy.deepcopy(current_arch)
                print(f"Now running FactorFlow on arch: {arch_name} and comp: {comp_name}...")
                if fitConstraintsToComp(current_arch_copy, current_comp, arch_name, comp_name):
                    continue
                edp, mops, energy, latency, utilization, end_time, _ = run_engine(current_arch_copy, current_comp, bias_read, verbose = False)
                table.add_row([arch_name, comp_name, f"{edp:.3e}", f"{mops[0]+mops[1]:.0f}", f"{latency:.3e}", f"{energy:.3e}", f"{utilization:.3e}", f"{end_time:.3f}"] + extra_constant_columns_values)
        print(table)
        
    elif options["gen-tests"]:
        #Here changing settings does not propagate to processes, which reimport and reset settings.py
        #Settings.forcedSettingsUpdate(arch)
        fitConstraintsToComp(arch, comp)
        edp, mops, energy, latency, utilization, _, arch = run_engine(arch, comp, bias_read, verbose = True)
        from test import generateTestMOPs, generateTestLatency
        print("\nGenerated tests:")
        generateTestMOPs(arch)
        generateTestLatency(arch)
    
    else:
        #Here changing settings does not propagate to processes, which reimport and reset settings.py
        #Settings.forcedSettingsUpdate(arch)
        fitConstraintsToComp(arch, comp)
        run_engine(arch, comp, bias_read, verbose = True)