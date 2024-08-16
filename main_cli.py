from prettytable import PrettyTable
import multiprocessing
import signal
import code
import copy
import time
import sys

from architectures import *
from solutions_db import *
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



# RUN FF

def basic_run(arch, comp, bias_read, verbose = False):
    start_time = time.time()

    if Settings.MULTITHREADED:
        manager = multiprocessing.Manager()
        return_list = manager.list([None]*Settings.THREADS_COUNT)
        processes = []
        for i in range(Settings.THREADS_COUNT):
            p = multiprocessing.Process(target=optimizeDataflows, args=(copy.deepcopy(arch), copy.deepcopy(comp), bias_read, i, Settings.THREADS_COUNT, return_list, Settings.VERBOSE))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        assert None not in return_list, f"Some threads failed to return or found no valid mapping, see above logs..."
        arch, wart, _ = max(return_list, key=lambda res : res[1])
    else:
        #arch, _ = factorFlow(arch, comp, bias_read, verbose = VERBOSE)
        arch, wart, _ = optimizeDataflows(arch, comp, bias_read, verbose = Settings.VERBOSE)
    
    end_time = time.time() - start_time

    edp = EDP(arch, bias_read, True)
    energy = Energy(arch, True)
    latency = Latency(arch)

    if verbose:
        print(f"\nFinished in: {end_time:.3f}s")

        print(f"\nBest mapping found with:\n\tWart: {wart:.3e}\n\tEDP: {edp:.3e} (J*cycle)\n\tEnergy: {energy:.3e} (uJ)\n\tLatency: {latency:.3e} (cc)")
        printFactors(arch)

        print("\nFinal MOPs per memory level:")
        printMOPsNew(arch)
        print("\nFinal Latency per level:")
        printLatencyNew(arch)
    
        if Settings.PADDED_MAPPINGS:
            print("")
            printPadding(arch, comp)

    return edp, energy, latency, end_time


# DEFAULT SPECIFICATION:

comp = comp_BERT_large['KQV']
bias_read = False # True if bias is not 0 - outputs are read even the first time

arch = arch_eyeriss


## MAIN:

if __name__ == "__main__":
    options = parse_options()
    
    if options["help"]:
        print("Supported options:")
        print("-h, --help\t\tDisplay this help menu.")
        print("-b --bias\t\tIf set, the bias is considered present in the GEMM, otherwise it is assumed absent.")
        print("-p --processes\t\tSets the number of concurrent processes to use.")
        print("-ta --tryall\t\tOverrides normal execution, runs FF for all known architectures and GEMMs.")
        print("-gt --gen-tests\t\tOverrides normal execution, runs FF and generates tests to enforce the obtained results.")
        sys.exit(1)
        
    #supported_archs = ["gemmini", "eyeriss", "simba", "tpu"]
    #if len(sys.argv) < 2 or sys.argv[1] not in supported_archs:
    #    print("The first argument must be a valid architecture name. Please choose one of the following:")
    #    for arch in supported_archs:
    #        print(f"- {arch}")
    #    sys.exit(1)
    #arch_name = sys.argv[1]
    
    bias_read = options["bias"]
    
    if options["processes"]:
        Settings.THREADS_COUNT = options["processes"]
    
    # EXECUTION
    
    if options["tryall"]:
        extra_constant_columns_names = ["N"]
        extra_constant_columns_values = ["1"]
        
        Settings.VERBOSE = False
        comp_BERT_large.pop("Out")
        comp_BERT_large.pop("FF2")
        table = PrettyTable(["Arch", "Comp", "EDP[J*cycle]", "Latency[cc]", "Energy[uJ]", "Runtime"] + extra_constant_columns_names)
        for arch_name, current_arch in zip(["Gemmini", "Eyeriss", "Simba", "TPUv1"], [arch_gemmini, arch_eyeriss, arch_simba, arch_tpu]):
            for comp_name, current_comp in zip(list(comp_BERT_large.keys()) + list(comp_maestro_blas.keys()), list(comp_BERT_large.values()) + list(comp_maestro_blas.values())):
            #for comp_name, current_comp in zip(list(comp_BERT_large.keys()), list(comp_BERT_large.values())):
                current_arch_copy = copy.deepcopy(current_arch)
                print(f"Now running FactorFlow on arch: {arch_name} and comp: {comp_name}...")
                if fitConstraintsToComp(current_arch_copy, current_comp, arch_name, comp_name):
                    continue
                edp, energy, latency, end_time = basic_run(current_arch_copy, current_comp, bias_read, verbose = False)
                table.add_row([arch_name, comp_name, f"{edp:.3e}", f"{latency:.3e}", f"{energy:.3e}", f"{end_time:.3f}"] + extra_constant_columns_values)
        print(table)
        
    elif options["gen-tests"]:
        Settings.forcedSettingsUpdate(arch)
        fitConstraintsToComp(arch, comp)
        basic_run(arch, comp, bias_read, verbose = True)
        from test import generateTestMOPs, generateTestLatency
        print("\nGenerated tests:")
        generateTestMOPs(arch)
        generateTestLatency(arch)
    
    else:
        Settings.forcedSettingsUpdate(arch)
        fitConstraintsToComp(arch, comp)
        basic_run(arch, comp, bias_read, verbose = True)