# GOAL: Empirically prove the soundness of starting from all factors on the first level by randomly trying many starting points.

from itertools import combinations, product, groupby
from prettytable import PrettyTable
from functools import reduce
import signal
import random
import time
import code
import math
import copy
import sys

try:
    from ..architectures.architectures import *
    from ..computations import *
    from ..architectures.solutions_db import *
    from ..computations import *
    from ..settings import *
    from ..factors import *
    from ..engine import *
    from ..levels import *
    from ..prints import *
    from ..utils import *
except:
    sys.path.append("..")
    from architectures.architectures import *
    from computations import *
    from architectures.solutions_db import *
    from computations import *
    from settings import *
    from factors import *
    from engine import *
    from levels import *
    from prints import *
    from utils import *

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
        "max_tries": args_match_and_remove("-mt", True) or args_match_and_remove("--max_tries", True),
        "random_moves": args_match_and_remove("-rm", True) or args_match_and_remove("--random_moves", True),
        "store_mappings": args_match_and_remove("-sm") or args_match_and_remove("--store_mappings"),
        "all_comps": args_match_and_remove("-ac") or args_match_and_remove("--all_comps"),
        "print_arrays": args_match_and_remove("-pa") or args_match_and_remove("--print_arrays"),
        "print_interval": args_match_and_remove("-pi", True) or args_match_and_remove("--print_interval", True)
    }
    return options

def randomDataflows(arch):
    for level in arch:
        if isinstance(level, MemLevel):
            level.dataflow = random.sample(interleave(level.dataflow_constraints, [dim for dim in level.dataflow if dim not in level.dataflow_constraints]), 1)[0]
        #else:
        #    level.dataflow = random.shuffle(level.dataflow)

# Random-ish but faster
def randomFactorsInitializationsFast(arch, comp):
    arch.initFactors(comp)
    arch.enforceFactorsConstraints()
    
    factors = reduce(lambda l, a : l + a, [[(dim, f) for f in arch[0].factors.toList(dim)] for dim in ['M', 'K', 'N']], [])
    
    def randomMoves(arch):
        random.shuffle(factors)
        for dim, factor in factors:
            choices = list(range(len(arch)))
            random.shuffle(choices)
            for dst_level_idx in choices:
                if dst_level_idx == 0 or arch.moveFactor(0, dst_level_idx, dim, factor, 1):
                    break

    already_seen = []
    
    fails = 0
    while True:
        random_arch = copy.deepcopy(arch)
        randomMoves(random_arch)
        randomDataflows(arch)
        hash = random_arch.hashFromFactors()
        if hash not in already_seen:
            already_seen.append(hash)
            fails = 0
            yield random_arch
        else:
            fails += 1
            if fails >= DUPLICATES_TO_STOP:
                print(f"WARNING: early termination triggered, could not generate {MAX_TRIES} different mappings...")
                return

# Truly random, but slower
def randomFactorsInitializationsSlow(arch, comp, random_moves = 10):
    arch.initFactors(comp)
    arch.enforceFactorsConstraints()
    
    def randomMoves(arch, n):
        mems = list(filter(lambda l : isinstance(l, MemLevel), arch))
        dims = ['M', 'K', 'N']
        random.shuffle(dims)
        for dim in dims:
            factors = mems[0].factors.toList(dim)
            random.shuffle(factors)
            for factor in mems[0].factors.toList(dim):
                for _ in range(n):
                    dst_level_idx = random.choice(range(len(mems)))
                    if dst_level_idx == 0 or arch.moveFactor(0, dst_level_idx, dim, factor, 1):
                        break
    
    already_seen = []
    
    fails = 0
    while True:
        random_arch = copy.deepcopy(arch)
        randomMoves(random_arch, random_moves)
        randomDataflows(arch)
        hash = random_arch.hashFromFactors()
        if hash not in already_seen:
            already_seen.append(hash)
            fails = 0
            yield random_arch
        else:
            fails += 1
            if fails >= DUPLICATES_TO_STOP:
                print(f"WARNING: early termination triggered, could not generate {MAX_TRIES} different mappings...")
                return

if __name__ == "__main__":
    # CONFIGURATION:

    MAX_TRIES = 10000
    DUPLICATES_TO_STOP = MAX_TRIES*10
    RANDOM_MOVES = 20
    STORE_INITIAL_CONDITIONS = False
    PRINT_INTERVAL = 5

    # PARSE CLI ARGS:

    options = parse_options()

    if options["help"]:
        print("Supported options:")
        print("-h, --help\t\t\tDisplay this help menu.")
        print("-mt, --max_tries <tries>\tSets to <tries> the number of starting point tried. Default is 10000.")
        print("-rm, --random_moves <moves>\tSets to <moves> the number of moves attempted for each prime factor in the mapping. Default is 20. ->DEPRECATED<-")
        print("-sm, --store_mappings\tIf given, the generated mappings are also stored and displayed at the end.")
        print("-ac, --all_comps\t\tTries all computations for the specified arch, and summarizes results in a table.")
        print("-pa, --print_arrays\t\tPrints all the stored EDP and Wart values. Only works without '-ac'.")
        print("-pi. --print_interval <secs>\tSets to <secs> the seconds between progress updates are printed. Default is 5 s.")
        sys.exit(1)

    MAX_TRIES = int(options["max_tries"]) if options["max_tries"] else MAX_TRIES
    DUPLICATES_TO_STOP = MAX_TRIES*10
    RANDOM_MOVES = int(options["random_moves"]) if options["random_moves"] else RANDOM_MOVES
    STORE_INITIAL_CONDITIONS = STORE_INITIAL_CONDITIONS or options["store_mappings"]
    PRINT_INTERVAL = PRINT_INTERVAL or options["store_mappings"]
    PRINT_INTERVAL = int(options["print_interval"]) if options["print_interval"] else PRINT_INTERVAL

    supported_archs = ["gemmini", "eyeriss", "simba", "tpu"]
    if len(sys.argv) < 2 or sys.argv[1] not in supported_archs:
        print("The first argument must be a valid architecture name. Please choose one of the following:")
        for arch in supported_archs:
            print(f"- {arch}")
        sys.exit(1)

    arch_name = sys.argv[1]

    # SPECIFICATION:
    
    if options["all_comps"]:
        comps = {"BERT large " + k : v for k, v in comp_BERT_large.items()}
        comps.update({"MAESTRO-BLAS " + k[-1] : v for k, v in comp_maestro_blas.items()})
    else:
        comps = {'BERT large KQV': comp_BERT_large['KQV']}
    bias_read = False

    # << Gemmini >>
    if arch_name == "gemmini":
        arch = arch_gemmini

    # << Eyeriss >>
    elif arch_name == "eyeriss":
        arch = arch_eyeriss
    
    # << Simba >>
    elif arch_name == "simba":
        arch = arch_simba
    
    # << TPU >>
    elif arch_name == "tpu":
        arch = arch_tpu
    
    #Here changing settings is fine, there are no processes
    forcedSettingsUpdate(arch, False)
    
    last_print_time = time.monotonic()
    
    if options["all_comps"]:
        table = PrettyTable(["Comp", "Arch", "FF - EDP", "Random - Min EDP", "Random - Max EDP", "Random - Avg. EDP"])
    for comp_name, comp in comps.items():
        print(f"Working on comp: {comp_name} and arch: {arch_name}...")
        tried = 0
        warts = []
        edps = []
        initial_conditions = []

        #print("Generating random starting points...")
        arch_copy = copy.deepcopy(arch)
        if not arch_copy.fitConstraintsToComp(comp, comp_name):
            continue
        random_archs = randomFactorsInitializationsFast(arch_copy, comp)
        #_ = next(random_archs)
        print(f"Starting generation of {MAX_TRIES} random mappings:")
        for current_arch in random_archs:
            try:
                assert current_arch.checkFactorsConstraints()
                if STORE_INITIAL_CONDITIONS: initial_conditions.append(factorsString(current_arch))
                edp = EDP(current_arch, bias_read, True)
                wart = Wart(current_arch, comp, bias_read)
            except AssertionError:
                continue
            warts.append(wart)
            edps.append(edp)
            
            if math.floor((tried/MAX_TRIES)*10) > math.floor(((tried - 1)/MAX_TRIES)*10) or time.monotonic() - last_print_time > PRINT_INTERVAL:
                print(f"Progress: {tried}/{MAX_TRIES} tried...")
                last_print_time = time.monotonic()
            if tried >= MAX_TRIES:
                break
            else:
                tried += 1
        
        if not options["all_comps"]:
            print(f"FF INIT: ", end="")
            printFactors(arch)
            factorflow_edp, _, _, _, _, _, arch = run_engine(arch, comp, bias_read, False)
            factorflow_wart = Wart(arch, comp, bias_read)
            print(f"FF FINAL: ", end="")
            printFactors(arch)
            
            print("\nResults for Wart (higher is better):")
            print(f"FF result - Wart: {factorflow_wart:.3e}")
            print(f"Random mappings:\n\t- avg. Wart: {sum(warts)/len(warts):.3e}\n\t- min Wart: {min(warts):.3e}\n\t- max Wart: {max(warts):.3e}")
            short_warts = [f"{w:.3e}" for w in warts]
            if STORE_INITIAL_CONDITIONS:
                for initial_condition, wart in zip(initial_conditions, short_warts):
                    print(f"Mapping: {initial_condition}, Wart: {wart}")
            if options["print_arrays"]:
                print(f"\nComplete mappings Warts: {short_warts}")
            
            print("\nResults for EDP (lower is better):")
            print(f"FF result - EDP: {factorflow_edp:.3e}")
            print(f"Random mappings:\n\t- avg. EDP: {sum(edps)/len(edps):.3e}\n\t- min EDP: {min(edps):.3e}\n\t- max EDP: {max(edps):.3e}")
            short_edps = [f"{e:.3e}" for e in edps]
            if STORE_INITIAL_CONDITIONS:
                for initial_condition, edp in zip(initial_conditions, short_edps):
                    print(f"Mapping: {initial_condition}, EDP: {edp}")
            if options["print_arrays"]:
                print(f"\nComplete mappings EDPs: {short_edps}")
        else:
            arch_ff = copy.deepcopy(arch)
            if not arch_ff.fitConstraintsToComp(comp, comp_name):
                continue
            factorflow_edp, _, _, _, _, _, arch_ff = run_engine(arch_ff, comp, bias_read, False)
            factorflow_wart = Wart(arch_ff, comp, bias_read)
            table.add_row([comp_name, arch_name.title(), f"{factorflow_edp:.3e}", f"{min(edps):.3e}", f"{max(edps):.3e}", f"{sum(edps)/len(edps):.3e}"])
    if options["all_comps"]:
        print(table)
