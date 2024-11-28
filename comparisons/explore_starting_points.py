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
        "store_init_conds": args_match_and_remove("-sic") or args_match_and_remove("--store_init_conds"),
        "all_comps": args_match_and_remove("-ac") or args_match_and_remove("--all_comps"),
        "print_interval": args_match_and_remove("-pi", True) or args_match_and_remove("--print_interval", True)
    }
    return options

"""
def partitions(arr, n, seen=None):
    if seen is None:
        seen = set()
    if n == len(arr):
        yield [[x] for x in arr]
    elif n == 1:
        yield [arr]
    else:
        for i in range(1, len(arr)-n+2):
            for first in combinations(range(len(arr)), i):
                first_set = set(first)
                arr1 = [arr[j] for j in first]
                arr2 = [arr[j] for j in range(len(arr)) if j not in first_set]
                for rest in partitions(arr2, n-1, seen):
                    partition = [arr1] + rest
                    partition_str = str(partition)
                    if partition_str not in seen:
                        seen.add(partition_str)
                        yield partition

def disjoint_partitions(arr1, arr2, arr3, n):
    part1 = list(partitions(arr1, n))
    print("Generation permutations 1/3...")
    part2 = list(partitions(arr2, n))
    print("Generation permutations 2/3...")
    part3 = list(partitions(arr3, n))
    print("Generation permutations 3/3...")
    print("Generating combinations...")

    all_partitions = list(product(part1, part2, part3))
    random.shuffle(all_partitions)

    for partition in all_partitions:
        yield partition

def randomFactorsInitializations(arch, comp):
    initFactors(arch, comp)
    enforceFactorsConstraints(arch)
    setupBypasses(arch)
    arch.setupSpatialLevelPointers()
    updateInstances(arch)
    
    mems = list(filter(lambda l : isinstance(l, MemLevel), arch))
    random_disjoint_primes_lists = disjoint_partitions(primeFactorsList(arch[0].factors.dimProduct('M')), primeFactorsList(arch[0].factors.dimProduct('M')), primeFactorsList(arch[0].factors.dimProduct('M')), len(mems))
    
    for disjoint_primes_list in random_disjoint_primes_lists:
        dim_to_idx = {'M': 0, 'K': 1, 'N': 2}
        for dim in dim_to_idx.keys():
            for level_idx in range(len(mems)):
                factors = Factors()
                for prime in disjoint_primes_list[dim_to_idx[dim]][level_idx]:
                    factors.addFactor(dim, prime, 1)
                arch[level_idx].factors = factors
        yield copy.deepcopy(arch)
"""

# Random-ish but faster
def randomFactorsInitializationsFast(arch, comp, random_moves = 10):
    arch.initFactors(comp)
    arch.enforceFactorsConstraints()
    arch.setupBypasses()
    arch.setupSpatialLevelPointers()
    arch.updateInstances()
    
    mems = list(filter(lambda l : isinstance(l, MemLevel), arch))
    factors = reduce(lambda l, a : l + a, [[(dim, f) for f in mems[0].factors.toList(dim)] for dim in ['M', 'K', 'N']], [])
    
    def randomMoves(arch, n):
        random.shuffle(factors)
        for dim, factor in factors:
            for _ in range(n):
                dst_level_idx = random.choice(range(len(mems)))
                if dst_level_idx == 0 or arch.moveFactor(0, dst_level_idx, dim, factor, 1):
                    break
    
    already_seen = [arch.hashFromFactors()]
    
    fails = 0
    while True:
        random_arch = copy.deepcopy(arch)
        randomMoves(random_arch, random_moves)
        hash = random_arch.hashFromFactors()
        if hash not in already_seen:
            already_seen.append(hash)
            fails = 0
            yield random_arch
        else:
            fails += 1
            if fails >= DUPLICATES_TO_STOP:
                print(f"WARNING: early termination triggered, could not generate {MAX_TRIES} different starting points...")
                return

# Truly random, but slower
def randomFactorsInitializationsSlow(arch, comp, random_moves = 10):
    arch.initFactors(comp)
    arch.enforceFactorsConstraints()
    arch.setupBypasses()
    arch.setupSpatialLevelPointers()
    arch.updateInstances()
    
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
    
    already_seen = [arch.hashFromFactors()]
    
    fails = 0
    while True:
        random_arch = copy.deepcopy(arch)
        randomMoves(random_arch, random_moves)
        hash = random_arch.hashFromFactors()
        if hash not in already_seen:
            already_seen.append(hash)
            fails = 0
            yield random_arch
        else:
            fails += 1
            if fails >= DUPLICATES_TO_STOP:
                print(f"WARNING: early termination triggered, could not generate {MAX_TRIES} different starting points...")
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
        print("-rm, --random_moves <moves>\tSets to <moves> the number of moves attempted for each prime factor in the mapping. Default is 20.")
        print("-sic, --store_init_conds\tIf given, the initial random starting points are also stored and displayed at the end.")
        print("-ac, --all_comps\t\tTries all computations for the specified arch, and summarizes results in a table.")
        print("-pi. --print_interval <secs>\tSets to <secs> the seconds between progress updates are printed. Default is 5 s.")
        sys.exit(1)

    MAX_TRIES = int(options["max_tries"]) if options["max_tries"] else MAX_TRIES
    DUPLICATES_TO_STOP = MAX_TRIES*10
    RANDOM_MOVES = int(options["random_moves"]) if options["random_moves"] else RANDOM_MOVES
    STORE_INITIAL_CONDITIONS = STORE_INITIAL_CONDITIONS or options["store_init_conds"]
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
        base_arch = arch_gemmini
        arch = arch_gemmini_factorflow_2

    # << Eyeriss >>
    elif arch_name == "eyeriss":
        base_arch = arch_eyeriss
        arch = arch_eyeriss_factorflow_1
        arch[2].dims = ['M', 'K']
        arch[2].dataflow = ['M', 'K']
    
    # << Simba >>
    elif arch_name == "simba":
        base_arch = arch_simba
        arch = arch_simba_factorflow_1
        arch[2].dims = ['M', 'K']
        arch[2].dataflow_constraints = ['M', 'K']
    
    # << TPU >>
    elif arch_name == "tpu":
        base_arch = arch_tpu
        arch = arch_tpu_factorflow_1
        Settings.STEPS_TO_EXPLORE = 2
        print(f"INFO: tpu selected, forcefully updating setting STEPS_TO_EXPLORE to {Settings.STEPS_TO_EXPLORE}.")
        #arch[0].dataflow_constraints = ['N', 'M', 'K']
        #arch[1].dataflow_constraints = ['M', 'K', 'N']
        #arch[2].dataflow_constraints = ['N', 'K', 'M']
        #arch[3].dataflow_constraints = ['N', 'K', 'M']
        #arch[5].dataflow_constraints = ['N', 'M', 'K']
        #arch[7].dataflow_constraints = ['M', 'K', 'N']
        #arch[0].dataflow = ['N', 'M', 'K']
        #arch[1].dataflow = ['M', 'K', 'N']
        #arch[2].dataflow = ['N', 'K', 'M']
        #arch[3].dataflow = ['N', 'K', 'M']
        #arch[5].dataflow = ['N', 'M', 'K']
        #arch[7].dataflow = ['M', 'K', 'N']
    
    for level_idx in range(len(arch)):
        #if not isinstance(arch[level_idx], ComputeLevel):
        arch[level_idx].factors_constraints = base_arch[level_idx].factors_constraints
    #Here changing settings is fine, there are no processes
    forcedSettingsUpdate(arch, False)
    
    last_print_time = time.monotonic()
    
    if options["all_comps"]:
        table = PrettyTable(["Comp", "Arch", "All on Level 0 - EDP", "Random - Min EDP", "Random - Max EDP", "Random - Avg. EDP"])
    for comp_name, comp in comps.items():
        print(f"Working on comp: {comp_name} and arch: {arch_name}...")
        tried = 0
        warts = []
        edps = []
        initial_conditions = []

        #print("Generating random starting points...")
        arch_copy = copy.deepcopy(arch)
        if arch_copy.fitConstraintsToComp(comp, comp_name):
            continue
        random_archs = randomFactorsInitializationsFast(arch_copy, comp, RANDOM_MOVES)
        #_ = next(random_archs)
        print(f"Starting optimization of {MAX_TRIES} different starting points:")
        for current_arch in random_archs:
            try:
                assert not current_arch.findConstraintsViolation(False)
                if STORE_INITIAL_CONDITIONS: initial_conditions.append(factorsString(current_arch))
                current_arch, wart = factorFlow(current_arch, comp, bias_read, already_initialized = True)
                edp = EDP(current_arch, bias_read, True)
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
            print(f"FF INIT: {factorsString(arch)}")
            arch, factorflow_wart = factorFlow(arch, comp, bias_read)
            print(f"FF FINAL: {factorsString(arch)}")
            factorflow_edp = EDP(arch, bias_read, True)
            
            print("\nResults for Wart (higher is better):")
            print(f"All factors on first level - Wart: {factorflow_wart:.3e}")
            print(f"Random starting points:\n\t- avg. Wart: {sum(warts)/len(warts):.3e}\n\t- min Wart: {min(warts):.3e}\n\t- max Wart: {max(warts):.3e}")
            short_warts = [f"{w:.3e}" for w in warts]
            if STORE_INITIAL_CONDITIONS:
                for initial_condition, wart in zip(initial_conditions, short_warts):
                    print(f"Initial condition: {initial_condition}, Wart: {wart}")
            #else:
            #    print(f"\nComplete random starting point Warts: {short_warts}")
            
            print("\nResults for EDP (lower is better):")
            print(f"All factors on first level - EDP: {factorflow_edp:.3e}")
            print(f"Random starting points:\n\t- avg. EDP: {sum(edps)/len(edps):.3e}\n\t- min EDP: {min(edps):.3e}\n\t- max EDP: {max(edps):.3e}")
            short_edps = [f"{e:.3e}" for e in edps]
            if STORE_INITIAL_CONDITIONS:
                for initial_condition, edp in zip(initial_conditions, short_edps):
                    print(f"Initial condition: {initial_condition}, EDP: {edp}")
            #else:
            #    print(f"\nComplete random starting point Warts: {short_warts}")
        else:
            arch_ff = copy.deepcopy(arch)
            if arch_ff.fitConstraintsToComp(comp, comp_name):
                continue
            arch_ff, factorflow_wart = factorFlow(arch_ff, comp, bias_read)
            factorflow_edp = EDP(arch_ff, bias_read, True)
            table.add_row([comp_name, arch_name.title(), f"{factorflow_edp:.3e}", f"{min(edps):.3e}", f"{max(edps):.3e}", f"{sum(edps)/len(edps):.3e}"])
    if options["all_comps"]:
        print(table)
