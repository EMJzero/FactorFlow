# GOAL: Empirically prove the soundness of starting from all factors on the first level by randomly trying many starting points.

from itertools import combinations, product, groupby
from functools import reduce
import random
import math
import copy

import sys
sys.path.append("..")

from main import factorFlow, EDP
from architectures import *
from computations import *
from solutions_db import *
from computations import *
from settings import *
from factors import *
from levels import *
from prints import *
from utils import *

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
    updateInstances(arch)
    
    mems = list(filter(lambda l : isinstance(l, MemLevel), arch))
    random_disjoint_primes_lists = disjoint_partitions(primeFactorsList(arch[0].factors.dimProduct('D')), primeFactorsList(arch[0].factors.dimProduct('D')), primeFactorsList(arch[0].factors.dimProduct('D')), len(mems))
    #print(primeFactorsList(arch[0].factors.dimProduct('D')), primeFactorsList(arch[0].factors.dimProduct('D')), primeFactorsList(arch[0].factors.dimProduct('D')), len(mems))
    #print(list(random_disjoint_primes_lists))
    
    for disjoint_primes_list in random_disjoint_primes_lists:
        dim_to_idx = {'D': 0, 'E': 1, 'L': 2}
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
    initFactors(arch, comp)
    enforceFactorsConstraints(arch)
    setupBypasses(arch)
    updateInstances(arch)
    
    mems = list(filter(lambda l : isinstance(l, MemLevel), arch))
    factors = reduce(lambda l, a : l + a, [[(dim, f) for f in mems[0].factors.toList(dim)] for dim in ['D', 'E', 'L']], [])
    
    def randomMoves(arch, n):
        random.shuffle(factors)
        for dim, factor in factors:
            for _ in range(n):
                dst_level_idx = random.choice(range(len(mems)))
                if dst_level_idx == 0 or moveFactor(arch, 0, dst_level_idx, dim, factor, 1):
                    break
    
    already_seen = [hashFromFactors(arch)]
    
    while True:
        random_arch = copy.deepcopy(arch)
        randomMoves(random_arch, random_moves)
        hash = hashFromFactors(random_arch)
        if hash not in already_seen:
            already_seen.append(hash)
            yield random_arch

# Truly random, but slower
def randomFactorsInitializationsSlow(arch, comp, random_moves = 10):
    initFactors(arch, comp)
    enforceFactorsConstraints(arch)
    setupBypasses(arch)
    updateInstances(arch)
    
    def randomMoves(arch, n):
        mems = list(filter(lambda l : isinstance(l, MemLevel), arch))
        dims = ['D', 'E', 'L']
        random.shuffle(dims)
        for dim in dims:
            factors = mems[0].factors.toList(dim)
            random.shuffle(factors)
            for factor in mems[0].factors.toList(dim):
                for _ in range(n):
                    dst_level_idx = random.choice(range(len(mems)))
                    if dst_level_idx == 0 or moveFactor(arch, 0, dst_level_idx, dim, factor, 1):
                        break
    
    already_seen = [hashFromFactors(arch)]
    
    while True:
        random_arch = copy.deepcopy(arch)
        randomMoves(random_arch, random_moves)
        hash = hashFromFactors(random_arch)
        if hash not in already_seen:
            already_seen.append(hash)
            yield random_arch

if __name__ == "__main__":
    # CONFIGURATION:

    MAX_TRIES = 10000
    RANDOM_MOVES = 20
    STORE_INITIAL_CONDITIONS = False

    # PARSE CLI ARGS:

    options = parse_options()

    if options["help"]:
        print("Supported options:")
        print("-h, --help\t\t\tDisplay this help menu.")
        print("-mt, --max_tries <tries>\tSets to <tries> the number of starting point tried. Default is 10000.")
        print("-rm, --random_moves <moves>\tSets to <moves> the number of moves attempted for each prime factor in the mapping. Default is 20.")
        print("-sic, --store_init_conds\tIf given, the initial random starting points are also stored and displayed at the end.")
        sys.exit(1)

    MAX_TRIES = int(options["max_tries"]) if options["max_tries"] else MAX_TRIES
    RANDOM_MOVES = int(options["random_moves"]) if options["random_moves"] else RANDOM_MOVES
    STORE_INITIAL_CONDITIONS = STORE_INITIAL_CONDITIONS or options["store_init_conds"]

    supported_archs = ["gemmini", "eyeriss", "simba", "tpu"]
    if len(sys.argv) < 2 or sys.argv[1] not in supported_archs:
        print("The first argument must be a valid architecture name. Please choose one of the following:")
        for arch in supported_archs:
            print(f"- {arch}")
        sys.exit(1)

    arch_name = sys.argv[1]

    # SPECIFICATION:
    
    comp = comp_BERT_large['KQV']
    bias_read = False

    # << Gemmini >>
    if arch_name == "gemmini":
        base_arch = arch_gemmini
        arch = arch_gemmini_factorflow_2

    # << Eyeriss >>
    elif arch_name == "eyeriss":
        base_arch = arch_eyeriss
        arch = arch_eyeriss_factorflow_1
        arch[2].dims = ['D', 'E']
        arch[2].dataflow = ['D', 'E']
    
    # << Simba >>
    elif arch_name == "simba":
        base_arch = arch_simba
        arch = arch_simba_factorflow_1
        arch[2].dims = ['D', 'E']
        arch[2].dataflow_constraints = ['D', 'E']
    
    # << TPU >>
    elif arch_name == "tpu":
        base_arch = arch_tpu
        arch = arch_tpu_factorflow_1
        Settings.STEPS_TO_EXPLORE = 2
        print(f"INFO: tpu selected, forcefully updating setting STEPS_TO_EXPLORE to {Settings.STEPS_TO_EXPLORE}.")
        #arch[0].dataflow_constraints = ['L', 'D', 'E']
        #arch[1].dataflow_constraints = ['D', 'E', 'L']
        #arch[2].dataflow_constraints = ['L', 'E', 'D']
        #arch[3].dataflow_constraints = ['L', 'E', 'D']
        #arch[5].dataflow_constraints = ['L', 'D', 'E']
        #arch[7].dataflow_constraints = ['D', 'E', 'L']
        #arch[0].dataflow = ['L', 'D', 'E']
        #arch[1].dataflow = ['D', 'E', 'L']
        #arch[2].dataflow = ['L', 'E', 'D']
        #arch[3].dataflow = ['L', 'E', 'D']
        #arch[5].dataflow = ['L', 'D', 'E']
        #arch[7].dataflow = ['D', 'E', 'L']
    
    for level_idx in range(len(arch)):
        #if not isinstance(arch[level_idx], ComputeLevel):
        arch[level_idx].factors_contraints = base_arch[level_idx].factors_contraints
    Settings.forcedSettingsUpdate(arch, False)

    tried = 0
    warts = []
    edps = []
    initial_conditions = []
    
    #print("Generating random starting points...")
    random_archs = randomFactorsInitializationsFast(copy.deepcopy(arch), comp, RANDOM_MOVES)
    #_ = next(random_archs)
    print(f"Starting optimization of {MAX_TRIES} different starting points:")
    for current_arch in random_archs:
        try:
            assert not findConstraintsViolation(current_arch, False)
            if STORE_INITIAL_CONDITIONS: initial_conditions.append(factorsString(current_arch))
            current_arch, wart = factorFlow(current_arch, comp, bias_read, already_initialized = True)
            edp = EDP(current_arch, bias_read, True)
        except AssertionError:
            continue
        warts.append(wart)
        edps.append(edp)
        
        if math.floor((tried/MAX_TRIES)*10) > math.floor(((tried - 1)/MAX_TRIES)*10):
            print(f"Progress: {tried}/{MAX_TRIES} tried...")
        if tried >= MAX_TRIES:
            break
        else:
            tried += 1
            
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