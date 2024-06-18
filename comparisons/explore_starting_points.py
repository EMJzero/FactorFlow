# GOAL: Empirically prove the soundness of starting from all factors on the first level by randomly trying many starting points.

from itertools import combinations, product, groupby
from functools import reduce
import random
import math
import copy

import sys
sys.path.append("..")

from main import factorFlow, forcedSettingsUpdate
from architectures import *
from computations import *
from solutions_db import *
from computations import *
from factors import *
from levels import *
from prints import *
from utils import *

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
def randomFactorsInitializationsSlow(arch, comp, random_moves = 10):
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
def randomFactorsInitializationsFast(arch, comp, random_moves = 10):
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

    MAX_TRIES = 1000
    RANDOM_MOVES = 20
    STORE_INITIAL_CONDITIONS = False

    # SPECIFICATION:
    
    comp = comp_BERT_large['KQV']
    bias_read = False

    # << Gemmini >>
    base_arch = arch_gemmini
    arch = arch_gemmini_factorflow_2

    # << Eyeriss >>
    #base_arch = arch_eyeriss
    #arch = arch_eyeriss_factorflow_1
    #arch[2].dims = ['D', 'E']
    #arch[2].dataflow = ['D', 'E']
    
    for level_idx in range(len(arch)):
        #if not isinstance(arch[level_idx], ComputeLevel):
        arch[level_idx].factors_contraints = base_arch[level_idx].factors_contraints
    forcedSettingsUpdate(arch, False)

    tried = 0
    warts = []
    initial_conditions = []
    
    #print("Generating random starting points...")
    random_archs = randomFactorsInitializationsSlow(copy.deepcopy(arch), comp, RANDOM_MOVES)
    #_ = next(random_archs)
    print(f"Starting optimization of {MAX_TRIES} different starting points:")
    for current_arch in random_archs:
        try:
            assert not findConstraintsViolation(current_arch, False)
            if STORE_INITIAL_CONDITIONS: initial_conditions.append(factorsString(current_arch))
            _, wart = factorFlow(current_arch, comp, bias_read, already_initialized = True)
        except AssertionError:
            continue
        warts.append(wart)
        
        if math.floor((tried/MAX_TRIES)*10) > math.floor(((tried - 1)/MAX_TRIES)*10):
            print(f"Progress: {tried}/{MAX_TRIES} tried...")
        if tried >= MAX_TRIES:
            break
        else:
            tried += 1
            
    _, factorflow_wart = factorFlow(arch, comp, bias_read)
    
    print("\nResults (higher is better):")
    print(f"All factors on first level - Wart: {factorflow_wart:.3e}")
    print(f"Random starting points:\n\t- avg. Wart: {sum(warts)/len(warts):.3e}\n\t- min Wart: {min(warts):.3e}\n\t- max Wart: {max(warts):.3e}")
    short_warts = [f"{w:.3e}" for w in warts]
    if STORE_INITIAL_CONDITIONS:
        for initial_condition, wart in zip(initial_conditions, short_warts):
            print(f"Initial condition: {initial_condition}, Wart: {wart}")
    else:
        print(f"\nComplete random starting point Warts: {short_warts}")