from itertools import chain, combinations, permutations
import math

from levels import *

# >>> Miscellaneus functions working with primes and iterables/arrays (snake_cased)

"""
Computes the prime factors of 'n' and returns them as a dictionary where
keys are n's prime factors, and values are their arities (occurrencies).
"""
def prime_factors(n):
    i = 2
    factors = {}
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors[i] = factors.get(i, 0) + 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

"""
Computes the prime factors of 'n' and returns them as a list.
"""
def prime_factors_list(n):
    factors = []
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    for i in range(3, int(n**0.5) + 1, 2):
        while n % i == 0:
            factors.append(i)
            n //= i
    if n > 2:
        factors.append(n)
    return factors

"""
Returns the powerset of an iterable, that being all its possible subsets,
including the complete and empty ones.
"""
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

"""
Interleaves the entries of 'elements' in all possible ways in between the
elements of 'array'. The array elements retain their order, while those of
'elements' may not.
"""
def interleave(array, elements):
    def recursive_insert(arr, elems):
        if not elems:
            return [arr]
        results = []
        for i in range(len(arr) + 1):
            new_arr = arr[:i] + [elems[0]] + arr[i:]
            results.extend(recursive_insert(new_arr, elems[1:]))
        return results
    return recursive_insert(array, elements)

"""
Returns a list of all versions of 'array' that differ by a rotation (or shift).
"""
def rotations(array):
    return [array[i:] + array[:i] for i in range(len(array))] or [[]]

"""
Returns, if any, the sets of elements which can undergo a cyclic shift and
in so reach the configuration of 'arr2' starting from 'arr1'.
"""
def single_cyclic_shift(arr1, arr2):
    n = len(arr1)
    for i in range(n):
        # Right Shift
        if arr1[-1] == arr2[i] and arr1[:-1] == arr2[i+1:]:
            return [[arr1[-1]], arr1[:-1]]
        # Left Shift
        if arr1[0] == arr2[i] and arr1[1:] == arr2[:i]:
            return [[arr2[i]], arr1[1:]]
    return []

"""
Returns all elements which need to be involved in swaps between neighbours
to reach 'arr2' from 'arr1' (no cyclic behaviour allowed).
"""
def pairwise_swaps(arr1, arr2):
    swaps = set()
    arr1 = arr1.copy()
    for i in range(len(arr1)):
        if arr1[i] != arr2[i]:
            j = i + 1
            while arr1[j] != arr2[i]:
                j += 1
            while j > i:
                arr1[j], arr1[j - 1] = arr1[j - 1], arr1[j]
                swaps.add(arr1[j])
                swaps.add(arr1[j - 1])
                j -= 1
    return swaps

"""
Finds the subarray of 'arr', if any, whose product exceeds 'target' by
the least amount and returns that subarray and the excess amount.
(the product is allowed to be equal to the target)
"""
def smallest_product_greater_than(arr, target):
    min_excess = math.inf
    best_subarray = []
    for r in range(1, len(arr) + 1):
        for comb in combinations(arr, r):
            product = math.prod(comb)
            if product >= target and (product - target) < min_excess:
                min_excess = product - target
                best_subarray = comb
    return list(best_subarray), min_excess

"""
Finds the subarray of 'arr', if any, whose product is short of 'target'
by the least amount and returns that subarray and the defect amount.
(the product is allowed to be equal to the target)
"""
def largest_product_less_than(arr, target):
    min_defect = math.inf
    best_subarray = []

    for r in range(1, len(arr) + 1):
        for comb in combinations(arr, r):
            product = math.prod(comb)
            if product <= target and (target - product) < min_defect:
                min_defect = target - product
                best_subarray = comb

    return list(best_subarray), min_defect