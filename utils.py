from itertools import chain, combinations
from functools import reduce
import math

from typing import Iterable, Iterator, TypeVar
T = TypeVar('T')

from settings import *

# >>> Miscellaneus functions working with primes and iterables/arrays (snake_cased)

"""
Computes the prime factors of 'n' and returns them as a dictionary where
keys are n's prime factors, and values are their arities (occurrencies).
"""
def prime_factors(n : int) -> dict[int, int]:
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
def prime_factors_list(n : int) -> list[int]:
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
def powerset(iterable : Iterable[T]) -> Iterator[tuple[T, ...]]:
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

"""
Interleaves the entries of 'elements' in all possible ways in between the
elements of 'array'. The array elements retain their order, while those of
'elements' may not.
"""
def interleave(array : list[T], elements : list[T]) -> list[list[T]]:
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
def rotations(array : list[T]) -> list[list[T]]:
    return [array[i:] + array[:i] for i in range(len(array))] or [[]]

"""
Returns, if any, the sets of elements which can undergo a cyclic shift and
in so reach the configuration of 'arr2' starting from 'arr1'.
"""
def single_cyclic_shift(arr1 : list[T], arr2 : list[T]) -> list[list[T]]:
    n = len(arr1)
    if n != len(arr2):
        return []
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
def pairwise_swaps(arr1 : list[T], arr2 : list[T]) -> set[T]:
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
def smallest_product_greater_than(arr : list[T], target : T) -> tuple[list[T], T]:
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
def largest_product_less_than(arr : list[T], target : T) -> tuple[list[T], T]:
    min_defect = math.inf
    best_subarray = []

    for r in range(1, len(arr) + 1):
        for comb in combinations(arr, r):
            product = math.prod(comb)
            if product <= target and (target - product) < min_defect:
                min_defect = target - product
                best_subarray = comb

    return list(best_subarray), min_defect

"""
Flattens a list by up to two levels of nesting.
"""
def flatten_two_levels_list(lst):
    return [subitem for item in lst for subitem in (item if isinstance(item, list) else [item])]

"""
Checks if a list has at most two levels of nesting.
"""
def is_two_levels_list(lst):
    return all(not (isinstance(item, list) and any(isinstance(subitem, list) for subitem in item)) for item in lst)

"""
Consider a vector V indicized as V[x_const_1*x_1+...+x_const_n*x_n]
for a certain n > 0, with x_i in [0, X_1) for i in [1, n]. This function
receive the values for X_1, ..., X_n, x_const_1, ..., x_const_n, and
provides the number of distinct elements of V which can be accessed
from at least one combination of valid x_i-s.
Here 'Xs' and 'x_consts' are lists containing the above n values.
"""
def distinct_values(Xs : list[int], x_consts : list[int]):
    if len(Xs) == 1: # trivial case
        return Xs[0]
    elif (cnt := Xs.count(1)) == len(Xs): # all x_i ranges contain only 0
        return 1
    elif not Settings.OVERESTIMATE_DISTINCT_VALUES and cnt == len(Xs) - 1: # all but one x_i ranges contain only 0
        return next((X for X in Xs if X != 1), 1)
    elif all(x_const == 1 for x_const in x_consts): # all coefficients are 1
        return sum(Xs) - len(Xs) + 1
    elif Settings.OVERESTIMATE_DISTINCT_VALUES and len(Xs) == 2: # simpler formula which slightly overestimates the count of distinct values (this is used by Timeloop)
        return math.floor((x_consts[0]*(Xs[0] - 1) + x_consts[1]*(Xs[1] - 1))/math.gcd(x_consts[0], x_consts[1])) + 1
    elif len(Xs) == 2 and (max_idx := x_consts[0]*(Xs[0]-1)+x_consts[1]*(Xs[1]-1)) >= 2*(f := x_consts[0]*x_consts[1] - x_consts[0] - x_consts[1]): # Frobenius coin problem approach - case z_max >= 2*f, no overlap between head and tail
        return max_idx - (x_consts[0] - 1)*(x_consts[1] - 1) + 1 - (1 if max_idx == f else 0)
    elif False and len(Xs) == 2: # Frobenius coin problem approach - case z_max < 2*f, overlap between head and tail
        # INCORRECT
        g = math.gcd(*x_consts)
        return min((x_consts[0]//g)*(Xs[0] - 1) + (x_consts[1]//g)*(Xs[1] - 1) + 1, Xs[0]*Xs[1])
    else: # general case, count all distinct values with dynamic programming
        g = math.gcd(*x_consts)
        step_0 = x_consts[0]//g
        current_values = set(range(0, step_0*Xs[0], step_0))
        for i in range(1, len(x_consts)):
            step = x_consts[i]//g
            X = Xs[i]
            additions = {v + step*x for v in current_values for x in range(X)}
            current_values.update(additions)
        return len(current_values)