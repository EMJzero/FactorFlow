from itertools import chain, combinations, permutations
from functools import reduce
import math

from typing import Iterable, Iterator, TypeVar, Union
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
Given a 'template' containing some entries of 'elements' and some
placeholder elements (having value 'placeholder'), inserts in all
possible ways the remaining entries of 'elements' in the template.
"""
def slot_in(template : list[T], elements : list[T], placeholder: T) -> list[list[T]]:
    placeholder_indices = [i for i, x in enumerate(template) if x == placeholder]
    num_placeholders = len(placeholder_indices)
    remaining_elements = [x for x in elements if x not in template]
    if len(remaining_elements) < num_placeholders:
        raise ValueError("Not enough elements to fill placeholders.")
    permutations_of_elements = permutations(remaining_elements, num_placeholders)
    results = []
    for perm in permutations_of_elements:
        filled_template = template[:]
        for idx, value in zip(placeholder_indices, perm):
            filled_template[idx] = value
        results.append(filled_template)
    return results

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
def flatten_two_levels_list(lst : list[Union[list[T], T]]) -> list[T]:
    return [subitem for item in lst for subitem in (item if isinstance(item, list) else [item])]

"""
Checks if a list has at most two levels of nesting.
"""
def is_two_levels_list(lst : list[Union[list[T], T]]) -> bool:
    return all(not (isinstance(item, list) and any(isinstance(subitem, list) for subitem in item)) for item in lst)

"""
Consider a vector V indicized as V[x_const_1*x_1+...+x_const_n*x_n]
for a certain n > 0, with x_i in [0, X_1) for i in [1, n]. This function
receive the values for X_1, ..., X_n, x_const_1, ..., x_const_n, and
provides the number of distinct elements of V which can be accessed
from at least one combination of valid x_i-s.
Here 'Xs' and 'x_consts' are lists containing the above n values.
"""
def distinct_values(Xs : list[int], x_consts : list[int]) -> int:
    if len(Xs) == 1: # trivial case
        return Xs[0]
    elif (cnt := Xs.count(1)) == len(Xs): # all x_i ranges contain only 0
        return 1
    elif not Settings.OVERESTIMATE_DISTINCT_VALUES and cnt == len(Xs) - 1: # all but one x_i ranges contain only 0
        return next((X for X in Xs if X != 1), 1)
    elif all(x_const == 1 for x_const in x_consts): # all coefficients are 1
        return sum(Xs) - len(Xs) + 1
    else:
        g = math.gcd(*x_consts)
        # remove any common denominator
        x_consts = list(map(lambda cst : cst//g, x_consts))
        if len(Xs) == 2:
            if Settings.OVERESTIMATE_DISTINCT_VALUES: # simpler formula which slightly overestimates the count of distinct values (this is used by Timeloop)
                return (x_consts[0]*(Xs[0] - 1) + x_consts[1]*(Xs[1] - 1)) + 1
            elif Xs[0] >= x_consts[1] and Xs[1] >= x_consts[0]: # Frobenius coin problem approach - case of negative-y lattice box fully contained in the valid box
                return (Xs[0]-1)*x_consts[0]+(Xs[1]-1)*x_consts[1]+1 - (x_consts[0]-1)*(x_consts[1]-1) #+1 is because we count 0 too
            else: # Frobenius coin problem approach - case of all valid lattice point being distinct values (x_costs vector can't fit in the valid box)
                return Xs[0]*Xs[1]
        else: # general case, count all distinct values with dynamic programming
            step_0 = x_consts[0]
            current_values = set(range(0, step_0*Xs[0], step_0))
            for i in range(1, len(x_consts)):
                step = x_consts[i]
                X = Xs[i]
                additions = {v + step*x for v in current_values for x in range(X)}
                current_values.update(additions)
            return len(current_values)