from itertools import chain, combinations
from functools import reduce
import math

from settings import *

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

#"""
#Calculates how many integers in the range [0, N) are multiples
#of at least one number in 'numbers'.
#"""
# TODO: remove this during cleanup
#def count_multiples(N, numbers):
#    total_multiples = 0
#    k = len(numbers)
#    for size in range(1, k + 1):
#        for subset in combinations(numbers, size):
#            subset_lcm = reduce(lambda a, b : (a * b) // math.gcd(a, b), subset)
#            count = N // subset_lcm
#            if size % 2 == 1:
#                total_multiples += count
#            else: 
#                total_multiples -= count
#    return total_multiples # return N - total_multiples for the total of non-multiples

"""
Consider a vector V indicized as V[x_const_1*x_1+...+x_const_n*x_n]
for a certain n > 0, with x_i in [0, X_1) for i in [1, n]. This function
receive the values for X_1, ..., X_n, x_const_1, ..., x_const_n, and
provides the number of distinct elements of V which can be accessed
from at least one combination of valid x_i-s.
Here 'Xs' and 'x_consts' are lists containing the above n values.
"""
def distinct_values(Xs : list[int], x_consts : list[int]):
    #indicized = (sum(x_const*(X - 1) for x_const, X in zip(x_consts, Xs)) + 1)/math.gcd(*x_consts)
    #return min(math.prod(Xs), indicized)
    
    if len(Xs) == 1: # trivial case
        return Xs[0]
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

#"""
#Continuing from the documentation for 'distinct_values'...
#Assume that V[x_const_1*x_1+...+x_const_n*x_n] is indexed twice, once
#with all x_i-s in range [0, X_i), and once with an x_j in [X_j, 2*X_j)
#for a give 'j', while all other ranges are unchanged. This function
#returns the common (overlapped) indices between the original indexing
#and the one with an updated j-th range.
#"""
#def overlapped_values(Xs : list[int], x_consts : list[int], j : int):
    #range_of_overlap = max(0, sum(x_const*(X - 1) for x_const, X in zip(x_consts, Xs)) + 1 - x_consts[j]*Xs[j])
    # TODO: possible overhead, try removing generators (e.g. enumerate and zip)
    #range_of_overlap = max(0, min(sum(x_const*(X - 1) for x_const, X in zip(x_consts, Xs)), x_consts[j]*(2*Xs[j] - 1) + sum(x_const*(X - 1) for i, (x_const, X) in enumerate(zip(x_consts, Xs)) if i != j)) - max(0, x_consts[j]*Xs[j]) + 1)
    #range_of_overlap = sum(x_const*(X - 1) for x_const, X in zip(x_consts, Xs)) + 1 # WRONG -> FIX ME!
    #range_of_overlap = sum(x_const*(X - 1) for x_const, X in zip(x_consts, Xs)) + x_consts[j]
    #return 1 + (range_of_overlap/math.gcd(*x_consts))
    #gcd = math.gcd(*x_consts)
    #return 0 if x_consts[j] % gcd != 0...