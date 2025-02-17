from itertools import chain, combinations, permutations
from collections import defaultdict, deque
import threading
import heapq
import math

from types import TracebackType
from typing import Generator, Generic, Iterable, Iterator, Optional, Type, TypeVar, Union
T = TypeVar('T')
U = TypeVar('U')
S = TypeVar('S')
W = TypeVar('W')

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

# NOT IN USE ANYMORE
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

# NOT IN USE ANYMORE
"""
Filters a list of permutations, keeping only the first permutation seen for
each unique configuration of the N innermost (highest indices) elements.
"""
def filter_by_unique_innermost_elems(perms_list : list[list[T]], N : int) -> list[T]:
    seen = set()
    result = []
    for perm in perms_list:
        innermost = tuple(perm[-N:])
        if innermost not in seen:
            seen.add(innermost)
            result.append(perm)

    return result

"""
Returns a list of all versions of 'array' that differ by a rotation (or shift).
"""
def rotations(array : list[T]) -> list[list[T]]:
    return [array[i:] + array[:i] for i in range(len(array))] or [[]]

# NOT IN USE ANYMORE
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

# NOT IN USE ANYMORE
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
Generates all ordered pairs which can be sampled from a list.
"""
def ordered_pairs(lst : list[T]) -> Iterator[tuple[T, T]]:
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            yield (lst[i], lst[j])

# NOT IN USE ANYMORE
"""
Reorders the lists in 'lists' according to their N-elements tail.
Each tail can never occur again until all other remaining ones
are picked once (similarly to a round-robin schedule).
"""
def roundrobin_lists_reordering(lists : list[list[T]], N : int = 1) -> list[list[T]]:
    groups = defaultdict(deque)
    for lst in lists:
        key = tuple(lst[-N:])
        groups[key].append(lst)
    queue, result = deque(groups.keys()), []
    while queue:
        seen, temp_queue = set(), deque()
        while queue:
            key = queue.popleft()
            if key in seen and groups[key]:
                temp_queue.append(key)
                continue
            if groups[key]:  
                result.append(groups[key].popleft())
                seen.add(key)
            if groups[key]:
                temp_queue.append(key)
        queue = temp_queue
    return result

"""
Filters a list of permutations by removing equivalent
ones based on the position of certain relevant elements.

Two permutations are equivalent iff for each set of relevant elements:
- They have the same k (the highest index of a relevant element).
- Elements strictly after k are the same but may be in different order.
- Elements from k-N+1 to k (if N > 0) are exactly the same and in the same order.
- Elements strictly before k are the same but may be in different order.
"""
def filter_equivalent_perms(perms : list[list[T]], relevant_elements_sets : set[set[T]], N : int = 0) -> list[list[T]]:
    def key_func(perm, rel_elems):
        k = max((i for i, el in enumerate(perm) if el in rel_elems), default=-1)
        fixed_prefix = tuple(perm[max(0, k - N + 1):k + 1]) if N > 0 else ()
        unordered_before = frozenset(perm[:max(0, k - N + 1)])
        unordered_after = frozenset(perm[k + 1:])
        return k, fixed_prefix, unordered_before, unordered_after

    seen = {}
    for perm in perms:
        key = tuple(chain(key_func(perm, rel_elems) for rel_elems in relevant_elements_sets))
        if key not in seen:
            seen[key] = perm
    return list(seen.values())

"""
Generates a valid ordering of elements based on the given relative
order pairs. Let 'pairs' be a list of tuples each representing the
relative order between two elements. The result will be a list
containing a valid ordering of elements that satisfies all constraints.
"""
def satisfy_relative_order(pairs : list[tuple[T, T]]) -> list[T]:
    elements = set()
    for a, b in pairs:
        elements.add(a)
        elements.add(b)
    graph = defaultdict(list)
    in_degree = {element: 0 for element in elements}
    for a, b in pairs:
        graph[a].append(b)
        in_degree[b] += 1
    queue = deque([node for node in elements if in_degree[node] == 0])
    result = []
    while queue:
        current = queue.popleft()
        result.append(current)
        for neighbor in graph[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    if len(result) != len(elements):
        raise ValueError("The input pairs contain a cycle or are otherwise invalid.")
    return result

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


# >>> Miscellaneus support data structures and classes

"""
Thread-safe heap implementation where each entry is a pair (value, data),
with the heap being ordered w.r.t. value.

Additionally a counter and a set, both thread-safe, are made available
alongside each heap, to track additional information.

Constructor arguments:
- is_min_heap: set to True for a min-heap, False for a max-heap.
- initial_counter: initial value for the integrated counter.
"""
class ThreadSafeHeap(Generic[T, U, S, W]):
    def __init__(self, is_min_heap : bool = False, initial_counter : int = 0):
        self.heap : list[tuple[T, U]] = []
        self.counter : int = initial_counter
        self.dict : dict[S, W] = {}
        self.is_min_heap = is_min_heap
        self.lock = threading.RLock()

    def _wrap_value(self, value : T) -> T:
        return value if self.is_min_heap else -value

    def _unwrap_value(self, value : T) -> T:
        return value if self.is_min_heap else -value

    def push(self, value : T, data : U, increase_counter : int = 1) -> None:
        with self.lock:
            heapq.heappush(self.heap, (self._wrap_value(value), id(data), data))
            self.counter += increase_counter

    def pop(self) -> tuple[T, U]:
        with self.lock:
            if not self.heap:
                raise IndexError("pop from an empty heap")
            wrapped_value, _, data = heapq.heappop(self.heap)
            return self._unwrap_value(wrapped_value), data

    def peek(self) -> tuple[T, U]:
        with self.lock:
            if not self.heap:
                raise IndexError("peek from an empty heap")
            wrapped_value, _, data = self.heap[0]
            return self._unwrap_value(wrapped_value), data

    def isEmpty(self) -> bool:
        with self.lock:
            return len(self.heap) == 0

    def getCounter(self) -> int:
        with self.lock:
            return self.counter

    def increaseCounter(self, amount : int = 1) -> None:
        with self.lock:
            self.counter += amount

    def addToDict(self, key : S, val : W) -> None:
        with self.lock:
            self.dict[key] = val

    def isInDict(self, key : S) -> bool:
        with self.lock:
            return key in self.dict

    def getFromDict(self, key : S, default : S) -> W:
        with self.lock:
            return self.dict[key] if key in self.dict else default

    def __iter__(self) -> Generator[tuple, None, None]:
        with self.lock:
            return iter((self._unwrap_value(wrapped_value), data) for wrapped_value, _, data in self.heap)

    def __len__(self):
        with self.lock:
            return len(self.heap)

    def __enter__(self) -> bool:
        self.lock.acquire()

    def __exit__(self, exc_type : Optional[Type[BaseException]], exc_val : Optional[BaseException], exc_tb : Optional[TracebackType]) -> bool:
        self.lock.release()

"""
Helper class mean to wrap a lock that may not exist.
Used in "optimizeDataflows".
"""
class OptionalLock:
    def __init__(self, lock : Optional[threading.Lock]):
        self.lock = lock

    def __enter__(self) -> bool:
        if self.lock is not None:
            self.lock.acquire()

    def __exit__(self, exc_type : Optional[Type[BaseException]], exc_val : Optional[BaseException], exc_tb : Optional[TracebackType]) -> bool:
        if self.lock is not None:
            self.lock.release()