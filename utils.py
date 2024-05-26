from itertools import chain, combinations, permutations
from levels import *


def findConstraintsViolation(arch):
    violation = False
    for level in arch:
        for dim in ['D', 'E', 'L']:
            if dim not in level.dataflow and len(level.factors[dim]) != 0:
                print(f"Factor-dataflow missmatch for constraint ({dim}: {level.factors.dimProduct(dim)}) enforced on level \"{level.name}\"")
                violation = True
            if dim in level.factors_contraints and level.factors_contraints[dim] != level.factors.dimProduct(dim):
                print(f"Constraint violation, desired was ({dim}: {level.factors_contraints[dim]}), obtained was ({dim}: {level.factors.dimProduct(dim)}), on level \"{level.name}\"")
                violation = True
    return violation

"""
Computes the prime factors of 'n' and returns them as a dictionary where
keys are n's prime factors, and values are their arities (occurrencies).
"""
def primeFactors(n):
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

# moves a factor and updates tile sizes accordingly, returns False (and reverts changes) if it violates any constraint
def moveFactor(arch, src_level_idx, dst_level_idx, dimension, factor, amount = 1, skip_src_constraints = False, skip_dst_constraints = False):
    # check that the factor exists in the required amount
    if not arch[src_level_idx].removeFactor(dimension, factor, amount):
        return False
    # check src constraints
    if not arch[src_level_idx].checkConstraints() and not skip_src_constraints:
        arch[src_level_idx].addFactor(dimension, factor, amount)
        return False   
    arch[dst_level_idx].addFactor(dimension, factor, amount)
    # update tile sizes
    factor_to_amount = factor**amount
    if src_level_idx < dst_level_idx:
        for i in range(src_level_idx, dst_level_idx):
            arch[i].tile_sizes[dimension] *= factor_to_amount
    elif src_level_idx > dst_level_idx:
        for i in range(dst_level_idx, src_level_idx):
            arch[i].tile_sizes[dimension] /= factor_to_amount
    # check dst and all in between constraints
    constraints_check = [level.checkConstraints() for level in (arch[src_level_idx:dst_level_idx+1] if src_level_idx < dst_level_idx else arch[dst_level_idx:src_level_idx+1])]
    if not all(constraints_check) and not skip_dst_constraints:
        arch[src_level_idx].addFactor(dimension, factor, amount)
        assert arch[dst_level_idx].removeFactor(dimension, factor, amount) # something is broken, cannot undo the move
        if src_level_idx < dst_level_idx:
            for i in range(src_level_idx, dst_level_idx):
                arch[i].tile_sizes[dimension] /= factor_to_amount
        elif src_level_idx > dst_level_idx:
            for i in range(dst_level_idx, src_level_idx):
                arch[i].tile_sizes[dimension] *= factor_to_amount
        return False
    return True

def initFactors(arch, comp):
    # initialize with all factors on first level, all tile sizes of 1!
    arch[0].factors = Factors(D = primeFactors(comp.D), E = primeFactors(comp.E), L = primeFactors(comp.L))
    # TODO: to support a random starting point, add here a set of moveFactor
    # invocations, and add a method to "reset" the architecture (tile sizes and all)

def enforceFactorsConstraints(arch):
    # assuming that initially all factors are on the first level
    for i in range(1, len(arch)):
        level = arch[i]
        assert 'D' not in level.factors_contraints or 'D' in level.dataflow # cannot enforce constraint on element dimension not in dataflow
        assert 'E' not in level.factors_contraints or 'E' in level.dataflow # cannot enforce constraint on element dimension not in dataflow
        assert 'L' not in level.factors_contraints or 'L' in level.dataflow # cannot enforce constraint on element dimension not in dataflow
        constr_factors = Factors(
            D = primeFactors(level.factors_contraints['D']) if 'D' in level.factors_contraints else {},
            E = primeFactors(level.factors_contraints['E']) if 'E' in level.factors_contraints else {},
            L = primeFactors(level.factors_contraints['L']) if 'L' in level.factors_contraints else {}
            )
        if arch[0].factors.isSubset(constr_factors):
            for dim in ['D', 'E', 'L']:
                for fact, amount in constr_factors[dim].items():
                    assert moveFactor(arch, 0, i, dim, fact, amount, True, True) #if this assert fails, constraints are not satisfiable!

def checkDataflowConstraints(arch):
    for level in filter(lambda l : isinstance(l, MemLevel), arch):
        dim_idx = 0
        for dim in level.dataflow_constraints:
            if dim_idx == len(level.dataflow):
                return False
            while dim_idx < len(level.dataflow):
                if dim == level.dataflow[dim_idx]:
                    break
                dim_idx += 1
    return True

def setupBypasses(arch):
    # bypasses at the initial layer simply skip the cost of operands
    for bypass in ['in', 'w', 'out']:
        # if the first MemLevel has a bypass, no need to initialize it, just let it
        # affect its internal MOPs computation and that's it!
        last_before_bypass = 0
        i = 1
        while i < len(arch):
            if not isinstance(arch[i], MemLevel):
                i += 1
            elif bypass in arch[i].bypasses:
                last_bypassing = i
                while i < len(arch) and (not isinstance(arch[i], MemLevel) or bypass in arch[i].bypasses):
                    if isinstance(arch[i], MemLevel):
                        last_bypassing = i
                    i += 1
                arch[last_before_bypass].initBypass(bypass, arch[last_before_bypass+1:last_bypassing+1])
                last_before_bypass = i
                i += 1
            else:
                last_before_bypass = i
                i += 1

def updateInstances(arch):
    spatial_fanout = 1
    for i in range(len(arch)):
        level = arch[i]
        if isinstance(level, FanoutLevel):
            # consider only factors -> only actually used instances
            spatial_fanout *= level.factors.fullProduct()
        elif isinstance(level, MemLevel):
            level.instances = spatial_fanout
            level.next_is_compute = isinstance(arch[i+1], ComputeLevel) if i+1 < len(arch) else False

def hashFromFactors(arch):
    hsh = ""
    for level_idx in range(len(arch)):
        hsh += f"|{level_idx}"
        for dim in arch[level_idx].dataflow:
            hsh += f"{dim}"
            for factor, amount in arch[level_idx].factors[dim].items():
                hsh += f"{factor}{amount}"
    return hash(hsh)

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

# Interleaves the entries of "elements" of in all possible ways with the
# elements of "array". The array elements retain their order, while elements may not
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
Returns, if any, the sets of elements which can undergo a cyclic shift and
in so reach the configuration of arr2 starting from arr1
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
to reach arr2 from arr1 (no cyclic behaviour allowed).
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