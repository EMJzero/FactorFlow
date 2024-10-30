from itertools import chain, combinations, permutations
import math

from settings import *
from levels import *


# >>> Helper functions to operate on the architecture (camelCase ones)

"""
Checks factors allocation constraints. Returns False if a violation is found.
"""
def findConstraintsViolation(arch, verbose = True):
    violation = False
    for level in arch:
        for dim in ['M', 'K', 'N']:
            if dim not in level.dataflow and len(level.factors[dim]) != 0:
                if verbose: print(f"Factor-dataflow missmatch for constraint ({dim}: {level.factors.dimProduct(dim)}) enforced on level \"{level.name}\"")
                violation = True
            if dim in level.factors_constraints and level.factors_constraints[dim] != level.factors.dimProduct(dim):
                if verbose: print(f"Constraint violation, desired was ({dim}: {level.factors_constraints[dim]}), obtained was ({dim}: {level.factors.dimProduct(dim)}), on level \"{level.name}\"")
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

"""
Computes the prime factors of 'n' and returns them as a list.
"""
def primeFactorsList(n):
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
Moves a factor between the same dimension of two levels, transitioning
between adjacent mappings. it also updates tile sizes accordingly for all levels
between the affected ones.

Returns False (and reverts changes) if the move violates any constraint.

Arguments:
- src_level_idx: level giving up the prime factor
- dst_level_idx: level receiving the prime factor
- dimension: the computation's dimension of the moved factor
- factor: specify which prime factor is moved
- amount: which arity of the factor is moved
- skip_src_constraints: if True, any constraints violation on the source level
                        is ignored.
- skip_dst_constraints: if True, any constraints violation on the destination level
                        is ignored.
"""
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

"""
Initializes the architecture's mapping with all the computation's prime
factors placed on the first level. This is the mapper's starting point.
"""
def initFactors(arch, comp):
    # initialize with all factors on first level, all tile sizes of 1!
    arch[0].factors = Factors(M = primeFactors(comp.M), K = primeFactors(comp.K), N = primeFactors(comp.N))

"""
This function must start from arch having all factors on its first level,
then it ensures that all constraints of all levels are satisfied.
If 'allow_padding' is True then computation dimensions not exactly
divided by constraints will be padded up to a multiple of constraints.

This function assumes that the computation is larger than what required
by constraints, if this is not satisfied, an assertion will be triggered.
-> use 'fitConstraintsToComp' to prevent such a situation.
"""
def enforceFactorsConstraints(arch, allow_padding = False, verbose_padding = True):
    # assuming that initially all factors are on the first level
    for i in range(1, len(arch)):
        level = arch[i]
        for k in level.factors_constraints.keys():
            assert k[0] in level.dataflow, f"Level {level.name}: cannot enforce constraint on dimension {k[0]} that is not in dataflow {level.dataflow}."
        # initialize only == and >= constraints, leave <= set to 1
        constr_factors = Factors(
            M = primeFactors(level.factors_constraints['M']) if 'M' in level.factors_constraints else (primeFactors(level.factors_constraints['M>=']) if 'M>=' in level.factors_constraints else {}),
            K = primeFactors(level.factors_constraints['K']) if 'K' in level.factors_constraints else (primeFactors(level.factors_constraints['K>=']) if 'K>=' in level.factors_constraints else {}),
            N = primeFactors(level.factors_constraints['N']) if 'N' in level.factors_constraints else (primeFactors(level.factors_constraints['N>=']) if 'N>=' in level.factors_constraints else {}),
            )
        if arch[0].factors.isSubset(constr_factors) or allow_padding:
            for dim in ['M', 'K', 'N']:
                dim_size = arch[0].factors.dimProduct(dim)
                constraint = constr_factors.dimProduct(dim)
                if dim_size%constraint == 0:
                    for fact, amount in constr_factors[dim].items():
                        assert moveFactor(arch, 0, i, dim, fact, amount, True, True), f"Constraints are not satisfiable! Cannot give {amount} instances of prime factor {fact} to level {level.name} on dimension {dim}."
                else:
                    padded_dim_size = dim_size + constraint - dim_size%constraint
                    if verbose_padding: print(f"PADDING: enlarged {dim} from {dim_size} to {padded_dim_size} to respect constraints on level {level.name}")
                    arch[0].factors[dim] = primeFactors(padded_dim_size)
                    arch[0].factors.resetDimProducts([dim])
                    for fact, amount in constr_factors[dim].items():
                        # be wary that here we skip constraints checks in moveFactor, so one must follow up this method with findConstraintsViolation
                        assert moveFactor(arch, 0, i, dim, fact, amount, True, True), "Failed to enforce constraints even with padding..."

"""
Checks dataflow (loop ordering) constraints. Returns False if a violation is found.
"""
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

"""
Initializes bypasses between MemLevels of the architecture.
For each operand and bypass, setup a pointer in the last MemLevel before the bypass
to the list of levels affected by the bypass.

This way such last level can compute its MOPs according to the level it actually
supplies data to.
"""
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

"""
Updates the count of ACTIVE instances throughout Mem- and Compute- Levels.
An instance is ACTIVE if a FanoutLevel maps a spatial iteration to it.
"""
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
        elif isinstance(level, ComputeLevel):
            level.instances = spatial_fanout

"""
Clears the accumulators for incrementally updated tile sizes and
prime factors products in the architecture.
"""
def resetTilesAndFactors(arch):
    for level in arch:
        level.factors.clear()
        for dim in ['M', 'K', 'N']:
            level.tile_sizes[dim] = 1

"""
Returns the overall utilization of spatial instances of a mapping.
"""
def fanoutsUtilization(arch):
    utilization = 1
    for level in arch:
        if isinstance(level, FanoutLevel):
            utilization *= level.factors.fullProduct()/level.mesh
    return utilization

"""
Hash unique for each factors allocation and dataflows pair.
"""
def hashFromFactors(arch):
    hsh = ""
    for level_idx in range(len(arch)):
        hsh += f"|{level_idx}"
        for dim in arch[level_idx].dataflow:
            hsh += f"{dim}"
            for factor, amount in arch[level_idx].factors[dim].items():
                hsh += f"{factor}{amount}"
    return hash(hsh)

"""
Reduces constraints to the largest possible ones that can be satisfied
by the present computations.

If arch and comp names are provided, an error is printed and the return
value must be checked for failure. Otherwise, an assertion requires
constraints comply.
"""
def fitConstraintsToComp(arch, comp, arch_name = None, comp_name = None):
    failed = False
    for dim in ['M', 'K', 'N']:
        total_constraint = 1
        for level in arch:
            # only == and >= constraints may be unsatisfiable at this point, since <= ones are forbidden on the
            # outermost architecture level, therefore factors can always stay there if constraints are too tight!
            if dim + '>=' in level.factors_constraints:
                eq = '>='
            else:
                eq = ''
            if dim in level.factors_constraints or dim + '>=' in level.factors_constraints:
                if comp[dim] // total_constraint < level.factors_constraints[dim + eq]:
                    if comp[dim] // total_constraint <= 0:
                        if arch_name and comp_name:
                            print(f"ERROR: failed to fit comp: {comp_name} to arch: {arch_name} because the constraint on level: {level.name} and dimension: {dim} ({eq}{level.factors_constraints[dim + eq]}) cannot be satisfied by comp ({dim}: {comp[dim]})!")
                        else:
                            assert False, f"Failed to fit comp to arch because the constraint on level: {level.name} and dimension: {dim} ({eq}{level.factors_constraints[dim + eq]}) cannot be satisfied by comp ({dim}: {comp[dim]})!"
                        failed = True
                        break
                    print(f"WARNING: updating constraint ({dim}: {level.factors_constraints[dim + eq]}) on level \"{level.name}\" to ({dim}: {comp[dim] // total_constraint}) to fit the computation.")
                    level.factors_constraints[dim + eq] = comp[dim] // total_constraint
                elif eq == '' and (comp[dim] // total_constraint) % level.factors_constraints[dim] != 0 and not Settings.PADDED_MAPPINGS:
                    assert False, f"Failed to fit comp to arch because the constraint on level {level.name} ({dim}: {level.factors_constraints[dim]}) does not divide comp dimension {dim} ({comp[dim]}) exactly. To compensate, consider setting 'Settings.PADDED_MAPPINGS' to True."
                total_constraint *= level.factors_constraints[dim + eq]
        if failed:
            break
    return failed


# >>> Miscellaneus functions working with iterables/arrays (snake_cased ones)

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