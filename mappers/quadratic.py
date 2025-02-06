from functools import reduce
from copy import deepcopy
import math

from typing import Iterator, Union
import threading

from settings import *
from factors import *
from levels import *
from prints import *
from model import *
from utils import *
from arch import *


"""
Mapper Step 2: allocate to fanout levels the maximum number of iterations
               which can fit on their instances.
"""
def fanoutMaximization(arch : Arch, comp : Shape, bias_read : bool, verbose : bool = False) -> None:
    # TECHNIQUE: Find the prime factors of the mesh, and pick the largest common ones with the dimension
    # mapped along that mesh, continue picking from the largest ones in common until you run out!
    # NOTE: When making this handle Fanouts with multiple unrolled dimensions, the issue is dividing the fanout size across dimensions
    # IDEA: Use a binary (ternary) tree expansion, start with 50-50 (30-30-30), explore each child and prune worse branches?
    # IMPORTANT: from optimizeDataflow, if there are unconstrained dimensions, those are always the first ones!
    # NOTE: This step applies to ComputeLevels too!
    if verbose: print("\nStarting fanout maximization:\n")
    if Settings.ONLY_MAXIMIZE_ONE_FANOUT_DIM:
        if Settings.PADDED_MAPPINGS:
            for dim in arch.coupling.dims:
                total_mesh = math.prod([level.mesh for level in arch if isinstance(level, SpatialLevel) and level.dataflow[0] == dim])
                mesh_factors = [f for level in arch if isinstance(level, SpatialLevel) and level.dataflow[0] == dim for f in prime_factors_list(level.mesh)]
                dim_size = arch[0].factors.dimProduct(dim)
                dim_factors = arch[0].factors.toList(dim)
                if total_mesh > dim_size:
                    used_factors, padding = smallest_product_greater_than(mesh_factors, dim_size)
                    if padding != math.inf and not all([f in dim_factors for f in used_factors]): # only pad if some different factor achieved higher utilization
                        if Settings.VERBOSE_PADDED_MAPPINGS: print(f"PADDING: Arch: {arch.name}: enlarged {dim} from {dim_size} to {dim_size + padding}")
                        arch[0].factors[dim] = prime_factors(dim_size + padding)
                        arch[0].factors.resetDimProducts([dim])
                else:
                    if not all([f in dim_factors for f in mesh_factors]): # only pad if you are not already a multiple
                        padded_dim_size = dim_size + total_mesh - dim_size%total_mesh
                        if Settings.VERBOSE_PADDED_MAPPINGS: print(f"PADDING: Arch: {arch.name}: enlarged {dim} from {dim_size} to {padded_dim_size}")
                        arch[0].factors[dim] = prime_factors(padded_dim_size)
                        arch[0].factors.resetDimProducts([dim])
        
        for i in range(1, len(arch) - 1): # first round: start from common factors
            level = arch[i]
            if isinstance(level, SpatialLevel):
                dim = level.dataflow[0]
                common_mesh_factors = [f for f in prime_factors(level.mesh).keys() if f in arch[0].factors[dim]]
                for f in sorted(common_mesh_factors, reverse=True): # try largest factors first
                    amount = arch[0].factors[dim][f]
                    while amount > 0:
                        if arch.moveFactor(0, i, dim, f, amount):
                            break
                        amount -= 1 # lower the amount until you succeed
        
        for i in range(1, len(arch) - 1): # second round: fill any remaining space as best as you can
            level = arch[i]
            if isinstance(level, SpatialLevel):
                for dim in level.dataflow: # as a last resort, try dimensions beyond the first one
                    if dim in level.factors_constraints:
                        continue
                    if level.factors.fullProduct() < level.mesh:
                        space = level.mesh // level.factors.fullProduct()
                        factors, _ = largest_product_less_than(arch[0].factors.toList(dim), space)
                        for f in factors:
                            if not arch.moveFactor(0, i, dim, f, 1) and verbose:
                                print(f"Arch: {arch.name}: fanout maximization failed to fill up the leftover space on level {level.name}, dim {dim} with factor {f} (mesh: {level.mesh}, space: {space})...")
    
    else:
        for i in range(1, len(arch) - 1):
            level = arch[i]
            if isinstance(level, SpatialLevel):
                assert len(level.dims) <= 2, f"Arch: {arch.name} -> Level: {level.name}: CURRENT LIMITATION - at most 2 dimensions on the same fanout, limit dims ({level.dims}) to at most 2 entries when Settings.ONLY_MAXIMIZE_ONE_FANOUT_DIM is False."
                # TODO: as of now this accepts only at most 2 dimensions on same fanout - pick mesh_split_factor >= 3 to allow more?
                mesh_prime_factors = prime_factors(level.mesh)
                common_mesh_factors = [f for f in mesh_prime_factors.keys() if f in [ft for dim in level.dims for ft in arch[0].factors[dim]]]
                ping_pong = 0
                for f in sorted(common_mesh_factors, reverse=True): # try largest factors first
                    for _ in range(max(map(lambda dim : arch[0].factors[dim][f] if f in arch[0].factors[dim] else 0, level.dims))):
                        if f not in arch[0].factors[level.dims[ping_pong]]:
                            ping_pong = len(level.dims) - 1 - ping_pong
                        if arch.moveFactor(0, i, level.dims[ping_pong], f):
                            ping_pong = len(level.dims) - 1 - ping_pong
        
    if verbose: print(f"After fanout maximization (Wart: {Wart(arch, comp, bias_read):.3e}):")
    if verbose: printFactors(arch)


"""
Generates/Enumerates all moves producing adjacent mappings to the provided one.
Returned moves may violate constraints, use utils.moveFactor to apply them.

Arguments:
- iterate_amounts: if True, adjacency is extended to the idea of moving
                   any arity of a factor between loops on the same dimension.
- skip_spatial: if True, spatial levels are not considered for adjacency.
"""
def factorsIterator(arch : Arch, iterate_amounts : bool = False, skip_spatial : bool = False) -> Iterator[tuple[int, str, int, int]]:
    for level_idx in range(len(arch)):
        if skip_spatial and (isinstance(arch[level_idx], SpatialLevel)):
            continue
        for dim in arch[level_idx].dataflow:
            for factor in list(arch[level_idx].factors[dim].keys()):
                if iterate_amounts:
                    for amount in range(1, arch[level_idx].factors[dim][factor] + 1):
                        yield level_idx, dim, factor, amount
                else:
                    yield level_idx, dim, factor, 1

"""
Mapper Step 3: greedy descent factors allocation, navigating the map-space
               via adjacent mappings, until a locally optimal one is found.

Adjacency: two mappings are adjacent if one can be constructed from the other
           by moving exactly one prime factor between two loops/levels on the
           same dimension.
"""
def factorFlow(arch : Arch, comp : Shape, bias_read : bool, verbose : bool = False) -> tuple[Arch, float]:
    if verbose: print("-------- factorFlow --------")
    already_initialized = arch.initialized
    if not already_initialized:
        arch.initFactors(comp)
        arch.enforceFactorsConstraints(Settings.PADDED_MAPPINGS, Settings.VERBOSE_PADDED_MAPPINGS)
    assert arch.checkFactorsConstraints() and arch.checkDataflowConstraints(), ("Ill-posed constraints:" if not already_initialized else "Improperly initialized arch:") + f"\n{arch.logConstraintsViolations()}"
    if verbose: print(f"Initial condition (Wart: {Wart(arch, comp, bias_read):.3e}):")
    if verbose: printFactors(arch)
    
    # maximize fanout dimensions
    if not already_initialized:
        fanoutMaximization(arch, comp, bias_read, verbose)
    else:
        if verbose: print("\nSkipping fanout maximization...\n")
    
    if verbose: print("\nStarting FactorFlow tiling optimization:\n")
    
    # never re-visit the same mapping
    already_seen = [arch.hashFromFactors()]
    # one-factor-steps greedy optimization
    best_wart = Wart(arch, comp, bias_read)
    # track the count of moves performed
    moves_count = 0
    
    def exploreOneStep(remaining_steps = 1, target_src_level_idx = None, target_dim = None, target_factor = None, target_dst_level_idx = None):
        choices = {}
        for src_level_idx, dim, factor, amount in factorsIterator(arch, iterate_amounts = Settings.ITERATE_AMOUNTS, skip_spatial= Settings.FREEZE_SA):
            if target_src_level_idx and target_dim and target_factor and (target_src_level_idx != src_level_idx or target_dim != dim or target_factor != factor):
                continue
            #print("Initial proposal: ", src_level_idx, dim, factor)
            
            # go back to the first valid source
            while src_level_idx >= 0 and dim in arch[src_level_idx].factors_constraints:
                src_level_idx -= 1
            if src_level_idx < 0 or factor not in arch[src_level_idx].factors[dim]:
                continue
            
            # pick the factor from the highest level that you can
            #new_src = src_level_idx - 1
            #while new_src >= 0 and factor in arch[new_src].factors[dim]:
            #    if dim not in arch[new_src].factors_constraints:
            #        src_level_idx = new_src
            #    new_src -= 1
            
            for dst_level_idx in range(len(arch)):
                if dst_level_idx != src_level_idx and dim not in arch[dst_level_idx].factors_constraints and dim in arch[dst_level_idx].dataflow and (not Settings.FREEZE_SA or not isinstance(arch[dst_level_idx], SpatialLevel)):
                    #print(src_level_idx, dst_level_idx, dim, factor)
                    if target_dst_level_idx and dst_level_idx != target_dst_level_idx:
                        continue
                    if arch.moveFactor(src_level_idx, dst_level_idx, dim, factor, amount, skip_src_constraints = Settings.NO_CONSTRAINTS_CHECK_DURING_MULTISTEP and remaining_steps > 1):
                        hsh = arch.hashFromFactors()
                        if hsh not in already_seen:
                            already_seen.append(hsh)
                            if remaining_steps > 1:
                                nested_choices = exploreOneStep(remaining_steps - 1, target_dst_level_idx = src_level_idx if Settings.LIMIT_NEXT_STEP_DST_TO_CURRENT_SRC else None)
                                if len(nested_choices) == 0:
                                    choices[(src_level_idx, dst_level_idx, dim, factor, amount)] = Wart(arch, comp, bias_read)
                                else:
                                    wart = Wart(arch, comp, bias_read)
                                    # >>> GREEDY MOVE <<<
                                    best_choice = max(nested_choices, key=nested_choices.get)
                                    if nested_choices[best_choice] >= wart:
                                        choices[(src_level_idx, dst_level_idx, dim, factor, amount) + best_choice] = nested_choices[best_choice]
                                    else:
                                        choices[(src_level_idx, dst_level_idx, dim, factor, amount)] = wart
                            else:
                                choices[(src_level_idx, dst_level_idx, dim, factor, amount)] = Wart(arch, comp, bias_read)
                        assert arch.moveFactor(dst_level_idx, src_level_idx, dim, factor, amount, skip_dst_constraints = Settings.NO_CONSTRAINTS_CHECK_DURING_MULTISTEP and remaining_steps > 1) # something went wrong, unreversible move of a factor    
        return choices
    
    while True:
        choices = exploreOneStep(remaining_steps = Settings.STEPS_TO_EXPLORE)
        if len(choices) == 0:
            if verbose: print(f"No valid follow-up configuration, stopping, current Wart: {best_wart:.3e}")
            break
        # >>> GREEDY MOVE <<<
        best_choice = max(choices, key=choices.get)
        if choices[best_choice] < best_wart:
            if verbose: print(f"Stopping with current Wart: {best_wart:.3e}, while best choice is: {choices[best_choice]:.3e}")
            break
        else:
            # each individual choice is defined by 5 parameters, chained to another 5 for each nested exploration step
            multisteps = len(best_choice) // 5
            moves_count += multisteps
            for i in range(multisteps):
                assert arch.moveFactor(best_choice[5*i + 0], best_choice[5*i + 1], best_choice[5*i + 2], best_choice[5*i + 3], best_choice[5*i + 4], skip_src_constraints = Settings.NO_CONSTRAINTS_CHECK_DURING_MULTISTEP and i < multisteps - 1) # best choice is an invalid mapping
            best_wart = choices[best_choice]
    
    updateStats(arch, bias_read)
    if verbose: print(f"Final condition:\nWart: {best_wart}\nEDP: {EDP(arch, bias_read, True):.3e} (J*cycle)")
    if verbose: printFactors(arch)
    return arch, best_wart, moves_count

"""
Mapper Step 1: exhaustively iterate loop permutations/dataflows.

IDEA: pre-exclude some permutations!
HOW-TO: exploit equi-dataflow matches to skip redundant permutations. However we cannot truly skip permutations,
but we can re-start the factorFlow exploration from where it was left -> adaptive programming!

Adaptive (memory) programming: a permutation is in equi-dataflow with an already
         tried one if they have the same order of loops with more than one iteration.
         That is an equi-dataflow match, and we restart step 3 from the mapping known
         for the past permutation, with the current permutation, rather than starting
         with all prime factors on the first level.
NOTE: in brief, we cannot truly skip the configuration, but we can re-start the
      "factorFlow" exploration from where it was left -> adaptive programming!
         
The function is meant as the entry point for multiple processes or threads:
- thread_idx: set to -1 if multiple tasks are not used, otherwise set to the current
              task's index.
- threads_count: total number of employed tasks.
- return_list: shared list among tasks for the return values.
"""
def optimizeDataflows(arch : Arch, comp : Shape, bias_read : bool, thread_idx : int = -1, threads_count : int = 1, past_perms : dict[tuple[int, ...], ThreadSafeHeap[float, list[LevelCore], int, int]] = None, lock : threading.Lock = None, barrier : threading.Barrier = None, verbose : bool = False) -> Optional[tuple[Arch, float]]:
    if verbose and thread_idx <= 0: print("-------- optimizeDataflows --------")
    # this approach requires spatial levels to be the first explored/permuted ones
    targets = list(filter(lambda l : Settings.ONLY_MAXIMIZE_ONE_FANOUT_DIM and isinstance(l, SpatialLevel) and len(l.dataflow) > 1, arch)) + list(filter(lambda l : isinstance(l, MemLevel), arch))
    def gen_permutations(level):
        if isinstance(level, MemLevel):
            if '_' in level.dataflow_constraints:
                perms = [perm for perm in slot_in(level.dataflow_constraints, level.dataflow, '_')]
            else:
                perms = [perm for perm in interleave(level.dataflow_constraints, [dim for dim in level.dataflow if dim not in level.dataflow_constraints])]
            # in the following two cases we speculate on all loops having more than one iteration, we then take advantage of single-iteration loops with equi-dataflow matches
            if Settings.DISTINCT_REUSE_OPPORTUNITIES and level.multiple_reuses:
                # considering skipped dimensions and halo reuse, for each operand changing order of loops before and after the innermost iterated dimension coupled to the operand doesn't impact reuse, while such innermost dimension dictates the halo reuse
                # => remove permutations with a different order of loops inside those determining the dataflow or outside them for each operand
                return filter_equivalent_perms(perms, {frozenset(arch.coupling.flat_in_coupling), frozenset(arch.coupling.flat_w_coupling), frozenset(arch.coupling.flat_out_coupling)}, 1)
            elif Settings.DISTINCT_REUSE_OPPORTUNITIES:
                # same as above, but we don't have halo reuse
                return filter_equivalent_perms(perms, {frozenset(arch.coupling.flat_in_coupling), frozenset(arch.coupling.flat_w_coupling), frozenset(arch.coupling.flat_out_coupling)})
            else:
                return perms
        elif isinstance(level, SpatialLevel):
            return [rot + [dim for dim in level.dims if dim in level.factors_constraints] for rot in rotations([dim for dim in level.dims if dim not in level.factors_constraints])]
    # list of lists of all valid permutations for each targeted level
    permutations = list(map(gen_permutations, targets))
    perms_ranges = [(0, len(perm)) for perm in permutations] # as in [low_idx, high_idx) to explore in the current thread
    
    # divide permutations across threads (if multithreading is enabled)
    if thread_idx != -1:
        partitions_per_level = [1 for _ in permutations]
        factors_per_level = [prime_factors_list(len(perm)) for perm in permutations]
        cumulative_product, circular_idx = 1, -1
        # circularly partition each level's permutations by the smaller of their amount's prime factors, until there are more partitions than threads
        while cumulative_product < threads_count:
            circular_idx = next(((i + circular_idx + 1) % len(factors_per_level) for i, factors in enumerate(factors_per_level[circular_idx + 1:] + factors_per_level[:circular_idx + 1]) if len(factors) > 0), None)
            if circular_idx == None:
                break
            factor = factors_per_level[circular_idx].pop(0)
            partitions_per_level[circular_idx] *= factor
            cumulative_product *= factor
        if circular_idx == None and thread_idx >= cumulative_product:
            # more threads than permutations, terminate extra threads
            if barrier:
                barrier.wait()
            return
        tmp_thread_idx = thread_idx
        threads_on_current_permutation = threads_count
        # partition permutations by removing those not assigned to the present thread
        for i in range(len(permutations)):
            partitioning_idx = tmp_thread_idx % partitions_per_level[i]
            partitioning_stride = len(permutations[i]) // partitions_per_level[i]
            if threads_on_current_permutation < partitions_per_level[i]:
                extra = partitions_per_level[i] - threads_on_current_permutation
                extra_per_thread = math.ceil(extra // threads_on_current_permutation)
                perms_ranges[i] = (partitioning_idx*(partitioning_stride*(1 + min(extra_per_thread, extra - (partitioning_idx - 1)*extra_per_thread))), (partitioning_idx + 1)*(partitioning_stride*(1 + min(extra_per_thread, extra - partitioning_idx*extra_per_thread))))
                break
            else:
                perms_ranges[i] = (partitioning_idx*partitioning_stride, (partitioning_idx + 1)*partitioning_stride)
            threads_on_current_permutation = (threads_on_current_permutation // partitions_per_level[i]) + (partitioning_idx < threads_on_current_permutation % partitions_per_level[i])
            tmp_thread_idx //= partitions_per_level[i]
    
    total_perms = sum((range[1] - range[0])*(len(perms_ranges) - i) for i, range in enumerate(perms_ranges))
    #print(f"Total permutations: {total_perms}")
    
    if verbose and thread_idx <= 0: print(f"Starting MSE:\n")
    if verbose:
        if thread_idx == -1: print(f"Dataflow permutations to try: {total_perms}")
        else: print(f"Dataflow permutations to try in thread {thread_idx}: {total_perms}")
    
    # ripple search (backtracking) algorithm counters:
    current_level = 0 # level whose permutations are currently being explored (and indicized by 'current_perms'), increment by 1 once perms on this level are exhausted
    innermost_level_to_explore = 1 # innermost level that has to be explored next, once it is explored reset 'current_level' to 0
    current_perms = [low_idx for low_idx, _ in perms_ranges] # permutations currently being explored (fixed for every level except 'current_level' that is currently being updated)
    current_best_perm = (0, current_perms[current_level]) # best (wart, permutations_idx) pair found so far on level 'current_level', reset when changing the explored level
    final_mapping = (0, None) # (wart, mapping) pair filled when current_level == len(targets) -1, that is, when the innermost level is being explored
    # NOTE: no need to re-evaluate the Wart immediately after moving down, since the present Wart is the best one found on the above level (that you froze while using the current permutations)!
    # Thread-shared cache of past solutions to get equi-dataflow matches:
    # Key: current_perms[0:current_level] + ['_'] + current_perms[current_level + 1:innermost_level_to_explore], as it should depend on all fixed levels, and not on the current one or the yet unexplored ones.
    # => check for equi-dataflow on the level whose index is the same as the '_' placeholder, all other relevant ones are already implied to have the same permutation.
    # Value: a max-heap of past permutations and their mappings tried for the key's configuration of permutations, sorted by Wart.
    # Place in key (<empty tuple>)'s heap the best solution found by each thread, so that the caller can extract it.
    # This datastructure does not propagate solutions, it merely caches them, it's each thread's job to pick its own best solution once it finishes exploring the innermost level.
    # Delete entries when their heap contains as many values as the permutations on the level with the same index as '_'.
    # No need to store anything for permutations on SpatialLevels.
    past_perms = past_perms if past_perms else {() : ThreadSafeHeap()}
    
    """
    Returns the key for 'past_perms' based on the current state of exploration.
    """
    def getKey() -> tuple[Union[str, int]]:
        #return tuple(current_perms[0:current_level] + ['_'] + current_perms[current_level + 1:innermost_level_to_explore]) # issue: can't distinguish the cache of threads working on different inner not yet explored perms...
        return tuple(current_perms[:current_level] + ['_'] + current_perms[current_level + 1:])
    
    # counters for explored permutations
    tried_perms = 0
    eqmatched_perms = 0
    
    # initialize shared data structures and synch threads
    lock = OptionalLock(lock)
    if isinstance(targets[current_level], MemLevel):
        with lock:
            key = getKey()
            if key not in past_perms:
                # new heap
                past_perms[key] = ThreadSafeHeap()
            else:
                # the heap is already used by somebody else, reduce the counter by the number of times you will be adding entries
                past_perms[key].increaseCounter(-(perms_ranges[current_level][1] - perms_ranges[current_level][0]))
    if barrier:
        barrier.wait()
    
    """
    Returns a dictionary indicating for each dimension if it has only a single iteration on level 'level_idx' of 'mapping'.
    """
    def factorsAtOne(mapping : Union[list[LevelCore], Arch], level_idx : int) -> dict[str, bool]:
        return {dim:  mapping[level_idx].factors.dimProduct(dim) == 1 for dim in arch.coupling.dims}
    
    """
    Checks for an equi-dataflow match with past solutions. Returns the best match found or None if no match was found. Let 'level_idx' be the level on which to check for a match.
    """
    def equiDataflowMatch() -> Union[list[LevelCore], bool]:
        if not Settings.PERM_SKIP or not isinstance(targets[current_level], MemLevel):
            return None
        new_perm = permutations[current_level][current_perms[current_level]]
        for _, mapping in past_perms[getKey()]:
            old_perm = mapping[current_level].dataflow
            new_perm_effective, old_perm_effective = list(filter(lambda dim : not factorsAtOne(mapping, current_level)[dim], new_perm[::-1])), list(filter(lambda dim : not factorsAtOne(mapping, current_level)[dim], old_perm[::-1]))
            if len(new_perm_effective) == 0:
                return mapping
            # Conditions for an equi-dataflow match (must hold for each operand independently):
            # 1) all innermost iterated dimensions orthogonal to an operand must be the same, but not necessarily in the same order
            # 2) the innermost non-orthogonal iterated dimension must be the same IFF it is part of a sum of indices and either no orthogonal dimension was iterated before it or multiple reuse types are supported on the level
            # 3) for bypassed operands, only the level that first dictates a dataflow among those the bypass spans over, needs to have an equi-dataflow match, all others match by default
            # NOTE: (3) is currently disabled, and does not help with many matches because the non-bypassed operands prevent it!
            i, j = next((i for i, dim in enumerate(new_perm_effective) if dim in arch.coupling.flat_in_coupling), 0), next((j for j, dim in enumerate(old_perm_effective) if dim in arch.coupling.flat_in_coupling), 0)
            #input_ok = not targets[level_idx].in_bp and not targets[level_idx].bp_stationarity_solved_here['in']
            input_ok = i == j and set(new_perm_effective[:i]) == set(old_perm_effective[:j]) and ((new_perm_effective[i] == old_perm_effective[j] or (not arch.coupling.getDimSum('in', new_perm_effective[i], 2) and not arch.coupling.getDimSum('in', old_perm_effective[j], 2))) or (not targets[current_level].multiple_reuses and i != 0))
            if input_ok:
                i, j = next((i for i, dim in enumerate(new_perm_effective) if dim in arch.coupling.flat_w_coupling), 0), next((j for j, dim in enumerate(old_perm_effective) if dim in arch.coupling.flat_w_coupling), 0)
                #weights_ok = not targets[level_idx].w_bp and not targets[level_idx].bp_stationarity_solved_here['w']
                weights_ok = i == j and set(new_perm_effective[:i]) == set(old_perm_effective[:j]) and ((new_perm_effective[i] == old_perm_effective[j] or (not arch.coupling.getDimSum('w', new_perm_effective[i], 2) and not arch.coupling.getDimSum('w', old_perm_effective[j], 2))) or (not targets[current_level].multiple_reuses and i != 0))
                if weights_ok:
                    i, j = next((i for i, dim in enumerate(new_perm_effective) if dim in arch.coupling.flat_out_coupling), 0), next((j for j, dim in enumerate(old_perm_effective) if dim in arch.coupling.flat_out_coupling), 0)
                    #output_ok = not targets[level_idx].out_bp and not targets[level_idx].bp_stationarity_solved_here['out']
                    output_ok = i == j and set(new_perm_effective[:i]) == set(old_perm_effective[:j]) and ((new_perm_effective[i] == old_perm_effective[j] or (not arch.coupling.getDimSum('out', new_perm_effective[i], 2) and not arch.coupling.getDimSum('out', old_perm_effective[j], 2))) or (not targets[current_level].multiple_reuses and i != 0))
                    if output_ok:
                        return mapping
        return None
    
    """
    Step forward the exploration by:
    - permuting the present level if it still has some permutations to explore
    - moving to the next inner level when the present one has been fully explored
    - backtrack to the outermost level when a never-before-explored inner level has been fully explored
    Returns True if the exploration is to be continued, False when is to be ended.
    """
    def nextPermutations() -> bool:
        nonlocal current_level, innermost_level_to_explore, current_perms, current_best_perm
        if current_perms[current_level] + 1 < perms_ranges[current_level][1]:
            current_perms[current_level] += 1
        else:
            current_perms[current_level] = current_best_perm[1]
            if current_level == innermost_level_to_explore:
                current_level = 0
                innermost_level_to_explore += 1
                if innermost_level_to_explore == len(targets):
                    return False
            else:
                current_level += 1
            current_perms[current_level] = perms_ranges[current_level][0]
            current_best_perm = (0, current_perms[current_level])
            if isinstance(targets[current_level], MemLevel):
                with lock:
                    key = getKey()
                    if key not in past_perms:
                        past_perms[key] = ThreadSafeHeap()
                    else:
                        past_perms[key].increaseCounter(-(perms_ranges[current_level][1] - perms_ranges[current_level][0]))
        return True
    
    def updateTriedCount(amount : int = 1) -> None:
        nonlocal tried_perms
        tried_perms += amount
        if verbose and math.floor((tried_perms/total_perms)*10) > math.floor(((tried_perms - amount)/total_perms)*10):
            if thread_idx == -1: print(f"Progress: {tried_perms}/{total_perms} tried...")
            else: print(f"Progress in thread {thread_idx}: {tried_perms}/{total_perms} tried...")
    
    continue_exploring = True
    
    while continue_exploring and not Settings.forced_termination_flag:
        equidataflow_past_solution = equiDataflowMatch()
        
        if not equidataflow_past_solution:
            # prepare present permutation and optimize its mapping
            arch.resetFactors(copy = True)
            for mem_idx in range(len(targets)):
                targets[mem_idx].dataflow = permutations[mem_idx][current_perms[mem_idx]]
            arch, wart, moves_count = factorFlow(arch, comp, bias_read)
        elif not Settings.HARD_PERM_SKIP:
            # copy over the mapping and quickly re-optimize it
            if not arch.initialized:
                arch.initFactors(comp)
            arch.importMapping(deepcopy(equidataflow_past_solution))
            targets[current_level].dataflow = permutations[current_level][current_perms[current_level]]
            arch, wart, moves_count = factorFlow(arch, comp, bias_read)
        
        if wart > current_best_perm[0]:
            current_best_perm = (wart, current_perms[current_level])
            if current_level == len(targets) - 1:
                final_mapping = (wart, arch.exportMapping(copy = True))
        
        key = getKey()
        # if all permutation have been explored ('counter_threshold' reached), delete the past solutions entry for the current state (save time by doing so before adding the present entry)
        # IDEA: instead of each thread storing the 'current_best_perm', they could fetch it from the top of the heap in the current past_perms entry!
        counter_threshold = perms_ranges[current_level][1] - perms_ranges[current_level][0] - 1
        if isinstance(targets[current_level], MemLevel):
            if not equidataflow_past_solution:
                # store past solutions (in order of Wart)
                with past_perms[key].lock:
                    if past_perms[key].counter == counter_threshold:
                        del past_perms[key]
                    else:
                        past_perms[key].push(wart, arch.exportMapping(copy = True))
            else:
                # store past solutions iff there was at least one move (in order of Wart)
                with past_perms[key].lock:
                    if past_perms[key].counter == counter_threshold:
                        del past_perms[key]
                    else:
                        if moves_count > 0:
                            past_perms[key].push(wart, arch.exportMapping(copy = True))
                        else:
                            past_perms[key].increaseCounter()
                eqmatched_perms += 1
        updateTriedCount()
        
        continue_exploring = nextPermutations()
    
    past_perms[()].push(final_mapping[0], final_mapping[1])
    
    if verbose and thread_idx != -1: print(f"Terminating thread {thread_idx} with Wart {final_mapping[0]:.3e}, eq-matched perms: {eqmatched_perms}/{tried_perms}.")
    #with past_perms[()].lock:
    #    print(f"Thread {thread_idx}: ", [(k, len(p), p.counter) for k, p in past_perms.items()])
    
    if thread_idx == -1:
        known_wart, mapping = past_perms[()].peek()
        arch.importMapping(mapping)
        wart = Wart(arch, comp, bias_read)
        assert known_wart == wart, "There is something wrong in the archival of past permutations or in the import/export of mappings from Arch..."
        return arch, wart