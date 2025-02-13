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
Update Settings to best target the provided architecture with the present mapper.
"""
def mapperForcedSettingsUpdate(arch : Arch, verbose : bool = True) -> None:
    if len(list(filter(lambda l : isinstance(l, MemLevel), arch))) < 6:
        steps_to_explore = max(4, Settings.STEPS_TO_EXPLORE)
    else:
        steps_to_explore = max(2, Settings.STEPS_TO_EXPLORE)
    if Settings.STEPS_TO_EXPLORE != steps_to_explore and verbose: print(f"INFO: forcefully updating setting STEPS_TO_EXPLORE to {steps_to_explore}")
    Settings.STEPS_TO_EXPLORE = steps_to_explore
    for level in arch:
        if isinstance(level, SpatialLevel) and len(level.dims) >= 2:
            Settings.LOCAL_SEARCH_SPATIAL_LEVELS = True
            if verbose: print(f"INFO: forcefully updating setting LOCAL_SEARCH_SPATIAL_LEVELS to {Settings.LOCAL_SEARCH_SPATIAL_LEVELS}")
            Settings.LIMIT_NEXT_STEP_DST_TO_CURRENT_SRC = True
            if verbose: print(f"INFO: forcefully updating setting LIMIT_NEXT_STEP_DST_TO_CURRENT_SRC to {Settings.LIMIT_NEXT_STEP_DST_TO_CURRENT_SRC}")
            Settings.NO_CONSTRAINTS_CHECK_DURING_MULTISTEP = True
            if verbose: print(f"INFO: forcefully updating setting NO_CONSTRAINTS_CHECK_DURING_MULTISTEP to {Settings.NO_CONSTRAINTS_CHECK_DURING_MULTISTEP}")
            if verbose: print(f"INFO: --> the cause of this is the presence of a Fanout level ({level.name}) with multiple mapped dimensions ({level.dims}). Runtime might increase slightly...")
            break

"""
Pad comp's size to fully exploit the available spatial instances.
"""
def padCompToFanoutLevels(arch : Arch, comp : Shape, verbose : bool = False) -> Shape:
    for dim in arch.coupling.dims:
        total_mesh = math.prod([level.mesh for level in arch if isinstance(level, SpatialLevel) and len(level.dataflow) > 0 and level.dataflow[0] == dim])
        mesh_factors = [f for level in arch if isinstance(level, SpatialLevel) and len(level.dataflow) > 0 and level.dataflow[0] == dim for f in prime_factors_list(level.mesh)]
        dim_size = comp[dim]
        dim_factors = prime_factors_list(dim_size)
        if total_mesh > dim_size:
            used_factors, padding = smallest_product_greater_than(mesh_factors, dim_size)
            if padding != math.inf and not all([f in dim_factors for f in used_factors]): # only pad if some different factor achieved higher utilization
                if verbose: print(f"PADDING: Arch: {arch.name}: enlarged {dim} from {dim_size} to {dim_size + padding}")
                comp[dim] = dim_size + padding
        else:
            if not all([f in dim_factors for f in mesh_factors]): # only pad if you are not already a multiple
                padded_dim_size = dim_size + total_mesh - dim_size%total_mesh
                if verbose: print(f"PADDING: Arch: {arch.name}: enlarged {dim} from {dim_size} to {padded_dim_size}")
                comp[dim] = padded_dim_size
    return comp

"""
Mapper Step 2: allocate to fanout levels the maximum number of iterations
               which can fit on their instances.

NOTE: when LOCAL_SEARCH_SPATIAL_LEVELS is True, this is ditched and
      replaced by an exploration of spatial levels in 'factorFlow'.
"""
def fanoutMaximization(arch : Arch, comp : Shape, bias_read : bool, verbose : bool = False) -> None:
    # TECHNIQUE: Find the prime factors of the mesh, and pick the largest common ones with the dimension
    # mapped along that mesh, continue picking from the largest ones in common until you run out!
    # IMPORTANT: from optimizeDataflow, if there are unconstrained dimensions, those are always the first ones!
    # NOTE: This step applies to ComputeLevels too!
    if verbose: print("\nStarting fanout maximization:\n")
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
def factorsIterator(arch : Arch, iterate_amounts : bool = False, skip_spatial : bool = False, skip_memories : bool = False) -> Iterator[tuple[int, str, int, int]]:
    for level_idx in range(len(arch)):
        if skip_spatial and isinstance(arch[level_idx], SpatialLevel) or skip_memories and isinstance(arch[level_idx], MemLevel):
            continue
        for dim in arch[level_idx].dataflow:
            for factor in list(arch[level_idx].factors[dim].keys()):
                # check constraints on factors to avoid proposing invalid mappings
                if dim not in arch[level_idx].factors_constraints:
                    if iterate_amounts:
                        for amount in range(1, arch[level_idx].factors[dim][factor] + 1 if (key := dim + '>=') not in arch[level_idx].factors_constraints else arch[level_idx].factors_constraints[key]):
                            yield level_idx, dim, factor, amount
                    else:
                        if (key := dim + '>=') in arch[level_idx].factors_constraints and arch[level_idx].factors.dimProduct(dim)//factor < arch[level_idx].factors_constraints[key]:
                            continue
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
        arch.enforceFactorsConstraints(Settings.PADDED_MAPPINGS, verbose)
    assert arch.checkFactorsConstraints() and arch.checkDataflowConstraints(), ("Ill-posed constraints:" if not already_initialized else "Improperly initialized arch:") + f"\n{arch.logConstraintsViolations()}"
    if verbose: print(f"Initial condition (Wart: {Wart(arch, comp, bias_read):.3e}):")
    if verbose: printFactors(arch)
    
    if verbose: print("\nStarting FactorFlow tiling optimization:\n")
    
    # never re-visit the same mapping
    already_seen = {arch.hashFromFactors()}
    # one-factor-steps greedy optimization
    best_wart = Wart(arch, comp, bias_read)
    # track the count of moves performed
    moves_count = 0
    
    # recursive function that explores and returns all mappings up to 'remaining_steps' away from the present one
    # NOTE: 'freeze_memories' only prevents memories from being destinations (receiving factors)
    # NOTE: 'freeze_spatials' prevents spatial levels from being both sources and destinations (neither giving nor receiving factors)
    def exploreOneStep(remaining_steps : int = 1, target_dst_level_idx : Optional[int] = None, freeze_memories : bool = False, freeze_spatials : bool = False, only_flow_inward : bool = True) -> dict[tuple[Union[int, str]], float]:
        choices = {}
        for src_level_idx, dim, factor, amount in factorsIterator(arch, iterate_amounts = Settings.ITERATE_AMOUNTS, skip_spatial = freeze_spatials):
            
            # go back to the first valid source
            #while src_level_idx >= 0 and dim in arch[src_level_idx].factors_constraints:
            #    src_level_idx -= 1
            #if src_level_idx < 0 or factor not in arch[src_level_idx].factors[dim]:
            #    continue
            
            # pick the target level:
            # - only among inner levels under normal circumstances
            # - anywhere after hitting a dead end
            for dst_level_idx in (range(src_level_idx + 1, len(arch)) if only_flow_inward else range(len(arch))):
                if (dst_level_idx == src_level_idx or dim not in arch[dst_level_idx].dataflow or dim in arch[dst_level_idx].factors_constraints or # check constraints on factors to avoid exploring invalid mappings
                    ((key := dim + '<=') in arch[dst_level_idx].factors_constraints and arch[dst_level_idx].factors.dimProduct(dim)*factor > arch[dst_level_idx].factors_constraints[key]) or
                    (freeze_spatials and isinstance(arch[dst_level_idx], SpatialLevel)) or (freeze_memories and isinstance(arch[dst_level_idx], MemLevel)) or (target_dst_level_idx and dst_level_idx != target_dst_level_idx)): # abide to the provided arguments
                    continue
                if arch.moveFactor(src_level_idx, dst_level_idx, dim, factor, amount, skip_src_constraints = Settings.NO_CONSTRAINTS_CHECK_DURING_MULTISTEP and remaining_steps > 1):
                    hsh = arch.hashFromFactors()
                    if hsh not in already_seen:
                        already_seen.add(hsh)
                        if remaining_steps > 1:
                            nested_choices = exploreOneStep(remaining_steps - 1, target_dst_level_idx = src_level_idx if Settings.LIMIT_NEXT_STEP_DST_TO_CURRENT_SRC else None, freeze_memories = freeze_memories, freeze_spatials = freeze_spatials, only_flow_inward = only_flow_inward)
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
    
    def localSearch(initial_steps_to_explore : int = 1, final_steps_to_explore : int = 1, steps_to_explore_increment : int = 2, freeze_memories : bool = False, freeze_spatials : bool = False) -> None:
        nonlocal best_wart, moves_count
        # when failing to find a better mapping, increase the explored hops ('steps_to_explore') until they reach Settings.STEPS_TO_EXPLORE, then terminate if no better mapping is found, otherswise reset the hops to one
        steps_to_explore = initial_steps_to_explore
        only_flow_inward = True
        while True:
            choices = exploreOneStep(remaining_steps = steps_to_explore, freeze_spatials = freeze_spatials, freeze_memories = freeze_memories, only_flow_inward = only_flow_inward)
            # >>> GREEDY MOVE <<<
            best_choice = max(choices, key = choices.get, default = None)
            if not best_choice or choices[best_choice] < best_wart:
                if steps_to_explore < final_steps_to_explore:
                    steps_to_explore *= steps_to_explore_increment
                    if steps_to_explore == final_steps_to_explore:
                        only_flow_inward = False
                    continue
                else:
                    if verbose: print(f"No valid follow-up configuration, stopping, current Wart: {best_wart:.3e}" if len(choices) == 0 else f"Stopping with current Wart: {best_wart:.3e}, while best choice is: {choices[best_choice]:.3e}")
                    break
            else:
                # each individual choice is defined by 5 parameters, chained to another 5 for each nested exploration step
                multisteps = len(best_choice) // 5
                moves_count += multisteps
                for i in range(multisteps):
                    if verbose: print(f"Moving {arch[best_choice[5*i + 0]].name} --{best_choice[5*i + 2]}:{best_choice[5*i + 3]*best_choice[5*i + 4]}--> {arch[ best_choice[5*i + 1]].name}")
                    assert arch.moveFactor(best_choice[5*i + 0], best_choice[5*i + 1], best_choice[5*i + 2], best_choice[5*i + 3], best_choice[5*i + 4], skip_src_constraints = Settings.NO_CONSTRAINTS_CHECK_DURING_MULTISTEP and i < multisteps - 1) # best choice is an invalid mapping
                best_wart = choices[best_choice]
                steps_to_explore = initial_steps_to_explore
                only_flow_inward = True
    
    if not already_initialized:
        localSearch(final_steps_to_explore = Settings.STEPS_TO_EXPLORE, freeze_memories = False, freeze_spatials = True) # keep a single iteration on spatial levels, optimize memory levels
        if Settings.LOCAL_SEARCH_SPATIAL_LEVELS:
            localSearch(final_steps_to_explore = Settings.STEPS_TO_EXPLORE, freeze_memories = True, freeze_spatials = False) # optimize spatial levels, may only remove factors from memory levels
        else:
            fanoutMaximization(arch, comp, bias_read, verbose) # saturate fanout dimensions
        already_seen.clear()
        already_seen.add(arch.hashFromFactors())
    localSearch(final_steps_to_explore = Settings.STEPS_TO_EXPLORE, freeze_memories = False, freeze_spatials = False) # co-optimize memory and spatial levels
    
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
    
    # if enabled, pad the computation to exploit all spatial instances
    if Settings.PADDED_MAPPINGS:
        comp = padCompToFanoutLevels(arch, comp, verbose)
    
    # this approach requires spatial levels to be the first explored/permuted ones
    targets = list(filter(lambda l : not Settings.LOCAL_SEARCH_SPATIAL_LEVELS and isinstance(l, SpatialLevel) and len(l.dataflow) > 1, arch)) + list(filter(lambda l : isinstance(l, MemLevel), arch))
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
    
    # do not explore levels with a single valid permutation
    for i in range(len(permutations) - 1, -1, -1):
        if len(permutations[i]) <= 1:
            del targets[i], permutations[i], perms_ranges[i]
    
    # divide permutations across threads (if multithreading is enabled)
    if thread_idx != -1:
        first_mem_level_idx = next(i for i, t in enumerate(targets) if isinstance(t, MemLevel))
        partitions_per_sp_level, partitions_per_mem_level = [1] * first_mem_level_idx, [1] * (len(permutations) - first_mem_level_idx + 1)
        factors_per_sp_level, factors_per_mem_level = [prime_factors_list(len(perm)) for perm in permutations[:first_mem_level_idx]], [prime_factors_list(len(perm)) for perm in permutations[first_mem_level_idx:]]
        cumulative_product, circular_idx = 1, -1
        # circularly partition each level's permutations by the smaller of their amount's prime factors, until there are more partitions than threads
        # -> handle spatial levels first
        while cumulative_product < threads_count:
            circular_idx = next(((i + circular_idx + 1) % len(factors_per_sp_level) for i, factors in enumerate(factors_per_sp_level[circular_idx + 1:] + factors_per_sp_level[:circular_idx + 1]) if len(factors) > 0), None)
            if circular_idx == None:
                break
            factor = factors_per_sp_level[circular_idx].pop(0)
            partitions_per_sp_level[circular_idx] *= factor
            cumulative_product *= factor
        # -> then memory levels
        circular_idx = -1
        while cumulative_product < threads_count:
            circular_idx = next(((i + circular_idx + 1) % len(factors_per_mem_level) for i, factors in enumerate(factors_per_mem_level[circular_idx + 1:] + factors_per_mem_level[:circular_idx + 1]) if len(factors) > 0), None)
            if circular_idx == None:
                break
            factor = factors_per_mem_level[circular_idx].pop(0)
            partitions_per_mem_level[circular_idx] *= factor
            cumulative_product *= factor
        if circular_idx == None and thread_idx >= cumulative_product:
            # more threads than permutations, terminate extra threads
            if barrier:
                barrier.wait()
            return
        partitions_per_level = partitions_per_sp_level + partitions_per_mem_level
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
    
    total_perms = reduce(lambda tot, range : tot * (range[1] - range[0]), perms_ranges, 1)
    #print(f"Total permutations: {total_perms}")
    
    if verbose and thread_idx <= 0: print(f"Starting MSE:\n")
    if verbose:
        if thread_idx == -1: print(f"Dataflow permutations to try: {total_perms}")
        else: print(f"Dataflow permutations to try in thread {thread_idx}: {total_perms}")
    
    # clockwork counter to try all permutations at all levels (in lexicographical order) (indices constructing the current permutation)
    current_perms = [low_idx for low_idx, _ in perms_ranges]
    # ensure that the first increments yields a first permutation of all 0s
    current_perms[-1] -= 1
    # The last memory hierarchy level which had its permutation updated
    last_iterated_perm = len(current_perms) - 1
    # The empty list (tuple([])) key contains the list for the permutations on the first level, each subsequent list is
    # assigned as key the hash of the sublist current_perms[0, level_idx] (after it is converted in a tuple for hashing)!
    # In other words, keys are like the current_perms[:i] for level 'i' (the current permutation for 'i' must be excluded).
    # Remember, each list is ordered from best to worse mapping (managed as a max-heap)!
    # NOTE: since we check equi-dataflow matches on the last permuted level, which thus follows an inner-levels-first order, and skip entirely inner permutations of an outer permutation
    #       when there is a match, we never end up in a situation where equi-dataflow matches need to be checked on a level other than the last permuted one, as outer ones have always
    #       been already checked, while inner ones already skipped if needed!
    # NOTE: store permutations for FanoutLevels too, since this datastructure is also used to propagate solutions upwards and pick the best one.
    # NOTE: each list also has a counter associated, which is used to determine when all permutations in the list have been added and it can be delete after the best among them is preserved.
    # NOTE: each list also contains a dictionary of permutations (keyed by current_perms[:i+1]) to skip on its level since they have already been determined to be an equi-dataflow match or a
    #       pruned permutation. This prevents two threads that share some of their permutation from exploring twice the same skipped permutation and breaking the above counters.
    #       The dictionary's values are instead the number of times each permutation has been skipped.
    past_perms = past_perms if past_perms else {}
    # counts explored permutations
    tried_perms = 0
    skipped_perms_total = 0
    pruned_perms_total = 0
    
    # TODO: SHARE THEM AMONG THREADS SOMEHOW!
    # Pruning conditions:
    # 1) any dimensions which ends up with 1 iteration after X non-equi-dataflow match explorations can be locked as an outermost (or even innermost) dimension on a level
    # 2) any pair of dimensions which maintains a certain relative order Y times in the best permutation found when going up from a level can be locked in that order
    # NOTE: store a tuple for each ordered pair; NOTE: locking can be realized either with a datastructure of "perm skip rules" shared among threads, or by removing entries from "permutations" in all threads
    # 3) any dimension which maintains a certain absolute position Z times in the best permutation found when going up from a level can be locked in that position
    # NOTE: a good value for X, Y, and Z is 2; NOTE: pruning conditions apply per-level regardless of the above permutations
    tentative_positional_locks = [{} for _ in targets] # here is where we keep the count of the conditions that need to occur to insert a new positional rule
    tentative_order_locks = [{} for _ in targets] # here is where we keep the count of the conditions that need to occur to insert a new ordering rule
    positional_locks = [{} for _ in targets] # list of dictionaries dim:idx
    relative_order_locks = [[] for _ in targets] # list of lists of ordered pairs
    
    # initialize shared data structures
    lock = OptionalLock(lock)
    first_key = tuple(current_perms[:-1])
    with lock:
        for i in range(len(first_key) + 1):
            if first_key[:i] not in past_perms:
                past_perms[first_key[:i]] = ThreadSafeHeap(initial_counter = -1 if i != len(first_key) else 0)
            elif i != len(first_key) and tuple(first_key[:i + 1]) not in past_perms:
                past_perms[first_key[:i]].increaseCounter(-1)
    if barrier:
        barrier.wait()
    
    """
    Returns a dictionary indicating for each dimension if it has only a single iteration on level 'level_idx' of 'mapping'.
    """
    def factorsAtOne(mapping : Union[list[LevelCore], Arch], level_idx : int) -> dict[str, bool]:
        return {dim: mapping[level_idx].factors.dimProduct(dim) == 1 for dim in arch.coupling.dims}
    
    """
    Checks for an equi-dataflow match with past solutions. Returns the best match found or None if no match was found. Let 'level_idx' be the level on which to check for a match.
    """
    def equiDataflowMatch(level_idx : int) -> Optional[list[LevelCore]]:
        if not Settings.PERM_SKIP or level_idx < 0 or isinstance(targets[level_idx], SpatialLevel):
            return None
        new_perm = permutations[level_idx][current_perms[level_idx]]
        for _, mapping in past_perms[tuple(current_perms[:level_idx])]:
            old_perm = mapping[level_idx].dataflow
            new_perm_effective, old_perm_effective = list(filter(lambda dim : not factorsAtOne(mapping, level_idx)[dim], new_perm[::-1])), list(filter(lambda dim : not factorsAtOne(mapping, level_idx)[dim], old_perm[::-1]))
            if len(new_perm_effective) == 0:
                return mapping
            # Conditions for an equi-dataflow match (must hold for each operand independently):
            # 1) all innermost iterated dimensions orthogonal to an operand must be the same, but not necessarily in the same order
            # 2) the innermost non-orthogonal iterated dimension must be the same IFF it is part of a sum of indices and either no orthogonal dimension was iterated before it or multiple reuse types are supported on the level
            # 3) for bypassed operands, only the level that first dictates a dataflow among those the bypass spans over, needs to have an equi-dataflow match, all others match by default
            # NOTE: (3) is currently disabled, and does not help with many matches because the non-bypassed operands prevent it!
            i, j = next((i for i, dim in enumerate(new_perm_effective) if dim in arch.coupling.flat_in_coupling), 0), next((j for j, dim in enumerate(old_perm_effective) if dim in arch.coupling.flat_in_coupling), 0)
            #input_ok = not targets[level_idx].in_bp and not targets[level_idx].bp_stationarity_solved_here['in']
            input_ok = i == j and set(new_perm_effective[:i]) == set(old_perm_effective[:j]) and ((new_perm_effective[i] == old_perm_effective[j] or (not arch.coupling.getDimSum('in', new_perm_effective[i], 2) and not arch.coupling.getDimSum('in', old_perm_effective[j], 2))) or (not targets[level_idx].multiple_reuses and i != 0))
            if input_ok:
                i, j = next((i for i, dim in enumerate(new_perm_effective) if dim in arch.coupling.flat_w_coupling), 0), next((j for j, dim in enumerate(old_perm_effective) if dim in arch.coupling.flat_w_coupling), 0)
                #weights_ok = not targets[level_idx].w_bp and not targets[level_idx].bp_stationarity_solved_here['w']
                weights_ok = i == j and set(new_perm_effective[:i]) == set(old_perm_effective[:j]) and ((new_perm_effective[i] == old_perm_effective[j] or (not arch.coupling.getDimSum('w', new_perm_effective[i], 2) and not arch.coupling.getDimSum('w', old_perm_effective[j], 2))) or (not targets[level_idx].multiple_reuses and i != 0))
                if weights_ok:
                    i, j = next((i for i, dim in enumerate(new_perm_effective) if dim in arch.coupling.flat_out_coupling), 0), next((j for j, dim in enumerate(old_perm_effective) if dim in arch.coupling.flat_out_coupling), 0)
                    #output_ok = not targets[level_idx].out_bp and not targets[level_idx].bp_stationarity_solved_here['out']
                    output_ok = i == j and set(new_perm_effective[:i]) == set(old_perm_effective[:j]) and ((new_perm_effective[i] == old_perm_effective[j] or (not arch.coupling.getDimSum('out', new_perm_effective[i], 2) and not arch.coupling.getDimSum('out', old_perm_effective[j], 2))) or (not targets[level_idx].multiple_reuses and i != 0))
                    if output_ok:
                        return mapping
        return None
    
    """
    Step forward the clockwork counter by permuting w.r.t. lexicographical order the next level that has index <= than "i".
    If "reset" is True, it steps forward by iterating the innermost level.
    """
    def nextPermutations() -> Optional[list[LevelCore]]:
        nonlocal last_iterated_perm, pruned_perms_total
        i = last_iterated_perm
        while i >= 0:
            if current_perms[i] + 1 == perms_ranges[i][1]:
                current_perms[i] = perms_ranges[i][0]
                # clear current past solutions buffer and move the best solution upward
                if i > 0:
                    current_key = tuple(current_perms[:i])
                    try:
                        perm = past_perms[current_key]
                    except KeyError:
                        i -= 1
                        continue
                    with perm:
                        if current_key in past_perms:
                            wart, mapping = perm.peek()
                            outer_perm = past_perms[current_key[:-1]]
                            outer_perm.push(wart, mapping, increase_counter = 0)
                            # exploration complete or to be terminated early since other threads skipped the present permutation
                            # NOTE: for this to work, the partitioning of permutations among threads MUST assign per level a number of threads which strictly divides the number of permutations
                            # TODO: all be it unlikely, a thread might reach this point and NOT execute the push up, only for then another thread skipping this permutation, with the push up's delete and counter increase thus never occurring...this is not too bad, but should still be prevented!
                            #print(f"Thread {thread_idx}: {current_key} - {current_perms[i]} -", perm.getCounter(), len(permutations[i]), outer_perm.getFromDict(current_perms[i - 1], 0)*(perms_ranges[i][1] - perms_ranges[i][0]))
                            if perm.getCounter() == len(permutations[i]) - outer_perm.getFromDict(current_perms[i - 1], 0)*(perms_ranges[i][1] - perms_ranges[i][0]):
                                #print(f"Thread {thread_idx}: {current_key[:-1]} - {current_perms[i - 1]} - increasing counter push up.")
                                # counter behaviour: increase by 2 on "push up" (unless a +1 already happened due to pruning or equi-dataflow matches), decrease by 1 when starting an exploration of a lower level
                                outer_perm.increaseCounter(2 if not outer_perm.isInDict(current_perms[i - 1]) else 1)
                                del past_perms[current_key]
                            # update pruning condition (2) and (3) about repeated relative order and absolute position
                            if Settings.PERM_PRUNING:
                                for pair in ordered_pairs(mapping[i].dataflow):
                                    tentative_order_locks[i][pair] = tentative_order_locks[i][pair] + 1 if pair in tentative_order_locks[i] else 1
                                relative_order_locks[i] = satisfy_relative_order([pair for pair, count in tentative_order_locks[i].items() if count >= Settings.RELATIVE_ORDER_COUNT_BEFORE_LOCK])
                                for idx, dim in enumerate(mapping[i].dataflow):
                                    if (key := (idx, dim)) not in positional_locks[i]:
                                        tentative_positional_locks[i][key] = tentative_positional_locks[i][key] + 1 if key in tentative_positional_locks[i] else 1
                                        if tentative_positional_locks[i][key] == Settings.POSITIONAL_COUNT_BEFORE_LOCK:
                                            actual_idx = min(actual_idx for actual_idx in range(idx, len(arch[last_iterated_perm].dataflow) + 1) if actual_idx not in positional_locks[last_iterated_perm].values())
                                            if actual_idx < len(arch[last_iterated_perm].dataflow):
                                                positional_locks[i][dim] = actual_idx
                                            tentative_positional_locks[i].pop(key)
                i -= 1
            else:
                current_perms[i] += 1
                key = tuple(current_perms[:i])
                with past_perms[key]: # all checks with a lock to ensure the inclusion of equi-dataflow matches and pruned permutation in the past_perms's set before another thread picks them up
                    # permutation already equi-dataflow matched or pruned by another thread, do not repeat
                    if past_perms[key].isInDict(current_perms[i]):
                        past_perms[key].addToDict(current_perms[i], past_perms[key].getFromDict(current_perms[i], 0) + 1)
                        continue
                    # enforce pruning rules
                    if Settings.PERM_PRUNING:
                        # WARNING: the addition of a new pruning rule and the triggering of this case may leave a past_perm list of a lower level incomplete and unable to push upward its results if its permutation is pruned for some of the threads after it was first entered...
                        new_perm = permutations[i][current_perms[i]]
                        prune = any(new_perm[idx] != dim for dim, idx in positional_locks[i].items())
                        if not prune:
                            j, k = 0, 0
                            while j < len(new_perm) and k < len(relative_order_locks[i]):
                                k = k + 1 if new_perm[j] == relative_order_locks[i][k] else k
                                j += 1
                            prune = k != len(relative_order_locks[i])
                        if prune:
                            pruned_perms = reduce(lambda tot, range : tot * (range[1] - range[0]), perms_ranges[i + 1:len(perms_ranges)], 1)
                            pruned_perms_total += pruned_perms
                            past_perms[key].increaseCounter()
                            #print(f"Thread {thread_idx}: {key} - {current_perms[i]} - increasing counter pruning.")
                            past_perms[key].addToDict(current_perms[i], 1)
                            updateTriedCount(pruned_perms)
                            continue
                    # if there is an equi-dataflow match, permute the same level again, otherwise re-start from the innermost one
                    eq_match = equiDataflowMatch(i) if Settings.PERM_SKIP else None
                    if eq_match:
                        past_perms[key].addToDict(current_perms[i], 1)
                    elif i != len(current_perms) - 1 and tuple(current_perms[:i + 1]) not in past_perms:
                        # decrease the counter by 1 to prevent early deletion of the past_perm iff you are the first thread initiating the exploration of such lower level, that is, the past_perm for the level below you does note exist yet (do not do this on the innermost level where there is not "going down")
                        #print(f"Thread {thread_idx}: {key} - {current_perms[i]} - decreasing counter {past_perms[key].getCounter()} -> {past_perms[key].getCounter() - 1}.")
                        past_perms[key].increaseCounter(-1)
                if eq_match:
                    last_iterated_perm = i
                else:
                    last_iterated_perm = len(current_perms) - 1
                    for j in range(i + 1, len(current_perms)):
                        inner_key = tuple(current_perms[:j])
                        with past_perms[tuple(current_perms[:j - 1])].lock:
                            if inner_key not in past_perms:
                                past_perms[inner_key] = ThreadSafeHeap(initial_counter = -1 if j != len(current_perms) - 1 else 0)
                            elif j != len(current_perms) - 1 and tuple(current_perms[:j + 1]) not in past_perms:
                                #print(f"Thread {thread_idx}: {key} - {current_perms[i]} - decreasing counter {past_perms[inner_key].getCounter()} -> {past_perms[inner_key].getCounter() - 1}.")
                                past_perms[inner_key].increaseCounter(-1)
                return eq_match
        last_iterated_perm = i

    """
    Prints the current exploration progress every time that:
    - print_on_doubling = False: a new 10% of permutations has been explored
    - print_on_doubling = True: the number of explored permutations has increased tenfold
    """
    def updateTriedCount(amount : int = 1, print_on_doubling : bool = True) -> None:
        nonlocal tried_perms
        tried_perms += amount
        if (verbose and (not print_on_doubling and math.floor((tried_perms/total_perms)*10) > math.floor(((tried_perms - amount)/total_perms)*10)) or
            (print_on_doubling and math.floor(math.log10(max(1, tried_perms))) > math.floor(math.log10(max(1, tried_perms - amount))))):
            if thread_idx == -1: print(f"Progress: {tried_perms}/{total_perms} tried...")
            else: print(f"Progress in thread {thread_idx}: {tried_perms}/{total_perms} tried...")
    
    equidataflow_past_solution = nextPermutations()
    
    while last_iterated_perm >= 0 and not Settings.forced_termination_flag:
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
            targets[last_iterated_perm].dataflow = permutations[last_iterated_perm][current_perms[last_iterated_perm]]
            arch, wart, moves_count = factorFlow(arch, comp, bias_read)
        
        key = tuple(current_perms[:last_iterated_perm])
        if not equidataflow_past_solution:
            # store past solutions (in order of Wart)
            past_perms[key].push(wart, arch.exportMapping(copy = True))
            #print(f"Thread {thread_idx}: {key} - {current_perms[last_iterated_perm]} - increasing counter std.")
            # update pruning condition (1) for dimensions with always a single iteration
            for dim, at_one in factorsAtOne(arch, last_iterated_perm).items():
                if at_one and Settings.PERM_PRUNING and dim not in positional_locks[last_iterated_perm]:
                    tentative_positional_locks[last_iterated_perm][dim] = tentative_positional_locks[last_iterated_perm][dim] + 1 if dim in tentative_positional_locks[last_iterated_perm] else 1
                    if tentative_positional_locks[last_iterated_perm][dim] == Settings.DIM_AT_1_COUNT_BEFORE_LOCK:
                        positional_locks[last_iterated_perm][dim] = min(idx for idx in range(len(arch[last_iterated_perm].dataflow)) if idx not in positional_locks[last_iterated_perm].values()) # lock always-at-one dimensions in the outermost spot
                        tentative_positional_locks[last_iterated_perm].pop(dim)
            updateTriedCount()
        else:
            # store past solutions (in order of Wart)
            if moves_count > 0:
                past_perms[key].push(wart, arch.exportMapping(copy = True))
            else:
                past_perms[key].increaseCounter()
            #print(f"Thread {thread_idx}: {key} - {current_perms[last_iterated_perm]} - increasing counter eq-match.")
            skipped_perms = reduce(lambda tot, range : tot * (range[1] - range[0]), perms_ranges[last_iterated_perm + 1:len(perms_ranges)], 1)
            skipped_perms_total += skipped_perms
            updateTriedCount(skipped_perms)
        
        equidataflow_past_solution = nextPermutations()
    
    if verbose and thread_idx != -1: print(f"Terminating thread {thread_idx}, eq-matched perms: {skipped_perms_total}, pruned perms: {pruned_perms_total}.")
    #print(f"Thread {thread_idx}: ", [(k, len(p), p.counter) for k, p in past_perms.items()])
    
    if thread_idx == -1:
        known_wart, mapping = past_perms[()].peek()
        arch.importMapping(mapping)
        wart = Wart(arch, comp, bias_read)
        assert known_wart == wart, "There is something wrong in the archival of past permutations or in the import/export of mappings from Arch..."
        return arch, wart