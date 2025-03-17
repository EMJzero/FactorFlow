from copy import deepcopy
import traceback
import math
import time

from typing import Iterator, Union
from queue import Queue, Empty
import threading

from settings import *
from factors import *
from levels import *
from prints import *
from model import *
from utils import *
from arch import *

# TODO: put me in an inner scope!!!
candidate_perms_per_mem_level : list[list[str]] = []

"""
Update Settings to best target the provided architecture with the present mapper.
"""
def mapperForcedSettingsUpdate(arch : Arch, verbose : bool = True) -> None:
    mem_levels = [mem_l for mem_l in arch if isinstance(mem_l, MemLevel)]
    sp_levels = [sp_l for sp_l in arch if isinstance(sp_l, SpatialLevel)]
    # Memories and Co-opt local search steps settings:
    steps_to_explore = max(2, Settings.STEPS_TO_EXPLORE)
    if len(mem_levels) < 6: # small architecture (less than six memories)
        steps_to_explore = max(4, Settings.STEPS_TO_EXPLORE)
    Settings.ITERATE_AMOUNTS = True # -> set this to False to save time, but set this to True to co-optimize mem. and sp. levels effectively!
    Settings.STEPS_TO_EXPLORE = steps_to_explore
    co_opt_steps_to_explore = max(2, Settings.CO_OPT_STEPS_TO_EXPLORE)
    if sum(len(prime_factors_list(sp_l.mesh)) for sp_l in sp_levels) < 12: # not excessively large fanout levels (sum of all primes less than twelve)
        co_opt_steps_to_explore = max(4, Settings.CO_OPT_STEPS_TO_EXPLORE)
    Settings.CO_OPT_STEPS_TO_EXPLORE = co_opt_steps_to_explore
    initial_steps_to_explore = min(max(2, Settings.INITIAL_STEPS_TO_EXPLORE), steps_to_explore)
    Settings.INITIAL_STEPS_TO_EXPLORE = initial_steps_to_explore
    # Local search multistep settings:
    if any(len(sp_l.dims) >= 2 for sp_l in sp_levels): # a spatial fanout supports multiple dimensions
        Settings.LOCAL_SEARCH_SPATIAL_LEVELS = True
        if Settings.STEPS_TO_EXPLORE > 1:
            Settings.LIMIT_NEXT_STEP_DST_TO_CURRENT_SRC = False # -> keep False to better explore memories!
            Settings.CO_OPT_LIMIT_NEXT_STEP_DST_TO_CURRENT_SRC = True # -> set to True to save on execution time!
            Settings.NO_CONSTRAINTS_CHECK_DURING_MULTISTEP = True # -> set to True when x_LIMIT_NEXT_STEP_DST_TO_CURRENT_SRC is True!
        if verbose: print(f"INFO: --> the cause of this is the presence of Fanout levels ({', '.join(sp_l.name for sp_l in sp_levels if len(sp_l.dims) >= 2)}) with multiple mapped dimensions ({', '.join(str(sp_l.dims) for sp_l in sp_levels if len(sp_l.dims) >= 2)}). Runtime might increase slightly...")
    # Spatial levels local search steps settings:
    if Settings.LOCAL_SEARCH_SPATIAL_LEVELS: # handling of spatial fanouts commuted from blind maximization to a preliminary local search
        spatial_steps_to_explore = max(max(len(prime_factors(sp_l.mesh).keys()) for sp_l in sp_levels), Settings.SPATIAL_STEPS_TO_EXPLORE, 4)
        Settings.SPATIAL_STEPS_TO_EXPLORE = spatial_steps_to_explore
        Settings.SPATIAL_LIMIT_NEXT_STEP_DST_TO_CURRENT_SRC = False
        Settings.SPATIAL_ITERATE_AMOUNTS = True

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
    # NOTE: This step applies to ComputeLevels too!
    for i in range(1, len(arch) - 1): # first round: start from common factors
        level = arch[i]
        if isinstance(level, SpatialLevel):
            dim = level.dataflow[0]
            common_mesh_factors = [f for f in prime_factors(level.mesh).keys() if f in arch[0].factors[dim]]
            for f in sorted(common_mesh_factors, reverse=True): # try largest factors first
                amount = arch[0].factors[dim][f]
                while amount > 0:
                    if arch.moveFactor(0, i, dim, f, amount):
                        if verbose: print(f"╶ Moving {arch[0].name} --{dim}:{f**amount}--> {arch[i].name}")
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
                        if not arch.moveFactor(0, i, dim, f, 1):
                            if verbose: print(f"Arch: {arch.name}: fanout maximization failed to fill up the leftover space on level {level.name}, dim {dim} with factor {f} (mesh: {level.mesh}, space: {space})...")
                        else:
                            if verbose: print(f"╶ Moving {arch[0].name} --{dim}:{f}--> {arch[i].name}")


"""
Removes from 'perms' all but one permutation for each set that is in equi-dataflow match on 'level'.
Use 'in/w/out_matters' to specify wheather an operand is or not relevant to determine the equi-dataflow.
"""
def filterEquiDataflowPerms(level : MemLevel, coupling : Coupling, perms : list[list[str]], in_matters : bool = True, w_matters : bool = True, out_matters : bool = True) -> list[list[str]]:
    """
    Returns a dictionary indicating for each dimension if it has only a single iteration on 'level'.
    """
    def dimsAtOne(level : MemLevel) -> dict[str, bool]:
        return {dim: level.factors.dimProduct(dim) == 1 for dim in coupling.dims}
    
    unique_perms = []
    for perm in perms:
        eq_match = False
        for unique_perm in unique_perms:
            perm_effective, unique_perm_effective = list(filter(lambda dim : not dimsAtOne(level)[dim], perm[::-1])), list(filter(lambda dim : not dimsAtOne(level)[dim], unique_perm[::-1]))
            if len(perm_effective) == 0:
                eq_match = True
                break
            # Conditions for an equi-dataflow match (must hold for each operand independently):
            # 1) all innermost iterated dimensions orthogonal to an operand must be the same, but not necessarily in the same order
            # 2) the innermost non-orthogonal iterated dimension must be the same IFF it is part of a sum of indices and either no orthogonal dimension was iterated before it or multiple reuse types are supported on the level
            # 3) for bypassed operands, only the level that first dictates a dataflow among those the bypass spans over, needs to have an equi-dataflow match, all others match by default
            # NOTE: (3) is passively handled by in/w/out_matters, that are given also according to whether the present level solves a bypass dataflow or not.
            i, j = next((i for i, dim in enumerate(perm_effective) if dim in coupling.flat_in_coupling), 0), next((j for j, dim in enumerate(unique_perm_effective) if dim in coupling.flat_in_coupling), 0)
            input_ok = not in_matters or i == j and set(perm_effective[:i]) == set(unique_perm_effective[:j]) and ((perm_effective[i] == unique_perm_effective[j] or (not coupling.getDimSum('in', perm_effective[i], 2) and not coupling.getDimSum('in', unique_perm_effective[j], 2))) or (not level.multiple_reuses and i != 0))
            if input_ok:
                i, j = next((i for i, dim in enumerate(perm_effective) if dim in coupling.flat_w_coupling), 0), next((j for j, dim in enumerate(unique_perm_effective) if dim in coupling.flat_w_coupling), 0)
                weights_ok = not w_matters or i == j and set(perm_effective[:i]) == set(unique_perm_effective[:j]) and ((perm_effective[i] == unique_perm_effective[j] or (not coupling.getDimSum('w', perm_effective[i], 2) and not coupling.getDimSum('w', unique_perm_effective[j], 2))) or (not level.multiple_reuses and i != 0))
                if weights_ok:
                    i, j = next((i for i, dim in enumerate(perm_effective) if dim in coupling.flat_out_coupling), 0), next((j for j, dim in enumerate(unique_perm_effective) if dim in coupling.flat_out_coupling), 0)
                    output_ok = not out_matters or i == j and set(perm_effective[:i]) == set(unique_perm_effective[:j]) and ((perm_effective[i] == unique_perm_effective[j] or (not coupling.getDimSum('out', perm_effective[i], 2) and not coupling.getDimSum('out', unique_perm_effective[j], 2))) or (not level.multiple_reuses and i != 0))
                    if output_ok:
                        eq_match = True
                        break
        if not eq_match:
            unique_perms.append(perm)
    return unique_perms


"""
Select the best permutations to maximize reuse on a certain factors allocation.
Method: iterate all meaningful permutations and pick the best performing one.
"""
def pickBestPermsIteratively(arch : Arch) -> None:
    # NOTE: even without permutations being defined, we can go inside->out from the first level storing an operand after it has been bypassed,
    # and find the first level with a factor on a dimension coupled to that operand, that is the level solving the dataflow for the bypass!
    # Once the level handling the dataflow has been found, the amount of reuse is still ONLY determined by the tile sizes at THAT level and
    # the iterations across which the reuse occurs. HENCE it is only a question of determining which of the three operands whe should consider
    # when computing reuse on a level, then the reuse calculations can be made locally to the level.
    levels_handling_bypass_dataflows = {'in': None, 'w': None, 'out': None} # operand->level_idx, for a bypassed operand, indicates the first level iterating on a dimension coupled to it, that is, the level solving the bypass's dataflow
    mem_levels = list(filter(lambda l : isinstance(l, MemLevel), arch))
    for i in range(len(mem_levels)):
        level = mem_levels[i]
        for operand, next_levels in level.next_levels_with_bypass.items():
            if next_levels != None: # bypass starting after this level, fetch the first followup level with more than one iteration on a dimension coupled to the present operand
                levels_handling_bypass_dataflows[operand] = next((mem_levels.index(l) for l in next_levels if isinstance(l, MemLevel) and any(l.factors.dimProduct(dim) > 1 for dim in arch.coupling.flatCouplingByOperand(operand))), i)

        dims_not_at_one = [dim for dim in arch.coupling.dims if level.factors.dimProduct(dim) > 1]
        if len(dims_not_at_one) <= 1: # no iterations or a single dimension is iterated, permutations don't matter
            continue
        
        in_matters = 'in' not in level.bypasses or levels_handling_bypass_dataflows['in'] == i
        w_matters = 'w' not in level.bypasses or levels_handling_bypass_dataflows['w'] == i
        out_matters = 'out' not in level.bypasses or levels_handling_bypass_dataflows['out'] == i
        
        if len(dims_not_at_one) == 2: # two dimension iterated, pick the best order between them
            dims_at_one = [dim for dim in arch.coupling.dims if level.factors.dimProduct(dim) == 1]
            candidate_perms = [dims_at_one + dims_not_at_one, dims_at_one + dims_not_at_one[::-1]]
        elif len(dims_not_at_one) < 6: # some dimensions not iterated, check equi-dataflow matches
            candidate_perms = candidate_perms_per_mem_level[i]
            # TODO: try to do another round of filter_equivalent_perms here, giving as sets only the couplings of operands that "matter"!
            candidate_perms = filterEquiDataflowPerms(level, arch.coupling, candidate_perms, in_matters, w_matters, out_matters)
        else: # six dimensions iterated, all candidates must be tried
            candidate_perms = candidate_perms_per_mem_level[i]
        
        best_perm, best_mops = None, math.inf
        for perm in candidate_perms:
            level.dataflow = perm
            # TODO: unfair, because the cost of reads and write is not identical...and we are ignoring other mops types...this is not a great proxy for reuse!
            in_reads, w_reads, out_reads, out_writes, _ = level.MOPs(in_matters, w_matters, out_matters, True)
            mops = in_reads + w_reads + out_reads + out_writes
            if mops < best_mops:
                best_perm, best_mops = perm, mops
        level.dataflow = best_perm


"""
Variant of Queue that has a timeout on the join.
"""
class JoinableQueue(Queue):
    """
    Returns True if the join succeeded, False if it timed out.
    """
    def join(self, timeout : Optional[float] = None) -> bool:
        with self.all_tasks_done:
            if self.unfinished_tasks:
                return self.all_tasks_done.wait(timeout)
            return True


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
def factorFlow(arch : Arch, comp : Shape, bias_read : bool, verbose : bool = False) -> tuple[Arch, float, int]:
    if verbose: print("-------- factorFlow --------")
    already_initialized = arch.initialized
    if not already_initialized:
        arch.initFactors(comp)
        arch.enforceFactorsConstraints(Settings.PADDED_MAPPINGS, verbose)
    assert arch.checkFactorsConstraints() and arch.checkDataflowConstraints(), ("Ill-posed constraints:" if not already_initialized else "Improperly initialized arch:") + f"\n{arch.logConstraintsViolations()}"
    if verbose: print(f"Initial condition (Wart: {Wart(arch, comp, bias_read):.3e}):")
    if verbose: printFactors(arch)
    
    if verbose: print("\nStarting FactorFlow tiling optimization:\n")
    
    # never re-visit the same mapping (unless you reach it with fewer moves)
    already_seen = {arch.hashFromFactors(ignore_dataflows = True, return_string = True): 0} # mapping hash -> moves to reach it
    # one-factor-steps greedy optimization
    best_wart = Wart(arch, comp, bias_read)
    # track the count of moves performed
    moves_count = 0
    
    # setup threads and shared data structures
    choices = dict()
    
    if Settings.MULTITHREADED:
        stay_alive = True
        align_threads = True
        queue = JoinableQueue()
        lock = threading.Lock()
        barrier = threading.Barrier(Settings.THREADS_COUNT + 1)
        update_local_arch = [True for _ in range(Settings.THREADS_COUNT)]
        
        def threadWorker(thread_idx : int) -> None:
            nonlocal choices
            local_arch = deepcopy(arch)
            barrier.wait()
            while stay_alive and not Settings.forced_termination_flag:
                if align_threads:
                    time.sleep(Settings.TIMEOUT)
                    continue
                try:
                    args = queue.get(timeout = Settings.TIMEOUT)
                except Empty:
                    #print(f"Thread {thread_idx} idle...")
                    continue
                try:
                    if update_local_arch[thread_idx]:
                        local_arch.transferMapping(arch, True, False)
                        update_local_arch[thread_idx] = False
                    if 'factors_iterator' in args:
                        local_choices = exploreOneStep(arch = local_arch, **args)
                    else:
                        local_choices = exploreOneStepFurther(arch = local_arch, **args)
                    with lock:
                        choices = choices | local_choices
                except Exception:
                    print(f"EXCEPTION IN WORKER THREAD {thread_idx}:", traceback.format_exc())
                finally:
                    queue.task_done()
        
        threads = []
        for i in range(Settings.THREADS_COUNT):
            t = threading.Thread(target=threadWorker,args=(i,))
            t.start()
            threads.append(t)
        barrier.wait()
    else:
        lock = OptionalLock(None)
    
    """
    Recursive function that explores all adjacent mappings to the present one, recurring to explore up to
    'remaining_steps' adjacency hops away and then returning each explored branch/trajectory.
    
    Arguments:
    - arch: reference model to explore.
    - remaining_steps: number of recursions/steps to explore.
    - recursion_depth: number of recursive calls this function effectuated.
    - factors_iterator: iterator for the moves to explore on the outermost recursion, defaults to all moves.
    - target_dst_level_idx: when specified, it enforces the specified level to receive any factors moved during the present recursion.
    - freeze_memories: (only) prevents memories from being destinations (receiving factors), except for the outermost memory.
    - freeze_spatials: prevents spatial levels from being both sources and destinations (neither giving nor receiving factors).
    - freeze_perms: disables exploration of permutations, using the present ones for all model evaluations.
    - only_flow_inward: allows factors to only be moved from an outer level (lower idx) to an inner one.
    - iterate_amounts: explore also moves that involve multiplicities of prime factors >1 (between the same pair of levels).
    - limit_n_dst_to_c_src: forces any next step's source level to be the one that was the destination in the previous step.
    """
    def exploreOneStep(arch : Arch, remaining_steps : int = 1, recursion_depth : int = 1, factors_iterator : Optional[Iterator[tuple[int, str, int, int]]] = None, target_dst_level_idx : Optional[int] = None, freeze_memories : bool = False, freeze_spatials : bool = False, freeze_perms : bool = False, only_flow_inward : bool = False, iterate_amounts : bool = False, limit_n_dst_to_c_src : bool = False) -> dict[tuple[Union[int, str], ...], float]:
        choices = {}
        if not factors_iterator:
            factors_iterator = factorsIterator(arch, iterate_amounts = iterate_amounts, skip_spatial = freeze_spatials)
        for src_level_idx, dim, factor, amount in factors_iterator:
            # pick the target level:
            # - only among inner levels under normal circumstances
            # - anywhere after hitting a dead end
            for dst_level_idx in ((range(src_level_idx + 1, len(arch)) if only_flow_inward else range(len(arch))) if target_dst_level_idx == None else (target_dst_level_idx,)):
                if (Settings.forced_termination_flag or dst_level_idx == src_level_idx or (target_dst_level_idx != None and only_flow_inward and target_dst_level_idx < src_level_idx) or # invalid source-destination pair
                    dim not in arch[dst_level_idx].dataflow or dim in arch[dst_level_idx].factors_constraints or ((key := dim + '<=') in arch[dst_level_idx].factors_constraints and arch[dst_level_idx].factors.dimProduct(dim)*(factor**amount) > arch[dst_level_idx].factors_constraints[key]) or # check constraints on factors to avoid exploring invalid mappings
                    (freeze_spatials and isinstance(arch[dst_level_idx], SpatialLevel)) or (freeze_memories and dst_level_idx > 0 and isinstance(arch[dst_level_idx], MemLevel))): # abide to the provided arguments
                    continue
                # predict the hash and anticipate the 'already_seen' check to save the time required for 'moveFactor'!
                hsh = arch.hashFromFactorsAfterMove(src_level_idx, dst_level_idx, dim, factor, amount, ignore_dataflows = True, return_string = True)
                moves = moves_count + recursion_depth
                if (not_in := hsh not in already_seen) or already_seen[hsh] > moves:
                    with lock:
                        already_seen[hsh] = moves if not_in else min(moves, already_seen[hsh]) # be it valid or invalid, don't try an already seen mapping ever again.
                    if arch.moveFactor(src_level_idx, dst_level_idx, dim, factor, amount, skip_src_constraints = Settings.NO_CONSTRAINTS_CHECK_DURING_MULTISTEP):
                        if not freeze_perms: pickBestPermsIteratively(arch)
                        if remaining_steps > 1:
                            nested_choices = exploreOneStep(arch, remaining_steps - 1, recursion_depth = recursion_depth + 1, target_dst_level_idx = src_level_idx if limit_n_dst_to_c_src else None, freeze_memories = freeze_memories, freeze_spatials = freeze_spatials, freeze_perms = freeze_perms, only_flow_inward = only_flow_inward, iterate_amounts = iterate_amounts, limit_n_dst_to_c_src = limit_n_dst_to_c_src)
                            if len(nested_choices) == 0:
                                choices[(src_level_idx, dst_level_idx, dim, factor, amount)] = Wart(arch, comp, bias_read, utilization_exponent = Settings.UTIL_EXP_IN_SEARCH_SPATIAL_LEVELS if freeze_memories else 1) if not Settings.NO_CONSTRAINTS_CHECK_DURING_MULTISTEP or arch[src_level_idx].checkFactorsConstraints() else -1
                            else:
                                for nested_choice, nested_wart in nested_choices.items():
                                    choices[(src_level_idx, dst_level_idx, dim, factor, amount) + nested_choice] = nested_wart
                        else:
                            choices[(src_level_idx, dst_level_idx, dim, factor, amount)] = Wart(arch, comp, bias_read, utilization_exponent = Settings.UTIL_EXP_IN_SEARCH_SPATIAL_LEVELS if freeze_memories else 1) if not Settings.NO_CONSTRAINTS_CHECK_DURING_MULTISTEP or arch[src_level_idx].checkFactorsConstraints() else -1 # a negative wart will never be picked, but remains as a starting point for calls to 'exploreOneStepFurther'
                        assert arch.moveFactor(dst_level_idx, src_level_idx, dim, factor, amount, skip_dst_constraints = Settings.NO_CONSTRAINTS_CHECK_DURING_MULTISTEP) # something went wrong, unreversible move of a factor    
        return choices
    
    """
    Same as 'exploreOneStep', but restarts from a known set of mappings and explores them one step further.
    Then, for each starting mapping, it greedily keeps only the best mapping reached, if it improved on (or was tied with) the original.
    """
    def exploreOneStepFurther(arch : Arch, choices : dict[tuple[Union[int, str], ...], float], remaining_steps : int = 1, freeze_memories : bool = False, freeze_spatials : bool = False, freeze_perms : bool = False, only_flow_inward : bool = False, iterate_amounts : bool = False, limit_n_dst_to_c_src : bool = False) -> dict[tuple[Union[int, str], ...], float]:
        further_choices = {}
        for choice, wart in choices.items():
            if remaining_steps >= 1:
                multisteps = len(choice) // 5
                for i in range(multisteps):
                    assert arch.moveFactor(choice[5*i + 0], choice[5*i + 1], choice[5*i + 2], choice[5*i + 3], choice[5*i + 4], skip_src_constraints = Settings.NO_CONSTRAINTS_CHECK_DURING_MULTISTEP) # some previous choice was invalid
                
                # >>> GREEDY BREADTH-FIRST: here we explore 'remaining_steps' further steps for each chioce and collaps all those "further" steps the best one for each!
                # NOTE: this means that low values of 'remaining_steps' collaps more often and see less branches.
                nested_choices = exploreOneStep(arch, remaining_steps, recursion_depth = multisteps + 1, target_dst_level_idx = choice[5*(multisteps - 1) + 0] if limit_n_dst_to_c_src else None, freeze_memories = freeze_memories, freeze_spatials = freeze_spatials, freeze_perms = freeze_perms, only_flow_inward = only_flow_inward, iterate_amounts = iterate_amounts, limit_n_dst_to_c_src = limit_n_dst_to_c_src)
                if len(nested_choices) > 0:
                    # >>> GREEDY MOVE <<<
                    best_choice = max(nested_choices, key=nested_choices.get)
                    if nested_choices[best_choice] >= wart:
                        further_choices[choice + best_choice] = nested_choices[best_choice]
            
                for i in range(multisteps - 1, -1, -1):
                    assert arch.moveFactor(choice[5*i + 1], choice[5*i + 0], choice[5*i + 2], choice[5*i + 3], choice[5*i + 4], skip_dst_constraints = Settings.NO_CONSTRAINTS_CHECK_DURING_MULTISTEP) # some previous choice was invalid
        return further_choices
    
    """
    Runs a greedy local search around the present mapping up until a local optimum is found. To save time, only downward factor moves and up
    to 'initial_steps_to_explore' adjacency hops are explored until the search gets stuck, both limitations get then released, with explored
    hops progressively increasing by 'steps_to_explore_increment' up to 'final_steps_to_explore' or until an improving move is found, if any.
    The arguments 'freeze_memories' and 'freeze_spatials' control which levels partake in the exploration, while 'freeze_perms' determines
    whether permutations are explored or kept fixed and 'limit_n_dst_to_c_src' forcefully creates a chain of moves, see 'exploreOneStep'.
    """
    def localSearch(initial_steps_to_explore : int = 1, final_steps_to_explore : int = 1, steps_to_explore_increment : int = 1, freeze_memories : bool = False, freeze_spatials : bool = False, freeze_perms : bool = False, iterate_amounts : bool = False, limit_n_dst_to_c_src : bool = False) -> None:
        nonlocal best_wart, moves_count, choices, align_threads
        # when failing to find a better mapping, increase the explored hops ('steps_to_explore') until they reach Settings.STEPS_TO_EXPLORE, then terminate if no better mapping is found, otherswise reset the hops to one
        steps_to_explore = initial_steps_to_explore
        while not Settings.forced_termination_flag:
            if steps_to_explore == initial_steps_to_explore:
                if Settings.MULTITHREADED:
                    for task in factorsIterator(arch, iterate_amounts = iterate_amounts, skip_spatial = freeze_spatials):
                        src_level_idx, dim, factor, amount = task
                        for target_dst_level_idx in range(len(arch)):
                            if (src_level_idx != target_dst_level_idx and dim in arch[target_dst_level_idx].dataflow and dim not in arch[target_dst_level_idx].factors_constraints and
                                not ((key := dim + '<=') in arch[target_dst_level_idx].factors_constraints and arch[target_dst_level_idx].factors.dimProduct(dim)*(factor**amount) > arch[target_dst_level_idx].factors_constraints[key]) and
                                not (freeze_spatials and isinstance(arch[target_dst_level_idx], SpatialLevel)) and not (freeze_memories and target_dst_level_idx > 0 and isinstance(arch[target_dst_level_idx], MemLevel))):
                                hsh = arch.hashFromFactorsAfterMove(src_level_idx, target_dst_level_idx, dim, factor, amount, ignore_dataflows = True, return_string = True)
                                if hsh not in already_seen or already_seen[hsh] > moves_count + 1:
                                    queue.put({'factors_iterator': (task,), 'target_dst_level_idx': target_dst_level_idx, 'remaining_steps': initial_steps_to_explore, 'freeze_spatials': freeze_spatials, 'freeze_memories': freeze_memories, 'freeze_perms': freeze_perms, 'iterate_amounts': iterate_amounts, 'limit_n_dst_to_c_src': limit_n_dst_to_c_src})
                    align_threads = False
                    all_done = False
                    while not (all_done or Settings.forced_termination_flag):
                        all_done = queue.join(Settings.TIMEOUT)
                    align_threads = True
                else:
                    choices = exploreOneStep(arch, remaining_steps = initial_steps_to_explore, freeze_spatials = freeze_spatials, freeze_memories = freeze_memories, freeze_perms = freeze_perms, iterate_amounts = iterate_amounts, limit_n_dst_to_c_src = limit_n_dst_to_c_src)
            else:
                # >>> GREEDY BREADH-FIST: at this point we retain only the best "further" choice for each choice we already have, as to not grow exponentially the number of choices.
                if Settings.MULTITHREADED:
                    for choice, wart in choices.items():
                        # TODO: idea, bring forward only the best choices...
                        #if wart > best_wart*0.9:
                        queue.put({'choices': {choice: wart}, 'remaining_steps': steps_to_explore_increment, 'freeze_spatials': freeze_spatials, 'freeze_memories': freeze_memories, 'freeze_perms': freeze_perms, 'iterate_amounts': iterate_amounts, 'limit_n_dst_to_c_src': limit_n_dst_to_c_src})
                    choices.clear()
                    align_threads = False
                    all_done = False
                    while not (all_done or Settings.forced_termination_flag):
                        all_done = queue.join(Settings.TIMEOUT)
                    align_threads = True
                else:
                    choices = exploreOneStepFurther(arch, choices, remaining_steps = steps_to_explore_increment, freeze_spatials = freeze_spatials, freeze_memories = freeze_memories, freeze_perms = freeze_perms, iterate_amounts = iterate_amounts, limit_n_dst_to_c_src = limit_n_dst_to_c_src)
            # >>> GREEDY MOVE <<<
            best_choice = max(choices, key = choices.get, default = None)
            if not best_choice or choices[best_choice] < best_wart:
                if best_choice and steps_to_explore < final_steps_to_explore:
                    steps_to_explore += steps_to_explore_increment
                else:
                    if verbose: print(f"No valid follow-up configuration, stopping, current Wart: {best_wart:.3e}" if not best_choice else f"Stopping with current Wart: {best_wart:.3e}, while best choice is: {choices[best_choice]:.3e}")
                    break
            else:
                # each individual choice is defined by 5 parameters, chained to another 5 for each nested exploration step
                multisteps = len(best_choice) // 5
                moves_count += multisteps
                for i in range(multisteps):
                    if verbose: print(f"{'╶' if multisteps == 1 else ('┌' if i == 0 else ('└' if i == multisteps - 1 else '│'))} Moving {arch[best_choice[5*i + 0]].name} --{best_choice[5*i + 2]}:{best_choice[5*i + 3]**best_choice[5*i + 4]}--> {arch[best_choice[5*i + 1]].name}")
                    assert arch.moveFactor(best_choice[5*i + 0], best_choice[5*i + 1], best_choice[5*i + 2], best_choice[5*i + 3], best_choice[5*i + 4], skip_src_constraints = Settings.NO_CONSTRAINTS_CHECK_DURING_MULTISTEP and i < multisteps - 1) # best choice is an invalid mapping
                best_wart = choices[best_choice]
                choices.clear()
                steps_to_explore = initial_steps_to_explore
                if Settings.MULTITHREADED:
                    for i in range(len(update_local_arch)):
                        update_local_arch[i] = True
        if not freeze_perms:
            pickBestPermsIteratively(arch)
    
    if not already_initialized:
        if Settings.LOCAL_SEARCH_SPATIAL_LEVELS:
            # NOTE: use a higher STEPS_TO_EXPLORE and 'iterate_amounts' to prevent disjoint large prime factors to contend with many small ones shared with the spatial level's available mesh.
            # NOTE: do not use "freeze_perms" here, as it makes the comparison unfair and has little overhead anyway, working only on the outermost memory!
            if verbose: print("-- local search of spatial levels --")
            localSearch(initial_steps_to_explore = Settings.SPATIAL_STEPS_TO_EXPLORE, final_steps_to_explore = Settings.SPATIAL_STEPS_TO_EXPLORE, freeze_memories = True, freeze_spatials = False, iterate_amounts = Settings.SPATIAL_ITERATE_AMOUNTS, limit_n_dst_to_c_src = Settings.SPATIAL_LIMIT_NEXT_STEP_DST_TO_CURRENT_SRC) # optimize spatial levels, may only remove factors from memory levels
        else:
            if verbose: print("-- fanout maximization --")
            fanoutMaximization(arch, comp, bias_read, verbose) # saturate fanout dimensions
        if verbose: print("-- local search of memory levels --")
        localSearch(initial_steps_to_explore = Settings.INITIAL_STEPS_TO_EXPLORE, final_steps_to_explore = Settings.STEPS_TO_EXPLORE, steps_to_explore_increment = 2, freeze_memories = False, freeze_spatials = True, iterate_amounts = Settings.ITERATE_AMOUNTS, limit_n_dst_to_c_src = Settings.LIMIT_NEXT_STEP_DST_TO_CURRENT_SRC) # optimize memory levels
        #already_seen.clear()
        #already_seen[arch.hashFromFactors(ignore_dataflows = True, return_string = True)] = moves_count
    if verbose: print("-- spatial-memory levels co-optimization --")
    localSearch(initial_steps_to_explore = Settings.INITIAL_STEPS_TO_EXPLORE, final_steps_to_explore = Settings.CO_OPT_STEPS_TO_EXPLORE, steps_to_explore_increment = 2, freeze_memories = False, freeze_spatials = False, iterate_amounts = Settings.ITERATE_AMOUNTS, limit_n_dst_to_c_src = Settings.CO_OPT_LIMIT_NEXT_STEP_DST_TO_CURRENT_SRC) # co-optimize memory and spatial levels
    
    if Settings.MULTITHREADED:
        stay_alive = False
        for t in threads:
            t.join()
    
    updateStats(arch, bias_read)
    if verbose: print(f"\nFinal condition:\nWart: {best_wart}\nEDP: {EDP(arch, bias_read, True):.3e} (J*cycle)")
    if verbose: printFactors(arch)
    if verbose: print(f"\nVisited {len(already_seen)} mappings.")
    return arch, best_wart, moves_count

"""
Pre-compute the meaningful permutations to explore for each memory level.
Delegate all optimization to 'factorFlow'.
"""
def optimizeDataflows(arch : Arch, comp : Shape, bias_read : bool, thread_idx : int = -1, threads_count : int = 1, past_perms : dict[tuple[int, ...], ThreadSafeHeap[float, list[LevelCore], int, int]] = None, lock : threading.Lock = None, barrier : threading.Barrier = None, verbose : bool = False) -> Optional[tuple[Arch, float]]:
    # if enabled, pad the computation to exploit all spatial instances
    if Settings.PADDED_MAPPINGS:
        comp = padCompToFanoutLevels(arch, comp, verbose)
    
    # consider for each level only permutations introducing a distinct set of reuse opportunities
    candidate_perms_per_mem_level.clear()
    for level in arch:
        if isinstance(level, MemLevel):
            if '_' in level.dataflow_constraints:
                candidate_perms = [perm for perm in slot_in(level.dataflow_constraints, level.dataflow, '_')]
            else:
                candidate_perms = [perm for perm in interleave(level.dataflow_constraints, [dim for dim in level.dataflow if dim not in level.dataflow_constraints])]
            candidate_perms = filter_equivalent_perms(candidate_perms, {frozenset(arch.coupling.flat_in_coupling), frozenset(arch.coupling.flat_w_coupling), frozenset(arch.coupling.flat_out_coupling)})
            candidate_perms_per_mem_level.append(candidate_perms)
    
    arch, wart, moves = factorFlow(arch, comp, bias_read, verbose)
    if verbose: print(f"\nFinished in {moves} moves.")
    if thread_idx == -1:
        return arch, wart
    elif thread_idx == 0:
        past_perms[()].push(wart, arch.exportMapping())