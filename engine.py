from functools import reduce
from copy import deepcopy
import threading
import math
import time

from computations import *
from settings import *
from factors import *
from levels import *
from prints import *
from utils import *
from arch import *


"""
Entry point for the analytical model.
Updates the MOPs and Latency data of each level w.r.t. the current mapping.
"""
def updateStats(arch, bias_read):
    # MOPs:
    WMOPs = 0
    temporal_iterations = 1
    spatial_iterations = 1
    last_in_reads, last_w_reads, last_out_reads, last_out_writes = 0, 0, 0, 0
    acc_out_reads_factors = 1
    for i in range(len(arch)):
        level = arch[i]
        if isinstance(level, MemLevel):
            # multiply by spatial_iterations too because memory is replicated spatially
            level_mops = level.MOPs()
            in_reads, w_reads, out_reads, out_writes = map(lambda m : m*temporal_iterations*spatial_iterations, level_mops[:4])
            out_reads_factors = level_mops[4]*acc_out_reads_factors
            if not bias_read and out_reads_factors != 0:
                out_reads = (out_reads*(out_reads_factors - 1))//out_reads_factors
            if 'in' not in level.bypasses:
                in_writes = last_in_reads # reads above are written here
                last_in_reads = in_reads
            else:
                in_writes = 0
            if 'w' not in level.bypasses:
                w_writes = last_w_reads # reads above are written here
                last_w_reads = w_reads
            else:
                w_writes = 0
            if 'out' not in level.bypasses:
                level.setAboveMOPs(last_out_reads, last_out_writes)
                out_writes += last_out_reads # reads above are written here
                last_out_reads = out_reads
                if not Settings.FREE_DRAINS:
                    out_reads += last_out_writes # writes above where read here
                    last_out_writes = out_writes
            else:
                level.setAboveMOPs(0, 0)
            level.setMOPs(in_reads = in_reads, w_reads = w_reads, out_reads = out_reads, in_writes = in_writes, w_writes = w_writes, out_writes = out_writes)
            level.temporal_iterations = temporal_iterations
            #print(f"Wordline padding extra reads: {sum(level.getWordlinesPad(in_reads, w_reads, out_reads))}, extra writes {sum(level.getWordlinesPad(in_writes, w_writes, out_writes))}")
            reads = in_reads + w_reads + out_reads + sum(level.getWordlinesPad(in_reads, w_reads, out_reads))
            writes = in_writes + w_writes + out_writes + sum(level.getWordlinesPad(in_writes, w_writes, +out_writes))
            WMOPs += level.WMOPs(reads, writes)
            temporal_iterations *= level.factors.fullProduct()
            acc_out_reads_factors *= level.factors.dimProduct('K')
        elif isinstance(level, FanoutLevel):
            spatial_iterations *= level.factors.fullProduct()
            # spatial multicast of an operand occurs if the fanout is along a dimension not relative
            # to such operand hence, the operand is read once, but written once per instance
            for dim in level.dataflow:
                iterations = level.factors.dimProduct(dim)
                if dim == 'M':
                    if not level.spatial_multicast_support: # we don't have spatial multicast capabilities, increment retroactively the reads on the above level
                        if i > 0:
                            arch[i-1].in_reads *= iterations
                    last_in_reads *= iterations
                if dim == 'K':
                    if not level.spatial_multicast_support: # we don't have spatial multicast capabilities, increment retroactively the reads on the above level
                        if i > 0: # no need to account for bias_read here, as the first read is skipped by all instances of the fanout
                            arch[i-1].out_reads = (arch[i-1].out_reads - arch[i-1].last_out_writes)*iterations + arch[i-1].last_out_writes
                    last_out_reads *= iterations
                    if not level.spatial_reduction_support: # we don't have spatial reduction capabilities, increment retroactively the writes on the above level
                        if i > 0:
                            arch[i-1].out_writes = (arch[i-1].out_writes - arch[i-1].last_out_reads)*iterations + arch[i-1].last_out_reads
                        pass
                    last_out_writes *= iterations
                if dim == 'N':
                    if not level.spatial_multicast_support: # we don't have spatial multicast capabilities, increment retroactively the reads on the above level
                        if i > 0:
                            arch[i-1].w_reads *= iterations
                    last_w_reads *= iterations
        elif isinstance(level, ComputeLevel):
            # TODO: remove cost of first output accumulate if bias_read is False!
            # => not needed because the cost of the add is << than the multiply!
            level.temporal_iterations = temporal_iterations
            WMOPs += level.computeCost(temporal_iterations*spatial_iterations)
            # compute is meant to be the innermost level
            break

    # Latency:
    max_latency = 0
    cc_per_tile = arch[len(arch)-1].latency()
    # IMPROVEMENT:
    # - data for the operand which is stationary can be drained/filled immediately as all iterations unfold (cc_per_tile * fullProduct)
    # - data for the non-stationary operands can be filled/drained only during the last iteration of the outermost loop (cc_per_tile * product of only the two inner dimensions)
    # - if double buffering, add the outermost dimension to the second product too
    # WARNING:
    # - the stationary operand only ever occupies a PART of its allotted space, so the size constraint shall be relaxed according to the number of instance
    #   of such operand which are scheduled to be kept at the same time
    for i in range(len(arch) - 2, -1, -1):
        level = arch[i]
        previous_fanout_pe_to_pe_warmup = 0
        if isinstance(level, MemLevel):
            scaling = level.instances*level.temporal_iterations
            ideal_bandwidth_read = (level.getRead()/scaling)/(cc_per_tile*level.factors.fullProduct())
            ideal_bandwidth_update = (level.getUpdate()/scaling)/(cc_per_tile*level.factors.fullProduct())
            # TODO: this should get divided per-operand, as depending on the dataflow, an operand may have more or less iterations to be loaded
            # TODO: support double buffering on a per-operand basis
            # NOTE: the current implementation coincides with Timeloop's notion of Buffets, but it is not exact...
            # NOTE: my bandwidth is already a bit more accurate since Timeloop ignores drains...
            #outermost_available_iterations = 1 if level.multiple_buffering == 1 else level.factors.dimProduct(level.dataflow[0])*(level.multiple_buffering - 1)
            #ideal_bandwidth_fill = (level.getFill()/scaling)/(cc_per_tile*level.factors.dimProduct(level.dataflow[1])*level.factors.dimProduct(level.dataflow[2])*outermost_available_iterations)
            #ideal_bandwidth_drain = (level.getDrain()/scaling)/(cc_per_tile*level.factors.dimProduct(level.dataflow[1])*level.factors.dimProduct(level.dataflow[2])*outermost_available_iterations)
            ideal_bandwidth_fill = (level.getFill()/scaling)/(cc_per_tile*level.factors.fullProduct())
            ideal_bandwidth_drain = (level.getDrain()/scaling)/(cc_per_tile*level.factors.fullProduct())
            # bandwidth is statically divided between reads and writes
            # NOTE: warmup cycles cannot be used to compensate for a lack of bandiwidth at regime
            if ideal_bandwidth_read + ideal_bandwidth_drain <= level.read_bandwidth:
                latency_read_drain = cc_per_tile*level.factors.fullProduct() + previous_fanout_pe_to_pe_warmup*cc_per_tile
            else:
                latency_read_drain = ((level.getRead() + level.getDrain())/scaling)*(1/level.read_bandwidth)
            if ideal_bandwidth_fill + ideal_bandwidth_update <= level.write_bandwidth:
                latency_fill_update = cc_per_tile*level.factors.fullProduct() + previous_fanout_pe_to_pe_warmup*cc_per_tile
            else:
                latency_fill_update = ((level.getFill() + level.getUpdate())/scaling)*(1/level.write_bandwidth)
            latency = max(latency_read_drain, latency_fill_update)
            stall_cycles = latency - cc_per_tile*level.factors.fullProduct()
            level.setLatency(latency_read_drain = latency_read_drain*level.temporal_iterations, latency_fill_update = latency_fill_update*level.temporal_iterations, cc_per_tile = cc_per_tile, stall_cycles = stall_cycles*level.temporal_iterations, ideal_bandwidth_read = ideal_bandwidth_read, ideal_bandwidth_update = ideal_bandwidth_update, ideal_bandwidth_fill = ideal_bandwidth_fill, ideal_bandwidth_drain = ideal_bandwidth_drain)
            #Timeloop does this (loosing knowledge of true behaviour): cc_per_tile = cc_per_tile*level.factors.fullProduct()
            previous_fanout_pe_to_pe_warmup = 0
            cc_per_tile = latency
        elif isinstance(level, FanoutLevel) and level.pe_to_pe:
            # pe-to-pe forwarding implies that the SA operates as a PIPELINE, which has overall latency equal to that of an operation but a warmup dependent on the mesh size
            previous_fanout_pe_to_pe_warmup = level.mesh - 1

    # Active instances, leakage, and final latency:
    temporal_iterations = 1
    active_instances = 1
    for i in range(len(arch)):
        level = arch[i]
        if isinstance(level, MemLevel):
            max_latency = max(max_latency, level.getSettedLatency())
            #print(f"Leakage level {level.name}: {level.Leakage(level.getSettedLatency())*active_instances}")
            WMOPs += level.Leakage(level.getSettedLatency())*active_instances
            temporal_iterations *= level.factors.fullProduct()
        elif isinstance(level, FanoutLevel):
            max_latency = max(max_latency, level.latency()*temporal_iterations)
            if level.power_gating_support:
                active_instances *= level.factors.fullProduct()
            else:
                active_instances *= level.mesh
        elif isinstance(level, ComputeLevel):
            max_latency = max(max_latency, level.latency()*temporal_iterations)
            WMOPs += level.Leakage(level.latency())*active_instances
            break

    return WMOPs, max_latency

"""
Weighted Arithmetic Intensity (WART)

It is equivalent to FLOPs/EDPoU, where EDPoU = (Energy * Latency) / Utilization
=> Maximizing the WART minimizes the EDPoU.
"""
def Wart(arch, comp, bias_read):
    FLOPs = comp.FLOPs()
    WMOPs, max_latency = updateStats(arch, bias_read)
    utilization = arch.fanoutsUtilization() if Settings.UTILIZATION_IN_WART else 1
    return (FLOPs/(WMOPs*max_latency))*utilization

"""
Energy-Delay Product [pJ*cc]

If pJ_to_J is True, the returned value is in [J*cc].
"""
def EDP(arch, bias_read, pJ_to_J = False):
    WMOPs, max_latency = updateStats(arch, bias_read)
    return WMOPs*max_latency*(10**-12 if pJ_to_J else 1)

"""
Latency [cc]
"""
def Latency(arch):
    max_latency = 0
    for level in arch:
        if isinstance(level, MemLevel):
            if max_latency <= level.getSettedLatency():
                max_latency = level.getSettedLatency()
        elif isinstance(level, FanoutLevel):
            continue
        elif isinstance(level, ComputeLevel):
            break
    return max_latency

"""
Energy [pJ]

If pJ_to_uJ is True, the returned value is in [uJ].
"""
def Energy(arch, pJ_to_uJ = False):
    WMOPs = 0
    for level in arch:
        if isinstance(level, MemLevel):
            reads = level.in_reads + level.w_reads + level.out_reads
            writes = level.in_writes + level.w_writes + level.out_writes
            WMOPs += level.WMOPs(reads, writes)
        elif isinstance(level, FanoutLevel):
            continue
        elif isinstance(level, ComputeLevel):
            WMOPs += level.computeCost(level.temporal_iterations*level.instances)
            break
    return WMOPs*(10**-6 if pJ_to_uJ else 1)

"""
Total read and write Memory Operations (MOPs)
"""
def MOPs(arch):
    tot_reads, tot_writes = 0, 0
    for level in arch:
        if isinstance(level, MemLevel):
            reads = level.in_reads + level.w_reads + level.out_reads
            writes = level.in_writes + level.w_writes + level.out_writes
            tot_reads += reads
            tot_writes += writes
        elif isinstance(level, FanoutLevel):
            continue
        elif isinstance(level, ComputeLevel):
            break
    return tot_reads, tot_writes

"""
Mapper Step 2: allocate to fanout levels the maximum number of iterations
               which can fit on their instances.
"""
def fanoutMaximization(arch, comp, bias_read, verbose = False):
    # TECHNIQUE: Find the prime factors of the mesh, and pick the largest common ones with the dimension
    # mapped along that mesh, continue picking from the largest ones in common until you run out!
    # NOTE: When making this handle Fanouts with multiple unrolled dimensions, the issue is dividing the fanout size across dimensions
    # IDEA: Use a binary (ternary) tree expansion, start with 50-50 (30-30-30), explore each child and prune worse branches?
    # IMPORTANT: from optimizeDataflow, if there are unconstrained dimensions, those are always the first ones!
    # TODO: Evaluate whether to apply this step to ComputeLevels too (see fanoutMaximization TODOs as well)!
    if verbose: print("\nStarting fanout maximization:\n")
    if Settings.ONLY_MAXIMIZE_ONE_FANOUT_DIM:
        if Settings.PADDED_MAPPINGS:
            for dim in ['M', 'K', 'N']:
                total_mesh = math.prod([level.mesh for level in arch if isinstance(level, FanoutLevel) and level.dataflow[0] == dim])
                mesh_factors = [f for level in arch if isinstance(level, FanoutLevel) and level.dataflow[0] == dim for f in prime_factors_list(level.mesh)]
                dim_size = arch[0].factors.dimProduct(dim)
                dim_factors = arch[0].factors.toList(dim)
                if total_mesh > dim_size:
                    used_factors, padding = smallest_product_greater_than(mesh_factors, dim_size)
                    if padding != math.inf and not all([f in dim_factors for f in used_factors]): # only pad if some different factor achieved higher utilization
                        if Settings.VERBOSE_PADDED_MAPPINGS: print(f"PADDING: enlarged {dim} from {dim_size} to {dim_size + padding}")
                        arch[0].factors[dim] = prime_factors(dim_size + padding)
                        arch[0].factors.resetDimProducts([dim])
                else:
                    if not all([f in dim_factors for f in mesh_factors]): # only pad if you are not already a multiple
                        padded_dim_size = dim_size + total_mesh - dim_size%total_mesh
                        if Settings.VERBOSE_PADDED_MAPPINGS: print(f"PADDING: enlarged {dim} from {dim_size} to {padded_dim_size}")
                        arch[0].factors[dim] = prime_factors(padded_dim_size)
                        arch[0].factors.resetDimProducts([dim])
        
        for i in range(1, len(arch) - 1): # first round: start from common factors
            level = arch[i]
            if isinstance(level, FanoutLevel):
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
            if isinstance(level, FanoutLevel):
                for dim in level.dataflow: # as a last resort, try dimensions beyond the first one
                    if dim in level.factors_constraints:
                        continue
                    if level.factors.fullProduct() < level.mesh:
                        space = level.mesh // level.factors.fullProduct()
                        factors, _ = largest_product_less_than(arch[0].factors.toList(dim), space)
                        for f in factors:
                            if not arch.moveFactor(0, i, dim, f, 1) and verbose:
                                print(f"Fanout maximization failed to fill up the leftover space on level {level.name}, dim {dim} with factor {f} (mesh: {level.mesh}, space: {space})...")
    
    else:
        for i in range(1, len(arch) - 1):
            level = arch[i]
            if isinstance(level, FanoutLevel):
                assert len(level.dims) <= 2, f"Level: {level.name}: CURRENT LIMITATION - at most 2 dimensions on the same fanout, limit dims ({level.dims}) to at most 2 entries when Settings.ONLY_MAXIMIZE_ONE_FANOUT_DIM is False."
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
        
    arch.updateInstances()
    if verbose: print(f"After fanout maximization (Wart: {Wart(arch, comp, bias_read):.3e}):")
    if verbose: printFactors(arch)


"""
Generates/Enumerates all moves producing adjacent mappings to the provided one.
Returned moves may violate constraints, use utils.moveFactor to apply them.

Arguments:
- iterate_amounts: if True, adjacency is extended to the idea of moving
                   any arity of a factor between loops on the same dimension.
- skip_fanouts: if True, fanout levels are not considered for adjacency.
"""
def factorsIterator(arch, iterate_amounts = False, skip_fanouts = False):
    for level_idx in range(len(arch)):
        if skip_fanouts and (isinstance(arch[level_idx], FanoutLevel)):
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
def factorFlow(arch, comp, bias_read, already_initialized = False, verbose = False):
    if verbose: print("-------- factorFlow --------")
    if not already_initialized:
        arch.initFactors(comp)
        arch.enforceFactorsConstraints(Settings.PADDED_MAPPINGS, Settings.VERBOSE_PADDED_MAPPINGS)
    assert not arch.findConstraintsViolation(), "Factor constraints or dataflow violation in the given architecture."
    constraints_check = [level.checkConstraints() for level in arch]
    assert all(constraints_check), f"Ill-posed constraints:\n{arch[constraints_check.index(False)].logConstraintsViolation()}"
    if not already_initialized:
        arch.setupBypasses()
        arch.updateInstances()
    assert isinstance(arch[0], MemLevel), f"The first/outermost level must a MemoryLevel, the provided one is {type(arch[-1])}."
    assert isinstance(arch[-1], ComputeLevel), f"The last/innermost level must a ComputeLevel, the provided one is {type(arch[-1])}."
    assert not any(map(lambda l : isinstance(l, ComputeLevel), arch[:-1])), "No other compute levels admitted beside the last/innermost one."
    assert arch.checkDataflowConstraints(), "Dataflow constraints violated."

    if verbose: print(f"Initial condition (Wart: {Wart(arch, comp, bias_read):.3e}):")
    if verbose: printFactors(arch)

    # never re-visit the same mapping
    already_seen = [arch.hashFromFactors()]

    # maximize fanout dimensions
    if not already_initialized:
        fanoutMaximization(arch, comp, bias_read, verbose)
    else:
        if verbose: print("\nSkipping fanout maximization...\n")
    
    if verbose: print("\nStarting FactorFlow tiling optimization:\n")

    # one-factor-steps greedy optimization
    best_wart = Wart(arch, comp, bias_read)
    
    def exploreOneStep(remaining_steps = 1, target_src_level_idx = None, target_dim = None, target_factor = None, target_dst_level_idx = None):
        choices = {}
        for src_level_idx, dim, factor, amount in factorsIterator(arch, iterate_amounts = Settings.ITERATE_AMOUNTS, skip_fanouts= Settings.FREEZE_SA):
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
                if dst_level_idx != src_level_idx and dim not in arch[dst_level_idx].factors_constraints and dim in arch[dst_level_idx].dataflow and (not Settings.FREEZE_SA or not isinstance(arch[dst_level_idx], FanoutLevel)):
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
            for i in range(multisteps):
                assert arch.moveFactor(best_choice[5*i + 0], best_choice[5*i + 1], best_choice[5*i + 2], best_choice[5*i + 3], best_choice[5*i + 4], skip_src_constraints = Settings.NO_CONSTRAINTS_CHECK_DURING_MULTISTEP and i < multisteps - 1) # best choice is an invalid mapping
            best_wart = choices[best_choice]

    arch.updateInstances()
    updateStats(arch, bias_read)
    if verbose: print(f"Final condition:\nWart: {best_wart}\nEDP: {EDP(arch, bias_read, True):.3e} (J*cycle)")
    if verbose: printFactors(arch)
    return arch, best_wart

"""
Helper class to store past mappings together with their Wart and factors at one.
Used in "optimizeDataflows".
"""
class PastSolution:
    def __init__(self, mapping, wart):
        self.mapping : list[LevelCore] = mapping
        self.wart = wart
        
    def factorsAtOne(self, level_idx):
        return {dim:self.mapping[level_idx].factors.dimProduct(dim) == 1 for dim in ['M', 'K', 'N']}

"""
Helper class for a lock which might not exist.
Used in "optimizeDataflows".
"""
class OptionalLock:
    def __init__(self, lock):
        self.lock = lock

    def __enter__(self):
        if self.lock is not None:
            self.lock.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.lock is not None:
            self.lock.release()

"""
Helper class with all data structures required to explore permutations.
Used in 'optimizeDataflows'.
"""
class ExplorationState():
    def __init__(self, thread_safe = False):
        self.lock = OptionalLock(threading.Lock() if thread_safe else None) # clockwork algorithm and equi-dataflow check lock
        self.barrier = threading.Barrier(Settings.THREADS_COUNT) if thread_safe else None # post-initialization synch barrier
        # clockwork counter (indices constructing the current permutation)
        self.current_perms = []
        # The last memory hierarchy level which had its permutation updated
        self.last_iterated_perm = 0
        # The empty list (tuple([])) key contains the list for the permutations on the first level, each subsequent list is
        # assigned as key the hash of the sublist current_perms[0, level_idx] (after it is converted in a tuple for hashing)!
        # Remember, each list is ordered from best to worse mapping (TODO: manage them as a max-heap)!
        # NOTE: since we check equi-dataflow matches on the last permuted level, which thus follows an inner-levels-first order, and skip entirely inner permutations of an outer permutation
        #       when there is a match, we never end up in a situation where equi-dataflow matches need to be checked on a level other than the last permuted one, as outer ones have always
        #       been already checked, while inner ones already skipped if needed!
        # NOTE: store permutations for FanoutLevels too, since this datastructure is then used to propagate solutions upwards and pick the best one.
        self.past_perms = {}
        # With the same keys as "past_perms", tracks how many threads are working on a certain level of the permutations tree,
        # when its entries reach zero, can be deleted, together with the matching entry in "past_perms".
        # In other words, this is the count of processes working on a certain permutation or permutations of its inner (non specified) levels.
        self.past_perms_process_counts = {}
        # counts explored permutations
        self.tried_perms = 0

"""
Mapper Step 1: exhaustively iterate loop permutations/dataflows.

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
def optimizeDataflows(arch, comp, bias_read, thread_idx = -1, state = None, verbose = False):
    #Here changing settings is fine, we are within a process
    Settings.forcedSettingsUpdate(arch, False)
    
    if verbose and thread_idx <= 0: print("-------- optimizeDataflows --------")
    targets = list(filter(lambda l : isinstance(l, MemLevel) or isinstance(l, ComputeLevel) or (Settings.ONLY_MAXIMIZE_ONE_FANOUT_DIM and isinstance(l, FanoutLevel)), arch))
    permutations = [[perm for perm in interleave(level.dataflow_constraints, [dim for dim in level.dataflow if dim not in level.dataflow_constraints])] if not isinstance(level, FanoutLevel) else [rot + [dim for dim in level.dims if dim in level.factors_constraints] for rot in rotations([dim for dim in level.dims if dim not in level.factors_constraints])] for level in targets]
    total_perms = reduce(lambda tot, perms : tot * len(perms), permutations, 1)
    #print(f"Total permutations: {total_perms}")
    
    if verbose and thread_idx <= 0:
        print(f"Starting MSE:\n")
        print(f"Dataflow permutations to try: {total_perms}")
    
    # clockwork counter to try all permutations at all levels (in lexicographical order)
    if thread_idx <= 0:
        state = state if state else ExplorationState()
        for _ in range(len(targets) - 1):
            state.current_perms.append(0)
        # ensure that the first increments yields a first permutation of all 0s
        state.current_perms.append(-1)
        state.last_iterated_perm = len(targets) - 1
        first_key = tuple(0 for _ in range(len(targets) - 1))
        for i in range(len(first_key) + 1):
            state.past_perms[first_key[:i]] = []
            state.past_perms_process_counts[first_key[:i]] = Settings.THREADS_COUNT if Settings.MULTITHREADED else 1
    with state.lock:
        local_current_perms = state.current_perms.copy()
    local_last_iterated_perm = len(targets) - 1
    #skipped_perms_total = 0
    
    if state.barrier:
        state.barrier.wait()
    
    """
    Checks for an equi-dataflow match with past solutions. Returns the best match found or None if no match was found. Let "i" be the level on which to check for a match.
    """
    def equiDataflowMatch(i):
        if isinstance(targets[i], FanoutLevel):
            return None
        #print(f"559 {thread_idx}:", state.past_perms_process_counts, tuple(state.current_perms[:i]), local_last_iterated_perm)
        for perm in state.past_perms[tuple(state.current_perms[:i])]:
            # True if any set of dimensions which with a cyclic shift can yield the previous permutation from the current one is entirely made of dims with a factor of 1,
            # or true if of all dimensions involved in swaps between neighbouring nested loops to go from the previous permutation to the current one at least all except one have a factor of 1.
            if (any(map(lambda dims : all([perm.factorsAtOne(i)[dim] for dim in dims]), single_cyclic_shift(permutations[i][local_current_perms[i]], permutations[i][local_current_perms[i] - 1])))
                    or (lambda dims : sum([perm.factorsAtOne(i)[dim] for dim in dims]) >= len(dims) - 1)(pairwise_swaps(permutations[i][local_current_perms[i]], permutations[i][local_current_perms[i] - 1]))):
                return perm.mapping
        return None
    
    """
    Step forward the clockwork counter by permuting w.r.t. lexicographical order the next level that has index <= than "i".
    If "reset" is True, it steps forward by iterating the innermost level.
    """
    def nextPermutations():
        nonlocal local_last_iterated_perm
        with state.lock:
            i = state.last_iterated_perm
            while i >= 0:
                if state.current_perms[i] + 1 == len(permutations[i]):
                    state.current_perms[i] = 0
                    i -= 1
                else:
                    state.current_perms[i] += 1
                    break
            
            changed_up_to = 0
            while state.current_perms[changed_up_to] == local_current_perms[changed_up_to]:
                changed_up_to += 1
            changed_up_to = min(local_last_iterated_perm, changed_up_to, i)
            # go up to the last common permutation point with your previous permutation, clearing up past permutations along the way
            for j in range(local_last_iterated_perm, changed_up_to, -1):
                key = tuple(local_current_perms[:j])
                state.past_perms_process_counts[key] -= 1
                if state.past_perms_process_counts[key] == 0 and j > 0:
                    # clear current past solutions buffer and move the best solution upward
                    state.past_perms_process_counts.pop(key)
                    upper_key = tuple(local_current_perms[:j - 1])
                    for k in range(len(state.past_perms[upper_key]) + 1):
                        if k == len(state.past_perms[upper_key]) or state.past_perms[upper_key][k].wart < state.past_perms[key][0].wart:
                            state.past_perms[upper_key].insert(k, state.past_perms[key][0])
                            break
                    state.past_perms.pop(key)
            # go back down to the present permutation, preparing past permutations lists along the way
            for j in range(changed_up_to + 1, i + 1):
                key = tuple(state.current_perms[:j])
                if key in state.past_perms:
                    state.past_perms_process_counts[key] += 1
                else:
                    state.past_perms_process_counts[key] = 1
                    state.past_perms[key] = []
            
            #new_key = tuple(state.current_perms[:i])
            # permuting an inner level, setup new past permutations list
            #if i > local_last_iterated_perm:
            #    if new_key in state.past_perms:
            #        state.past_perms_process_counts[new_key] += 1
            #    else:
            #        state.past_perms_process_counts[new_key] = 1
            #        state.past_perms[new_key] = []
            # decrement count of processes working on a permutation only when permuting an outer level
            #if i < local_last_iterated_perm:
            #    old_key = tuple(local_current_perms[:local_last_iterated_perm])
            #    state.past_perms_process_counts[old_key] -= 1
            #    if new_key not in state.past_perms:
            #        state.past_perms_process_counts[new_key] = 1
            #        state.past_perms[new_key] = []
            #    if state.past_perms_process_counts[old_key] == 0:
                    # clear current past solutions buffer and move the best solution upward
            #        state.past_perms_process_counts.pop(old_key)
            #        for j in range(len(state.past_perms[new_key]) + 1):
            #            if j == len(state.past_perms[new_key]) or state.past_perms[new_key][j].wart < state.past_perms[old_key][0].wart:
            #                state.past_perms[new_key].insert(j, state.past_perms[old_key][0])
            #                break
            #        state.past_perms.pop(old_key)
            # stop when finishing permutations on the outermost level
            if i == -1:
                state.last_iterated_perm = -1
                local_last_iterated_perm = -1
                return None, None
            # if there is an equi-dataflow match, permute the same level again, otherwise re-start from the innermost one
            eq_match = equiDataflowMatch(i) if Settings.PERM_SKIP else None
            if eq_match:
                local_last_iterated_perm = i
                state.last_iterated_perm = i
            else:
                state.last_iterated_perm = len(targets) - 1
                local_last_iterated_perm = len(targets) - 1
                for j in range(i + 1, len(targets)):
                    key = tuple(state.current_perms[:j])
                    if key in state.past_perms:
                        state.past_perms_process_counts[key] += 1
                    else:
                        state.past_perms_process_counts[key] = 1
                        state.past_perms[key] = []
            return [idx for idx in state.current_perms], eq_match
    
    local_current_perms, equidataflow_past_solution = nextPermutations()
    
    while local_last_iterated_perm >= 0:
        if not equidataflow_past_solution:
            # prepare present permutation and optimize its mapping
            arch.resetFactors(copy = True)
            for mem_idx in range(len(targets)):
                targets[mem_idx].dataflow = permutations[mem_idx][state.current_perms[mem_idx]]
            arch, wart = factorFlow(arch, comp, bias_read)
        elif not Settings.HARD_PERM_SKIP:
            # copy over the mapping and quickly re-optimize it
            arch.importMapping(deepcopy(equidataflow_past_solution))
            targets[local_last_iterated_perm].dataflow = permutations[local_last_iterated_perm][state.current_perms[local_last_iterated_perm]]
            arch, wart = factorFlow(arch, comp, bias_read, True)
        
        key = tuple(local_current_perms[:local_last_iterated_perm])
        if not equidataflow_past_solution:
            # store past solutions (in order of Wart)
            past_solution = PastSolution(arch.exportMapping(copy = True), wart)
            with state.lock:
                for j in range(len(state.past_perms[key]) + 1):
                    if j == len(state.past_perms[key]) or state.past_perms[key][j].wart < wart:
                        state.past_perms[key].insert(j, past_solution)
                        break
            state.tried_perms += 1
            local_tried_perms = state.tried_perms
            if verbose and math.floor((local_tried_perms/total_perms)*10) > math.floor(((local_tried_perms - 1)/total_perms)*10):
                print(f"Progress: {local_tried_perms}/{total_perms} tried...")
        else:
            # store past solutions (in order of Wart)
            # TODO: store IFF you moved at least one factor!
            past_solution = PastSolution(arch.exportMapping(copy = True), wart)
            with state.lock:
                for j in range(len(state.past_perms[key]) + 1):
                    if j == len(state.past_perms[key]) or state.past_perms[key][j].wart < wart:
                        state.past_perms[key].insert(j, past_solution)
                        break
            skipped_perms = reduce(lambda tot, perms : tot * len(perms), permutations[local_last_iterated_perm+1:len(permutations)], 1)
            state.tried_perms += skipped_perms
            local_tried_perms = state.tried_perms
            #skipped_perms_total += skipped_perms
            if verbose and math.floor((local_tried_perms/total_perms)*10) > math.floor(((local_tried_perms - skipped_perms)/total_perms)*10):
                print(f"Progress: {local_tried_perms}/{total_perms} tried...")
        
        local_current_perms, equidataflow_past_solution = nextPermutations()
    
    #print(f"SKIPPED (thread {thread_idx}):", skipped_perms_total)
    
    if verbose and thread_idx != -1: print(f"Terminating thread {thread_idx}...")
    #print(f"Terminating thread {thread_idx} with skipped permutations (total): {skipped_perms_total}")
    
    if thread_idx == -1:
        arch.importMapping(state.past_perms[()][0].mapping)
        wart = Wart(arch, comp, bias_read)
        assert wart == state.past_perms[()][0].wart
        return arch, wart

"""
Mapper entry point.
"""
def run_engine(arch, comp, bias_read, verbose = False):
    start_time = time.time()

    # TODO: this is redundant with it being done at the start of "optimizeDataflows". Remove this when you manage to share settings between processes!
    if verbose:
        Settings.forcedSettingsUpdate(arch)

    if Settings.MULTITHREADED:
        state = ExplorationState(thread_safe = True)
        threads = []
        for i in range(Settings.THREADS_COUNT):
            t = threading.Thread(target=optimizeDataflows, args=(deepcopy(arch), deepcopy(comp), bias_read, i, state, Settings.VERBOSE))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        assert () in state.past_perms and len(state.past_perms[()]) > 0, f"All threads failed to return or found no valid mapping, see above logs..."
        arch.importMapping(state.past_perms[()][0].mapping)
        wart = Wart(arch, comp, bias_read)
    else:
        #arch, _ = factorFlow(arch, comp, bias_read, verbose = VERBOSE)
        arch, wart = optimizeDataflows(arch, comp, bias_read, verbose = Settings.VERBOSE)
    
    end_time = time.time() - start_time

    edp = EDP(arch, bias_read, True)
    mops = MOPs(arch)
    energy = Energy(arch, True)
    latency = Latency(arch)
    utilization = arch.fanoutsUtilization()

    if verbose:
        print(f"\nFinished in: {end_time:.3f}s")

        print(f"\nBest mapping found with:\n\tWart: {wart:.3e}\n\tEDP: {edp:.3e} (J*cycle)\n\tEnergy: {energy:.3e} (uJ)\n\tLatency: {latency:.3e} (cc)")
        printFactors(arch)

        print("\nFinal MOPs per memory level:")
        printMOPsNew(arch)
        print("\nFinal Latency per level:")
        printLatencyNew(arch)
    
        if Settings.PADDED_MAPPINGS:
            print("")
            printPadding(arch, comp)

    return edp, mops, energy, latency, utilization, end_time, arch