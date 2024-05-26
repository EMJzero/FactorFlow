from itertools import chain, combinations, permutations
from functools import reduce
import math
import copy
import time
import sys

from architectures import *
from factors import *
from levels import *
from prints import *
from utils import *

# SETTINGS:

# If False, FF only searches for better solutions at a one-factor distance from the current one,
# if True, FF searches for solutions at a distance of multiple factors, all be it only arity is
# varied, with the tried factor being just one.
ITERATE_AMOUNTS = False
# If True, factors allocated on fanout levels will not be optimized (as if they were constraints),
# and this is done after factor allocation in the fanouts is maximized.
# NOTE: automatically set to False in case of 2 dimensions on the same fanout.
FREEZE_SA = True
# Number of one-factor steps to try after of which the best choice is picked.
# NOTE: automatically raised to (at least) 2 in case of 2 dimensions on the same fanout.
STEPS_TO_EXPLORE = 1
# If True, any recursively explored step after the first one, will only attempt to move factors
# into the destination level which was the source for the previous move.
# NOTE: automatically set to True in case of 2 dimensions on the same fanout.
LIMIT_FUTURE_STEPS_DST_TO_CURRENT_SRC = False
# If True, any intermediate step of a multi-step exploration will not need to satisfy architectural
# constraints, on the condition that the final step will satisfy them.
# NOTE: can be True iif LIMIT_FUTURE_STEPS_DST_TO_CURRENT_SRC is True
# NOTE: automatically set to True in case of 2 dimensions on the same fanout.
NO_CONSTRAINTS_CHECK_DURING_MULTISTEP = LIMIT_FUTURE_STEPS_DST_TO_CURRENT_SRC and False
# If True, saves time by assuming that any permutation differing from an optimal one by the order
# of dimensions involving one with a single iteration cannot be optimized further, thus skips
# it entirely. Setting it to False can slightly improve the best found mapping.
HARD_PERM_SKIP = False 

def forcedSettingsUpdate(arch):
    global FREEZE_SA, STEPS_TO_EXPLORE, LIMIT_FUTURE_STEPS_DST_TO_CURRENT_SRC, NO_CONSTRAINTS_CHECK_DURING_MULTISTEP
    for level in arch:
        if isinstance(level, FanoutLevel) and len(level.dims) >= 2:
            FREEZE_SA = False
            print(f"INFO: forcefully updating setting FREEZE_SA to {FREEZE_SA}")
            STEPS_TO_EXPLORE = max(2, STEPS_TO_EXPLORE)
            print(f"INFO: forcefully updating setting STEPS_TO_EXPLORE to {STEPS_TO_EXPLORE}")
            LIMIT_FUTURE_STEPS_DST_TO_CURRENT_SRC = True
            print(f"INFO: forcefully updating setting LIMIT_FUTURE_STEPS_DST_TO_CURRENT_SRC to {LIMIT_FUTURE_STEPS_DST_TO_CURRENT_SRC}")
            NO_CONSTRAINTS_CHECK_DURING_MULTISTEP = True
            print(f"INFO: forcefully updating setting NO_CONSTRAINTS_CHECK_DURING_MULTISTEP to {NO_CONSTRAINTS_CHECK_DURING_MULTISTEP}")
            print(f"INFO: --> the cause of this is the presence of a Fanout level ({level.name}) with multiple mapped dimensions({level.dims}). Runtime might increase to a few seconds...")
            print("")
            break

# updates the MOPs and Latency data of each level w.r.t. the current mapping
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
                out_reads += last_out_writes # writes above where read here
                last_out_writes = out_writes
            else:
                level.setAboveMOPs(0, 0)
            level.setMOPs(in_reads = in_reads, w_reads = w_reads, out_reads = out_reads, in_writes = in_writes, w_writes = w_writes, out_writes = out_writes)
            level.temporal_iterations = temporal_iterations
            MOPs = in_reads + w_reads + out_reads + in_writes + w_writes + out_writes
            WMOPs += level.WMOPs(MOPs)
            temporal_iterations *= level.factors.fullProduct()
            acc_out_reads_factors *= level.factors.dimProduct('E')
        elif isinstance(level, FanoutLevel):
            iterations = level.factors.fullProduct()
            spatial_iterations *= iterations
            # spatial multicast of an operand occurs if the fanout is along a dimension not relative
            # to such operand hence, the operand is read once, but written once per instance
            for dim in level.dataflow:
                if dim == 'D':
                    if not level.spatial_multicast_support: # we don't have spatial multicast capabilities, increment retroactively the reads on the above level
                        if i > 0:
                            arch[i-1].in_reads *= iterations
                    last_in_reads *= iterations
                if dim == 'E':
                    if not level.spatial_multicast_support: # we don't have spatial multicast capabilities, increment retroactively the reads on the above level
                        if i > 0: # no need to account for bias_read here, as the first read is skipped by all instances of the fanout
                            arch[i-1].out_reads = (arch[i-1].out_reads - arch[i-1].last_out_writes)*iterations + arch[i-1].last_out_writes
                    last_out_reads *= iterations
                    if not level.spatial_reduction_support: # we don't have spatial reduction capabilities, increment retroactively the writes on the above level
                        if i > 0:
                            arch[i-1].out_writes = (arch[i-1].out_writes - arch[i-1].last_out_reads)*iterations + arch[i-1].last_out_reads
                        pass
                    last_out_writes *= iterations
                if dim == 'L':
                    if not level.spatial_multicast_support: # we don't have spatial multicast capabilities, increment retroactively the reads on the above level
                        if i > 0:
                            arch[i-1].w_reads *= iterations
                    last_w_reads *= iterations
        elif isinstance(level, ComputeLevel):
            # TODO: remove cost of first output accumulate if bias_read is False!
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

    temporal_iterations = 1
    spatial_iterations = 1
    for i in range(len(arch)):
        level = arch[i]
        if isinstance(level, MemLevel):
            max_latency = max(max_latency, level.getSettedLatency())
            temporal_iterations *= level.factors.fullProduct()
        elif isinstance(level, FanoutLevel):
            max_latency = max(max_latency, level.latency()*temporal_iterations)
            spatial_iterations *= level.factors.fullProduct()
        elif isinstance(level, ComputeLevel):
            max_latency = max(max_latency, level.latency()*temporal_iterations)
            break

    return WMOPs, max_latency

# Weighted Arithmetic Intensity
def Wart(arch, comp, bias_read):
    FLOPs = comp.FLOPs()
    WMOPs, max_latency = updateStats(arch, bias_read)
    return FLOPs/(WMOPs*max_latency)

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

def factorFlow(arch, comp, bias_read, already_initialized = False, verbose = False):
    if verbose: print("-------- factorFlow --------")
    if not already_initialized:
        initFactors(arch, comp)
        enforceFactorsConstraints(arch)
    assert not findConstraintsViolation(arch) # factor constraints or dataflow violation in the given architecture
    constraints_check = [level.checkConstraints() for level in arch]
    if not all(constraints_check):
        print(f"Constraints violation on level \"{arch[constraints_check.index(False)].name}\"")
        assert False # ill-posed constraints (usually due to a dimension missmatch)
    if not already_initialized:
        setupBypasses(arch)
        updateInstances(arch)
    assert isinstance(arch[-1], ComputeLevel) # the last/innermost level must be compute
    assert not any(map(lambda l : isinstance(l, ComputeLevel), arch[:-1])) # no other compute levels admitted beside the last/innermost one
    assert any(map(lambda l : isinstance(l, MemLevel), arch[:-1])) # at least one memory level must be present
    assert checkDataflowConstraints(arch) # dataflow constraints violated

    if verbose: print(f"Initial condition (Wart: {Wart(arch, comp, bias_read):.3e}):")
    if verbose: printFactors(arch)

    already_seen = [hashFromFactors(arch)]

    # maximize fanout dimensions
    # TECHNIQUE: Find the prime factors of the mesh, and pick the largest common ones with the dimension
    # mapped along that mesh, continue picking from the largest ones in common until you run out!
    # NOTE: When making this handle Fanouts with multiple unrolled dimensions, the issue is dividing the fanout size across dimensions
    # IDEA: use a binary (ternary) tree expansion, start with 50-50 (30-30-30), explore each child and prune worse branches?
    if not already_initialized:
        if verbose: print("\nStarting fanout maximization:\n")
        for i in range(1, len(arch) - 1):
            level = arch[i]
            if isinstance(level, FanoutLevel):
                # TODO: as of now this accepts only at most 2 dimensions on same fanout - pick mesh_split_factor >= 3 to allow more?
                mesh_prime_factors = primeFactors(level.mesh)
                # use the smallest available factor to split the mesh among dimensions
                mesh_split_factor = min(mesh_prime_factors) if len(level.dims) >= 2 else 1
                for j in range(len(level.dims)):
                    common_mesh_factors = [f for f in mesh_prime_factors.keys() if f in arch[0].factors[level.dims[j]]]
                    for f in sorted(common_mesh_factors, reverse=True): # try largest factors first
                        amount = arch[0].factors[level.dims[j]][f]
                        while amount > 0:
                            # lower the amount until you succeed
                            if moveFactor(arch, 0, i, level.dims[j], f, amount):
                                if level.checkConstraints(mesh_split_factor) or j != 0:
                                    break
                                else:
                                    assert moveFactor(arch, i, 0, level.dims[j], f, amount) # failed to revert move
                            amount -= 1
        updateInstances(arch)
        if verbose: print(f"After fanout maximization (Wart: {Wart(arch, comp, bias_read):.3e}):")
        if verbose: printFactors(arch)
    else:
        if verbose: print("\nSkipping fanout maximization...\n")
    
    if verbose: print("\nStarting FactorFlow tiling optimization:\n")

    # one-factor-steps greedy optimization
    best_wart = Wart(arch, comp, bias_read)
    
    def exploreOneStep(remaining_steps = 1, target_src_level_idx = None, target_dim = None, target_factor = None, target_dst_level_idx = None):
        choices = {}
        for src_level_idx, dim, factor, amount in factorsIterator(arch, iterate_amounts = ITERATE_AMOUNTS, skip_fanouts= FREEZE_SA):
            if target_src_level_idx and target_dim and target_factor and (target_src_level_idx != src_level_idx or target_dim != dim or target_factor != factor):
                continue
            #print("Initial proposal: ", src_level_idx, dim, factor)
            # make this -2, -3, etc. if the next layer is bound by a constraint
            #init_dst_level_idx = src_level_idx + 1
            #dst_level_idx = src_level_idx + 1
            #while dst_level_idx < len(arch) and (dim in arch[dst_level_idx].factors_contraints or dim not in arch[dst_level_idx].dataflow):
            #    dst_level_idx += 1
            #if dst_level_idx >= len(arch):
            #    continue

            # go back to the first valid source
            while src_level_idx >= 0 and dim in arch[src_level_idx].factors_contraints:
                src_level_idx -= 1
            if src_level_idx < 0 or factor not in arch[src_level_idx].factors[dim]:
                continue

            # pick the factor from the highest level that you can
            new_src = src_level_idx - 1
            while new_src >= 0 and factor in arch[new_src].factors[dim]:
                if dim not in arch[new_src].factors_contraints:
                    src_level_idx = new_src
                new_src -= 1

            for dst_level_idx in range(len(arch)):
                if dst_level_idx != src_level_idx and dim not in arch[dst_level_idx].factors_contraints and dim in arch[dst_level_idx].dataflow:
                    #print(src_level_idx, dst_level_idx, dim, factor)
                    if target_dst_level_idx and dst_level_idx != target_dst_level_idx:
                        continue
                    if moveFactor(arch, src_level_idx, dst_level_idx, dim, factor, amount, skip_src_constraints = NO_CONSTRAINTS_CHECK_DURING_MULTISTEP and remaining_steps > 1):
                        hsh = hashFromFactors(arch)
                        if hsh not in already_seen:
                            already_seen.append(hsh)
                            if remaining_steps > 1:
                                nested_choices = exploreOneStep(remaining_steps - 1, target_dst_level_idx = src_level_idx if LIMIT_FUTURE_STEPS_DST_TO_CURRENT_SRC else None)
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
                        assert moveFactor(arch, dst_level_idx, src_level_idx, dim, factor, amount, skip_dst_constraints = NO_CONSTRAINTS_CHECK_DURING_MULTISTEP and remaining_steps > 1) # something went wrong, unreversible move of a factor    
        return choices
    
    while True:
        choices = exploreOneStep(remaining_steps = STEPS_TO_EXPLORE)
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
                assert moveFactor(arch, best_choice[5*i + 0], best_choice[5*i + 1], best_choice[5*i + 2], best_choice[5*i + 3], best_choice[5*i + 4], skip_src_constraints = NO_CONSTRAINTS_CHECK_DURING_MULTISTEP and i < multisteps - 1) # best choice is an invalid mapping
            best_wart = choices[best_choice]

    updateInstances(arch)
    updateStats(arch, bias_read)
    if verbose: print("Final condition:")
    if verbose: printFactors(arch)
    return arch, best_wart

def optimizeDataflows(arch, comp, bias_read, verbose = False):
    if verbose: print("-------- optimizeDataflows --------")
    mems = list(filter(lambda l : isinstance(l, MemLevel) or isinstance(l, ComputeLevel), arch))
    # TODO: pre-exclude some permutations according to the work on derivatives
    # HOW-TO:
    # - prevent from permuting any dimension with iterations/factors constrainted at 1
    # - come up with something from derivatives...
    # - TOP TIER: exploit the fact that you are going in order! If the difference between the
    #   next configuration and the current one involves a dimension which had in the previous
    #   best mapping a factor of 1, SKIP THE CONFIGURATION
    # NOTE: we cannot truly skip the configuration, but we can re-start the factorFlow exploration
    # from where it was left!
    permutations = [[perm for perm in interleave(level.dataflow_constraints, [dim for dim in level.dataflow if dim not in level.dataflow_constraints])] for level in mems]
    total_perms = reduce(lambda tot, perms : tot * len(perms), permutations, 1)

    if verbose: print(f"Starting MSE:\n")
    if verbose: print(f"Dataflow permutations to try: {total_perms}")
    
    # clockwork counter to try all permutations at all levels
    current_perms = [0 for _ in mems]
    # optimization: If the difference between the next permutation and the current one involves solely dimension which had in the previous best mapping a factor of 1, skip the permutation
    # TODO: could be improved further by looking at the whole history of swapped dimensions
    factors_at_one = [{'D': False, 'E': False, 'L': False} for _ in mems]
    best_perm, best_arch, best_wart = current_perms.copy(), None, 0
    tried_perms = 0
    
    def nextPermutations(i):
        while i >= 0:
            if current_perms[i] + 1 == len(permutations[i]):
                current_perms[i] = 0
                for k in factors_at_one[i]:
                    factors_at_one[i][k] = False
                i -= 1
            else:
                current_perms[i] += 1
                for dim in factors_at_one[i].keys():
                    factors_at_one[i][dim] = current_mems[i].factors.dimProduct(dim) == 1
                break
        return i
    
    while True:
        # TODO: remove this deepcopy and just reset factors (requires a "reset" method implemented in each layer)
        #resetTilesAndFactors(arch)
        current_arch = copy.deepcopy(arch)
        current_mems = list(filter(lambda l : isinstance(l, MemLevel) or isinstance(l, ComputeLevel), current_arch))
        for mem_idx in range(len(current_mems)):
            current_mems[mem_idx].dataflow = permutations[mem_idx][current_perms[mem_idx]]
        current_arch, wart = factorFlow(current_arch, comp, bias_read)
        if wart > best_wart:
            best_perm = current_perms.copy()
            best_arch = current_arch
            best_wart = wart
            
        tried_perms += 1
        if verbose and math.floor((tried_perms/total_perms)*10) > math.floor(((tried_perms - 1)/total_perms)*10):
            print(f"Progress: {tried_perms}/{total_perms} tried...")
        # NOTE: "i" is the outermost memory hierarchy level which had its permutation updated right now
        i = nextPermutations(len(mems) - 1)
        
        # True if any set of dimensions which with a cyclic shift can yield the previous permutation from the current one is entirely made of dims with a factor of 1,
        # or true if of all dimensions involved in swaps between neighbouring nested loops to go from the previous permutation to the current one at least all except one have a factor of 1.
        while i >= 0 and (any(map(lambda dims : all([factors_at_one[i][dim] for dim in dims]), single_cyclic_shift(permutations[i][current_perms[i]], permutations[i][current_perms[i] - 1])))
               or (lambda dims : sum([factors_at_one[i][dim] for dim in dims]) >= len(dims) - 1)(pairwise_swaps(permutations[i][current_perms[i]], permutations[i][current_perms[i] - 1]))):
            if not HARD_PERM_SKIP:
                current_mems[i].dataflow = permutations[i][current_perms[i]]
                for j in range(i + 1, len(mems)):
                    current_mems[j].dataflow = permutations[j][best_perm[j]]
                current_arch, wart = factorFlow(current_arch, comp, bias_read, True)
                if wart > best_wart:
                    best_perm = current_perms.copy()
                    best_arch = current_arch
                    best_wart = wart
            
            skipped_perms = reduce(lambda tot, perms : tot * len(perms), permutations[i+1:len(permutations)], 1)
            tried_perms += skipped_perms
            if verbose and math.floor((tried_perms/total_perms)*10) > math.floor(((tried_perms - skipped_perms)/total_perms)*10):
                print(f"Progress: {tried_perms}/{total_perms} tried...")
            i = nextPermutations(i)
            if i < 0:
                break
        if i < 0:
            break
    
    if verbose: print(f"\nBest mapping found with Wart: {best_wart:.3e}")
    if verbose: printFactors(best_arch)
    return best_arch, best_wart, best_perm


# SPECIFICATION:

comp = Shape(
    D = 1024*3, #768*3
    E = 1024, #768
    L = 4096 #1024
    )
bias_read = False # True if bias is not 0 - outputs are read even the first time

arch = arch_eyeriss

## MAIN:

if __name__ == "__main__":
    forcedSettingsUpdate(arch)
    
    start_time = time.time()

    #arch, _ = factorFlow(arch, comp, bias_read, verbose = True)
    arch, _, _ = optimizeDataflows(arch, comp, bias_read, verbose = True)
    
    print(f"\nFinished in: {time.time() - start_time:.3f}s")

    print("\nFinal MOPs per memory level:")
    printMOPsNew(arch)
    print("\nFinal Latency per level:")
    printLatencyNew(arch)

    if len(sys.argv) > 1 and sys.argv[1] == "--gen-tests":
        from test import generateTestMOPs, generateTestLatency
        print("\nGenerated tests:")
        generateTestMOPs(arch)
        generateTestLatency(arch)