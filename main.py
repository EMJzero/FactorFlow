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
FREEZE_SA = True
# If True also logs the factors immediately after the initialization.
LOG_INITIAL_CONDITION = False
# If True also logs the factors immediately after the maximization of fanouts.
LOG_SA_MAXIMIZATION = False


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
        elif isinstance(level, FanoutLevel1D) or isinstance(level, FanoutLevel2D):
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
            if ideal_bandwidth_read + ideal_bandwidth_drain <= level.read_bandwidth:
                latency_read_drain = cc_per_tile*level.factors.fullProduct()
            else:
                latency_read_drain = ((level.getRead() + level.getDrain())/scaling)*(1/level.read_bandwidth)
            if ideal_bandwidth_fill + ideal_bandwidth_update <= level.write_bandwidth:
                latency_fill_update = cc_per_tile*level.factors.fullProduct()
            else:
                latency_fill_update = ((level.getFill() + level.getUpdate())/scaling)*(1/level.write_bandwidth)
            latency = max(latency_read_drain, latency_fill_update)
            stall_cycles = latency - cc_per_tile*level.factors.fullProduct()
            level.setLatency(latency_read_drain = latency_read_drain*level.temporal_iterations, latency_fill_update = latency_fill_update*level.temporal_iterations, cc_per_tile = cc_per_tile, stall_cycles = stall_cycles*level.temporal_iterations, ideal_bandwidth_read = ideal_bandwidth_read, ideal_bandwidth_update = ideal_bandwidth_update, ideal_bandwidth_fill = ideal_bandwidth_fill, ideal_bandwidth_drain = ideal_bandwidth_drain)
            #Timeloop does this (loosing knowledge of true behaviour): cc_per_tile = cc_per_tile*level.factors.fullProduct()
            cc_per_tile = latency

    temporal_iterations = 1
    spatial_iterations = 1
    for i in range(len(arch)):
        level = arch[i]
        if isinstance(level, MemLevel):
            max_latency = max(max_latency, level.getSettedLatency())
            temporal_iterations *= level.factors.fullProduct()
        elif isinstance(level, FanoutLevel1D) or isinstance(level, FanoutLevel2D):
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
        if skip_fanouts and (isinstance(arch[level_idx], FanoutLevel1D) or isinstance(arch[level_idx], FanoutLevel2D)):
            continue
        for dim in arch[level_idx].dataflow:
            for factor in list(arch[level_idx].factors[dim].keys()):
                if iterate_amounts:
                    for amount in range(1, arch[level_idx].factors[dim][factor] + 1):
                        yield level_idx, dim, factor, amount
                else:
                    yield level_idx, dim, factor, 1

def factorFlow(arch, comp, bias_read, verbose = False):
    if verbose: print("-------- factorFlow --------")
    initFactors(arch, comp)
    enforceFactorsConstraints(arch)
    assert not findConstraintsViolation(arch) # factor constraints or dataflow violation in the given architecture
    constraints_check = [level.checkConstraints() for level in arch]
    if not all(constraints_check):
        print(f"Constraints violation on level \"{arch[constraints_check.index(False)].name}\"")
        assert False # ill-posed constraints (usually due to a dimension missmatch)
    setupBypasses(arch)
    updateInstances(arch)
    assert isinstance(arch[-1], ComputeLevel) # the last/innermost level must be compute
    assert not any(map(lambda l : isinstance(l, ComputeLevel), arch[:-1])) # no other compute levels admitted beside the last/innermost one
    assert any(map(lambda l : isinstance(l, MemLevel), arch[:-1])) # at least one memory level must be present
    assert checkDataflowConstraints(arch) # dataflow constraints violated

    if LOG_INITIAL_CONDITION and verbose: print(f"Initial condition (Wart: {Wart(arch, comp, bias_read):.3e}):")
    if LOG_INITIAL_CONDITION and verbose: printFactors(arch)
    if LOG_SA_MAXIMIZATION and verbose: print("\nStarting fanout maximization:\n")

    already_seen = []

    # TODO: behaviour is suboptimal when SAs are not fixed, so, before going forward, maximize here SA usage,
    # doing so in the limit imposed by the factors you have...
    # TECHNIQUE: find the prime factors of the mesh, and pick the largest common ones with the dimension
    # mapped along that mesh, continue picking from the largest ones in common until you run out!

    # maximize fanout dimensions
    for i in range(1, len(arch) - 1):
        level = arch[i]
        if isinstance(level, FanoutLevel1D):
            common_mesh_factors = [f for f in primeFactors(level.mesh).keys() if f in arch[0].factors[level.dim]]
            for f in sorted(common_mesh_factors, reverse=True): # try largest factors first
                amount = arch[0].factors[level.dim][f]
                while amount > 0:
                    # lower the amount until you succeed
                    if moveFactor(arch, 0, i, level.dim, f, amount):
                        break
                    amount -= 1
        elif isinstance(level, FanoutLevel2D):
            assert False # FanoutLevel2D not yet supported, use pairs of FanoutLevel1D instead!
    updateInstances(arch)

    if LOG_SA_MAXIMIZATION and verbose: print(f"After fanout maximization (Wart: {Wart(arch, comp, bias_read):.3e}):")
    if LOG_SA_MAXIMIZATION and verbose: printFactors(arch)
    if verbose: print("\nStarting FactorFlow tiling optimization:\n")

    # one-factor-steps greedy optimization
    best_wart = Wart(arch, comp, bias_read)
    while True:
        choices = {}
        for src_level_idx, dim, factor, amount in factorsIterator(arch, iterate_amounts = ITERATE_AMOUNTS, skip_fanouts= FREEZE_SA):
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
                if dim not in arch[dst_level_idx].factors_contraints and dim in arch[dst_level_idx].dataflow:
                    #print(src_level_idx, dst_level_idx, dim, factor)
                    if moveFactor(arch, src_level_idx, dst_level_idx, dim, factor, amount):
                        hsh = hashFromFactors(arch)
                        if hsh not in already_seen:
                            already_seen.append(hsh)
                            choices[(src_level_idx, dst_level_idx, dim, factor, amount)] = Wart(arch, comp, bias_read)
                        assert moveFactor(arch, dst_level_idx, src_level_idx, dim, factor, amount) # something went wrong, unreversible move of a factor

        if len(choices) == 0:
            if verbose: print(f"No valid follow-up configuration, stopping, current Wart: {best_wart:.3e}")
            break
        # >>> GREEDY MOVE <<<
        best_choice = max(choices, key=choices.get)
        # no need for <= since factors flow only in one direction, you cannot oscillate
        if choices[best_choice] < best_wart:
            if verbose: print(f"Stopping with current Wart: {best_wart:.3e}, while best choice is: {choices[best_choice]:.3e}")
            break
        else:
            assert moveFactor(arch, best_choice[0], best_choice[1], best_choice[2], best_choice[3], best_choice[4]) # best choice is an invalid mapping
            best_wart = choices[best_choice]

    updateInstances(arch)
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
        # TODO: remove this deepcopy and just reset factors
        current_arch = copy.deepcopy(arch)
        current_mems = list(filter(lambda l : isinstance(l, MemLevel) or isinstance(l, ComputeLevel), current_arch))
        for mem_idx in range(len(current_mems)):
            current_mems[mem_idx].dataflow = permutations[mem_idx][current_perms[mem_idx]]
        _, wart = factorFlow(current_arch, comp, bias_read)
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
        # NOTE: this is already robust enough to work with the history of permutations
        while i >= 0 and (any(map(lambda dims : all([factors_at_one[i][dim] for dim in dims]), single_cyclic_shift(permutations[i][current_perms[i]], permutations[i][current_perms[i] - 1])))
               or (lambda dims : sum([factors_at_one[i][dim] for dim in dims]) >= len(dims) - 1)(pairwise_swaps(permutations[i][current_perms[i]], permutations[i][current_perms[i] - 1]))):
            #print(f"Skipping permutation {permutations[i][current_perms[i]]} at level {mems[i].name}!")
            
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

arch = arch_gemmini

## MAIN:

if __name__ == "__main__":
    start_time = time.time()

    #factorFlow(arch, comp, bias_read, True)
    arch, _, _ = optimizeDataflows(arch, comp, bias_read, True)
    
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