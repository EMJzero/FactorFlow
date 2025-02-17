# GOAL: Parse and re-evaluate zigzag-produced mapping with FF's analytical model.

from prettytable import PrettyTable
from copy import deepcopy
from typing import Any
import math
import sys
import os
import re


try:
    from ..architectures.architectures import *
    from ..computations import *
    from ..architectures.solutions_db import *
    from ..computations import *
    from ..settings import *
    from ..factors import *
    from ..engine import *
    from ..levels import *
    from ..prints import *
    from ..model import *
    from ..utils import *
    from ..arch import *
except:
    sys.path.append("..")
    from architectures.architectures import *
    from computations import *
    from architectures.solutions_db import *
    from computations import *
    from settings import *
    from factors import *
    from engine import *
    from levels import *
    from prints import *
    from model import *
    from utils import *
    from arch import *

def read_n_to_last_line(filename : str, n : int = 1) -> str:
    """Returns the nth before last line of a file (n=1 gives last line)"""
    num_newlines = 0
    with open(filename, 'rb') as f:
        try:
            f.seek(-2, os.SEEK_END)    
            while num_newlines < n:
                f.seek(-2, os.SEEK_CUR)
                if f.read(1) == b'\n':
                    num_newlines += 1
        except OSError:
            f.seek(0)
        last_line = f.readline().decode()
    return last_line

def shorten_path(path: str) -> str:
    parts = path.split(os.sep)
    if len(parts) > 2:
        return f"(...){os.sep}{parts[-2]}{os.sep}{parts[-1]}"
    return path

def largest_product_less_than_with_tags(arr : list[tuple[Any, ...]], target : int, tuple_idx_to_compare : int = 0) -> tuple[list[tuple[Any, ...]], int]:
    min_defect = math.inf
    best_subarray = []

    for r in range(1, len(arr) + 1):
        for comb in combinations(arr, r):
            product = math.prod(map(lambda a : a[tuple_idx_to_compare], comb))
            if product <= target and (target - product) < min_defect:
                min_defect = target - product
                best_subarray = comb

    return list(best_subarray), min_defect


if __name__ == "__main__":

    if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]) or not os.path.isdir(sys.argv[1]):
        print(f"The first argument ({sys.argv[1]}) must be a path to a directory containing one or more Timeloop output directories.")
        print("E.g.: say that '~/outputs/test_1' and '~/outputs/test_2' are where Timeloop wrote its outputs, hence you have file like '~/outputs/test_1/loma.txt' or '~/outputs/test_1/salsa.txt', the first argument shall be '~/outputs'. Note also that the middle directory's name, like 'test_1' shall contain an underscore to separate the name of the used architecture and the used computation like 'arch_comp'.")
        sys.exit(1)
    root = sys.argv[1]
    
    archs = {"gemmini": arch_gemmini, "eyeriss": arch_eyeriss, "simba": arch_simba, "tpu": arch_tpu, "gemmini-conv": arch_gemmini_conv, "eyeriss-conv": arch_eyeriss_conv, "simba-conv": arch_simba_conv, "tpu-conv": arch_tpu_conv}
    gemm_comps = comp_BERT_large | comp_maestro_blas
    conv_comps = {"VGG16-" + k: v for k, v in comp_vgg_16.items()} | {"ResNet18-" + k: v for k, v in comp_resnet_18.items()} | benchmark_convs
    conv_transp = benchmark_convs_transposed
    table = PrettyTable(["Mapper", "Arch", "Comp", "EDP[J*cycle]", "MOPs", "Latency[cc]", "Energy[uJ]", "Utilization[/]", "Runtime"])
    for subdir in os.listdir(root):
        subdir = os.path.join(root, subdir)
        if os.path.isdir(subdir):
            arch_comp = os.path.basename(subdir)
            if "_" in arch_comp:
                arch_name, comp_name = arch_comp.split("_", 1)
            else:
                continue
            
            if comp_name not in gemm_comps and comp_name not in conv_comps and comp_name not in conv_transp:
                print(f"Invalid computation ({comp_name}) in:", subdir)
                continue
            if comp_name in gemm_comps:
                coupling = gemm_coupling
                dims_renaming = {'D': 'M', 'E': 'K', 'L': 'N'}
                comp = gemm_comps[comp_name]
                arch_tail = ""
            elif comp_name in conv_comps:
                coupling = conv_coupling_with_stride
                dims_renaming = {'D': 'M', 'E': 'C', 'L': 'P'}
                comp = conv_comps[comp_name]
                arch_tail = "-conv"
            else:
                coupling = transposed_conv_coupling
                dims_renaming = {'D': 'M', 'E': 'C', 'L': 'P'}
                comp = conv_transp[comp_name]
                arch_tail = "-conv"
            
            if not any(k.startswith(arch_name.lower()) for k in archs.keys()):
                print(f"Invalid architecture ({arch_name}) in:", subdir)
                continue
            arch = archs[arch_name + arch_tail]
            arch.resetFactors()
            #level_name_to_idx = {l.name.lower(): i for i, l in enumerate(arch)}
            
            class simpleLevel():
                def __init__(self, name : str, idx : int):
                    self.name = name
                    self.idx = idx
            
            class loopsline():
                def __init__(self, in_level, w_level, out_level):
                    self.in_level : simpleLevel = in_level
                    self.w_level : simpleLevel= w_level
                    self.out_level : simpleLevel= out_level
                    self.dataflow : list[str] = []
                    self.loops : list[tuple[str, int]] = []
            
            for filename in ["loma", "salsa"]:
                
                loopslines = [
                    loopsline(
                        simpleLevel(arch[(idx := next((i for i, l in enumerate(filter(lambda l : isinstance(l, MemLevel), arch)) if 'in' not in l.bypasses), -1))].name.lower(), idx),
                        simpleLevel(arch[(idx := next((i for i, l in enumerate(filter(lambda l : isinstance(l, MemLevel), arch)) if 'w' not in l.bypasses), -1))].name.lower(), idx),
                        simpleLevel(arch[(idx := next((i for i, l in enumerate(filter(lambda l : isinstance(l, MemLevel), arch)) if 'out' not in l.bypasses), -1))].name.lower(), idx)
                    )
                ]
                
                level_names_renaming = {
                    "globb": "GlobalBuffer".lower(),
                    "peinb": "PEInputBuffer".lower(),
                    "peweib": "PEWeightBuffer".lower(),
                    "peaccb": "PEAccuBuffer".lower(),
                    "peweireg": "PEWeightRegs".lower(),
                    "sram_1m": "GlobalBuffer".lower(),
                    "rf_64b_a": "InRegister".lower(),
                    "rf_64b_w": "WRegister".lower(),
                    "rf_16b": "OutRegister".lower(),
                    "iodram": "DRAM".lower(),
                    "wdram": "WeightsDRAM".lower(),
                    "unigb": "UnifiedBuffer".lower(),
                    "wfifo": "WeightsFIFO".lower(),
                    "accum": "Accumulator".lower(),
                    "peweir": "Register".lower(),
                    "glbb": "Scratchpad".lower(),
                    "accum": "Accumulator".lower(),
                    "rf_w": "Register".lower()
                }
                
                filepath = os.path.join(subdir, filename + ".txt")
                
                if not os.path.exists(filepath):
                    print("Can't find file:", filepath)
                    continue
                
                last_line = read_n_to_last_line(filepath)
                runtime = re.search(r'time:\s*([\d.eE+-]+)', last_line)
                if not runtime:
                    print("Could not find the mapper's runtime in:", filepath)
                    continue
                runtime = float(runtime.group(1))
                
                with open(filepath, 'r') as file:
                    operand_to_group = {'I': 3, 'W': 4, 'O': 5}
                    fanouts = []
                    for line in file.readlines():
                        if (match := re.match(r'Temporal\sLoops\s+([IOW])\s+([IOW])\s+([IOW])', line)):
                            operands = [match.group(1), match.group(2), match.group(3)]
                            operand_to_group = {k: operands.index(k) + 3 for k in operand_to_group.keys()}
                        elif (match := re.match(r'\s*for\s([a-zA-Z])\sin\s\[0\,\s(\d+)\)\:\s*(\w+)\s+(\w+)\s+(\w+)', line)):
                            # 1) break each level in 3, each instance storing one operand and bypassing the other two. Make less than 3 if the level by default has some bypass
                            #    => give each instance ~infinite capacity, but the same access costs and divide the bandwidth evenly.
                            # 2) the question then becomes when to switch to the next level. Assume that you start with DRAM for each operand and go down the loops:
                            #    => when encountering a loop with a lower level for an operand, if the loop is NOT orthogonal to the operand and the level does not bypass the operand, switch the current level
                            #       for that operand to the new one.
                            # 3) bundle instances of the same level for different operands together (and also sum back up the available bandwidth) in a single instance when:
                            #    => an instance stores an operand and, the other stores its operand too OR the other's operand is already stored by an inner level (NOT an outer one).
                            #    => for SP fanout levels, place them down as soon as you have the first level inside the SP fanout being used.
                            # 4) so long as no change to the levels occur, accumulate factors on the innermost of the currently in use levels, in case of a tie, target the inputs-holding instance of the level first,
                            #    then the weights-holding one, and lastly the outputs one. When bundling levels, one will in automatic have 0 iterations associated.
                            dim = match.group(1)
                            if dim in dims_renaming:
                                dim = dims_renaming[dim]
                            factor = match.group(2)
                            
                            mem_in, mem_w, mem_out = match.group(operand_to_group['I']).lower(), match.group(operand_to_group['W']).lower(), match.group(operand_to_group['O']).lower()
                            if mem_in in level_names_renaming:
                                mem_in = level_names_renaming[mem_in]
                            if mem_w in level_names_renaming:
                                mem_w = level_names_renaming[mem_w]
                            if mem_out in level_names_renaming:
                                mem_out = level_names_renaming[mem_out]
                            new_in_level_idx, new_w_level_idx, new_out_level_idx = loopslines[-1].in_level.idx, loopslines[-1].w_level.idx, loopslines[-1].out_level.idx
                            new_loopsline_in, new_loopsline_w, new_loopsline_out = False, False, False
                            if dim in coupling.flat_in_coupling and not loopslines[-1].in_level.name.startswith(mem_in):
                                new_in_level_idx = next((i + new_in_level_idx + 1 for i, l in enumerate(arch[new_in_level_idx + 1:]) if l.name.lower() == mem_in and 'in' not in l.bypasses), -1)
                                assert new_in_level_idx != -1, f"Issue in {shorten_path(filepath)}: can't find a level storing inputs with name {mem_in}..."
                                new_loopsline_in = True
                            if dim in coupling.flat_w_coupling and not loopslines[-1].w_level.name.startswith(mem_w):
                                new_w_level_idx = next((i + new_w_level_idx + 1 for i, l in enumerate(arch[new_w_level_idx + 1:]) if l.name.lower() == mem_w and 'w' not in l.bypasses), -1)
                                assert new_w_level_idx != -1, f"Issue in {shorten_path(filepath)}: can't find a level storing inputs with name {mem_w}..."
                                new_loopsline_w = True
                            if dim in coupling.flat_out_coupling and not loopslines[-1].out_level.name.startswith(mem_out):
                                new_out_level_idx = next((i + new_out_level_idx + 1 for i, l in enumerate(arch[new_out_level_idx + 1:]) if l.name.lower() == mem_out and 'out' not in l.bypasses), -1)
                                assert new_out_level_idx != -1, f"Issue in {shorten_path(filepath)}: can't find a level storing inputs with name {mem_out}..."
                                new_loopsline_out = True
                            
                            #print("--------------------------") # REMOVE ME
                            #print(mem_in, mem_w, mem_out, new_loopsline_in, new_loopsline_w, new_loopsline_out, new_in_level_idx, new_w_level_idx, new_out_level_idx)
                            if new_loopsline_in or new_loopsline_w or new_loopsline_out:
                                mem_in_ = mem_in + "(I" + ("W" if mem_in == mem_w else "") + ("O" if mem_in == mem_out else "") + ")"
                                mem_w_ = mem_w + "(" + ("I" if mem_w == mem_in else "") + "W" + ("O" if mem_w == mem_out else "") + ")"
                                mem_out_ = mem_out + "(" + ("I" if mem_out == mem_in else "") + ("W" if mem_out == mem_w else "") + "O)"
                                loopslines.append(loopsline(
                                    simpleLevel(mem_in_ if new_loopsline_in else loopslines[-1].in_level.name, new_in_level_idx),
                                    simpleLevel(mem_w_ if new_loopsline_w else loopslines[-1].w_level.name, new_w_level_idx),
                                    simpleLevel(mem_out_ if new_loopsline_out else loopslines[-1].out_level.name, new_out_level_idx)
                                ))
                            
                            if dim in loopslines[-1].dataflow:
                                loopslines[-1].dataflow.remove(dim)
                            loopslines[-1].dataflow.append(dim)
                            loopslines[-1].loops.append((dim, int(factor)))
                        
                        elif (match := re.match(r'\s*parfor\s([a-zA-Z])\sin\s\[0\,\s(\d+)\)', line)):
                            dim = match.group(1)
                            if dim in dims_renaming:
                                dim = dims_renaming[dim]
                            factor = match.group(2)
                            fanouts.append((dim, int(factor)))
                    
                    new_arch = []
                    factor_moves = []
                    next_spatial = next((i for i, l in enumerate(arch) if isinstance(l, FanoutLevel)), len(arch))
                    spatial_indices = []
                    for ll in loopslines:
                        levels = [ll.in_level, ll.w_level, ll.out_level]
                        levels.sort(key = lambda l : l.idx)
                        for l in levels:
                            if not any(nal.name == l.name for nal in new_arch):
                                assert isinstance(arch[l.idx], MemLevel), f"Issue in {shorten_path(filepath)}: name of a non-MemLevel used in the mapping..."
                                new_arch.append(deepcopy(arch[l.idx]))
                                new_arch[-1].name = l.name
                                new_arch[-1].size *= 3 # to give some room for padding
                                new_arch[-1].bypasses = ['in', 'w', 'out']
                                new_arch[-1].in_bp = 0
                                new_arch[-1].w_bp = 0
                                new_arch[-1].out_bp = 0
                                new_arch[-1]._value_access_energy = None
                                new_arch[-1]._word_bits = 256
                                new_arch[-1]._value_bits = 8
                                new_arch[-1]._wordline_access_energy = None
                                new_arch[-1]._read_wordline_access_energy = None
                                new_arch[-1]._write_wordline_access_energy = None
                                new_arch[-1]._read_value_access_energy = new_arch[-1].read_access_energy
                                new_arch[-1]._write_value_access_energy = new_arch[-1].write_access_energy
                                new_arch[-1]._bandwidth = None
                                new_arch[-1].factors_constraints = {}
                                while l.idx > next_spatial:
                                    spatial_indices.append(len(new_arch) - 1)
                                    next_spatial = next((i + next_spatial + 1 for i, l in enumerate(arch[next_spatial + 1:]) if isinstance(l, FanoutLevel)), len(arch))
                            if l == ll.in_level:
                                if 'in' in new_arch[-1].bypasses:
                                    new_arch[-1].bypasses.remove('in')
                                new_arch[-1].in_bp = 1
                            if l == ll.w_level:
                                if 'w' in new_arch[-1].bypasses:
                                    new_arch[-1].bypasses.remove('w')
                                new_arch[-1].w_bp = 1
                            if l == ll.out_level:
                                if 'out' in new_arch[-1].bypasses:
                                    new_arch[-1].bypasses.remove('out')
                                new_arch[-1].out_bp = 1
                        new_arch[-1].dataflow = [dim for dim in coupling.dims if dim not in ll.dataflow] + ll.dataflow
                        new_arch[-1].size = math.inf
                        new_arch[-1].read_bandwidth *= (3 - len(new_arch[-1].bypasses))/(3 - len(arch[l.idx].bypasses))
                        new_arch[-1].write_bandwidth *= (3 - len(new_arch[-1].bypasses))/(3 - len(arch[l.idx].bypasses))
                        factor_moves.append((new_arch[-1], ll.loops))
                    
                    #print("-------")
                    #for l in new_arch: # REMOVE ME
                    #    print(l.name)
                    
                    for si, sl in list(zip(spatial_indices + [math.inf]*len(arch), list(filter(lambda l : isinstance(l, FanoutLevel), arch))))[::-1]:
                        si = min(si, len(new_arch))
                        new_arch.insert(si, deepcopy(sl))
                        new_arch[si]._dim = None
                        new_arch[si].dims = coupling.dims
                        new_arch[si].dataflow = coupling.dims
                        new_arch[si].factors_constraints = {}

                    new_arch.append(deepcopy(arch[-1]))
                    new_arch[-1]._dim = None
                    
                    padded_comp = {dim : 1 for dim in comp.keys() if not dim.endswith("stride") and not dim.endswith("dilation")}
                    for _, fms in factor_moves:
                        for dim, factor in fms:
                            padded_comp[dim] *= factor
                    for dim, factor in fanouts:
                        padded_comp[dim] *= factor
                    comp_ = deepcopy(comp)
                    for dim in padded_comp.keys():
                        if padded_comp[dim] > comp_[dim]:
                            print(f"Warning in {shorten_path(filepath)}: padding detected on dim {dim}, {comp[dim]}->{padded_comp[dim]}")
                            comp_[dim] = padded_comp[dim]
                    if any(padded_comp[dim] < comp_[dim] for dim in padded_comp.keys()):
                        #print(f"Issue in {shorten_path(filepath)}: not all comp factors are present (comp: {comp}, allocated: {padded_comp})...")
                        #continue
                        for dim in padded_comp.keys():
                            if padded_comp[dim] < comp_[dim]:
                                factor = next((i for i in range(10000) if i*padded_comp[dim] >= comp_[dim]), -1)
                                comp_[dim] = padded_comp[dim]*factor
                                assert factor > 0, "What?! ZigZag left more than 10000 iterations unallocated? This is a bug..."
                                factor_moves.append((new_arch[0], [(dim, factor)]))
                        print(f"Warning in {shorten_path(filepath)}: not all comp factors are present (comp: {comp}, allocated: {padded_comp}, new_comp: {comp_}), assuming that missing ones are on DRAM...")
                    
                    new_arch = Arch(new_arch, coupling)
                    new_arch.initFactors(comp_)
                    total_per_dim = {dim : 1 for dim in comp_.keys() if not dim.endswith("stride") and not dim.endswith("dilation")}
                    for dst, fms in factor_moves:
                        dst = new_arch.index(dst)
                        for dim, factor in fms:
                            for f in prime_factors_list(factor):
                                if not new_arch.moveFactor(0, dst, dim, f):
                                    print(f"Issue in {shorten_path(filepath)}: failed to move factor {f} on dim {dim} from mem. level 0 to mem. level {dst}...")
                                total_per_dim[dim] *= f
                    
                    # AKA: while not all factors were allocated, increase by 1 the remaining ones and RETRY!
                    increased_fanouts = {}
                    while math.prod(factor for _, factor in fanouts) <= math.prod(l.mesh//l.factors.fullProduct() for l in new_arch if isinstance(l, FanoutLevel)) and len(fanouts) > 0:
                        sp_idx = next((i for i, l in enumerate(new_arch) if isinstance(l, FanoutLevel)), -1)
                        #for i in range(len(fanouts) - 1, -1, -1):
                        #    dim, factor = fanouts[i]
                        #    for f in prime_factors_list(factor):
                        #        while sp_idx != -1:
                        #            if new_arch.moveFactor(0, sp_idx, dim, f):
                        #                fanouts[i] = (dim, factor//f)
                        #                total_per_dim[dim] *= f
                        #                break
                        #            sp_idx = next((i + sp_idx + 1 for i, l in enumerate(new_arch[sp_idx + 1:]) if isinstance(l, FanoutLevel)), -1)
                        #    if sp_idx != -1:
                        #        fanouts.pop(i)
                        #for i in range(len(fanouts)):
                        #    fanouts[i] = (fanouts[i][0], fanouts[i][1] + 1)
                        #    if fanouts[i] in increased_fanouts:
                        #        increased_fanouts[fanouts[i]][0] += 1
                        #    else:
                        #        increased_fanouts[fanouts[i]] = [fanouts[i][1], fanouts[i][1] - 1, fanouts[i][0]]
                        while sp_idx != -1:
                            for dim, f in largest_product_less_than_with_tags([(dim, p) for dim, f in fanouts for p in prime_factors_list(f)], new_arch[sp_idx].mesh//new_arch[sp_idx].factors.fullProduct(), 1)[0]:
                                if not new_arch.moveFactor(0, sp_idx, dim, f):
                                    print(f"Issue in {shorten_path(filepath)}: failed to move factor {f} on dim {dim} from sp. fanout level 0 to sp. fanout level {dst}...")
                                total_per_dim[dim] *= f
                                for i in range(len(fanouts)):
                                    if fanouts[i][0] == dim and fanouts[i][1] % f == 0:
                                        fanouts[i] = (fanouts[i][0], fanouts[i][1]//f)
                                        break
                                    if i == len(fanouts) - 1:
                                        print(f"Issue in {shorten_path(filepath)}: something exploded while allocating factor {f} of dim {dim} on level {sp_idx}...")
                            sp_idx = next((i + sp_idx + 1 for i, l in enumerate(new_arch[sp_idx + 1:]) if isinstance(l, FanoutLevel)), -1)
                        for i in range(len(fanouts) - 1, -1, -1):
                            if fanouts[i][1] == 1:
                                fanouts.pop(i)
                                continue
                            fanouts[i] = (fanouts[i][0], fanouts[i][1] + 1)
                            if fanouts[i] in increased_fanouts:
                                increased_fanouts[fanouts[i]][0] += 1
                            else:
                                increased_fanouts[fanouts[i]] = [fanouts[i][1], fanouts[i][1] - 1, fanouts[i][0]]
                    if increased_fanouts:
                        paddings = ", ".join(f"{dim}:{base}->{padded}" for padded, base, dim in increased_fanouts.values())
                        print(f"Warning in {shorten_path(filepath)}: fanouts padded {paddings}")
                    
                    #print("-------")
                    #for l in new_arch: # REMOVE ME
                    #    print(l.name)
                    
                    if not all(comp_[dim] == total_per_dim[dim] for dim in total_per_dim.keys()):
                        print(f"Issue in {shorten_path(filepath)}: not all comp factors were allocated correctly (comp: {comp_}, allocated: {total_per_dim})...")
                        continue
                    
                    edp = EDP(new_arch, False, True)
                    mops = MOPs(new_arch)
                    energy = Energy(new_arch, True)
                    latency = Latency(new_arch)
                    utilization = new_arch.spatialUtilization()
                    
                    #printFactors(new_arch)
                    #printMOPs(new_arch)
                    #printLatency(new_arch)
                    #for level in new_arch:
                    #    print(level.name, level.dataflow, level.bypasses if isinstance(level, MemLevel) else level.mesh)
                    
                    table.add_row([filename, arch_name, comp_name, f"{edp:.3e}", f"{mops[0]+mops[1]:.0f}", f"{latency:.3e}", f"{energy:.3e}", f"{utilization:.3e}", f"{runtime:.3f}"])

    print(table)


"""
#OLD JUNK:

            class simpleLevel():
                def __init__(self, name : str, associated_idx : int):
                    self.name = name
                    self.associated_idx = associated_idx
                    self.loops : list[tuple[str, int]] = []
            
            in_levels = [simpleLevel(arch[(idx := next((i for i, l in enumerate(arch) if l.in_bp == 1 FIX ME), -1))].name.lower(), idx)]
            w_levels = [simpleLevel(arch[(idx := next((i for i, l in enumerate(arch) if l.w_bp == 1 FIX ME), -1))].name.lower(), idx)]
            out_levels = [simpleLevel(arch[(idx := next((i for i, l in enumerate(arch) if l.out_bp == 1 FIX ME), -1))].name.lower(), idx)]
            future_in_levels = []
            future_w_levels = []
            future_out_levels = []
            
            stats = os.path.join(subdir, "timeloop-mapper.stats.txt")
            mapping = os.path.join(subdir, "timeloop-mapper.map.txt")
            
            if not os.path.exists(stats) or not os.path.exists(mapping):
                print("Unrecognized results files in:", subdir)
                continue
            
            last_line = read_n_to_last_line(stats)
            runtime = re.search(r'time:\s*([\d.eE+-]+)', last_line)
            if not runtime:
                print("Could not find the mapper's runtime in:", stats)
                continue
            runtime = float(runtime.group(1))
            
            with open(mapping, 'r') as file:
                innermost_level = in_levels[0]
                total_per_dim = {dim : 1 for dim in comp.keys() if not dim.endswith("stride") or dim.endswith("dilation")}
                for line in file.readlines():
                    if (match := re.match(r'\s*for\s([a-zA-Z])\sin\s\[0\,\s(\d+)\)\s*(\w+)\s+(\w+)\s+(\w+)', line)):
                        # 1) break each level in 3, each instance storing one operand and bypassing the other two. Make less than 3 if the level by default has some bypass
                        #    => give each instance ~infinite capacity, but the same access costs and divide the bandwidth evenly.
                        # 2) the question then becomes when to switch to the next level. Assume that you start with DRAM for each operand and go down the loops:
                        #    => when encountering a loop with a lower level for an operand, if the loop is NOT orthogonal to the operand and the level does not bypass the operand, switch the current level
                        #       for that operand to the new one.
                        # 3) bundle instances of the same level for different operands together (and also sum back up the available bandwidth) in a single instance when:
                        #    => an instance stores an operand and, the other stores its operand too OR the other's operand is already stored by an inner level (NOT an outer one).
                        #    => for SP fanout levels, place them down as soon as you have the first level inside the SP fanout being used.
                        # 4) so long as no change to the levels occur, accumulate factors on the innermost of the currently in use levels, in case of a tie, target the inputs-holding instance of the level first,
                        #    then the weights-holding one, and lastly the outputs one. When bundling levels, one will in automatic have 0 iterations associated.
                        dim = match.group(1)
                        if dim in dims_renaming:
                            dim = dims_renaming[dim]
                        factor = match.group(2)
                        
                        #mem_in, mem_w, mem_out = level_name_to_idx[match.group(3).lower()], level_name_to_idx[match.group(4).lower()], level_name_to_idx[match.group(5).lower()]
                        # TODO: check that the order is I, W, O from the first line!!!!!
                        mem_in, mem_w, mem_out = match.group(3).lower(), match.group(4).lower(), match.group(5).lower()
                        if dim in coupling.flat_in_coupling and in_levels[-1].name.lower() != mem_in:
                            level_idx = next((i + ... for i, l in enumerate(arch[in_levels[-1].associated_idx:]) if l.name.lower() == mem_in and l.in_bp FIX ME == 1), -1)
                            assert level_idx != -1, f"Issue in {subdir}: can't find a level storing inputs with name {mem_in}..."
                            
                            i = 0
                            while i < len(future_in_levels):
                                if level_idx > future_in_levels[i].associated_idx:
                                    if not any(l.associated_idx == future_in_levels[i].associated_idx for l in in_levels):
                                        in_levels.append(future_in_levels[i])
                                    future_in_levels.pop(i)
                                else:
                                    i += 1
                            
                            in_levels.append(simpleLevel(mem_in, level_idx))
                            
                            if arch[in_levels[-1].associated_idx].w_bp FIX ME == 1 and not any(l.associated_idx == in_levels[-1].associated_idx for l in w_levels):
                                future_w_levels.append(simpleLevel(mem_in, level_idx))
                            if arch[in_levels[-1].associated_idx].out_bp FIX ME == 1 and not any(l.associated_idx == in_levels[-1].associated_idx for l in out_levels):
                                future_out_levels.append(simpleLevel(mem_in, level_idx))
                                
                            if innermost_level.associated_idx < level_idx:
                                innermost_level = in_levels[-1]
                        if dim in coupling.flat_w_coupling and w_levels[-1].name.lower() != mem_w:
                            level_idx = next((i + ... for i, l in enumerate(arch[w_levels[-1].associated_idx:]) if l.name.lower() == mem_w and l.w_bp FIX ME == 1), -1)
                            assert level_idx != -1, f"Issue in {subdir}: can't find a level storing weights with name {mem_w}..."
                            
                            i = 0
                            while i < len(future_w_levels):
                                if level_idx > future_w_levels[i].associated_idx:
                                    if not any(l.associated_idx == future_w_levels[i].associated_idx for l in w_levels):
                                        w_levels.append(future_w_levels[i])
                                    future_w_levels.pop(i)
                                else:
                                    i += 1
                            
                            w_levels.append(simpleLevel(mem_w, level_idx))
                            
                            if arch[w_levels[-1].associated_idx].in_bp FIX ME == 1 and not any(l.associated_idx == w_levels[-1].associated_idx for l in in_levels):
                                future_in_levels.append(simpleLevel(mem_w, level_idx))
                            if arch[w_levels[-1].associated_idx].out_bp FIX ME == 1 and not any(l.associated_idx == w_levels[-1].associated_idx for l in out_levels):
                                future_out_levels.append(simpleLevel(mem_w, level_idx))
                            
                            if innermost_level.associated_idx < level_idx:
                                innermost_level = w_levels[-1]
                        if dim in coupling.flat_out_coupling and out_levels[-1].name.lower() != mem_out:
                            level_idx = next((i + ... for i, l in enumerate(arch[out_levels[-1].associated_idx:]) if l.name.lower() == mem_out and l.out_bp FIX ME == 1), -1)
                            assert level_idx != -1, f"Issue in {subdir}: can't find a level storing outputs with name {mem_out}..."
                            
                            i = 0
                            while i < len(future_out_levels):
                                if level_idx > future_out_levels[i].associated_idx:
                                    if not any(l.associated_idx == future_out_levels[i].associated_idx for l in out_levels):
                                        out_levels.append(future_out_levels[i])
                                    future_out_levels.pop(i)
                                else:
                                    i += 1
                            
                            out_levels.append(simpleLevel(mem_out, level_idx))
                            
                            if arch[out_levels[-1].associated_idx].in_bp FIX ME == 1 and not any(l.associated_idx == out_levels[-1].associated_idx for l in in_levels):
                                future_in_levels.append(simpleLevel(mem_out, level_idx))
                            if arch[out_levels[-1].associated_idx].w_bp FIX ME == 1 and not any(l.associated_idx == out_levels[-1].associated_idx for l in w_levels):
                                future_w_levels.append(simpleLevel(mem_out, level_idx))
                            
                            if innermost_level.associated_idx < level_idx:
                                innermost_level = out_levels[-1]
                        
                        innermost_level.loop.append((dim, factor))
                    
                    elif True: # math SP fanouts here
                        continue
                
                levels = []
                while in_levels or w_levels or out_levels:
                    in_level = in_levels[0].associated_idx if in_levels else math.inf
                    w_level = w_levels[0].associated_idx if w_levels else math.inf
                    out_level = out_levels[0].associated_idx if out_levels else math.inf
                    outermost = min(in_level, w_level, out_level)
                    if in_level == outermost:
                        levels[]
"""