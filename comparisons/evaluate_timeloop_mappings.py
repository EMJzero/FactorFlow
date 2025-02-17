# GOAL: Parse and re-evaluate timeloop-produced mapping with FF's analytical model.

from prettytable import PrettyTable
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

def read_n_to_last_line(filename, n = 1):
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


if __name__ == "__main__":

    if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]) or not os.path.isdir(sys.argv[1]):
        print(f"The first argument ({sys.argv[1]}) must be a path to a directory containing one or more Timeloop output directories.")
        print("E.g.: say that '~/outputs/test_1' and '~/outputs/test_2' are where Timeloop wrote its outputs, hence you have file like '~/outputs/test_1/timeloop-mapper.map.txt' and '~/outputs/test_1/timeloop-mapper.stats.txt', the first argument shall be '~/outputs'. Note also that the middle directory's name, like 'test_1' shall contain an underscore to separate the name of the used architecture and the used computation like 'arch_comp'.")
        sys.exit(1)
    root = sys.argv[1]
    
    archs = {"gemmini": arch_gemmini, "eyeriss": arch_eyeriss, "simba": arch_simba, "tpu": arch_tpu, "gemmini-conv": arch_gemmini_conv, "eyeriss-conv": arch_eyeriss_conv, "simba-conv": arch_simba_conv, "tpu-conv": arch_tpu_conv}
    gemm_comps = comp_BERT_large | comp_maestro_blas
    conv_comps = {"VGG16-" + k: v for k, v in comp_vgg_16.items()} | {"ResNet18-" + k: v for k, v in comp_resnet_18.items()} | benchmark_convs
    conv_transp = benchmark_convs_transposed
    table = PrettyTable(["Arch", "Comp", "EDP[J*cycle]", "MOPs", "Latency[cc]", "Energy[uJ]", "Utilization[/]", "Runtime"])
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
            arch.initFactors(comp)
            
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
                level_idx = -1
                dataflow = []
                total_per_dim = {dim : 1 for dim in comp.keys() if not dim.endswith("stride") and not dim.endswith("dilation")}
                for line in file.readlines():
                    if (match := re.match(r'(\w+)\s*\[', line)): # new level line
                        #print(match.group(1), level_idx)
                        arch[level_idx].dataflow = [dim for dim in arch[level_idx].dataflow if dim not in dataflow] + dataflow.copy()
                        dataflow.clear()
                        level_idx += 1
                    elif (match := re.match(r'\|\s*for\s([a-zA-Z])\sin\s\[0\:(\d+)\)\n', line)): # temporal loop line
                        #print(match.group(1), match.group(2), level_idx)
                        dim = match.group(1)
                        if dim in dims_renaming:
                            dim = dims_renaming[dim]
                        if level_idx > 0:
                            for factor in prime_factors_list(int(match.group(2))):
                                if not arch.moveFactor(0, level_idx, dim, factor):
                                    print(f"Issue in {subdir}: violated factors constraints while allocating factor {factor} on level {arch[level_idx].name}...")
                        for factor in prime_factors_list(int(match.group(2))):
                            total_per_dim[dim] *= factor
                        assert dim in arch[level_idx].dataflow, f"Unsupported dimension ({dim}) on level {arch[level_idx].name}."
                        dataflow.append(dim)
                    elif (match := re.match(r'\|\s*for\s([a-zA-Z])\sin\s\[0\:(\d+)\)\s\(Spatial', line)): # spatial loop line
                        #print(match.group(1), match.group(2), level_idx)
                        dim = match.group(1)
                        if dim in dims_renaming:
                            dim = dims_renaming[dim]
                        for factor in prime_factors_list(int(match.group(2))):
                            if not arch.moveFactor(0, level_idx, dim, factor):
                                print(f"Issue in {subdir}: violated factors constraints while allocating factor {factor} on level {arch[level_idx].name}...")
                            total_per_dim[dim] *= factor
                        assert dim in arch[level_idx].dataflow, f"Unsupported dimension ({dim}) on level {arch[level_idx].name}."
                
                if not all(comp[dim] == total_per_dim[dim] for dim in total_per_dim.keys()):
                    print(f"Issue in {subdir}: not all comp factors were allocated correctly (comp: {comp}, allocated: {total_per_dim})...")
                
                edp = EDP(arch, False, True)
                mops = MOPs(arch)
                energy = Energy(arch, True)
                latency = Latency(arch)
                utilization = arch.spatialUtilization()
                
                #printFactors(arch)
                #printMOPs(arch)
                #printLatency(arch)
                
                table.add_row([arch_name, comp_name, f"{edp:.3e}", f"{mops[0]+mops[1]:.0f}", f"{latency:.3e}", f"{energy:.3e}", f"{utilization:.3e}", f"{runtime:.3f}"])

    print(table)