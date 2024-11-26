import collections.abc

from levels import *
from factors import *

"""
Returns a string with a pretty textual representation of the provided dictionary.
"""
def prettyFormatDict(dictionary, level = 0):
    string = ""
    for key, value in (dictionary.items() if isinstance(dictionary, dict) else zip(["" for i in dictionary], dictionary)):
        string += ''*level + (f"{key}: " if key != "" else "- ")
        if isinstance(value, dict):
            string += "\n" + prettyFormatDict(value, level + 4)
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
            string += "\n" + prettyFormatDict(value, level + 4)
        else:
            string += str(value)
        string += "\n"
    return string.rstrip()

"""
Prints to stdout the provided object in a nicely formatted way.
Explicitly supported objects are: classes, iterables, dictionaries, and atomics.
Any other object should also work reasonably well.

Any attribute or key appearing in 'omit_fields' will not be printed.
"""
def prettyPrint(obj, omit_fields = None):
    omit_fields = omit_fields if omit_fields else []
    seen = set()
    res = ""
    
    def pp(obj, indent=0, keep_first_indend = True):
        nonlocal res
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key not in omit_fields:
                    res += f"{' ' * (indent + 4)}{key}:\n"
                    pp(value, indent + 4)
            if len(obj) == 0:
                res += f"{' ' * (indent + 4)}<empty>\n"
            return
        if isinstance(obj, collections.abc.Iterable) and not isinstance(obj, str):
            if len(obj) == 0:
                res += "[]\n"
                return
            res += f"{' ' * indent * keep_first_indend}[\n"
            for i, item in enumerate(obj):
                res += f"{' ' * (indent + 4)}Item {i}:\n"
                pp(item, indent + 8)
            res += f"{' ' * indent}]\n"
            return
        if hasattr(obj, "__dict__"):
            name = obj.name if hasattr(obj, "name") else id(obj)
            if id(obj) in seen:
                res += f"{' ' * indent * keep_first_indend}- Reference to {obj.__class__.__name__}({name}) already printed\n"
                return
            seen.add(id(obj))
            res += f"{' ' * indent * keep_first_indend}{obj.__class__.__name__}({name}):\n"
            for attr, value in obj.__dict__.items():
                if attr.startswith('_') or attr in omit_fields:
                    continue
                if hasattr(value, "__dict__"):
                    res += f"{' ' * (indent + 4)}{attr}:\n"
                    pp(value, indent + 4)
                elif isinstance(value, dict):
                    res += f"{' ' * (indent + 4)}{attr}:\n"
                    pp(value, indent + 4)
                elif isinstance(value, collections.abc.Iterable) and not isinstance(value, str):
                    res += f"{' ' * (indent + 4)}{attr}: "
                    pp(value, indent + 4, False)
                else:
                    res += f"{' ' * (indent + 4)}{attr}: {value}\n"
        else:
            res += f"{' ' * (indent + 4)}- {obj}\n"
    
    pp(obj)
    print(res)

"""
Print to stdout a summary of the factors allocated to each dimension across the
entire architecture. Dimensions order also reflects dataflows.
If 'omitOnes' is True, dimension with a single iterations are omitted.
"""
def printFactors(arch, omitOnes = True):
    for level in arch:
        fac_str = f"{level.name} -> "
        for dim in level.dataflow:
            if not (level.factors.dimProduct(dim) == 1 and omitOnes):
                fac_str += f"{dim}: {level.factors.dimProduct(dim)}, "
        print(fac_str[:-2])

"""
Returns a summary string representing the factors allocated to each dimension
across the entire architecture.
"""
def factorsString(arch):
    res = ""
    for level in arch:
        res += f"{level.name}["
        for dim in level.dataflow:
            res += f"{dim}{level.factors.dimProduct(dim)} "
        res = res[:-1] + "] "
    return res

"""
Print to stdout a summary of the tile sizes for each dimension across the
entire architecture. Dimensions order also reflects dataflows.
"""
def printTileSizes(arch):
    for level in arch:
        fac_str = f"{level.name} -> "
        for dim in level.dataflow:
            fac_str += f"{dim}: {level.tile_sizes[dim]}, "
        print(fac_str[:-2])

"""
Print to stdout a summary of the memory operations (MOPs) across the memory levels
in the architecture, broken down per-operand. A few notes:
- If "per_instance" is True, reported MOPs are divided by the number of instances
  of a certain component, otherwise they are aggregate across all such instances.
  Default is False.
- R is short for READS, while W for WRITES.
- the fill/drain/read/update terms refer to the Buffet model adopted to describe
  the memory levels, and are explicitated only for levels not bypassing the outputs,
  since otherwise drain and updates are 0, while fill and read can be inferred
  from Tot_W and Tot_R respectively.
"""
def printMOPs(arch, per_instance = False):
    tot_reads = 0
    tot_writes = 0
    WMOPs = 0
    for level in arch:
        if isinstance(level, MemLevel):
            scaling = level.instances if per_instance else 1
            if 'out' not in level.bypasses:
                print(f"{level.name}:{chr(9) * (2 - (len(level.name) + 1)//8)}Out_R = {(level.out_reads - level.last_out_writes)/scaling:.0f} (reads) + {level.last_out_writes/scaling:.0f} (drains), Out_W = {(level.out_writes - level.last_out_reads)/scaling:.0f} (updates) + {level.last_out_reads/scaling:.0f} (fills)")
            reads = level.in_reads + level.w_reads + level.out_reads
            writes = level.in_writes + level.w_writes + level.out_writes
            print(f"{level.name}:{chr(9) * (2 - (len(level.name) + 1)//8)}{level.in_reads/scaling:.0f} In_R, {level.w_reads/scaling:.0f} W_R, {level.out_reads/scaling:.0f} Our_R, {reads/scaling:.0f} Tot_R,\n\t\t{level.in_writes/scaling:.0f} In_W, {level.w_writes/scaling:.0f} W_W, {level.out_writes/scaling:.0f} Out_W, {writes/scaling:.0f} Tot_W")
            tot_reads += reads
            tot_writes += writes
            WMOPs += level.WMOPs(reads, writes)
        elif isinstance(level, FanoutLevel):
            continue
        elif isinstance(level, ComputeLevel):
            WMOPs += level.computeCost(level.temporal_iterations*level.instances)
            break
    print(f"Totals:\t\t{tot_reads:.0f} R, {tot_writes:.0f} W, {tot_reads+tot_writes:.0f} Tot")
    print(f"Energy:\t\t{WMOPs*10**-6:.3f} uJ")

"""
Print to stdout a summary of the latency, bandwidth and stalls across the levels
in the architecture, broken down per operation. A few notes:
- reported bandwidths are in values/cycle, where a value has the bitwidth
  (value_bits) specified on the level.
- R is short for READS, while W for WRITES.
- RD is short for READ & DRAIN (the two Buffet read operations), while FU for
  FILL & UPDATE (the two Buffet write operations).
- The "ideal bandwidth" represents the bandwidth that would have been required to
  incur in zero stall cycles. It follows then that "stall cycles" are the cycles
  required to move data which exceed those required by the computation, thus
  forcing the latter to wait/stall.
"""
def printLatency(arch):
    max_latency, max_latency_level_name = 0, "<<Unavailable>>"
    for level in arch:
        if isinstance(level, MemLevel):
            if max_latency <= level.getSettedLatency():
                max_latency = level.getSettedLatency()
                max_latency_level_name = level.name
            print(f"{level.name}:{chr(9) * (2 - (len(level.name) + 1)//8)}{level.latency_read_drain:.0f}cc RD and {level.latency_fill_update:.0f}cc FU Latency, {level.read_bandwidth:.1f} R and {level.write_bandwidth:.1f} W Bandwidth,\n\t\t{level.ideal_bandwidth_read:.3f} R and {level.ideal_bandwidth_update:.3f} U and {level.ideal_bandwidth_fill:.3f} F and {level.ideal_bandwidth_drain:.3f} D Ideal Bandwidth,\n\t\t{level.cc_per_tile:.0f}cc per Tile, {level.stall_cycles:.0f} Stall Cycles")
        elif isinstance(level, FanoutLevel):
            continue
        elif isinstance(level, ComputeLevel):
            break
    print(f"Max Latency:\t{max_latency:.0f}cc of level {max_latency_level_name}")

"""
Print to stdout the total amount of padding required by the different dimensions
of the computation. This is non-zero iif the PADDED_MAPPINGS is True.
"""
def printPadding(arch, comp):
    total_iterations = {dim: 1 for dim in arch.coupling.dims}
    for level in arch:
        for dim in arch.coupling.dims:
            total_iterations[dim] *= level.factors.dimProduct(dim)
    print("Padding required:")
    for dim in arch.coupling.dims:
        print(f"\t{dim}: {total_iterations[dim] - comp[dim]:.0f} ({comp[dim]} -> {total_iterations[dim]})")

"""
Print to stdout the energy per action of the levels in the give architecture.
"""
def printEnergyPerAction(arch):
    for level in arch:
        if isinstance(level, MemLevel):
            print(f"{level.name}:{chr(9) * (2 - (len(level.name) + 1)//8)}read {level.read_access_energy:.3e} pJ, write {level.write_access_energy:.3e} pJ, leak {level.leakage_energy:.3e} pJ/cc (values per wordline {level.values_per_wordline})")
        if isinstance(level, ComputeLevel):
            print(f"{level.name}:{chr(9) * (2 - (len(level.name) + 1)//8)}compute {level.compute_energy:.3e} pJ, leak {level.leakage_energy:.3e} pJ/cc")

"""
Print to stdout the area of the levels in the give architecture.
"""
def printAreaPerLevel(arch):
    physical_instances = 1
    for level in arch:
        if level.area != None:
            print(f"{level.name}:{chr(9) * (2 - (len(level.name) + 1)//8)}total level area {level.area*physical_instances:.3e} um^2, per instance area {level.area:.3e} um^2")
        else:
            print(f"{level.name}:{chr(9) * (2 - (len(level.name) + 1)//8)}N/A")
        if isinstance(level, SpatialLevel):
            physical_instances *= level.mesh

"""
Shorthand to invoke prettyPrint on an architecture.
"""
def printArch(arch):
    prettyPrint(arch[::-1], ['arch'])