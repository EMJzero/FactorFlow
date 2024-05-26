import collections.abc

from levels import *
from factors import *

def pretty_format_dict(dictionary, level = 0):
    string = ""
    for key, value in (dictionary.items() if isinstance(dictionary, dict) else zip(["" for i in dictionary], dictionary)):
        string += ''*level + (f"{key}: " if key != "" else "- ")
        if isinstance(value, dict):
            string += "\n" + pretty_format_dict(value, level + 4)
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
            string += "\n" + pretty_format_dict(value, level + 4)
        else:
            string += str(value)
        string += "\n"
    return string.rstrip()

# use it with arch[::-1]!!
def prettyPrint(obj):
    seen = set()
    
    def pp(obj, indent=0, keep_first_indend = True):
        if isinstance(obj, dict):
            for key, value in obj.items():
                print(f"{' ' * (indent + 4)}{key}: ")
                pp(value, indent + 4)
            if len(obj) == 0:
                print(f"{' ' * (indent + 4)}<empty>")
            return
        if isinstance(obj, collections.abc.Iterable) and not isinstance(obj, str):
            if len(obj) == 0:
                print("[]")
                return
            print(f"{' ' * indent * keep_first_indend}[")
            for i, item in enumerate(obj):
                print(f"{' ' * (indent + 4)}Item {i}: ")
                pp(item, indent + 8)
            print(f"{' ' * indent}]")
            return
        if hasattr(obj, "__dict__"):
            name = obj.name if hasattr(obj, "name") else id(obj)
            if id(obj) in seen:
                print(f"{' ' * indent * keep_first_indend}- Reference to {obj.__class__.__name__}({name}) already printed")
                return
            seen.add(id(obj))
            print(f"{' ' * indent * keep_first_indend}{obj.__class__.__name__}({name}): ")
            for attr, value in obj.__dict__.items():
                if attr.startswith('_'):
                    continue
                if hasattr(value, "__dict__"):
                    print(f"{' ' * (indent + 4)}{attr}: ")
                    pp(value, indent + 4)
                elif isinstance(value, dict):
                    print(f"{' ' * (indent + 4)}{attr}: ")
                    pp(value, indent + 4)
                elif isinstance(value, collections.abc.Iterable) and not isinstance(value, str):
                    print(f"{' ' * (indent + 4)}{attr}: ", end='')
                    pp(value, indent + 4, False)
                else:
                    print(f"{' ' * (indent + 4)}{attr}: {value}")
        else:
            print(f"{' ' * (indent + 4)}- {obj}")
    pp(obj)

def printFactors(arch):
    for level in arch:
        fac_str = f"{level.name} -> "
        for dim in level.dataflow:
            fac_str += f"{dim}: {level.factors.dimProduct(dim)}, "
        print(fac_str[:-2])

def printTileSizes(arch):
    for level in arch:
        fac_str = f"{level.name} -> "
        for dim in level.dataflow:
            fac_str += f"{dim}: {level.tile_sizes[dim]}, "
        print(fac_str[:-2])

def printMOPs(arch, per_instance = False):
    temporal_iterations_inputs = 1
    temporal_iterations_weights = 1
    temporal_iterations_outputs = 1
    spatial_iterations_inputs = 1
    spatial_iterations_weights = 1
    spatial_iterations_outputs = 1
    last_in_reads, last_w_reads, last_out_reads, last_out_writes = 0, 0, 0, 0
    for i in range(len(arch)):
        level = arch[i]
        if isinstance(level, MemLevel):
            # multiply by spatial_iterations too because memory is replicated spatially
            in_reads, w_reads, out_reads, out_writes = level.MOPs()
            in_reads *= temporal_iterations_inputs
            w_reads *= temporal_iterations_weights
            out_reads *= temporal_iterations_outputs
            out_writes *= temporal_iterations_outputs
            if not per_instance:
                in_reads *= spatial_iterations_inputs
                w_reads *= spatial_iterations_weights
                out_reads *= spatial_iterations_outputs
                out_writes *= spatial_iterations_outputs
            if 'in' not in level.bypasses:
                in_writes = last_in_reads #reads above are written here
                last_in_reads = in_reads
            else:
                in_writes = 0
            if 'w' not in level.bypasses:
                w_writes = last_w_reads #reads above are written here
                last_w_reads = w_reads
            else:
                w_writes = 0
            if 'out' not in level.bypasses:
                print(f"{level.name}:{chr(9) * (2 - (len(level.name) + 1)//8)}Out_R = {out_reads:.0f} (read) + {last_out_writes:.0f} (drain), Out_W = {out_writes:.0f} (updates) + {last_out_reads:.0f} (fills)")
                out_writes += last_out_reads #reads above are written here
                last_out_reads = out_reads
                out_reads += last_out_writes #writes above where read here
                last_out_writes = out_writes
            dataflow = level.actualDataflow()
            # outer datataflows do not cause affect iterations at inner levels
            temporal_iterations_inputs *= level.factors.fullProduct()
            temporal_iterations_weights *= level.factors.fullProduct()
            temporal_iterations_outputs *= level.factors.fullProduct()
            # outer dataflows affect inner iterations (WRONG)
            #if dataflow == "WS":
            #    temporal_iterations_weights *= level.factors.dimProduct('D')*level.factors.dimProduct('E')
            #    temporal_iterations_inputs *= level.factors.fullProduct()
            #    temporal_iterations_outputs *= level.factors.fullProduct()
            #elif dataflow == "OS":
            #    temporal_iterations_weights *= level.factors.fullProduct()
            #    temporal_iterations_inputs *= level.factors.fullProduct()
            #    temporal_iterations_outputs *= level.factors.dimProduct('D')*level.factors.dimProduct('L')
            #elif dataflow == "IS":
            #    temporal_iterations_weights *= level.factors.fullProduct()
            #    temporal_iterations_inputs *= level.factors.dimProduct('E')*level.factors.dimProduct('L')
            #    temporal_iterations_outputs *= level.factors.fullProduct()
            reads = in_reads + w_reads + out_reads
            writes = in_writes + w_writes + out_writes
            print(f"{level.name}:{chr(9) * (2 - (len(level.name) + 1)//8)}{in_reads:.0f} In_R, {w_reads:.0f} W_R, {out_reads:.0f} Our_R, {reads:.0f} Tot_R,\n\t\t{in_writes:.0f} In_W, {w_writes:.0f} W_W, {out_writes:.0f} Out_W, {writes:.0f} Tot_W")
        elif isinstance(level, FanoutLevel):
            # We are interested in all instances at once, essentially, so it is fine!
            #spatial_iterations_inputs, spatial_iterations_weights, spatial_iterations_outputs, _ = level.mulByDim(spatial_iterations_inputs, spatial_iterations_weights, spatial_iterations_outputs, 0)
            spatial_iterations_inputs *= level.factors.fullProduct()
            spatial_iterations_weights *= level.factors.fullProduct()
            spatial_iterations_outputs *= level.factors.fullProduct()
            # We don't need to scale those down, as they are added back after reads and writes have already been multiplied by iterations!
            # (Not need to use it, but the next line is the right one) With PE->PE forwarding, the last_*_reads and writes are reduced, because reused with forwarding, so we don't need to divide again!
            #last_in_reads, last_w_reads, last_out_reads, last_out_writes = level.divByDim(last_in_reads, last_w_reads, last_out_reads, last_out_writes)
            if per_instance:
                last_in_reads //= level.factors.fullProduct()
                last_w_reads //= level.factors.fullProduct()
                last_out_reads //= level.factors.fullProduct()
                last_out_writes //= level.factors.fullProduct()

def printMOPsNew(arch, per_instance = False):
    for level in arch:
        if isinstance(level, MemLevel):
            scaling = level.instances if per_instance else 1
            if 'out' not in level.bypasses:
                print(f"{level.name}:{chr(9) * (2 - (len(level.name) + 1)//8)}Out_R = {(level.out_reads - level.last_out_writes)/scaling:.0f} (reads) + {level.last_out_writes/scaling:.0f} (drains), Out_W = {(level.out_writes - level.last_out_reads)/scaling:.0f} (updates) + {level.last_out_reads/scaling:.0f} (fills)")
            reads = level.in_reads + level.w_reads + level.out_reads
            writes = level.in_writes + level.w_writes + level.out_writes
            print(f"{level.name}:{chr(9) * (2 - (len(level.name) + 1)//8)}{level.in_reads/scaling:.0f} In_R, {level.w_reads/scaling:.0f} W_R, {level.out_reads/scaling:.0f} Our_R, {reads/scaling:.0f} Tot_R,\n\t\t{level.in_writes/scaling:.0f} In_W, {level.w_writes/scaling:.0f} W_W, {level.out_writes/scaling:.0f} Out_W, {writes/scaling:.0f} Tot_W")
        elif isinstance(level, FanoutLevel):
            continue
        elif isinstance(level, ComputeLevel):
            break

def printLatency(arch):
    max_latency, max_latency_level_name = 0, "<<Error>>"
    temporal_iterations = 1
    spatial_iterations = 1
    last_in_reads, last_w_reads, last_out_reads, last_out_writes = 0, 0, 0, 0

    def printAndUpdate(latency, name, bandwidth = None, MOPs = None):
        nonlocal max_latency, max_latency_level_name
        print(f"{name}:{chr(9) * (2 - (len(name) + 1)//8)} {latency:.0f}cc Latency, {bandwidth} Bandwidth, {int(MOPs) if MOPs else MOPs} MOPs")
        if max_latency < latency:
            max_latency = latency
            max_latency_level_name = name
            
    for i in range(len(arch)):
        level = arch[i]
        if isinstance(level, MemLevel):
            in_reads, w_reads, out_reads, out_writes = level.MOPs()
            in_reads *= temporal_iterations*spatial_iterations
            w_reads *= temporal_iterations*spatial_iterations
            out_reads *= temporal_iterations*spatial_iterations
            out_writes *= temporal_iterations*spatial_iterations
            if 'in' not in level.bypasses:
                in_writes = last_in_reads #reads above are written here
                last_in_reads = in_reads
            else:
                in_writes = 0
            if 'w' not in level.bypasses:
                w_writes = last_w_reads #reads above are written here
                last_w_reads = w_reads
            else:
                w_writes = 0
            if 'out' not in level.bypasses:
                out_writes += last_out_reads #reads above are written here
                last_out_reads = out_reads
                out_reads += last_out_writes #writes above where read here
                last_out_writes = out_writes
            dataflow = level.actualDataflow()
            temporal_iterations *= level.factors.fullProduct()
            MOPs = in_reads + w_reads + out_reads + in_writes + w_writes + out_writes
            printAndUpdate(level.latency(MOPs//spatial_iterations), level.name, level.bandwidth, MOPs)
        elif isinstance(level, FanoutLevel):
            printAndUpdate(level.latency()*temporal_iterations, level.name)
            spatial_iterations *= level.factors.fullProduct()
        elif isinstance(level, ComputeLevel):
            printAndUpdate(level.latency()*temporal_iterations, level.name)
            break
    print(f"Max Latency:\t{max_latency:.0f}cc of level {max_latency_level_name}")

def printLatencyNew(arch):
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

def printArch(arch):
    prettyPrint(arch[::-1])