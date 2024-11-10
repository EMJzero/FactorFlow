from math import log2, ceil

from architectures.accelergy_hw_data import accelergy_estimate_energy, accelergy_estimate_area
from architectures.architectures import WS, OS, IS
from prints import printEnergyPerAction, printAreaPerLevel
from levels import *
from arch import *


"""
Helper function to instantiate a smartbuffer register file (see Accelergy docs)

Set "energy" to True to estimate energy, set it to "False" for area.
"""
def smartbuffer_registerfile(depth, word_bits, value_bits, cycle_seconds, technology, action = None, energy = True):
    assert not energy or action in ["read", "write", "leak"], f"SmartBuffer SRAM Estimator: Action ({action}) must be one of 'read', 'write', or 'leak' when estimating energy!"
    std_width, std_depth = 32, 64
    # NOTE: currently Accelergy is bugged and does not apply those scales, for consistency we do not apply them too...
    dynamic_energy_scale = 1#(16/std_width)*((12/std_depth)**(1.56/2))
    static_energy_scale = 1#(16/std_width)*(12/std_depth) # = area_scale

    registers_attributes = {
        "global_cycle_seconds": cycle_seconds,
        "technology": technology,
    }
    registers_address_gen_attributes = {
        "n_bits": value_bits,
        "precision": value_bits,
        "datawidth": value_bits,
        "technology": technology,
        "global_cycle_seconds": cycle_seconds,
        "cycle_seconds": cycle_seconds,
    }
    if energy:
        return accelergy_estimate_energy(query={
                "class_name": "aladdin_register",
                "attributes": registers_attributes,
                "action_name": action,
                "arguments": {"global_cycle_seconds": cycle_seconds}
            })*max(std_width, word_bits)*(dynamic_energy_scale if action != "leak" else static_energy_scale*max(std_depth, depth)) + accelergy_estimate_energy(query={
                "class_name": "aladdin_comparator",
                "attributes": registers_attributes,
                "action_name": ("compare" if action != "leak" else action),
                "arguments": {"global_cycle_seconds": cycle_seconds}
            })*max(std_depth, depth)*(dynamic_energy_scale if action != "leak" else static_energy_scale) + accelergy_estimate_energy(query={
                "class_name": "intadder",
                "attributes": registers_address_gen_attributes,
                "action_name": ("add" if action != "leak" else action),
                "arguments": {"global_cycle_seconds": cycle_seconds}
            })*(1 if action != "leak" else 2)
    else:
        return accelergy_estimate_area(query={
                "class_name": "aladdin_register",
                "attributes": registers_attributes,
            })*max(std_width, word_bits)*static_energy_scale*max(std_depth, depth) + accelergy_estimate_area(query={
                "class_name": "aladdin_comparator",
                "attributes": registers_attributes,
            })*max(std_depth, depth)*static_energy_scale + accelergy_estimate_area(query={
                "class_name": "intadder",
                "attributes": registers_address_gen_attributes,
            })*2

"""
Helper function to instantiate a smartbuffer SRAM (see Accelergy docs)

Set "energy" to True to estimate energy, set it to "False" for area.
"""
def smartbuffer_SRAM(depth, word_bits, rw_ports, banks, cycle_seconds, technology, action = None, energy = True):
    assert not energy or action in ["read", "write", "leak"], f"SmartBuffer SRAM Estimator: Action ({action}) must be one of 'read', 'write', or 'leak' when estimating energy!"
    
    SRAM_attributes = {
        "n_rw_ports": rw_ports,
        "depth": depth,
        "width": word_bits,
        "technology": technology,
        "cycle_seconds": cycle_seconds,
        "n_banks": banks,
    }
    address_gen_attributes = {
        "n_bits": max(1, ceil(log2(depth))) if depth >= 1 else 1,
        "precision": max(1, ceil(log2(depth))) if depth >= 1 else 1,
        "datawidth": max(1, ceil(log2(depth))) if depth >= 1 else 1,
        "technology": technology,
        "global_cycle_seconds": cycle_seconds,
        "cycle_seconds": cycle_seconds,
    }
    
    if energy:
        return accelergy_estimate_energy(query={
                "class_name": "SRAM",
                "attributes": SRAM_attributes,
                "action_name": action,
                "arguments": {"global_cycle_seconds": cycle_seconds}
            }) + accelergy_estimate_energy(query={
                "class_name": "intadder",
                "attributes": address_gen_attributes,
                "action_name": ("add" if action != "leak" else action),
                "arguments": {"global_cycle_seconds": cycle_seconds}
            })*(1 if action != "leak" else 2)
    else:
        return accelergy_estimate_area(query={
                "class_name": "SRAM",
                "attributes": SRAM_attributes,
            }) + accelergy_estimate_area(query={
                "class_name": "intadder",
                "attributes": address_gen_attributes,
            })*2


# >>> GEMMINI <<<
# > WS version  <

def get_arch_gemmini_hw_data():
    cycle_seconds = 1.2e-09
    technology = "32nm"
    arguments = { # in theory, this is not needed on anything but "leak"
        "global_cycle_seconds": cycle_seconds,
    }
    DRAM_attributes = {
        "type": "LPDDR4",
        "width": 64,
        "technology": technology,
        "cycle_seconds": cycle_seconds,
    }
    scratchpad_attributes = {
        "n_rw_ports": 2,
        "depth": 16384*2,
        "width": 128,
        "technology": technology,
        "cycle_seconds": cycle_seconds,
        "n_banks": 4,
    }
    accumulator_attributes = {
        "n_rw_ports": 2,
        "depth": 4096,
        "width": 32,
        "technology": technology,
        "cycle_seconds": cycle_seconds,
        "n_banks": 2,
    }
    register_attributes = {
        "n_rw_ports": 2,
        "depth": 1,
        "width": 8,
        "technology": technology,
        "cycle_seconds": cycle_seconds,
        "n_banks": 1,
    }
    
    arch = Arch([
        MemLevel(
            name = "DRAM",
            dataflow_constraints = [], # ['N', 'M', 'K'],
            size = 2**64-1, # number of entries
            read_wordline_access_energy = accelergy_estimate_energy({
                "class_name": "DRAM",
                "attributes": DRAM_attributes,
                "action_name": "read",
                "arguments": arguments
            }),
            write_wordline_access_energy = accelergy_estimate_energy({
                "class_name": "DRAM",
                "attributes": DRAM_attributes,
                "action_name": "write",
                "arguments": arguments
            }),
            leakage_energy = accelergy_estimate_energy({
                "class_name": "DRAM",
                "attributes": DRAM_attributes,
                "action_name": "leak",
                "arguments": arguments
            }),
            word_bits = 64,
            value_bits = 8,
            area = accelergy_estimate_area({
                "class_name": "DRAM",
                "attributes": DRAM_attributes,
            }),
            bandwidth = 8, # operands per cycle (shared)
            factors_constraints = {},
            bypasses = []
        ),
        MemLevel(
            name = "Scratchpad",
            dataflow_constraints = [], #WS,
            size = 512*(2**10), # number of entries
            read_wordline_access_energy = accelergy_estimate_energy({
                "class_name": "SRAM",
                "attributes": scratchpad_attributes,
                "action_name": "read",
                "arguments": arguments
            }),
            write_wordline_access_energy = accelergy_estimate_energy({
                "class_name": "SRAM",
                "attributes": scratchpad_attributes,
                "action_name": "write",
                "arguments": arguments
            }),
            leakage_energy = accelergy_estimate_energy({
                "class_name": "SRAM",
                "attributes": scratchpad_attributes,
                "action_name": "leak",
                "arguments": arguments
            }),
            word_bits = 128,
            value_bits = 8,
            area = accelergy_estimate_area({
                "class_name": "SRAM",
                "attributes": scratchpad_attributes,
            }),
            bandwidth = 32, # operands per cycle (shared)
            factors_constraints = {},
            bypasses = ['out']
        ),
        FanoutLevel(
            name = "SARows",
            dim = WS[0],
            mesh = 16,
            # pe_to_pe should be used, since Gemmini uses a systolic array, but Timeloop
            # does not have this feature, so for sake of comparison, it is turned off
            #pe_to_pe = True, 
            area = 0,
            factors_constraints = {'M': 16}
        ),
        MemLevel(
            name = "Accumulator",
            dataflow_constraints = [], #WS,
            size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remember to account for operand size)
            read_wordline_access_energy = accelergy_estimate_energy({
                "class_name": "SRAM",
                "attributes": accumulator_attributes,
                "action_name": "read",
                "arguments": arguments
            }),
            write_wordline_access_energy = accelergy_estimate_energy({
                "class_name": "SRAM",
                "attributes": accumulator_attributes,
                "action_name": "write",
                "arguments": arguments
            }),
            leakage_energy = accelergy_estimate_energy({
                "class_name": "SRAM",
                "attributes": accumulator_attributes,
                "action_name": "leak",
                "arguments": arguments
            }),
            word_bits = 32,
            value_bits = 32,
            area = accelergy_estimate_area({
                "class_name": "SRAM",
                "attributes": accumulator_attributes,
            }),
            bandwidth = 8, # operands per cycle (shared)
            factors_constraints = {}, # the systolic array does a 16x16 matmul in this case
            bypasses = ['in', 'w']
        ),
        FanoutLevel(
            name = "SACols",
            dim = WS[1],
            mesh = 16,
            # pe_to_pe should be used, since Gemmini uses a systolic array, but Timeloop
            # does not have this feature, so for sake of comparison, it is turned off
            #pe_to_pe = True, 
            area = 0,
            factors_constraints = {'K': 16}
        ),
        MemLevel(
            name = "Register",
            dataflow_constraints = WS,
            size = 1, # number of entries
            read_wordline_access_energy = accelergy_estimate_energy({
                "class_name": "SRAM",
                "attributes": register_attributes,
                "action_name": "read",
                "arguments": arguments
            }),
            write_wordline_access_energy = accelergy_estimate_energy({
                "class_name": "SRAM",
                "attributes": register_attributes,
                "action_name": "write",
                "arguments": arguments
            }),
            leakage_energy = accelergy_estimate_energy({
                "class_name": "SRAM",
                "attributes": register_attributes,
                "action_name": "leak",
                "arguments": arguments
            }),
            word_bits = 8,
            value_bits = 8,
            area = accelergy_estimate_area({
                "class_name": "SRAM",
                "attributes": register_attributes,
            }),
            bandwidth = 2, # operands per cycle (shared)
            factors_constraints = {'M': 1, 'K': 1, 'N': 16},
            bypasses = ['in', 'out']
        ),
        ComputeLevel(
            name = "Compute",
            dim = WS[2],
            mesh = 1,
            compute_energy = accelergy_estimate_energy({
                "class_name": "aladdin_adder",
                "attributes": {
                    "width_a": 8,
                    "width_b": 8,
                    "technology": technology,
                },
                "action_name": "read",
                "arguments": arguments
            }) + accelergy_estimate_energy({
                "class_name": "aladdin_adder",
                "attributes": {
                    "width": 16,
                    "technology": technology,
                },
                "action_name": "read",
                "arguments": arguments
            }),
            leakage_energy = accelergy_estimate_energy({
                "class_name": "aladdin_adder",
                "attributes": {
                    "width_a": 8,
                    "width_b": 8,
                    "technology": technology,
                },
                "action_name": "leak",
                "arguments": arguments
            }) + accelergy_estimate_energy({
                "class_name": "aladdin_adder",
                "attributes": {
                    "width": 16,
                    "technology": technology,
                },
                "action_name": "leak",
                "arguments": arguments
            }),
            area = accelergy_estimate_area({
                "class_name": "aladdin_adder",
                "attributes": {
                    "width_a": 8,
                    "width_b": 8,
                    "technology": technology,
                }
            }) + accelergy_estimate_area({
                "class_name": "aladdin_adder",
                "attributes": {
                    "width": 16,
                    "technology": technology,
                }
            }),
            cycles = 1,
            factors_constraints = {'N': 1}
        )
    ], name="Gemmini (Accelergy data)")

    print(f"\nEnergy per action in {arch.name}:")
    printEnergyPerAction(arch)
    print(f"\nArea per level in {arch.name}:")
    printAreaPerLevel(arch)
    print(f"Total area of {arch.name}: {arch.totalArea(True):.3e} um^2")
    return arch


# >>> EYERISS <<<

# C -> K
# M -> M
# P -> N
def get_arch_eyeriss_hw_data():
    cycle_seconds = 1.2e-09
    technology = "32nm"
    arguments = { # in theory, this is not needed on anything but "leak"
        "global_cycle_seconds": cycle_seconds,
    }
    DRAM_attributes = {
        "type": "LPDDR4",
        "width": 64,
        "technology": technology,
        "cycle_seconds": cycle_seconds,
    }
    SRAM_attributes = {
        "n_rw_ports": 1,
        "depth": 16384,
        "width": 64,
        "technology": technology,
        "cycle_seconds": cycle_seconds,
        "n_banks": 2,
    }
    
    arch = Arch([
        MemLevel(
            name = "DRAM",
            dataflow_constraints = [], # ['N', 'M', 'K'],
            size = 2**64-1, # number of entries
            read_wordline_access_energy = accelergy_estimate_energy({
                "class_name": "DRAM",
                "attributes": DRAM_attributes,
                "action_name": "read",
                "arguments": arguments
            }),
            write_wordline_access_energy = accelergy_estimate_energy({
                "class_name": "DRAM",
                "attributes": DRAM_attributes,
                "action_name": "write",
                "arguments": arguments
            }),
            leakage_energy = accelergy_estimate_energy({
                "class_name": "DRAM",
                "attributes": DRAM_attributes,
                "action_name": "leak",
                "arguments": arguments
            }),
            word_bits = 64,
            value_bits = 8,
            area = accelergy_estimate_area({
                "class_name": "DRAM",
                "attributes": DRAM_attributes,
            }),
            bandwidth = 8, # operands per cycle (shared)
            factors_constraints = {},
            bypasses = []
        ),
        MemLevel(
            name = "GlobalBuffer",
            dataflow_constraints = [], #WS,
            size = 16384*8, # number of entries
            read_wordline_access_energy = accelergy_estimate_energy({
                "class_name": "SRAM",
                "attributes": SRAM_attributes,
                "action_name": "read",
                "arguments": arguments
            }),
            write_wordline_access_energy = accelergy_estimate_energy({
                "class_name": "SRAM",
                "attributes": SRAM_attributes,
                "action_name": "write",
                "arguments": arguments
            }),
            leakage_energy = accelergy_estimate_energy({
                "class_name": "SRAM",
                "attributes": SRAM_attributes,
                "action_name": "leak",
                "arguments": arguments
            }),
            word_bits = 64,
            value_bits = 8,
            area = accelergy_estimate_area({
                "class_name": "SRAM",
                "attributes": SRAM_attributes,
            }),
            bandwidth = 32, # operands per cycle (shared)
            factors_constraints = {},
            bypasses = ['w']
        ),
        FanoutLevel(
            name = "SACols",
            mesh = 14,
            dims = WS[:2],
            area = 0,
            factors_constraints = {} #{'M': 8}
        ),
        FanoutLevel(
            name = "SARows",
            mesh = 12,
            dims = WS[:1],
            area = 0,
            factors_constraints = {} #{'M': 12}
        ),
        MemLevel(
            name = "InRegister",
            dataflow_constraints = [], #WS,
            size = 12*2, # number of entries
            read_wordline_access_energy = smartbuffer_registerfile(12, 16, 8, cycle_seconds, technology, "read"),
            write_wordline_access_energy = smartbuffer_registerfile(12, 16, 8, cycle_seconds, technology, "write"),
            leakage_energy = smartbuffer_registerfile(12, 16, 8, cycle_seconds, technology, "leak"),
            word_bits = 16,
            value_bits = 8,
            area = smartbuffer_registerfile(12, 16, 8, cycle_seconds, technology, energy = False),
            bandwidth = 4, # operands per cycle (shared)
            factors_constraints = {'M': 1, 'K': 1, 'N': 1},
            bypasses = ['w', 'out']
        ),
        MemLevel(
            name = "WRegister",
            dataflow_constraints = [], #WS,
            size = 192*2, # number of entries
            read_wordline_access_energy = smartbuffer_registerfile(192, 16, 8, cycle_seconds, technology, "read"),
            write_wordline_access_energy = smartbuffer_registerfile(192, 16, 8, cycle_seconds, technology, "write"),
            leakage_energy = smartbuffer_registerfile(192, 16, 8, cycle_seconds, technology, "leak"),
            word_bits = 16,
            value_bits = 8,
            area = smartbuffer_registerfile(192, 16, 8, cycle_seconds, technology, energy = False),
            bandwidth = 4, # operands per cycle (shared)
            factors_constraints = {'M': 1, 'N': 1},
            bypasses = ['in', 'out']
        ),
        MemLevel(
            name = "OutRegister",
            dataflow_constraints = [], #WS,
            size = 16*2, # number of entries
            read_wordline_access_energy = smartbuffer_registerfile(16, 16, 16, cycle_seconds, technology, "read"),
            write_wordline_access_energy = smartbuffer_registerfile(16, 16, 16, cycle_seconds, technology, "write"),
            leakage_energy = smartbuffer_registerfile(16, 16, 16, cycle_seconds, technology, "leak"),
            word_bits = 16,
            value_bits = 16,
            area = smartbuffer_registerfile(16, 16, 16, cycle_seconds, technology, energy = False),
            bandwidth = 4, # operands per cycle (shared)
            factors_constraints = {'K': 1, 'N': 1},
            bypasses = ['in', 'w']
        ),
        ComputeLevel(
            name = "Compute",
            dim = WS[2],
            mesh = 1,
            compute_energy = accelergy_estimate_energy({
                "class_name": "aladdin_adder", # multiplier
                "attributes": {
                    "width_a": 8,
                    "width_b": 8,
                    "technology": technology,
                },
                "action_name": "read",
                "arguments": arguments
            }) + accelergy_estimate_energy({
                "class_name": "aladdin_adder", # adder
                "attributes": {
                    "width": 16,
                    "technology": technology,
                },
                "action_name": "read",
                "arguments": arguments
            }),
            leakage_energy = accelergy_estimate_energy({
                "class_name": "aladdin_adder", # multiplier
                "attributes": {
                    "width_a": 8,
                    "width_b": 8,
                    "technology": technology,
                },
                "action_name": "leak",
                "arguments": arguments
            }) + accelergy_estimate_energy({
                "class_name": "aladdin_adder", # adder
                "attributes": {
                    "width": 16,
                    "technology": technology,
                },
                "action_name": "leak",
                "arguments": arguments
            }),
            area = accelergy_estimate_area({
                "class_name": "aladdin_adder",
                "attributes": {
                    "width_a": 8,
                    "width_b": 8,
                    "technology": technology,
                }
            }) + accelergy_estimate_area({
                "class_name": "aladdin_adder",
                "attributes": {
                    "width": 16,
                    "technology": technology,
                }
            }),
            cycles = 1,
            factors_constraints = {'N': 1}
        )
    ], name="Eyeriss (Accelergy data)")
    
    print(f"\nEnergy per action in {arch.name}:")
    printEnergyPerAction(arch)
    print(f"\nArea per level in {arch.name}:")
    printAreaPerLevel(arch)
    print(f"Total area of {arch.name}: {arch.totalArea(True):.3e} um^2")
    return arch


# >>> SIMBA <<<

# C -> K
# M -> M
# P -> N
def get_arch_simba_hw_data():
    cycle_seconds = 1.2e-09
    technology = "45nm"
    arguments = { # in theory, this is not needed on anything but "leak"
        "global_cycle_seconds": cycle_seconds,
    }
    DRAM_attributes = {
        "type": "LPDDR4",
        "width": 64,
        "technology": technology,
        "cycle_seconds": cycle_seconds,
    }
    
    arch = Arch([
        MemLevel(
            name = "DRAM",
            dataflow_constraints = [],
            size = 2**64-1, # number of entries
            read_wordline_access_energy = accelergy_estimate_energy({
                "class_name": "DRAM",
                "attributes": DRAM_attributes,
                "action_name": "read",
                "arguments": arguments
            }),
            write_wordline_access_energy = accelergy_estimate_energy({
                "class_name": "DRAM",
                "attributes": DRAM_attributes,
                "action_name": "write",
                "arguments": arguments
            }),
            leakage_energy = accelergy_estimate_energy({
                "class_name": "DRAM",
                "attributes": DRAM_attributes,
                "action_name": "leak",
                "arguments": arguments
            }),
            area = accelergy_estimate_area({
                "class_name": "DRAM",
                "attributes": DRAM_attributes,
            }),
            word_bits = 64,
            value_bits = 8,
            bandwidth = 8, # operands per cycle (shared)
            factors_constraints = {},
            bypasses = []
        ),
        MemLevel(
            name = "GlobalBuffer",
            dataflow_constraints = [], #WS,
            size = 65536, # number of entries
            read_wordline_access_energy = smartbuffer_SRAM(2048, 256, 2, 4, cycle_seconds, technology, "read"),
            write_wordline_access_energy = smartbuffer_SRAM(2048, 256, 2, 4, cycle_seconds, technology, "write"),
            leakage_energy = smartbuffer_SRAM(2048, 256, 2, 4, cycle_seconds, technology, "leak"),
            word_bits = 256,
            value_bits = 8,
            area = smartbuffer_SRAM(2048, 256, 2, 4, cycle_seconds, technology, energy = False),
            bandwidth = 2**10, # operands per cycle (shared)
            factors_constraints = {},
            bypasses = ['w']
        ),
        FanoutLevel(
            name = "PEs",
            mesh = 16,
            dims = ['K', 'M'],
            area = 0,
            factors_constraints = {}
        ),
        MemLevel(
            name = "PEInputBuffer",
            dataflow_constraints = [],
            size = 65536, # number of entries
            read_wordline_access_energy = smartbuffer_registerfile(8192, 64, 8, cycle_seconds, technology, "read"),
            write_wordline_access_energy = smartbuffer_registerfile(8192, 64, 8, cycle_seconds, technology, "write"),
            leakage_energy = smartbuffer_registerfile(8192, 64, 8, cycle_seconds, technology, "leak"),
            word_bits = 64,
            value_bits = 8,
            area = smartbuffer_registerfile(8192, 64, 8, cycle_seconds, technology, energy = False),
            bandwidth = 2**10, # operands per cycle (shared)
            factors_constraints = {},
            bypasses = ['w', 'out']
        ),
        FanoutLevel(
            name = "DistributionBuffers",
            mesh = 4,
            dims = ['M'],
            area = 0,
            factors_constraints = {}
        ),
        MemLevel(
            name = "PEWeightBuffer",
            dataflow_constraints = [],
            size = 32768, # number of entries
            read_wordline_access_energy = smartbuffer_registerfile(4096, 64, 8, cycle_seconds, technology, "read"),
            write_wordline_access_energy = smartbuffer_registerfile(4096, 64, 8, cycle_seconds, technology, "write"),
            leakage_energy = smartbuffer_registerfile(4096, 64, 8, cycle_seconds, technology, "leak"),
            word_bits = 64,
            value_bits = 8,
            area = smartbuffer_registerfile(4096, 64, 8, cycle_seconds, technology, energy = False),
            bandwidth = 2**10, # operands per cycle (shared)
            factors_constraints = {},
            bypasses = ['in', 'out']
        ),
        MemLevel(
            name = "PEAccuBuffer",
            dataflow_constraints = [],
            size = 128, # number of entries
            read_wordline_access_energy = smartbuffer_registerfile(128, 24, 24, cycle_seconds, technology, "read"),
            write_wordline_access_energy = smartbuffer_registerfile(128, 24, 24, cycle_seconds, technology, "write"),
            leakage_energy = smartbuffer_registerfile(128, 24, 24, cycle_seconds, technology, "leak"),
            word_bits = 24,
            value_bits = 24,
            area = smartbuffer_registerfile(128, 24, 24, cycle_seconds, technology, energy = False),
            bandwidth = 2**10, # operands per cycle (shared)
            factors_constraints = {},
            bypasses = ['in', 'w']
        ),
        FanoutLevel(
            name = "RegMac",
            mesh = 4,
            dims = ['K'],
            area = 0,
            factors_constraints = {}
        ),
        MemLevel(
            name = "PEWeightRegs",
            dataflow_constraints = [],
            size = 1, # number of entries (64 in TL)
            read_wordline_access_energy = smartbuffer_registerfile(1, 512, 8, cycle_seconds, technology, "read"),
            write_wordline_access_energy = smartbuffer_registerfile(1, 512, 8, cycle_seconds, technology, "write"),
            leakage_energy = smartbuffer_registerfile(1, 512, 8, cycle_seconds, technology, "leak"),
            word_bits = 512,
            value_bits = 8,
            area = smartbuffer_registerfile(1, 512, 8, cycle_seconds, technology, energy = False),
            bandwidth = 2**10, # operands per cycle (shared)
            factors_constraints = {},
            bypasses = ['in', 'out']
        ),
        ComputeLevel(
            name = "Compute",
            dim = WS[2],
            mesh = 1,
            compute_energy = accelergy_estimate_energy({
                "class_name": "aladdin_adder",
                "attributes": {
                    "width_a": 8,
                    "width_b": 8,
                    "technology": technology,
                },
                "action_name": "read",
                "arguments": arguments
            }) + accelergy_estimate_energy({
                "class_name": "aladdin_adder",
                "attributes": {
                    "width": 16,
                    "technology": technology,
                },
                "action_name": "read",
                "arguments": arguments
            }),
            leakage_energy = accelergy_estimate_energy({
                "class_name": "aladdin_adder",
                "attributes": {
                    "width_a": 8,
                    "width_b": 8,
                    "technology": technology,
                },
                "action_name": "leak",
                "arguments": arguments
            }) + accelergy_estimate_energy({
                "class_name": "aladdin_adder",
                "attributes": {
                    "width": 16,
                    "technology": technology,
                },
                "action_name": "leak",
                "arguments": arguments
            }),
            area = accelergy_estimate_area({
                "class_name": "aladdin_adder",
                "attributes": {
                    "width_a": 8,
                    "width_b": 8,
                    "technology": technology,
                }
            }) + accelergy_estimate_area({
                "class_name": "aladdin_adder",
                "attributes": {
                    "width": 16,
                    "technology": technology,
                }
            }),
            cycles = 1,
            factors_constraints = {'N': 1}
        )
    ], name="Simba")
    
    print(f"\nEnergy per action in {arch.name}:")
    printEnergyPerAction(arch)
    print(f"\nArea per level in {arch.name}:")
    printAreaPerLevel(arch)
    print(f"Total area of {arch.name}: {arch.totalArea(True):.3e} um^2")
    return arch


# >>>  TPU  <<<
# > 8bit mode <

def get_arch_tpu_hw_data():
    cycle_seconds = 1.2e-09
    technology = "28nm"
    arguments = { # in theory, this is not needed on anything but "leak"
        "global_cycle_seconds": cycle_seconds,
    }
    DRAM_attributes = {
        "type": "DDR3",
        "width": 64,
        "technology": technology,
        "cycle_seconds": cycle_seconds,
    }
    unifiedbuffer_attributes = {
        "n_rw_ports": 2,
        "depth": 24*(2**16),
        "width": 128,
        "technology": technology,
        "cycle_seconds": cycle_seconds,
        "n_banks": 4,
    }
    weightsfifo_attributes = {
        "n_rw_ports": 2,
        "depth": 2**14,
        "width": 128,
        "technology": technology,
        "cycle_seconds": cycle_seconds,
        "n_banks": 4,
    }
    accumulator_attributes = {
        "n_rw_ports": 2,
        "depth": 2**12,
        "width": 32,
        "technology": technology,
        "cycle_seconds": cycle_seconds,
        "n_banks": 2,
    }
    register_attributes = {
        "n_rw_ports": 2,
        "depth": 2,
        "width": 8,
        "technology": technology,
        "cycle_seconds": cycle_seconds,
        "n_banks": 1,
    }
    
    arch = Arch([
        MemLevel(
            name = "DRAM",
            dataflow_constraints = [],
            size = 8*2**30, # number of entries
            read_wordline_access_energy = accelergy_estimate_energy({
                "class_name": "DRAM",
                "attributes": DRAM_attributes,
                "action_name": "read",
                "arguments": arguments
            }),
            write_wordline_access_energy = accelergy_estimate_energy({
                "class_name": "DRAM",
                "attributes": DRAM_attributes,
                "action_name": "write",
                "arguments": arguments
            }),
            leakage_energy = accelergy_estimate_energy({
                "class_name": "DRAM",
                "attributes": DRAM_attributes,
                "action_name": "leak",
                "arguments": arguments
            }),
            word_bits = 64,
            value_bits = 8,
            area = accelergy_estimate_area({
                "class_name": "DRAM",
                "attributes": DRAM_attributes,
            }),
            bandwidth = 8, # operands per cycle (shared)
            factors_constraints = {},
            bypasses = ['w']
        ),
        MemLevel(
            name = "WeightsDRAM",
            dataflow_constraints = [],
            size = 8*2**30, # number of entries
            read_wordline_access_energy = accelergy_estimate_energy({
                "class_name": "DRAM",
                "attributes": DRAM_attributes,
                "action_name": "read",
                "arguments": arguments
            }),
            write_wordline_access_energy = accelergy_estimate_energy({
                "class_name": "DRAM",
                "attributes": DRAM_attributes,
                "action_name": "write",
                "arguments": arguments
            }),
            leakage_energy = accelergy_estimate_energy({
                "class_name": "DRAM",
                "attributes": DRAM_attributes,
                "action_name": "leak",
                "arguments": arguments
            }),
            word_bits = 64,
            value_bits = 8,
            area = accelergy_estimate_area({
                "class_name": "DRAM",
                "attributes": DRAM_attributes,
            }),
            bandwidth = 8, # operands per cycle (shared)
            factors_constraints = {},
            bypasses = ['in', 'out']
        ),
        MemLevel(
            name = "UnifiedBuffer",
            dataflow_constraints = [],
            size = 24*(2**20), # number of entries
            read_wordline_access_energy = accelergy_estimate_energy({
                "class_name": "SRAM",
                "attributes": unifiedbuffer_attributes,
                "action_name": "read",
                "arguments": arguments
            }),
            write_wordline_access_energy = accelergy_estimate_energy({
                "class_name": "SRAM",
                "attributes": unifiedbuffer_attributes,
                "action_name": "write",
                "arguments": arguments
            }),
            leakage_energy = accelergy_estimate_energy({
                "class_name": "SRAM",
                "attributes": unifiedbuffer_attributes,
                "action_name": "leak",
                "arguments": arguments
            }),
            word_bits = 128,
            value_bits = 8,
            area = accelergy_estimate_area({
                "class_name": "SRAM",
                "attributes": unifiedbuffer_attributes,
            }),
            bandwidth = 32, # operands per cycle (shared)
            factors_constraints = {},
            # The Unified Buffer also stores outputs after the activation is
            # performed. Not modeled as we are only interested in GEMMs.
            bypasses = ['w', 'out']
        ),
        MemLevel(
            name = "WeightsFIFO",
            dataflow_constraints = [],
            size = 4*2**16, # number of entries
            read_wordline_access_energy = accelergy_estimate_energy({
                "class_name": "SRAM",
                "attributes": weightsfifo_attributes,
                "action_name": "read",
                "arguments": arguments
            }),
            write_wordline_access_energy = accelergy_estimate_energy({
                "class_name": "SRAM",
                "attributes": weightsfifo_attributes,
                "action_name": "write",
                "arguments": arguments
            }),
            leakage_energy = accelergy_estimate_energy({
                "class_name": "SRAM",
                "attributes": weightsfifo_attributes,
                "action_name": "leak",
                "arguments": arguments
            }),
            word_bits = 128,
            value_bits = 8,
            area = accelergy_estimate_area({
                "class_name": "SRAM",
                "attributes": weightsfifo_attributes,
            }),
            bandwidth = 8, # operands per cycle (shared)
            factors_constraints = {},
            bypasses = ['in', 'out']
        ),
        FanoutLevel(
            name = "SARows",
            dim = WS[0],
            mesh = 256,
            # pe_to_pe should be used, since the TPU uses a systolic array, but Timeloop
            # does not have this feature, so for sake of comparison, it is turned off
            #pe_to_pe = True,
            area = 0,
            factors_constraints = {'M': 256}
        ),
        MemLevel(
            name = "Accumulator",
            dataflow_constraints = [],
            size = 4096, # number of entries (PER ONE INSTANCE!!) (remember to account for operand size)
            read_wordline_access_energy = accelergy_estimate_energy({
                "class_name": "SRAM",
                "attributes": accumulator_attributes,
                "action_name": "read",
                "arguments": arguments
            }),
            write_wordline_access_energy = accelergy_estimate_energy({
                "class_name": "SRAM",
                "attributes": accumulator_attributes,
                "action_name": "write",
                "arguments": arguments
            }),
            leakage_energy = accelergy_estimate_energy({
                "class_name": "SRAM",
                "attributes": accumulator_attributes,
                "action_name": "leak",
                "arguments": arguments
            }),
            word_bits = 32,
            value_bits = 32,
            area = accelergy_estimate_area({
                "class_name": "SRAM",
                "attributes": accumulator_attributes,
            }),
            bandwidth = 8, # operands per cycle (shared)
            multiple_buffering = 2,
            factors_constraints = {},
            bypasses = ['in', 'w']
        ),
        FanoutLevel(
            name = "SACols",
            dim = WS[1],
            mesh = 256,
            # pe_to_pe should be used, since the TPU uses a systolic array, but Timeloop
            # does not have this feature, so for sake of comparison, it is turned off
            #pe_to_pe = True,
            area = 0,
            factors_constraints = {'K': 256}
        ),
        MemLevel(
            name = "Register",
            dataflow_constraints = WS,
            size = 2, # number of entries
            read_wordline_access_energy = accelergy_estimate_energy({
                "class_name": "SRAM",
                "attributes": register_attributes,
                "action_name": "read",
                "arguments": arguments
            }),
            write_wordline_access_energy = accelergy_estimate_energy({
                "class_name": "SRAM",
                "attributes": register_attributes,
                "action_name": "write",
                "arguments": arguments
            }),
            leakage_energy = accelergy_estimate_energy({
                "class_name": "SRAM",
                "attributes": register_attributes,
                "action_name": "leak",
                "arguments": arguments
            }),
            word_bits = 8,
            value_bits = 8,
            area = accelergy_estimate_area({
                "class_name": "SRAM",
                "attributes": register_attributes,
            }),
            bandwidth = 2, # operands per cycle (shared)
            multiple_buffering = 2,
            factors_constraints = {'M': 1, 'K': 1}, # L is free
            bypasses = ['in', 'out']
        ),
        ComputeLevel(
            name = "Compute",
            dim = WS[2],
            mesh = 1,
            compute_energy = accelergy_estimate_energy({
                "class_name": "aladdin_adder",
                "attributes": {
                    "width_a": 8,
                    "width_b": 8,
                    "technology": technology,
                },
                "action_name": "read",
                "arguments": arguments
            }) + accelergy_estimate_energy({
                "class_name": "aladdin_adder",
                "attributes": {
                    "width": 32,
                    "technology": technology,
                },
                "action_name": "read",
                "arguments": arguments
            }),
            leakage_energy = accelergy_estimate_energy({
                "class_name": "aladdin_adder",
                "attributes": {
                    "width_a": 8,
                    "width_b": 8,
                    "technology": technology,
                },
                "action_name": "leak",
                "arguments": arguments
            }) + accelergy_estimate_energy({
                "class_name": "aladdin_adder",
                "attributes": {
                    "width": 32,
                    "technology": technology,
                },
                "action_name": "leak",
                "arguments": arguments
            }),
            area = accelergy_estimate_area({
                "class_name": "aladdin_adder",
                "attributes": {
                    "width_a": 8,
                    "width_b": 8,
                    "technology": technology,
                }
            }) + accelergy_estimate_area({
                "class_name": "aladdin_adder",
                "attributes": {
                    "width": 32,
                    "technology": technology,
                }
            }),
            cycles = 1,
            factors_constraints = {'N': 1}
        )
    ], name="TPUv1")

    print(f"\nEnergy per action in {arch.name}:")
    printEnergyPerAction(arch)
    print(f"\nArea per level in {arch.name}:")
    printAreaPerLevel(arch)
    print(f"Total area of {arch.name}: {arch.totalArea(True):.3e} um^2")
    return arch