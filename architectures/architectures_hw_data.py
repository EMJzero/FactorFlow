from architectures.accelergy_hw_data import accelergy_estimate_energy
from architectures.architectures import WS, OS, IS
from levels import *
from arch import *


# >>> EYERISS <<<

# C -> K
# M -> M
# P -> N
def arch_eyeriss_hw_data():
    cycle_seconds = 1.2e-09
    technology = "32nm"
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
    }
    
    return Arch([
        MemLevel(
            name = "DRAM",
            dataflow_constraints = [], # ['N', 'M', 'K'],
            size = 2**64-1, # number of entries
            read_wordline_access_energy = accelergy_estimate_energy({
                "class_name": "DRAM",
                "attributes": DRAM_attributes,
                "action_name": "read",
                "arguments": {}
            }),
            write_wordline_access_energy = accelergy_estimate_energy({
                "class_name": "DRAM",
                "attributes": DRAM_attributes,
                "action_name": "write",
                "arguments": {}
            }),
            leakage_energy = accelergy_estimate_energy({
                "class_name": "DRAM",
                "attributes": DRAM_attributes,
                "action_name": "leak",
                "arguments": {}
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
            size = 16384*8, # number of entries
            read_wordline_access_energy = accelergy_estimate_energy({
                "class_name": "SRAM",
                "attributes": SRAM_attributes,
                "action_name": "read",
                "arguments": {}
            }),
            write_wordline_access_energy = accelergy_estimate_energy({
                "class_name": "SRAM",
                "attributes": SRAM_attributes,
                "action_name": "write",
                "arguments": {}
            }),
            leakage_energy = accelergy_estimate_energy({
                "class_name": "SRAM",
                "attributes": SRAM_attributes,
                "action_name": "leak",
                "arguments": {}
            }),
            word_bits = 64,
            value_bits = 8,
            bandwidth = 32, # operands per cycle (shared)
            factors_constraints = {},
            bypasses = ['w']
        ),
        FanoutLevel(
            name = "SACols",
            mesh = 14,
            dims = WS[:2],
            factors_constraints = {} #{'M': 8}
        ),
        FanoutLevel(
            name = "SARows",
            mesh = 12,
            # PATHOLOGICAL CASE: dims = WS[:2],
            dims = WS[:1],
            factors_constraints = {} #{'M': 12}
        ),
        MemLevel(
            name = "InRegister",
            dataflow_constraints = [], #WS,
            size = 12*2, # number of entries
            read_wordline_access_energy = accelergy_estimate_energy({
                "class_name": "SRAM",
                "attributes": SRAM_attributes,
                "action_name": "read",
                "arguments": {}
            }),
            write_wordline_access_energy = accelergy_estimate_energy({
                "class_name": "SRAM",
                "attributes": SRAM_attributes,
                "action_name": "write",
                "arguments": {}
            }),
            leakage_energy = accelergy_estimate_energy({
                "class_name": "SRAM",
                "attributes": SRAM_attributes,
                "action_name": "leak",
                "arguments": {}
            }),
            word_bits = 16,
            value_bits = 8,
            bandwidth = 4, # operands per cycle (shared)
            factors_constraints = {'M': 1, 'K': 1, 'N': 1},
            bypasses = ['w', 'out']
        ),
        MemLevel(
            name = "WRegister",
            dataflow_constraints = [], #WS,
            size = 192*2, # number of entries
            wordline_access_energy = 3.94, # per operand/scalar access (pJ)
            leakage_energy = 0.244/168, # per cycle (pJ)
            word_bits = 16,
            value_bits = 8,
            bandwidth = 4, # operands per cycle (shared)
            factors_constraints = {'M': 1, 'N': 1},
            bypasses = ['in', 'out']
        ),
        MemLevel(
            name = "OutRegister",
            dataflow_constraints = [], #WS,
            size = 16*2, # number of entries
            wordline_access_energy = 1.44, # per operand/scalar access (pJ)
            leakage_energy = 0.0815/168, # per cycle (pJ)
            word_bits = 16,
            value_bits = 16,
            bandwidth = 4, # operands per cycle (shared)
            factors_constraints = {'K': 1, 'N': 1},
            bypasses = ['in', 'w']
        ),
        ComputeLevel(
            name = "Compute",
            dataflow = WS[2],
            size = 1,
            compute_energy = accelergy_estimate_energy({
                "class_name": "aladdin_adder",
                "attributes": {
                    "width_a": 8,
                    "width_b": 8,
                    "technology": "32nm",
                },
                "action_name": "read",
                "arguments": {}
            }) + accelergy_estimate_energy({
                "class_name": "aladdin_adder",
                "attributes": {
                    "width": 16,
                    "technology": "32nm",
                },
                "action_name": "read",
                "arguments": {}
            }),
            leakage_energy = accelergy_estimate_energy({
                "class_name": "aladdin_adder",
                "attributes": {
                    "width_a": 8,
                    "width_b": 8,
                    "technology": "32nm",
                },
                "action_name": "leak",
                "arguments": {}
            }) + accelergy_estimate_energy({
                "class_name": "aladdin_adder",
                "attributes": {
                    "width": 16,
                    "technology": "32nm",
                },
                "action_name": "leak",
                "arguments": {}
            }),
            cycles = 1,
            factors_constraints = {'N': 1}
        )
    ], name="Eyeriss (Accelergy data)")

arch_eyeriss_hw_data = arch_eyeriss_hw_data()