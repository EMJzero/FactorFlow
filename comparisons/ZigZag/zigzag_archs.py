from architectures import WS, OS, IS
from levels import *

# ZigZag conversions:
# K -> D
# C -> E
# OY -> L

# >>> GEMMINI <<<
# > WS version  <

# Modified architecture with split-memories
arch_gemmini_zigzag_compatible = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = [], #['L', 'E', 'D'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = [], #WS,
        size = 320*(2**10), # number of entries
        access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {},
        bypasses = ['out', 'in']
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = [], #WS,
        size = 320*(2**10), # number of entries
        access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {},
        bypasses = ['out', 'w']
    ),
    FanoutLevel(
        name = "SARows",
        dim = WS[0],
        mesh = 16,
        # pe_to_pe should be used, since Gemmini uses a systolic array, but Timeloop
        # does not have this feature, so for sake of comparison, it is turned off
        #pe_to_pe = True, 
        factors_contraints = {'D': 16}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = [], #WS,
        size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        access_energy = 4.01, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {}, # the systolic array does a 16x16 matmul in this case
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "SACols",
        dim = WS[1],
        mesh = 16,
        # pe_to_pe should be used, since Gemmini uses a systolic array, but Timeloop
        # does not have this feature, so for sake of comparison, it is turned off
        #pe_to_pe = True, 
        factors_contraints = {'E': 16}
    ),
    MemLevel(
        name = "Register",
        dataflow_constraints = WS,
        size = 1, # number of entries
        access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 16},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dataflow = WS[2],
        size = 1,
        compute_energy = 0.28, # per compute (pJ)
        cycles = 1,
        factors_contraints = {'L': 1}
    )]

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# NOTE: unfair compared to the Gemmini architecture in "architectures.py"
#       because LOMA can "split" memories at will. Hence compare only
#       with the Gemmini version in this file!
# Comp: bert large KQV
arch_gemmini_zigzag_loma_1 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 16},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = WS,
        size = 512*(2**10), # number of entries
        access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D':16, 'E': 1, 'L': 1},
        bypasses = ['out', 'w']
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = WS,
        size = 512*(2**10), # number of entries
        access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D':12, 'E': 1, 'L': 1},
        bypasses = ['out', 'in']
    ),
    FanoutLevel(
        name = "SARows",
        dim = WS[0],
        mesh = 16,
        factors_contraints = {'D': 16}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = WS,
        size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        access_energy = 4.01, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 64, 'L': 16}, # the systolic array does a 16x16 matmul in this case
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "SACols",
        dim = WS[1],
        mesh = 16,
        factors_contraints = {'E': 16}
    ),
    MemLevel(
        name = "Register",
        dataflow_constraints = WS,
        size = 1, # number of entries
        access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared) -> 2 is a classic register which can be read & written in the same cc
        factors_contraints = {'D': 1, 'E': 1, 'L': 16},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dataflow = WS[2],
        size = 1,
        compute_energy = 0.28, # per compute (pJ)
        cycles = 1,
        factors_contraints = {'L': 1}
    )]

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: bert large VScores
arch_gemmini_zigzag_loma_2 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 512*(2**10), # number of entries
        access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D':1, 'E': 16, 'L': 16},
        bypasses = ['out', 'in']
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = WS,
        size = 512*(2**10), # number of entries
        access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D':1, 'E': 1, 'L': 1},
        bypasses = ['out', 'w']
    ),
    FanoutLevel(
        name = "SARows",
        dim = WS[0],
        mesh = 16,
        factors_contraints = {'D': 16}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = WS,
        size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        access_energy = 4.01, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 4, 'E': 16, 'L': 16}, # the systolic array does a 16x16 matmul in this case
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "SACols",
        dim = WS[1],
        mesh = 16,
        factors_contraints = {'E': 16}
    ),
    MemLevel(
        name = "Register",
        dataflow_constraints = WS,
        size = 1, # number of entries
        access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared) -> 2 is a classic register which can be read & written in the same cc
        factors_contraints = {'D': 1, 'E': 1, 'L': 16},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dataflow = WS[2],
        size = 1,
        compute_energy = 0.28, # per compute (pJ)
        cycles = 1,
        factors_contraints = {'L': 1}
    )]


# >>> EYERISS <<<

# Modified architecture with split-memories
arch_eyeriss_zigzag_compatible = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = [], # ['L', 'D', 'E'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = [], #WS,
        size = 16384*8, # number of entries
        access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        mesh = 14,
        dims = WS[:2],
        factors_contraints = {} #{'D': 8}
    ),
    FanoutLevel(
        name = "SARows",
        mesh = 12,
        # PATHOLOGICAL CASE: dims = WS[:2],
        dims = WS[0],
        factors_contraints = {} #{'D': 12}
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = [], #WS,
        size = 192*2, # number of entries
        access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'L': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = [], #WS,
        size = 16*2, # number of entries
        access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'E': 1, 'L': 1},
        bypasses = ['in', 'w']
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = [], #WS,
        size = 12*2, # number of entries
        access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1},
        bypasses = ['w', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dataflow = WS[2],
        size = 1,
        compute_energy = 0.21, # per compute (pJ)
        cycles = 1,
        factors_contraints = {'L': 1}
    )]

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# NOTE: solution obtained by constraining the fanout levels to the known best. Ignore.
# Comp: bert large KQV
arch_eyeriss_zigzag_loma_1 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['D', 'L', 'E'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 32, 'E': 16, 'L': 16},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'L', 'D'],
        size = 16384*8, # number of entries
        access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dim = 'D',
        mesh = 14,
        factors_contraints = {'D': 8}
    ),
    FanoutLevel(
        name = "SARows",
        dim = 'D', # one of D or E
        mesh = 12,
        factors_contraints = {'D': 12}
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = WS,
        size = 12*2, # number of entries
        access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w', 'out', 'in']
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['L', 'D', 'E'],
        size = 192*2, # number of entries
        access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 256},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = WS,
        size = 16*2, # number of entries
        access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 64, 'L': 1},
        bypasses = ['in', 'w']
    ),
    ComputeLevel(
        name = "Compute",
        dataflow = WS[2],
        size = 1,
        compute_energy = 0.21, # per compute (pJ)
        cycles = 1,
        factors_contraints = {'L': 1}
    )]

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# NOTE: unfair compared to the Eyeriss architecture in "architectures.py"
#       because LOMA can "split" memories at will. Hence compare only
#       with the Eyeriss version in this file!
# Comp: bert large KQV
arch_eyeriss_zigzag_loma_2 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 16, 'E': 8, 'L': 256},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'L', 'D'],
        size = 16384*8, # number of entries
        access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dim = 'E',
        mesh = 14,
        factors_contraints = {'E': 8}
    ),
    FanoutLevel(
        name = "SARows",
        dim = 'D', # one of D or E
        mesh = 12,
        factors_contraints = {'D': 12}
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['L', 'D', 'E'],
        size = 192*2, # number of entries
        access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 16, 'E': 1, 'L': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 16*2, # number of entries
        access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 16, 'L': 1},
        bypasses = ['in', 'w']
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = WS,
        size = 12*2, # number of entries
        access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 16},
        bypasses = ['w', 'out', 'in']
    ),
    ComputeLevel(
        name = "Compute",
        dataflow = WS[2],
        size = 1,
        compute_energy = 0.21, # per compute (pJ)
        cycles = 1,
        factors_contraints = {'L': 1}
    )]

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# NOTE: unfair compared to the Eyeriss architecture in "architectures.py"
#       because LOMA can "split" memories at will. Hence compare only
#       with the Eyeriss version in this file!
# Comp: bert large VScores
arch_eyeriss_zigzag_loma_3 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['D', 'E', 'L'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 2, 'E': 16, 'L': 256},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'L', 'D'],
        size = 16384*8, # number of entries
        access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dim = 'E',
        mesh = 14,
        factors_contraints = {'E': 8}
    ),
    FanoutLevel(
        name = "SARows",
        dim = 'D', # one of D or E
        mesh = 12,
        factors_contraints = {'D': 8}
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['L', 'D', 'E'],
        size = 192*2, # number of entries
        access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 16},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 16*2, # number of entries
        access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 32, 'L': 1},
        bypasses = ['in', 'w']
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = WS,
        size = 12*2, # number of entries
        access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 4, 'E': 1, 'L': 1},
        bypasses = ['w', 'out', 'in']
    ),
    ComputeLevel(
        name = "Compute",
        dataflow = WS[2],
        size = 1,
        compute_energy = 0.21, # per compute (pJ)
        cycles = 1,
        factors_contraints = {'L': 1}
    )]


# >>> SIMBA <<<

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: bert large KQV
arch_simba_zigzag_loma_1 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['D', 'L', 'E'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 12, 'E': 1, 'L': 256},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'D', 'L'], #WS,
        size = 65536, # number of entries
        access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['D'],
        factors_contraints = {'D': 16}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = WS,
        size = 65536, # number of entries
        access_energy = 30.26, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "DistributionBuffers",
        mesh = 4,
        dims = ['E'],
        factors_contraints = {'E': 4}
    ),
    MemLevel(
        name = "PEWeightBuffer",
        dataflow_constraints = ['L', 'D', 'E'],
        size = 32768, # number of entries
        access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 16, 'E': 1, 'L': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['D', 'E', 'L'],
        size = 128, # number of entries
        access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 64, 'L': 1},
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "RegMac",
        mesh = 4,
        dims = ['E'],
        factors_contraints = {'E': 4}
    ),
    MemLevel(
        name = "PEWeightRegs",
        dataflow_constraints = WS,
        size = 1, # number of entries
        access_energy = 0.70, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 16},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dataflow = WS[2],
        size = 1,
        compute_energy = 0.32, # per compute (pJ)
        cycles = 1,
        factors_contraints = {'L': 1}
    )]

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: bert large VScores
arch_simba_zigzag_loma_2 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['D', 'L', 'E'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 16},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'D', 'L'], #WS,
        size = 65536, # number of entries
        access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 256},
        bypasses = ['w', 'in']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['D'],
        factors_contraints = {'D': 16}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = WS,
        size = 65536, # number of entries
        access_energy = 30.26, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "DistributionBuffers",
        mesh = 4,
        dims = ['E'],
        factors_contraints = {'E': 4}
    ),
    MemLevel(
        name = "PEWeightBuffer",
        dataflow_constraints = ['L', 'D', 'E'],
        size = 32768, # number of entries
        access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 128, # number of entries
        access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 4, 'E': 256, 'L': 1},
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "RegMac",
        mesh = 4,
        dims = ['E'],
        factors_contraints = {'E': 4}
    ),
    MemLevel(
        name = "PEWeightRegs",
        dataflow_constraints = WS,
        size = 1, # number of entries
        access_energy = 0.70, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dataflow = WS[2],
        size = 1,
        compute_energy = 0.32, # per compute (pJ)
        cycles = 1,
        factors_contraints = {'L': 1}
    )]


# >>> TPU <<<

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: bert large KQV
arch_tpu_zigzag_loma_1 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 8*2**30, # number of entries
        access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w']
    ),
    MemLevel(
        name = "WeightsDRAM",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 8*2**30, # number of entries
        access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 4, 'E': 1, 'L': 16},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 512*(2**10), # number of entries
        access_energy = 2.69, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 3, 'E': 2, 'L': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WeightsFIFO",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 4*2**16, # number of entries
        access_energy = 2.05, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['in', 'out']
    ),
    FanoutLevel(
        name = "SARows",
        dim = WS[0],
        mesh = 256,
        factors_contraints = {'D': 256}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 4096, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        access_energy = 3.03, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_contraints = {'D': 1, 'E': 2, 'L': 256},
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "SACols",
        dim = WS[1],
        mesh = 256,
        factors_contraints = {'E': 256}
    ),
    MemLevel(
        name = "Register",
        dataflow_constraints = WS,
        size = 2, # number of entries
        access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dataflow = WS[2],
        size = 1,
        compute_energy = 0.15, # per compute (pJ)
        cycles = 1,
        factors_contraints = {'L': 1}
    )]

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: bert large VScores
arch_tpu_zigzag_loma_2 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 8*2**30, # number of entries
        access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w']
    ),
    MemLevel(
        name = "WeightsDRAM",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 8*2**30, # number of entries
        access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "WeightsFIFO",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 4*2**16, # number of entries
        access_energy = 2.05, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 4, 'L': 16},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 512*(2**10), # number of entries
        access_energy = 2.69, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "SARows",
        dim = WS[0],
        mesh = 256,
        factors_contraints = {'D': 64}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 4096, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        access_energy = 3.03, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_contraints = {'D': 1, 'E': 4, 'L': 1},
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "SACols",
        dim = WS[1],
        mesh = 256,
        factors_contraints = {'E': 256}
    ),
    MemLevel(
        name = "Register",
        dataflow_constraints = WS,
        size = 2, # number of entries
        access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_contraints = {'D': 1, 'E': 1, 'L': 256},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dataflow = WS[2],
        size = 1,
        compute_energy = 0.15, # per compute (pJ)
        cycles = 1,
        factors_contraints = {'L': 1}
    )]
