from architectures.architectures import WS, OS, IS
from levels import *


# >>> GEMMINI <<<
# > WS version  <

# SOLUTION GIVEN BY TIMELOOP:
# Comp: bert large KQV
arch_gemmini_timeloop = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 64, 'E': 2, 'L': 16},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = WS,
        size = 512*(2**10), # number of entries
        value_access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 8, 'L': 1},
        bypasses = ['out']
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
        value_access_energy = 4.01, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 3, 'E': 4, 'L': 16}, # the systolic array does a 16x16 matmul in this case
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
        value_access_energy = 0.01, # per operand (pJ)
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

# SOLUTION GIVEN BY FF:
# Comp: bert large KQV
arch_gemmini_factorflow_1 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 2**64-1, # number of entries
        value_access_energy = 512.00, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 6, 'E': 2, 'L': 8},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = WS,
        size = 512*(2**10), # number of entries
        value_access_energy = 57.04, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D':32, 'E': 1, 'L': 32},
        bypasses = ['out']
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
        value_access_energy = 4.54, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 32, 'L': 1}, # the systolic array does a 16x16 matmul in this case
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
        value_access_energy = 0.01, # per operand (pJ)
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

# SOLUTION GIVEN BY FF:
# Comp: bert large KQV
arch_gemmini_factorflow_2 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 2**64-1, # number of entries
        value_access_energy = 512.00, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 6, 'E': 16, 'L': 1},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = WS,
        size = 512*(2**10), # number of entries
        value_access_energy = 57.04, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D':32, 'E': 4, 'L': 1},
        bypasses = ['out']
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
        value_access_energy = 4.54, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 256}, # the systolic array does a 16x16 matmul in this case
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
        value_access_energy = 0.01, # per operand (pJ)
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

# SOLUTION GIVEN BY TIMELOOP:
# Comp: bert large KQV
arch_eyeriss_timeloop = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['L', 'D', 'E'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 16, 'E': 64, 'L': 16},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = WS,
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 256},
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
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = WS,
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 16, 'L': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = WS,
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 2, 'E': 1, 'L': 1},
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

# SOLUTION GIVEN BY TIMELOOP EXHAUSTIVE:
# Comp: maestro BLAS 4
arch_eyeriss_timeloop_ex_1 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['L', 'D', 'E'],
        size = 2**64-1, # number of entries
        wordline_access_energy = 512.00, # per operand/scalar access (pJ)
        leakage_energy = 0, # per cycle (pJ)
        word_bits = 64,
        value_bits = 8,
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 8, 'L': 16},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = WS,
        size = 16384*8, # number of entries
        wordline_access_energy = 18.18, # per operand/scalar access (pJ)
        leakage_energy = 0.0068, # per cycle (pJ)
        word_bits = 64,
        value_bits = 8,
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 512},
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
        dim = 'E', # one of D or E
        mesh = 12,
        factors_contraints = {'E': 8}
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = WS,
        size = 12*2, # number of entries
        wordline_access_energy = 1.44, # per operand/scalar access (pJ)
        leakage_energy = 0.0815/168, # per cycle (pJ)
        word_bits = 16,
        value_bits = 8,
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = WS,
        size = 192*2, # number of entries
        wordline_access_energy = 3.94, # per operand/scalar access (pJ)
        leakage_energy = 0.244/168, # per cycle (pJ)
        word_bits = 16,
        value_bits = 8,
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 16, 'L': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = WS,
        size = 16*2, # number of entries
        wordline_access_energy = 1.44, # per operand/scalar access (pJ)
        leakage_energy = 0.0815/168, # per cycle (pJ)
        word_bits = 16,
        value_bits = 16,
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
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

# SOLUTION GIVEN BY TIMELOOP EXHAUSTIVE:
# Comp: bert large KQV
arch_eyeriss_timeloop_ex_2 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['L', 'D', 'E'],
        size = 2**64-1, # number of entries
        wordline_access_energy = 512.00, # per operand/scalar access (pJ)
        leakage_energy = 0, # per cycle (pJ)
        word_bits = 64,
        value_bits = 8,
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 8, 'E': 8, 'L': 16},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = WS,
        size = 16384*8, # number of entries
        wordline_access_energy = 18.18, # per operand/scalar access (pJ)
        leakage_energy = 0.0068, # per cycle (pJ)
        word_bits = 64,
        value_bits = 8,
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 2, 'E': 1, 'L': 256},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dim = 'D',
        mesh = 14,
        factors_contraints = {'D': 12}
    ),
    FanoutLevel(
        name = "SARows",
        dim = 'E', # one of D or E
        mesh = 12,
        factors_contraints = {'E': 8}
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = WS,
        size = 12*2, # number of entries
        wordline_access_energy = 1.44, # per operand/scalar access (pJ)
        leakage_energy = 0.122/168, # per cycle (pJ)
        word_bits = 16,
        value_bits = 8,
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = WS,
        size = 192*2, # number of entries
        wordline_access_energy = 3.94, # per operand/scalar access (pJ)
        leakage_energy = 0.366/168, # per cycle (pJ)
        word_bits = 16,
        value_bits = 8,
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 16, 'L': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = WS,
        size = 16*2, # number of entries
        wordline_access_energy = 1.44, # per operand/scalar access (pJ)
        leakage_energy = 0.122/168, # per cycle (pJ)
        word_bits = 16,
        value_bits = 16,
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 16, 'E': 1, 'L': 1},
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

# SOLUTION GIVEN BY FF:
# Comp: bert large KQV
arch_eyeriss_factorflow_1 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['L', 'D', 'E'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 8, 'E': 8, 'L': 16},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = WS,
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 8, 'L': 256},
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
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = WS,
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 16, 'L': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = WS,
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 4, 'E': 1, 'L': 1},
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


# >>> SIMBA <<<

# SOLUTION GIVEN BY TIMELOOP:
# Comp: bert large KQV
arch_simba_timeloop = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['L', 'D', 'E'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 3, 'E': 1, 'L': 128},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'D', 'L'], #WS,
        size = 65536, # number of entries
        value_access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 16, 'E': 16, 'L': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['E', 'D'],
        factors_contraints = {'D': 2, 'E': 2}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = WS,
        size = 65536, # number of entries
        value_access_energy = 30.26, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 2, 'L': 1},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "DistributionBuffers",
        mesh = 4,
        dims = ['D'],
        factors_contraints = {'D': 2}
    ),
    MemLevel(
        name = "PEWeightBuffer",
        dataflow_constraints = ['L', 'D', 'E'],
        size = 32768, # number of entries
        value_access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 16, 'E': 2, 'L': 8},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['D', 'E', 'L'],
        size = 128, # number of entries
        value_access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 2, 'L': 2},
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
        value_access_energy = 0.70, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 2},
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

# SOLUTION GIVEN BY FF:
# Comp: bert large KQV
arch_simba_factorflow_1 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 2, 'E': 1, 'L': 256},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 65536, # number of entries
        value_access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['E', 'D'],
        factors_contraints = {'D': 8, 'E': 2,}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 65536, # number of entries
        value_access_energy = 30.26, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "DistributionBuffers",
        mesh = 4,
        dims = ['D'],
        factors_contraints = {'D': 4}
    ),
    MemLevel(
        name = "PEWeightBuffer",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 32768, # number of entries
        value_access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 6, 'E': 1, 'L': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 128, # number of entries
        value_access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 8, 'E': 128, 'L': 16},
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
        dataflow_constraints = ['E', 'L', 'D'],
        size = 1, # number of entries
        value_access_energy = 0.70, # per operand (pJ)
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

# SOLUTION GIVEN BY FF:
# Comp: bert large KQV
arch_tpu_factorflow_1 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 8*2**30, # number of entries
        value_access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w']
    ),
    MemLevel(
        name = "WeightsDRAM",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 8*2**30, # number of entries
        value_access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 32},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 24*(2**20), # number of entries
        value_access_energy = 2.69, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WeightsFIFO",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 4*2**16, # number of entries
        value_access_energy = 2.12, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['in', 'out']
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
        value_access_energy = 3.08, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_contraints = {'D': 1, 'E': 16, 'L': 128},
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
        value_access_energy = 0.01, # per operand (pJ)
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

# SOLUTION GIVEN BY TIMELOOP EXHAUSTIVE:
# Comp: maestro BLAS 6
arch_tpu_timeloop_ex = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 8*2**30, # number of entries
        wordline_access_energy = 4480.00, # per operand/scalar access (pJ)
        leakage_energy = 0, # per cycle (pJ)
        word_bits = 64,
        value_bits = 8,
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w']
    ),
    MemLevel(
        name = "WeightsDRAM",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 8*2**30, # number of entries
        wordline_access_energy = 4480.00, # per operand/scalar access (pJ)
        leakage_energy = 0, # per cycle (pJ)
        word_bits = 64,
        value_bits = 8,
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 24*(2**20), # number of entries
        wordline_access_energy = 315.87, # per operand (pJ)
        leakage_energy = 2.31, # per cycle (pJ)
        word_bits = 128,
        value_bits = 8,
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WeightsFIFO",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 4*2**16, # number of entries
        wordline_access_energy = 33.98, # per operand/scalar access (pJ)
        leakage_energy = 0.03, # per cycle (pJ)
        word_bits = 128,
        value_bits = 8,
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
        wordline_access_energy = 3.54, # per operand (pJ)
        leakage_energy = 0.51/256, # per cycle (pJ)
        word_bits = 32,
        value_bits = 32,
        bandwidth = 8, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_contraints = {'D': 2, 'E': 1, 'L': 1},
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
        wordline_access_energy = 0.01, # per operand (pJ)
        leakage_energy = 0, # per cycle (pJ)
        word_bits = 8,
        value_bits = 8,
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