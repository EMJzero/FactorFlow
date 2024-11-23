from architectures.architectures import WS, OS, IS
from computations import gemm_coupling, conv_coupling
from levels import *
from arch import *


# >>> GEMMINI <<<
# > WS version  <

# SOLUTION GIVEN BY TIMELOOP:
# Comp: bert large KQV
arch_gemmini_timeloop = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 64, 'K': 2, 'N': 16},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = WS,
        size = 512*(2**10), # number of entries
        value_access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 8, 'N': 1},
        bypasses = ['out']
    ),
    FanoutLevel(
        name = "SARows",
        dim = WS[0],
        mesh = 16,
        factors_constraints = {'M': 16}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = WS,
        size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        value_access_energy = 4.01, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 3, 'K': 4, 'N': 16}, # the systolic array does a 16x16 matmul in this case
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "SACols",
        dim = WS[1],
        mesh = 16,
        factors_constraints = {'K': 16}
    ),
    MemLevel(
        name = "Register",
        dataflow_constraints = WS,
        size = 1, # number of entries
        value_access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared) -> 2 is a classic register which can be read & written in the same cc
        factors_constraints = {'M': 1, 'K': 1, 'N': 16},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.28, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY FF:
# Comp: bert large KQV
arch_gemmini_factorflow_1 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 2**64-1, # number of entries
        value_access_energy = 512.00, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 6, 'K': 2, 'N': 8},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = WS,
        size = 512*(2**10), # number of entries
        value_access_energy = 57.04, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M':32, 'K': 1, 'N': 32},
        bypasses = ['out']
    ),
    FanoutLevel(
        name = "SARows",
        dim = WS[0],
        mesh = 16,
        factors_constraints = {'M': 16}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = WS,
        size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        value_access_energy = 4.54, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 32, 'N': 1}, # the systolic array does a 16x16 matmul in this case
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "SACols",
        dim = WS[1],
        mesh = 16,
        factors_constraints = {'K': 16}
    ),
    MemLevel(
        name = "Register",
        dataflow_constraints = WS,
        size = 1, # number of entries
        value_access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared) -> 2 is a classic register which can be read & written in the same cc
        factors_constraints = {'M': 1, 'K': 1, 'N': 16},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.28, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY FF:
# Comp: bert large KQV
arch_gemmini_factorflow_2 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 2**64-1, # number of entries
        value_access_energy = 512.00, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 6, 'K': 16, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = WS,
        size = 512*(2**10), # number of entries
        value_access_energy = 57.04, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M':32, 'K': 4, 'N': 1},
        bypasses = ['out']
    ),
    FanoutLevel(
        name = "SARows",
        dim = WS[0],
        mesh = 16,
        factors_constraints = {'M': 16}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = WS,
        size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        value_access_energy = 4.54, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 256}, # the systolic array does a 16x16 matmul in this case
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "SACols",
        dim = WS[1],
        mesh = 16,
        factors_constraints = {'K': 16}
    ),
    MemLevel(
        name = "Register",
        dataflow_constraints = WS,
        size = 1, # number of entries
        value_access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared) -> 2 is a classic register which can be read & written in the same cc
        factors_constraints = {'M': 1, 'K': 1, 'N': 16},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.28, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )], coupling=gemm_coupling)


# >>> EYERISS <<<

# SOLUTION GIVEN BY TIMELOOP:
# Comp: bert large KQV
arch_eyeriss_timeloop = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['N', 'M', 'K'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 16, 'K': 64, 'N': 16},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = WS,
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 256},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dim = 'M',
        mesh = 14,
        factors_constraints = {'M': 8}
    ),
    FanoutLevel(
        name = "SARows",
        dim = 'M', # one of M or K
        mesh = 12,
        factors_constraints = {'M': 12}
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = WS,
        size = 12*2, # number of entries
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = WS,
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 16, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = WS,
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 2, 'K': 1, 'N': 1},
        bypasses = ['in', 'w']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.21, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY TIMELOOP EXHAUSTIVE:
# Comp: maestro BLAS 4
arch_eyeriss_timeloop_ex_1 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['N', 'M', 'K'],
        size = 2**64-1, # number of entries
        wordline_access_energy = 512.00, # per operand/scalar access (pJ)
        leakage_energy = 0, # per cycle (pJ)
        word_bits = 64,
        value_bits = 8,
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 8, 'N': 16},
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
        factors_constraints = {'M': 1, 'K': 1, 'N': 512},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dim = 'M',
        mesh = 14,
        factors_constraints = {'M': 8}
    ),
    FanoutLevel(
        name = "SARows",
        dim = 'K', # one of M or K
        mesh = 12,
        factors_constraints = {'K': 8}
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
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
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
        factors_constraints = {'M': 1, 'K': 16, 'N': 1},
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
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'w']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.21, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY TIMELOOP EXHAUSTIVE:
# Comp: bert large KQV
arch_eyeriss_timeloop_ex_2 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['N', 'M', 'K'],
        size = 2**64-1, # number of entries
        wordline_access_energy = 512.00, # per operand/scalar access (pJ)
        leakage_energy = 0, # per cycle (pJ)
        word_bits = 64,
        value_bits = 8,
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 8, 'K': 8, 'N': 16},
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
        factors_constraints = {'M': 2, 'K': 1, 'N': 256},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dim = 'M',
        mesh = 14,
        factors_constraints = {'M': 12}
    ),
    FanoutLevel(
        name = "SARows",
        dim = 'K', # one of M or K
        mesh = 12,
        factors_constraints = {'K': 8}
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
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
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
        factors_constraints = {'M': 1, 'K': 16, 'N': 1},
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
        factors_constraints = {'M': 16, 'K': 1, 'N': 1},
        bypasses = ['in', 'w']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.21, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY FF:
# Comp: bert large KQV
arch_eyeriss_factorflow_1 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['N', 'M', 'K'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 8, 'K': 8, 'N': 16},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = WS,
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 8, 'N': 256},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dim = 'M',
        mesh = 14,
        factors_constraints = {'M': 8}
    ),
    FanoutLevel(
        name = "SARows",
        dim = 'M', # one of M or K
        mesh = 12,
        factors_constraints = {'M': 12}
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = WS,
        size = 12*2, # number of entries
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = WS,
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 16, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = WS,
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 4, 'K': 1, 'N': 1},
        bypasses = ['in', 'w']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.21, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )], coupling=gemm_coupling)


# >>> SIMBA <<<

# SOLUTION GIVEN BY TIMELOOP:
# Comp: bert large KQV
arch_simba_timeloop = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['N', 'M', 'K'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 3, 'K': 1, 'N': 128},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'M', 'N'], #WS,
        size = 65536, # number of entries
        value_access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 16, 'K': 16, 'N': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['K', 'M'],
        factors_constraints = {'M': 2, 'K': 2}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = WS,
        size = 65536, # number of entries
        value_access_energy = 30.26, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 2, 'N': 1},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "DistributionBuffers",
        mesh = 4,
        dims = ['M'],
        factors_constraints = {'M': 2}
    ),
    MemLevel(
        name = "PEWeightBuffer",
        dataflow_constraints = ['N', 'M', 'K'],
        size = 32768, # number of entries
        value_access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 16, 'K': 2, 'N': 8},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 128, # number of entries
        value_access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 2, 'N': 2},
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "RegMac",
        mesh = 4,
        dims = ['K'],
        factors_constraints = {'K': 4}
    ),
    MemLevel(
        name = "PEWeightRegs",
        dataflow_constraints = WS,
        size = 1, # number of entries
        value_access_energy = 0.70, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 2},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.32, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY FF:
# Comp: bert large KQV
arch_simba_factorflow_1 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 2, 'K': 1, 'N': 256},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 65536, # number of entries
        value_access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['K', 'M'],
        factors_constraints = {'M': 8, 'K': 2,}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 65536, # number of entries
        value_access_energy = 30.26, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "DistributionBuffers",
        mesh = 4,
        dims = ['M'],
        factors_constraints = {'M': 4}
    ),
    MemLevel(
        name = "PEWeightBuffer",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 32768, # number of entries
        value_access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 6, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 128, # number of entries
        value_access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 8, 'K': 128, 'N': 16},
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "RegMac",
        mesh = 4,
        dims = ['K'],
        factors_constraints = {'K': 4}
    ),
    MemLevel(
        name = "PEWeightRegs",
        dataflow_constraints = ['K', 'N', 'M'],
        size = 1, # number of entries
        value_access_energy = 0.70, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.32, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )], coupling=gemm_coupling)


# >>> TPU <<<

# SOLUTION GIVEN BY FF:
# Comp: bert large KQV
arch_tpu_factorflow_1 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 8*2**30, # number of entries
        value_access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w']
    ),
    MemLevel(
        name = "WeightsDRAM",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 8*2**30, # number of entries
        value_access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 32},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 24*(2**20), # number of entries
        value_access_energy = 2.69, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WeightsFIFO",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 4*2**16, # number of entries
        value_access_energy = 2.12, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    FanoutLevel(
        name = "SARows",
        dim = WS[0],
        mesh = 256,
        factors_constraints = {'M': 64}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 4096, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        value_access_energy = 3.08, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_constraints = {'M': 1, 'K': 16, 'N': 128},
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "SACols",
        dim = WS[1],
        mesh = 256,
        factors_constraints = {'K': 256}
    ),
    MemLevel(
        name = "Register",
        dataflow_constraints = WS,
        size = 2, # number of entries
        value_access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.15, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY TIMELOOP EXHAUSTIVE:
# Comp: maestro BLAS 6
arch_tpu_timeloop_ex = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 8*2**30, # number of entries
        wordline_access_energy = 4480.00, # per operand/scalar access (pJ)
        leakage_energy = 0, # per cycle (pJ)
        word_bits = 64,
        value_bits = 8,
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w']
    ),
    MemLevel(
        name = "WeightsDRAM",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 8*2**30, # number of entries
        wordline_access_energy = 4480.00, # per operand/scalar access (pJ)
        leakage_energy = 0, # per cycle (pJ)
        word_bits = 64,
        value_bits = 8,
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 24*(2**20), # number of entries
        wordline_access_energy = 315.87, # per operand (pJ)
        leakage_energy = 2.31, # per cycle (pJ)
        word_bits = 128,
        value_bits = 8,
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WeightsFIFO",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 4*2**16, # number of entries
        wordline_access_energy = 33.98, # per operand/scalar access (pJ)
        leakage_energy = 0.03, # per cycle (pJ)
        word_bits = 128,
        value_bits = 8,
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    FanoutLevel(
        name = "SARows",
        dim = WS[0],
        mesh = 256,
        factors_constraints = {'M': 256}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 4096, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        wordline_access_energy = 3.54, # per operand (pJ)
        leakage_energy = 0.51/256, # per cycle (pJ)
        word_bits = 32,
        value_bits = 32,
        bandwidth = 8, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_constraints = {'M': 2, 'K': 1, 'N': 1},
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "SACols",
        dim = WS[1],
        mesh = 256,
        factors_constraints = {'K': 256}
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
        factors_constraints = {'M': 1, 'K': 1, 'N': 256},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.15, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY TIMELOOP:
# Comp: VGG16 L5
arch_eyeriss_conv_timeloop_1 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['Q', 'P', 'M', 'C', 'R', 'S'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'Q': 4, 'P': 14, 'M': 8},
        multiple_reuses = False,
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['S', 'M', 'C', 'P', 'Q', 'R'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'S': 3, 'M': 2, 'C': 64, 'P': 4},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        mesh = 14,
        dims = ['Q', 'M'],
        factors_constraints = {'Q': 14}
    ),
    FanoutLevel(
        name = "SARows",
        mesh = 12,
        dims = ['S', 'C', 'M'],
        factors_constraints = {'M': 2, 'C': 4}
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = ['M', 'C', 'P', 'Q', 'R', 'S'],
        size = 12*2, # number of entries
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'C': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['M', 'P', 'Q', 'S', 'C', 'R'],
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'P': 1, 'Q': 1, 'S': 1, 'R': 3},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = ['C', 'P', 'Q', 'R', 'S', 'M'],
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'C': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': 1, 'M': 8},
        bypasses = ['in', 'w']
    ),
    ComputeLevel(
        name = "Compute",
        mesh = 1,
        compute_energy = 0.21, # per compute (pJ)
        cycles = 1,
    )], coupling=conv_coupling)

# SOLUTION GIVEN BY TIMELOOP:
# Comp: VGG16 L5 (ver 2)
arch_eyeriss_conv_timeloop_2 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['C', 'Q', 'M', 'P', 'R', 'S'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'C': 4, 'Q': 4, 'P': 8, 'M': 16},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['M', 'C', 'P', 'Q', 'R', 'S'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 4, 'C': 4, 'P': 7},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        mesh = 14,
        dims = ['Q', 'M'],
        factors_constraints = {'Q': 14}
    ),
    FanoutLevel(
        name = "SARows",
        mesh = 12,
        dims = ['S', 'C', 'M'],
        factors_constraints = {'M': 2, 'C': 2, 'S': 3}
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = ['M', 'C', 'P', 'Q', 'R', 'S'],
        size = 12*2, # number of entries
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'C': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['M', 'P', 'Q', 'S', 'C', 'R'],
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'P': 1, 'Q': 1, 'S': 1, 'C': 8, 'R': 3},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = ['C', 'P', 'Q', 'R', 'S', 'M'],
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'C': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': 1, 'M': 2},
        bypasses = ['in', 'w']
    ),
    ComputeLevel(
        name = "Compute",
        mesh = 1,
        compute_energy = 0.21, # per compute (pJ)
        cycles = 1,
    )], coupling=conv_coupling)

# SOLUTION GIVEN BY TIMELOOP:
# Comp: VGG16 L3+
arch_eyeriss_conv_timeloop_3 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['Q', 'C', 'M', 'P', 'R', 'S'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'Q': 7, 'C': 8, 'M': 16},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['S', 'M', 'C', 'Q', 'P', 'R'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'S': 3, 'M': 2, 'C': 16, 'Q': 2, 'P': 112},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        mesh = 14,
        dims = ['Q', 'M'],
        factors_constraints = {'Q': 8}
    ),
    FanoutLevel(
        name = "SARows",
        mesh = 12,
        dims = ['S', 'C', 'M'],
        factors_constraints = {'M': 4, 'S': 3}
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = ['M', 'C', 'P', 'Q', 'R', 'S'],
        size = 12*2, # number of entries
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'C': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['M', 'P', 'Q', 'S', 'C', 'R'],
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'P': 1, 'Q': 1, 'S': 1, 'R': 9},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = ['C', 'P', 'Q', 'R', 'S', 'M'],
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'C': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': 1, 'M': 1},
        bypasses = ['in', 'w']
    ),
    ComputeLevel(
        name = "Compute",
        mesh = 1,
        compute_energy = 0.21, # per compute (pJ)
        cycles = 1,
    )], coupling=conv_coupling)

# CUSTOM TEST FOR ADDITIONAL REUSE OPPORTUNITIES:
# Comp: VGG16 L3+
arch_eyeriss_conv_additional_reuse = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['C', 'M', 'P', 'R', 'S', 'Q'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'C': 8, 'M': 16, 'Q': 7},
        multiple_reuses = True,
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['S', 'C', 'Q', 'P', 'M', 'R'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'S': 3, 'C': 16, 'Q': 2, 'P': 112, 'M': 2},
        multiple_reuses = True,
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        mesh = 14,
        dims = ['Q', 'M'],
        factors_constraints = {'Q': 8},
        selective_multicast_support = True,
        selective_reduction_support = True
    ),
    FanoutLevel(
        name = "SARows",
        mesh = 12,
        dims = ['S', 'C', 'M'],
        factors_constraints = {'M': 4, 'S': 3},
        selective_multicast_support = True,
        selective_reduction_support = True
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = ['M', 'C', 'P', 'Q', 'R', 'S'],
        size = 12*2, # number of entries
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'C': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['M', 'P', 'Q', 'S', 'C', 'R'],
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'P': 1, 'Q': 1, 'S': 1, 'R': 9},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = ['C', 'P', 'Q', 'R', 'S', 'M'],
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'C': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': 1, 'M': 1},
        bypasses = ['in', 'w']
    ),
    ComputeLevel(
        name = "Compute",
        mesh = 1,
        compute_energy = 0.21, # per compute (pJ)
        cycles = 1,
    )], coupling=conv_coupling)

# CUSTOM TEST FOR INPUT BYPASS:
# Comp: VGG16 L3+
arch_eyeriss_conv_input_bypass = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['Q', 'M', 'C', 'R', 'S', 'P'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'C': 8, 'M': 16, 'Q': 7},
        multiple_reuses = False,
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['S', 'M', 'C', 'Q', 'P', 'R'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'S': 3, 'M': 2, 'C': 16, 'Q': 2, 'P': 112},
        multiple_reuses = False,
        bypasses = ['in']
    ),
    FanoutLevel(
        name = "SACols",
        mesh = 14,
        dims = ['Q', 'M'],
        factors_constraints = {'Q': 8},
    ),
    FanoutLevel(
        name = "SARows",
        mesh = 12,
        dims = ['S', 'C', 'M'],
        factors_constraints = {'M': 4, 'S': 3},
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = ['M', 'C', 'P', 'Q', 'R', 'S'],
        size = 12*2, # number of entries
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'C': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['M', 'P', 'Q', 'S', 'C', 'R'],
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'P': 1, 'Q': 1, 'S': 1, 'R': 9},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = ['C', 'P', 'Q', 'R', 'S', 'M'],
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'C': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': 1, 'M': 1},
        bypasses = ['in', 'w']
    ),
    ComputeLevel(
        name = "Compute",
        mesh = 1,
        compute_energy = 0.21, # per compute (pJ)
        cycles = 1,
    )], coupling=conv_coupling)