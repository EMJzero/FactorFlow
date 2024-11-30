from architectures.architectures import WS, OS, IS
from levels import *
from arch import *

# CoSA conversions:
# C -> K
# K -> M
# P -> N

# >>> GEMMINI <<<
# > WS version  <

# SOLUTION GIVEN BY CoSA:
# Comp: bert large KQV
arch_gemmini_cosa_kqv = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 16, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 512*(2**10), # number of entries
        value_access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 192, 'K': 1, 'N': 1},
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
        dataflow_constraints = ['M', 'K', 'N'],
        size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        value_access_energy = 4.01, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 4, 'N': 1}, # the systolic array does a 16x16 matmul in this case
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
        dataflow_constraints = ['M', 'K', 'N'],
        size = 1, # number of entries
        value_access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 4096},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.28, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )])

# SOLUTION GIVEN BY CoSA:
# Comp: bert large KTQ
arch_gemmini_cosa_ktq = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 512*(2**10), # number of entries
        value_access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 256, 'K': 1, 'N': 1},
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
        dataflow_constraints = ['M', 'K', 'N'],
        size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        value_access_energy = 4.01, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 4, 'N': 1}, # the systolic array does a 16x16 matmul in this case
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
        dataflow_constraints = ['M', 'K', 'N'],
        size = 1, # number of entries
        value_access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 4096},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.28, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )])

# SOLUTION GIVEN BY CoSA:
# Comp: bert large VScores
arch_gemmini_cosa_vscores = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 16, 'N': 4},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 512*(2**10), # number of entries
        value_access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
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
        dataflow_constraints = ['M', 'K', 'N'],
        size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        value_access_energy = 4.01, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 4, 'K': 16, 'N': 1}, # the systolic array does a 16x16 matmul in this case
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
        dataflow_constraints = ['M', 'K', 'N'],
        size = 1, # number of entries
        value_access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1024},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.28, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )])

# SOLUTION GIVEN BY CoSA:
# Comp: bert large FF1
arch_gemmini_cosa_ff1 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 16, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 512*(2**10), # number of entries
        value_access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 256, 'K': 1, 'N': 1},
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
        dataflow_constraints = ['M', 'K', 'N'],
        size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        value_access_energy = 4.01, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 4, 'N': 1}, # the systolic array does a 16x16 matmul in this case
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
        dataflow_constraints = ['M', 'K', 'N'],
        size = 1, # number of entries
        value_access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 4096},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.28, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )])

# SOLUTION GIVEN BY CoSA:
# Comp: MAESTRO-BLAS 1
arch_gemmini_cosa_mb1 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 256, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 512*(2**10), # number of entries
        value_access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1024, 'K': 1, 'N': 1},
        bypasses = ['out']
    ),
    FanoutLevel(
        name = "SARows",
        dims = ['M', 'N'],
        mesh = 16,
        factors_constraints = {'M': 8, 'N': 2}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = ['M', 'K', 'N'],
        size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        value_access_energy = 4.01, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 2, 'N': 1}, # the systolic array does a 16x16 matmul in this case
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
        dataflow_constraints = ['M', 'K', 'N'],
        size = 1, # number of entries
        value_access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 4096},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.28, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )])

# SOLUTION GIVEN BY CoSA:
# Comp: MAESTRO-BLAS 2
arch_gemmini_cosa_mb2 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 32, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 512*(2**10), # number of entries
        value_access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 16, 'K': 1, 'N': 1},
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
        dataflow_constraints = ['M', 'K', 'N'],
        size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        value_access_energy = 4.01, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 2, 'K': 32, 'N': 1}, # the systolic array does a 16x16 matmul in this case
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['M', 'K'],
        mesh = 16,
        factors_constraints = {'M': 2, 'K': 8}
    ),
    MemLevel(
        name = "Register",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 1, # number of entries
        value_access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1024},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.28, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )])

# SOLUTION GIVEN BY CoSA:
# Comp: MAESTRO-BLAS 3
arch_gemmini_cosa_mb3 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 512*(2**10), # number of entries
        value_access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['out']
    ),
    FanoutLevel(
        name = "SARows",
        dim = WS[1],
        mesh = 16,
        factors_constraints = {'K': 16}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = ['M', 'K', 'N'],
        size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        value_access_energy = 4.01, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 2, 'K': 128, 'N': 1}, # the systolic array does a 16x16 matmul in this case
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['M', 'K'],
        mesh = 16,
        factors_constraints = {'K': 4, 'M': 4}
    ),
    MemLevel(
        name = "Register",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 1, # number of entries
        value_access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 8},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.28, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )])

# SOLUTION GIVEN BY CoSA:
# Comp: MAESTRO-BLAS 4
arch_gemmini_cosa_mb4 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 32, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 512*(2**10), # number of entries
        value_access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['out']
    ),
    FanoutLevel(
        name = "SARows",
        dims = ['M', 'N'],
        mesh = 16,
        factors_constraints = {'M': 8, 'N': 2}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = ['M', 'K', 'N'],
        size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        value_access_energy = 4.01, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 2, 'N': 1}, # the systolic array does a 16x16 matmul in this case
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
        dataflow_constraints = ['M', 'K', 'N'],
        size = 1, # number of entries
        value_access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 4096},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.28, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )])

# SOLUTION GIVEN BY CoSA:
# Comp: MAESTRO-BLAS 5
arch_gemmini_cosa_mb5 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 2, 'K': 16, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 512*(2**10), # number of entries
        value_access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['out']
    ),
    FanoutLevel(
        name = "SARows",
        dims = ['M', 'K'],
        mesh = 16,
        factors_constraints = {'M': 8, 'K': 2}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = ['M', 'K', 'N'],
        size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        value_access_energy = 4.01, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 32, 'K': 32, 'N': 1}, # the systolic array does a 16x16 matmul in this case
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "SACols",
        dim = WS[0],
        mesh = 16,
        factors_constraints = {'M': 16}
    ),
    MemLevel(
        name = "Register",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 1, # number of entries
        value_access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 8},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.28, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )])

# SOLUTION GIVEN BY CoSA:
# Comp: MAESTRO-BLAS 6
arch_gemmini_cosa_mb6 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 2, 'K': 1, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 512*(2**10), # number of entries
        value_access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
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
        dataflow_constraints = ['M', 'K', 'N'],
        size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        value_access_energy = 4.01, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 2, 'K': 128, 'N': 1}, # the systolic array does a 16x16 matmul in this case
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['M', 'K'],
        mesh = 16,
        factors_constraints = {'M': 8, 'K': 2}
    ),
    MemLevel(
        name = "Register",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 1, # number of entries
        value_access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 256},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.28, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )])


# >>> EYERISS <<<

# SOLUTION GIVEN BY CoSA:
# Comp: bert large KQV
arch_eyeriss_cosa_kqv = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 128},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 16, 'K': 1024, 'N': 4},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dim = 'N',
        mesh = 14,
        factors_constraints = {'N': 8}
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
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = WS,
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
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
    )])

# SOLUTION GIVEN BY CoSA:
# Comp: bert large KTQ
arch_eyeriss_cosa_ktq = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 256},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['K'],
        mesh = 14,
        factors_constraints = {'K': 8}
    ),
    FanoutLevel(
        name = "SARows",
        dims = ['K'],
        mesh = 12,
        factors_constraints = {'K': 8}
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = WS,
        size = 12*2, # number of entries
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 4096, 'K': 1, 'N': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = WS,
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 16},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = WS,
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
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
    )])

# SOLUTION GIVEN BY CoSA:
# Comp: bert large VScores
arch_eyeriss_cosa_vscores = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 512},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 4096, 'N': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dim = 'N',
        mesh = 14,
        factors_constraints = {'N': 8}
    ),
    FanoutLevel(
        name = "SARows",
        dim = 'M', # one of M or K
        mesh = 12,
        factors_constraints = {'M': 8}
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
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = WS,
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 8, 'K': 1, 'N': 1},
        bypasses = ['in', 'w']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.21, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )])

# SOLUTION GIVEN BY CoSA:
# Comp: bert large FF1
arch_eyeriss_cosa_ff1 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 256},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 16, 'N': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['K'],
        mesh = 14,
        factors_constraints = {'K': 8}
    ),
    FanoutLevel(
        name = "SARows",
        dims = ['K'],
        mesh = 12,
        factors_constraints = {'K': 8}
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = WS,
        size = 12*2, # number of entries
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 4096, 'K': 1, 'N': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = WS,
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 16},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = WS,
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
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
    )])

# SOLUTION GIVEN BY CoSA:
# Comp: MAESTRO-BLAS 1
arch_eyeriss_cosa_mb1 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1024},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 64, 'N': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['K'],
        mesh = 14,
        factors_constraints = {'K': 8}
    ),
    FanoutLevel(
        name = "SARows",
        dims = ['K'],
        mesh = 12,
        factors_constraints = {'K': 8}
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = WS,
        size = 12*2, # number of entries
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 8192, 'K': 2, 'N': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = WS,
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 8},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = WS,
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
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
    )])

# SOLUTION GIVEN BY CoSA:
# Comp: MAESTRO-BLAS 2
arch_eyeriss_cosa_mb2 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 8, 'N': 16},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 64, 'N': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['K', 'N'],
        mesh = 14,
        factors_constraints = {'K': 2, 'N': 4}
    ),
    FanoutLevel(
        name = "SARows",
        dims = ['K'],
        mesh = 12,
        factors_constraints = {'K': 8}
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = WS,
        size = 12*2, # number of entries
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1024, 'K': 1, 'N': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = WS,
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 16},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = WS,
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
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
    )])

# SOLUTION GIVEN BY CoSA:
# Comp: MAESTRO-BLAS 3
arch_eyeriss_cosa_mb3 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 64, 'N': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['K'],
        mesh = 14,
        factors_constraints = {'K': 8}
    ),
    FanoutLevel(
        name = "SARows",
        dims = ['K'],
        mesh = 12,
        factors_constraints = {'K': 8}
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = WS,
        size = 12*2, # number of entries
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 8, 'K': 2, 'N': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = WS,
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 8},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = WS,
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
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
    )])

# SOLUTION GIVEN BY CoSA:
# Comp: MAESTRO-BLAS 4
arch_eyeriss_cosa_mb4 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 2, 'N': 64},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 32, 'N': 2},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['K', 'N'],
        mesh = 14,
        factors_constraints = {'K': 2, 'N': 4}
    ),
    FanoutLevel(
        name = "SARows",
        dims = ['K'],
        mesh = 12,
        factors_constraints = {'K': 8}
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = WS,
        size = 12*2, # number of entries
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 8, 'K': 1, 'N': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = WS,
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 16},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = WS,
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
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
    )])

# SOLUTION GIVEN BY CoSA:
# Comp: MAESTRO-BLAS 5
arch_eyeriss_cosa_mb5 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 8, 'N': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['K'],
        mesh = 14,
        factors_constraints = {'K': 8}
    ),
    FanoutLevel(
        name = "SARows",
        dims = ['K'],
        mesh = 12,
        factors_constraints = {'K': 8}
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = WS,
        size = 12*2, # number of entries
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 8192, 'K': 2, 'N': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = WS,
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 8},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = WS,
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
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
    )])

# SOLUTION GIVEN BY CoSA:
# Comp: MAESTRO-BLAS 6
arch_eyeriss_cosa_mb6 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 2},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 32, 'N': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['K', 'N'],
        mesh = 14,
        factors_constraints = {'K': 2, 'N': 4}
    ),
    FanoutLevel(
        name = "SARows",
        dims = ['K', 'N'],
        mesh = 12,
        factors_constraints = {'K': 4, 'N': 2}
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = WS,
        size = 12*2, # number of entries
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 512, 'K': 1, 'N': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = WS,
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 16},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = WS,
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
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
    )])


# >>> SIMBA <<<

# SOLUTION GIVEN BY CoSA:
# Comp: bert large KQV
arch_simba_cosa_kqv = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 256},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['M', 'K', 'N'], #WS,
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
        factors_constraints = {'M': 16, 'K': 1}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 65536, # number of entries
        value_access_energy = 30.26, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 2, 'K': 1, 'N': 1},
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
        dataflow_constraints = ['M', 'K', 'N'],
        size = 32768, # number of entries
        value_access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 3, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 128, # number of entries
        value_access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 4, 'K': 512, 'N': 1},
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "RegMac",
        mesh = 4,
        dims = ['M', 'K'],
        factors_constraints = {'M': 2, 'K': 2}
    ),
    MemLevel(
        name = "PEWeightRegs",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 1, # number of entries
        value_access_energy = 0.70, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 16},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.32, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )])

# SOLUTION GIVEN BY CoSA:
# Comp: bert large KTQ
arch_simba_cosa_ktq = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 32, 'K': 1, 'N': 16},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['M', 'K', 'N'], #WS,
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
        factors_constraints = {'M': 16, 'K': 1}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 65536, # number of entries
        value_access_energy = 30.26, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "DistributionBuffers",
        mesh = 4,
        dims = ['N'],
        factors_constraints = {'N': 4}
    ),
    MemLevel(
        name = "PEWeightBuffer",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 32768, # number of entries
        value_access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 4, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 128, # number of entries
        value_access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 2, 'K': 16, 'N': 1},
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "RegMac",
        mesh = 4,
        dims = ['M', 'K'],
        factors_constraints = {'M': 1, 'K': 4}
    ),
    MemLevel(
        name = "PEWeightRegs",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 1, # number of entries
        value_access_energy = 0.70, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 64},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.32, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )])

# SOLUTION GIVEN BY CoSA:
# Comp: bert large VScores
arch_simba_cosa_vscores = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 8, 'N': 128},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['M', 'K', 'N'], #WS,
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
        factors_constraints = {'M': 16, 'K': 1}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 65536, # number of entries
        value_access_energy = 30.26, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "DistributionBuffers",
        mesh = 4,
        dims = ['K'],
        factors_constraints = {'K': 4}
    ),
    MemLevel(
        name = "PEWeightBuffer",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 32768, # number of entries
        value_access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 128, # number of entries
        value_access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 128, 'N': 1},
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "RegMac",
        mesh = 4,
        dims = ['M', 'K'],
        factors_constraints = {'M': 4, 'K': 1}
    ),
    MemLevel(
        name = "PEWeightRegs",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 1, # number of entries
        value_access_energy = 0.70, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 32},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.32, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )])

# SOLUTION GIVEN BY CoSA:
# Comp: bert large FF1
arch_simba_cosa_ff1 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 2, 'K': 1, 'N': 256},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['M', 'K', 'N'], #WS,
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
        factors_constraints = {'M': 16, 'K': 1}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 65536, # number of entries
        value_access_energy = 30.26, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "DistributionBuffers",
        mesh = 4,
        dims = ['M', 'N'],
        factors_constraints = {'M': 4, 'N': 1}
    ),
    MemLevel(
        name = "PEWeightBuffer",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 32768, # number of entries
        value_access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 4, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 128, # number of entries
        value_access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 8, 'K': 256, 'N': 1},
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "RegMac",
        mesh = 4,
        dims = ['M', 'K'],
        factors_constraints = {'M': 1, 'K': 4}
    ),
    MemLevel(
        name = "PEWeightRegs",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 1, # number of entries
        value_access_energy = 0.70, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 16},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.32, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )])

# SOLUTION GIVEN BY CoSA:
# Comp: MAESTRO-BLAS 1
arch_simba_cosa_mb1 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 2, 'N': 2048},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['M', 'K', 'N'], #WS,
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
        factors_constraints = {'M': 16, 'K': 1}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 65536, # number of entries
        value_access_energy = 30.26, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 8, 'K': 2, 'N': 1},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "DistributionBuffers",
        mesh = 4,
        dims = ['K', 'M'],
        factors_constraints = {'M': 2, 'K': 2}
    ),
    MemLevel(
        name = "PEWeightBuffer",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 32768, # number of entries
        value_access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 128, # number of entries
        value_access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 32, 'K': 256, 'N': 1},
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "RegMac",
        mesh = 4,
        dims = ['M', 'K'],
        factors_constraints = {'M': 1, 'K': 4}
    ),
    MemLevel(
        name = "PEWeightRegs",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 1, # number of entries
        value_access_energy = 0.70, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 4},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.32, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )])

# SOLUTION GIVEN BY CoSA:
# Comp: MAESTRO-BLAS 2
arch_simba_cosa_mb2 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 4, 'N': 128},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['M', 'K', 'N'], #WS,
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
        factors_constraints = {'M': 16, 'K': 1}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = ['M', 'K', 'N'],
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
        dataflow_constraints = ['M', 'K', 'N'],
        size = 32768, # number of entries
        value_access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 128, # number of entries
        value_access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 16, 'K': 512, 'N': 1},
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "RegMac",
        mesh = 4,
        dims = ['M', 'K'],
        factors_constraints = {'M': 1, 'K': 4}
    ),
    MemLevel(
        name = "PEWeightRegs",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 1, # number of entries
        value_access_energy = 0.70, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 8},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.32, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )])

# SOLUTION GIVEN BY CoSA:
# Comp: MAESTRO-BLAS 3
arch_simba_cosa_mb3 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 4, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['M', 'K', 'N'], #WS,
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
        factors_constraints = {'M': 2, 'K': 8}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 65536, # number of entries
        value_access_energy = 30.26, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "DistributionBuffers",
        mesh = 4,
        dims = ['K'],
        factors_constraints = {'K': 4}
    ),
    MemLevel(
        name = "PEWeightBuffer",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 32768, # number of entries
        value_access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 128, # number of entries
        value_access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 64, 'N': 1},
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "RegMac",
        mesh = 4,
        dims = ['M', 'K'],
        factors_constraints = {'M': 4, 'K': 1}
    ),
    MemLevel(
        name = "PEWeightRegs",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 1, # number of entries
        value_access_energy = 0.70, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 8},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.32, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )])

# SOLUTION GIVEN BY CoSA:
# Comp: MAESTRO-BLAS 4
arch_simba_cosa_mb4 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 64, 'N': 8},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['M', 'K', 'N'], #WS,
        size = 65536, # number of entries
        value_access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['N', 'M'],
        factors_constraints = {'M': 8, 'N': 2}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 65536, # number of entries
        value_access_energy = 30.26, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "DistributionBuffers",
        mesh = 4,
        dims = ['N'],
        factors_constraints = {'N': 4}
    ),
    MemLevel(
        name = "PEWeightBuffer",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 32768, # number of entries
        value_access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 128, # number of entries
        value_access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 4, 'N': 1},
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "RegMac",
        mesh = 4,
        dims = ['M', 'K'],
        factors_constraints = {'M': 1, 'K': 4}
    ),
    MemLevel(
        name = "PEWeightRegs",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 1, # number of entries
        value_access_energy = 0.70, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 128},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.32, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )])

# SOLUTION GIVEN BY CoSA:
# Comp: MAESTRO-BLAS 5
arch_simba_cosa_mb5 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 2, 'K': 1, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['M', 'K', 'N'], #WS,
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
        factors_constraints = {'M': 16, 'K': 1}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 65536, # number of entries
        value_access_energy = 30.26, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 2, 'K': 1, 'N': 1},
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
        dataflow_constraints = ['M', 'K', 'N'],
        size = 32768, # number of entries
        value_access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 2, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 128, # number of entries
        value_access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 4, 'K': 1024, 'N': 1},
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "RegMac",
        mesh = 4,
        dims = ['M', 'K'],
        factors_constraints = {'M': 4, 'K': 1}
    ),
    MemLevel(
        name = "PEWeightRegs",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 1, # number of entries
        value_access_energy = 0.70, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 8},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.32, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )])

# SOLUTION GIVEN BY CoSA:
# Comp: MAESTRO-BLAS 6
arch_simba_cosa_mb6 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 4},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['M', 'K', 'N'], #WS,
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
        factors_constraints = {'M': 16, 'K': 1}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 65536, # number of entries
        value_access_energy = 30.26, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "DistributionBuffers",
        mesh = 4,
        dims = ['N'],
        factors_constraints = {'N': 4}
    ),
    MemLevel(
        name = "PEWeightBuffer",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 32768, # number of entries
        value_access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 4, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 128, # number of entries
        value_access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 4, 'K': 128, 'N': 1},
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "RegMac",
        mesh = 4,
        dims = ['M', 'K'],
        factors_constraints = {'M': 2, 'K': 2}
    ),
    MemLevel(
        name = "PEWeightRegs",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 1, # number of entries
        value_access_energy = 0.70, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 16},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.32, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )])


# >>> TPUv1 <<<

# SOLUTION GIVEN BY CoSA:
# Comp: bert large KQV
arch_tpu_cosa_kqv = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 8*2**30, # number of entries
        value_access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 24*(2**20), # number of entries
        value_access_energy = 2.69, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 12, 'K': 1, 'N': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WeightsFIFO",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 4*2**16, # number of entries
        value_access_energy = 2.05, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 2, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    FanoutLevel(
        name = "SARows",
        dims = ['M', 'N'],
        mesh = 256,
        factors_constraints = {'M': 128, 'N': 2}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 4096, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        value_access_energy = 3.03, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_constraints = {'M': 1, 'K': 4, 'N': 1},
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
        factors_constraints = {'M': 1, 'K': 1, 'N': 2048},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.15, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )])

# SOLUTION GIVEN BY CoSA:
# Comp: bert large KTQ
arch_tpu_cosa_ktq = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 8*2**30, # number of entries
        value_access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 24*(2**20), # number of entries
        value_access_energy = 2.69, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WeightsFIFO",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 4*2**16, # number of entries
        value_access_energy = 2.05, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 8, 'K': 1, 'N': 4},
        bypasses = ['in', 'out']
    ),
    FanoutLevel(
        name = "SARows",
        dims = ['M'],
        mesh = 256,
        factors_constraints = {'M': 256}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 4096, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        value_access_energy = 3.03, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_constraints = {'M': 2, 'K': 1, 'N': 1},
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['K'],
        mesh = 256,
        factors_constraints = {'K': 64}
    ),
    MemLevel(
        name = "Register",
        dataflow_constraints = WS,
        size = 2, # number of entries
        value_access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_constraints = {'M': 1, 'K': 1, 'N': 1024},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.15, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )])

# SOLUTION GIVEN BY CoSA:
# Comp: bert large VScores
arch_tpu_cosa_vscores = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 8*2**30, # number of entries
        value_access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 24*(2**20), # number of entries
        value_access_energy = 2.69, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WeightsFIFO",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 4*2**16, # number of entries
        value_access_energy = 2.05, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    FanoutLevel(
        name = "SARows",
        dims = ['N', 'K', 'M'],
        mesh = 256,
        factors_constraints = {'M': 64, 'N': 2, 'K': 2}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 4096, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        value_access_energy = 3.03, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_constraints = {'M': 1, 'K': 8, 'N': 1},
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
        factors_constraints = {'M': 1, 'K': 1, 'N': 2048},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.15, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )])

# SOLUTION GIVEN BY CoSA:
# Comp: bert large FF1
arch_tpu_cosa_ff1 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 8*2**30, # number of entries
        value_access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 24*(2**20), # number of entries
        value_access_energy = 2.69, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 16, 'K': 1, 'N': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WeightsFIFO",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 4*2**16, # number of entries
        value_access_energy = 2.05, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 2, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    FanoutLevel(
        name = "SARows",
        dims = ['M', 'N'],
        mesh = 256,
        factors_constraints = {'M': 128, 'N': 2}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 4096, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        value_access_energy = 3.03, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_constraints = {'M': 1, 'K': 4, 'N': 1},
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
        factors_constraints = {'M': 1, 'K': 1, 'N': 2048},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.15, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )])

# SOLUTION GIVEN BY CoSA:
# Comp: MAESTRO-BLAS 1
arch_tpu_cosa_mb1 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 8*2**30, # number of entries
        value_access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 4, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 24*(2**20), # number of entries
        value_access_energy = 2.69, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 64, 'K': 1, 'N': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WeightsFIFO",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 4*2**16, # number of entries
        value_access_energy = 2.05, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 2, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    FanoutLevel(
        name = "SARows",
        dims = ['M', 'N'],
        mesh = 256,
        factors_constraints = {'M': 64, 'N': 4}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 4096, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        value_access_energy = 3.03, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_constraints = {'M': 1, 'K': 8, 'N': 1},
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
        factors_constraints = {'M': 1, 'K': 1, 'N': 2048},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.15, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )])

# SOLUTION GIVEN BY CoSA:
# Comp: MAESTRO-BLAS 2
arch_tpu_cosa_mb2 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 8*2**30, # number of entries
        value_access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 24*(2**20), # number of entries
        value_access_energy = 2.69, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 32, 'K': 1, 'N': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WeightsFIFO",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 4*2**16, # number of entries
        value_access_energy = 2.05, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    FanoutLevel(
        name = "SARows",
        dims = ['M', 'K'],
        mesh = 256,
        factors_constraints = {'M': 16, 'K': 16}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 4096, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        value_access_energy = 3.03, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_constraints = {'M': 1, 'K': 4, 'N': 1},
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['M', 'K'],
        mesh = 256,
        factors_constraints = {'M': 2, 'K': 128}
    ),
    MemLevel(
        name = "Register",
        dataflow_constraints = WS,
        size = 2, # number of entries
        value_access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_constraints = {'M': 1, 'K': 1, 'N': 1024},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.15, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )])

# SOLUTION GIVEN BY CoSA:
# Comp: MAESTRO-BLAS 3
arch_tpu_cosa_mb3 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 8*2**30, # number of entries
        value_access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 24*(2**20), # number of entries
        value_access_energy = 2.69, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WeightsFIFO",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 4*2**16, # number of entries
        value_access_energy = 2.05, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    FanoutLevel(
        name = "SARows",
        dim = WS[1],
        mesh = 256,
        factors_constraints = {'K': 256}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 4096, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        value_access_energy = 3.03, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_constraints = {'M': 1, 'K': 32, 'N': 1},
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "SACols",
        dim = WS[0],
        mesh = 256,
        factors_constraints = {'M': 8}
    ),
    MemLevel(
        name = "Register",
        dataflow_constraints = WS,
        size = 2, # number of entries
        value_access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_constraints = {'M': 1, 'K': 1, 'N': 8},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.15, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )])

# SOLUTION GIVEN BY CoSA:
# Comp: MAESTRO-BLAS 4
arch_tpu_cosa_mb4 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 8*2**30, # number of entries
        value_access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 24*(2**20), # number of entries
        value_access_energy = 2.69, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WeightsFIFO",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 4*2**16, # number of entries
        value_access_energy = 2.05, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 4},
        bypasses = ['in', 'out']
    ),
    FanoutLevel(
        name = "SARows",
        dims = ['K', 'N'],
        mesh = 256,
        factors_constraints = {'K': 32, 'N': 8}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 4096, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        value_access_energy = 3.03, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_constraints = {'M': 1, 'K': 32, 'N': 1},
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "SACols",
        dim = WS[0],
        mesh = 256,
        factors_constraints = {'M': 8}
    ),
    MemLevel(
        name = "Register",
        dataflow_constraints = WS,
        size = 2, # number of entries
        value_access_energy = 0.01, # per operand (pJ)
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
    )])

# SOLUTION GIVEN BY CoSA:
# Comp: MAESTRO-BLAS 5
arch_tpu_cosa_mb5 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 8*2**30, # number of entries
        value_access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 24*(2**20), # number of entries
        value_access_energy = 2.69, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 32, 'K': 1, 'N': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WeightsFIFO",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 4*2**16, # number of entries
        value_access_energy = 2.05, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    FanoutLevel(
        name = "SARows",
        dim = WS[1],
        mesh = 256,
        factors_constraints = {'K': 256}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 4096, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        value_access_energy = 3.03, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_constraints = {'M': 2, 'K': 2, 'N': 1},
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['M', 'K'],
        mesh = 256,
        factors_constraints = {'M': 128, 'K': 2}
    ),
    MemLevel(
        name = "Register",
        dataflow_constraints = WS,
        size = 2, # number of entries
        value_access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_constraints = {'M': 1, 'K': 1, 'N': 8},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.15, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )])

# SOLUTION GIVEN BY CoSA:
# Comp: MAESTRO-BLAS 6
arch_tpu_cosa_mb6 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 8*2**30, # number of entries
        value_access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 24*(2**20), # number of entries
        value_access_energy = 2.69, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WeightsFIFO",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 4*2**16, # number of entries
        value_access_energy = 2.05, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    FanoutLevel(
        name = "SARows",
        dims = ['M', 'K'],
        mesh = 256,
        factors_constraints = {'M': 64, 'K': 4}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 4096, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        value_access_energy = 3.03, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_constraints = {'M': 1, 'K': 2, 'N': 1},
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['M', 'K'],
        mesh = 256,
        factors_constraints = {'M': 8, 'K': 32}
    ),
    MemLevel(
        name = "Register",
        dataflow_constraints = WS,
        size = 2, # number of entries
        value_access_energy = 0.01, # per operand (pJ)
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
    )])