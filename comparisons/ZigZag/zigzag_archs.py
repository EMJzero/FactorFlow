from architectures.architectures import WS, OS, IS
from computations import gemm_coupling
from levels import *
from arch import *
from arch import *

# ZigZag conversions:
# K -> M
# C -> K
# OY -> N

# >>> GEMMINI <<<
# > WS version  <

# Modified architecture with split-memories
arch_gemmini_zigzag_compatible = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = [], #['N', 'K', 'M'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = [], #WS,
        size = 320*(2**10), # number of entries
        value_access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {},
        bypasses = ['out', 'in']
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = [], #WS,
        size = 320*(2**10), # number of entries
        value_access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {},
        bypasses = ['out', 'w']
    ),
    FanoutLevel(
        name = "SARows",
        dim = WS[0],
        mesh = 16,
        # pe_to_pe should be used, since Gemmini uses a systolic array, but Timeloop
        # does not have this feature, so for sake of comparison, it is turned off
        #pe_to_pe = True, 
        factors_constraints = {'M': 16}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = [], #WS,
        size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        value_access_energy = 4.01, # per operand (pJ)
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
        factors_constraints = {'K': 16}
    ),
    MemLevel(
        name = "Register",
        dataflow_constraints = WS,
        size = 1, # number of entries
        value_access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
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

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# NOTE: unfair compared to the Gemmini architecture in "architectures.py"
#       because LOMA can "split" memories at will. Hence compare only
#       with the Gemmini version in this file!
# Comp: bert large KQV
arch_gemmini_zigzag_loma_kqv = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 16},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = WS,
        size = 512*(2**10), # number of entries
        value_access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 16, 'K': 1, 'N': 1},
        bypasses = ['out', 'w']
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = WS,
        size = 512*(2**10), # number of entries
        value_access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 12, 'K': 1, 'N': 1},
        bypasses = ['out', 'in']
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
        factors_constraints = {'M': 1, 'K': 64, 'N': 16}, # the systolic array does a 16x16 matmul in this case
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

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: bert large VScores
arch_gemmini_zigzag_loma_vscores = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 512*(2**10), # number of entries
        value_access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 16, 'N': 16},
        bypasses = ['out', 'in']
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = WS,
        size = 512*(2**10), # number of entries
        value_access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M':1, 'K': 1, 'N': 1},
        bypasses = ['out', 'w']
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
        factors_constraints = {'M': 4, 'K': 16, 'N': 16}, # the systolic array does a 16x16 matmul in this case
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

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: bert large KTQ
arch_gemmini_zigzag_loma_ktq = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 512*(2**10), # number of entries
        value_access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 256, 'K': 1, 'N': 16},
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
        dataflow_constraints = WS,
        size = 1, # number of entries
        value_access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared) -> 2 is a classic register which can be read & written in the same cc
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
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: bert large FF1
arch_gemmini_zigzag_loma_ff1 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 16, 'K': 1, 'N': 16},
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
        factors_constraints = {'M': 1, 'K': 64, 'N': 1}, # the systolic array does a 16x16 matmul in this case
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
        bandwidth = 2, # operands per cycle (shared) -> 2 is a classic register which can be read & written in the same cc
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
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: MAESTRO-BLAS 1
arch_gemmini_zigzag_loma_mb1 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 32, 'K': 16, 'N': 32},
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
        dataflow_constraints = ['M', 'K', 'N'],
        size = 1, # number of entries
        value_access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared) -> 2 is a classic register which can be read & written in the same cc
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
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: MAESTRO-BLAS 2
arch_gemmini_zigzag_loma_mb2 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand (pJ)
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
        factors_constraints = {'M': 64, 'K': 1, 'N': 1},
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
        factors_constraints = {'M': 1, 'K': 16, 'N': 1}, # the systolic array does a 16x16 matmul in this case
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
        bandwidth = 2, # operands per cycle (shared) -> 2 is a classic register which can be read & written in the same cc
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
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: MAESTRO-BLAS 3
arch_gemmini_zigzag_loma_mb3 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand (pJ)
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
        dim = WS[0],
        mesh = 16,
        factors_constraints = {'M': 8}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = ['M', 'K', 'N'],
        size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        value_access_energy = 4.01, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 512, 'N': 1}, # the systolic array does a 16x16 matmul in this case
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
        bandwidth = 2, # operands per cycle (shared) -> 2 is a classic register which can be read & written in the same cc
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
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: MAESTRO-BLAS 4
arch_gemmini_zigzag_loma_mb4 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 16},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 512*(2**10), # number of entries
        value_access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M':1, 'K': 4, 'N': 1},
        bypasses = ['out', 'in']
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 512*(2**10), # number of entries
        value_access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['out', 'w']
    ),
    FanoutLevel(
        name = "SARows",
        dim = WS[0],
        mesh = 16,
        factors_constraints = {'M': 8}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = ['M', 'K', 'N'],
        size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        value_access_energy = 4.01, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 16, 'N': 1}, # the systolic array does a 16x16 matmul in this case
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
        bandwidth = 2, # operands per cycle (shared) -> 2 is a classic register which can be read & written in the same cc
        factors_constraints = {'M': 1, 'K': 1, 'N': 512},
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

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: MAESTRO-BLAS 5
arch_gemmini_zigzag_loma_mb5 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 16, 'K': 1, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 512*(2**10), # number of entries
        value_access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M':1, 'K': 16, 'N': 1},
        bypasses = ['out', 'w']
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 512*(2**10), # number of entries
        value_access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['out', 'in']
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
        factors_constraints = {'M': 32, 'K': 4, 'N': 1}, # the systolic array does a 16x16 matmul in this case
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
        bandwidth = 2, # operands per cycle (shared) -> 2 is a classic register which can be read & written in the same cc
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
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: MAESTRO-BLAS 6
arch_gemmini_zigzag_loma_mb6 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand (pJ)
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
        factors_constraints = {'M': 8, 'K': 1, 'N': 1},
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
        bandwidth = 2, # operands per cycle (shared) -> 2 is a classic register which can be read & written in the same cc
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
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: bert large KQV
arch_gemmini_zigzag_salsa_kqv = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 8, 'K': 1, 'N':32},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 512*(2**10), # number of entries
        value_access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 2, 'K': 1, 'N': 1},
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
        factors_constraints = {'M': 12, 'K': 64, 'N': 1}, # the systolic array does a 16x16 matmul in this case
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
        bandwidth = 2, # operands per cycle (shared) -> 2 is a classic register which can be read & written in the same cc
        factors_constraints = {'M': 1, 'K': 1, 'N': 128},
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

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: bert large KTQ
arch_gemmini_zigzag_salsa_ktq = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['N', 'M', 'K'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 256, 'K': 1, 'N': 2},
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
        bandwidth = 2, # operands per cycle (shared) -> 2 is a classic register which can be read & written in the same cc
        factors_constraints = {'M': 1, 'K': 1, 'N': 2048},
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

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: bert large VScores
arch_gemmini_zigzag_salsa_vscores = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 512*(2**10), # number of entries
        value_access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M':1, 'K': 8, 'N': 8},
        bypasses = ['out', 'in']
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 512*(2**10), # number of entries
        value_access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['out', 'w']
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
        factors_constraints = {'M': 4, 'K': 32, 'N': 1}, # the systolic array does a 16x16 matmul in this case
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
        bandwidth = 2, # operands per cycle (shared) -> 2 is a classic register which can be read & written in the same cc
        factors_constraints = {'M': 1, 'K': 1, 'N': 512},
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

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: bert large FF1
arch_gemmini_zigzag_salsa_ff1 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 16, 'K': 1, 'N': 16},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 512*(2**10), # number of entries
        value_access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 2, 'K': 1, 'N': 1},
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
        factors_constraints = {'M': 8, 'K': 64, 'N': 1}, # the systolic array does a 16x16 matmul in this case
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
        bandwidth = 2, # operands per cycle (shared) -> 2 is a classic register which can be read & written in the same cc
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
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: MAESTRO-BLAS 1
arch_gemmini_zigzag_salsa_mb1 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 32, 'K': 8, 'N': 32},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 512*(2**10), # number of entries
        value_access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 2, 'K': 1, 'N': 1},
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
        dataflow_constraints = ['K', 'M', 'N'],
        size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        value_access_energy = 4.01, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 8, 'K': 64, 'N': 1}, # the systolic array does a 16x16 matmul in this case
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
        bandwidth = 2, # operands per cycle (shared) -> 2 is a classic register which can be read & written in the same cc
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
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: MAESTRO-BLAS 2
arch_gemmini_zigzag_salsa_mb2 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['K', 'N', 'M'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 4, 'K': 8, 'N': 4},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 512*(2**10), # number of entries
        value_access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 2, 'K': 1, 'N': 1},
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
        dataflow_constraints = ['K', 'M', 'N'],
        size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        value_access_energy = 4.01, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 8, 'K': 64, 'N': 1}, # the systolic array does a 16x16 matmul in this case
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
        bandwidth = 2, # operands per cycle (shared) -> 2 is a classic register which can be read & written in the same cc
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
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: MAESTRO-BLAS 3
arch_gemmini_zigzag_salsa_mb3 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand (pJ)
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
        dim = WS[0],
        mesh = 16,
        factors_constraints = {'M': 8}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = ['M', 'K', 'N'],
        size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        value_access_energy = 4.01, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 512, 'N': 1}, # the systolic array does a 16x16 matmul in this case
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
        bandwidth = 2, # operands per cycle (shared) -> 2 is a classic register which can be read & written in the same cc
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
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: MAESTRO-BLAS 4
arch_gemmini_zigzag_salsa_mb4 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 512*(2**10), # number of entries
        value_access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 8, 'N': 4},
        bypasses = ['out', 'in']
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 512*(2**10), # number of entries
        value_access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['out', 'w']
    ),
    FanoutLevel(
        name = "SARows",
        dim = WS[0],
        mesh = 16,
        factors_constraints = {'M': 8}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = ['M', 'K', 'N'],
        size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        value_access_energy = 4.01, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 8, 'N': 1}, # the systolic array does a 16x16 matmul in this case
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
        bandwidth = 2, # operands per cycle (shared) -> 2 is a classic register which can be read & written in the same cc
        factors_constraints = {'M': 1, 'K': 1, 'N': 2048},
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

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: MAESTRO-BLAS 5
arch_gemmini_zigzag_salsa_mb5 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand (pJ)
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
        factors_constraints = {'M': 2, 'K': 1, 'N': 1},
        bypasses = ['out', 'w']
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
        factors_constraints = {'M': 2, 'K': 8, 'N': 1}, # the systolic array does a 16x16 matmul in this case
        bypasses = ['in', 'w']
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 512*(2**10), # number of entries
        value_access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 128, 'K': 8, 'N': 1},
        bypasses = ['out', 'in']
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
        bandwidth = 2, # operands per cycle (shared) -> 2 is a classic register which can be read & written in the same cc
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
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: MAESTRO-BLAS 6
arch_gemmini_zigzag_salsa_mb6 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand (pJ)
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
        factors_constraints = {'M': 4, 'K': 1, 'N': 1},
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
        factors_constraints = {'M': 8, 'K': 16, 'N': 1}, # the systolic array does a 16x16 matmul in this case
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
        bandwidth = 2, # operands per cycle (shared) -> 2 is a classic register which can be read & written in the same cc
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
    )], coupling=gemm_coupling)


# >>> EYERISS <<<

# Modified architecture with split-memories
arch_eyeriss_zigzag_compatible = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = [], # ['N', 'M', 'K'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = [], #WS,
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
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
        dims = WS[0],
        factors_constraints = {} #{'M': 12}
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = [], #WS,
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = [], #WS,
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'K': 1, 'N': 1},
        bypasses = ['in', 'w']
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = [], #WS,
        size = 12*2, # number of entries
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1},
        bypasses = ['w', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.21, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )], coupling=gemm_coupling)

# Modified architecture without constraints
arch_eyeriss_zigzag_compatible_2 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = [], # ['N', 'M', 'K'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = [], #WS,
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
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
        dims = WS[0],
        factors_constraints = {} #{'M': 12}
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = [], #WS,
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = [], #WS,
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {},
        bypasses = ['in', 'w']
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = [], #WS,
        size = 12*2, # number of entries
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {},
        bypasses = ['w', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.21, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# NOTE: solution obtained by constraining the fanout levels to the known best. Ignore.
# Comp: bert large KQV
arch_eyeriss_zigzag_loma_kqv = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 32, 'K': 16, 'N': 16},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'N', 'M'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
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
        bypasses = ['w', 'out', 'in']
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['N', 'M', 'K'],
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 256},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = WS,
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 64, 'N': 1},
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

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# NOTE: unfair compared to the Eyeriss architecture in "architectures.py"
#       because LOMA can "split" memories at will. Hence compare only
#       with the Eyeriss version in this file!
# Comp: bert large VScores
arch_eyeriss_zigzag_loma_vscores = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 16, 'K': 8, 'N': 256},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'N', 'M'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dim = 'K',
        mesh = 14,
        factors_constraints = {'K': 8}
    ),
    FanoutLevel(
        name = "SARows",
        dim = 'M', # one of M or K
        mesh = 12,
        factors_constraints = {'M': 12}
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['N', 'M', 'K'],
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 16, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 16, 'N': 1},
        bypasses = ['in', 'w']
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = WS,
        size = 12*2, # number of entries
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 16},
        bypasses = ['w', 'out', 'in']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.21, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# NOTE: unfair compared to the Eyeriss architecture in "architectures.py"
#       because LOMA can "split" memories at will. Hence compare only
#       with the Eyeriss version in this file!
# Comp: bert large VScores
arch_eyeriss_zigzag_loma_vscores = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 2, 'K': 16, 'N': 256},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'N', 'M'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dim = 'K',
        mesh = 14,
        factors_constraints = {'K': 8}
    ),
    FanoutLevel(
        name = "SARows",
        dim = 'M', # one of M or K
        mesh = 12,
        factors_constraints = {'M': 8}
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['N', 'M', 'K'],
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 16},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 32, 'N': 1},
        bypasses = ['in', 'w']
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = WS,
        size = 12*2, # number of entries
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 4, 'K': 1, 'N': 1},
        bypasses = ['w', 'out', 'in']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.21, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: bert large KTQ
arch_eyeriss_zigzag_loma_ktq = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 32, 'K': 1, 'N': 256},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'N', 'M'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['M', 'K'],
        mesh = 14,
        factors_constraints = {'K': 8}
    ),
    FanoutLevel(
        name = "SARows",
        dim = 'M', # one of M or K
        mesh = 12,
        factors_constraints = {'M': 8}
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['N', 'M', 'K'],
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 16, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = WS,
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 8, 'N': 1},
        bypasses = ['in', 'w']
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = WS,
        size = 12*2, # number of entries
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 16},
        bypasses = ['w', 'out', 'in']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.21, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: bert large FF1
arch_eyeriss_zigzag_loma_ff1 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 32, 'K': 8, 'N': 256},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'N', 'M'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['M', 'K'],
        mesh = 14,
        factors_constraints = {'K': 8}
    ),
    FanoutLevel(
        name = "SARows",
        dim = 'M', # one of M or K
        mesh = 12,
        factors_constraints = {'M': 8}
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['N', 'M', 'K'],
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 16, 'K': 16, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = WS,
        size = 12*2, # number of entries
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w', 'out', 'in']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = WS,
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 16},
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

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: MAESTRO-BLAS 1
arch_eyeriss_zigzag_loma_mb1 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 64, 'K': 64, 'N': 256},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'N', 'M'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['M', 'K'],
        mesh = 14,
        factors_constraints = {'K': 8}
    ),
    FanoutLevel(
        name = "SARows",
        dim = 'M', # one of M or K
        mesh = 12,
        factors_constraints = {'M': 8}
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 16, 'K': 1, 'N': 32},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = WS,
        size = 12*2, # number of entries
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w', 'out', 'in']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = WS,
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 16, 'N': 1},
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

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: MAESTRO-BLAS 2
arch_eyeriss_zigzag_loma_mb2 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 16, 'K': 64, 'N': 64},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'N', 'M'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 16},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['M', 'K'],
        mesh = 14,
        factors_constraints = {'K': 8}
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
        bypasses = ['w', 'out', 'in']
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['N', 'M', 'K'],
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 8, 'K': 16, 'N': 1},
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

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: MAESTRO-BLAS 3
arch_eyeriss_zigzag_loma_mb3 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'N', 'M'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 4, 'N': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['M', 'K'],
        mesh = 14,
        factors_constraints = {'K': 8}
    ),
    FanoutLevel(
        name = "SARows",
        dim = 'M', # one of M or K
        mesh = 12,
        factors_constraints = {'M': 8}
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['N', 'M', 'K'],
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 16, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 16, 'N': 4},
        bypasses = ['in', 'w']
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = WS,
        size = 12*2, # number of entries
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 2},
        bypasses = ['w', 'out', 'in']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.21, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: MAESTRO-BLAS 4
arch_eyeriss_zigzag_loma_mb4 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 16},
        bypasses = []
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['M', 'K'],
        mesh = 14,
        factors_constraints = {'K': 8}
    ),
    FanoutLevel(
        name = "SARows",
        dim = 'M', # one of M or K
        mesh = 12,
        factors_constraints = {'M': 8}
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['N', 'M', 'K'],
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 4, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'N', 'M'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 16},
        bypasses = ['w', 'in']
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'N', 'M'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 32},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = WS,
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 8, 'N': 1},
        bypasses = ['in', 'w']
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = WS,
        size = 12*2, # number of entries
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 4, 'N': 1},
        bypasses = ['w', 'out', 'in']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.21, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: MAESTRO-BLAS 5
arch_eyeriss_zigzag_loma_mb5 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 64, 'K': 1, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'N', 'M'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 16, 'N': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['M', 'K'],
        mesh = 14,
        factors_constraints = {'K': 8}
    ),
    FanoutLevel(
        name = "SARows",
        dim = 'M', # one of M or K
        mesh = 12,
        factors_constraints = {'M': 8}
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['N', 'M', 'K'],
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 16, 'K': 1, 'N': 2},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = WS,
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 8, 'N': 1},
        bypasses = ['in', 'w']
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = WS,
        size = 12*2, # number of entries
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 4},
        bypasses = ['w', 'out', 'in']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.21, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: MAESTRO-BLAS 6
arch_eyeriss_zigzag_loma_mb6 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'N', 'M'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 16, 'K': 1, 'N': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'N', 'M'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w', 'in']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['M', 'K'],
        mesh = 14,
        factors_constraints = {'K': 8}
    ),
    FanoutLevel(
        name = "SARows",
        dim = 'M', # one of M or K
        mesh = 12,
        factors_constraints = {'M': 8}
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['N', 'M', 'K'],
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 256},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = WS,
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 8, 'N': 1},
        bypasses = ['in', 'w']
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 12*2, # number of entries
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 4, 'K': 4, 'N': 1},
        bypasses = ['w', 'out', 'in']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.21, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: bert large KQV
arch_eyeriss_zigzag_salsa_kqv = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['N', 'M', 'K'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 16, 'K': 2, 'N': 32},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'N', 'M'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 4, 'K': 1, 'N': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dim = 'K',
        mesh = 14,
        factors_constraints = {'K': 8}
    ),
    FanoutLevel(
        name = "SARows",
        dim = 'M', # one of M or K
        mesh = 12,
        factors_constraints = {'M': 12}
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 2, 'K': 1, 'N': 16},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 32, 'N': 1},
        bypasses = ['in', 'w']
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 12*2, # number of entries
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 2, 'K': 2, 'N': 8},
        bypasses = ['w', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.21, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: bert large KTQ
arch_eyeriss_zigzag_salsa_ktq = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['N', 'M', 'K'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 16, 'K': 1, 'N': 4},
        bypasses = []
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['N', 'M', 'K'],
        size = 192*2*8*8, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 4, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['N', 'M', 'K'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 4, 'K': 1, 'N': 128},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['M', 'K'],
        mesh = 14,
        factors_constraints = {'M': 1, 'K': 8}
    ),
    FanoutLevel(
        name = "SARows",
        dim = 'M', # one of M or K
        mesh = 12,
        factors_constraints = {'M': 8}
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 4, 'N': 1},
        bypasses = ['in', 'w']
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 12*2, # number of entries
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 2, 'K': 2, 'N': 8},
        bypasses = ['w', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.21, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: bert large VScores
arch_eyeriss_zigzag_salsa_vscores = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['N', 'M', 'K'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 16, 'N': 8},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'N', 'M'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 2},
        bypasses = ['w', 'in']
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'N', 'M'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['M', 'K'],
        mesh = 14,
        factors_constraints = {'M': 1, 'K': 8}
    ),
    FanoutLevel(
        name = "SARows",
        dim = 'M', # one of M or K
        mesh = 12,
        factors_constraints = {'M': 8}
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['N', 'M', 'K'],
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 4, 'K': 1, 'N': 32},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 16, 'N': 1},
        bypasses = ['in', 'w']
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 12*2, # number of entries
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 2, 'K': 2, 'N': 8},
        bypasses = ['w', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.21, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: bert large FF1
arch_eyeriss_zigzag_salsa_ff1 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['N', 'M', 'K'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 16, 'K': 2, 'N': 32},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'N', 'M'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 8, 'K': 1, 'N': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['M', 'K'],
        mesh = 14,
        factors_constraints = {'M': 1, 'K': 8}
    ),
    FanoutLevel(
        name = "SARows",
        dim = 'M', # one of M or K
        mesh = 12,
        factors_constraints = {'M': 8}
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 2, 'K': 1, 'N': 16},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 32, 'N': 1},
        bypasses = ['in', 'w']
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 12*2, # number of entries
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 2, 'K': 2, 'N': 8},
        bypasses = ['w', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.21, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: MAESTRO-BLAS 1
arch_eyeriss_zigzag_salsa_mb1 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 32, 'K': 16, 'N': 64},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'N', 'M'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 8, 'K': 1, 'N': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['M', 'K'],
        mesh = 14,
        factors_constraints = {'M': 1, 'K': 8}
    ),
    FanoutLevel(
        name = "SARows",
        dim = 'M', # one of M or K
        mesh = 12,
        factors_constraints = {'M': 8}
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['N', 'M', 'K'],
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 2, 'K': 1, 'N': 16},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 32, 'N': 1},
        bypasses = ['in', 'w']
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 12*2, # number of entries
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 2, 'K': 2, 'N': 8},
        bypasses = ['w', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.21, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: MAESTRO-BLAS 2
arch_eyeriss_zigzag_salsa_mb2 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 4, 'K': 16, 'N': 8},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'N', 'M'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 8, 'K': 1, 'N': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['M', 'K'],
        mesh = 14,
        factors_constraints = {'M': 1, 'K': 8}
    ),
    FanoutLevel(
        name = "SARows",
        dim = 'M', # one of M or K
        mesh = 12,
        factors_constraints = {'M': 8}
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 2, 'K': 1, 'N': 16},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 32, 'N': 1},
        bypasses = ['in', 'w']
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 12*2, # number of entries
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 2, 'K': 2, 'N': 8},
        bypasses = ['w', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.21, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: MAESTRO-BLAS 3
arch_eyeriss_zigzag_salsa_mb3 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'N', 'M'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['M', 'K'],
        mesh = 14,
        factors_constraints = {'M': 1, 'K': 8}
    ),
    FanoutLevel(
        name = "SARows",
        dim = 'M', # one of M or K
        mesh = 12,
        factors_constraints = {'M': 8}
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 4, 'N': 1},
        bypasses = ['in', 'w']
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['N', 'M', 'K'],
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 128, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 12*2, # number of entries
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 2, 'N': 8},
        bypasses = ['w', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.21, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: MAESTRO-BLAS 4
arch_eyeriss_zigzag_salsa_mb4 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'N', 'M'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 16, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 128},
        bypasses = ['w', 'in']
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'N', 'M'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 16, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 4},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['M', 'K'],
        mesh = 14,
        factors_constraints = {'M': 1, 'K': 8}
    ),
    FanoutLevel(
        name = "SARows",
        dim = 'M', # one of M or K
        mesh = 12,
        factors_constraints = {'M': 8}
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['N', 'M', 'K'],
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 128, 'N': 1},
        bypasses = ['in', 'w']
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 12*2, # number of entries
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 16},
        bypasses = ['w', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.21, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: MAESTRO-BLAS 5
arch_eyeriss_zigzag_salsa_mb5 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'N', 'M'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 512, 'K': 1, 'N': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['M', 'K'],
        mesh = 14,
        factors_constraints = {'M': 1, 'K': 8}
    ),
    FanoutLevel(
        name = "SARows",
        dim = 'M', # one of M or K
        mesh = 12,
        factors_constraints = {'M': 8}
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['N', 'M', 'K'],
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 64, 'N': 1},
        bypasses = ['in', 'w']
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 12*2, # number of entries
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 2, 'K': 2, 'N': 8},
        bypasses = ['w', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.21, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: MAESTRO-BLAS 6
arch_eyeriss_zigzag_salsa_mb6 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 2, 'K': 1, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'N', 'M'],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 4, 'K': 1, 'N': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['M', 'K'],
        mesh = 14,
        factors_constraints = {'M': 1, 'K': 8}
    ),
    FanoutLevel(
        name = "SARows",
        dim = 'M', # one of M or K
        mesh = 12,
        factors_constraints = {'M': 8}
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 4, 'K': 1, 'N': 32},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 16, 'N': 1},
        bypasses = ['in', 'w']
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 12*2, # number of entries
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 2, 'K': 2, 'N': 8},
        bypasses = ['w', 'out']
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

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: bert large KQV
arch_simba_zigzag_loma_kqv = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 12, 'K': 1, 'N': 256},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'M', 'N'], #WS,
        size = 65536, # number of entries
        value_access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['M'],
        factors_constraints = {'M': 16}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = WS,
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
        dataflow_constraints = ['N', 'M', 'K'],
        size = 32768, # number of entries
        value_access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 16, 'K': 1, 'N': 1},
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
        dims = ['K'],
        factors_constraints = {'K': 4}
    ),
    MemLevel(
        name = "PEWeightRegs",
        dataflow_constraints = WS,
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
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: bert large VScores
arch_simba_zigzag_loma_vscores = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 16},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'M', 'N'], #WS,
        size = 65536, # number of entries
        value_access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 256},
        bypasses = ['w', 'in']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['M'],
        factors_constraints = {'M': 16}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = WS,
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
        dataflow_constraints = ['N', 'M', 'K'],
        size = 32768, # number of entries
        value_access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 128, # number of entries
        value_access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 4, 'K': 256, 'N': 1},
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

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: bert large KTQ
arch_simba_zigzag_loma_ktq = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 16},
        bypasses = []
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['M'],
        factors_constraints = {'M': 16}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = WS,
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
        dataflow_constraints = ['M', 'N', 'K'],
        size = 32768, # number of entries
        value_access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 64, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'M', 'N'], #WS,
        size = 65536, # number of entries
        value_access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 256},
        bypasses = ['w', 'in']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 128, # number of entries
        value_access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 16, 'N': 1},
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

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: bert large FF1
arch_simba_zigzag_loma_ff1 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 4, 'K': 1, 'N': 256},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'M', 'N'], #WS,
        size = 65536, # number of entries
        value_access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w', 'in']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['M'],
        factors_constraints = {'M': 16,}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = WS,
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
        dataflow_constraints = ['N', 'M', 'K'],
        size = 32768, # number of entries
        value_access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 16, 'K': 16, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 128, # number of entries
        value_access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 16, 'N': 1},
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
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: MAESTRO-BLAS 1
arch_simba_zigzag_loma_mb1 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['K', 'N', 'M'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 8, 'K': 16, 'N': 256},
        bypasses = []
    ),
    #MemLevel(
    #    name = "GlobalBuffer",
    #    dataflow_constraints = ['K', 'M', 'N'], #WS,
    #    size = 65536, # number of entries
    #    value_access_energy = 1.85, # per operand (pJ)
    #    bandwidth = 2**10, # operands per cycle (shared)
    #    factors_constraints = {'M': 1, 'K': 1, 'N': 1},
    #    bypasses = ['w', 'in']
    #),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['M'],
        factors_constraints = {'M': 16}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = WS,
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
        dataflow_constraints = ['N', 'M', 'K'],
        size = 32768, # number of entries
        value_access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 16, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 128, # number of entries
        value_access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 128, 'N': 1},
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
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: MAESTRO-BLAS 2
arch_simba_zigzag_loma_mb2 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 4, 'K': 1, 'N': 16},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'M', 'N'], #WS,
        size = 65536, # number of entries
        value_access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w', 'in']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['M'],
        factors_constraints = {'M': 16}
    ),
    FanoutLevel(
        name = "DistributionBuffers",
        mesh = 4,
        dims = ['M'],
        factors_constraints = {'M': 4}
    ),
    MemLevel(
        name = "PEWeightBuffer",
        dataflow_constraints = ['N', 'M', 'K'],
        size = 32768, # number of entries
        value_access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 16, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = WS,
        size = 65536, # number of entries
        value_access_energy = 30.26, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 64},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 128, # number of entries
        value_access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 4, 'K': 128, 'N': 1},
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

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: MAESTRO-BLAS 3
arch_simba_zigzag_loma_mb3 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'M', 'N'], #WS,
        size = 65536, # number of entries
        value_access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w', 'in']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['K', 'M'],
        factors_constraints = {'M': 2, 'K': 8}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = WS,
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
        dataflow_constraints = ['N', 'M', 'K'],
        size = 32768, # number of entries
        value_access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 16, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 128, # number of entries
        value_access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 16, 'N': 1},
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
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: MAESTRO-BLAS 4
arch_simba_zigzag_loma_mb4 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'M', 'N'], #WS,
        size = 65536, # number of entries
        value_access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 32},
        bypasses = ['w', 'in']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['K', 'M'],
        factors_constraints = {'M': 2, 'K': 8}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = WS,
        size = 65536, # number of entries
        value_access_energy = 30.26, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 16},
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
        dataflow_constraints = ['N', 'M', 'K'],
        size = 32768, # number of entries
        value_access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 4, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 128, # number of entries
        value_access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 8, 'N': 1},
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
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: MAESTRO-BLAS 5
arch_simba_zigzag_loma_mb5 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 16, 'K': 1, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'M', 'N'], #WS,
        size = 65536, # number of entries
        value_access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w', 'in']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['K', 'M'],
        factors_constraints = {'M': 4, 'K': 4}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = WS,
        size = 65536, # number of entries
        value_access_energy = 30.26, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 2},
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
        dataflow_constraints = ['N', 'M', 'K'],
        size = 32768, # number of entries
        value_access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 32, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 128, # number of entries
        value_access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 64, 'N': 1},
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
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: MAESTRO-BLAS 6
arch_simba_zigzag_loma_mb6 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'M', 'N'], #WS,
        size = 6553600, # number of entries
        value_access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w', 'in']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['K', 'M'],
        factors_constraints = {'M': 4, 'K': 4}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = WS,
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
        dataflow_constraints = ['M', 'N', 'K'],
        size = 32768, # number of entries
        value_access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 4, 'K': 1, 'N': 256},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 128, # number of entries
        value_access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 8, 'K': 16, 'N': 1},
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

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: bert large KQV
arch_simba_zigzag_salsa_kqv = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 2, 'K': 1, 'N': 8},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'M', 'N'], #WS,
        size = 655360, # number of entries
        value_access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 2},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['K', 'M'],
        factors_constraints = {'M': 4, 'K': 4}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = WS,
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
        dataflow_constraints = ['M', 'N', 'K'],
        size = 32768, # number of entries
        value_access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 4, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'N', 'M'], #WS,
        size = 655360, # number of entries
        value_access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 24, 'K': 1, 'N': 4},
        bypasses = ['w', 'in']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 128, # number of entries
        value_access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 64, 'N': 1},
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
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: bert large KTQ
arch_simba_zigzag_salsa_ktq = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'M', 'N'], #WS,
        size = 262144, # number of entries
        value_access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['K', 'M'],
        factors_constraints = {'M': 4, 'K': 4}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = WS,
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
        dataflow_constraints = ['M', 'N', 'K'],
        size = 32768, # number of entries
        value_access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 16, 'K': 1, 'N': 4},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'M', 'N'], #WS,
        size = 655360, # number of entries
        value_access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 16, 'K': 1, 'N': 16},
        bypasses = ['w', 'in']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 128, # number of entries
        value_access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 4, 'N': 1},
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
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: bert large VScores
arch_simba_zigzag_salsa_vscores = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 4},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'M', 'N'], #WS,
        size = 655360, # number of entries
        value_access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 2},
        bypasses = ['w', 'in']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['K', 'M'],
        factors_constraints = {'M': 4, 'K': 4}
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
        factors_constraints = {'M': 1, 'K': 2, 'N': 2},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'M', 'N'], #WS,
        size = 655360, # number of entries
        value_access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 2},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 65536, # number of entries
        value_access_energy = 30.26, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 4, 'K': 1, 'N': 2},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 128, # number of entries
        value_access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 128, 'N': 1},
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
        dataflow_constraints = ['K', 'M', 'N'],
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
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: bert large FF1
arch_simba_zigzag_salsa_ff1 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 2, 'K': 1, 'N': 8},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'M', 'N'], #WS,
        size = 655360, # number of entries
        value_access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 2},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['K', 'M'],
        factors_constraints = {'M': 4, 'K': 4}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = ['K', 'M', 'N'],
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
        dataflow_constraints = ['M', 'N', 'K'],
        size = 32768, # number of entries
        value_access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 4, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'M', 'N'], #WS,
        size = 655360, # number of entries
        value_access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 32, 'K': 1, 'N': 4},
        bypasses = ['w', 'in']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 128, # number of entries
        value_access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 64, 'N': 1},
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
        dataflow_constraints = ['K', 'M', 'N'],
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
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: MAESTRO-BLAS 1
arch_simba_zigzag_salsa_mb1 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 8, 'K': 4, 'N': 32},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'M', 'N'], #WS,
        size = 655360, # number of entries
        value_access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 2},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['K', 'M'],
        factors_constraints = {'M': 4, 'K': 4}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = ['K', 'M', 'N'],
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
        dataflow_constraints = ['M', 'N', 'K'],
        size = 32768, # number of entries
        value_access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 2, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'N', 'M'], #WS,
        size = 655360, # number of entries
        value_access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 32, 'K': 1, 'N': 2},
        bypasses = ['w', 'in']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 128, # number of entries
        value_access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 128, 'N': 1},
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
        dataflow_constraints = ['K', 'M', 'N'],
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
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: MAESTRO-BLAS 2
arch_simba_zigzag_salsa_mb2 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 4, 'N': 4},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'M', 'N'], #WS,
        size = 655360, # number of entries
        value_access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 2},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['K', 'M'],
        factors_constraints = {'M': 4, 'K': 4}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 655360, # number of entries
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
        dataflow_constraints = ['M', 'N', 'K'],
        size = 32768, # number of entries
        value_access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 2, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'N', 'M'], #WS,
        size = 65536, # number of entries
        value_access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 32, 'K': 1, 'N': 2},
        bypasses = ['w', 'in']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 128, # number of entries
        value_access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 128, 'N': 1},
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
        dataflow_constraints = ['K', 'M', 'N'],
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
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: MAESTRO-BLAS 3
arch_simba_zigzag_salsa_mb3 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'M', 'N'], #WS,
        size = 655360, # number of entries
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
        dataflow_constraints = ['K', 'M', 'N'],
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
        dataflow_constraints = ['M', 'N', 'K'],
        size = 32768, # number of entries
        value_access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 128, # number of entries
        value_access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 256, 'N': 1},
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
        dataflow_constraints = ['K', 'M', 'N'],
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
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: MAESTRO-BLAS 4
arch_simba_zigzag_salsa_mb4 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'M', 'N'], #WS,
        size = 6553600, # number of entries
        value_access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 2},
        bypasses = ['w', 'in']
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'M', 'N'], #WS,
        size = 6553600, # number of entries
        value_access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 8},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['K', 'M'],
        factors_constraints = {'M': 2, 'K': 8}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 65536, # number of entries
        value_access_energy = 30.26, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 8},
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
        dataflow_constraints = ['M', 'N', 'K'],
        size = 32768, # number of entries
        value_access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 128, # number of entries
        value_access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 32, 'N': 1},
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
        dataflow_constraints = ['K', 'M', 'N'],
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
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: MAESTRO-BLAS 5
arch_simba_zigzag_salsa_mb5 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'M', 'N'], #WS,
        size = 655360, # number of entries
        value_access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['K', 'M'],
        factors_constraints = {'M': 4, 'K': 4}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 65536, # number of entries
        value_access_energy = 30.26, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 4, 'K': 1, 'N': 1},
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
        dataflow_constraints = ['M', 'N', 'K'],
        size = 32768, # number of entries
        value_access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 16, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 128, # number of entries
        value_access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 8, 'K': 64, 'N': 1},
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
        dataflow_constraints = ['K', 'M', 'N'],
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
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: MAESTRO-BLAS 6
arch_simba_zigzag_salsa_mb6 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['K', 'M', 'N'], #WS,
        size = 655360, # number of entries
        value_access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['K', 'M'],
        factors_constraints = {'M': 4, 'K': 4}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = ['K', 'M', 'N'],
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
        dataflow_constraints = ['N', 'M', 'K'],
        size = 32768, # number of entries
        value_access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 32, 'K': 1, 'N': 4},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['K', 'M', 'N'],
        size = 128, # number of entries
        value_access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 16, 'N': 1},
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
        dataflow_constraints = ['K', 'M', 'N'],
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
    )], coupling=gemm_coupling)


# >>> TPU <<<

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: bert large KQV
arch_tpu_zigzag_loma_kqv = Arch([
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
        dataflow_constraints = ['M', 'K', 'N'],
        size = 8*2**30, # number of entries
        value_access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 4, 'K': 1, 'N': 16},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 24*(2**20), # number of entries
        value_access_energy = 19.66, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 2, 'N': 1},
        bypasses = ['w', 'out']
    ),
    #MemLevel(
    #    name = "WeightsFIFO",
    #    dataflow_constraints = ['N', 'K', 'M'],
    #    size = 4*2**16, # number of entries
    #    value_access_energy = 2.05, # per operand/scalar access (pJ)
    #    bandwidth = 8, # operands per cycle (shared)
    #    factors_constraints = {'M': 1, 'K': 1, 'N': 1},
    #    bypasses = ['in', 'out']
    #),
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
        value_access_energy = 3.03, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_constraints = {'M': 3, 'K': 1, 'N': 1},
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
        size = 4, # number of entries
        value_access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_constraints = {'M': 1, 'K': 2, 'N': 256},
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

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: bert large KTQ
arch_tpu_zigzag_loma_ktq = Arch([
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
        dataflow_constraints = ['M', 'K', 'N'],
        size = 8*2**30, # number of entries
        value_access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 24*(2**20), # number of entries
        value_access_energy = 19.66, # per operand (pJ)
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
        factors_constraints = {'M': 8, 'K': 1, 'N': 16},
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
        value_access_energy = 3.03, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "SACols",
        dim = WS[1],
        mesh = 256,
        factors_constraints = {'K': 64}
    ),
    MemLevel(
        name = "Register",
        dataflow_constraints = WS,
        size = 4, # number of entries
        value_access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_constraints = {'M': 2, 'K': 1, 'N': 256},
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

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: bert large VScores
arch_tpu_zigzag_loma_vscores = Arch([
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
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "WeightsFIFO",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 4*2**16, # number of entries
        value_access_energy = 2.05, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 24*(2**20), # number of entries
        value_access_energy = 19.66, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 8, 'N': 16},
        bypasses = ['w', 'out']
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
        value_access_energy = 3.03, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
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
        size = 4, # number of entries
        value_access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_constraints = {'M': 1, 'K': 2, 'N': 256},
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

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: bert large FF1
arch_tpu_zigzag_loma_ff1 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 8*2**30, # number of entries
        value_access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 16},
        bypasses = ['w']
    ),
    MemLevel(
        name = "WeightsDRAM",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 8*2**30, # number of entries
        value_access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 4, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    #MemLevel(
    #    name = "WeightsFIFO",
    #    dataflow_constraints = ['N', 'K', 'M'],
    #    size = 4*2**16, # number of entries
    #    value_access_energy = 2.05, # per operand/scalar access (pJ)
    #    bandwidth = 8, # operands per cycle (shared)
    #    factors_constraints = {'M': 1, 'K': 1, 'N': 1},
    #    bypasses = ['in', 'out']
    #),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 24*(2**20), # number of entries
        value_access_energy = 19.66, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 2, 'N': 1},
        bypasses = ['w', 'out']
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
        value_access_energy = 3.03, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_constraints = {'M': 4, 'K': 1, 'N': 1},
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
        size = 4, # number of entries
        value_access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_constraints = {'M': 1, 'K': 2, 'N': 256},
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

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: MAESTRO-BLAS 1
arch_tpu_zigzag_loma_mb1 = Arch([
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
        dataflow_constraints = ['K', 'N', 'M'],
        size = 8*2**30, # number of entries
        value_access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 4, 'N': 32},
        bypasses = ['in', 'out']
    ),
    #MemLevel(
    #    name = "WeightsFIFO",
    #    dataflow_constraints = ['N', 'K', 'M'],
    #    size = 4*2**16, # number of entries
    #    value_access_energy = 2.05, # per operand/scalar access (pJ)
    #    bandwidth = 8, # operands per cycle (shared)
    #    factors_constraints = {'M': 1, 'K': 1, 'N': 1},
    #    bypasses = ['in', 'out']
    #),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 24*(2**20), # number of entries
        value_access_energy = 19.66, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 8, 'K': 1, 'N': 1},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "SARows",
        dim = WS[0],
        mesh = 256,
        factors_constraints = {'M': 256}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 4096, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        value_access_energy = 3.03, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_constraints = {'M': 4, 'K': 8, 'N': 1},
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
        size = 4, # number of entries
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
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: MAESTRO-BLAS 2
arch_tpu_zigzag_loma_mb2 = Arch([
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
        factors_constraints = {'M': 2, 'K': 8, 'N': 1},
        bypasses = ['in', 'out']
    ),
    #MemLevel(
    #    name = "WeightsFIFO",
    #    dataflow_constraints = ['N', 'K', 'M'],
    #    size = 4*2**16, # number of entries
    #    value_access_energy = 2.05, # per operand/scalar access (pJ)
    #    bandwidth = 8, # operands per cycle (shared)
    #    factors_constraints = {'M': 1, 'K': 1, 'N': 1},
    #    bypasses = ['in', 'out']
    #),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 24*(2**20), # number of entries
        value_access_energy = 19.66, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 4, 'N': 1},
        bypasses = ['w', 'out']
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
        value_access_energy = 3.03, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_constraints = {'M': 1, 'K': 1, 'N': 1024},
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
        size = 4, # number of entries
        value_access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_constraints = {'M': 2, 'K': 1, 'N': 1},
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

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: MAESTRO-BLAS 3
arch_tpu_zigzag_loma_mb3 = Arch([
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
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "WeightsFIFO",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 4*2**16, # number of entries
        value_access_energy = 2.05, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 24*(2**20), # number of entries
        value_access_energy = 19.66, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 8, 'N': 1},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "SARows",
        dim = WS[0],
        mesh = 256,
        factors_constraints = {'M': 8}
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
        size = 4, # number of entries
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
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: MAESTRO-BLAS 4
arch_tpu_zigzag_loma_mb4 = Arch([
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
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "WeightsFIFO",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 4*2**16, # number of entries
        value_access_energy = 2.05, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 24*(2**20), # number of entries
        value_access_energy = 19.66, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 16},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "SARows",
        dim = WS[0],
        mesh = 256,
        factors_constraints = {'M': 8}
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
        dim = WS[1],
        mesh = 256,
        factors_constraints = {'K': 256}
    ),
    MemLevel(
        name = "Register",
        dataflow_constraints = WS,
        size = 4, # number of entries
        value_access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_constraints = {'M': 1, 'K': 2, 'N': 512},
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

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: MAESTRO-BLAS 5
arch_tpu_zigzag_loma_mb5 = Arch([
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
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    #MemLevel(
    #    name = "WeightsFIFO",
    #    dataflow_constraints = ['N', 'K', 'M'],
    #    size = 4*2**16, # number of entries
    #    value_access_energy = 2.05, # per operand/scalar access (pJ)
    #    bandwidth = 8, # operands per cycle (shared)
    #    factors_constraints = {'M': 1, 'K': 1, 'N': 1},
    #    bypasses = ['in', 'out']
    #),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 24*(2**20), # number of entries
        value_access_energy = 19.66, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 4, 'N': 1},
        bypasses = ['w', 'out']
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
        value_access_energy = 3.03, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_constraints = {'M': 32, 'K': 1, 'N': 1},
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
        size = 4, # number of entries
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
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: MAESTRO-BLAS 6
arch_tpu_zigzag_loma_mb6 = Arch([
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
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "WeightsFIFO",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 4*2**16, # number of entries
        value_access_energy = 2.05, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 24*(2**20), # number of entries
        value_access_energy = 19.66, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w', 'out']
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
        value_access_energy = 3.03, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_constraints = {'M': 1, 'K': 1, 'N': 256},
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
        size = 4, # number of entries
        value_access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_constraints = {'M': 2, 'K': 1, 'N': 1},
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

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: bert large KQV
arch_tpu_zigzag_salsa_kqv = Arch([
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
        dataflow_constraints = ['M', 'K', 'N'],
        size = 8*2**30, # number of entries
        value_access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 2},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 24*(2**20), # number of entries
        value_access_energy = 19.66, # per operand (pJ)
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
        factors_constraints = {'M': 1, 'K': 1, 'N': 2},
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
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: bert large KTQ
arch_tpu_zigzag_salsa_ktq = Arch([
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
        dataflow_constraints = ['M', 'K', 'N'],
        size = 8*2**30, # number of entries
        value_access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 24*(2**20), # number of entries
        value_access_energy = 19.66, # per operand (pJ)
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
        factors_constraints = {'M': 16, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "Register",
        dataflow_constraints = WS,
        size = 2**20, # number of entries
        value_access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_constraints = {'M': 1, 'K': 1, 'N': 2},
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
        value_access_energy = 3.03, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_constraints = {'M': 1, 'K': 1, 'N': 2048},
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "SACols",
        dim = WS[1],
        mesh = 256,
        factors_constraints = {'K': 64}
    ),
    ComputeLevel(
        name = "Compute",
        dim= WS[2],
        mesh = 1,
        compute_energy = 0.15, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: bert large VScores
arch_tpu_zigzag_salsa_vscores = Arch([
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
        dataflow_constraints = ['M', 'K', 'N'],
        size = 8*2**30, # number of entries
        value_access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "WeightsFIFO",
        dataflow_constraints = ['M', 'N', 'K'],
        size = 4*2**16, # number of entries
        value_access_energy = 2.05, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 2, 'N': 4},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 24*(2**20), # number of entries
        value_access_energy = 19.66, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w', 'out']
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
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: bert large FF1
arch_tpu_zigzag_salsa_ff1 = Arch([
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
        dataflow_constraints = ['M', 'K', 'N'],
        size = 8*2**30, # number of entries
        value_access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 2},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 24*(2**20), # number of entries
        value_access_energy = 19.66, # per operand (pJ)
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
        factors_constraints = {'M': 1, 'K': 1, 'N': 2},
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
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: MAESTRO-BLAS 1
arch_tpu_zigzag_salsa_mb1 = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['K', 'N', 'M'],
        size = 8*2**30, # number of entries
        value_access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 4, 'N': 8},
        bypasses = ['w']
    ),
    MemLevel(
        name = "WeightsDRAM",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 8*2**30, # number of entries
        value_access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['N', 'M', 'K'],
        size = 24*(2**20), # number of entries
        value_access_energy = 19.66, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 32, 'K': 2, 'N': 1},
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
        dim = WS[0],
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
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: MAESTRO-BLAS 2
arch_tpu_zigzag_salsa_mb2 = Arch([
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
        dataflow_constraints = ['M', 'K', 'N'],
        size = 8*2**30, # number of entries
        value_access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 4, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['N', 'M', 'K'],
        size = 24*(2**20), # number of entries
        value_access_energy = 19.66, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 4, 'K': 2, 'N': 1},
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
        dim = WS[0],
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
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: MAESTRO-BLAS 3
arch_tpu_zigzag_salsa_mb3 = Arch([
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
        dataflow_constraints = ['M', 'K', 'N'],
        size = 8*2**30, # number of entries
        value_access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 24*(2**20), # number of entries
        value_access_energy = 19.66, # per operand (pJ)
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
        dim = WS[0],
        mesh = 256,
        factors_constraints = {'M': 8}
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
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: MAESTRO-BLAS 4
arch_tpu_zigzag_salsa_mb4 = Arch([
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
        dataflow_constraints = ['M', 'K', 'N'],
        size = 8*2**30, # number of entries
        value_access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
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
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 24*(2**20), # number of entries
        value_access_energy = 19.66, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 2},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "SARows",
        dim = WS[0],
        mesh = 256,
        factors_constraints = {'M': 8}
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
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: MAESTRO-BLAS 5
arch_tpu_zigzag_salsa_mb5 = Arch([
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
        dataflow_constraints = ['M', 'K', 'N'],
        size = 8*2**30, # number of entries
        value_access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 24*(2**20), # number of entries
        value_access_energy = 19.66, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "SARows",
        dim = WS[0],
        mesh = 256,
        factors_constraints = {'M': 256}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 4096, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        value_access_energy = 3.03, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_constraints = {'M': 8, 'K': 4, 'N': 1},
        bypasses = ['in', 'w']
    ),
    MemLevel(
        name = "WeightsFIFO",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 4*2**16, # number of entries
        value_access_energy = 2.05, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 4, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
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
    )], coupling=gemm_coupling)

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: MAESTRO-BLAS 6
arch_tpu_zigzag_salsa_mb6 = Arch([
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
        dataflow_constraints = ['M', 'K', 'N'],
        size = 8*2**30, # number of entries
        value_access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['N', 'K', 'M'],
        size = 24*(2**20), # number of entries
        value_access_energy = 19.66, # per operand (pJ)
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
        dim = WS[0],
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