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
arch_gemmini_zigzag_loma_kqv = [
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
arch_gemmini_zigzag_loma_vscores = [
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

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: bert large KTQ
arch_gemmini_zigzag_loma_ktq = [
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
        dataflow_constraints = ['D', 'L', 'E'],
        size = 512*(2**10), # number of entries
        access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D':256, 'E': 1, 'L': 16},
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
        access_energy = 4.01, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 4, 'L': 1}, # the systolic array does a 16x16 matmul in this case
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
        factors_contraints = {'D': 1, 'E': 1, 'L': 256},
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
# Comp: bert large FF1
arch_gemmini_zigzag_loma_ff1 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['D', 'E', 'L'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 16, 'E': 1, 'L': 16},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = ['D', 'E', 'L'],
        size = 512*(2**10), # number of entries
        access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D':16, 'E': 1, 'L': 1},
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
        dataflow_constraints = ['D', 'E', 'L'],
        size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        access_energy = 4.01, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 64, 'L': 1}, # the systolic array does a 16x16 matmul in this case
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
        dataflow_constraints = ['D', 'E', 'L'],
        size = 1, # number of entries
        access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared) -> 2 is a classic register which can be read & written in the same cc
        factors_contraints = {'D': 1, 'E': 1, 'L': 256},
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
# Comp: MAESTRO-BLAS 1
arch_gemmini_zigzag_loma_mb1 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 32, 'E': 16, 'L': 32},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = ['D', 'E', 'L'],
        size = 512*(2**10), # number of entries
        access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D':16, 'E': 1, 'L': 1},
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
        dataflow_constraints = ['D', 'E', 'L'],
        size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        access_energy = 4.01, # per operand (pJ)
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
        dataflow_constraints = ['D', 'E', 'L'],
        size = 1, # number of entries
        access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared) -> 2 is a classic register which can be read & written in the same cc
        factors_contraints = {'D': 1, 'E': 1, 'L': 256},
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
# Comp: MAESTRO-BLAS 2
arch_gemmini_zigzag_loma_mb2 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 32, 'L': 1},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = ['D', 'E', 'L'],
        size = 512*(2**10), # number of entries
        access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D':64, 'E': 1, 'L': 1},
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
        dataflow_constraints = ['D', 'E', 'L'],
        size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        access_energy = 4.01, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 16, 'L': 1}, # the systolic array does a 16x16 matmul in this case
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
        dataflow_constraints = ['D', 'E', 'L'],
        size = 1, # number of entries
        access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared) -> 2 is a classic register which can be read & written in the same cc
        factors_contraints = {'D': 1, 'E': 1, 'L': 1024},
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
# Comp: MAESTRO-BLAS 3
arch_gemmini_zigzag_loma_mb3 = [
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
        dataflow_constraints = ['D', 'E', 'L'],
        size = 512*(2**10), # number of entries
        access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D':1, 'E': 1, 'L': 1},
        bypasses = ['out']
    ),
    FanoutLevel(
        name = "SARows",
        dim = WS[0],
        mesh = 16,
        factors_contraints = {'D': 8}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = ['D', 'E', 'L'],
        size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        access_energy = 4.01, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 512, 'L': 1}, # the systolic array does a 16x16 matmul in this case
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
        dataflow_constraints = ['D', 'E', 'L'],
        size = 1, # number of entries
        access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared) -> 2 is a classic register which can be read & written in the same cc
        factors_contraints = {'D': 1, 'E': 1, 'L': 8},
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
# Comp: MAESTRO-BLAS 4
arch_gemmini_zigzag_loma_mb4 = [
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
        dataflow_constraints = ['D', 'E', 'L'],
        size = 512*(2**10), # number of entries
        access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D':1, 'E': 4, 'L': 1},
        bypasses = ['out', 'in']
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = ['D', 'E', 'L'],
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
        factors_contraints = {'D': 8}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = ['D', 'E', 'L'],
        size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        access_energy = 4.01, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 16, 'L': 1}, # the systolic array does a 16x16 matmul in this case
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
        dataflow_constraints = ['D', 'E', 'L'],
        size = 1, # number of entries
        access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared) -> 2 is a classic register which can be read & written in the same cc
        factors_contraints = {'D': 1, 'E': 1, 'L': 512},
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
# Comp: MAESTRO-BLAS 5
arch_gemmini_zigzag_loma_mb5 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 16, 'E': 1, 'L': 1},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = ['D', 'E', 'L'],
        size = 512*(2**10), # number of entries
        access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D':1, 'E': 16, 'L': 1},
        bypasses = ['out', 'w']
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = ['D', 'E', 'L'],
        size = 512*(2**10), # number of entries
        access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D':1, 'E': 1, 'L': 1},
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
        dataflow_constraints = ['D', 'E', 'L'],
        size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        access_energy = 4.01, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 32, 'E': 4, 'L': 1}, # the systolic array does a 16x16 matmul in this case
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
        dataflow_constraints = ['D', 'E', 'L'],
        size = 1, # number of entries
        access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared) -> 2 is a classic register which can be read & written in the same cc
        factors_contraints = {'D': 1, 'E': 1, 'L': 8},
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
# Comp: MAESTRO-BLAS 6
arch_gemmini_zigzag_loma_mb6 = [
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
        dataflow_constraints = ['D', 'E', 'L'],
        size = 512*(2**10), # number of entries
        access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D':8, 'E': 1, 'L': 1},
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
        dataflow_constraints = ['D', 'E', 'L'],
        size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        access_energy = 4.01, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 4, 'E': 16, 'L': 1}, # the systolic array does a 16x16 matmul in this case
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
        dataflow_constraints = ['D', 'E', 'L'],
        size = 1, # number of entries
        access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared) -> 2 is a classic register which can be read & written in the same cc
        factors_contraints = {'D': 1, 'E': 1, 'L': 256},
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

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# NOTE: unfair compared to the Gemmini architecture in "architectures.py"
#       because SALSA can "split" memories at will. Hence compare only
#       with the Gemmini version in this file!
# Comp: bert large KQV
arch_gemmini_zigzag_salsa_kqv = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['D', 'E', 'L'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 8, 'E': 1, 'L': 32},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = ['D', 'E', 'L'],
        size = 512*(2**10), # number of entries
        access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D':2, 'E': 1, 'L': 1},
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
        dataflow_constraints = ['D', 'E', 'L'],
        size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        access_energy = 4.01, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 12, 'E': 64, 'L': 1}, # the systolic array does a 16x16 matmul in this case
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
        factors_contraints = {'D': 1, 'E': 1, 'L': 128},
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

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# NOTE: unfair compared to the Gemmini architecture in "architectures.py"
#       because SALSA can "split" memories at will. Hence compare only
#       with the Gemmini version in this file!
# Comp: bert large VScores
arch_gemmini_zigzag_salsa_vscores = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['D', 'E', 'L'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 8},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = ['D', 'E', 'L'],
        size = 512*(2**10), # number of entries
        access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D':1, 'E': 8, 'L': 1},
        bypasses = ['out', 'in']
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = ['D', 'E', 'L'],
        size = 512*(2**10), # number of entries
        access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D':1, 'E': 2, 'L': 1},
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
        dataflow_constraints = ['D', 'E', 'L'],
        size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        access_energy = 4.01, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 4, 'E': 16, 'L': 1}, # the systolic array does a 16x16 matmul in this case
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
        factors_contraints = {'D': 1, 'E': 1, 'L': 512},
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

# Modified architecture without constraints
arch_eyeriss_zigzag_compatible_2 = [
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
        factors_contraints = {},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = [], #WS,
        size = 16*2, # number of entries
        access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {},
        bypasses = ['in', 'w']
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = [], #WS,
        size = 12*2, # number of entries
        access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {},
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
arch_eyeriss_zigzag_loma_kqv = [
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
# Comp: bert large VScores
arch_eyeriss_zigzag_loma_vscores = [
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
arch_eyeriss_zigzag_loma_vscores = [
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

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: bert large KTQ
arch_eyeriss_zigzag_loma_ktq = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['D', 'L', 'E'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 32, 'E': 1, 'L': 256},
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
        dims = ['D', 'E'],
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
        factors_contraints = {'D': 16, 'E': 1, 'L': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = WS,
        size = 16*2, # number of entries
        access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 8, 'L': 1},
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
# Comp: bert large FF1
arch_eyeriss_zigzag_loma_ff1 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 32, 'E': 8, 'L': 256},
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
        dims = ['D', 'E'],
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
        factors_contraints = {'D': 16, 'E': 16, 'L': 1},
        bypasses = ['in', 'out']
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
        name = "OutRegister",
        dataflow_constraints = WS,
        size = 16*2, # number of entries
        access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 16},
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
# Comp: MAESTRO-BLAS 1
arch_eyeriss_zigzag_loma_mb1 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['D', 'L', 'E'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 64, 'E': 64, 'L': 256},
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
        dims = ['D', 'E'],
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
        dataflow_constraints = ['D', 'L', 'E'],
        size = 192*2, # number of entries
        access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 16, 'E': 1, 'L': 32},
        bypasses = ['in', 'out']
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
        name = "OutRegister",
        dataflow_constraints = WS,
        size = 16*2, # number of entries
        access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 16, 'L': 1},
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
# Comp: MAESTRO-BLAS 2
arch_eyeriss_zigzag_loma_mb2 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 16, 'E': 64, 'L': 64},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'L', 'D'],
        size = 16384*8, # number of entries
        access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 16},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['D', 'E'],
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
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 16*2, # number of entries
        access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 8, 'E': 16, 'L': 1},
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
# Comp: MAESTRO-BLAS 3
arch_eyeriss_zigzag_loma_mb3 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['D', 'L', 'E'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'L', 'D'],
        size = 16384*8, # number of entries
        access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 4, 'L': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['D', 'E'],
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
        factors_contraints = {'D': 1, 'E': 16, 'L': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 16*2, # number of entries
        access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 16, 'L': 4},
        bypasses = ['in', 'w']
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = WS,
        size = 12*2, # number of entries
        access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 2},
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
# Comp: MAESTRO-BLAS 4
arch_eyeriss_zigzag_loma_mb4 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['D', 'L', 'E'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 16},
        bypasses = []
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['D', 'E'],
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
        factors_contraints = {'D': 1, 'E': 4, 'L': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'L', 'D'],
        size = 16384*8, # number of entries
        access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 16},
        bypasses = ['w', 'in']
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'L', 'D'],
        size = 16384*8, # number of entries
        access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 32},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = WS,
        size = 16*2, # number of entries
        access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 8, 'L': 1},
        bypasses = ['in', 'w']
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = WS,
        size = 12*2, # number of entries
        access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 4, 'L': 1},
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
# Comp: MAESTRO-BLAS 5
arch_eyeriss_zigzag_loma_mb5 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['D', 'L', 'E'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 64, 'E': 1, 'L': 1},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'L', 'D'],
        size = 16384*8, # number of entries
        access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 16, 'L': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['D', 'E'],
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
        factors_contraints = {'D': 16, 'E': 1, 'L': 2},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = WS,
        size = 16*2, # number of entries
        access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 8, 'L': 1},
        bypasses = ['in', 'w']
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = WS,
        size = 12*2, # number of entries
        access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 4},
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
# Comp: MAESTRO-BLAS 6
arch_eyeriss_zigzag_loma_mb6 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['D', 'L', 'E'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'L', 'D'],
        size = 16384*8, # number of entries
        access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 16, 'E': 1, 'L': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'L', 'D'],
        size = 16384*8, # number of entries
        access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w', 'in']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['D', 'E'],
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
        factors_contraints = {'D': 1, 'E': 1, 'L': 256},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = WS,
        size = 16*2, # number of entries
        access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 8, 'L': 1},
        bypasses = ['in', 'w']
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 12*2, # number of entries
        access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 4, 'E': 4, 'L': 1},
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

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: bert large KQV
arch_eyeriss_zigzag_salsa_kqv = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['L', 'D', 'E'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 16, 'E': 2, 'L': 32},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'L', 'D'],
        size = 16384*8, # number of entries
        access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 4, 'E': 1, 'L': 1},
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
        dataflow_constraints = ['D', 'L', 'E'],
        size = 192*2, # number of entries
        access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 2, 'E': 1, 'L': 16},
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
        dataflow_constraints = ['E', 'D', 'L'],
        size = 12*2, # number of entries
        access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 2, 'E': 2, 'L': 8},
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

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: bert large KTQ
arch_eyeriss_zigzag_salsa_ktq = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['L', 'D', 'E'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 16, 'E': 1, 'L': 4},
        bypasses = []
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['L', 'D', 'E'],
        size = 192*2*8*8, # number of entries
        access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 4, 'E': 1, 'L': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['L', 'D', 'E'],
        size = 16384*8, # number of entries
        access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 4, 'E': 1, 'L': 128},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['D', 'E'],
        mesh = 14,
        factors_contraints = {'D': 1, 'E': 8}
    ),
    FanoutLevel(
        name = "SARows",
        dim = 'D', # one of D or E
        mesh = 12,
        factors_contraints = {'D': 8}
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = ['D', 'E', 'L'],
        size = 16*2, # number of entries
        access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 4, 'L': 1},
        bypasses = ['in', 'w']
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 12*2, # number of entries
        access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 2, 'E': 2, 'L': 8},
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

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: bert large VScores
arch_eyeriss_zigzag_salsa_vscores = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['L', 'D', 'E'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 16, 'L': 8},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'L', 'D'],
        size = 16384*8, # number of entries
        access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 2},
        bypasses = ['w', 'in']
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'L', 'D'],
        size = 16384*8, # number of entries
        access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['D', 'E'],
        mesh = 14,
        factors_contraints = {'D': 1, 'E': 8}
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
        factors_contraints = {'D': 4, 'E': 1, 'L': 32},
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
        dataflow_constraints = ['E', 'D', 'L'],
        size = 12*2, # number of entries
        access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 2, 'E': 2, 'L': 8},
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

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: bert large FF1
arch_eyeriss_zigzag_salsa_ff1 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['L', 'D', 'E'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 16, 'E': 2, 'L': 32},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'L', 'D'],
        size = 16384*8, # number of entries
        access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 8, 'E': 1, 'L': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['D', 'E'],
        mesh = 14,
        factors_contraints = {'D': 1, 'E': 8}
    ),
    FanoutLevel(
        name = "SARows",
        dim = 'D', # one of D or E
        mesh = 12,
        factors_contraints = {'D': 8}
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['D', 'L', 'E'],
        size = 192*2, # number of entries
        access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 2, 'E': 1, 'L': 16},
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
        dataflow_constraints = ['E', 'D', 'L'],
        size = 12*2, # number of entries
        access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 2, 'E': 2, 'L': 8},
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

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: MAESTRO-BLAS 1
arch_eyeriss_zigzag_salsa_mb1 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['D', 'L', 'E'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 32, 'E': 16, 'L': 64},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'L', 'D'],
        size = 16384*8, # number of entries
        access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 8, 'E': 1, 'L': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['D', 'E'],
        mesh = 14,
        factors_contraints = {'D': 1, 'E': 8}
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
        factors_contraints = {'D': 2, 'E': 1, 'L': 16},
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
        dataflow_constraints = ['D', 'E', 'L'],
        size = 12*2, # number of entries
        access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 2, 'E': 2, 'L': 8},
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

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: MAESTRO-BLAS 2
arch_eyeriss_zigzag_salsa_mb2 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['D', 'L', 'E'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 4, 'E': 16, 'L': 8},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'L', 'D'],
        size = 16384*8, # number of entries
        access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 8, 'E': 1, 'L': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['D', 'E'],
        mesh = 14,
        factors_contraints = {'D': 1, 'E': 8}
    ),
    FanoutLevel(
        name = "SARows",
        dim = 'D', # one of D or E
        mesh = 12,
        factors_contraints = {'D': 8}
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['D', 'L', 'E'],
        size = 192*2, # number of entries
        access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 2, 'E': 1, 'L': 16},
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
        dataflow_constraints = ['D', 'E', 'L'],
        size = 12*2, # number of entries
        access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 2, 'E': 2, 'L': 8},
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

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: MAESTRO-BLAS 3
arch_eyeriss_zigzag_salsa_mb3 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'L', 'D'],
        size = 16384*8, # number of entries
        access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['D', 'E'],
        mesh = 14,
        factors_contraints = {'D': 1, 'E': 8}
    ),
    FanoutLevel(
        name = "SARows",
        dim = 'D', # one of D or E
        mesh = 12,
        factors_contraints = {'D': 8}
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 16*2, # number of entries
        access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 4, 'L': 1},
        bypasses = ['in', 'w']
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['L', 'D', 'E'],
        size = 192*2, # number of entries
        access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 128, 'L': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 12*2, # number of entries
        access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 2, 'L': 8},
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

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: MAESTRO-BLAS 4
arch_eyeriss_zigzag_salsa_mb4 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'L', 'D'],
        size = 16384*8, # number of entries
        access_energy = 2.02, # per operand (pJ)
        bandwidth = 16, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 128},
        bypasses = ['w', 'in']
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'L', 'D'],
        size = 16384*8, # number of entries
        access_energy = 2.02, # per operand (pJ)
        bandwidth = 16, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 4},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['D', 'E'],
        mesh = 14,
        factors_contraints = {'D': 1, 'E': 8}
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
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 16*2, # number of entries
        access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 128, 'L': 1},
        bypasses = ['in', 'w']
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 12*2, # number of entries
        access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 16},
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

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: MAESTRO-BLAS 5
arch_eyeriss_zigzag_salsa_mb5 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'L', 'D'],
        size = 16384*8, # number of entries
        access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 512, 'E': 1, 'L': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['D', 'E'],
        mesh = 14,
        factors_contraints = {'D': 1, 'E': 8}
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
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 16*2, # number of entries
        access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 64, 'L': 1},
        bypasses = ['in', 'w']
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = ['D', 'E', 'L'],
        size = 12*2, # number of entries
        access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 2, 'E': 2, 'L': 8},
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

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: MAESTRO-BLAS 6
arch_eyeriss_zigzag_salsa_mb6 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 2, 'E': 1, 'L': 1},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'L', 'D'],
        size = 16384*8, # number of entries
        access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 4, 'E': 1, 'L': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        dims = ['D', 'E'],
        mesh = 14,
        factors_contraints = {'D': 1, 'E': 8}
    ),
    FanoutLevel(
        name = "SARows",
        dim = 'D', # one of D or E
        mesh = 12,
        factors_contraints = {'D': 8}
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['D', 'L', 'E'],
        size = 192*2, # number of entries
        access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 4, 'E': 1, 'L': 32},
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
        dataflow_constraints = ['E', 'D', 'L'],
        size = 12*2, # number of entries
        access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 2, 'E': 2, 'L': 8},
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


# >>> SIMBA <<<

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: bert large KQV
arch_simba_zigzag_loma_kqv = [
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
arch_simba_zigzag_loma_vscores = [
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

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: bert large KTQ
arch_simba_zigzag_loma_ktq = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['D', 'L', 'E'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 16},
        bypasses = []
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
        dims = ['D'],
        factors_contraints = {'D': 4}
    ),
    MemLevel(
        name = "PEWeightBuffer",
        dataflow_constraints = ['D', 'L', 'E'],
        size = 32768, # number of entries
        access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 64, 'E': 1, 'L': 1},
        bypasses = ['in', 'out']
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
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 128, # number of entries
        access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 16, 'L': 1},
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

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: bert large FF1
arch_simba_zigzag_loma_ff1 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['D', 'L', 'E'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 4, 'E': 1, 'L': 256},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'D', 'L'], #WS,
        size = 65536, # number of entries
        access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w', 'in']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['D'],
        factors_contraints = {'D': 16,}
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
        dims = ['D'],
        factors_contraints = {'D': 4}
    ),
    MemLevel(
        name = "PEWeightBuffer",
        dataflow_constraints = ['L', 'D', 'E'],
        size = 32768, # number of entries
        access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 16, 'E': 16, 'L': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 128, # number of entries
        access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 16, 'L': 1},
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
# Comp: MAESTRO-BLAS 1
arch_simba_zigzag_loma_mb1 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['E', 'L', 'D'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 8, 'E': 16, 'L': 256},
        bypasses = []
    ),
    #MemLevel(
    #    name = "GlobalBuffer",
    #    dataflow_constraints = ['E', 'D', 'L'], #WS,
    #    size = 65536, # number of entries
    #    access_energy = 1.85, # per operand (pJ)
    #    bandwidth = 2**10, # operands per cycle (shared)
    #    factors_contraints = {'D': 1, 'E': 1, 'L': 1},
    #    bypasses = ['w', 'in']
    #),
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
        dims = ['D'],
        factors_contraints = {'D': 4}
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
        dataflow_constraints = ['E', 'D', 'L'],
        size = 128, # number of entries
        access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 128, 'L': 1},
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
        factors_contraints = {'D': 1, 'E': 1, 'L': 32},
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
# Comp: MAESTRO-BLAS 2
arch_simba_zigzag_loma_mb2 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['D', 'L', 'E'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 4, 'E': 1, 'L': 16},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'D', 'L'], #WS,
        size = 65536, # number of entries
        access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w', 'in']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['D'],
        factors_contraints = {'D': 16}
    ),
    FanoutLevel(
        name = "DistributionBuffers",
        mesh = 4,
        dims = ['D'],
        factors_contraints = {'D': 4}
    ),
    MemLevel(
        name = "PEWeightBuffer",
        dataflow_constraints = ['L', 'D', 'E'],
        size = 32768, # number of entries
        access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 16, 'L': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = WS,
        size = 65536, # number of entries
        access_energy = 30.26, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 64},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 128, # number of entries
        access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 4, 'E': 128, 'L': 1},
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

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: MAESTRO-BLAS 3
arch_simba_zigzag_loma_mb3 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['D', 'L', 'E'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'D', 'L'], #WS,
        size = 65536, # number of entries
        access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w', 'in']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['E', 'D'],
        factors_contraints = {'D': 2, 'E': 8}
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
        dims = ['D'],
        factors_contraints = {'D': 4}
    ),
    MemLevel(
        name = "PEWeightBuffer",
        dataflow_constraints = ['L', 'D', 'E'],
        size = 32768, # number of entries
        access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 16, 'L': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 128, # number of entries
        access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 16, 'L': 1},
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
        factors_contraints = {'D': 1, 'E': 1, 'L': 8},
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
# Comp: MAESTRO-BLAS 4
arch_simba_zigzag_loma_mb4 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['D', 'L', 'E'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'D', 'L'], #WS,
        size = 65536, # number of entries
        access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 32},
        bypasses = ['w', 'in']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['E', 'D'],
        factors_contraints = {'D': 2, 'E': 8}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = WS,
        size = 65536, # number of entries
        access_energy = 30.26, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 16},
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
        dataflow_constraints = ['L', 'D', 'E'],
        size = 32768, # number of entries
        access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 4, 'L': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 128, # number of entries
        access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 8, 'L': 1},
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
# Comp: MAESTRO-BLAS 5
arch_simba_zigzag_loma_mb5 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['D', 'L', 'E'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 16, 'E': 1, 'L': 1},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'D', 'L'], #WS,
        size = 65536, # number of entries
        access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w', 'in']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['E', 'D'],
        factors_contraints = {'D': 4, 'E': 4}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = WS,
        size = 65536, # number of entries
        access_energy = 30.26, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 2},
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
        dataflow_constraints = ['L', 'D', 'E'],
        size = 32768, # number of entries
        access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 32, 'E': 1, 'L': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['E', 'D', 'L'],
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
        factors_contraints = {'D': 1, 'E': 1, 'L': 4},
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
# Comp: MAESTRO-BLAS 6
arch_simba_zigzag_loma_mb6 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['D', 'L', 'E'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'D', 'L'], #WS,
        size = 6553600, # number of entries
        access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w', 'in']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['E', 'D'],
        factors_contraints = {'D': 4, 'E': 4}
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
        dims = ['D'],
        factors_contraints = {'D': 4}
    ),
    MemLevel(
        name = "PEWeightBuffer",
        dataflow_constraints = ['D', 'L', 'E'],
        size = 32768, # number of entries
        access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 4, 'E': 1, 'L': 256},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 128, # number of entries
        access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 8, 'E': 16, 'L': 1},
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

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: bert large KQV
arch_simba_zigzag_salsa_kqv = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['D', 'L', 'E'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 2, 'E': 1, 'L': 8},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'D', 'L'], #WS,
        size = 655360, # number of entries
        access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 2},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['E', 'D'],
        factors_contraints = {'D': 4, 'E': 4}
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
        dims = ['D'],
        factors_contraints = {'D': 4}
    ),
    MemLevel(
        name = "PEWeightBuffer",
        dataflow_constraints = ['D', 'L', 'E'],
        size = 32768, # number of entries
        access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 4, 'E': 1, 'L': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'L', 'D'], #WS,
        size = 655360, # number of entries
        access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 24, 'E': 1, 'L': 4},
        bypasses = ['w', 'in']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['E', 'D', 'L'],
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
        factors_contraints = {'D': 1, 'E': 1, 'L': 64},
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

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: bert large KTQ
arch_simba_zigzag_salsa_ktq = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['D', 'L', 'E'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'D', 'L'], #WS,
        size = 262144, # number of entries
        access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['E', 'D'],
        factors_contraints = {'D': 4, 'E': 4}
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
        dims = ['D'],
        factors_contraints = {'D': 4}
    ),
    MemLevel(
        name = "PEWeightBuffer",
        dataflow_constraints = ['D', 'L', 'E'],
        size = 32768, # number of entries
        access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 16, 'E': 1, 'L': 4},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'D', 'L'], #WS,
        size = 655360, # number of entries
        access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 16, 'E': 1, 'L': 16},
        bypasses = ['w', 'in']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 128, # number of entries
        access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 4, 'L': 1},
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
        factors_contraints = {'D': 1, 'E': 1, 'L': 64},
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

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: bert large VScores
arch_simba_zigzag_salsa_vscores = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['D', 'L', 'E'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 4},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'D', 'L'], #WS,
        size = 655360, # number of entries
        access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 2},
        bypasses = ['w', 'in']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['E', 'D'],
        factors_contraints = {'D': 4, 'E': 4}
    ),
    FanoutLevel(
        name = "DistributionBuffers",
        mesh = 4,
        dims = ['D'],
        factors_contraints = {'D': 4}
    ),
    MemLevel(
        name = "PEWeightBuffer",
        dataflow_constraints = ['D', 'E', 'L'],
        size = 32768, # number of entries
        access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 2, 'L': 2},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'D', 'L'], #WS,
        size = 655360, # number of entries
        access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 2},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 65536, # number of entries
        access_energy = 30.26, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 4, 'E': 1, 'L': 2},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 128, # number of entries
        access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 128, 'L': 1},
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
        dataflow_constraints = ['E', 'D', 'L'],
        size = 1, # number of entries
        access_energy = 0.70, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 64},
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

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: bert large FF1
arch_simba_zigzag_salsa_ff1 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['D', 'L', 'E'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 2, 'E': 1, 'L': 8},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'D', 'L'], #WS,
        size = 655360, # number of entries
        access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 2},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['E', 'D'],
        factors_contraints = {'D': 4, 'E': 4}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 65536, # number of entries
        access_energy = 30.26, # per operand (pJ)
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
        dataflow_constraints = ['D', 'L', 'E'],
        size = 32768, # number of entries
        access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 4, 'E': 1, 'L': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'D', 'L'], #WS,
        size = 655360, # number of entries
        access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 32, 'E': 1, 'L': 4},
        bypasses = ['w', 'in']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['E', 'D', 'L'],
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
        dataflow_constraints = ['E', 'D', 'L'],
        size = 1, # number of entries
        access_energy = 0.70, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 64},
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

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: MAESTRO-BLAS 1
arch_simba_zigzag_salsa_mb1 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['D', 'E', 'L'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 8, 'E': 4, 'L': 32},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'D', 'L'], #WS,
        size = 655360, # number of entries
        access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 2},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['E', 'D'],
        factors_contraints = {'D': 4, 'E': 4}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 65536, # number of entries
        access_energy = 30.26, # per operand (pJ)
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
        dataflow_constraints = ['D', 'L', 'E'],
        size = 32768, # number of entries
        access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 2, 'E': 1, 'L': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'L', 'D'], #WS,
        size = 655360, # number of entries
        access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 32, 'E': 1, 'L': 2},
        bypasses = ['w', 'in']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 128, # number of entries
        access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 128, 'L': 1},
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
        dataflow_constraints = ['E', 'D', 'L'],
        size = 1, # number of entries
        access_energy = 0.70, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 64},
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

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: MAESTRO-BLAS 2
arch_simba_zigzag_salsa_mb2 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['D', 'E', 'L'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 4, 'L': 4},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'D', 'L'], #WS,
        size = 655360, # number of entries
        access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 2},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['E', 'D'],
        factors_contraints = {'D': 4, 'E': 4}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 655360, # number of entries
        access_energy = 30.26, # per operand (pJ)
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
        dataflow_constraints = ['D', 'L', 'E'],
        size = 32768, # number of entries
        access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 2, 'E': 1, 'L': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'L', 'D'], #WS,
        size = 65536, # number of entries
        access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 32, 'E': 1, 'L': 2},
        bypasses = ['w', 'in']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 128, # number of entries
        access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 128, 'L': 1},
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
        dataflow_constraints = ['E', 'D', 'L'],
        size = 1, # number of entries
        access_energy = 0.70, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 64},
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

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: MAESTRO-BLAS 3
arch_simba_zigzag_salsa_mb3 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['D', 'L', 'E'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'D', 'L'], #WS,
        size = 655360, # number of entries
        access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['E', 'D'],
        factors_contraints = {'D': 2, 'E': 8}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 65536, # number of entries
        access_energy = 30.26, # per operand (pJ)
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
        dataflow_constraints = ['D', 'L', 'E'],
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
        factors_contraints = {'D': 1, 'E': 256, 'L': 1},
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
        dataflow_constraints = ['E', 'D', 'L'],
        size = 1, # number of entries
        access_energy = 0.70, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 8},
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

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: MAESTRO-BLAS 4
arch_simba_zigzag_salsa_mb4 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['D', 'L', 'E'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'D', 'L'], #WS,
        size = 6553600, # number of entries
        access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 2},
        bypasses = ['w', 'in']
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'D', 'L'], #WS,
        size = 6553600, # number of entries
        access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 8},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['E', 'D'],
        factors_contraints = {'D': 2, 'E': 8}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 65536, # number of entries
        access_energy = 30.26, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 8},
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
        dataflow_constraints = ['D', 'L', 'E'],
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
        factors_contraints = {'D': 1, 'E': 32, 'L': 1},
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
        dataflow_constraints = ['E', 'D', 'L'],
        size = 1, # number of entries
        access_energy = 0.70, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 64},
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

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: MAESTRO-BLAS 5
arch_simba_zigzag_salsa_mb5 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['D', 'L', 'E'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'D', 'L'], #WS,
        size = 655360, # number of entries
        access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['E', 'D'],
        factors_contraints = {'D': 4, 'E': 4}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 65536, # number of entries
        access_energy = 30.26, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 4, 'E': 1, 'L': 1},
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
        dataflow_constraints = ['D', 'L', 'E'],
        size = 32768, # number of entries
        access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 16, 'E': 1, 'L': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 128, # number of entries
        access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 8, 'E': 64, 'L': 1},
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
        dataflow_constraints = ['E', 'D', 'L'],
        size = 1, # number of entries
        access_energy = 0.70, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 8},
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

# SOLUTION GIVEN BY ZigZag WITH SALSA:
# Comp: MAESTRO-BLAS 6
arch_simba_zigzag_salsa_mb6 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['D', 'L', 'E'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['E', 'D', 'L'], #WS,
        size = 655360, # number of entries
        access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['E', 'D'],
        factors_contraints = {'D': 4, 'E': 4}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 65536, # number of entries
        access_energy = 30.26, # per operand (pJ)
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
        dataflow_constraints = ['L', 'D', 'E'],
        size = 32768, # number of entries
        access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 32, 'E': 1, 'L': 4},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 128, # number of entries
        access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 16, 'L': 1},
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
        dataflow_constraints = ['E', 'D', 'L'],
        size = 1, # number of entries
        access_energy = 0.70, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 64},
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
arch_tpu_zigzag_loma_kqv = [
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
        dataflow_constraints = ['D', 'E', 'L'],
        size = 8*2**30, # number of entries
        access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 4, 'E': 1, 'L': 16},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 24*(2**20), # number of entries
        access_energy = 19.66, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 2, 'L': 1},
        bypasses = ['w', 'out']
    ),
    #MemLevel(
    #    name = "WeightsFIFO",
    #    dataflow_constraints = ['L', 'E', 'D'],
    #    size = 4*2**16, # number of entries
    #    access_energy = 2.05, # per operand/scalar access (pJ)
    #    bandwidth = 8, # operands per cycle (shared)
    #    factors_contraints = {'D': 1, 'E': 1, 'L': 1},
    #    bypasses = ['in', 'out']
    #),
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
        factors_contraints = {'D': 3, 'E': 1, 'L': 1},
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
        size = 4, # number of entries
        access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_contraints = {'D': 1, 'E': 2, 'L': 256},
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
# Comp: bert large KTQ
arch_tpu_zigzag_loma_ktq = [
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
        dataflow_constraints = ['D', 'E', 'L'],
        size = 8*2**30, # number of entries
        access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 24*(2**20), # number of entries
        access_energy = 19.66, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WeightsFIFO",
        dataflow_constraints = ['D', 'E', 'L'],
        size = 4*2**16, # number of entries
        access_energy = 2.05, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 8, 'E': 1, 'L': 16},
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
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "SACols",
        dim = WS[1],
        mesh = 256,
        factors_contraints = {'E': 64}
    ),
    MemLevel(
        name = "Register",
        dataflow_constraints = WS,
        size = 4, # number of entries
        access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_contraints = {'D': 2, 'E': 1, 'L': 256},
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
arch_tpu_zigzag_loma_vscores = [
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
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 24*(2**20), # number of entries
        access_energy = 19.66, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 8, 'L': 16},
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
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
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
        size = 4, # number of entries
        access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_contraints = {'D': 1, 'E': 2, 'L': 256},
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
# Comp: bert large FF1
arch_tpu_zigzag_loma_ff1 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 8*2**30, # number of entries
        access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 16},
        bypasses = ['w']
    ),
    MemLevel(
        name = "WeightsDRAM",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 8*2**30, # number of entries
        access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 4, 'E': 1, 'L': 1},
        bypasses = ['in', 'out']
    ),
    #MemLevel(
    #    name = "WeightsFIFO",
    #    dataflow_constraints = ['L', 'E', 'D'],
    #    size = 4*2**16, # number of entries
    #    access_energy = 2.05, # per operand/scalar access (pJ)
    #    bandwidth = 8, # operands per cycle (shared)
    #    factors_contraints = {'D': 1, 'E': 1, 'L': 1},
    #    bypasses = ['in', 'out']
    #),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 24*(2**20), # number of entries
        access_energy = 19.66, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 2, 'L': 1},
        bypasses = ['w', 'out']
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
        factors_contraints = {'D': 4, 'E': 1, 'L': 1},
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
        size = 4, # number of entries
        access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_contraints = {'D': 1, 'E': 2, 'L': 256},
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
# Comp: MAESTRO-BLAS 1
arch_tpu_zigzag_loma_mb1 = [
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
        dataflow_constraints = ['E', 'L', 'D'],
        size = 8*2**30, # number of entries
        access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 4, 'L': 32},
        bypasses = ['in', 'out']
    ),
    #MemLevel(
    #    name = "WeightsFIFO",
    #    dataflow_constraints = ['L', 'E', 'D'],
    #    size = 4*2**16, # number of entries
    #    access_energy = 2.05, # per operand/scalar access (pJ)
    #    bandwidth = 8, # operands per cycle (shared)
    #    factors_contraints = {'D': 1, 'E': 1, 'L': 1},
    #    bypasses = ['in', 'out']
    #),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 24*(2**20), # number of entries
        access_energy = 19.66, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 8, 'E': 1, 'L': 1},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "SARows",
        dim = WS[0],
        mesh = 256,
        factors_contraints = {'D': 256}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = ['D', 'E', 'L'],
        size = 4096, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        access_energy = 3.03, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_contraints = {'D': 4, 'E': 8, 'L': 1},
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
        size = 4, # number of entries
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

# SOLUTION GIVEN BY ZigZag WITH LOMA:
# Comp: MAESTRO-BLAS 2
arch_tpu_zigzag_loma_mb2 = [
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
        factors_contraints = {'D': 2, 'E': 8, 'L': 1},
        bypasses = ['in', 'out']
    ),
    #MemLevel(
    #    name = "WeightsFIFO",
    #    dataflow_constraints = ['L', 'E', 'D'],
    #    size = 4*2**16, # number of entries
    #    access_energy = 2.05, # per operand/scalar access (pJ)
    #    bandwidth = 8, # operands per cycle (shared)
    #    factors_contraints = {'D': 1, 'E': 1, 'L': 1},
    #    bypasses = ['in', 'out']
    #),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 24*(2**20), # number of entries
        access_energy = 19.66, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 4, 'L': 1},
        bypasses = ['w', 'out']
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
        factors_contraints = {'D': 1, 'E': 1, 'L': 1024},
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
        size = 4, # number of entries
        access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_contraints = {'D': 2, 'E': 1, 'L': 1},
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
# Comp: MAESTRO-BLAS 3
arch_tpu_zigzag_loma_mb3 = [
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
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 24*(2**20), # number of entries
        access_energy = 19.66, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 8, 'L': 1},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "SARows",
        dim = WS[0],
        mesh = 256,
        factors_contraints = {'D': 8}
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
        size = 4, # number of entries
        access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_contraints = {'D': 1, 'E': 1, 'L': 8},
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
# Comp: MAESTRO-BLAS 4
arch_tpu_zigzag_loma_mb4 = [
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
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 24*(2**20), # number of entries
        access_energy = 19.66, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 16},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "SARows",
        dim = WS[0],
        mesh = 256,
        factors_contraints = {'D': 8}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = ['E', 'D', 'L'],
        size = 4096, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        access_energy = 3.03, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_contraints = {'D': 1, 'E': 2, 'L': 1},
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
        size = 4, # number of entries
        access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_contraints = {'D': 1, 'E': 2, 'L': 512},
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
# Comp: MAESTRO-BLAS 5
arch_tpu_zigzag_loma_mb5 = [
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
    #MemLevel(
    #    name = "WeightsFIFO",
    #    dataflow_constraints = ['L', 'E', 'D'],
    #    size = 4*2**16, # number of entries
    #    access_energy = 2.05, # per operand/scalar access (pJ)
    #    bandwidth = 8, # operands per cycle (shared)
    #    factors_contraints = {'D': 1, 'E': 1, 'L': 1},
    #    bypasses = ['in', 'out']
    #),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 24*(2**20), # number of entries
        access_energy = 19.66, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 4, 'L': 1},
        bypasses = ['w', 'out']
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
        factors_contraints = {'D': 32, 'E': 1, 'L': 1},
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
        size = 4, # number of entries
        access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_contraints = {'D': 1, 'E': 1, 'L': 8},
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
# Comp: MAESTRO-BLAS 6
arch_tpu_zigzag_loma_mb6 = [
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
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 24*(2**20), # number of entries
        access_energy = 19.66, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w', 'out']
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
        factors_contraints = {'D': 1, 'E': 1, 'L': 256},
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
        size = 4, # number of entries
        access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_contraints = {'D': 2, 'E': 1, 'L': 1},
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