from levels import *

# DATAFLOWS (outer to inner):
WS = ['D', 'E', 'L']
OS = ['D', 'L', 'E']
IS = ['E', 'L', 'D']


# >>> GEMMINI <<<
arch_gemmini = [
    MemLevel(
        name = "DRAM",
        dataflow = ['L', 'E', 'D'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow = WS,
        size = 512*(2**10), # number of entries
        access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {},
        bypasses = ['out']
    ),
    #FanoutLevel2D(
    #    name = "SA",
    #    dimX = WS[0],
    #    dimY = WS[1],
    #    meshX = 16,
    #    meshY = 16,
    #    factors_contraints = {}
    #),
    FanoutLevel1D(
        name = "SARows",
        dim = WS[0],
        mesh = 16,
        pe_to_pe = True,
        factors_contraints = {'D': 16}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow = WS,
        size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        access_energy = 4.01, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {}, # the systolic array does a 16x16 matmul in this case
        bypasses = ['in', 'w']
    ),
    FanoutLevel1D(
        name = "SACols",
        dim = WS[1],
        mesh = 16,
        pe_to_pe = True,
        factors_contraints = {'E': 16}
    ),
    MemLevel(
        name = "Register",
        dataflow = WS,
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

# SOLUTION GIVEN BY TIMELOOP:
arch_gemmini_timeloop = [
    MemLevel(
        name = "DRAM",
        dataflow = ['L', 'E', 'D'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 64, 'E': 2, 'L': 16},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow = WS,
        size = 512*(2**10), # number of entries
        access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 8, 'L': 1},
        bypasses = ['out']
    ),
    FanoutLevel1D(
        name = "SARows",
        dim = WS[0],
        mesh = 16,
        pe_to_pe = True,
        factors_contraints = {'D': 16}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow = WS,
        size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        access_energy = 4.01, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 3, 'E': 4, 'L': 16}, # the systolic array does a 16x16 matmul in this case
        bypasses = ['in', 'w']
    ),
    FanoutLevel1D(
        name = "SACols",
        dim = WS[1],
        mesh = 16,
        pe_to_pe = True,
        factors_contraints = {'E': 16}
    ),
    MemLevel(
        name = "Register",
        dataflow = WS,
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

# SOLUTION GIVEN BY FF1:
arch_gemmini_factorflow_1 = [
    MemLevel(
        name = "DRAM",
        dataflow = ['L', 'E', 'D'],
        size = 2**64-1, # number of entries
        access_energy = 512.00, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 6, 'E': 2, 'L': 8},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow = WS,
        size = 512*(2**10), # number of entries
        access_energy = 57.04, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D':32, 'E': 1, 'L': 32},
        bypasses = ['out']
    ),
    FanoutLevel1D(
        name = "SARows",
        dim = WS[0],
        mesh = 16,
        pe_to_pe = True,
        factors_contraints = {'D': 16}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow = WS,
        size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        access_energy = 4.54, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 32, 'L': 1}, # the systolic array does a 16x16 matmul in this case
        bypasses = ['in', 'w']
    ),
    FanoutLevel1D(
        name = "SACols",
        dim = WS[1],
        mesh = 16,
        pe_to_pe = True,
        factors_contraints = {'E': 16}
    ),
    MemLevel(
        name = "Register",
        dataflow = WS,
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

# SOLUTION GIVEN BY FF2:
arch_gemmini_factorflow_2 = [
    MemLevel(
        name = "DRAM",
        dataflow = ['L', 'E', 'D'],
        size = 2**64-1, # number of entries
        access_energy = 512.00, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 6, 'E': 16, 'L': 1},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow = WS,
        size = 512*(2**10), # number of entries
        access_energy = 57.04, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D':32, 'E': 4, 'L': 1},
        bypasses = ['out']
    ),
    FanoutLevel1D(
        name = "SARows",
        dim = WS[0],
        mesh = 16,
        pe_to_pe = True,
        factors_contraints = {'D': 16}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow = WS,
        size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        access_energy = 4.54, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 256}, # the systolic array does a 16x16 matmul in this case
        bypasses = ['in', 'w']
    ),
    FanoutLevel1D(
        name = "SACols",
        dim = WS[1],
        mesh = 16,
        pe_to_pe = True,
        factors_contraints = {'E': 16}
    ),
    MemLevel(
        name = "Register",
        dataflow = WS,
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

# C -> E
# M -> D
# P -> L
arch_eyeriss = [
    MemLevel(
        name = "DRAM",
        dataflow = ['L', 'D', 'E'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow = WS,
        size = 16384*8, # number of entries
        access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {},
        bypasses = ['w']
    ),
    FanoutLevel1D(
        name = "SACols",
        dim = 'D',
        mesh = 14,
        pe_to_pe = False,
        factors_contraints = {} #{'D': 8}
    ),
    FanoutLevel1D(
        name = "SARows",
        dim = 'D', # one of D or E
        mesh = 12,
        pe_to_pe = False,
        factors_contraints = {} #{'D': 12}
    ),
    MemLevel(
        name = "InRegister",
        dataflow = WS,
        size = 12*2, # number of entries
        access_energy = 0.01, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WRegister",
        dataflow = WS,
        size = 192*2, # number of entries
        access_energy = 0.01, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'L': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow = WS,
        size = 16*2, # number of entries
        access_energy = 0.01, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'E': 1, 'L': 1},
        bypasses = ['in', 'w']
    ),
    ComputeLevel(
        name = "Compute",
        dataflow = WS[2],
        size = 1,
        compute_energy = 0.28, # per compute (pJ)
        cycles = 1,
        factors_contraints = {'L': 1}
    )]

# SOLUTION GIVEN BY TIMELOOP:
arch_eyeriss_timeloop = [
    MemLevel(
        name = "DRAM",
        dataflow = ['L', 'D', 'E'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 16, 'E': 64, 'L': 16},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow = WS,
        size = 16384*8, # number of entries
        access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 256},
        bypasses = ['w']
    ),
    FanoutLevel1D(
        name = "SACols",
        dim = 'D',
        mesh = 14,
        pe_to_pe = False,
        factors_contraints = {'D': 8}
    ),
    FanoutLevel1D(
        name = "SARows",
        dim = 'D', # one of D or E
        mesh = 12,
        pe_to_pe = False,
        factors_contraints = {'D': 12}
    ),
    MemLevel(
        name = "InRegister",
        dataflow = WS,
        size = 12*2, # number of entries
        access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WRegister",
        dataflow = WS,
        size = 192*2, # number of entries
        access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 16, 'L': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow = WS,
        size = 16*2, # number of entries
        access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 2, 'E': 1, 'L': 1},
        bypasses = ['in', 'w']
    ),
    ComputeLevel(
        name = "Compute",
        dataflow = WS[2],
        size = 1,
        compute_energy = 0.28, # per compute (pJ)
        cycles = 1,
        factors_contraints = {'L': 1}
    )]