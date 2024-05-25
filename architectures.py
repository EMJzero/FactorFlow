from levels import *

# DATAFLOWS (outer to inner):
WS = ['D', 'E', 'L']
OS = ['D', 'L', 'E']
IS = ['E', 'L', 'D']

# TODO: currently a dataflow is always required, but in the absence of constraints and with
# permutations exploration enabled, such dataflow will be subject to change. It might be better
# to refactor it like this:
# - on MemLevels do not allow dataflow to be set, set if by default to ['D', 'E', 'L'] and then act according to constraints
# - add dataflow_constraints as an attribute
# - on FanoutLevel1D use "dim" to act as both constraint and exploration space. If dim has more than one dimension
#   additional ones will be explored, otherwise the only provided one is used.

# >>> GEMMINI <<<
arch_gemmini = [
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
        dataflow_constraints = [], #WS,
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

# SOLUTION GIVEN BY TIMELOOP:
arch_gemmini_timeloop = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 64, 'E': 2, 'L': 16},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = WS,
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
        dataflow_constraints = WS,
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

# SOLUTION GIVEN BY FF:
arch_gemmini_factorflow_1 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 2**64-1, # number of entries
        access_energy = 512.00, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 6, 'E': 2, 'L': 8},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = WS,
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
        dataflow_constraints = WS,
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

# SOLUTION GIVEN BY FF:
arch_gemmini_factorflow_2 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['L', 'E', 'D'],
        size = 2**64-1, # number of entries
        access_energy = 512.00, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 6, 'E': 16, 'L': 1},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = WS,
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
        dataflow_constraints = WS,
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

# C -> E
# M -> D
# P -> L
arch_eyeriss = [
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
    FanoutLevel1D(
        name = "SACols",
        mesh = 14,
        candidate_dims = WS,
        pe_to_pe = False,
        factors_contraints = {} #{'D': 8}
    ),
    FanoutLevel1D(
        name = "SARows",
        mesh = 12,
        candidate_dims = WS,
        pe_to_pe = False,
        factors_contraints = {} #{'D': 12}
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = [], #WS,
        size = 12*2, # number of entries
        access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w', 'out']
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
    ComputeLevel(
        name = "Compute",
        dataflow = WS[2],
        size = 1,
        compute_energy = 0.21, # per compute (pJ)
        cycles = 1,
        factors_contraints = {'L': 1}
    )]

# SOLUTION GIVEN BY TIMELOOP:
arch_eyeriss_timeloop = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['L', 'D', 'E'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 16, 'E': 64, 'L': 16},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = WS,
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
        dataflow_constraints = WS,
        size = 12*2, # number of entries
        access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = WS,
        size = 192*2, # number of entries
        access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 16, 'L': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = WS,
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
        compute_energy = 0.21, # per compute (pJ)
        cycles = 1,
        factors_contraints = {'L': 1}
    )]

# SOLUTION GIVEN BY FF:
arch_eyeriss_factorflow_1 = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['L', 'D', 'E'],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {'D': 8, 'E': 8, 'L': 16},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = WS,
        size = 16384*8, # number of entries
        access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 8, 'L': 256},
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
        dataflow_constraints = WS,
        size = 12*2, # number of entries
        access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = WS,
        size = 192*2, # number of entries
        access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 16, 'L': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = WS,
        size = 16*2, # number of entries
        access_energy = 1.34, # per operand (pJ)
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
# C -> E
# M -> D
# P -> L
arch_simba = [
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
    #TODO: there should be two dimensions in fanout here... maybe using here Fanout2D could allow for that? Simply add to Fanout2D the case
    FanoutLevel1D(
        name = "SACols",
        mesh = 14,
        candidate_dims = WS,
        pe_to_pe = False,
        factors_contraints = {} #{'D': 8}
    ),
    MemLevel(
        name = "InBuffer",
        dataflow_constraints = [], #WS,
        size = 65536, # number of entries
        access_energy = 0.69, # per operand (pJ)
        bandwidth = 2**64-1, # operands per cycle (shared)
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
        bypasses = ['w', 'out']
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
    ComputeLevel(
        name = "Compute",
        dataflow = WS[2],
        size = 1,
        compute_energy = 0.21, # per compute (pJ)
        cycles = 1,
        factors_contraints = {'L': 1}
    )]
