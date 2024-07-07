from levels import *

# DATAFLOWS (outer to inner):
WS = ['D', 'E', 'L']
OS = ['D', 'L', 'E']
IS = ['E', 'L', 'D']


# >>> GEMMINI <<<
# > WS version  <

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
        size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remember to account for operand size)
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


# >>> TRUE GEMMINI <<<

arch_true_gemmini = [
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
    FanoutLevel(
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
        factors_contraints = {'D': 1, 'E': 1, 'L': 16}, # the systolic array does a 16x16 matmul in this case
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
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
        factors_contraints = {'D': 1, 'E': 1, 'L': 1},
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
        dims = WS[:1],
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


# >>> SIMBA <<<

# C -> E
# M -> D
# P -> L
# REMEMBER: number of entries, aka size, is (width*depth)/(cluster_size*datawidth)
arch_simba = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = [],
        size = 2**64-1, # number of entries
        access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = [], #WS,
        size = 65536, # number of entries
        access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['E', 'D'],
        factors_contraints = {}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = [],
        size = 65536, # number of entries
        access_energy = 30.26, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "DistributionBuffers",
        mesh = 4,
        dims = ['D'],
        factors_contraints = {}
    ),
    MemLevel(
        name = "PEWeightBuffer",
        dataflow_constraints = [],
        size = 32768, # number of entries
        access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = [],
        size = 128, # number of entries
        access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {},
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "RegMac",
        mesh = 4,
        dims = ['E'],
        factors_contraints = {}
    ),
    MemLevel(
        name = "PEWeightRegs",
        dataflow_constraints = [],
        size = 1, # number of entries (64 in TL)
        access_energy = 0.70, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_contraints = {},
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


# >>>  TPU  <<<
# > 8bit mode <

# NOTE: use STEPS_TO_EXPLORE = 2

arch_tpu = [
    MemLevel(
        name = "DRAM",
        dataflow_constraints = [],
        size = 8*2**30, # number of entries
        access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {},
        bypasses = ['w']
    ),
    MemLevel(
        name = "WeightsDRAM",
        dataflow_constraints = [],
        size = 8*2**30, # number of entries
        access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = [],
        size = 24*(2**20), # number of entries
        access_energy = 19.66, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_contraints = {},
        # The Unified Buffer also stores outputs after the activation is
        # performed. Not modeled as we are only interested in GEMMs.
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WeightsFIFO",
        dataflow_constraints = [],
        size = 4*2**16, # number of entries
        access_energy = 2.11, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_contraints = {},
        bypasses = ['in', 'out']
    ),
    FanoutLevel(
        name = "SARows",
        dim = WS[0],
        mesh = 256,
        # pe_to_pe should be used, since the TPU uses a systolic array, but Timeloop
        # does not have this feature, so for sake of comparison, it is turned off
        #pe_to_pe = True, 
        factors_contraints = {'D': 256}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = [],
        size = 4096, # number of entries (PER ONE INSTANCE!!) (remember to account for operand size)
        access_energy = 3.03, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_contraints = {},
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "SACols",
        dim = WS[1],
        mesh = 256,
        # pe_to_pe should be used, since the TPU uses a systolic array, but Timeloop
        # does not have this feature, so for sake of comparison, it is turned off
        #pe_to_pe = True, 
        factors_contraints = {'E': 256}
    ),
    MemLevel(
        name = "Register",
        dataflow_constraints = WS,
        size = 2, # number of entries
        access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_contraints = {'D': 1, 'E': 1}, # L is free
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
