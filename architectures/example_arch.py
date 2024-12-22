from computations import gemm_coupling
from levels import *
from arch import *

arch = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = [],
        size = 2**64-1, # number of entries
        value_access_energy = 64.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {},
        bypasses = []
    ),
    MemLevel(
        name = "Scratchpad",
        dataflow_constraints = [],
        size = 512*(2**10), # number of entries
        value_access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {'K<=': 128},
        bypasses = ['out']
    ),
    FanoutLevel(
        name = "PEs",
        #dim = 'M',
        dims = ['M', 'K'],
        mesh = 32,
        pe_to_pe = False, 
        factors_constraints = {'M<=': 8, 'K>=': 2}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = [],
        size = 256, # number of entries (PER ONE INSTANCE!!) (remeber to account for operand size)
        value_access_energy = 4.01, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N>=': 16}, # the systolic array does a 16x16 matmul in this case
        bypasses = ['in', 'w']
    ),
    MemLevel(
        name = "Registers",
        dataflow_constraints = ['M', 'K', 'N'],
        size = 4, # number of entries
        value_access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dims = ['N'],
        mesh = 2,
        compute_energy = 0.28, # per compute (pJ)
        cycles = 1,
        factors_constraints = {}
    )], coupling=gemm_coupling, name="Example Architecture")
