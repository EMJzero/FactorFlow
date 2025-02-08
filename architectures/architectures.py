from computations import gemm_coupling, conv_coupling, conv_coupling_with_stride
from levels import *
from arch import *

# DATAFLOWS (outer to inner) (GEMMs only):
WS = ['M', 'K', 'N']
OS = ['M', 'N', 'K']
IS = ['K', 'N', 'M']


# >>> GEMMINI <<<
# > WS version  <

arch_gemmini = Arch([
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
        size = 512*(2**10), # number of entries
        value_access_energy = 3.47, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {},
        bypasses = ['out']
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
        size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remember to account for operand size)
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
        dim = WS[2],
        mesh = 1,
        compute_energy = 0.28, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )], coupling=gemm_coupling, name="Gemmini")

arch_gemmini_conv = Arch([
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
        factors_constraints = {},
        bypasses = ['out']
    ),
    FanoutLevel(
        name = "SARows",
        dim = 'M',
        mesh = 16,
        # pe_to_pe should be used, since Gemmini uses a systolic array, but Timeloop
        # does not have this feature, so for sake of comparison, it is turned off
        #pe_to_pe = True,
        factors_constraints = {}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = [],
        size = (256//4)*(2**10)//16, # number of entries (PER ONE INSTANCE!!) (remember to account for operand size)
        value_access_energy = 4.01, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'R': 1, 'S': 1}, # the systolic array does a 16x16 matmul in this case
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "SACols",
        dim = 'C',
        mesh = 16,
        # pe_to_pe should be used, since Gemmini uses a systolic array, but Timeloop
        # does not have this feature, so for sake of comparison, it is turned off
        #pe_to_pe = True,
        factors_constraints = {}
    ),
    MemLevel(
        name = "Register",
        dataflow_constraints = ['M', 'Q', 'C', 'R', 'S', 'P'],
        size = 1, # number of entries
        value_access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'Q': 1, 'C': 1, 'R': 1, 'S': 1},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        mesh = 1,
        compute_energy = 0.28, # per compute (pJ)
        cycles = 1,
    )], coupling=conv_coupling_with_stride, name="Gemmini")


# >>> EYERISS <<<

# C -> K
# M -> M
# P -> N
arch_eyeriss = Arch([
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
        dims = WS[:1],
        factors_constraints = {} #{'M': 12}
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = [], #WS,
        size = 12*2, # number of entries
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'K': 1, 'N': 1},
        bypasses = ['w', 'out']
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
    ComputeLevel(
        name = "Compute",
        dim = WS[2],
        mesh = 1,
        compute_energy = 0.21, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )], coupling=gemm_coupling, name="Eyeriss")

arch_eyeriss_conv = Arch([
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
        name = "GlobalBuffer",
        dataflow_constraints = [],
        size = 16384*8, # number of entries
        value_access_energy = 2.02, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        mesh = 14,
        dims = ['Q', 'M'],
        factors_constraints = {}
    ),
    FanoutLevel(
        name = "SARows",
        mesh = 12,
        dims = ['S', 'C', 'M'],
        factors_constraints = {}
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = ['S', 'R', 'Q', 'P', 'C', 'M'],
        size = 12*2, # number of entries
        value_access_energy = 0.69, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'C': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['R', 'C', 'S', 'Q', 'P', 'M'],
        size = 192*2, # number of entries
        value_access_energy = 1.97, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'M': 1, 'P': 1, 'Q': 1, 'S': 1},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = ['M', 'S', 'R', 'Q', 'P', 'C'],
        size = 16*2, # number of entries
        value_access_energy = 1.34, # per operand (pJ)
        bandwidth = 4, # operands per cycle (shared)
        factors_constraints = {'C': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': 1},
        bypasses = ['in', 'w']
    ),
    ComputeLevel(
        name = "Compute",
        mesh = 1,
        compute_energy = 0.21, # per compute (pJ)
        cycles = 1,
    )], coupling=conv_coupling_with_stride, name="Eyeriss (conv)")


# >>> SIMBA <<<

# C -> K
# M -> M
# P -> N
# REMEMBER: number of entries, aka size, is (width*depth)/(cluster_size*datawidth)
arch_simba = Arch([
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
        name = "GlobalBuffer",
        dataflow_constraints = [], #WS,
        size = 65536, # number of entries
        value_access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['K', 'M'],
        factors_constraints = {}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = [],
        size = 65536, # number of entries
        value_access_energy = 30.26, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "DistributionBuffers",
        mesh = 4,
        dims = ['M'],
        factors_constraints = {}
    ),
    MemLevel(
        name = "PEWeightBuffer",
        dataflow_constraints = [],
        size = 32768, # number of entries
        value_access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = [],
        size = 128, # number of entries
        value_access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {},
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "RegMac",
        mesh = 4,
        dims = ['K'],
        factors_constraints = {}
    ),
    MemLevel(
        name = "PEWeightRegs",
        dataflow_constraints = [],
        size = 1, # number of entries (64 in TL)
        value_access_energy = 0.70, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim = WS[2],
        mesh = 1,
        compute_energy = 0.32, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )], coupling=gemm_coupling, name="Simba")

arch_simba_conv = Arch([
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
        name = "GlobalBuffer",
        dataflow_constraints = [], #WS,
        size = 65536, # number of entries
        value_access_energy = 1.85, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {},
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "PEs",
        mesh = 16,
        dims = ['M', 'C'],
        factors_constraints = {}
    ),
    MemLevel(
        name = "PEInputBuffer",
        dataflow_constraints = [],
        size = 65536, # number of entries
        value_access_energy = 30.26, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {},
        bypasses = ['w', 'out']
    ),
    FanoutLevel(
        name = "DistributionBuffers",
        mesh = 4,
        dims = ['M'],
        factors_constraints = {}
    ),
    MemLevel(
        name = "PEWeightBuffer",
        dataflow_constraints = [],
        size = 32768, # number of entries
        value_access_energy = 15.16, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "PEAccuBuffer",
        dataflow_constraints = [],
        size = 128, # number of entries
        value_access_energy = 3.93, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {},
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "RegMac",
        mesh = 4,
        dims = ['C'],
        factors_constraints = {}
    ),
    MemLevel(
        name = "PEWeightRegs",
        dataflow_constraints = [],
        size = 1, # number of entries (64 in TL)
        value_access_energy = 0.70, # per operand (pJ)
        bandwidth = 2**10, # operands per cycle (shared)
        factors_constraints = {},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        mesh = 1,
        compute_energy = 0.32, # per compute (pJ)
        cycles = 1,
    )], coupling=conv_coupling_with_stride, name="Simba")


# >>>  TPU  <<<
# > 8bit mode <

# NOTE: use STEPS_TO_EXPLORE = 2

arch_tpu = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = [],
        size = 8*2**30, # number of entries
        value_access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {},
        bypasses = ['w']
    ),
    MemLevel(
        name = "WeightsDRAM",
        dataflow_constraints = [],
        size = 8*2**30, # number of entries
        value_access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = [],
        size = 24*(2**20), # number of entries
        value_access_energy = 19.66, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {},
        # The Unified Buffer also stores outputs after the activation is
        # performed. Not modeled as we are only interested in GEMMs.
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WeightsFIFO",
        dataflow_constraints = [],
        size = 4*2**16, # number of entries
        value_access_energy = 2.11, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {},
        bypasses = ['in', 'out']
    ),
    FanoutLevel(
        name = "SARows",
        dim = WS[0],
        mesh = 256,
        # pe_to_pe should be used, since the TPU uses a systolic array, but Timeloop
        # does not have this feature, so for sake of comparison, it is turned off
        #pe_to_pe = True, 
        factors_constraints = {'M': 256}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = [],
        size = 4096, # number of entries (PER ONE INSTANCE!!) (remember to account for operand size)
        value_access_energy = 3.03, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_constraints = {},
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "SACols",
        dim = WS[1],
        mesh = 256,
        # pe_to_pe should be used, since the TPU uses a systolic array, but Timeloop
        # does not have this feature, so for sake of comparison, it is turned off
        #pe_to_pe = True, 
        factors_constraints = {'K': 256}
    ),
    MemLevel(
        name = "Register",
        dataflow_constraints = WS,
        size = 2, # number of entries
        value_access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_constraints = {'M': 1, 'K': 1}, # L is free
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        dim = WS[2],
        mesh = 1,
        compute_energy = 0.15, # per compute (pJ)
        cycles = 1,
        factors_constraints = {'N': 1}
    )], coupling=gemm_coupling, name="TPUv1")

arch_tpu_conv = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = [],
        size = 8*2**30, # number of entries
        value_access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {},
        bypasses = ['w']
    ),
    MemLevel(
        name = "WeightsDRAM",
        dataflow_constraints = [],
        size = 8*2**30, # number of entries
        value_access_energy = 560.00, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "UnifiedBuffer",
        dataflow_constraints = [],
        size = 24*(2**20), # number of entries
        value_access_energy = 19.66, # per operand (pJ)
        bandwidth = 32, # operands per cycle (shared)
        factors_constraints = {},
        # The Unified Buffer also stores outputs after the activation is
        # performed. Not modeled as we are only interested in the conv.
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WeightsFIFO",
        dataflow_constraints = [],
        size = 4*2**16, # number of entries
        value_access_energy = 2.11, # per operand/scalar access (pJ)
        bandwidth = 8, # operands per cycle (shared)
        factors_constraints = {},
        bypasses = ['in', 'out']
    ),
    FanoutLevel(
        name = "SARows",
        dim = 'M',
        mesh = 256,
        # pe_to_pe should be used, since the TPU uses a systolic array, but Timeloop
        # does not have this feature, so for sake of comparison, it is turned off
        #pe_to_pe = True, 
        factors_constraints = {'M': 256}
    ),
    MemLevel(
        name = "Accumulator",
        dataflow_constraints = ['C', 'R', 'S', '_', '_', '_'],
        size = 4096, # number of entries (PER ONE INSTANCE!!) (remember to account for operand size)
        value_access_energy = 3.03, # per operand (pJ)
        bandwidth = 8, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_constraints = {'C': 1, 'R': 1, 'S': 1}, # output stationary
        bypasses = ['in', 'w']
    ),
    FanoutLevel(
        name = "SACols",
        dim = 'C',
        mesh = 256,
        # pe_to_pe should be used, since the TPU uses a systolic array, but Timeloop
        # does not have this feature, so for sake of comparison, it is turned off
        #pe_to_pe = True, 
        factors_constraints = {'C': 256}
    ),
    MemLevel(
        name = "Register",
        dataflow_constraints = ['M', 'Q', 'C', 'R', 'S', 'P'],
        size = 2, # number of entries
        value_access_energy = 0.01, # per operand (pJ)
        bandwidth = 2, # operands per cycle (shared)
        multiple_buffering = 2,
        factors_constraints = {'M': 1, 'Q': 1, 'C': 1, 'R': 1, 'S': 1},
        bypasses = ['in', 'out']
    ),
    ComputeLevel(
        name = "Compute",
        mesh = 1,
        compute_energy = 0.15, # per compute (pJ)
        cycles = 1,
    )], coupling=conv_coupling_with_stride, name="TPUv1")


# >>>  NVDLA  <<<
# >  small ver. <

#arch_nvdla = [
#]

"""
ChatGPT gathering information on NVDLA:

--------------------------------------------------------------------------------

The exact configuration of the NVDLA (Nvidia Deep Learning Accelerator) used in the experiments is as
follows:

- **PE Array**: 64x8
- **NoC (Network-on-Chip)**: Bus + Tree
- **Dataflow**: Weight-stationary

This configuration is part of the specifications listed for the target spatial accelerator architectures
in the document. The weight-stationary dataflow implies that the weight matrix
(input matrix B in GEMM operation) remains stationary, and the input matrix A is streamed through the
PEs (Processing Elements)[^1^][1]. The NoC design facilitates the communication between the PEs and the
global shared scratchpad memory[^2^][2].

--------------------------------------------------------------------------------

The NVIDIA® Deep Learning Accelerator (NVDLA) is a hardware accelerator designed for deep learning
inference operations[^1^][1]. It is highly configurable and modular, allowing it to be tailored for
various applications. Here's a summary of the memory hierarchy for a "small" NVDLA accelerator based
on the requirements you provided:

- **PE Array Configuration**: The PE (Processing Element) array is configured as 64x8, which indicates
  the parallelism in the input feature channel dimension (Atomic-C) and output feature channel dimension
  (Atomic-K)[^2^][2]. This impacts the total number of MAC (Multiply-Accumulate) operations, convolution
  buffer read bandwidth, and accumulator instance number[^3^][3][^4^][4].

- **Memory Hierarchy Levels**:
  - **Convolution Buffer (CBUF)**: This is the primary memory level where both weight data and feature
    data for the convolution function are stored[^5^][5]. It is configurable to accommodate different
    ratios between feature and weight data[^6^][6]. The size of the CBUF depends on the CNN size, and it
    is preferable if the full size of either weight data or feature data of one hardware layer can be
    stored in the CBUF[^7^][7].
  - **Dedicated SRAM**: The NVDLA can connect to on-chip SRAM or other high-bandwidth low-latency
    buses through the SRAMIF interface[^8^][8]. This is used for lower latency and higher throughput.

- **Operand Size**:
  - The operand size at each level can vary based on the precision required by the application.
    NVDLA supports multiple data types, including INT8, INT16, and FP16[^9^][9]. For a "small" implementation,
    INT8 (8 bits) is likely sufficient.

- **Operand Storage**:
  - **Inputs**: Stored in the Convolution Buffer and fetched from the system memory or dedicated SRAM.
  - **Weights**: Also stored in the Convolution Buffer, with support for sparse weight compression to
    save memory bandwidth[^10^][10].
  - **Outputs**: Stored temporarily in the accumulator buffers and then written back to the system memory.

- **Operand Flow**:
  - Operands flow from one layer to the other through the command-execute-interrupt cycle.
    The management processor sends down the configuration of one hardware layer, along with an "activate"
    command. Once a hardware engine finishes its task, it issues an interrupt to report completion, and the
    process begins again for the next layer[^11^][11].

- **MAC Compute Units Location**:
  - The MAC compute units are part of the Convolution core pipeline stages[^12^][12]. They are involved
    in the direct convolution mode and are responsible for performing the multiply-accumulate operations.

Please note that the specific technology and exact sizes (depth, width) of the memory hierarchy levels are
not detailed in the document and would typically be defined in the NVDLA hardware design specification based
on the application's performance, area, and power requirements. The flow of operands and the location of MAC
compute units are conceptual and based on the logical organization of the NVDLA.
"""