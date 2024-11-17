from architectures.architectures import *
from computations import *
from engine import *
from utils import *

#from architectures.solutions_db import *
#from architectures.architectures_hw_data import *
#from comparisons.ZigZag.zigzag_archs import *
#from comparisons.CoSA.cosa_archs import *


# SPECIFICATION:

comp = comp_BERT_large['KQV'] #comp_maestro_blas['MB6']
coupling = gemm_coupling
bias_read = False # True if bias is not 0 - outputs are read even the first time

arch = arch_eyeriss

## MAIN:

if __name__ == "__main__":
    #Here changing settings does not propagate to processes, which reimport and reset settings.py
    #Settings.forcedSettingsUpdate(arch)

    arch.checkCouplingCompatibility(coupling, comp, verbose = True)
    arch.fitConstraintsToComp(comp, enforce = True)
    edp, mops, energy, latency, utilization, end_time, arch = run_engine(arch, comp, coupling, bias_read, verbose = True)