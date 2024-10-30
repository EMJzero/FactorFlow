from architectures.architectures import *
from engine import *
from utils import *

#from architectures.solutions_db import *
#from comparisons.ZigZag.zigzag_archs import *
#from comparisons.CoSA.cosa_archs import *


# SPECIFICATION:

comp = comp_BERT_large['KQV'] #comp_maestro_blas['MB6']
bias_read = False # True if bias is not 0 - outputs are read even the first time

arch = arch_eyeriss

## MAIN:

if __name__ == "__main__":
    #Here changing settings does not propagate to processes, which reimport and reset settings.py
    #Settings.forcedSettingsUpdate(arch)

    fitConstraintsToComp(arch, comp)
    edp, mops, energy, latency, utilization, end_time, arch = run_engine(arch, comp, bias_read, verbose = True)