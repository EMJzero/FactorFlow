import multiprocessing
import copy
import time
import sys

from architectures import *
from solutions_db import *
from computations import *
from settings import *
from factors import *
from engine import *
from levels import *
from prints import *
from utils import *

#from comparisons.ZigZag.zigzag_archs import *
#from comparisons.CoSA.cosa_archs import *
#from comparisons.MAESTRO.maestro_archs import *


# SPECIFICATION:

comp = comp_BERT_large['KQV']
bias_read = False # True if bias is not 0 - outputs are read even the first time

arch = arch_eyeriss

## MAIN:

if __name__ == "__main__":
    Settings.forcedSettingsUpdate(arch)
    
    start_time = time.time()

    fitConstraintsToComp(arch, comp)

    if Settings.MULTITHREADED:
        manager = multiprocessing.Manager()
        return_list = manager.list([None]*Settings.THREADS_COUNT)
        processes = []
        for i in range(Settings.THREADS_COUNT):
            p = multiprocessing.Process(target=optimizeDataflows, args=(copy.deepcopy(arch), copy.deepcopy(comp), bias_read, i, Settings.THREADS_COUNT, return_list, Settings.VERBOSE))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        assert None not in return_list, f"Some threads failed to return or found no valid mapping, see above logs..."
        arch, wart, _ = max(return_list, key=lambda res : res[1])
    else:
        #arch, _ = factorFlow(arch, comp, bias_read, verbose = VERBOSE)
        arch, wart, _ = optimizeDataflows(arch, comp, bias_read, verbose = Settings.VERBOSE)
    
    print(f"\nFinished in: {time.time() - start_time:.3f}s")

    print(f"\nBest mapping found with Wart: {wart:.3e}, EDP: {EDP(arch, bias_read, True):.3e} (J*cycle)")
    printFactors(arch)

    print("\nFinal MOPs per memory level:")
    printMOPsNew(arch)
    print("\nFinal Latency per level:")
    printLatencyNew(arch)
    
    if Settings.PADDED_MAPPINGS:
        print("")
        printPadding(arch, comp)

    if len(sys.argv) > 1 and sys.argv[1] == "--gen-tests":
        from test import generateTestMOPs, generateTestLatency
        print("\nGenerated tests:")
        generateTestMOPs(arch)
        generateTestLatency(arch)