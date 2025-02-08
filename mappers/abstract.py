import threading

from settings import *
from model import *
from utils import *
from arch import *


"""
Update Settings to best target the provided architecture with the present mapper.
"""
def mapperForcedSettingsUpdate(arch : Arch, verbose : bool = True) -> None:
    pass

"""
Mapper Step 2: allocate to fanout levels the maximum number of iterations
               which can fit on their instances.

NOTE: this is a placeholder.
"""
def fanoutMaximization(arch : Arch, comp : Shape, bias_read : bool, verbose : bool = False) -> None:
    pass

"""
Mapper Step 3: greedy descent factors allocation, navigating the map-space
               via adjacent mappings, until a locally optimal one is found.

NOTE: this is a placeholder.
"""
def factorFlow(arch : Arch, comp : Shape, bias_read : bool, verbose : bool = False) -> tuple[Arch, float]:
    already_initialized = arch.initialized
    if not already_initialized:
        arch.initFactors(comp)
        arch.enforceFactorsConstraints(Settings.PADDED_MAPPINGS, verbose)
    assert arch.checkFactorsConstraints() and arch.checkDataflowConstraints(), ("Ill-posed constraints:" if not already_initialized else "Improperly initialized arch:") + f"\n{arch.logConstraintsViolations()}"
    if not already_initialized:
        fanoutMaximization(arch, comp, bias_read, verbose)
    pass

"""
Mapper Step 1: exhaustively iterate loop permutations/dataflows.

NOTE: this is a placeholder.
"""
def optimizeDataflows(arch : Arch, comp : Shape, bias_read : bool, thread_idx : int = -1, threads_count : int = 1, past_perms : dict[tuple[int, ...], ThreadSafeHeap[float, list[LevelCore], int, int]] = None, lock : threading.Lock = None, barrier : threading.Barrier = None, verbose : bool = False) -> Optional[tuple[Arch, float]]:
    if verbose: print("WARNING: placeholder implementation, no exploration is being carried out, returning the default mapping.")
    if thread_idx == 0:
        arch.initFactors(comp)
        # multithreading: the caller expects the best mapping found to be the top of the heap in past_perms[()]
        past_perms[()].push(Wart(arch, comp, bias_read), arch.exportMapping(copy = True))
        return
    elif thread_idx == -1:
        arch.initFactors(comp)
        # no multithreading: the caller expects the best mapping found to be returned
        return arch, Wart(arch, comp, bias_read)