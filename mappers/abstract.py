import threading

from settings import *
from model import *
from utils import *
from arch import *


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