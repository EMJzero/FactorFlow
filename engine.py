from copy import deepcopy
from typing import TYPE_CHECKING
import time
import sys

import importlib
import threading

from settings import *
from factors import *
from levels import *
from prints import *
from model import *
from utils import *
from arch import *

mapper = importlib.import_module("mappers." + Settings.MAPPER)
globals().update(vars(mapper))

if TYPE_CHECKING:
    from mappers.abstract import *


"""
Update Settings:
- initialize some depeding on runtime information;
- set some to best target the provided architecture.
"""
def forcedSettingsUpdate(arch : Arch, verbose : bool = True) -> None:
    mapperForcedSettingsUpdate(arch, verbose)
    if Settings.MULTITHREADED:
        if sys.version_info[1] < 13 or sys._is_gil_enabled() and verbose: print(f"WARNING: running on a Python version without free-threading (GIL enabled), this program relies heavily on true multithreading, thus expect a severe performance hit with your current setup. Consider updating to Python 3.13t or newer.")
        Settings.THREADS_COUNT = Settings.THREADS_COUNT if Settings.THREADS_COUNT else os.cpu_count()
        if verbose: print(f"INFO: running multithreaded with THREADS_COUNT = {Settings.THREADS_COUNT}")
    if not Settings.VERBOSE:
        if verbose: print(f"INFO: VERBOSE output disabled, wait patiently...")
    if verbose: print("")

"""
Mapper entry point.
"""
def run_engine(arch : Arch, comp : Shape, coupling : Coupling, bias_read : bool, verbose : bool = False) -> tuple[float, int, float, int, float, float, Arch]:
    forcedSettingsUpdate(arch, verbose = Settings.VERBOSE)
    start_time = time.time()
    
    if Settings.MULTITHREADED:
        past_perms = {(): ThreadSafeHeap()}
        lock = threading.Lock()
        barrier = threading.Barrier(Settings.THREADS_COUNT)
        threads = []
        for i in range(Settings.THREADS_COUNT):
            t = threading.Thread(target=optimizeDataflows, args=(deepcopy(arch), deepcopy(comp), bias_read, i, Settings.THREADS_COUNT, past_perms, lock, barrier, Settings.VERBOSE))
            threads.append(t)
            t.start()
            t.join
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(0.001)
        assert () in past_perms and len(past_perms[()]) > 0, f"All threads failed to return or found no valid mapping, see above logs..."
        _, mapping = past_perms[()].peek()
        arch.initFactors(comp)
        arch.importMapping(mapping)
        wart = Wart(arch, comp, bias_read)
    else:
        arch, wart = optimizeDataflows(arch, comp, bias_read, verbose = Settings.VERBOSE)
    
    end_time = time.time() - start_time
    
    edp = EDP(arch, bias_read, True)
    mops = MOPs(arch)
    energy = Energy(arch, True)
    latency = Latency(arch)
    utilization = arch.spatialUtilization()
    
    if verbose:
        print(f"\nFinished in: {end_time:.3f}s")
        
        print(f"\nBest mapping found with:\n\tWart: {wart:.3e}\n\tEDP: {edp:.3e} (J*cycle)\n\tEnergy: {energy:.3e} (uJ)\n\tLatency: {latency:.3e} (cc)")
        printFactors(arch)
        
        print("\nFinal MOPs per memory level:")
        printMOPs(arch)
        print("\nFinal Latency per level:")
        printLatency(arch)
        
        if Settings.PADDED_MAPPINGS:
            print("")
            printPadding(arch, comp)
    
    return edp, mops, energy, latency, utilization, end_time, arch

"""
Forcefully stop a running engine's threads.
"""
def stop_engine() -> None:
    Settings.forced_termination_flag = True
    if Settings.MULTITHREADED:
        threads = threading.enumerate()
        threads.remove(threading.current_thread())
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(0.001)