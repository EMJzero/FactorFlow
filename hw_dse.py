from itertools import combinations, product
from prettytable import PrettyTable
from typing import Callable, Self, Any

from architectures.architectures_hw_data import *
from computations import *
from engine import *
from utils import *

"""
An entry for the Pareto set class.
"""
class ParetoEntry:
    def __init__(self, value : tuple[float], data : Any):
        self.value = value
        self.data = data

"""
The Pareto class manages a Pareto front of non-dominated points in a multi-dimensional space.
Each dimension can be configured to be maximized or minimized, and the class maintains only
non-dominated elements based on these rules.

Constructor arguments:
- dimensions: (int) The number of dimensions for each point.
- maximize_minimize: (list of int) A list of +1 or -1 values for each dimension, indicating
  whether to maximize (+1) or minimize (-1) that dimension.
"""
class Pareto(set[ParetoEntry]):
    def __init__(self, dimensions : int, maximize_minimize : list[int]):
        # Enforce dimensions and maximize_minimize constraints
        assert isinstance(dimensions, int) and dimensions > 0, f"Dimensions ({dimensions}) must be a positive integer."
        assert len(maximize_minimize) == dimensions, f"The length of maximize_minimize ({maximize_minimize}) must match the number of dimensions."
        assert all(m in (+1, -1) for m in maximize_minimize), f"Each element in maximize_minimize ({maximize_minimize}) must be +1 or -1."
        
        self.dimensions = dimensions
        self.maximize_minimize = maximize_minimize
        super().__init__()
    
    """
    Check if tuple `a` dominates tuple `b` based on the maximize/minimize vector.
    """
    def dominates(self, entry_a : ParetoEntry, entry_b : ParetoEntry) -> bool:
        a, b = entry_a.value, entry_b.value
        assert len(a) == self.dimensions and len(b) == self.dimensions, "Elements must match the number of dimensions."
        
        dominance = False
        for i in range(self.dimensions):
            if self.maximize_minimize[i] == 1: # maximize
                if a[i] < b[i]:
                    return False
                elif a[i] > b[i]:
                    dominance = True
            else:  # minimize
                if a[i] > b[i]:
                    return False
                elif a[i] < b[i]:
                    dominance = True
        return dominance

    """
    Insert an entry into the Pareto set, removing anything that gets dominated.
    Returns True if the element was added to the pareto set, False otherwise.
    """
    def insert(self, value : tuple[float], data : Any) -> bool:
        assert isinstance(value, tuple) and len(value) == self.dimensions, "Element must be a tuple with exactly `dimensions` elements."
        new_entry = ParetoEntry(value, data)
        
        to_remove = set()
        for existing in self:
            if self.dominates(new_entry, existing):
                to_remove.add(existing)
            elif self.dominates(existing, new_entry):
                return False
        self.difference_update(to_remove)
        self.add(new_entry)
        return True

"""
Design space for an architecture built through Accelergy.

Constructor arguments:
- arch_callback: function that returns a built architecture w.r.t. the provided parameters.
- dictionary of lists: lists of values to try for each architectural parameter, parameter names must match argument names in the callback.
"""
class DesignSpace(dict[str, list[Any]]):
    def __init__(self, arch_callback : Callable[..., Arch], *args, **kwargs):
        self.arch_callback = arch_callback
        super().__init__(*args, **kwargs)
        assert self.arch_callback.__code__.co_argcount == len(self) and all(arg in self for arg in self.arch_callback.__code__.co_varnames[:self.arch_callback.__code__.co_argcount]), f"The provided dictionary's keys ({list(self.keys())}) do not match the arguments for the provided callback '{self.arch_callback.__name__}' ({self.arch_callback.__code__.co_varnames[:self.arch_callback.__code__.co_argcount]})."

    """
    Partitions the design space into 'n' almost equi-sized subspaces, each returned as an instance of DesignSpace.
    """
    def getSubSpaces(self, n : int = 1) -> list[Self]:
        n = min(n, prod(len(v) for v in self.values()))

        partitions = {k : 1 for k in self.keys()}
        primes = prime_factors_list(n)
        assigned = True
        while len(primes) > 0 and assigned:
            assigned = False
            for k in partitions.keys():
                if partitions[k]*primes[0] < len(self[k]):
                    partitions[k] *= primes.pop(0)
                    assigned = True
                    break
        
        partition_idx = {k : 0 for k in self.keys()}
        result = []
        while True:
            result.append(DesignSpace({k : v[(delta := math.ceil(len(v)/partitions[k]))*partition_idx[k]:(delta*(partition_idx[k] + 1) if partition_idx[k] + 1 < partitions[k] else len(v))] for k, v in self.items()}))
            done = False
            for k in self.keys():
                if partition_idx[k] + 1 < partitions[k]:
                    partition_idx[k] += 1
                    done = True
                    break
                else:
                    partition_idx[k] = 0
            if not done:
                break
        
        return result

    """
    Generates one after the other the architectures corresponding to points in the design space.
    """
    def __iter__(self) -> Iterable[tuple[Arch, dict[str, Any]]]:
        idx = {k : 0 for k in self.keys()}
        while True:
            args = {k : v[idx[k]] for k, v in self.items()}
            yield self.arch_callback(**args), args
            done = False
            for k in self.keys():
                if idx[k] + 1 < len(self[k]):
                    idx[k] += 1
                    done = True
                    break
                else:
                    idx[k] = 0
            if not done:
                break

def evalDesign(design : Arch, benchmarks : dict[str, Shape], couplings : dict[str, Coupling], config : Optional[dict[str, Any]]) -> tuple[float, float, float, float, float, float]:
    tot_edp, tot_energy, tot_latency, tot_utilization, tot_time = 0, 0, 0, 0, 0
    area = design.totalArea()
    print("Exploring Eyeriss: ", config)
    for benchmark_name, benchmark in benchmarks.items():
        print(f"Now running FactorFlow on comp: {benchmark_name}...")
        design.checkCouplingCompatibility(couplings[benchmark_name], benchmark, verbose = False)
        design.fitConstraintsToComp(benchmark, enforce = True)
        edp, _, energy, latency, utilization, time, _ = run_engine(design, benchmark, couplings[benchmark_name], False, False)
        tot_edp += edp
        tot_energy += energy
        tot_latency += latency
        tot_utilization += utilization
        tot_time += time
    
    return tot_edp, tot_energy, tot_latency, tot_utilization, tot_time, area


def runDSE(benchmarks : dict[str, Shape], couplings : dict[str, Coupling]) -> None:
    eyeriss_ds = DesignSpace(get_arch_eyeriss_hw_data, {
        'global_buffer_size': [16384*8, 16384*8*2, 16384*6*2, 16384*6],
        'global_buffer_banks': [2],
        'sa_cols': [14, 16, 12, 24],
        'sa_rows': [12, 16, 8, 24],
        'in_reg_size': [12*2],
        'w_reg_size': [192*2],
        'out_reg_size': [16*2]
    })

    table = PrettyTable([k for k in eyeriss_ds.keys()] + ["Tot. EDP[J*cycle]", "Tot. Latency[cc]", "Tot. Energy[uJ]", "Avg. Util.[/]", "Area[um^2]", "Expl. Time[s]"])
    table.sortby = "Tot. EDP[J*cycle]"
    table.sort_key = lambda l1, l2 : float(l1[0]) > float(l2[0])

    design : Arch = None
    for design, config in eyeriss_ds:
        tot_edp, tot_energy, tot_latency, tot_utilization, tot_time, area = evalDesign(design, benchmarks, couplings, config)
        table.add_row([config[k] for k in eyeriss_ds.keys()] + [f"{tot_edp:.3e}", f"{tot_latency:.3e}", f"{tot_energy:.3e}", f"{tot_utilization/len(benchmarks):.3e}", f"{area:.3e}" if area else 'N/A', f"{tot_time:.3f}"])

    print(table)

def runAnnealingDSE(benchmarks : dict[str, Shape], couplings : dict[str, Coupling]) -> None:
    eyeriss_args = {
        'global_buffer_size': 16384*8,
        'global_buffer_banks': 2,
        'sa_cols': 14,
        'sa_rows': 12,
        'in_reg_size': 12*2,
        'w_reg_size': 192*2,
        'out_reg_size': 16*2
    }

    table = PrettyTable([k for k in eyeriss_args.keys()] + ["Tot. EDP[J*cycle]", "Tot. Latency[cc]", "Tot. Energy[uJ]", "Avg. Util.[/]", "Area[um^2]", "Expl. Time[s]"])
    table.sortby = "Tot. EDP[J*cycle]"
    table.sort_key = lambda l1 : float(l1[0])

    # IDEA: binary tree to explore one parameter, each node has two childs, one with its parameter * step, one with its parameter * 1/step.
    #       Only keep alive nodes that are not dominated in area and EDP by any other node. Complete the tree layer by layer, each new layer
    #       has its step reduced by a certain constants, and with that step all nodes at that layer are built. This gives the pareto!

    designs : list[tuple[Arch, dict[str, Any]]] = [(get_arch_eyeriss_hw_data(**eyeriss_args), eyeriss_args)]
    pareto_designs : list[tuple[Arch, dict[str, Any], float, float]] = []
    future_designs : list[tuple[Arch, dict[str, Any]]] = []
    step = 2.0
    while True:
        for design, config in designs:
            tot_edp, tot_energy, tot_latency, tot_utilization, tot_time, area = evalDesign(design, benchmarks, couplings, config)
            table.add_row([config[k] for k in config.keys()] + [f"{tot_edp:.3e}", f"{tot_latency:.3e}", f"{tot_energy:.3e}", f"{tot_utilization/len(benchmarks):.3e}", f"{area:.3e}" if area else 'N/A', f"{tot_time:.3f}"])
            if not any(pd_edp < tot_edp and pd_area < area for _, _,pd_edp, pd_area in pareto_designs):
                down_config = config.copy()
                down_config['global_buffer_size'] *= 1/step
                future_designs.append((get_arch_eyeriss_hw_data(**down_config), down_config))
                up_config = config.copy()
                up_config['global_buffer_size'] *= step
                future_designs.append((get_arch_eyeriss_hw_data(**up_config), up_config))
                pareto_designs.append((design, config, tot_edp, area))
        step -= 0.1
        if step <= 1.0 or len(future_designs) == 0:
            break
        del designs
        designs = future_designs
        future_designs = []
    print(table)

def runMultiNudgeDSE(benchmarks : dict[str, Shape], couplings : dict[str, Coupling]) -> None:
    parameters = {
        'global_buffer_size': 16384*8,
        'global_buffer_banks': 2,
        'sa_cols': 8,
        'sa_rows': 8,
        'in_reg_size': 12*2,
        'w_reg_size': 192*2,
        'out_reg_size': 16*2
    }
    step_sizes = {
        'global_buffer_size': 4.0,
        'global_buffer_banks': 1.0, # do not explore
        'sa_cols': 4.0,
        'sa_rows': 4.0,
        'in_reg_size': 2.0,
        'w_reg_size': 2.0,
        'out_reg_size': 2.0
    }
    step_decay = 0.2
    simultaneous_nudges = 2

    table = PrettyTable([k for k in parameters.keys()] + ["Tot. EDP[J*cycle]", "Tot. Latency[cc]", "Tot. Energy[uJ]", "Avg. Util.[/]", "Area[um^2]", "Expl. Time[s]"])
    table.sortby = "Tot. EDP[J*cycle]"
    table.sort_key = lambda l1 : float(l1[0])

    # IDEA for the DSE multi-nudge algorithm:
    # - keep one step size per parameter, that decreses by a fixed constant at every round
    # - keep the pareto set of designs
    # - start from a decent design
    # - try all of designs where only at most N parameters have been nudge up or down
    # - add to the pareto set the best designs found
    # - repeat starting once from each mapping in the pareto set, decreasing N each time the pareto set surpasses a certain size
    # - stop exploring a parameter when its step size becomes <=1, stop when all parameters are unexplorable

    designs : list[tuple[Arch, dict[str, Any]]] = [(get_arch_eyeriss_hw_data(**parameters), parameters)]
    future_designs : list[tuple[Arch, dict[str, Any]]] = []
    pareto_designs = Pareto(2, [-1, -1])
    while True:
        for design, config in designs:
            tot_edp, tot_energy, tot_latency, tot_utilization, tot_time, area = evalDesign(design, benchmarks, couplings, config)
            table.add_row([config[k] for k in config.keys()] + [f"{tot_edp:.3e}", f"{tot_latency:.3e}", f"{tot_energy:.3e}", f"{tot_utilization/len(benchmarks):.3e}", f"{area:.3e}" if area else 'N/A', f"{tot_time:.3f}"])
            if pareto_designs.insert((tot_edp, area), config):
                for param_keys in combinations([k for k in parameters.keys() if step_sizes[k] > 1.0], r = simultaneous_nudges):
                    for direction in product([1, -1], repeat = simultaneous_nudges):
                        new_config = config.copy()
                        for i, param_key in enumerate(param_keys):
                            new_config[param_key] *= step_sizes[param_key]**direction[i]
                        future_designs.append((get_arch_eyeriss_hw_data(**new_config), new_config))
        for k in step_sizes.keys():
            step_sizes[k] -= step_decay
        if all(v <= 1.0 for v in step_sizes.values()) or len(future_designs) == 0:
            break
        del designs
        designs = future_designs
        future_designs = []
    print("Visited design space points:")
    print(table)
    
    pareto_table = PrettyTable(["Tot. EDP[J*cycle]", "Area[um^2]"] + [k for k in parameters.keys()])
    pareto_table.sort_key = lambda l1 : float(l1[0])
    for entry in pareto_designs:
        pareto_table.add_row([v for v in entry.value] + [entry.data[k] for k in parameters.keys()])
    print("\nPareto set:")
    print(pareto_table)

def runFromAshesToAssets(benchmarks : dict[str, Shape], couplings : dict[str, Coupling]) -> None:
    # IDEA:
    # If we always increment design parameters, and never lower them, we can continously reuse the previous mappings
    # with just one call to QuickFlow's co-optimization round to re-gain mapping optimality on the larger hardware.
    #
    # DSE axioms:
    # - each buffer size must be strictly a product of prime factors present in the kernels dimensions;
    # - each dimension to explore must have an upper and lower bound;
    # => the exploration space becomes a finite N-dimensional lattice!
    # - the objective is a Pareto set of area VS EDP;
    # - the direction (not the magnitude) of the gradient for area is known ahead of time and tends to decrease
    #   along with each dimension’s quantity;
    # - the gradient for EDP is an extremely convoluted function of each dimension;
    # => start with a design of experiments like Box-Banken to get an approximation for the gradients?
    # - if the problem can be solved efficiently even for just one dimension, then handling multiple dimensions
    #   is just a matter of scaling up and going in // across many cores;
    #
    # Considerations:
    # - beyond a certain point the EDP will not decrease anymore, question is, where is this “equi-level” line
    #   of EDP in the lattice?
    # - to dodge the above mentioned part of the space, what about starting from the minimum of all dimensions and
    #   increase the best option every time, then go beam search and keep up to the n-th best option?
    # => this way you don’t need to restart MSE every time from scratch, because so long as you increase dimensions,
    #    and not shrink them, you can just rerun the co-opt. MSE step starting from the previous best mapping!
    # => the greedy-ascent beam search from before needs to keep the n-best recent configurations seen, we can define
    #    the “best” as the most recently seen ones added to the Pareto set;
    #
    # TL;DR:
    # - we have a finite multidimensional lattice to search;
    # - try first by increasing dimensions only, since that allows you to reuse mappings;
    # - try to solve along 1 (at most 2) dimensions first, then we can scale either in // or by seeing
    #   if the algorithm generalizes well;
    #
    # -> easy to get the “low” part of the curve where area is low, harder to get the “high” part since you risk diverging while lowering the EDP.
    params_lower_bound = {
        'global_buffer_size': 16384//2,
        'global_buffer_banks': 2,
        'sa_cols': 32,
        'sa_rows': 32,
        'in_reg_size': 12*8,
        'w_reg_size': 192*8,
        'out_reg_size': 16*8
    }
    params_upper_bound = {
        'global_buffer_size': 16384*16,
        'global_buffer_banks': 2,
        'sa_cols': 1,
        'sa_rows': 1,
        'in_reg_size': 12,
        'w_reg_size': 192,
        'out_reg_size': 16
    }

    # distinct prime factors appearing in kernels
    kernels_primes = set(p for kernel in benchmarks.values() for dim_size in kernel.values() for p in prime_factors_list(dim_size))

if __name__ == "__main__":
    benchmarks = benchmark_convs | benchmark_convs_batched
    couplings = {k : conv_coupling_with_stride for k in benchmark_convs.keys()} | {k : conv_coupling_with_stride_and_batches for k in benchmark_convs_batched.keys()}
    #benchmarks = {'I': benchmark_convs['I'], 'II': benchmark_convs['II']}
    #couplings = {'I': conv_coupling_with_stride, 'II': conv_coupling_with_stride}

    #runDSE(benchmarks, couplings)
    #runAnnealingDSE(benchmarks, couplings)
    #runMultiNudgeDSE(benchmarks, couplings)
    runFromAshesToAssets(benchmarks, couplings)