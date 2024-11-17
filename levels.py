from __future__ import annotations
from typing import TYPE_CHECKING
from math import prod

from factors import *

# fix static typechecking without recursive imports
if TYPE_CHECKING:
    from arch import Arch

# Remember: on the present level you store everything for the iterations you have there, while
# stationarity means finding the operands that remaing constant (and don't need to be read again
# to either be used or sent to a level below) during consecutive (thus, innermost) iterations.

"""
Class with the minimal information characterizing a level's mapping.
"""
class LevelCore:
    # IMPORTANT:
    # Use the entries of "dataflow" to access "factors", so that you only
    # see the factors over which you are actually supposed to iterate!
    dataflow: None # order of the loops | e.g. ['M', 'K', 'N']
    # NOTE: tile sizes are updated in "moveFactor".
    factors: None # iterations done for the dimensions at this lever
    tile_sizes: None # indicate the size of a tile used in the level BELOW
    
    def __init__(self, dataflow, factors, tile_sizes):
        self.dataflow = dataflow
        self.factors = factors
        self.tile_sizes = tile_sizes
    
    def __str__(self):
        return f"dataflow: {self.dataflow}, factors: {self.factors}, tile_sizes: {self.tile_sizes}"


"""
Abstract class representing a level of the accelerator's architecture.

NOTE: the hierarchy is read just as the architecture is specified, with
"below" levels being closer to the computation and "above" ones closer
to DRAM or slower memories.
"""
class Level(LevelCore):
    name: None
    arch: None
    factors_constraints: None
    area: None

    """
    Add "amount" instances of the provided factor to those of
    "dimension" in the current level.
    """
    def addFactor(self, dimension, factor, amount = 1):
        self.factors.addFactor(dimension, factor, amount)

    """
    Removes "amount" instances of the provided factor from those of
    "dimension" in the current level.
    Return False if the removal failed because the current level does not
    have at least "amount" instances of "factor" along "dimension".
    """
    def removeFactor(self, dimension, factor, amount = 1):
        return self.factors.removeFactor(dimension, factor, amount)

    """
    Returns True iif factors present on this level satisfy all of
    its constraints.
    """
    def checkConstraints(self):
        return (all([(dim not in self.factors_constraints or self.factors_constraints[dim] == self.factors.dimProduct(dim)) for dim in self.dataflow]) and
                all([(dim + '<=' not in self.factors_constraints or self.factors_constraints[dim + '<='] >= self.factors.dimProduct(dim)) for dim in self.dataflow]) and
                all([(dim + '>=' not in self.factors_constraints or self.factors_constraints[dim + '>='] <= self.factors.dimProduct(dim)) for dim in self.dataflow]))

    """
    Returns a string describing the current violation of constraints,
    if any.
    """
    def logConstraintsViolation(self):
        if not self.checkConstraints():
            return (f"CONSTRAINTS VIOLATION: Arch: {self.arch.name} -> Level: {self.name}: "
                + ', '.join([f"constrained {dim} == {self.factors_constraints[dim]} VS obtained {dim}: {self.factors.dimProduct(dim)}, " for dim in self.dataflow if (dim in self.factors_constraints and self.factors_constraints[dim] != self.factors.dimProduct(dim))])
                + ', '.join([f"constrained {dim} <= {self.factors_constraints[dim + '<=']} VS obtained {dim}: {self.factors.dimProduct(dim)}, " for dim in self.dataflow if (dim + '<=' in self.factors_constraints and self.factors_constraints[dim + '<='] >= self.factors.dimProduct(dim))])
                + ', '.join([f"constrained {dim} >= {self.factors_constraints[dim + '>=']} VS obtained {dim}: {self.factors.dimProduct(dim)}, " for dim in self.dataflow if (dim + '>=' in self.factors_constraints and self.factors_constraints[dim + '>='] <= self.factors.dimProduct(dim))]))
        return ""

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __str__(self):
        return f"{self.name}: ({super().__str__()}, factors_constraints: {self.factors_constraints})"


"""
A Memory Level within the architecture, with the possibility to store
data and provide it as tiles to the levels below it.

Constructor arguments:
- name: the level's name
- size: the capacity (in number-of-operands, disregarding bits per operand)
- value_access_energy: energy required by each access of one value (in pJ)
                       [w.r.t. Timeloop this is the vector access energy / elements per
                       vector, also called energy-per-scalar-access]
  - Note: if specified, it has priority over wordline_access_energy. At least one of
          value_access_energy and wordline_access_energy must be specified.
- wordline_access_energy: energy required by each wordline access (in pJ)
  - Note: at least one of value_access_energy and wordline_access_energy must be specified.
          Specifying wordline_access_energy requires word_bits and value_bits to be both
          specified as well.
- word_bits: size in bits of the memory's wordlines.
- value_bits: size in bits of the values stored on the memory. This is the same for all
          operands, castings are implicitly assumed to take place whenever needed.
- leakage_energy: energy leaked each clock cycle by the component (in pJ/cc)
- area: the area occupied by the memory (in um^2).
- bandwidth: the bandwidth for reads and writes, it will be divided in 1/2 for
             read and 1/2 for write (in operands/clock-cycle)
- dataflow: specifies the dimensions over which to iterate, defaults to all dimensions
- factors: specifies the initial factors for this level, should not be normally
           specified aside for initializing MSE from a specific configuration
- tile_sizes: specifies the initial tile sizes for this level, should not be normally
              specified aside for initializing MSE from a specific configuration,
              in which case it must be consistent with any other factors initialization
- factors_constraints: constraints on the factors that must be placed on this level.
                      Valid dictionary keys use dimension names, e.g. for a GEMM:
                          - 'M', 'K', and 'N' for exact values;
                          - 'M<=', 'K<=', and 'N<=' for upper bounds;
                          - 'M>=', 'K>=', and 'N>=' for lower bounds;
                      NOTE: the use of the '<=' and '>=' constraints does not shorten the
                            mapper's runtime as much as exact constraints.
- dataflow_constraints: constraints for the order of loops at this level, for any dim
                        not specified here, all permutations are tried while keeping
                        fixed the relative order of constrained dimensions.
                        E.g., for GEMMs, valid strings are 'M', 'K', and 'N'.
- bypasses: list of operands which should bypass this level (i.o.w. not be stored here),
            valid strings are 'in', 'w', and 'out'.
- multiple_buffering: factor of multiple buffering employed by this level, must be >1
- read_value_access_energy: energy required for reads accesses, if specified overrides value_access_energy
- write_value_access_energy: energy required for write accesses, if specified overrides value_access_energy
  - Note: either both or none of read_value_access_energy and write_value_access_energy must be specified.
- read_wordline_access_energy: energy required for reads accesses, if specified overrides wordline_access_energy
- write_wordline_access_energy: energy required for write accesses, if specified overrides wordline_access_energy
  - Note: either both or none of read_wordline_access_energy and write_wordline_access_energy must be specified.
- read_bandwidth: bandwidth allocated for reads, if specified overrides "bandwidth"
- write_bandwidth: bandwidth allocated for writes, if specified overrides "bandwidth"
  - Note: either both or none of read_bandwidth and write_bandwidth must be specified
"""
class MemLevel(Level):
    def __init__(self, name, size, value_access_energy = None, wordline_access_energy = None, word_bits = None, value_bits = None, leakage_energy = 0, area = None, bandwidth = None, dataflow = None, factors = None, tile_sizes = None, factors_constraints = None, dataflow_constraints = None, bypasses = None, multiple_buffering = 1,  read_value_access_energy = None, write_value_access_energy = None, read_wordline_access_energy = None, write_wordline_access_energy = None, read_bandwidth = None, write_bandwidth = None):
        self.name = name
        self.dataflow = dataflow
        self.size = size
        self.word_bits = word_bits
        self.value_bits = value_bits
        self.wordline_access_energy = wordline_access_energy
        self.value_access_energy = value_access_energy
        self.read_wordline_access_energy = read_wordline_access_energy
        self.write_wordline_access_energy = write_wordline_access_energy
        self.read_value_access_energy = read_value_access_energy
        self.write_value_access_energy = write_value_access_energy
        self.leakage_energy = leakage_energy
        self.area = area
        self.bandwidth = bandwidth
        self.read_bandwidth = read_bandwidth
        self.write_bandwidth = write_bandwidth
        self.factors = factors
        self.tile_sizes = tile_sizes
        self.factors_constraints = factors_constraints if factors_constraints else {}
        self.dataflow_constraints = dataflow_constraints if dataflow_constraints else []
        self.bypasses = bypasses if bypasses else []
        self.in_bp = 0 if (bypasses and 'in' in bypasses) else 1
        self.w_bp = 0 if (bypasses and 'w' in bypasses) else 1
        self.out_bp = 0 if (bypasses and 'out' in bypasses) else 1
        self.multiple_buffering = multiple_buffering

        # STATISTICS:
        self.instances = 1 # this are the used/active instances
        self.next_is_compute = False
        self.temporal_iterations = 0
        self.in_reads = 0
        self.w_reads = 0
        self.out_reads = 0
        self.in_writes = 0
        self.w_writes = 0
        self.out_writes = 0
        self.last_out_reads = 0
        self.last_out_writes = 0
        self.latency_read_drain = 0
        self.latency_fill_update = 0
        self.cc_per_tile = 0
        self.stall_cycles = 0
        self.ideal_bandwidth_read = 0
        self.ideal_bandwidth_update = 0
        self.ideal_bandwidth_fill = 0
        self.ideal_bandwidth_drain = 0

        self.next_levels_with_bypass = {'in': None, 'w': None, 'out': None}

    """
    Sets up a pointer back to the whole architecture.
    Ultimates the initialization of the level and validates its attributes.
    """
    def initArch(self, arch : Arch):
        self.arch = arch
        # NOTE: this way of constructing the dataflow from the constraints is redundant, but useful if one wants to skip the
        # exploration of permutations since with this method the dataflow will be immediately consistent with constraints.
        self.dataflow = self.dataflow if self.dataflow else (self.dataflow_constraints + [dim for dim in arch.coupling.dims if dim not in self.dataflow_constraints] if self.dataflow_constraints else arch.coupling.dims) # dimensions over which to iterate
        assert all(dim in arch.coupling.dims for dim in self.dataflow), f"Arch: {arch.name} -> Level: {self.name}: accepted names for dimensions, as per the present coupling, are solely {arch.coupling.dims} provided ones were {self.dataflow}."
        assert self.size >= 0, f"Arch: {arch.name} -> Level: {self.name}: a negative size ({self.size}) does not mean anything."
        # read_access_energy and write_access_energy are intended always for one value, remember to bring accessed values to a multiple of values_per_wordline for the correct total energy
        assert (self.value_access_energy or (self.read_value_access_energy and self.write_value_access_energy)) or (self.word_bits and self.value_bits and (self.wordline_access_energy or (self.read_wordline_access_energy and self.write_wordline_access_energy))), f"Arch: {arch.name} -> Level: {self.name}: either value_access_energy ({self.value_access_energy}) or read_value_access_energy ({self.read_value_access_energy}) and write_value_access_energy ({self.write_value_access_energy}) must be specified, alternatively, you can specify word_bits ({self.word_bits}) and value_bits ({self.value_bits}) and either wordline_access_energy ({self.wordline_access_energy}) or read_wordline_access_energy ({self.read_wordline_access_energy}) and write_wordline_access_energy ({self.write_wordline_access_energy}). In any case, when if either of read_*_access_energy or write_*_access_energy is specified, the other must be present as well."
        if (self.value_access_energy or (self.read_value_access_energy and self.write_value_access_energy)):
            self.values_per_wordline = 1
            self.read_access_energy = self.read_value_access_energy if self.read_value_access_energy else self.value_access_energy
            self.write_access_energy = self.write_value_access_energy if self.write_value_access_energy else self.value_access_energy
        else:
            self.values_per_wordline = self.word_bits // self.value_bits
            self.read_access_energy = (self.read_wordline_access_energy if self.read_wordline_access_energy else self.wordline_access_energy) / self.values_per_wordline
            self.write_access_energy = (self.write_wordline_access_energy if self.write_wordline_access_energy else self.wordline_access_energy) / self.values_per_wordline
        del self.word_bits, self.value_bits, self.wordline_access_energy, self.value_access_energy, self.read_wordline_access_energy, self.write_wordline_access_energy, self.read_value_access_energy, self.write_value_access_energy
        assert self.read_access_energy >= 0 and self.write_access_energy >= 0 and self.leakage_energy >= 0, f"Arch: {arch.name} -> Level: {self.name}: a negative access energy ({self.read_access_energy} read, {self.read_access_energy} write), ({self.leakage_energy} leak), does not mean anything (unless you are into sci-fi stuff)."
        assert not self.area or self.area >= 0, f"Arch: {arch.name} -> Level: {self.name}: a negative area ({self.area}) does not mean anything."
        # NOTE: 1/2 split of bandwidth for consistency with Timeloop - not a true must...
        assert (self.bandwidth and not self.read_bandwidth and not self.write_bandwidth) or (self.read_bandwidth and self.write_bandwidth), f"Arch: {arch.name} -> Level: {self.name}: either bandwidth ({self.bandwidth}) or read_bandwidth ({self.read_bandwidth}) and write_bandwidth ({self.write_bandwidth}) must be specified, if either of read_bandwidth or write_bandwidth is specified, the other must be specified as well."
        self.read_bandwidth = self.read_bandwidth if self.read_bandwidth else self.bandwidth/2
        self.write_bandwidth = self.write_bandwidth if self.write_bandwidth else self.bandwidth/2
        del self.bandwidth
        self.factors = self.factors if self.factors else Factors(arch.coupling.dims)
        self.tile_sizes = self.tile_sizes if self.tile_sizes else Shape({dim: 1 for dim in arch.coupling.dims})
        assert self.read_bandwidth >= 0 and self.write_bandwidth >= 0, f"Arch: {arch.name} -> Level: {self.name}: a negative bandwidth ({self.read_bandwidth} R, {self.write_bandwidth} W) does not mean anything."
        assert all([constr[0] in self.dataflow and constr[1:] in ['', '>=', '<='] for constr in self.factors_constraints.keys()]), f"Arch: {arch.name} -> Level: {self.name}: all keys within factor constraints ({list(self.factors_constraints.keys())}) must be a dimension of the dataflow ({self.dataflow}) and in the form 'dim', 'dim<=', or 'dim>='."
        assert all([sum(constr[0] == dim for constr in self.factors_constraints.keys()) <= 1 for dim in self.dataflow]), f"Arch: {arch.name} -> Level: {self.name}: each dimension must occur at most once in constraints ({list(self.factors_constraints.keys())}), regardless of the use of '>=' or '<='."
        assert all([value > 0 for value in self.factors_constraints.values()]), f"Arch: {arch.name} -> Level: {self.name}: all constraints ({self.factors_constraints}) must have a value strictly > 0."
        assert all([constr in self.dataflow for constr in self.dataflow_constraints]), f"Arch: {arch.name} -> Level: {self.name}: all dims specified as dataflow constraints ({self.dataflow_constraints}) must be part of the dataflow ({self.dataflow})."
        assert self.multiple_buffering >= 1, f"Arch: {arch.name} -> Level: {self.name}: multiple buffering ({self.multiple_buffering}) must be at least 1."

    """
    Initializes the bypasses which start from this level.
    Let "levels" be all levels starting from the next one going downward
    up until and including the last one bypassing "operand".

    => This method must be invoked while iterating from outer to inner levels
    as it updates the level's notion of bypassed operations.
    """
    def initBypass(self, operand, levels):
        self.next_levels_with_bypass[operand] = levels
        if operand == 'in':
            self.in_bp = 0
        elif operand == 'w':
            self.w_bp = 0
        else:
            self.out_bp = 0

    """
    Sets MOPs statistics for this level.
    Those must include both operations with the above and below level.
    """
    def setMOPs(self, in_reads, w_reads, out_reads, in_writes, w_writes, out_writes):
        self.in_reads = in_reads
        self.w_reads = w_reads
        self.out_reads = out_reads
        self.in_writes = in_writes
        self.w_writes = w_writes
        self.out_writes = out_writes

    # fill, drain, read, and update are intended in the "Buffet" sense, and to be
    # computed they require "last_out_writes"  and "last_out_reads" from the level
    # ABOVE to distinguish the components of "out_reads" and "out_writes"!

    """
    Sets MOPs statistics for the interations between this level and
    the MemLevel immediately above it.
    """
    def setAboveMOPs(self, last_out_reads, last_out_writes):
        self.last_out_reads = last_out_reads
        self.last_out_writes = last_out_writes

    """
    Returns "Fills" as intended for Buffets, thus the incoming writes
    from an higher level.
    """
    def getFill(self):
        return self.in_writes + self.w_writes + self.last_out_reads #include fills for outputs

    """
    Returns "Drains" as intended for Buffets, thus the outgoing reads
    towards an higher level.
    """
    def getDrain(self):
        return self.last_out_writes

    """
    Returns "Reads" as intended for Buffets, thus the outgoing reads
    towards a lower level.
    """
    def getRead(self):
        return self.in_reads + self.w_reads + (self.out_reads - self.last_out_writes) #ignore reads done to drain

    """
    Returns "Updates" as intended for Buffets, thus the incoming writes
    from a lower level.
    """
    def getUpdate(self):
        return self.out_writes - self.last_out_reads #ignore updates coming from fills

    """
    Returns the total MOPs previoulsy stored by setMOPs().
    """
    def getSettedMOPs(self):
        return self.in_reads + self.w_reads + self.out_reads + self.in_writes + self.w_writes + self.out_writes
 
    """
    Sets Latency and related statistics for this level.
    """
    def setLatency(self, latency_read_drain, latency_fill_update, cc_per_tile, stall_cycles, ideal_bandwidth_read, ideal_bandwidth_update, ideal_bandwidth_fill, ideal_bandwidth_drain):
        self.latency_read_drain = latency_read_drain
        self.latency_fill_update = latency_fill_update
        self.cc_per_tile = cc_per_tile
        self.stall_cycles = stall_cycles
        self.ideal_bandwidth_read = ideal_bandwidth_read
        self.ideal_bandwidth_update = ideal_bandwidth_update
        self.ideal_bandwidth_fill = ideal_bandwidth_fill
        self.ideal_bandwidth_drain = ideal_bandwidth_drain

    """
    Returns the Latency previoulsy stored by setLatency().
    """
    def getSettedLatency(self):
        return max(self.latency_read_drain, self.latency_fill_update)

    # Memory operation between this level and the one below it! Specifically: returns reads outgoing
    # (downward) from this level and writes incoming (upward) from the below level.
    # In other words, reads and writes are between a level and the one below it.
    # Let "size" be the tile_size of the above level!
    # => The returned value must be multiplied by the factors above it.
    """
    Returns the memory operations between this level and the one below it.
    Specifically: returns reads going downward from this level and writes
    coming upward from lower levels. In other words, reads and writes
    between this level and the one(s) below it.

    => The returned value must be multiplied by the iterations happening
       on levels above this one in order to get the true total.

    Bypasses are handled like this:
    If this level is the last before a bypass of a certain operand, it
    will be invoking this same method on the last level before the next
    one which does not bypass the operand in question.
    This way MOPs are computed w.r.t. the tile size of such level, which
    corresponds with the actual tiles being accessed from this level.
    For each level encountered in between, MOPs are scaled accordingly.

    Lastly, factors encountered on the K dimension are returned too, to
    allow the caller to remove one such iteration to account for the
    presence or absence of the bias (bias_read flag).
    
    Arguments:
    - in_bp, w_bp, out_bp: whether the respective operands are bypassed or not,
                           defaults to the instance's setting if not provided.
                           Values are '0' for bypass, '1' for keep.
    - ignore_bypasses: if True, MOPs are returned only relative to this level's
                       accesses from strictly adjacent levels, bypassed operands
                       will thus show 0 MOPs.
    """
    def MOPs(self, in_bp = None, w_bp = None, out_bp= None, ignore_bypasses = False):
        in_bp = in_bp if in_bp != None else self.in_bp
        w_bp = w_bp if w_bp != None else self.w_bp
        out_bp = out_bp if out_bp != None else self.out_bp
        # ignore loops at one
        actual_dataflow = list(filter(lambda dim : self.factors.dimProduct(dim) > 1, self.dataflow))
        # stationarity calculation for inputs
        in_reads = in_bp
        if in_reads:
            # elements per tile - size of the tiles moved between this level and the one below; for summed indices, their tile_sizes must be added (with a -1 per sum), and them multiplied with the others
            in_reads = prod(map(lambda dim_sum : sum(self.tile_sizes[dim] for dim in dim_sum) - len(dim_sum) + 1, self.arch.coupling.in_coupling))
            i = len(actual_dataflow) - 1
            if not self.next_is_compute:
                while i >= 0 and (actual_dataflow[i] not in self.arch.coupling.flat_in_coupling): # skip contigous innermost orthogonal dimensions (determining stationarity)
                    i -= 1
            # try handling the innermost actual iteration by itself to manage a potential sum of indices: for a pair of innermost consecutive summed indices, if the inner one has 2 iterations, one is read/write is not needed
            if i > 0 and self.factors.dimProduct(actual_dataflow[i]) == 2 and any(actual_dataflow[i] in dim_sum and actual_dataflow[i-1] in dim_sum for dim_sum in self.arch.coupling.in_coupling):
                i -= 1
            for dim in actual_dataflow[:i+1]: # accumulate the remaining iterations
                in_reads *= self.factors.dimProduct(dim)
        # stationarity calculation for weights
        w_reads = w_bp
        if w_reads:
            w_reads = prod(map(lambda dim_sum : sum(self.tile_sizes[dim] for dim in dim_sum) - len(dim_sum) + 1, self.arch.coupling.w_coupling))
            i = len(actual_dataflow) - 1
            if not self.next_is_compute:
                while i >= 0 and (actual_dataflow[i] not in self.arch.coupling.flat_w_coupling):
                    i -= 1
            if i > 0 and self.factors.dimProduct(actual_dataflow[i]) == 2 and any(actual_dataflow[i] in dim_sum and actual_dataflow[i-1] in dim_sum for dim_sum in self.arch.coupling.w_coupling):
                i -= 1
            for dim in actual_dataflow[:i+1]:
                w_reads *= self.factors.dimProduct(dim)
        # stationarity calculation for outputs
        out_reads = out_bp
        out_reads_factors = out_bp # this collects the factors along dimensions orthogonal to the output, returned to handle the presence/absence of the bias
        if out_reads:
            out_reads = prod(map(lambda dim_sum : sum(self.tile_sizes[dim] for dim in dim_sum) - len(dim_sum) + 1, self.arch.coupling.out_coupling))
            i = len(actual_dataflow) - 1
            if not self.next_is_compute:
                while i >= 0 and (actual_dataflow[i] not in self.arch.coupling.flat_out_coupling):
                    i -= 1
            if i > 0 and self.factors.dimProduct(actual_dataflow[i]) == 2 and any(actual_dataflow[i] in dim_sum and actual_dataflow[i-1] in dim_sum for dim_sum in self.arch.coupling.out_coupling):
                i -= 1
            for dim in actual_dataflow[:i+1]:
                out_reads *= self.factors.dimProduct(dim)
                if dim not in self.arch.coupling.flat_out_coupling:
                    out_reads_factors *= self.factors.dimProduct(dim)
        out_writes = out_reads

        #print("BEFORE BYPASS:\n", f"{self.name}:{chr(9) * (2 - len(self.name)//8)}{in_reads} In_R, {w_reads} W_R, {out_reads} Our_R, {in_reads + w_reads + out_reads} Tot_R, {out_writes} Out_W, {out_reads_factors} Out_R_Fac")
        # handle bypasses
        if not ignore_bypasses:
            for operand, levels in self.next_levels_with_bypass.items():
                if levels != None:
                    in_between, level = levels[:-1], levels[-1]
                    in_reads_bp, w_reads_bp, out_reads_bp, out_writes_bp, out_reads_bp_factors = level.MOPs(operand == 'in', operand == 'w', operand == 'out', True)
                    # bulid the inner-most dataflow, by piling one against the other all non-1 loops, then look at which is the innermost dimension, that is the one that matters!
                    # TL;DR: consider the dataflow only w.r.t. the innermost loop, ignoring those with 1 iteration!
                    if any(level.factors.dimProduct(dim) > 1 for dim in level.dataflow): # at least a loop not at one implies that this level has a dataflow
                        stationarity_to_address = False
                    else:
                        stationarity_to_address = True
                    for in_btwn in in_between[::-1]:
                        if isinstance(in_btwn, MemLevel):
                            if stationarity_to_address:
                                # all inner loops were 1s, deal with the dataflow now!
                                # ignore loops at one
                                actual_dataflow_bp = list(filter(lambda dim : in_btwn.factors.dimProduct(dim) > 1, in_btwn.dataflow))
                                if len(actual_dataflow_bp) > 0:
                                    stationarity_to_address = False
                                # stationarity calculation for inputs
                                if in_reads_bp:
                                    i = len(actual_dataflow_bp) - 1
                                    while i >= 0 and (actual_dataflow_bp[i] not in self.arch.coupling.flat_in_coupling):
                                        i -= 1
                                    if i > 0 and in_btwn.factors.dimProduct(actual_dataflow_bp[i]) == 2 and any(actual_dataflow_bp[i] in dim_sum and actual_dataflow_bp[i-1] in dim_sum for dim_sum in self.arch.coupling.in_coupling):
                                        i -= 1
                                    for dim in actual_dataflow_bp[:i+1]:
                                        in_reads_bp *= in_btwn.factors.dimProduct(dim)
                                # stationarity calculation for weights
                                if w_reads_bp:
                                    i = len(actual_dataflow_bp) - 1
                                    while i >= 0 and (actual_dataflow_bp[i] not in self.arch.coupling.flat_w_coupling):
                                        i -= 1
                                    if i > 0 and in_btwn.factors.dimProduct(actual_dataflow_bp[i]) == 2 and any(actual_dataflow_bp[i] in dim_sum and actual_dataflow_bp[i-1] in dim_sum for dim_sum in self.arch.coupling.w_coupling):
                                        i -= 1
                                    for dim in actual_dataflow_bp[:i+1]:
                                        w_reads_bp *= in_btwn.factors.dimProduct(dim)
                                # stationarity calculation for outputs
                                if out_reads_bp:
                                    i = len(actual_dataflow_bp) - 1
                                    while i >= 0 and (actual_dataflow_bp[i] not in self.arch.coupling.flat_out_coupling):
                                        i -= 1
                                    if i > 0 and in_btwn.factors.dimProduct(actual_dataflow_bp[i]) == 2 and any(actual_dataflow_bp[i] in dim_sum and actual_dataflow_bp[i-1] in dim_sum for dim_sum in self.arch.coupling.out_coupling):
                                        i -= 1
                                    for dim in actual_dataflow_bp[:i+1]:
                                        out_reads_bp *= in_btwn.factors.dimProduct(dim)
                                        out_writes_bp *= in_btwn.factors.dimProduct(dim)
                                        if dim not in self.arch.coupling.flat_out_coupling:
                                            out_reads_bp_factors *= in_btwn.factors.dimProduct(dim)
                            else:
                                # dataflow already handled among inner loops
                                in_btwn_factors_full = in_btwn.factors.fullProduct()
                                w_reads_bp = in_btwn_factors_full*w_reads_bp
                                in_reads_bp = in_btwn_factors_full*in_reads_bp
                                out_reads_bp = in_btwn_factors_full*out_reads_bp
                                out_writes_bp = in_btwn_factors_full*out_writes_bp
                                out_reads_bp_factors *= prod(in_btwn.factors.dimProduct(dim) for dim in in_btwn.dataflow if dim not in self.arch.coupling.flat_out_coupling)
                        else:
                            in_reads_bp, w_reads_bp, out_reads_bp, out_writes_bp = in_btwn.mulByDim(in_reads_bp, w_reads_bp, out_reads_bp, out_writes_bp)
                            # do not update out_reads_bp_factors here, because in it go only iterations of which the first one is skipped,
                            # while in a fanout all fanned-out copies of the inner loop behave the same, there isn't a first different spatial iteration or anything
                        #print("IN BETWEEN BYPASS:\n", f"{in_btwn.name}:{chr(9) * (2 - len(in_btwn.name)//8)}{in_reads_bp} In_R, {w_reads_bp} W_R, {out_reads_bp} Our_R, {in_reads_bp + w_reads_bp + out_reads_bp} Tot_R, {out_writes_bp} Out_W, {out_reads_bp_factors} Out_R_Fac")
                    # consider the dataflow only among the three innermost loops, unless all loops seen until now were 1s
                    if stationarity_to_address:
                        # all inner loops were 1s, deal with the dataflow now!
                        if in_reads_bp:
                            i = len(actual_dataflow) - 1
                            while i >= 0 and (actual_dataflow[i] not in self.arch.coupling.flat_in_coupling):
                                i -= 1
                            if i > 0 and self.factors.dimProduct(actual_dataflow[i]) == 2 and any(actual_dataflow[i] in dim_sum and actual_dataflow[i-1] in dim_sum for dim_sum in self.arch.coupling.in_coupling):
                                i -= 1
                            for dim in actual_dataflow[:i+1]:
                                in_reads_bp *= self.factors.dimProduct(dim)
                        # stationarity calculation for weights
                        if w_reads_bp:
                            i = len(actual_dataflow) - 1
                            while i >= 0 and (actual_dataflow[i] not in self.arch.coupling.flat_w_coupling):
                                i -= 1
                            if i > 0 and self.factors.dimProduct(actual_dataflow[i]) == 2 and any(actual_dataflow[i] in dim_sum and actual_dataflow[i-1] in dim_sum for dim_sum in self.arch.coupling.w_coupling):
                                i -= 1
                            for dim in actual_dataflow[:i+1]:
                                w_reads_bp *= self.factors.dimProduct(dim)
                        # stationarity calculation for outputs
                        if out_reads_bp:
                            i = len(actual_dataflow) - 1
                            while i >= 0 and (actual_dataflow[i] not in self.arch.coupling.flat_out_coupling):
                                i -= 1
                            if i > 0 and self.factors.dimProduct(actual_dataflow[i]) == 2 and any(actual_dataflow[i] in dim_sum and actual_dataflow[i-1] in dim_sum for dim_sum in self.arch.coupling.out_coupling):
                                i -= 1
                            for dim in actual_dataflow[:i+1]:
                                out_reads_bp *= self.factors.dimProduct(dim)
                                out_writes_bp *= self.factors.dimProduct(dim)
                                if dim not in self.arch.coupling.flat_out_coupling:
                                    out_reads_bp_factors *= self.factors.dimProduct(dim)
                    else:
                        # dataflow handled among inner loops
                        factors_full = self.factors.fullProduct()
                        w_reads_bp = factors_full*w_reads_bp
                        in_reads_bp = factors_full*in_reads_bp
                        out_reads_bp = factors_full*out_reads_bp
                        out_writes_bp = factors_full*out_writes_bp
                        out_reads_bp_factors *= prod(self.factors.dimProduct(dim) for dim in self.dataflow if dim not in self.arch.coupling.flat_out_coupling)
                    #print(f"BYPASS ({operand}):\n", f"{self.name}->{level.name}:{chr(9) * (3 - (len(self.name)+len(level.name))//8)}{in_reads_bp} In_R, {w_reads_bp} W_R, {out_reads_bp} Our_R, {in_reads_bp + w_reads_bp + out_reads_bp} Tot_R, {out_writes_bp} Out_W, {out_reads_bp_factors} Out_R_Fac")
                    in_reads += in_reads_bp
                    w_reads += w_reads_bp
                    out_reads += out_reads_bp
                    out_writes += out_writes_bp
                    if operand == 'out':
                        out_reads_factors = out_reads_bp_factors
        #print("AFTER BYPASS:\n", f"{self.name}:{chr(9) * (2 - len(self.name)//8)}{in_reads} In_R, {w_reads} W_R, {out_reads} Our_R, {in_reads + w_reads + out_reads} Tot_R, {out_writes} Out_W, {out_reads_factors} Out_R_Fac\n")
        return in_reads, w_reads, out_reads, out_writes, out_reads_factors                                                   #S;G

    """
    Returns the provided MOPs (or newly calculated MOPs for this level)
    scaled by the MOPs's weight/energy at this level.
    """
    def WMOPs(self, reads = None, writes = None):
        if not (reads and writes):
            reads = reads if reads else self.in_reads + self.w_reads + self.out_reads
            writes = writes if writes else self.in_writes + self.w_writes + self.out_writes
        return self.read_access_energy * reads + self.write_access_energy * writes

    """
    Returns the total leaked energy during the provided clock cycles.
    """
    def Leakage(self, cycles):
        return cycles*self.leakage_energy

    """
    Returns True iif factors present on this level satisfy all of
    its constraints, including fitting in the available memory.
    """
    def checkConstraints(self):
        return self.factors.mem_footprint(self.tile_sizes, self.arch.coupling, not self.bypasses or 'in' not in self.bypasses, not self.bypasses or 'w' not in self.bypasses, not self.bypasses or 'out' not in self.bypasses) <= self.size/self.multiple_buffering and super().checkConstraints()

    """
    Returns a string describing the current violation of constraints, if any.
    """
    def logConstraintsViolation(self):
        if not super().checkConstraints():
            return super().logConstraintsViolation()
        elif not self.checkConstraints():
            mem_footprint = self.factors.mem_footprint(self.tile_sizes, self.arch.coupling, not self.bypasses or 'in' not in self.bypasses, not self.bypasses or 'w' not in self.bypasses, not self.bypasses or 'out' not in self.bypasses)
            return f"CONSTRAINTS VIOLATION: Arch: {self.arch.name} -> Level: {self.name}: memory used: {mem_footprint} VS memory available: {self.size/self.multiple_buffering:.0f}"
        return ""

    def __str__(self):
        return f"{super().__str__()}, size: {self.size}, read_access_energy: {self.read_access_energy}, write_access_energy: {self.write_access_energy}, leakage_energy: {self.leakage_energy}, read_bandwidth: {self.read_bandwidth}, write_bandwidth: {self.write_bandwidth}, bypasses: {self.bypasses}, multiple_buffering: {self.multiple_buffering}"


"""
Abstract class for a level introducing multiple spatial instances.
"""
class SpatialLevel(Level):
    dims: None
    mesh: None


"""
A Spatial Fanout Level within the architecture, the core of a spatial architecture,
all subsequent levels will be replicated "mesh" times, each replica executing one
of the iterations done by this level, by each running the same inner loops as the
others, with partially different data.

Constructor arguments:
- name: the level's name
- mesh: the maximum spatial fanout available at this level
- dim: single dimension to spatially unroll at this level, if dim is specified,
       dims must not be specified
- dims: list of dimensions to spatially unroll at this level, if dims is specified,
        dim must not be specified. The order in dims does not matter.
- area: the area occupied by the entire interconnect (in um^2).
- pe_to_pe: if True, data is not multicasted in one shot, but sent to only the first
            fanout entry, which then forwards it to the second, and so on, just like
            the operands flowing in/out of a systolic array (or, like in a pipeline).
            This adds a warmup overhead to latency, but does not affect the total MOPs.
- spatial_multicast_support: True (default) if this level supports spatial multicast
- spatial_reduction_support: True (default) if this level supports spatial reduction
- power_gating_support: True if instances immediately following this level can be
                        power-gated when not in use, saving leakage power.
- factors: specifies the initial factors for this level, should not be normally
           specified aside for initializing MSE from a specific configuration
- tile_sizes: specifies the initial tile sizes for this level, should not be normally
              specified aside for initializing MSE from a specific configuration,
              in which case it must be consistent with any other factors initialization
- factors_constraints: constraints on the factors that must be placed on this level.
                      Valid dictionary keys use dimension names, e.g. for a GEMM:
                          - 'M', 'K', and 'N' for exact values;
                          - 'M<=', 'K<=', and 'N<=' for upper bounds;
                          - 'M>=', 'K>=', and 'N>=' for lower bounds;
                      NOTE: the use of the '<=' and '>=' constraints does not shorten the
                            mapper's runtime as much as exact constraints.
"""
# IMPORTANT:
# Currently fanout levels reuse all operands mapped on them, period. However this should be up to hardware support.
# Therefore add here a value N which determines how many operands can be spatially reused.
# Practically, if N = 1 and you have 2 loops with some iterations, for the inner loop operate as if spatial_multicast_support
# and spatial_reduction_support were True (if they were False to begin with, let them be False (&&)), for the second set
# them both to false.
# Obviously, in case N < |dims| you need to change the "iterate permutations" step to actually permute spatial loops!!!
# BETTER: make spatial_reduction_support and spatial_multicast_support be specified per-dimension!
class FanoutLevel(SpatialLevel):
    def __init__(self, name, mesh, dim : str = None, dims : list[str] = None, area = None, pe_to_pe = False, spatial_multicast_support = True, spatial_reduction_support = True, power_gating_support = False, factors = None, tile_sizes = None, factors_constraints = None):
        self.name = name
        self.dim = dim
        self.dims =  dims
        self.mesh = mesh
        self.area = area
        self.pe_to_pe = pe_to_pe # True in all cases where the operand independent of "dim" (e.g.: in a GEMM, if dim = M, such operand is the input) is forwarded pe->pe rather than multicasted
        self.spatial_multicast_support = spatial_multicast_support
        self.spatial_reduction_support = spatial_reduction_support
        self.power_gating_support = power_gating_support
        self.factors = factors
        self.tile_sizes = tile_sizes
        self.factors_constraints = factors_constraints if factors_constraints else {}

    """
    Sets up a pointer back to the whole architecture.
    Ultimates the initialization of the level and validates its attributes.
    """
    def initArch(self, arch : Arch):
        self.arch = arch
        assert (self.dim and not self.dims) or (self.dims and not self.dim), f"Arch: {arch.name} -> Level: {self.name}: exactly one of dim ({self.dim}) or dims ({self.dims}) must be specified."
        self.dims = [self.dim] if self.dim else self.dims
        del self.dim
        self.dataflow = self.dims
        assert all([dim in arch.coupling.dims for dim in self.dataflow]), f"Arch: {arch.name} -> Level: {self.name}: accepted names for dimensions are solely M, K and N, provided ones were {self.dataflow}."
        assert self.mesh > 0, f"Arch: {arch.name} -> Level: {self.name}: a spatial fanout must have a mesh ({self.mesh}) of at least 1."
        assert not self.area or self.area >= 0, f"Arch: {arch.name} -> Level: {self.name}: a negative area ({self.area}) does not mean anything."
        assert not self.pe_to_pe or (self.spatial_multicast_support and self.spatial_reduction_support), f"Arch: {arch.name} -> Level: {self.name}: pe-to-pe forwarding is a form of spatial multicast or reduction, which must then both be supported to use it."
        self.factors = self.factors if self.factors else Factors(arch.coupling.dims)
        self.tile_sizes = self.tile_sizes if self.tile_sizes else Shape({dim: 1 for dim in arch.coupling.dims})
        assert all([constr[0] in self.dataflow and constr[1:] in ['', '>=', '<='] for constr in self.factors_constraints.keys()]), f"Arch: {arch.name} -> Level: {self.name}: all keys within factor constraints ({list(self.factors_constraints.keys())}) must be a dimension of the dataflow ({self.dataflow}) and in the form 'dim', 'dim<=', or 'dim>='."
        assert all([sum(constr[0] == dim for constr in self.factors_constraints.keys()) <= 1 for dim in self.dataflow]), f"Arch: {arch.name} -> Level: {self.name}: each dimension must occur at most once in constraints ({list(self.factors_constraints.keys())}), regardless of the use of '>=' or '<='."
        assert all([value > 0 for value in self.factors_constraints.values()]), f"Arch: {arch.name} -> Level: {self.name}: all constraints ({self.factors_constraints}) must have a value strictly > 0."

    """
    Let inputs be the amount of operations occuring on a level below this fanout,
    this method returns the amount of operations the seen above the fanout, accounting
    for spatial multicast and spatial reduction support of operands!
    """
    def mulByDim(self, in_reads, w_reads, out_reads, out_writes):
        in_reads *= prod(map(lambda dim_sum : sum(self.factors.dimProduct(dim) for dim in dim_sum) - len(dim_sum) + 1, self.arch.coupling.in_coupling))
        w_reads *= prod(map(lambda dim_sum : sum(self.factors.dimProduct(dim) for dim in dim_sum) - len(dim_sum) + 1, self.arch.coupling.w_coupling))
        out_reads *= prod(map(lambda dim_sum : sum(self.factors.dimProduct(dim) for dim in dim_sum) - len(dim_sum) + 1, self.arch.coupling.out_coupling))
        out_writes = out_reads
        if not self.spatial_multicast_support:
            in_reads *= prod(self.factors.dimProduct(dim) for dim in self.dataflow if dim not in self.arch.coupling.flat_in_coupling)
            w_reads *= prod(self.factors.dimProduct(dim) for dim in self.dataflow if dim not in self.arch.coupling.flat_w_coupling)
            out_reads *= prod(self.factors.dimProduct(dim) for dim in self.dataflow if dim not in self.arch.coupling.flat_out_coupling)
        if not self.spatial_reduction_support:
            out_writes *= prod(self.factors.dimProduct(dim) for dim in self.dataflow if dim not in self.arch.coupling.flat_out_coupling)
        
        return in_reads, w_reads, out_reads, out_writes

    """
    Returns the clock cycles required by this fanout level to sustain the bandwidth
    to move all operands across the above and below memory/compute levels.
    
    TODO: implement this w.r.t. a NoC model.
    
    => The returned value must be multiplied by the factors above it.
    """
    def latency(self):
        return 0 # change this if we model the network's latency

    """
    Returns True iif factors present on this level satisfy all of its constraints,
    including not exceeding the physical mesh.
    """
    def checkConstraints(self):
        return self.factors.fullProduct() <= self.mesh and super().checkConstraints()

    """
    Returns a string describing the current violation of constraints, if any.
    """
    def logConstraintsViolation(self):
        if not super().checkConstraints():
            return super().logConstraintsViolation()
        elif not self.checkConstraints():
            return f"CONSTRAINTS VIOLATION: Arch: {self.arch.name} -> Level: {self.name}: spatial iterations used: {self.factors.fullProduct()} VS available instances (mesh): {self.mesh}"
        return ""

    def __str__(self):
        return f"{super().__str__()}, mesh: {self.mesh}, pe_to_pe: {self.pe_to_pe}, spatial_multicast_support: {self.spatial_multicast_support}, spatial_reduction_support: {self.spatial_reduction_support}, power_gating_support: {self.power_gating_support}"


"""
A Compute Level within the architecture, it is a placeholder for any processing
element (PE) capable of multiply and accumulate (MAC).

Note: iterations at this level are equivalent to requiring that a PE can, in the
clock-cycles specified in "cycles", execute all such iterations. It is also assumed
that the data required for all iterations done at this level is held within hardware
registers whose energy cost is modeled by the "compute energy", together with the
actual compute cost. As such, iterations done here are equivalent to a PE capable
of multiple concurrent (parallel or pipelined (chaining accumulation output)) MACs.
Then intuitively, increasing the number of iterations done at this level linearly
increases the required bandwidth of all memory levels feeding it, as they need to
keep up with the concurrent MACs done here.

Constructor arguments:
- name: the level's name
- mesh: how many concurrent (parallel or pipelined) MACs the compute element
        can perform within "cycles" clock-cycles.
- compute_energy: the energy required for a single MAC (regardless of how many
                  you run concurrently), accounting for all computation-related
                  costs at the PE level.
- cycles: the number of clock cycles of latency required to execute "size" MACs
- leakage_energy: energy leaked each clock cycle by the component (in pJ/cc)
- area: the area occupied by the entire PE (in um^2).
- dim: single dimension along which MAC operations are picked to run concurrently
       on this level, if dim is specified, dims must not be specified
       NOTE: when mesh is 1, both dim and dims can be omitted
- dims: list of dimensions from which MAC operations are picked to run concurrently
       on this level, if dims is specified, dim must not be specified.
       The order in dims does not matter.
       NOTE: when mesh is 1, both dim and dims can be omitted
- factors: specifies the initial factors for this level, should not be normally
           specified aside for initializing MSE from a specific configuration
- tile_sizes: specifies the initial tile sizes for this level, should not be normally
              specified aside for initializing MSE from a specific configuration,
              in which case it must be consistent with any other factors initialization
- factors_constraints: constraints on the factors that must be placed on this level.
                      Valid dictionary keys use dimension names, e.g. for a GEMM:
                          - 'M', 'K', and 'N' for exact values;
                          - 'M<=', 'K<=', and 'N<=' for upper bounds;
                          - 'M>=', 'K>=', and 'N>=' for lower bounds;
                      NOTE: the use of the '<=' and '>=' constraints does not shorten the
                            mapper's runtime as much as exact constraints.
"""
class ComputeLevel(SpatialLevel):
    def __init__(self, name, mesh, compute_energy, cycles, dim : str = None, dims : list[str] = None, leakage_energy = 0, area = None, factors = None, tile_sizes = None, factors_constraints = None):
        self.name = name
        self.dim = dim
        self.dims = dims
        self.mesh = mesh # for a systolic array, this is the length of the operand buffers
        self.compute_energy = compute_energy
        self.leakage_energy = leakage_energy
        self.area = area
        self.cycles = cycles # clock cycles used per element in the inner dimension (latency of one MAC)
        self.factors = factors
        self.tile_sizes = tile_sizes
        self.factors_constraints = factors_constraints if factors_constraints else {}

        # STATISTICS:
        self.instances = 1 # this are the used/active instances
        self.temporal_iterations = 0

    """
    Sets up a pointer back to the whole architecture.
    Ultimates the initialization of the level and validates its attributes.
    """
    def initArch(self, arch : Arch):
        self.arch = arch
        assert self.mesh == 1 or (self.dim and not self.dims) or (self.dims and not self.dim), f"Arch: {arch.name} -> Level: {self.name}: when mesh ({self.mesh}) is > 1, exactly one of dim ({self.dim}) or dims ({self.dims}) must be specified."
        self.dims = ([self.dim] if self.dim else self.dims) if self.dim or self.dims else []
        self.dataflow = self.dims
        assert all([dim in arch.coupling.dims for dim in self.dataflow]), f"Arch: {arch.name} -> Level: {self.name}: accepted names for dimensions are solely M, K and N, provided ones were {self.dataflow}."
        assert self.mesh > 0, f"Arch: {arch.name} -> Level: {self.name}: a zero or negative size ({self.mesh}) does not make sense."
        assert self.compute_energy >= 0, f"Arch: {arch.name} -> Level: {self.name}: a negative compute energy ({self.compute_energy}) does not mean anything (unless you watched too much Gundam and discovered Minovsky particles...)."
        assert self.leakage_energy >= 0, f"Arch: {arch.name} -> Level: {self.name}: a negative leakage energy ({self.leakage_energy}) does not mean anything (unless you watched too much Gundam 00 and discovered GN particles...)."
        assert not self.area or self.area >= 0, f"Arch: {arch.name} -> Level: {self.name}: a negative area ({self.area}) does not mean anything."
        assert self.cycles >= 0, f"Arch: {arch.name} -> Level: {self.name}: a negative number of clock-cycles per MAC ({self.cycles}) does not mean anything."
        self.factors = self.factors if self.factors else Factors(arch.coupling.dims)
        self.tile_sizes = self.tile_sizes if self.tile_sizes else Shape({dim: 1 for dim in arch.coupling.dims})
        assert all([constr[0] in self.dataflow and constr[1:] in ['', '>=', '<='] for constr in self.factors_constraints.keys()]), f"Arch: {arch.name} -> Level: {self.name}: all keys within factor constraints ({list(self.factors_constraints.keys())}) must be a dimension of the dataflow ({self.dataflow}) and in the form 'dim', 'dim<=', or 'dim>='."
        assert all([sum(constr[0] == dim for constr in self.factors_constraints.keys()) <= 1 for dim in self.dataflow]), f"Arch: {arch.name} -> Level: {self.name}: each dimension must occur at most once in constraints ({list(self.factors_constraints.keys())}), regardless of the use of '>=' or '<='."
        assert all([value > 0 for value in self.factors_constraints.values()]), f"Arch: {arch.name} -> Level: {self.name}: all constraints ({self.factors_constraints}) must have a value strictly > 0."

    """
    Returns the clock cycles required by this compute level to perform ALL its
    allocated iterations. The reasoning being that such iterations occur all either
    within the same set of cycles or in pipeline with one step requiring the hereby
    returned amount of cycles.
    
    => The returned value must be multiplied by the factors above it.
    """
    def latency(self):
        return self.cycles

    """
    Returns the energy required by this level to perform all MACs across its internal
    iterations, times "iterations", which represents the number of iterations done by
    the hierarchy of levels on top of this one.
    """
    def computeCost(self, iterations = 1):
        return self.compute_energy*self.factors.fullProduct()*iterations

    """
    Returns the total leaked energy during the provided clock cycles.
    """
    def Leakage(self, cycles):
        return cycles*self.leakage_energy

    """
    Returns True iif factors present on this level satisfy all of its constraints,
    including not exceeding the phisical number of concurrent MACs performed here.
    """
    def checkConstraints(self):
        return self.factors.fullProduct() <= self.mesh and super().checkConstraints()

    """
    Returns a string describing the current violation of constraints, if any.
    """
    def logConstraintsViolation(self):
        if not super().checkConstraints():
            return super().logConstraintsViolation()
        elif not self.checkConstraints():
            return f"CONSTRAINTS VIOLATION: Arch: {self.arch.name} -> Level: {self.name}: concurrent MACs used: {self.factors.fullProduct()} VS concurrent MACs available: {self.mesh}"
        return ""

    def __str__(self):
        return f"{super().__str__()}, mesh: {self.mesh}, compute_energy: {self.compute_energy}, leakage_energy: {self.leakage_energy}, cycles: {self.cycles}"