from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Any
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
    dataflow : list[str] # order of the loops | e.g. ['M', 'K', 'N']
    factors : Factors # iterations done for the dimensions at this lever
    tile_sizes : Shape # indicate the size of a tile used in the level BELOW
    # NOTE: also, this is the data sent inward at each iterations
    # NOTE: tile sizes are, in other words, the number of elements jumped
    #       by one iteration at this level an any dimension
    # NOTE: another interpretation is that tile sizes are the amount by
    #       which inner levels will move along a dimension
    # NOTE: finally, tile sizes can be seen as the coefficient to be applied
    #       to a level's index along the respective dimension
    # NOTE: tile sizes in fanout levels represent the size of each tile
    #       stored in an instance below them
    # NOTE: tile sizes are updated in "moveFactor"
    
    def __init__(self, dataflow : list[str], factors : Factors, tile_sizes : Shape):
        self.dataflow = dataflow
        self.factors = factors
        self.tile_sizes = tile_sizes
    
    def __str__(self) -> str:
        return f"dataflow: {self.dataflow}, factors: {self.factors}, tile_sizes: {self.tile_sizes}"


"""
Abstract class representing a level of the accelerator's architecture.

NOTE: the hierarchy is read just as the architecture is specified, with
"below" levels being closer to the computation and "above" ones closer
to DRAM or slower memories.
"""
class Level(LevelCore):
    name : str
    arch : Arch
    factors_constraints : dict[str, int]
    area : Optional[float]

    """
    Add "amount" instances of the provided factor to those of
    "dimension" in the current level.
    """
    def addFactor(self, dimension : str, factor : int, amount : int = 1) -> None:
        self.factors.addFactor(dimension, factor, amount)

    """
    Removes "amount" instances of the provided factor from those of
    "dimension" in the current level.
    Return False if the removal failed because the current level does not
    have at least "amount" instances of "factor" along "dimension".
    """
    def removeFactor(self, dimension : str, factor : int, amount : int = 1) -> bool:
        return self.factors.removeFactor(dimension, factor, amount)

    """
    Returns True iif factors present on this level satisfy all of
    its constraints.
    """
    def checkConstraints(self) -> bool:
        return (all([(dim not in self.factors_constraints or self.factors_constraints[dim] == self.factors.dimProduct(dim)) for dim in self.dataflow]) and
                all([(dim + '<=' not in self.factors_constraints or self.factors_constraints[dim + '<='] >= self.factors.dimProduct(dim)) for dim in self.dataflow]) and
                all([(dim + '>=' not in self.factors_constraints or self.factors_constraints[dim + '>='] <= self.factors.dimProduct(dim)) for dim in self.dataflow]))

    """
    Returns a string describing the current violation of constraints,
    if any.
    """
    def logConstraintsViolation(self) -> str:
        if not self.checkConstraints():
            return (f"CONSTRAINTS VIOLATION: Arch: {self.arch.name} -> Level: {self.name}: "
                + ', '.join([f"constrained {dim} == {self.factors_constraints[dim]} VS obtained {dim}: {self.factors.dimProduct(dim)}, " for dim in self.dataflow if (dim in self.factors_constraints and self.factors_constraints[dim] != self.factors.dimProduct(dim))])
                + ', '.join([f"constrained {dim} <= {self.factors_constraints[dim + '<=']} VS obtained {dim}: {self.factors.dimProduct(dim)}, " for dim in self.dataflow if (dim + '<=' in self.factors_constraints and self.factors_constraints[dim + '<='] >= self.factors.dimProduct(dim))])
                + ', '.join([f"constrained {dim} >= {self.factors_constraints[dim + '>=']} VS obtained {dim}: {self.factors.dimProduct(dim)}, " for dim in self.dataflow if (dim + '>=' in self.factors_constraints and self.factors_constraints[dim + '>='] <= self.factors.dimProduct(dim))]))
        return ""

    def __getitem__(self, key : str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key : str, value : Any) -> None:
        setattr(self, key, value)

    def __str__(self) -> str:
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
- multiple_reuses: if True (default) both stationarity reuse (loops skip) and halo reuse can be
                   leveraged at once. Otherwise only the one bound to the innermost loop is leveraged.
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
    size : int
    values_per_wordline : int
    read_access_energy : float
    write_access_energy : float
    leakage_energy : float
    read_bandwidth : float
    write_bandwidth : float
    dataflow_constraints : list[str]
    bypasses : list[str]
    in_bp : bool
    w_bp : bool
    out_bp : bool
    multiple_buffering : int
    multiple_reuses : bool
    
    # POINTERS TO OTHER LEVELS:
    next_is_compute : bool # flag indicating if this is the last memory before the compute level (initialized in setupSpatialLevelPointers)
    next_spatials : Optional[list[Level]] # list of spatial levels following this one (can be empty, initialized in setupSpatialLevelPointers)
    next_levels_with_bypass : dict[str, Optional[list[Level]]] # pointers to the next level storing an operand hereby bypassed (initialized in setupBypasses)

    # STATISTICS:
    instances = 1 # this are the used/active instances
    temporal_iterations = 0
    in_reads = 0
    w_reads = 0
    out_reads = 0
    in_writes = 0
    w_writes = 0
    out_writes = 0
    last_out_reads = 0
    last_out_writes = 0
    latency_read_drain = 0
    latency_fill_update = 0
    cc_per_tile = 0
    stall_cycles = 0
    ideal_bandwidth_read = 0
    ideal_bandwidth_update = 0
    ideal_bandwidth_fill = 0
    ideal_bandwidth_drain = 0

    def __init__(self, name : str, size : int, value_access_energy : Optional[float] = None, wordline_access_energy : Optional[float] = None, word_bits : Optional[int] = None, value_bits : Optional[int] = None, leakage_energy : float = 0, area : Optional[float] = None, bandwidth : Optional[float] = None, dataflow : Optional[list[str]] = None, factors : Optional[Factors] = None, tile_sizes : Optional[Shape] = None, factors_constraints : Optional[dict[str, int]] = None, dataflow_constraints : Optional[list[str]] = None, bypasses : Optional[list[str]] = None, multiple_buffering : int = 1, multiple_reuses : bool = True, read_value_access_energy : Optional[float] = None, write_value_access_energy : Optional[float] = None, read_wordline_access_energy : Optional[float] = None, write_wordline_access_energy : Optional[float] = None, read_bandwidth : Optional[float] = None, write_bandwidth : Optional[float] = None):
        self.name = name
        self.dataflow = dataflow
        self.size = size
        self._word_bits = word_bits
        self._value_bits = value_bits
        self._wordline_access_energy = wordline_access_energy
        self._value_access_energy = value_access_energy
        self._read_wordline_access_energy = read_wordline_access_energy
        self._write_wordline_access_energy = write_wordline_access_energy
        self._read_value_access_energy = read_value_access_energy
        self._write_value_access_energy = write_value_access_energy
        self.leakage_energy = leakage_energy
        self.area = area
        self._bandwidth = bandwidth
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
        self.multiple_reuses = multiple_reuses

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
        assert (self._value_access_energy or (self._read_value_access_energy and self._write_value_access_energy)) or (self._word_bits and self._value_bits and (self._wordline_access_energy or (self._read_wordline_access_energy and self._write_wordline_access_energy))), f"Arch: {arch.name} -> Level: {self.name}: either value_access_energy ({self._value_access_energy}) or read_value_access_energy ({self._read_value_access_energy}) and write_value_access_energy ({self._write_value_access_energy}) must be specified, alternatively, you can specify word_bits ({self._word_bits}) and value_bits ({self._value_bits}) and either wordline_access_energy ({self._wordline_access_energy}) or read_wordline_access_energy ({self._read_wordline_access_energy}) and write_wordline_access_energy ({self._write_wordline_access_energy}). In any case, when if either of read_*_access_energy or write_*_access_energy is specified, the other must be present as well."
        if (self._value_access_energy or (self._read_value_access_energy and self._write_value_access_energy)):
            self.values_per_wordline = 1
            self.read_access_energy = self._read_value_access_energy if self._read_value_access_energy else self._value_access_energy
            self.write_access_energy = self._write_value_access_energy if self._write_value_access_energy else self._value_access_energy
        else:
            assert self._word_bits >= self._value_bits, f"Arch: {arch.name} -> Level: {self.name}: word_bits ({self._word_bits}) must be more than value_bits ({self._value_bits}), otherwise a value cannot fit on a single wordline."
            self.values_per_wordline = self._word_bits // self._value_bits
            self.read_access_energy = (self._read_wordline_access_energy if self._read_wordline_access_energy else self._wordline_access_energy) / self.values_per_wordline
            self.write_access_energy = (self._write_wordline_access_energy if self._write_wordline_access_energy else self._wordline_access_energy) / self.values_per_wordline
        del self._word_bits, self._value_bits, self._wordline_access_energy, self._value_access_energy, self._read_wordline_access_energy, self._write_wordline_access_energy, self._read_value_access_energy, self._write_value_access_energy
        assert self.read_access_energy >= 0 and self.write_access_energy >= 0 and self.leakage_energy >= 0, f"Arch: {arch.name} -> Level: {self.name}: a negative access energy ({self.read_access_energy} read, {self.read_access_energy} write), ({self.leakage_energy} leak), does not mean anything (unless you are into sci-fi stuff)."
        assert not self.area or self.area >= 0, f"Arch: {arch.name} -> Level: {self.name}: a negative area ({self.area}) does not mean anything."
        # NOTE: 1/2 split of bandwidth for consistency with Timeloop - not a true must...
        assert (self._bandwidth and not self.read_bandwidth and not self.write_bandwidth) or (self.read_bandwidth and self.write_bandwidth), f"Arch: {arch.name} -> Level: {self.name}: either bandwidth ({self._bandwidth}) or read_bandwidth ({self.read_bandwidth}) and write_bandwidth ({self.write_bandwidth}) must be specified, if either of read_bandwidth or write_bandwidth is specified, the other must be specified as well."
        self.read_bandwidth = self.read_bandwidth if self.read_bandwidth else self._bandwidth/2
        self.write_bandwidth = self.write_bandwidth if self.write_bandwidth else self._bandwidth/2
        del self._bandwidth
        self.factors = self.factors if self.factors else Factors(arch.coupling.dims)
        self.tile_sizes = self.tile_sizes if self.tile_sizes else Shape({dim: 1 for dim in arch.coupling.dims})
        assert self.read_bandwidth >= 0 and self.write_bandwidth >= 0, f"Arch: {arch.name} -> Level: {self.name}: a negative bandwidth ({self.read_bandwidth} R, {self.write_bandwidth} W) does not mean anything."
        assert all([constr[0] in self.dataflow and constr[1:] in ['', '>=', '<='] for constr in self.factors_constraints.keys()]), f"Arch: {arch.name} -> Level: {self.name}: all keys within factor constraints ({list(self.factors_constraints.keys())}) must be a dimension of the dataflow ({self.dataflow}) and in the form 'dim', 'dim<=', or 'dim>='."
        assert all([sum(constr[0] == dim for constr in self.factors_constraints.keys()) <= 1 for dim in self.dataflow]), f"Arch: {arch.name} -> Level: {self.name}: each dimension must occur at most once in constraints ({list(self.factors_constraints.keys())}), regardless of the use of '>=' or '<='."
        assert all([value > 0 for value in self.factors_constraints.values()]), f"Arch: {arch.name} -> Level: {self.name}: all constraints ({self.factors_constraints}) must have a value strictly > 0."
        assert all([constr in self.dataflow for constr in self.dataflow_constraints]), f"Arch: {arch.name} -> Level: {self.name}: all dims specified as dataflow constraints ({self.dataflow_constraints}) must be part of the dataflow ({self.dataflow})."
        assert self.multiple_buffering >= 1, f"Arch: {arch.name} -> Level: {self.name}: multiple buffering ({self.multiple_buffering}) must be at least 1."
        self.next_is_compute = False
        self.next_spatials = None
        self.next_levels_with_bypass = {'in': None, 'w': None, 'out': None}

    """
    Initializes the bypasses which start from this level.
    Let "levels" be all levels starting from the next one going downward
    up until and including the last one bypassing "operand".

    => This method must be invoked while iterating from outer to inner levels
    as it updates the level's notion of bypassed operations.
    """
    def initBypass(self, operand : str, levels : list[Level]) -> None:
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
    def setMOPs(self, in_reads : int, w_reads : int, out_reads : int, in_writes : int, w_writes : int, out_writes : int) -> None:
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
    def setAboveMOPs(self, last_out_reads : int, last_out_writes : int) -> None:
        self.last_out_reads = last_out_reads
        self.last_out_writes = last_out_writes

    """
    Returns "Fills" as intended for Buffets, thus the incoming writes
    from an higher level.
    """
    def getFill(self) -> int:
        return self.in_writes + self.w_writes + self.last_out_reads #include fills for outputs

    """
    Returns "Drains" as intended for Buffets, thus the outgoing reads
    towards an higher level.
    """
    def getDrain(self) -> int:
        return self.last_out_writes

    """
    Returns "Reads" as intended for Buffets, thus the outgoing reads
    towards a lower level.
    """
    def getRead(self) -> int:
        return self.in_reads + self.w_reads + (self.out_reads - self.last_out_writes) #ignore reads done to drain

    """
    Returns "Updates" as intended for Buffets, thus the incoming writes
    from a lower level.
    """
    def getUpdate(self) -> int:
        return self.out_writes - self.last_out_reads #ignore updates coming from fills

    """
    Returns the total MOPs previoulsy stored by setMOPs().
    """
    def getSettedMOPs(self) -> int:
        return self.in_reads + self.w_reads + self.out_reads + self.in_writes + self.w_writes + self.out_writes
 
    """
    Sets Latency and related statistics for this level.
    """
    def setLatency(self, latency_read_drain : int, latency_fill_update : int, cc_per_tile : int, stall_cycles : int, ideal_bandwidth_read : float, ideal_bandwidth_update : float, ideal_bandwidth_fill : float, ideal_bandwidth_drain : float) -> None:
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
    def getSettedLatency(self) -> int:
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

    Lastly, factors encountered on dimensions orthogonal to the output
    are returned too, to allow the caller to remove one such iteration
    to account for the presence or absence of the bias (bias_read flag).
    
    Arguments:
    - in_bp, w_bp, out_bp: whether the respective operands are bypassed or not,
                           defaults to the instance's setting if not provided.
                           Values are '0' for bypass, '1' for keep.
    - ignore_bypasses: if True, MOPs are returned only relative to this level's
                       accesses from strictly adjacent levels, bypassed operands
                       will thus show 0 MOPs.
    """
    def MOPs(self, in_bp : Optional[bool] = None, w_bp : Optional[bool] = None, out_bp : Optional[bool] = None, ignore_bypasses : bool = False) -> tuple[int, int, int, int, int]:
        in_bp = in_bp if in_bp != None else self.in_bp
        w_bp = w_bp if w_bp != None else self.w_bp
        out_bp = out_bp if out_bp != None else self.out_bp
        # ignore loops at one
        actual_dataflow = list(filter(lambda dim : self.factors.dimProduct(dim) > 1, self.dataflow))
        # stationarity calculation for inputs
        in_reads = in_bp
        if in_reads:
            i = len(actual_dataflow) - 1
            innermost_dim_sum = None
            if not self.next_is_compute:
                skipped = False
                # skip contigous innermost orthogonal dimensions (determining stationarity)
                while i >= 0 and (actual_dataflow[i] not in self.arch.coupling.flat_in_coupling):
                    i -= 1
                    skipped = True
                # dimensions that partake in a sum of indices for the input operand together with the innermost iterated dimension
                innermost_dim_sum = next((dim_sum for dim_sum in self.arch.coupling.in_coupling if actual_dataflow[i] in dim_sum and len(dim_sum) > 1), None) if i >= 0 else None
                # if any dimension in innermost_dim_sum is spatially unrolled after this level, no halo reuse can occur because the instance
                # storing reusable data from the previous iteration differs from the instance which needs that data for the next iteration
                # NOTE: halo reuse, when a sum of indices is involved, after reading the first tile, all subsequent ones are only read for the part not in common
                # with the preceeding tile [here a tile is intended as deriving from the product of sum of tile sizes matching the product of sum in the coupling]
                # TODO: think about how to consider memories with all iterations at 1. Should "next_spatials" cut through them and see fanouts beyond, or stop there as if the were used (this also has repercussions on bypass handling)?
                if innermost_dim_sum and (not all(all(sp_level.factors.dimProduct(dim) == 1 for dim in innermost_dim_sum) for sp_level in self.next_spatials) or (skipped and not self.multiple_reuses)):
                    innermost_dim_sum = None
                # reuse the halo left by the innermost iterated dimension on the iterations on dimensions part of a sum of indices with it
                if innermost_dim_sum:
                    # on the first iteration, you send down the sum of tile sizes for the sum of indices, then at each iteration you fill up only your tile size worth of new elements
                    # WARNING: this ignores potential reuse in the case in which (e.g.) R is iterated immediately around P, and part of the input can be reuse on inner levels!!! That may not be supported by HW tho...
                    #in_reads *= (distinct_values((tile_sizes := [self.tile_sizes[dim] for dim in innermost_dim_sum]), (strides := [self.arch.stride_values[self.arch.coupling.in_strides[dim]] for dim in innermost_dim_sum])) - (overlap := overlapped_values(tile_sizes, strides, innermost_dim_sum.index(actual_dataflow[i]))))*self.factors.dimProduct(actual_dataflow[i]) + overlap
                    in_reads *= distinct_values([self.factors.dimProduct(actual_dataflow[i])*self.tile_sizes[actual_dataflow[i]]] + [self.tile_sizes[dim] for dim in innermost_dim_sum if dim != actual_dataflow[i]], [self.arch.getInStride(actual_dataflow[i])] + [self.arch.getInStride(dim) for dim in innermost_dim_sum if dim != actual_dataflow[i]])
                    i -= 1
            if not self.next_spatials:
                # no spatial level(s) immediately follow;
                # compute the elements per tile - size of the tiles moved between this level and the one below; for summed indices, their tile_sizes must be added (with a -1 per sum), and them multiplied with the others.
                in_reads *= prod(distinct_values([self.tile_sizes[dim] for dim in dim_sum], [self.arch.getInStride(dim) for dim in dim_sum]) for dim_sum in self.arch.coupling.in_coupling if dim_sum is not innermost_dim_sum)
            else:
                # total iterations per dimension in the following consecutive fanout levels
                next_spatial_unroll = {dim: prod(level.factors.dimProduct(dim) for level in self.next_spatials if not level.selective_multicast_support) for dim in self.arch.coupling.dims} # for dim_sum in self.arch.coupling.in_coupling if len(dim_sum) > 1 for dim in dim_sim}
                # elements per tile - however, if immediately following fanouts don't support the multicasting of only certain parts of tiles (selective multicast), shared between instances, each seeing a different step
                # of the moving window, then the window required by each instance needs to be accessed independently, making the final tiles at this level contain duplicate elements for those which can't be multicasted.
                # NOTE: moving window, the access patter deriving from a sum of indices, where the inner iterated one creates a window, which is slid forward by the outer iterated index.
                in_reads *= prod(distinct_values([self.tile_sizes[dim]//next_spatial_unroll[dim] for dim in dim_sum], [self.arch.getInStride(dim) for dim in dim_sum])*prod(next_spatial_unroll[dim] for dim in dim_sum) for dim_sum in self.arch.coupling.in_coupling if dim_sum is not innermost_dim_sum)
            # accumulate the remaining iterations
            for dim in actual_dataflow[:i+1]:
                in_reads *= self.factors.dimProduct(dim)
        # stationarity calculation for weights
        w_reads = w_bp
        if w_reads:
            i = len(actual_dataflow) - 1
            innermost_dim_sum = None
            if not self.next_is_compute:
                skipped = False
                while i >= 0 and (actual_dataflow[i] not in self.arch.coupling.flat_w_coupling):
                    i -= 1
                    skipped = True
                innermost_dim_sum = next((dim_sum for dim_sum in self.arch.coupling.w_coupling if actual_dataflow[i] in dim_sum and len(dim_sum) > 1), None) if i >= 0 else None
                if innermost_dim_sum and (not all(all(sp_level.factors.dimProduct(dim) == 1 for dim in innermost_dim_sum) for sp_level in self.next_spatials) or (skipped and not self.multiple_reuses)):
                    innermost_dim_sum = None
                if innermost_dim_sum:
                    w_reads *= distinct_values([self.factors.dimProduct(actual_dataflow[i])*self.tile_sizes[actual_dataflow[i]]] + [self.tile_sizes[dim] for dim in innermost_dim_sum if dim != actual_dataflow[i]], [self.arch.getWStride(actual_dataflow[i])] + [self.arch.getWStride(dim) for dim in innermost_dim_sum if dim != actual_dataflow[i]])
                    i -= 1
            if not self.next_spatials:
                w_reads *= prod(distinct_values([self.tile_sizes[dim] for dim in dim_sum], [self.arch.getWStride(dim) for dim in dim_sum]) for dim_sum in self.arch.coupling.w_coupling if dim_sum is not innermost_dim_sum)
            else:
                next_spatial_unroll = {dim: prod(level.factors.dimProduct(dim) for level in self.next_spatials if not level.selective_multicast_support) for dim in self.arch.coupling.dims}
                w_reads *= prod(distinct_values([self.tile_sizes[dim]//next_spatial_unroll[dim] for dim in dim_sum], [self.arch.getWStride(dim) for dim in dim_sum])*prod(next_spatial_unroll[dim] for dim in dim_sum) for dim_sum in self.arch.coupling.w_coupling if dim_sum is not innermost_dim_sum)
            for dim in actual_dataflow[:i+1]:
                w_reads *= self.factors.dimProduct(dim)
        # stationarity calculation for outputs
        out_reads = out_bp
        out_reads_factors = out_bp # this collects the factors along dimensions orthogonal to the output, returned to handle the presence/absence of the bias
        if out_reads:
            i = len(actual_dataflow) - 1
            innermost_dim_sum = None
            if not self.next_is_compute:
                skipped = False
                while i >= 0 and (actual_dataflow[i] not in self.arch.coupling.flat_out_coupling):
                    i -= 1
                    skipped = True
                innermost_dim_sum = next((dim_sum for dim_sum in self.arch.coupling.out_coupling if actual_dataflow[i] in dim_sum and len(dim_sum) > 1), None) if i >= 0 else None
                if innermost_dim_sum and (not all(all(sp_level.factors.dimProduct(dim) == 1 for dim in innermost_dim_sum) for sp_level in self.next_spatials) or (skipped and not self.multiple_reuses)):
                    innermost_dim_sum = None
                if innermost_dim_sum:
                    out_reads *= distinct_values([self.factors.dimProduct(actual_dataflow[i])*self.tile_sizes[actual_dataflow[i]]] + [self.tile_sizes[dim] for dim in innermost_dim_sum if dim != actual_dataflow[i]], [self.arch.getOutStride(actual_dataflow[i])] + [self.arch.getOutStride(dim) for dim in innermost_dim_sum if dim != actual_dataflow[i]])
                    i -= 1
            if not self.next_spatials:
                out_reads *= prod(distinct_values([self.tile_sizes[dim] for dim in dim_sum], [self.arch.getOutStride(dim) for dim in dim_sum]) for dim_sum in self.arch.coupling.out_coupling if dim_sum is not innermost_dim_sum)
            else:
                next_spatial_unroll = {dim: prod(level.factors.dimProduct(dim) for level in self.next_spatials if not level.selective_multicast_support) for dim in self.arch.coupling.dims}
                out_reads *= prod(distinct_values([self.tile_sizes[dim]//next_spatial_unroll[dim] for dim in dim_sum], [self.arch.getOutStride(dim) for dim in dim_sum])*prod(next_spatial_unroll[dim] for dim in dim_sum) for dim_sum in self.arch.coupling.out_coupling if dim_sum is not innermost_dim_sum)
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
                    # 'level' is the level immediately above the one that stores the bypassed operand after the self level
                    in_between, level = levels[:-1], levels[-1]
                    in_reads_bp, w_reads_bp, out_reads_bp, out_writes_bp, out_reads_bp_factors = level.MOPs(operand == 'in', operand == 'w', operand == 'out', True)
                    # bulid the inner-most dataflow, by piling one against the other all non-1 loops, then look at which is the innermost dimension, that is the one that matters!
                    # TL;DR: consider the dataflow only w.r.t. the innermost loop, ignoring those with 1 iteration!
                    stationarity_to_address = not (any(level.factors.dimProduct(dim) > 1 and dim in self.arch.coupling.flatCouplingByOperand(operand) for dim in level.dataflow) if self.multiple_reuses else any(level.factors.dimProduct(dim) > 1 for dim in level.dataflow))
                    for in_btwn in in_between[::-1]:
                        if isinstance(in_btwn, MemLevel):
                            # ignore loops at one
                            actual_dataflow_bp = list(filter(lambda dim : in_btwn.factors.dimProduct(dim) > 1, in_btwn.dataflow))
                            in_btwn_factors_full = in_btwn.factors.fullProduct()
                            if in_reads_bp:
                                if stationarity_to_address:
                                    # all inner loops were 1s or orthogonal, deal with the dataflow now!
                                    i = len(actual_dataflow_bp) - 1
                                    while i >= 0 and (actual_dataflow_bp[i] not in self.arch.coupling.flat_in_coupling):
                                        i -= 1
                                    stationarity_to_address = i < 0 # postpone stationarity evaluation since not input-coupled dimension is iterated here
                                    # dimensions that partake in a sum of indices for the input operand together with the innermost iterated dimension on the currently traversed level between bypasses
                                    innermost_dim_sum = next((dim_sum for dim_sum in self.arch.coupling.in_coupling if actual_dataflow_bp[i] in dim_sum and len(dim_sum) > 1), None) if i >= 0 else None
                                    if innermost_dim_sum and (all(all(sp_level.factors.dimProduct(dim) == 1 for dim in innermost_dim_sum) for sp_level in in_btwn.next_spatials) and (i == len(actual_dataflow_bp) - 1 or self.multiple_reuses)):
                                        # before updating the tile size, remove from the original one the component relative to the dimension involved in the sum of indices
                                        # WARNING: this ignores potential reuse in the case in which (e.g.) R is iterated immediately around P, and part of the input can be reuse on inner levels!!! That may not be supported by HW tho...
                                        in_reads_bp //= distinct_values([in_btwn.tile_sizes[dim] for dim in innermost_dim_sum], [self.arch.getInStride(dim) for dim in innermost_dim_sum])
                                        in_reads_bp *= distinct_values([in_btwn.factors.dimProduct(actual_dataflow_bp[i])*in_btwn.tile_sizes[actual_dataflow_bp[i]]] + [in_btwn.tile_sizes[dim] for dim in innermost_dim_sum if dim != actual_dataflow_bp[i]], [self.arch.getInStride(actual_dataflow_bp[i])] + [self.arch.getInStride(dim) for dim in innermost_dim_sum if dim != actual_dataflow_bp[i]])
                                        i -= 1
                                    for dim in actual_dataflow_bp[:i+1]:
                                        in_reads_bp *= in_btwn.factors.dimProduct(dim)
                                else:
                                    # dataflow already handled among inner loops
                                    in_reads_bp = in_btwn_factors_full*in_reads_bp
                            if w_reads_bp:
                                if stationarity_to_address:
                                    i = len(actual_dataflow_bp) - 1
                                    while i >= 0 and (actual_dataflow_bp[i] not in self.arch.coupling.flat_w_coupling):
                                        i -= 1
                                    stationarity_to_address = i < 0
                                    innermost_dim_sum = next((dim_sum for dim_sum in self.arch.coupling.w_coupling if actual_dataflow_bp[i] in dim_sum and len(dim_sum) > 1), None) if i >= 0 else None
                                    if innermost_dim_sum and (all(all(sp_level.factors.dimProduct(dim) == 1 for dim in innermost_dim_sum) for sp_level in in_btwn.next_spatials) and (i == len(actual_dataflow_bp) - 1 or self.multiple_reuses)):
                                        w_reads_bp //= distinct_values([in_btwn.tile_sizes[dim] for dim in innermost_dim_sum], [self.arch.getWStride(dim) for dim in innermost_dim_sum])
                                        w_reads_bp *= distinct_values([in_btwn.factors.dimProduct(actual_dataflow_bp[i])*in_btwn.tile_sizes[actual_dataflow_bp[i]]] + [in_btwn.tile_sizes[dim] for dim in innermost_dim_sum if dim != actual_dataflow_bp[i]], [self.arch.getWStride(actual_dataflow_bp[i])] + [self.arch.getWStride(dim) for dim in innermost_dim_sum if dim != actual_dataflow_bp[i]])
                                        i -= 1
                                    for dim in actual_dataflow_bp[:i+1]:
                                        w_reads_bp *= in_btwn.factors.dimProduct(dim)
                                else:
                                    w_reads_bp = in_btwn_factors_full*w_reads_bp
                            if out_reads_bp:
                                if stationarity_to_address:
                                    i = len(actual_dataflow_bp) - 1
                                    while i >= 0 and (actual_dataflow_bp[i] not in self.arch.coupling.flat_out_coupling):
                                        i -= 1
                                    stationarity_to_address = i < 0
                                    innermost_dim_sum = next((dim_sum for dim_sum in self.arch.coupling.out_coupling if actual_dataflow_bp[i] in dim_sum and len(dim_sum) > 1), None) if i >= 0 else None
                                    if innermost_dim_sum and (all(all(sp_level.factors.dimProduct(dim) == 1 for dim in innermost_dim_sum) for sp_level in in_btwn.next_spatials) and (i == len(actual_dataflow_bp) - 1 or self.multiple_reuses)):
                                        out_reads_bp, out_writes_bp = out_reads_bp//(dvs := distinct_values([in_btwn.tile_sizes[dim] for dim in innermost_dim_sum], [self.arch.getOutStride(dim) for dim in innermost_dim_sum])), out_writes_bp//dvs
                                        out_reads_bp, out_writes_bp = out_reads_bp*(dvs := distinct_values([in_btwn.factors.dimProduct(actual_dataflow_bp[i])*in_btwn.tile_sizes[actual_dataflow_bp[i]]] + [in_btwn.tile_sizes[dim] for dim in innermost_dim_sum if dim != actual_dataflow_bp[i]], [self.arch.getOutStride(actual_dataflow_bp[i])] + [self.arch.getOutStride(dim) for dim in innermost_dim_sum if dim != actual_dataflow_bp[i]])), out_writes_bp*dvs
                                        i -= 1
                                    for dim in actual_dataflow_bp[:i+1]:
                                        out_reads_bp *= in_btwn.factors.dimProduct(dim)
                                        out_writes_bp *= in_btwn.factors.dimProduct(dim)
                                        if dim not in self.arch.coupling.flat_out_coupling:
                                            out_reads_bp_factors *= in_btwn.factors.dimProduct(dim)
                                else:
                                    out_reads_bp = in_btwn_factors_full*out_reads_bp
                                    out_writes_bp = in_btwn_factors_full*out_writes_bp
                                    out_reads_bp_factors *= prod(in_btwn.factors.dimProduct(dim) for dim in in_btwn.dataflow if dim not in self.arch.coupling.flat_out_coupling)
                        else:
                            in_reads_bp, w_reads_bp, out_reads_bp, out_writes_bp = in_btwn.mulByDim(in_reads_bp, w_reads_bp, out_reads_bp, out_writes_bp)
                            # do not update out_reads_bp_factors here, because in it go only iterations of which the first one is skipped,
                            # while in a fanout all fanned-out copies of the inner loop behave the same, there isn't a first different spatial iteration or anything
                        #print("IN BETWEEN BYPASS:\n", f"{in_btwn.name}:{chr(9) * (2 - len(in_btwn.name)//8)}{in_reads_bp} In_R, {w_reads_bp} W_R, {out_reads_bp} Our_R, {in_reads_bp + w_reads_bp + out_reads_bp} Tot_R, {out_writes_bp} Out_W, {out_reads_bp_factors} Out_R_Fac")
                    factors_full = self.factors.fullProduct()
                    if in_reads_bp:
                        if stationarity_to_address:
                            # all inner loops were 1s or orthogonal, deal with the dataflow now!
                            i = len(actual_dataflow) - 1
                            while i >= 0 and (actual_dataflow[i] not in self.arch.coupling.flat_in_coupling):
                                i -= 1
                            innermost_dim_sum = next((dim_sum for dim_sum in self.arch.coupling.in_coupling if actual_dataflow[i] in dim_sum and len(dim_sum) > 1), None) if i >= 0 else None
                            if innermost_dim_sum and (all(all(sp_level.factors.dimProduct(dim) == 1 for dim in innermost_dim_sum) for sp_level in self.next_spatials) and (i == len(actual_dataflow) - 1 or self.multiple_reuses)):
                                in_reads_bp //= distinct_values([self.tile_sizes[dim] for dim in innermost_dim_sum], [self.arch.getInStride(dim) for dim in innermost_dim_sum])
                                in_reads_bp *= distinct_values([self.factors.dimProduct(actual_dataflow[i])*self.tile_sizes[actual_dataflow[i]]] + [self.tile_sizes[dim] for dim in innermost_dim_sum if dim != actual_dataflow[i]], [self.arch.getInStride(actual_dataflow[i])] + [self.arch.getInStride(dim) for dim in innermost_dim_sum if dim != actual_dataflow[i]])
                                i -= 1
                            for dim in actual_dataflow[:i+1]:
                                in_reads_bp *= self.factors.dimProduct(dim)
                        else:
                            # dataflow handled among inner loops
                            in_reads_bp = factors_full*in_reads_bp
                    if w_reads_bp:
                        if stationarity_to_address:
                            i = len(actual_dataflow) - 1
                            while i >= 0 and (actual_dataflow[i] not in self.arch.coupling.flat_w_coupling):
                                i -= 1
                            innermost_dim_sum = next((dim_sum for dim_sum in self.arch.coupling.w_coupling if actual_dataflow[i] in dim_sum and len(dim_sum) > 1), None) if i >= 0 else None
                            if innermost_dim_sum and (all(all(sp_level.factors.dimProduct(dim) == 1 for dim in innermost_dim_sum) for sp_level in self.next_spatials) and (i == len(actual_dataflow) - 1 or self.multiple_reuses)):
                                w_reads_bp //= distinct_values([self.tile_sizes[dim] for dim in innermost_dim_sum], [self.arch.getWStride(dim) for dim in innermost_dim_sum])
                                w_reads_bp *= distinct_values([self.factors.dimProduct(actual_dataflow[i])*self.tile_sizes[actual_dataflow[i]]] + [self.tile_sizes[dim] for dim in innermost_dim_sum if dim != actual_dataflow[i]], [self.arch.getWStride(actual_dataflow[i])] + [self.arch.getWStride(dim) for dim in innermost_dim_sum if dim != actual_dataflow[i]])
                                i -= 1
                            for dim in actual_dataflow[:i+1]:
                                w_reads_bp *= self.factors.dimProduct(dim)
                        else:
                            w_reads_bp = factors_full*w_reads_bp
                    if out_reads_bp:
                        if stationarity_to_address:
                            i = len(actual_dataflow) - 1
                            while i >= 0 and (actual_dataflow[i] not in self.arch.coupling.flat_out_coupling):
                                i -= 1
                            innermost_dim_sum = next((dim_sum for dim_sum in self.arch.coupling.out_coupling if actual_dataflow[i] in dim_sum and len(dim_sum) > 1), None) if i >= 0 else None
                            if innermost_dim_sum and (all(all(sp_level.factors.dimProduct(dim) == 1 for dim in innermost_dim_sum) for sp_level in self.next_spatials) and (i == len(actual_dataflow) - 1 or self.multiple_reuses)):
                                out_reads_bp, out_writes_bp = out_reads_bp//(dvs := distinct_values([self.tile_sizes[dim] for dim in innermost_dim_sum], [self.arch.getOutStride(dim) for dim in innermost_dim_sum])), out_writes_bp//dvs
                                out_reads_bp, out_writes_bp = out_reads_bp*(dvs := distinct_values([self.factors.dimProduct(actual_dataflow[i])*self.tile_sizes[actual_dataflow[i]]] + [self.tile_sizes[dim] for dim in innermost_dim_sum if dim != actual_dataflow[i]], [self.arch.getOutStride(actual_dataflow[i])] + [self.arch.getOutStride(dim) for dim in innermost_dim_sum if dim != actual_dataflow[i]])), out_writes_bp*dvs
                                i -= 1
                            for dim in actual_dataflow[:i+1]:
                                out_reads_bp *= self.factors.dimProduct(dim)
                                out_writes_bp *= self.factors.dimProduct(dim)
                                if dim not in self.arch.coupling.flat_out_coupling:
                                    out_reads_bp_factors *= self.factors.dimProduct(dim)
                        else:
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
    def WMOPs(self, reads : Optional[int] = None, writes : Optional[int] = None) -> float:
        if not (reads and writes):
            reads = reads if reads else self.in_reads + self.w_reads + self.out_reads
            writes = writes if writes else self.in_writes + self.w_writes + self.out_writes
        return self.read_access_energy * reads + self.write_access_energy * writes

    """
    Returns the total leaked energy during the provided clock cycles.
    """
    def Leakage(self, cycles : int) -> float:
        return cycles*self.leakage_energy

    """
    Returns True iif factors present on this level satisfy all of
    its constraints, including fitting in the available memory.
    """
    def checkConstraints(self) -> bool:
        return self.factors.memFootprint(self.tile_sizes, self.arch.coupling, not self.bypasses or 'in' not in self.bypasses, not self.bypasses or 'w' not in self.bypasses, not self.bypasses or 'out' not in self.bypasses) <= self.size/self.multiple_buffering and super().checkConstraints()

    """
    Returns a string describing the current violation of constraints, if any.
    """
    def logConstraintsViolation(self) -> str:
        if not super().checkConstraints():
            return super().logConstraintsViolation()
        elif not self.checkConstraints():
            mem_footprint = self.factors.memFootprint(self.tile_sizes, self.arch.coupling, not self.bypasses or 'in' not in self.bypasses, not self.bypasses or 'w' not in self.bypasses, not self.bypasses or 'out' not in self.bypasses)
            return f"CONSTRAINTS VIOLATION: Arch: {self.arch.name} -> Level: {self.name}: memory used: {mem_footprint} VS memory available: {self.size/self.multiple_buffering:.0f}"
        return ""

    def __str__(self) -> str:
        return f"{super().__str__()}, size: {self.size}, read_access_energy: {self.read_access_energy}, write_access_energy: {self.write_access_energy}, leakage_energy: {self.leakage_energy}, read_bandwidth: {self.read_bandwidth}, write_bandwidth: {self.write_bandwidth}, bypasses: {self.bypasses}, multiple_buffering: {self.multiple_buffering}"


"""
Abstract class for a level introducing multiple spatial instances.
"""
class SpatialLevel(Level):
    dims : list[str]
    mesh : int
    spatial_multicast_support : bool
    spatial_reduction_support : bool
    selective_multicast_support : bool
    selective_reduction_support : bool


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
- spatial_multicast_support: True (default) if this level supports spatial multicast.
        By itself, this only allows the same identical tile of an operand to be sent
        to each instance at once identically.
- spatial_reduction_support: same as 'spatial_multicast_support' but for spatial reduction.
- selective_multicast_support: enables sending parts of a read tile selectively to one
        or more different instances. In other words, if enabled, all values required
        by all instances are read once, with each instance receiving only the part of its
        interest. When disabled, each instance [involved in the spatial unrolling of a
        dimension present in a sum of indices] issues its own independent read for the
        values it needs. Obviously, this is useful only when fanouts unfold spatially
        dimensions involved in a sum of indices (e.g. those of the input tensor of a
        convolution), it does not make a difference otherwise (e.g. on GEMMs).
        Can only be True if 'spatial_multicast_support' is already True. Default is False.
- selective_reduction_support: same as 'selective_multicast_support' but for reduction.
        Can only be True if 'spatial_reduction_support' is already True. Default is False.
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
    pe_to_pe : bool # True in all cases where the operand independent of "dim" (e.g.: in a GEMM, if dim = M, such operand is the input) is forwarded pe->pe rather than multicasted
    power_gating_support : bool

    def __init__(self, name : str, mesh : int, dim : Optional[str] = None, dims : Optional[list[str]] = None, area : Optional[float] = None, pe_to_pe : bool = False, spatial_multicast_support : bool = True, spatial_reduction_support : bool = True, selective_multicast_support : bool = False, selective_reduction_support : bool = False, power_gating_support : bool = False, factors : Optional[Factors] = None, tile_sizes : Optional[Shape] = None, factors_constraints : Optional[dict[str, int]] = None):
        self.name = name
        self._dim = dim
        self.dims =  dims
        self.mesh = mesh
        self.area = area
        self.pe_to_pe = pe_to_pe
        self.spatial_multicast_support = spatial_multicast_support
        self.spatial_reduction_support = spatial_reduction_support
        self.selective_multicast_support = selective_multicast_support
        self.selective_reduction_support = selective_reduction_support
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
        assert (self._dim and not self.dims) or (self.dims and not self._dim), f"Arch: {arch.name} -> Level: {self.name}: exactly one of dim ({self._dim}) or dims ({self.dims}) must be specified."
        self.dims = [self._dim] if self._dim else self.dims
        del self._dim
        self.dataflow = self.dims
        assert all([dim in arch.coupling.dims for dim in self.dataflow]), f"Arch: {arch.name} -> Level: {self.name}: accepted names for dimensions are solely M, K and N, provided ones were {self.dataflow}."
        assert self.mesh > 0, f"Arch: {arch.name} -> Level: {self.name}: a spatial fanout must have a mesh ({self.mesh}) of at least 1."
        assert not self.area or self.area >= 0, f"Arch: {arch.name} -> Level: {self.name}: a negative area ({self.area}) does not mean anything."
        assert not self.pe_to_pe or (self.spatial_multicast_support and self.spatial_reduction_support), f"Arch: {arch.name} -> Level: {self.name}: pe-to-pe ({self.pe_to_pe}) forwarding is a form of spatial multicast ({self.spatial_multicast_support}) or reduction ({self.spatial_reduction_support}), which must then both be supported to enable it."
        assert not self.selective_multicast_support or self.spatial_multicast_support, f"Arch: {arch.name} -> Level: {self.name}: selective multicast ({self.selective_multicast_support}) is a form of spatial multicast ({self.spatial_multicast_support}), which must then be supported to enable it."
        assert not self.selective_reduction_support or self.spatial_reduction_support, f"Arch: {arch.name} -> Level: {self.name}: selective reduction ({self.selective_reduction_support}) is a form of spatial reduction ({self.spatial_reduction_support}), which must then be supported to enable it."
        self.factors = self.factors if self.factors else Factors(arch.coupling.dims)
        self.tile_sizes = self.tile_sizes if self.tile_sizes else Shape({dim: 1 for dim in arch.coupling.dims})
        assert all([constr[0] in self.dataflow and constr[1:] in ['', '>=', '<='] for constr in self.factors_constraints.keys()]), f"Arch: {arch.name} -> Level: {self.name}: all keys within factor constraints ({list(self.factors_constraints.keys())}) must be a dimension of the dataflow ({self.dataflow}) and in the form 'dim', 'dim<=', or 'dim>='."
        assert all([sum(constr[0] == dim for constr in self.factors_constraints.keys()) <= 1 for dim in self.dataflow]), f"Arch: {arch.name} -> Level: {self.name}: each dimension must occur at most once in constraints ({list(self.factors_constraints.keys())}), regardless of the use of '>=' or '<='."
        assert all([value > 0 for value in self.factors_constraints.values()]), f"Arch: {arch.name} -> Level: {self.name}: all constraints ({self.factors_constraints}) must have a value strictly > 0."

    """
    Let inputs be the amount of operations occuring on a level below this fanout,
    this method returns the amount of operations seen from above the fanout,
    accounting for spatial multicast and spatial reduction support for operands.
    """
    # here you receive the reads/writes done by an instance, and need to
    # return the reads/writes that are needed for all instances.
    def mulByDim(self, in_reads : int, w_reads : int, out_reads : int, out_writes : int) -> tuple[int, int, int, int]:
        if self.selective_multicast_support:
            for dim_sum in self.arch.coupling.in_coupling:
                if len(dim_sum) > 1: #TODO: could optimize by skipping this whole if and its else if all dimProduct(dim)-s are 1!
                    strides = [self.arch.getInStride(dim) for dim in dim_sum]
                    in_reads //= distinct_values([self.tile_sizes[dim] for dim in dim_sum], strides)
                    in_reads *= distinct_values([self.factors.dimProduct(dim)*self.tile_sizes[dim] for dim in dim_sum], strides)
                else:
                    in_reads *= self.factors.dimProduct(dim_sum[0])
            for dim_sum in self.arch.coupling.w_coupling:
                if len(dim_sum) > 1:
                    strides = [self.arch.getWStride(dim) for dim in dim_sum]
                    w_reads //= distinct_values([self.tile_sizes[dim] for dim in dim_sum], strides)
                    w_reads *= distinct_values([self.factors.dimProduct(dim)*self.tile_sizes[dim] for dim in dim_sum], strides)
                else:
                    w_reads *= self.factors.dimProduct(dim_sum[0])
            for dim_sum in self.arch.coupling.out_coupling:
                if len(dim_sum) > 1:
                    strides = [self.arch.getOutStride(dim) for dim in dim_sum]
                    out_reads //= distinct_values([self.tile_sizes[dim] for dim in dim_sum], strides)
                    out_reads *= distinct_values([self.factors.dimProduct(dim)*self.tile_sizes[dim] for dim in dim_sum], strides)
                else:
                    out_reads *= self.factors.dimProduct(dim_sum[0])
        else:
            in_reads *= prod(self.factors.dimProduct(dim) for dim in self.arch.coupling.flat_in_coupling)
            w_reads *= prod(self.factors.dimProduct(dim) for dim in self.arch.coupling.flat_w_coupling)
            out_reads *= prod(self.factors.dimProduct(dim) for dim in self.arch.coupling.flat_out_coupling)
        if self.selective_reduction_support:
            for dim_sum in self.arch.coupling.out_coupling:
                if len(dim_sum) > 1:
                    strides = [self.arch.getOutStride(dim) for dim in dim_sum]
                    out_writes //= distinct_values([self.tile_sizes[dim] for dim in dim_sum], strides)
                    out_writes *= distinct_values([self.factors.dimProduct(dim)*self.tile_sizes[dim] for dim in dim_sum], strides)
                else:
                    out_writes *= self.factors.dimProduct(dim_sum[0])
        else:
            out_writes *= prod(self.factors.dimProduct(dim) for dim in self.arch.coupling.flat_out_coupling)
        
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
    def latency(self) -> int:
        return 0 # change this if we model the network's latency

    """
    Returns True iif factors present on this level satisfy all of its constraints,
    including not exceeding the physical mesh.
    """
    def checkConstraints(self) -> bool:
        return self.factors.fullProduct() <= self.mesh and super().checkConstraints()

    """
    Returns a string describing the current violation of constraints, if any.
    """
    def logConstraintsViolation(self) -> str:
        if not super().checkConstraints():
            return super().logConstraintsViolation()
        elif not self.checkConstraints():
            return f"CONSTRAINTS VIOLATION: Arch: {self.arch.name} -> Level: {self.name}: spatial iterations used: {self.factors.fullProduct()} VS available instances (mesh): {self.mesh}"
        return ""

    def __str__(self) -> str:
        return f"{super().__str__()}, mesh: {self.mesh}, pe_to_pe: {self.pe_to_pe}, spatial_multicast_support: {self.spatial_multicast_support}, spatial_reduction_support: {self.spatial_reduction_support}, power_gating_support: {self.power_gating_support}"


"""
A Compute Level within the architecture, it is a placeholder for any processing
element (PE) capable of multiply and accumulate (MAC).

NOTE: iterations at this level are equivalent to requiring that a PE can, in the
clock-cycles specified in "cycles", execute all such iterations. It is also assumed
that the data required for all iterations done at this level is held within hardware
registers whose energy cost is modeled by the "compute energy", together with the
actual compute cost. As such, iterations done here are equivalent to a PE capable
of multiple concurrent (parallel or pipelined (chaining accumulation output)) MACs.
Then intuitively, increasing the number of iterations done at this level linearly
increases the required bandwidth of all memory levels feeding it, as they need to
keep up with the concurrent MACs done here.

NOTE: no value is kept stationary on a ComputeLevel, the values for each MAC
operation are read/written every time. PE-level stationarity is realized by the
first Memory Level above compute.
NOTE: values that partake in multiple MAC operations performed concurrently in the
iterations of the ComputeLevel are read/written once per MAC.
TODO: move the multicast/reduction and selective_multicast/selective_reduction
to the SpatialLevel and make them available on ComputeLevel too. Then make the
above NOTEs conditional on the True or False of those properties too! These would
take effect during MemLevel.MOPs, just like they do for FanoutLevel, since also
the ComputeLevel is included in MemLevel.next_spatials!

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
    compute_energy : float
    leakage_energy : float
    cycles : int # clock cycles used per element in the inner dimension (latency of one MAC)

    # STATISTICS:
    instances = 1 # this are the used/active instances
    temporal_iterations = 0
    
    def __init__(self, name : str, mesh : int, compute_energy : float, cycles : int, dim : Optional[str] = None, dims : Optional[list[str]] = None, leakage_energy : float = 0, area : Optional[float] = None, factors : Optional[Factors] = None, tile_sizes : Optional[Shape] = None, factors_constraints : Optional[dict[str, int]] = None):
        self.name = name
        self._dim = dim
        self.dims = dims
        self.mesh = mesh # for a systolic array, this is the length of the operand buffers
        self.compute_energy = compute_energy
        self.leakage_energy = leakage_energy
        self.area = area
        self.spatial_multicast_support = False
        self.spatial_reduction_support = False
        self.selective_multicast_support = False
        self.selective_reduction_support = False
        self.cycles = cycles # clock cycles used per element in the inner dimension (latency of one MAC)
        self.factors = factors
        self.tile_sizes = tile_sizes
        self.factors_constraints = factors_constraints if factors_constraints else {}

    """
    Sets up a pointer back to the whole architecture.
    Ultimates the initialization of the level and validates its attributes.
    """
    def initArch(self, arch : Arch):
        self.arch = arch
        assert self.mesh == 1 or (self._dim and not self.dims) or (self.dims and not self._dim), f"Arch: {arch.name} -> Level: {self.name}: when mesh ({self.mesh}) is > 1, exactly one of dim ({self._dim}) or dims ({self.dims}) must be specified."
        self.dims = ([self._dim] if self._dim else self.dims) if self._dim or self.dims else []
        del self._dim
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
    def latency(self) -> int:
        return self.cycles

    """
    Returns the energy required by this level to perform all MACs across its internal
    iterations, times "iterations", which represents the number of iterations done by
    the hierarchy of levels on top of this one.
    """
    def computeCost(self, iterations : int = 1) -> float:
        return self.compute_energy*self.factors.fullProduct()*iterations

    """
    Returns the total leaked energy during the provided clock cycles.
    """
    def Leakage(self, cycles : int) -> float:
        return cycles*self.leakage_energy

    """
    Returns True iif factors present on this level satisfy all of its constraints,
    including not exceeding the phisical number of concurrent MACs performed here.
    """
    def checkConstraints(self) -> bool:
        return self.factors.fullProduct() <= self.mesh and super().checkConstraints()

    """
    Returns a string describing the current violation of constraints, if any.
    """
    def logConstraintsViolation(self) -> str:
        if not super().checkConstraints():
            return super().logConstraintsViolation()
        elif not self.checkConstraints():
            return f"CONSTRAINTS VIOLATION: Arch: {self.arch.name} -> Level: {self.name}: concurrent MACs used: {self.factors.fullProduct()} VS concurrent MACs available: {self.mesh}"
        return ""

    def __str__(self) -> str:
        return f"{super().__str__()}, mesh: {self.mesh}, compute_energy: {self.compute_energy}, leakage_energy: {self.leakage_energy}, cycles: {self.cycles}"