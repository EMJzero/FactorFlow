from typing import Optional, Any

from factors import *

"""
Class with the minimal information characterizing a level's mapping.
"""
class LevelCore:
    # IMPORTANT:
    # Use the entries of "dataflow" to access "factors", so that you only
    # see the factors over which you are actually supposed to iterate!
    dataflow : list[str] = None # order of the loops | e.g. ['M', 'K', 'N']
    # NOTE: tile sizes are updated in "moveFactor".
    factors : Factors = None # iterations done for the dimensions at this lever
    tile_sizes : Shape = None # indicate the size of a tile used in the level BELOW
    
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
    name : str = None
    factors_constraints : dict[str, int] = None
    area : float = None

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
    Returns the dataflow class for this level, one of:
    - Weight Stationary (WS)
    - Output Stationary (OS)
    - Input Stationary (IS)
    """
    def actualDataflow(self) -> Optional[Dataflow]:
        innermost = [f for f in self.dataflow if self.factors.dimProduct(f) > 1]
        if len(innermost) == 0:
            return None
        elif innermost[-1] == 'N':
            return Dataflow.WS
        elif innermost[-1] == 'K':
            return Dataflow.OS
        elif innermost[-1] == 'M':
            return Dataflow.IS
        else:
            raise Exception(f"Unrecognized Dataflow in level {self.name} with dimension {innermost[-1]} != M, K, or N!")

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
            return (f"CONSTRAINTS VIOLATION: level {self.name}, "
                + ', '.join([f"constrained {dim} == {self.factors_constraints[dim]} VS obtained {dim}: {self.factors.dimProduct(dim)}, " for dim in self.dataflow if (dim in self.factors_constraints and self.factors_constraints[dim] != self.factors.dimProduct(dim))])
                + ', '.join([f"constrained {dim} <= {self.factors_constraints[dim + '<=']} VS obtained {dim}: {self.factors.dimProduct(dim)}, " for dim in self.dataflow if (dim + '<=' in self.factors_constraints and self.factors_constraints[dim + '<='] >= self.factors.dimProduct(dim))])
                + ', '.join([f"constrained {dim} >= {self.factors_constraints[dim + '>=']} VS obtained {dim}: {self.factors.dimProduct(dim)}, " for dim in self.dataflow if (dim + '>=' in self.factors_constraints and self.factors_constraints[dim + '>='] <= self.factors.dimProduct(dim))]))
        return ""

    def __getitem__(self, key : str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key : str, value : Any):
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
                      Valid dictionary keys are:
                          - 'M', 'K', and 'N' for exact values;
                          - 'M<=', 'K<=', and 'N<=' for upper bounds;
                          - 'M>=', 'K>=', and 'N>=' for lower bounds;
                      NOTE: the use of the '<=' and '>=' constraints does not shorten the
                            mapper's runtime as much as exact constraints.
- dataflow_constraints: constraints for the order of loops at this level, for any dim
                        not specified here, all permutations are tried while keeping
                        fixed the relative order of constrained dimensions.
                        Valid strings are 'M', 'K', and 'N'.
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
    def __init__(self, name : str, size : int, value_access_energy : Optional[float] = None, wordline_access_energy : Optional[float] = None, word_bits : Optional[int] = None, value_bits : Optional[int] = None, leakage_energy : float = 0, area : Optional[float] = None, bandwidth : Optional[float] = None, dataflow : Optional[list[str]] = None, factors : Optional[Factors] = None, tile_sizes : Optional[Shape] = None, factors_constraints : Optional[dict[str, int]] = None, dataflow_constraints : Optional[list[str]] = None, bypasses : Optional[list[str]] = None, multiple_buffering : int = 1,  read_value_access_energy : Optional[float] = None, write_value_access_energy : Optional[float] = None, read_wordline_access_energy : Optional[float] = None, write_wordline_access_energy : Optional[float] = None, read_bandwidth : Optional[float] = None, write_bandwidth : Optional[float] = None):
        self.name :str = name
        # NOTE: this way of constructing the dataflow from the constraints is redundant, but useful if one wants to skip the
        # exploration of permutations since with this method the dataflow will be immediately consistent with constraints.
        self.dataflow : list[str] = dataflow if dataflow else (dataflow_constraints + [dim for dim in ['M', 'K', 'N'] if dim not in dataflow_constraints] if dataflow_constraints else ['M', 'K', 'N']) # dimensions over which to iterate
        assert all([dim in ['M', 'K', 'N'] for dim in self.dataflow]), f"Level: {name}: accepted dimension names in the dataflow are solely M, K and N, provided ones were {self.dataflow}."
        assert size >= 0, f"Level: {name}: a negative size ({size}) does not mean anything."
        self.size : int = size
        # read_access_energy and write_access_energy are intended always for one value, remember to bring accessed values to a multiple of values_per_wordline for the correct total energy
        assert (value_access_energy or (read_value_access_energy and write_value_access_energy)) or (word_bits and value_bits and (wordline_access_energy or (read_wordline_access_energy and write_wordline_access_energy))), f"Level: {name}: either value_access_energy ({value_access_energy}) or read_value_access_energy ({read_value_access_energy}) and write_value_access_energy ({write_value_access_energy}) must be specified, alternatively, you can specify word_bits ({word_bits}) and value_bits ({value_bits}) and either wordline_access_energy ({wordline_access_energy}) or read_wordline_access_energy ({read_wordline_access_energy}) and write_wordline_access_energy ({write_wordline_access_energy}). In any case, when if either of read_*_access_energy or write_*_access_energy is specified, the other must be present as well."
        if (value_access_energy or (read_value_access_energy and write_value_access_energy)):
            self.values_per_wordline : int = 1
            self.read_access_energy : float = read_value_access_energy if read_value_access_energy else value_access_energy
            self.write_access_energy : float = write_value_access_energy if write_value_access_energy else value_access_energy
        else:
            assert word_bits >= value_bits, f"Level: {name}: word_bits ({word_bits}) must be more than value_bits ({value_bits}), otherwise a value cannot fit on a single wordline."
            self.values_per_wordline : int = word_bits // value_bits
            self.read_access_energy : float = (read_wordline_access_energy if read_wordline_access_energy else wordline_access_energy) / self.values_per_wordline
            self.write_access_energy : float = (write_wordline_access_energy if write_wordline_access_energy else wordline_access_energy) / self.values_per_wordline
        self.leakage_energy : float = leakage_energy
        assert self.read_access_energy >= 0 and self.write_access_energy >= 0 and self.leakage_energy >= 0, f"Level: {name}: a negative access energy ({self.read_access_energy} read, {self.read_access_energy} write), ({self.leakage_energy} leak), does not mean anything (unless you are into sci-fi stuff)."
        assert not area or area >= 0, f"Level: {name}: a negative area ({area}) does not mean anything."
        self.area : Optional[float] = area
        # NOTE: 1/2 split of bandwidth for consistency with Timeloop - not a true must...
        assert (bandwidth and not read_bandwidth and not write_bandwidth) or (read_bandwidth and write_bandwidth), f"Level: {name}: either bandwidth ({bandwidth}) or read_bandwidth ({read_bandwidth}) and write_bandwidth ({write_bandwidth}) must be specified, if either of read_bandwidth or write_bandwidth is specified, the other must be specified as well."
        self.read_bandwidth : float = read_bandwidth if read_bandwidth else bandwidth/2
        self.write_bandwidth : float = write_bandwidth if write_bandwidth else bandwidth/2
        assert self.read_bandwidth >= 0 and self.write_bandwidth >= 0, f"Level: {name}: a negative bandwidth ({self.read_bandwidth} R, {self.write_bandwidth} W) does not mean anything."
        self.factors : Factors = factors if factors else Factors()
        self.tile_sizes : Shape = tile_sizes if tile_sizes else Shape(1, 1, 1)
        self.factors_constraints : dict[str, int] = factors_constraints if factors_constraints else {}
        assert all([constr[0] in self.dataflow and constr[1:] in ['', '>=', '<='] for constr in self.factors_constraints.keys()]), f"Level: {name}: all keys within factor constraints ({list(self.factors_constraints.keys())}) must be a dimension of the dataflow ({self.dataflow}) and in the form 'dim', 'dim<=', or 'dim>='."
        assert all([sum(constr[0] == dim for constr in self.factors_constraints.keys()) <= 1 for dim in self.dataflow]), f"Level: {name}: each dimension must occur at most once in constraints ({list(self.factors_constraints.keys())}), regardless of the use of '>=' or '<='."
        assert all([value > 0 for value in self.factors_constraints.values()]), f"Level: {name}: all constraints ({self.factors_constraints}) must have a value strictly > 0."
        self.dataflow_constraints : list[str] = dataflow_constraints if dataflow_constraints else []
        assert all([constr in self.dataflow for constr in self.dataflow_constraints]), f"Level: {name}: all dims specified as dataflow constraints ({self.dataflow_constraints}) must be part of the dataflow ({self.dataflow})."
        self.bypasses : list[str] = bypasses if bypasses else []
        self.in_bp : bool = 0 if (bypasses and 'in' in bypasses) else 1
        self.w_bp : bool = 0 if (bypasses and 'w' in bypasses) else 1
        self.out_bp : bool = 0 if (bypasses and 'out' in bypasses) else 1
        self.multiple_buffering : int = multiple_buffering
        assert self.multiple_buffering >= 1, f"Level: {name}: multiple buffering ({self.multiple_buffering}) must be at least 1."
        # NOTE: removed for consistency with Timeloop - not necessarily wrong...
        #if not self.in_bp and not self.w_bp:
        #    self.factors_constraints['K'] = 1
        #if not self.in_bp and not self.out_bp:
        #    self.factors_constraints['N'] = 1
        #if not self.out_bp and not self.w_bp:
        #    self.factors_constraints['M'] = 1

        # STATISTICS:
        self.active_instances = 1
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

        self.next_levels_with_bypass : dict[str, Optional[list[Level]]] = {'in': None, 'w': None, 'out': None}

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

    Lastly, factors encountered on the K dimension are returned too, to
    allow the caller to remove one such iteration to account for the
    presence or absence of the bias (bias_read flag).
    
    Arguments:
    - in_bp, w_bp, out_bp: whether the respective operands are bypassed or not,
                           defaults to the instance's setting if not provided
    - ignore_bypasses: if True, MOPs are returned only relative to this level's
                       accesses from strictly adjacent levels, bypassed operands
                       will thus show 0 MOPs.
    """
    def MOPs(self, in_bp : Optional[bool] = None, w_bp : Optional[bool] = None, out_bp : Optional[bool] = None, ignore_bypasses : bool = False) -> tuple[int, int, int, int, int]:
        in_bp = in_bp if in_bp != None else self.in_bp
        w_bp = w_bp if w_bp != None else self.w_bp
        out_bp = out_bp if out_bp != None else self.out_bp
        dataflow = self.actualDataflow()
        # size of the tiles moved between this level and the one below
        in_tile_elems = self.tile_sizes.K*self.tile_sizes.N
        w_tile_elems = self.tile_sizes.M*self.tile_sizes.K
        out_tile_elems = self.tile_sizes.M*self.tile_sizes.N
        factors_M = self.factors.dimProduct('M')
        factors_K = self.factors.dimProduct('K')
        factors_N = self.factors.dimProduct('N')
        # these are the complete tiles on this level, then updated across dataflows
        w_reads = factors_M*factors_K*w_tile_elems*w_bp
        in_reads = factors_K*factors_N*in_tile_elems*in_bp
        out_reads = factors_M*factors_N*out_tile_elems*out_bp
        out_writes = out_reads
        # this collects the factors along K, returned to handle the presence/absence of the bias
        out_reads_factors = out_bp
        # update w.r.t. the dataflow (orthogonal operands and stationarity)
        if self.next_is_compute:
            w_reads = w_reads*factors_N
            in_reads = in_reads*factors_M
            out_reads = out_reads*factors_K
            out_reads_factors *= factors_K
            out_writes = out_writes*factors_K
        elif dataflow == Dataflow.WS:
            in_reads = in_reads*factors_M
            out_reads = out_reads*factors_K
            out_reads_factors *= factors_K
            out_writes = out_writes*factors_K
        elif dataflow == Dataflow.OS:
            in_reads = in_reads*factors_M
            out_reads = out_reads
            w_reads = w_reads*factors_N
        elif dataflow == Dataflow.IS:
            out_reads = out_reads*factors_K
            out_reads_factors *= factors_K
            w_reads = w_reads*factors_N
            out_writes = out_writes*factors_K
        #print("BEFORE BYPASS:\n", f"{self.name}:{chr(9) * (2 - len(self.name)//8)}{in_reads} In_R, {w_reads} W_R, {out_reads} Our_R, {in_reads + w_reads + out_reads} Tot_R, {out_writes} Out_W, {out_reads_factors} Out_R_Fac")
        # handle bypasses
        if not ignore_bypasses:
            for operand, levels in self.next_levels_with_bypass.items():
                if levels != None:
                    in_between, level = levels[:-1], levels[-1]
                    in_reads_bp, w_reads_bp, out_reads_bp, out_writes_bp, out_reads_bp_factors = level.MOPs(operand == 'in', operand == 'w', operand == 'out', True)
                    # bulid the inner-most dataflow, by piling one against the other all non-1 loops, then look at which is the innermost dimension, that is the one that matters!
                    # TL;DR: consider the dataflow only w.r.t. the innermost loop, ignoring those with 1 iteration!
                    dataflow_bp = level.actualDataflow()
                    if dataflow_bp == None:
                        stationarity_to_address = True
                    else:
                        stationarity_to_address = False
                    for in_btwn in in_between[::-1]:
                        if isinstance(in_btwn, MemLevel):
                            in_btwn_factors_M = in_btwn.factors.dimProduct('M')
                            in_btwn_factors_K = in_btwn.factors.dimProduct('K')
                            in_btwn_factors_N = in_btwn.factors.dimProduct('N')
                            in_btwn_factors_full = in_btwn.factors.fullProduct()
                            if stationarity_to_address:
                                # all inner loops were 1s, deal with the dataflow now!
                                w_reads_bp = in_btwn_factors_M*in_btwn_factors_K*w_reads_bp
                                in_reads_bp = in_btwn_factors_K*in_btwn_factors_N*in_reads_bp
                                out_reads_bp = in_btwn_factors_M*in_btwn_factors_N*out_reads_bp
                                out_writes_bp = in_btwn_factors_M*in_btwn_factors_N*out_writes_bp
                                dataflow_bp = in_btwn.actualDataflow()
                                if dataflow_bp != None:
                                    stationarity_to_address = False
                                if dataflow_bp == Dataflow.WS:
                                    in_reads_bp = in_reads_bp*in_btwn_factors_M
                                    out_reads_bp = out_reads_bp*in_btwn_factors_K
                                    out_reads_bp_factors *= in_btwn_factors_K
                                    out_writes_bp = out_writes_bp*in_btwn_factors_K
                                elif dataflow_bp == Dataflow.OS:
                                    in_reads_bp = in_reads_bp*in_btwn_factors_M
                                    w_reads_bp = w_reads_bp*in_btwn_factors_N
                                elif dataflow_bp == Dataflow.IS:
                                    out_reads_bp = out_reads_bp*in_btwn_factors_K
                                    out_reads_bp_factors *= in_btwn_factors_K
                                    w_reads_bp = w_reads_bp*in_btwn_factors_N
                                    out_writes_bp = out_writes_bp*in_btwn_factors_K
                            else:
                                # dataflow handled among inner loops
                                w_reads_bp = in_btwn_factors_full*w_reads_bp
                                in_reads_bp = in_btwn_factors_full*in_reads_bp
                                out_reads_bp = in_btwn_factors_full*out_reads_bp
                                out_writes_bp = in_btwn_factors_full*out_writes_bp
                                out_reads_bp_factors *= in_btwn_factors_K
                        else:
                            in_reads_bp, w_reads_bp, out_reads_bp, out_writes_bp = in_btwn.mulByDim(in_reads_bp, w_reads_bp, out_reads_bp, out_writes_bp)
                            # do not update out_reads_bp_factors here, because in it go only iterations of which the first one is skipped,
                            # while in a fanout all fanned-out copies of the inner loop behave the same, there isn't a first different spatial iteration or anything
                        #print("IN BETWEEN BYPASS:\n", f"{in_btwn.name}:{chr(9) * (2 - len(in_btwn.name)//8)}{in_reads_bp} In_R, {w_reads_bp} W_R, {out_reads_bp} Our_R, {in_reads_bp + w_reads_bp + out_reads_bp} Tot_R, {out_writes_bp} Out_W, {out_reads_bp_factors} Out_R_Fac")
                    # consider the dataflow only among the three innermost loops, unless all loops seen until now were 1s
                    if stationarity_to_address:
                        # all inner loops were 1s, deal with the dataflow now!
                        w_reads_bp = factors_M*factors_K*w_reads_bp
                        in_reads_bp = factors_K*factors_N*in_reads_bp
                        out_reads_bp = factors_M*factors_N*out_reads_bp
                        out_writes_bp = factors_M*factors_N*out_writes_bp
                        if dataflow == Dataflow.WS:
                            in_reads_bp = in_reads_bp*factors_M
                            out_reads_bp = out_reads_bp*factors_K
                            out_reads_bp_factors *= factors_K
                            out_writes_bp = out_writes_bp*factors_K
                        elif dataflow == Dataflow.OS:
                            in_reads_bp = in_reads_bp*factors_M
                            w_reads_bp = w_reads_bp*factors_N
                        elif dataflow == Dataflow.IS:
                            out_reads_bp = out_reads_bp*factors_K
                            out_reads_bp_factors *= factors_K
                            w_reads_bp = w_reads_bp*factors_N
                            out_writes_bp = out_writes_bp*factors_K
                    else:
                        # dataflow handled among inner loops
                        factors_full = self.factors.fullProduct()
                        w_reads_bp = factors_full*w_reads_bp
                        in_reads_bp = factors_full*in_reads_bp
                        out_reads_bp = factors_full*out_reads_bp
                        out_writes_bp = factors_full*out_writes_bp
                        out_reads_bp_factors *= factors_K
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
        return self.factors.mem_footprint(self.tile_sizes, not self.bypasses or 'in' not in self.bypasses, not self.bypasses or 'w' not in self.bypasses, not self.bypasses or 'out' not in self.bypasses) <= self.size/self.multiple_buffering and super().checkConstraints()

    """
    Returns a string describing the current violation of constraints, if any.
    """
    def logConstraintsViolation(self) -> str:
        if not super().checkConstraints():
            return super().logConstraintsViolation()
        elif not self.checkConstraints():
            mem_footprint = self.factors.mem_footprint(self.tile_sizes, not self.bypasses or 'in' not in self.bypasses, not self.bypasses or 'w' not in self.bypasses, not self.bypasses or 'out' not in self.bypasses)
            return f"CONSTRAINTS VIOLATION: level {self.name}, memory used: {mem_footprint} VS memory available: {self.size/self.multiple_buffering:.0f}"
        return ""

    def __str__(self) -> str:
        return f"{super().__str__()}, size: {self.size}, read_access_energy: {self.read_access_energy}, write_access_energy: {self.write_access_energy}, leakage_energy: {self.leakage_energy}, read_bandwidth: {self.read_bandwidth}, write_bandwidth: {self.write_bandwidth}, bypasses: {self.bypasses}, multiple_buffering: {self.multiple_buffering}"


"""
Abstract class for a level introducing multiple spatial instances.
"""
class SpatialLevel(Level):
    dims : list[str] = None
    mesh : int = None


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
                      Valid dictionary keys are:
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
    def __init__(self, name : str, mesh : int, dim : Optional[str] = None, dims : Optional[list[str]] = None, area : Optional[float] = None, pe_to_pe : bool = False, spatial_multicast_support : bool = True, spatial_reduction_support : bool = True, power_gating_support : bool = False, factors : Optional[Factors] = None, tile_sizes : Optional[Shape] = None, factors_constraints : Optional[dict[str, int]] = None):
        self.name : str = name
        assert (dim and not dims) or (dims and not dim), f"Level: {name}: exactly one of dim ({dim}) or dims ({dims}) must be specified."
        self.dims : list[str] = [dim] if dim else dims
        self.dataflow : list[str] = self.dims
        assert all([dim in ['M', 'K', 'N'] for dim in self.dataflow]), f"Level: {name}: accepted names for dimensions are solely M, K and N, provided ones were {self.dataflow}."
        assert mesh > 0, f"Level: {name}: a spatial fanout must have a mesh ({mesh}) of at least 1."
        self.mesh : int = mesh
        assert not area or area >= 0, f"Level: {name}: a negative area ({area}) does not mean anything."
        self.area : Optional[float] = area
        assert not pe_to_pe or (spatial_multicast_support and spatial_reduction_support), f"Level: {name}: pe-to-pe forwarding is a form of spatial multicast or reduction, which must then both be supported to use it."
        self.pe_to_pe : bool = pe_to_pe # True in all cases where the operand independent (ex: if dim = D, the operand is the input) of "dim" is forwarded pe->pe rather than multicasted
        self.spatial_multicast_support : bool = spatial_multicast_support
        self.spatial_reduction_support : bool = spatial_reduction_support
        self.power_gating_support : bool = power_gating_support
        self.factors : Factors = factors if factors else Factors()
        self.tile_sizes : Shape = tile_sizes if tile_sizes else Shape(1, 1, 1)
        self.factors_constraints : dict[str, int] = factors_constraints if factors_constraints else {}
        assert all([constr[0] in self.dataflow and constr[1:] in ['', '>=', '<='] for constr in self.factors_constraints.keys()]), f"Level: {name}: all keys within factor constraints ({list(self.factors_constraints.keys())}) must be a dimension of the dataflow ({self.dataflow}) and in the form 'dim', 'dim<=', or 'dim>='."
        assert all([sum(constr[0] == dim for constr in self.factors_constraints.keys()) <= 1 for dim in self.dataflow]), f"Level: {name}: each dimension must occur at most once in constraints ({list(self.factors_constraints.keys())}), regardless of the use of '>=' or '<='."
        assert all([value > 0 for value in self.factors_constraints.values()]), f"Level: {name}: all constraints ({self.factors_constraints}) must have a value strictly > 0."

    """
    Let inputs be the amount of operations occuring on a level below this fanout,
    this method returns the amount of operations the seen above the fanout, accounting
    for spatial multicast and spatial reduction support of operands!
    """
    def mulByDim(self, in_reads : int, w_reads : int, out_reads : int, out_writes : int) -> tuple[int, int, int, int]:
        factor_M = self.factors.dimProduct('M')
        factor_K = self.factors.dimProduct('K')
        factor_N = self.factors.dimProduct('N')
        in_reads *= factor_K*factor_N
        w_reads *= factor_K*factor_M
        out_reads *= factor_N*factor_M
        if not self.spatial_multicast_support:
            in_reads *= factor_M
            w_reads *= factor_N
            out_reads *= factor_K
        out_writes *= factor_N*factor_M
        if not self.spatial_reduction_support:
            out_writes *= factor_K
                        
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
            return f"CONSTRAINTS VIOLATION: level {self.name}, spatial iterations used: {self.factors.fullProduct()} VS available instances (mesh): {self.mesh}"
        return ""

    def __str__(self) -> str:
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
                      Valid dictionary keys are:
                          - 'M', 'K', and 'N' for exact values;
                          - 'M<=', 'K<=', and 'N<=' for upper bounds;
                          - 'M>=', 'K>=', and 'N>=' for lower bounds;
                      NOTE: the use of the '<=' and '>=' constraints does not shorten the
                            mapper's runtime as much as exact constraints.
"""
class ComputeLevel(SpatialLevel):
    def __init__(self, name : str, mesh : int, compute_energy : float, cycles : int, dim : Optional[str] = None, dims : Optional[list[str]] = None, leakage_energy : float = 0, area : Optional[float] = None, factors : Optional[Factors] = None, tile_sizes : Optional[Shape] = None, factors_constraints : Optional[dict[str, int]] = None):
        self.name : str = name
        assert mesh == 1 or (dim and not dims) or (dims and not dim), f"Level: {name}: when mesh ({mesh}) is > 1, exactly one of dim ({dim}) or dims ({dims}) must be specified."
        self.dims : list[str] = ([dim] if dim else dims) if dim or dims else []
        self.dataflow : list[str] = self.dims
        assert all([dim in ['M', 'K', 'N'] for dim in self.dataflow]), f"Level: {name}: accepted names for dimensions are solely M, K and N, provided ones were {self.dataflow}."
        assert mesh > 0, f"Level: {name}: a zero or negative size ({mesh}) does not make sense."
        self.mesh : int = mesh # for a systolic array, this is the length of the operand buffers
        assert compute_energy >= 0, f"Level: {name}: a negative compute energy ({compute_energy}) does not mean anything (unless you watched too much Gundam and discovered Minovsky particles...)."
        self.compute_energy : float = compute_energy
        assert leakage_energy >= 0, f"Level: {name}: a negative leakage energy ({leakage_energy}) does not mean anything (unless you watched too much Gundam 00 and discovered GN particles...)."
        self.leakage_energy : float = leakage_energy
        assert not area or area >= 0, f"Level: {name}: a negative area ({area}) does not mean anything."
        self.area : Optional[float] = area
        assert cycles >= 0 # a negative number of clock-cycles per MAC does not mean anything
        self.cycles : int = cycles # clock cycles used per element in the inner dimension (latency of one MAC)
        self.factors : Factors = factors if factors else Factors()
        self.tile_sizes : Shape = tile_sizes if tile_sizes else Shape(1, 1, 1)
        self.factors_constraints : dict[str, int] = factors_constraints if factors_constraints else {}
        assert all([constr[0] in self.dataflow and constr[1:] in ['', '>=', '<='] for constr in self.factors_constraints.keys()]), f"Level: {name}: all keys within factor constraints ({list(self.factors_constraints.keys())}) must be a dimension of the dataflow ({self.dataflow}) and in the form 'dim', 'dim<=', or 'dim>='."
        assert all([sum(constr[0] == dim for constr in self.factors_constraints.keys()) <= 1 for dim in self.dataflow]), f"Level: {name}: each dimension must occur at most once in constraints ({list(self.factors_constraints.keys())}), regardless of the use of '>=' or '<='."
        assert all([value > 0 for value in self.factors_constraints.values()]), f"Level: {name}: all constraints ({self.factors_constraints}) must have a value strictly > 0."

        # STATISTICS:
        self.active_instances = 1
        self.temporal_iterations = 0

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
            return f"CONSTRAINTS VIOLATION: level {self.name}, concurrent MACs used: {self.factors.fullProduct()} VS concurrent MACs available: {self.mesh}"
        return ""

    def __str__(self) -> str:
        return f"{super().__str__()}, mesh: {self.mesh}, compute_energy: {self.compute_energy}, leakage_energy: {self.leakage_energy}, cycles: {self.cycles}"