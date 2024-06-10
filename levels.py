from factors import *


"""
Abstract class representing a level of the accelerator's architecture.

NOTE: the hierarchy is read just as the architecture is specified, with
"below" levels being closer to the computation and "above" ones closer
to DRAM or slower memories.
"""
class Level:
    # IMPORTANT:
    # Use the entries of "dataflow" to access "factors", so that you only
    # see the factors over which you are actually supposed to iterate!
    name: None
    dataflow: None # order of the loops | e.g. ['D', 'E', 'L']
    dataflow_constraints: None
    # NOTE:
    # Tile sizes are updated in "moveFactor".
    factors: None # iterations done for the dimensions at this lever
    tile_sizes: None # indicate the size of a tile used in the level BELOW
    constraints: None
    factors_contraints: None

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
    Returns the dataflow class for this level, one of:
    - Weight Stationary (WS)
    - Output Stationary (OS)
    - Input Stationary (IS)
    """
    def actualDataflow(self):
        innermost = [f for f in self.dataflow if self.factors.dimProduct(f) > 1]
        if len(innermost) == 0:
            return None
        elif innermost[-1] == 'L':
            return Dataflow.WS
        elif innermost[-1] == 'E':
            return Dataflow.OS
        elif innermost[-1] == 'D':
            return Dataflow.IS
        else:
            raise Exception(f"Unrecognized Dataflow in level {self.name} with dimension {innermost[-1]} != D, E, or L!")

    # TODO: add the possibility for < and > constraints!
    """
    Returns True iif factors present on this level satisfy all of
    its constraints.
    """
    def checkConstraints(self):
        return (('D' not in self.factors_contraints or self.factors_contraints['D'] == self.factors.dimProduct('D')) and
                ('E' not in self.factors_contraints or self.factors_contraints['E'] == self.factors.dimProduct('E')) and
                ('L' not in self.factors_contraints or self.factors_contraints['L'] == self.factors.dimProduct('L')))

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

"""
A Memory Level within the architecture, with the possibility to store
data and provide it as tiles to the levels below it.

Constructor arguments:
- name: the level's name
- size: the capacity (in number-of-operands, disregarding bits per operand)
- access_energy: energy required by each access (in pJ)
- bandwidth: the bandwidth for reads and writes, it will be divided in 1/2 for
             read and 1/2 for write (in operands/clock-cycle)
- dataflow: specifies the dimensions over which to iterate, defaults to all dimensions
- factors: specifies the initial factors for this level, should not be normally
           specified aside for initializing MSE from a specific configuration
- tile_sizes: specifies the initial tile sizes for this level, should not be normally
              specified aside for initializing MSE from a specific configuration,
              in which case it must be consistent with any other factors initialization
- factors_contraints: constraints on the factors that must be placed on this level.
                      Valid dictionary keys are 'D', 'E', and 'L'.
- dataflow_constraints: constraints for the order of loops at this level, for any dim
                        not specified here, all permutations are tried while keeping
                        fixed the relative order of constrained dimensions.
                        Valid strings are 'D', 'E', and 'L'.
- bypasses: list of operands which should bypass this level (i.o.w. not be stored here),
            valid strings are 'in', 'w', and 'out'.
- multiple_buffering: factor of multiple buffering employed by this level, must be >1
- read_access_energy: energy required for reads accesses, if specified overrides "access_energy"
- write_access_energy: energy required for write accesses, if specified overrides "access_energy"
  - Note: either both or none of read_access_energy and write_access_energy must be specified
- read_bandwidth: bandwidth allocated for reads, if specified overrides "bandwidth"
- write_bandwidth: bandwidth allocated for writes, if specified overrides "bandwidth"
  - Note: either both or none of read_bandwidth and write_bandwidth must be specified
"""
class MemLevel(Level):
    def __init__(self, name, size, access_energy = None, bandwidth = None, dataflow = None, factors = None, tile_sizes = None, factors_contraints = None, dataflow_constraints = None, bypasses = None, multiple_buffering = 1,  read_access_energy = None, write_access_energy = None, read_bandwidth = None, write_bandwidth = None):
        self.name = name
        # NOTE: this way of constructing the dataflow from the constraints is redundant, but useful if one wants to skip the
        # exploration of permutations since with this method the dataflow will be immediately consistent with constraints.
        self.dataflow = dataflow if dataflow else (dataflow_constraints + [dim for dim in ['D', 'E', 'L'] if dim not in dataflow_constraints] if dataflow_constraints else ['D', 'E', 'L']) # dimensions over which to iterate
        assert size >= 0, f"Level: {name}: a negative size ({size}) does not mean anything."
        self.size = size
        assert (access_energy and not read_access_energy and not write_access_energy) or (read_access_energy and write_access_energy), f"Level: {name}: either access_energy ({access_energy}) or read_access_energy ({read_access_energy}) and write_access_energy ({write_access_energy}) must be specified, if either of read_access_energy or write_access_energy is specified, the other must be specified as well."
        self.read_access_energy = read_access_energy if read_access_energy else access_energy
        self.write_access_energy = write_access_energy if write_access_energy else access_energy
        assert self.read_access_energy >= 0 and self.write_access_energy >= 0, f"Level: {name}: a negative access energy ({self.read_access_energy} R, {self.read_access_energy} W) does not mean anything (unless you are into sci-fi stuff)."
        # NOTE: 1/2 split of bandwidth for consistency with Timeloop - not a true must...
        assert (bandwidth and not read_bandwidth and not write_bandwidth) or (read_bandwidth and write_bandwidth), f"Level: {name}: either bandwidth ({bandwidth}) or read_bandwidth ({read_bandwidth}) and write_bandwidth ({write_bandwidth}) must be specified, if either of read_bandwidth or write_bandwidth is specified, the other must be specified as well."
        self.read_bandwidth = read_bandwidth if read_bandwidth else bandwidth/2
        self.write_bandwidth = write_bandwidth if write_bandwidth else bandwidth/2
        assert self.read_bandwidth >= 0 and self.write_bandwidth >= 0, f"Level: {name}: a negative bandwidth ({self.read_bandwidth} R, {self.write_bandwidth} W) does not mean anything."
        # This models how much an update costs w.r.t. a read. The true cost of an update is "update_cost" times cost of a read.
        self.update_cost = 1
        self.factors = factors if factors else Factors()
        self.tile_sizes = tile_sizes if tile_sizes else Shape(1, 1, 1)
        self.factors_contraints = factors_contraints if factors_contraints else {}
        assert all([constr in self.dataflow for constr in self.factors_contraints.keys()]), f"Level: {name}: all dims with factor constraints ({self.factors_contraints.keys()}) must be part of the dataflow ({self.dataflow})."
        self.dataflow_constraints = dataflow_constraints if dataflow_constraints else []
        assert all([constr in self.dataflow for constr in self.dataflow_constraints]), f"Level: {name}: all dims specified as dataflow constraints ({self.dataflow_constraints}) must be part of the dataflow ({self.dataflow})."
        self.bypasses = bypasses if bypasses else []
        self.in_bp = 0 if (bypasses and 'in' in bypasses) else 1
        self.w_bp = 0 if (bypasses and 'w' in bypasses) else 1
        self.out_bp = 0 if (bypasses and 'out' in bypasses) else 1
        self.multiple_buffering = multiple_buffering
        assert self.multiple_buffering >= 1, f"Level: {name}: multiple buffering ({self.multiple_buffering}) must be at least 1."
        # NOTE: removed for consistency with Timeloop - not necessarily wrong...
        #if not self.in_bp and not self.w_bp:
        #    self.factors_contraints['E'] = 1
        #if not self.in_bp and not self.out_bp:
        #    self.factors_contraints['L'] = 1
        #if not self.out_bp and not self.w_bp:
        #    self.factors_contraints['D'] = 1

        # STATISTICS:
        self.instances = 1
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

        self.next_layers_with_bypass = {'in': None, 'w': None, 'out': None}

    """
    Initializes the bypasses which start from this layer.
    Let "layers" be all layers starting from the next one going downward
    up until and including the last one bypassing "operand".

    => This method must be invoked while iterating from outer to inner layers
    as it updates the layer's notion of bypassed operations.
    """
    def initBypass(self, operand, layers):
        self.next_layers_with_bypass[operand] = layers
        if operand == 'in':
            self.in_bp = 0
        elif operand == 'w':
            self.w_bp = 0
        else:
            self.out_bp = 0

    """
    Sets MOPs statistics for this layer.
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
    # computed they require "last_out_writes"  and "last_out_reads" from the layer
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
    Sets Latency and related statistics for this layer.
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
    # Let "size" be the tile_size of the above layer!
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

    Lastly, factors encountered on the E dimension are returned too, to
    allow the caller to remove one such iteration to account for the
    presence or absence of the bias (bias_read flag).
    """
    def MOPs(self, in_bp = None, w_bp = None, out_bp= None, update_cost = None, ignore_bypasses = False):
        in_bp = in_bp if in_bp != None else self.in_bp
        w_bp = w_bp if w_bp != None else self.w_bp
        out_bp = out_bp if out_bp != None else self.out_bp
        update_cost = update_cost if update_cost else self.update_cost
        dataflow = self.actualDataflow()
        in_tile_elems = self.tile_sizes.E*self.tile_sizes.L
        w_tile_elems = self.tile_sizes.D*self.tile_sizes.E
        out_tile_elems = self.tile_sizes.D*self.tile_sizes.L
        factors_D = self.factors.dimProduct('D')
        factors_E = self.factors.dimProduct('E')
        factors_L = self.factors.dimProduct('L')
        # these are the base, then updated across dataflows
        w_reads = factors_D*factors_E*w_tile_elems*w_bp
        in_reads = factors_E*factors_L*in_tile_elems*in_bp
        out_reads = factors_D*factors_L*out_tile_elems*out_bp
        out_writes = update_cost*out_reads
        # this collects the factors along E, returned to handle the presence/absence of the bias
        out_reads_factors = out_bp
        # then updated w.r.t. each dataflow
        if self.next_is_compute:
            w_reads = w_reads*factors_L
            in_reads = in_reads*factors_D
            out_reads = out_reads*factors_E
            out_reads_factors *= factors_E
            out_writes = out_writes*factors_E
        elif dataflow == Dataflow.WS:
            in_reads = in_reads*factors_D
            out_reads = out_reads*factors_E
            out_reads_factors *= factors_E
            out_writes = out_writes*factors_E
        elif dataflow == Dataflow.OS:
            in_reads = in_reads*factors_D
            out_reads = out_reads
            w_reads = w_reads*factors_L
        elif dataflow == Dataflow.IS:
            out_reads = out_reads*factors_E
            out_reads_factors *= factors_E
            w_reads = w_reads*factors_L
            out_writes = out_writes*factors_E
        #print("BEFORE BYPASS:\n", f"{self.name}:{chr(9) * (2 - len(self.name)//8)}{in_reads} In_R, {w_reads} W_R, {out_reads} Our_R, {in_reads + w_reads + out_reads} Tot_R, {out_writes} Out_W, {out_reads_factors} Out_R_Fac")
        # handle bypasses
        if not ignore_bypasses:
            for operand, layers in self.next_layers_with_bypass.items():
                if layers != None:
                    in_between, layer = layers[:-1], layers[-1]
                    in_reads_bp, w_reads_bp, out_reads_bp, out_writes_bp, out_reads_bp_factors = layer.MOPs(operand == 'in', operand == 'w', operand == 'out', self.update_cost, True)
                    # bulid the inner-most dataflow, by piling one against the other all non-1 loops, then look at which is the innermost dimension, that is the one that matters!
                    # TL;DR: consider the dataflow only w.r.t. the innermost loop, ignoring those with 1 iteration!
                    dataflow_bp = layer.actualDataflow()
                    if dataflow_bp == None:
                        stationarity_to_address = True
                    else:
                        stationarity_to_address = False
                    for in_btwn in in_between[::-1]:
                        if isinstance(in_btwn, MemLevel):
                            if stationarity_to_address:
                                # all inner loops were 1s, deal with the dataflow now!
                                w_reads_bp = in_btwn.factors.dimProduct('D')*in_btwn.factors.dimProduct('E')*w_reads_bp
                                in_reads_bp = in_btwn.factors.dimProduct('E')*in_btwn.factors.dimProduct('L')*in_reads_bp
                                out_reads_bp = in_btwn.factors.dimProduct('D')*in_btwn.factors.dimProduct('L')*out_reads_bp
                                out_writes_bp = in_btwn.factors.dimProduct('D')*in_btwn.factors.dimProduct('L')*out_writes_bp
                                dataflow_bp = in_btwn.actualDataflow()
                                if dataflow_bp != None:
                                    stationarity_to_address = False
                                if dataflow_bp == Dataflow.WS:
                                    in_reads_bp = in_reads_bp*in_btwn.factors.dimProduct('D')
                                    out_reads_bp = out_reads_bp*in_btwn.factors.dimProduct('E')
                                    out_reads_bp_factors *= in_btwn.factors.dimProduct('E')
                                    out_writes_bp = out_writes_bp*in_btwn.factors.dimProduct('E')
                                elif dataflow_bp == Dataflow.OS:
                                    in_reads_bp = in_reads_bp*in_btwn.factors.dimProduct('D')
                                    w_reads_bp = w_reads_bp*in_btwn.factors.dimProduct('L')
                                elif dataflow_bp == Dataflow.IS:
                                    out_reads_bp = out_reads_bp*in_btwn.factors.dimProduct('E')
                                    out_reads_bp_factors *= in_btwn.factors.dimProduct('E')
                                    w_reads_bp = w_reads_bp*in_btwn.factors.dimProduct('L')
                                    out_writes_bp = out_writes_bp*in_btwn.factors.dimProduct('E')
                            else:
                                # dataflow handled among inner loops
                                w_reads_bp = in_btwn.factors.fullProduct()*w_reads_bp
                                in_reads_bp = in_btwn.factors.fullProduct()*in_reads_bp
                                out_reads_bp = in_btwn.factors.fullProduct()*out_reads_bp
                                out_writes_bp = in_btwn.factors.fullProduct()*out_writes_bp
                                out_reads_bp_factors *= in_btwn.factors.dimProduct('E')
                        else:
                            in_reads_bp, w_reads_bp, out_reads_bp, out_writes_bp = in_btwn.mulByDim(in_reads_bp, w_reads_bp, out_reads_bp, out_writes_bp)
                            # do not update out_reads_bp_factors here, because in it go only iterations of which the first one is skipped,
                            # while in a fanout all fanned-out copies of the inner loop behave the same, there isn't a first different spatial iteration or anything
                        #print("IN BETWEEN BYPASS:\n", f"{in_btwn.name}:{chr(9) * (2 - len(in_btwn.name)//8)}{in_reads_bp} In_R, {w_reads_bp} W_R, {out_reads_bp} Our_R, {in_reads_bp + w_reads_bp + out_reads_bp} Tot_R, {out_writes_bp} Out_W, {out_reads_bp_factors} Out_R_Fac")
                    # consider the dataflow only among the three innermost loops, unless all loops seen until now were 1s
                    if stationarity_to_address:
                        # all inner loops were 1s, deal with the dataflow now!
                        w_reads_bp = factors_D*factors_E*w_reads_bp
                        in_reads_bp = factors_E*factors_L*in_reads_bp
                        out_reads_bp = factors_D*factors_L*out_reads_bp
                        out_writes_bp = factors_D*factors_L*out_writes_bp
                        if dataflow == Dataflow.WS:
                            in_reads_bp = in_reads_bp*factors_D
                            out_reads_bp = out_reads_bp*factors_E
                            out_reads_bp_factors *= factors_E
                            out_writes_bp = out_writes_bp*factors_E
                        elif dataflow == Dataflow.OS:
                            in_reads_bp = in_reads_bp*factors_D
                            w_reads_bp = w_reads_bp*factors_L
                        elif dataflow == Dataflow.IS:
                            out_reads_bp = out_reads_bp*factors_E
                            out_reads_bp_factors *= factors_E
                            w_reads_bp = w_reads_bp*factors_L
                            out_writes_bp = out_writes_bp*factors_E
                    else:
                        # dataflow handled among inner loops
                        w_reads_bp = factors_D*factors_E*factors_L*w_reads_bp
                        in_reads_bp = factors_E*factors_L*factors_D*in_reads_bp
                        out_reads_bp = factors_D*factors_L*factors_E*out_reads_bp
                        out_writes_bp = factors_D*factors_L*factors_E*out_writes_bp
                        out_reads_bp_factors *= factors_E
                    #print(f"BYPASS ({operand}):\n", f"{self.name}->{layer.name}:{chr(9) * (3 - (len(self.name)+len(layer.name))//8)}{in_reads_bp} In_R, {w_reads_bp} W_R, {out_reads_bp} Our_R, {in_reads_bp + w_reads_bp + out_reads_bp} Tot_R, {out_writes_bp} Out_W, {out_reads_bp_factors} Out_R_Fac")
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
    Returns True iif factors present on this level satisfy all of
    its constraints, including fitting in the available memory.
    """
    def checkConstraints(self):
        return self.factors.mem_footprint(self.tile_sizes, not self.bypasses or 'in' not in self.bypasses, not self.bypasses or 'w' not in self.bypasses, not self.bypasses or 'out' not in self.bypasses) <= self.size/self.multiple_buffering and super().checkConstraints()

"""
A Spatial Fanout Level within the architecture, the core of a spatial architecture,
all subsequent levels will be replicated "mesh" times, each replica executing one
of the iterations done by this level, by each running the same inner loops as the
others, with partially different data.

Constructor arguments:
- name: the level's name
- mesh: the maximum spatial fanout available at this level
- dim: single dimension to spatially unroll at this level, if specified, dims
       must not be specified
- dims: list of dimensions to spatially unroll at this level, if specified, dims
        must not be specified. The order in dims does not matter
- pe_to_pe: if True, data is not multicasted in one shot, but sent to only the first
            fanout entry, which then forwards it to the second, and so on, just like
            the operands flowing in/out of a systolic array (or, like in a pipeline).
            This adds a warmup overhead to latency, but does not affect the total MOPs.
- spatial_multicast_support: True (default) if this level supports spatial multicast
- spatial_reduction_support: True (default) if this level supports spatial reduction
- factors: specifies the initial factors for this level, should not be normally
           specified aside for initializing MSE from a specific configuration
- tile_sizes: specifies the initial tile sizes for this level, should not be normally
              specified aside for initializing MSE from a specific configuration,
              in which case it must be consistent with any other factors initialization
- factors_contraints: constraints on the factors that must be placed on this level.
                      Valid dictionary keys are 'D', 'E', and 'L'.
"""
class FanoutLevel(Level):
    def __init__(self, name, mesh, dim = None, dims = None, pe_to_pe = False, spatial_multicast_support = True, spatial_reduction_support = True, factors = None, tile_sizes = None, factors_contraints = None):
        self.name = name
        assert (dim and not dims) or (dims and not dim), f"Level: {name}: exactly one of dim ({dim}) or dims ({dims}) must be specified."
        assert not dims or len(dims) <= 2, f"Level: {name}: CURRENT LIMITATION - at most 2 dimensions on the same fanout, limit dims ({dims}) to at most 2 entries."
        self.dims = [dim] if dim else dims
        self.dataflow = self.dims
        assert mesh > 0, f"Level: {name}: a spatial fanout must have a mesh ({mesh}) of at least 1."
        self.mesh = mesh
        assert not pe_to_pe or (spatial_multicast_support and spatial_reduction_support), f"Level: {name}: pe-to-pe forwarding is a form of spatial multicast or reduction, which must then both be supported to use it."
        self.pe_to_pe = pe_to_pe # True in all cases where the operand independent (ex: if dim = D, the operand is the input) of "dim" is forwarded pe->pe rather than multicasted
        self.spatial_multicast_support = spatial_multicast_support
        self.spatial_reduction_support = spatial_reduction_support
        self.factors = factors if factors else Factors()
        self.tile_sizes = tile_sizes if tile_sizes else Shape(1, 1, 1)
        self.factors_contraints = factors_contraints if factors_contraints else {}

    """
    Let inputs be the amount of operations occuring on a level below this fanout,
    this method returns the amount of operations the seen above the fanout, accounting
    for spatial multicast and spatial reduction support of operands!
    """
    def mulByDim(self, in_reads, w_reads, out_reads, out_writes):
        factor_D = self.factors.dimProduct('D')
        factor_E = self.factors.dimProduct('E')
        factor_L = self.factors.dimProduct('L')
        in_reads *= factor_E*factor_L
        w_reads *= factor_E*factor_D
        out_reads *= factor_L*factor_D
        if not self.spatial_multicast_support:
            in_reads *= factor_D
            w_reads *= factor_L
            out_reads *= factor_E
        out_writes *= factor_L*factor_D
        if not self.spatial_reduction_support:
            out_writes *= factor_E
                        
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
A Compute Level within the architecture, it is a placeholder for any processing
element (PE) capable of multiply and accumulate (MAC).

Note: iterations at this level are equivalent to requiring that a PE can, in the
clock-cycles specified in "cycles", execute all such iterations. It is also assumed
that the data required for all iterations done at this level is held within hardware
registers whose energy cost is modeled by the "compute energy", together with the
actual compute cost. As such, iterations done here are equivalent to a PE capable
of multiple concurrent (simultaneous or pipelined) MACs.
Then intuitively, increasing the number of iterations done at this level linearly
increases the required bandwidth of all memory levels feeding it, as they need to
keep up with the concurrent MACs done here.

Constructor arguments:
- name: the level's name
- size: how many concurrent (simultaneous or pipelined) MACs the compute element
        can perform within "cycles" clock-cycles.
- compute_energy: the energy required for a single MAC (regardless of how many
                  you run concurrently), accounting for all computation-related
                  costs at the PE level.
- cycles: the number of clock cycles of latency required to execute "size" MACs
- dataflow: specifies the dimensions over which to iterate, defaults to all dimensions
- factors: specifies the initial factors for this level, should not be normally
           specified aside for initializing MSE from a specific configuration
- tile_sizes: specifies the initial tile sizes for this level, should not be normally
              specified aside for initializing MSE from a specific configuration,
              in which case it must be consistent with any other factors initialization
- factors_contraints: constraints on the factors that must be placed on this level.
                      Valid dictionary keys are 'D', 'E', and 'L'.
- dataflow_constraints: constraints for the order of loops at this level, for any dim
                        not specified here, all permutations are tried while keeping
                        fixed the relative order of constrained dimensions.
                        Valid strings are 'D', 'E', and 'L'.
"""
class ComputeLevel(Level):
    def __init__(self, name, size, compute_energy, cycles, dataflow = None, factors = None, tile_sizes = None, factors_contraints = None, dataflow_constraints = None):
        self.name = name
        # NOTE: this way of constructing the dataflow from the constraints is redundant, but useful if one wants to skip the
        # exploration of permutations since with this method the dataflow will be immediately consistent with constraints.
        self.dataflow = dataflow if dataflow else (dataflow_constraints + [dim for dim in ['D', 'E', 'L'] if dim not in dataflow_constraints] if dataflow_constraints else ['D', 'E', 'L']) # dimensions over which to iterate
        assert size > 0, f"Level: {name}: a zero or negative size ({size}) does not make sense."
        self.size = size # for a systolic array, this is the length of the operand buffers
        assert compute_energy >= 0, f"Level: {name}: a negative compute energy ({compute_energy}) does not mean anything (unless you watched too much Gundam and discovered Minovsky particles...)."
        self.compute_energy = compute_energy
        assert cycles >= 0 # a negative number of clock-cycles per MAC does not mean anything
        self.cycles = cycles # clock cycles used per element in the inner dimension (latency of one MAC)
        self.factors = factors if factors else Factors()
        self.tile_sizes = tile_sizes if tile_sizes else Shape(1, 1, 1)
        self.factors_contraints = factors_contraints if factors_contraints else {}
        self.dataflow_constraints = dataflow_constraints if dataflow_constraints else []
        assert all([constr in self.dataflow for constr in self.dataflow_constraints]), f"Level: {name}: all dims specified as dataflow constraints ({self.dataflow_constraints}) must be part of the dataflow ({self.dataflow})."

        # STATISTICS:
        self.instances = 1
        self.temporal_iterations = 0

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
    Returns True iif factors present on this level satisfy all of its constraints,
    including not exceeding the phisical number of concurrent MACs performed here.
    """
    def checkConstraints(self):
        return self.factors.fullProduct() <= self.size and super().checkConstraints()