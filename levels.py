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
        if factor in self.factors[dimension]:
            self.factors[dimension][factor] += amount
        else:
            self.factors[dimension][factor] = amount

    """
    Remove "amount" instances of the provided factor from those of
    "dimension" in the current level.
    Return False if the removal failed because th current level does not
    have at least "amount" instances of "factor" along "dimension".
    """
    def removeFactor(self, dimension, factor, amount = 1):
        if self.factors[dimension][factor] < amount:
            return False
        self.factors[dimension][factor] -= amount
        if self.factors[dimension][factor] == 0:
            self.factors[dimension].pop(factor)
        return True

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
            return "WS"
        elif innermost[-1] == 'E':
            return "OS"
        elif innermost[-1] == 'D':
            return "IS"
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
data and provided it as tiles to the levels below it.
"""
class MemLevel(Level):
    def __init__(self, name, dataflow, size, access_energy, bandwidth, constraints = None, factors = None, tile_sizes = None, factors_contraints = None, bypasses = None, multiple_buffering = 1, read_bandwidth = None, write_bandwidth = None):
        self.name = name
        self.dataflow = dataflow
        self.size = size
        self.access_energy = access_energy
        # NOTE: 1/2 split of bandwidth for consistency with Timeloop - not a true must...
        self.read_bandwidth = read_bandwidth if read_bandwidth else bandwidth/2
        self.write_bandwidth = write_bandwidth if write_bandwidth else bandwidth/2
        # This models how much an update costs w.r.t. a read. The true cost of an update is "update_cost" times cost of a read.
        self.update_cost = 1
        self.constraints = constraints if constraints else {}
        self.factors = factors if factors else Factors()
        self.tile_sizes = tile_sizes if tile_sizes else Shape(1, 1, 1)
        self.factors_contraints = factors_contraints if factors_contraints else {}
        self.bypasses = bypasses if bypasses else []
        self.in_bp = 0 if (bypasses and 'in' in bypasses) else 1
        self.w_bp = 0 if (bypasses and 'w' in bypasses) else 1
        self.out_bp = 0 if (bypasses and 'out' in bypasses) else 1
        self.multiple_buffering = multiple_buffering
        assert self.multiple_buffering >= 1 # multiple buffering must be at least 1
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
    Returns the memory operation between this level and the one below it.
    Specifically: returns reads going downward from this level and writes
    coming upward from lower levels. In other words, reads and writes
    between this level and the one below it.

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
        elif dataflow == "WS":
            in_reads = in_reads*factors_D
            out_reads = out_reads*factors_E
            out_reads_factors *= factors_E
            out_writes = out_writes*factors_E
        elif dataflow == "OS":
            in_reads = in_reads*factors_D
            out_reads = out_reads
            w_reads = w_reads*factors_L
        elif dataflow == "IS":
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
                    # the inner-most dataflow, built by piling one against the other all non-1 loops, and then looking at which is the innermost dimension, is the one that matters!
                    # TL;DR: consider the dataflow only among the three innermost loops, unless they are all 1s!
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
                                if dataflow_bp == "WS":
                                    in_reads_bp = in_reads_bp*in_btwn.factors.dimProduct('D')
                                    out_reads_bp = out_reads_bp*in_btwn.factors.dimProduct('E')
                                    out_reads_bp_factors *= in_btwn.factors.dimProduct('E') if out_reads_bp_factors == 1 else 1
                                    out_writes_bp = out_writes_bp*in_btwn.factors.dimProduct('E')
                                elif dataflow_bp == "OS":
                                    in_reads_bp = in_reads_bp*in_btwn.factors.dimProduct('D')
                                    w_reads_bp = w_reads_bp*in_btwn.factors.dimProduct('L')
                                elif dataflow_bp == "IS":
                                    out_reads_bp = out_reads_bp*in_btwn.factors.dimProduct('E')
                                    out_reads_bp_factors *= in_btwn.factors.dimProduct('E') if out_reads_bp_factors == 1 else 1
                                    w_reads_bp = w_reads_bp*in_btwn.factors.dimProduct('L')
                                    out_writes_bp = out_writes_bp*in_btwn.factors.dimProduct('E')
                            else:
                                # dataflow handled among inner loops
                                w_reads_bp = in_btwn.factors.fullProduct()*w_reads_bp
                                in_reads_bp = in_btwn.factors.fullProduct()*in_reads_bp
                                out_reads_bp = in_btwn.factors.fullProduct()*out_reads_bp
                                out_writes_bp = in_btwn.factors.fullProduct()*out_writes_bp
                                out_reads_bp_factors *= in_btwn.factors.dimProduct('E') if out_reads_bp_factors == 1 else 1
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
                        if dataflow == "WS":
                            in_reads_bp = in_reads_bp*factors_D
                            out_reads_bp = out_reads_bp*factors_E
                            out_reads_bp_factors *= factors_E
                            out_writes_bp = out_writes_bp*factors_E
                        elif dataflow == "OS":
                            in_reads_bp = in_reads_bp*factors_D
                            w_reads_bp = w_reads_bp*factors_L
                        elif dataflow == "IS":
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
                    out_reads_factors = out_reads_bp_factors
        #print("AFTER BYPASS:\n", f"{self.name}:{chr(9) * (2 - len(self.name)//8)}{in_reads} In_R, {w_reads} W_R, {out_reads} Our_R, {in_reads + w_reads + out_reads} Tot_R, {out_writes} Out_W, {out_reads_factors} Out_R_Fac\n")
        return in_reads, w_reads, out_reads, out_writes, out_reads_factors                                                   #S;G

    """
    Returns the provided MOPs (or newly calculated MOPs for this level)
    scaled by the MOPs's weight/energy at this level.
    """
    def WMOPs(self, MOPs = None):
        return MOPs*self.access_energy if MOPs else sum(self.MOPs())*self.access_energy

    """
    Returns True iif factors present on this level satisfy all of
    its constraints, including fitting in the available memory.
    """
    def checkConstraints(self):
        return self.factors.mem_footprint(self.tile_sizes, not self.bypasses or 'in' not in self.bypasses, not self.bypasses or 'w' not in self.bypasses, not self.bypasses or 'out' not in self.bypasses) <= self.size/self.multiple_buffering and super().checkConstraints()

class FanoutLevel1D(Level):
    def __init__(self, name, dim, mesh, pe_to_pe = None, constraints = None, factors = None, tile_sizes = None, factors_contraints = None):
        self.name = name
        self.dataflow = [dim]
        self.dim = dim
        self.mesh = mesh
        self.pe_to_pe = pe_to_pe if pe_to_pe != None else False # True in all cases where the operand independent (ex: if dim = D, the operand is the input) of "dim" is forwarded pe->pe rather than read multiple times
        self.constraints = constraints if constraints else {}
        self.factors = factors if factors else Factors()
        self.tile_sizes = tile_sizes if tile_sizes else Shape(1, 1, 1)
        self.factors_contraints = factors_contraints if factors_contraints else {}

    # let inputs be the amount of operations for each leaf, returns the amount of
    # operations above the fanout accounting for PE-to-PE forwarding of operands!
    def mulByDim(self, in_reads, w_reads, out_reads, out_writes):
        iterations = self.factors.fullProduct()
        if self.pe_to_pe:
            if self.dim != 'D':
                in_reads *= iterations
            if self.dim != 'E':
                out_reads *= iterations
                out_writes *= iterations
            if self.dim != 'L':
                w_reads *= iterations
        else:
            w_reads *= iterations
            in_reads *= iterations
            out_reads *= iterations
            out_writes *= iterations
        return in_reads, w_reads, out_reads, out_writes

    def divByDim(self, in_reads, w_reads, out_reads, out_writes):
        iterations = self.factors.fullProduct()
        if self.pe_to_pe:
            if self.dim != 'D':
                in_reads //= iterations
            if self.dim != 'E':
                out_reads //= iterations
                out_writes //= iterations
            if self.dim != 'L':
                w_reads //= iterations
        else:
            w_reads //= iterations
            in_reads //= iterations
            out_reads //= iterations
            out_writes //= iterations
        return in_reads, w_reads, out_reads, out_writes

    # => The returned value must be multiplied by the factors above it.
    def latency(self):
        return 0 # change this if we model the network's latency

    def checkConstraints(self):
        return self.factors.dimProduct(self.dim) <= self.mesh and super().checkConstraints()

class FanoutLevel2D(Level):
    def __init__(self, name, dimX, dimY, meshX, meshY, pe_to_peX = None, pe_to_peY = None, constraints = None, factors = None, tile_sizes = None, factors_contraints = None):
        self.name = name
        self.dataflow = [dimX, dimY]
        self.dimX = dimX
        self.dimY = dimY
        self.meshX = meshX
        self.meshY = meshY
        self.pe_to_peX = pe_to_peX if pe_to_peX != None else False # True in all cases where the operand independent (ex: if dimX = D, the operand is the input) of "dimX" is forwarded pe->pe rather than read multiple times
        self.pe_to_peY = pe_to_peY if pe_to_peY != None else False # True in all cases where the operand independent (ex: if dimY = D, the operand is the input) of "dimY" is forwarded pe->pe rather than read multiple times
        self.constraints = constraints if constraints else {}
        self.factors = factors if factors else Factors()
        self.tile_sizes = tile_sizes if tile_sizes else Shape(1, 1, 1)
        self.factors_contraints = factors_contraints if factors_contraints else {}

    # let inputs be the amount of operations for each leaf, returns the amount of
    # operations above the fanout accounting for PE-to-PE forwarding of operands!
    def mulByDim(self, in_reads, w_reads, out_reads, out_writes):
        iterationsX = self.factors.dimProduct(self.dimX)
        if self.pe_to_peX:
            if self.dimX != 'D':
                in_reads *= iterationsX
            if self.dimX != 'E':
                out_reads *= iterationsX
                out_writes *= iterationsX
            if self.dimX != 'L':
                w_reads *= iterationsX
        else:
            w_reads *= iterationsX
            in_reads *= iterationsX
            out_reads *= iterationsX
            out_writes *= iterationsX
        iterationsY = self.factors.dimProduct(self.dimY)
        if self.pe_to_peY:
            if self.dimY != 'D':
                in_reads *= iterationsY
            if self.dimY != 'E':
                out_reads *= iterationsY
                out_writes *= iterationsY
            if self.dimY != 'L':
                w_reads *= iterationsY
        else:
            w_reads *= iterationsY
            in_reads *= iterationsY
            out_reads *= iterationsY
            out_writes *= iterationsY
        return in_reads, w_reads, out_reads, out_writes

    def divByDim(self, in_reads, w_reads, out_reads, out_writes):
        iterationsX = self.factors.dimProduct(self.dimX)
        if self.pe_to_peX:
            if self.dimX != 'D':
                in_reads //= iterationsX
            if self.dimX != 'E':
                out_reads //= iterationsX
                out_writes //= iterationsX
            if self.dimX != 'L':
                w_reads //= iterationsX
        else:
            w_reads //= iterationsX
            in_reads //= iterationsX
            out_reads //= iterationsX
            out_writes //= iterationsX
        iterationsY = self.factors.dimProduct(self.dimY)
        if self.pe_to_peY:
            if self.dimY != 'D':
                in_reads //= iterationsY
            if self.dimY != 'E':
                out_reads //= iterationsY
                out_writes //= iterationsY
            if self.dimY != 'L':
                w_reads //= iterationsY
        else:
            w_reads //= iterationsY
            in_reads //= iterationsY
            out_reads //= iterationsY
            out_writes //= iterationsY
        return in_reads, w_reads, out_reads, out_writes

    # => The returned value must be multiplied by the factors above it.
    def latency(self):
        return 0 # change this if we model the network's latency

    def checkConstraints(self):
        return (self.factors.dimProduct(self.dimX) <= self.meshX and
                self.factors.dimProduct(self.dimY) <= self.meshY and
                super().checkConstraints())
        
class ComputeLevel(Level):
    def __init__(self, name, dataflow, size, compute_energy, cycles, constraints = None, factors = None, tile_sizes = None, factors_contraints = None):
        self.name = name
        self.dataflow = dataflow
        self.size = size # for a systolic array, this is the length of the operand buffers
        self.compute_energy = compute_energy # 
        self.cycles = cycles # clock cycles used per element in the inner dimension (latency of one MAC)
        self.constraints = constraints if constraints else {}
        self.factors = factors if factors else Factors()
        self.tile_sizes = tile_sizes if tile_sizes else Shape(1, 1, 1)
        self.factors_contraints = factors_contraints if factors_contraints else {}

    # TODO: remove size, factors, and constraints from here, this must become just an empty shell

    # => The returned value must be multiplied by the factors above it.
    def latency(self):
        return self.factors.fullProduct()*self.cycles

    def computeCost(self, iterations = 1):
        return self.compute_energy * iterations

    def checkConstraints(self):
        return self.factors.fullProduct() <= self.size and super().checkConstraints()