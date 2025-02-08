from __future__ import annotations
from copy import deepcopy

from settings import *
from factors import *
from levels import *
from utils import *

"""
Class wrapping a list of levels into an architecture.
In FF, the terms outermost and innermost refer to the first and last elements
in the list, in accordance with the HW componenets represented by levels.

This class becomes operational in 4 steps:
1) construction of the 'Level' instances it wraps <- per-level arguments
2) construction of this class <- coupling
3) initialization and validation of all 'Level' instances with calls to 'initArch'
4) initialization of this class with a call to 'initFactors' <- computation

Constructor arguments:
- levels: the list of levels for the architecture
- coupling: dimensions and dimension-operand coupling
- name: the name for the architecture
"""
class Arch(list[Level]):
    def __init__(self, levels : list[Level], coupling : Coupling, name : str ="<unnamed architecture>"):
        self.name : str = name
        # TODO: maybe store Wart, EDP, etc. here?
        self.coupling : Coupling = coupling
        self.stride_values : dict[str, int] = {}
        self.initialized : bool = False
        
        super().__init__(levels)

        assert len(self) >= 2, f"Arch: {self.name}: at least two levels (one 'MemLevel', one 'ComputeLevel') are required for an architecture, {len(self)} provided."
        assert all(isinstance(item, Level) for item in self), f"Arch: {self.name}: all architecture entries must be levels (instances of the 'Level' class), provided ones are {list(map(lambda x : type(x).__name__, self))}."
        assert isinstance(self[0], MemLevel), f"Arch: {self.name}: the outermost (idx 0) level must be a memory level ('MemLevel' instance), {type(self[0]).__name__} provided."
        assert isinstance(self[-1], ComputeLevel), f"Arch: {self.name}: the innermost (idx len-1) level must be a compute level ('ComputeLevel' instance), {type(self[-1]).__name__} provided."
        assert not any(map(lambda l : isinstance(l, ComputeLevel), self[:-1])), f"Arch: {self.name}: no other compute levels admitted apart from the innermost one."
        assert all(constr[1:] == '>=' for constr in self[0].factors_constraints.keys()), f"Arch: {self.name}: the outermost (idx 0) level's constraints ({self[0].factors_constraints}) can only be of the '>=' kind."

        for level in self:
            level.initArch(self)

        self.setupBypasses()
        self.setupSpatialLevelPointers()
        next((level for level in self[::-1] if isinstance(level, MemLevel)), None).next_is_compute = True

    """
    Returns a compact representation of the current mapping.
    
    Arguments:
    - copy: if True, the returned mapping is a deep-copy of the current one
    """
    def exportMapping(self, copy : bool = False) -> list[LevelCore]:
        if copy:
            return [LevelCore(deepcopy(level.dataflow), deepcopy(level.factors), deepcopy(level.tile_sizes)) for level in self]
        else:
            return [LevelCore(level.dataflow, level.factors, level.tile_sizes) for level in self]

    """
    Loads a mapping (preferably coming from "exportMapping") into the architecture.
    
    Arguments:
    - mapping: the mapping to be loaded
    - from_level: optional outermost level included in the copy
    - to_level: optional innermost level included in the copy
    """
    def importMapping(self, mapping : list[LevelCore], from_level : int = 0, to_level : int = -1) -> None:
        for i in range(from_level, to_level % len(self)):
            self[i].dataflow = mapping[i].dataflow
            self[i].factors = mapping[i].factors
            self[i].tile_sizes = mapping[i].tile_sizes

    """
    Checks if the provided coupling is compatible with the architecture's.
    Updates the provided computation with its missing dimensions if needed.
    """
    def checkCouplingCompatibility(self, coupling : Coupling, comp : Shape, verbose : bool = False) -> None:
        assert coupling.isCompatibleComp(comp), f"The provided computation ({comp}) is not compatible with the provided coupling ({coupling.compactStr()}), note that each dimension and stride of the latter must appear in the computation."
        assert self.coupling.isCompatibleCoupling(coupling), f"The provided coupling ({coupling.compactStr()}) is not compatible with arch {self.name}'s coupling ({self.coupling.compactStr()})."
        if verbose and not self.coupling.isSubcoupling(coupling):
            print(f"WARNING: the used coupling ({coupling.compactStr()}) is not a subcoupling of arch {self.name}'s coupling ({self.coupling.compactStr()}), but is still compatible.")
        comp.fitToCoupling(coupling)

    """
    Moves a factor between the same dimension of two levels, transitioning
    between adjacent mappings. It also updates tile sizes accordingly for all levels
    between the affected ones.

    Returns False (and reverts changes) if the move violates any constraint.

    Arguments:
    - src_level_idx: level giving up the prime factor
    - dst_level_idx: level receiving the prime factor
    - dimension: the computation's dimension of the moved factor
    - factor: specify which prime factor is moved
    - amount: which arity of the factor is moved
    - skip_src_constraints: if True, any constraints violation on the source level
                            is ignored.
    - skip_dst_constraints: if True, any constraints violation on the destination level
                            is ignored.
    """
    def moveFactor(self, src_level_idx : int, dst_level_idx : int, dimension : str, factor : int, amount : int = 1, skip_src_constraints : bool = False, skip_dst_constraints : bool = False) -> bool:
        # check that the factor exists in the required amount
        if not self[src_level_idx].removeFactor(dimension, factor, amount):
            return False
        # check src constraints
        if not self[src_level_idx].checkFactorsConstraints() and not skip_src_constraints:
            self[src_level_idx].addFactor(dimension, factor, amount)
            return False   
        self[dst_level_idx].addFactor(dimension, factor, amount)
        # update tile sizes
        factor_to_amount = factor**amount
        if src_level_idx < dst_level_idx:
            for i in range(src_level_idx, dst_level_idx):
                self[i].tile_sizes[dimension] *= factor_to_amount
        elif src_level_idx > dst_level_idx:
            for i in range(dst_level_idx, src_level_idx):
                self[i].tile_sizes[dimension] //= factor_to_amount
        # check dst and all in between constraints
        constraints_check = [level.checkFactorsConstraints() for level in (self[src_level_idx:dst_level_idx+1] if src_level_idx < dst_level_idx else self[dst_level_idx:src_level_idx+1])]
        if not all(constraints_check) and not skip_dst_constraints:
            self[src_level_idx].addFactor(dimension, factor, amount)
            assert self[dst_level_idx].removeFactor(dimension, factor, amount) # something is broken, cannot undo the move
            if src_level_idx < dst_level_idx:
                for i in range(src_level_idx, dst_level_idx):
                    self[i].tile_sizes[dimension] //= factor_to_amount
            elif src_level_idx > dst_level_idx:
                for i in range(dst_level_idx, src_level_idx):
                    self[i].tile_sizes[dimension] *= factor_to_amount
            return False
        return True

    """
    Initializes the architecture's mapping with all the computation's prime
    factors placed on the first level. This is the mapper's starting point.
    Stride values are also imported from the computation.

    This method must always be called once before using the architecture.
    """
    def initFactors(self, comp : Shape) -> None:
        # initialize with all factors on first level, all tile sizes of 1!
        self[0].factors = Factors({dim: prime_factors(comp[dim]) for dim in self.coupling.dims})
        self.stride_values = {dim: (comp[dim] if dim in comp else 1) for dim in list(self.coupling.in_strides.values()) + list(self.coupling.w_strides.values()) + list(self.coupling.out_strides.values())}
        self.initialized = True

    """
    Resets the architecture's mapping to a blank state.
    Ensure to call 'initFactors' before using this instance again.
    [Dataflows are not affected, and remain arbitrary]
    
    Arguments:
    - copy: if True, it does not overwrite the previous state's
            datastructures, but creates new blank instances of them.
    """
    def resetFactors(self, copy : bool = False) -> None:
        self.initialized = False
        if copy:
            self.stride_values = {}
            for level in self:
                level.factors = Factors(self.coupling.dims)
                level.tile_sizes = Shape({dim: 1 for dim in self.coupling.dims})
        else:
            self.stride_values.clear()
            for level in self:
                level.factors.clear()
                level.tile_sizes.clear()

    """
    This function must start from arch having all factors on its first level,
    then it ensures that all constraints of all levels are satisfied.
    If 'allow_padding' is True then computation dimensions not exactly
    divided by constraints will be padded up to a multiple of constraints.

    This function assumes that the computation is larger than what required
    by constraints, if this is not satisfied, an assertion will be triggered.
    -> use 'fitConstraintsToComp' to prevent such a situation.
    """
    def enforceFactorsConstraints(self, allow_padding : bool = False, verbose_padding : bool = True) -> None:
        # assuming that initially all factors are on the first level
        for i in range(1, len(self)):
            level = self[i]
            for k in level.factors_constraints.keys():
                assert k[0] in level.dataflow, f"Arch: {self.name} -> Level: {level.name}: cannot enforce constraint on dimension {k[0]} that is not in dataflow {level.dataflow}."
            # initialize only == and >= constraints, leave <= set to 1
            constr_factors = Factors({dim: (prime_factors(level.factors_constraints[dim]) if dim in level.factors_constraints else (prime_factors(level.factors_constraints[f'{dim}>=']) if f'{dim}>=' in level.factors_constraints else {})) for dim in self.coupling.dims})
            if self[0].factors.isSubset(constr_factors) or allow_padding:
                for dim in self.coupling.dims:
                    dim_size = self[0].factors.dimProduct(dim)
                    constraint = constr_factors.dimProduct(dim)
                    if dim_size%constraint == 0:
                        for fact, amount in constr_factors[dim].items():
                            assert self.moveFactor(0, i, dim, fact, amount, True, True), f"Arch {self.name} -> Level: {level.name}: Constraints are not satisfiable! Cannot give {amount} instances of prime factor {fact} to level {level.name} on dimension {dim}."
                    else:
                        padded_dim_size = dim_size + constraint - dim_size%constraint
                        if verbose_padding: print(f"PADDING: Arch: {self.name} -> Level: {level.name}: enlarged {dim} from {dim_size} to {padded_dim_size} to respect constraints.")
                        self[0].factors[dim] = prime_factors(padded_dim_size)
                        self[0].factors.resetDimProducts([dim])
                        for fact, amount in constr_factors[dim].items():
                            # be wary that here we skip constraints checks in moveFactor, so one must follow up this method with checkFactorsConstraints
                            assert self.moveFactor(0, i, dim, fact, amount, True, True), f"Arch: {self.name} -> Level: {level.name}: Failed to enforce constraints even with padding..."

    """
    Checks factors allocation constraints. Returns True if no violation is found.
    """
    def checkFactorsConstraints(self) -> bool:
        return all(level.checkFactorsConstraints() for level in self)

    """
    Checks dataflow (loop ordering) constraints. Returns False if a violation is found.
    """
    def checkDataflowConstraints(self) -> bool:
        return all(level.checkDataflowConstraints() for level in self if isinstance(level, MemLevel))

    """
    Returns a string describing the current violations of constraints, if any.
    """
    def logConstraintsViolations(self) -> bool:
        return '\n'.join(filter(lambda s : len(s) > 0, (level.logConstraintsViolation() for level in self)))

    """
    Initializes bypasses between MemLevels of the architecture.
    For each operand and bypass, setup a pointer in the last MemLevel before the bypass
    to the list of levels affected by the bypass.

    This way such last level can compute its MOPs according to the level it actually
    supplies data to.
    """
    def setupBypasses(self) -> None:
        # bypasses at the initial level simply skip the cost of operands
        for bypass in ['in', 'w', 'out']:
            # if the first MemLevel has a bypass, no need to initialize it, just let it
            # affect its internal MOPs computation and that's it!
            last_before_bypass = 0
            i = 1
            while i < len(self):
                if not isinstance(self[i], MemLevel):
                    i += 1
                elif bypass in self[i].bypasses:
                    last_bypassing = i
                    while i < len(self) and (not isinstance(self[i], MemLevel) or bypass in self[i].bypasses):
                        if isinstance(self[i], MemLevel):
                            last_bypassing = i
                        i += 1
                    self[last_before_bypass].initBypass(bypass, self[last_before_bypass+1:last_bypassing+1])
                    last_before_bypass = i
                    i += 1
                else:
                    last_before_bypass = i
                    i += 1

    """
    Sets up a pointers list from each MemLevel to the possible group of consecutive
    spatial (fanouts and compute) levels which may follow it.
    The "next_is_compute" is also set on the last memory level before compute.
    """
    def setupSpatialLevelPointers(self) -> None:
        for i in range(len(self)):
            level = self[i]
            if isinstance(level, MemLevel):
                j = i + 1
                while j < len(self) and isinstance(self[j], SpatialLevel):
                    j += 1
                level.next_spatials = self[i+1:j]
                level.next_is_compute = isinstance(self[j-1], ComputeLevel)

    """
    Returns the overall utilization of spatial instances of a mapping.
    """
    def spatialUtilization(self) -> float:
        utilization = 1
        for level in self:
            if isinstance(level, SpatialLevel):
                utilization *= level.factors.fullProduct()/level.mesh
        return utilization

    """
    Hash unique for each factors allocation and dataflows pair.
    """
    def hashFromFactors(self) -> int:
        hsh = ""
        for level_idx in range(len(self)):
            hsh += f"|{level_idx}"
            for dim in self[level_idx].dataflow:
                hsh += f"{dim}"
                for factor, amount in self[level_idx].factors[dim].items():
                    hsh += f"{factor}{amount}"
        return hash(hsh)

    """
    Reduces constraints to the largest possible ones that can be satisfied
    by the present computations. Returns True if the fitting was possible.
    If enforce is False, in case of usatisfiable constraints an error is
    printed and the return value must be checked for failure. Otherwise,
    an assertion enforces constraints to comply.
    """
    def fitConstraintsToComp(self, comp : Shape, comp_name : Optional[str] = None, enforce : bool = False) -> bool:
        failed = False
        for dim in self.coupling.dims:
            total_constraint = 1
            for level in self:
                # only == and >= constraints may be unsatisfiable at this point, since <= ones are forbidden on the
                # outermost architecture level, therefore factors can always stay there if constraints are too tight!
                if dim + '>=' in level.factors_constraints:
                    eq = '>='
                else:
                    eq = ''
                if dim in level.factors_constraints or dim + '>=' in level.factors_constraints:
                    if comp[dim] // total_constraint < level.factors_constraints[dim + eq]:
                        if comp[dim] // total_constraint <= 0:
                            if enforce:
                                assert False, f"Arch: {self.name} -> Level: {level.name}: Failed to fit comp to arch because the level's constraint on dimension: {dim} ({eq}{level.factors_constraints[dim + eq]}) cannot be satisfied by comp ({dim}: {comp[dim]})!"
                            else:
                                print(f"ERROR: Arch: {self.name} -> Level: {level.name}: failed to fit comp: {comp_name if comp_name else comp} to arch because the level's constraint on dimension: {dim} ({eq}{level.factors_constraints[dim + eq]}) cannot be satisfied by comp ({dim}: {comp[dim]})!")
                            failed = True
                            break
                        print(f"WARNING: Arch: {self.name} -> Level: {level.name}: updating constraint ({dim}: {level.factors_constraints[dim + eq]}) to ({dim}: {comp[dim] // total_constraint}) to fit the computation.")
                        level.factors_constraints[dim + eq] = comp[dim] // total_constraint
                    elif eq == '' and (comp[dim] // total_constraint) % level.factors_constraints[dim] != 0 and not Settings.PADDED_MAPPINGS:
                        assert False, f"Arch: {self.name} -> Level: {level.name}: Failed to fit comp to arch because the level's constraint ({dim}: {level.factors_constraints[dim]}) does not divide comp dimension {dim} ({comp[dim]}) exactly. To compensate, consider setting 'Settings.PADDED_MAPPINGS' to True."
                    total_constraint *= level.factors_constraints[dim + eq]
            if failed:
                break
        return not failed

    """
    Returns the overall estimated area for the architecture (in um^2).
    Returns None if any component is missing an area estimate.
    """
    def totalArea(self, verbose : bool = False) -> float:
        area = 0
        physical_instances = 1
        for level in self:
            if level.area == None:
                if verbose:
                    print(f"WARNING: Arch: {self.name}: None value for area found on level {level.name}, area calculation aborted.")
                return None
            area += level.area*physical_instances
            if isinstance(level, SpatialLevel):
                physical_instances *= level.mesh
        return area

    """
    Returns the current stride for a dimension indexing the input tensor.
    """
    def getInStride(self, dim : str) -> int:
        return self.stride_values[self.coupling.in_strides[dim]] if dim in self.coupling.in_strides else 1

    """
    Returns the current stride for a dimension indexing the weights tensor.
    """
    def getWStride(self, dim : str) -> int:
        return self.stride_values[self.coupling.w_strides[dim]] if dim in self.coupling.w_strides else 1

    """
    Returns the current stride for a dimension indexing the output tensor.
    """
    def getOutStride(self, dim : str) -> int:
        return self.stride_values[self.coupling.out_strides[dim]] if dim in self.coupling.out_strides else 1

    def __repr__(self) -> str:
        return f"<object Arch: name: {self.name}, levels: {super().__repr__()}>"
    
    def __str__(self) -> str:
        return f"{self.name}: [" + ", ".join([level.__str__() for level in self]) + "]"