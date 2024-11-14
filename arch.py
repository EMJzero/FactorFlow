from copy import deepcopy

from levels import *
from settings import *
from utils import *

"""
Class wrapping a list of levels into an architecture.

Constructor arguments:
- iterable: the list of levels for the architecture
- name: the name for the architecture
"""
class Arch(list):
    def __init__(self, iterable=None, name="<unnamed architecture>"):
        self.name = name
        
        if iterable is None:
            iterable = []
        super().__init__(iterable)

        assert len(self) >= 2, f"Arch: {self.name}: at least two levels (one 'MemLevel', one 'ComputeLevel') are required for an architecture, {len(self)} provided."
        assert all(isinstance(item, Level) for item in self), f"Arch: {self.name}: all architecture entries must be levels (instances of the 'Level' class), provided ones are {list(map(lambda x : type(x).__name__, self))}."
        assert isinstance(self[0], MemLevel), f"Arch: {self.name}: the outermost (idx 0) level must be a memory level ('MemLevel' instance), {type(self[0]).__name__} provided."
        assert isinstance(self[-1], ComputeLevel), f"Arch: {self.name}: the innermost (idx len-1) level must be a compute level ('ComputeLevel' instance), {type(self[-1]).__name__} provided."
        assert not any(constr[1:] == '<=' for constr in self[0].factors_constraints.keys()), f"Arch: {self.name}: the outermost (idx 0) level's constraints ({self[0].factors_constraints}) must never be of the '<=' kind."

    """
    Returns a deep-copied compact representation of the current mapping.
    """
    def exportMapping(self):
        return [LevelCore(deepcopy(level.dataflow), deepcopy(level.factors), deepcopy(level.tile_sizes)) for level in self]

    """
    Loads a mapping (preferably coming from "exportMapping") into the architecture.
    """
    def importMapping(self, mapping : list[LevelCore]):
        for i in range(len(self)):
            self[i].dataflow = mapping[i].dataflow
            self[i].factors = mapping[i].factors
            self[i].tile_sizes = mapping[i].tile_sizes
            
    """
    Checks factors allocation constraints. Returns False if a violation is found.
    """
    def findConstraintsViolation(self, verbose = True):
        violation = False
        for level in self:
            for dim in ['M', 'K', 'N']:
                if dim not in level.dataflow and len(level.factors[dim]) != 0:
                    if verbose: print(f"Factor-dataflow missmatch for constraint ({dim}: {level.factors.dimProduct(dim)}) enforced on level \"{level.name}\"")
                    violation = True
                if dim in level.factors_constraints and level.factors_constraints[dim] != level.factors.dimProduct(dim):
                    if verbose: print(f"Constraint violation, desired was ({dim}: {level.factors_constraints[dim]}), obtained was ({dim}: {level.factors.dimProduct(dim)}), on level \"{level.name}\"")
                    violation = True
        return violation

    """
    Moves a factor between the same dimension of two levels, transitioning
    between adjacent mappings. it also updates tile sizes accordingly for all levels
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
    def moveFactor(self, src_level_idx, dst_level_idx, dimension, factor, amount = 1, skip_src_constraints = False, skip_dst_constraints = False):
        # check that the factor exists in the required amount
        if not self[src_level_idx].removeFactor(dimension, factor, amount):
            return False
        # check src constraints
        if not self[src_level_idx].checkConstraints() and not skip_src_constraints:
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
                self[i].tile_sizes[dimension] /= factor_to_amount
        # check dst and all in between constraints
        constraints_check = [level.checkConstraints() for level in (self[src_level_idx:dst_level_idx+1] if src_level_idx < dst_level_idx else self[dst_level_idx:src_level_idx+1])]
        if not all(constraints_check) and not skip_dst_constraints:
            self[src_level_idx].addFactor(dimension, factor, amount)
            assert self[dst_level_idx].removeFactor(dimension, factor, amount) # something is broken, cannot undo the move
            if src_level_idx < dst_level_idx:
                for i in range(src_level_idx, dst_level_idx):
                    self[i].tile_sizes[dimension] /= factor_to_amount
            elif src_level_idx > dst_level_idx:
                for i in range(dst_level_idx, src_level_idx):
                    self[i].tile_sizes[dimension] *= factor_to_amount
            return False
        return True

    """
    Initializes the architecture's mapping with all the computation's prime
    factors placed on the first level. This is the mapper's starting point.
    """
    def initFactors(self, comp):
        # initialize with all factors on first level, all tile sizes of 1!
        self[0].factors = Factors(M = prime_factors(comp.M), K = prime_factors(comp.K), N = prime_factors(comp.N))

    """
    This function must start from arch having all factors on its first level,
    then it ensures that all constraints of all levels are satisfied.
    If 'allow_padding' is True then computation dimensions not exactly
    divided by constraints will be padded up to a multiple of constraints.

    This function assumes that the computation is larger than what required
    by constraints, if this is not satisfied, an assertion will be triggered.
    -> use 'fitConstraintsToComp' to prevent such a situation.
    """
    def enforceFactorsConstraints(self, allow_padding = False, verbose_padding = True):
        # assuming that initially all factors are on the first level
        for i in range(1, len(self)):
            level = self[i]
            for k in level.factors_constraints.keys():
                assert k[0] in level.dataflow, f"Level {level.name}: cannot enforce constraint on dimension {k[0]} that is not in dataflow {level.dataflow}."
            # initialize only == and >= constraints, leave <= set to 1
            constr_factors = Factors(
                M = prime_factors(level.factors_constraints['M']) if 'M' in level.factors_constraints else (prime_factors(level.factors_constraints['M>=']) if 'M>=' in level.factors_constraints else {}),
                K = prime_factors(level.factors_constraints['K']) if 'K' in level.factors_constraints else (prime_factors(level.factors_constraints['K>=']) if 'K>=' in level.factors_constraints else {}),
                N = prime_factors(level.factors_constraints['N']) if 'N' in level.factors_constraints else (prime_factors(level.factors_constraints['N>=']) if 'N>=' in level.factors_constraints else {}),
                )
            if self[0].factors.isSubset(constr_factors) or allow_padding:
                for dim in ['M', 'K', 'N']:
                    dim_size = self[0].factors.dimProduct(dim)
                    constraint = constr_factors.dimProduct(dim)
                    if dim_size%constraint == 0:
                        for fact, amount in constr_factors[dim].items():
                            assert self.moveFactor(0, i, dim, fact, amount, True, True), f"Constraints are not satisfiable! Cannot give {amount} instances of prime factor {fact} to level {level.name} on dimension {dim}."
                    else:
                        padded_dim_size = dim_size + constraint - dim_size%constraint
                        if verbose_padding: print(f"PADDING: enlarged {dim} from {dim_size} to {padded_dim_size} to respect constraints on level {level.name}")
                        self[0].factors[dim] = prime_factors(padded_dim_size)
                        self[0].factors.resetDimProducts([dim])
                        for fact, amount in constr_factors[dim].items():
                            # be wary that here we skip constraints checks in moveFactor, so one must follow up this method with findConstraintsViolation
                            assert self.moveFactor(0, i, dim, fact, amount, True, True), "Failed to enforce constraints even with padding..."

    """
    Checks dataflow (loop ordering) constraints. Returns False if a violation is found.
    """
    def checkDataflowConstraints(self):
        for level in filter(lambda l : isinstance(l, MemLevel), self):
            dim_idx = 0
            for dim in level.dataflow_constraints:
                if dim_idx == len(level.dataflow):
                    return False
                while dim_idx < len(level.dataflow):
                    if dim == level.dataflow[dim_idx]:
                        break
                    dim_idx += 1
        return True

    """
    Initializes bypasses between MemLevels of the architecture.
    For each operand and bypass, setup a pointer in the last MemLevel before the bypass
    to the list of levels affected by the bypass.

    This way such last level can compute its MOPs according to the level it actually
    supplies data to.
    """
    def setupBypasses(self):
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
    Updates the count of ACTIVE instances throughout Mem- and Compute- Levels.
    An instance is ACTIVE if a FanoutLevel maps a spatial iteration to it.
    """
    def updateInstances(self):
        spatial_fanout = 1
        for i in range(len(self)):
            level = self[i]
            if isinstance(level, FanoutLevel):
                # consider only factors -> only actually used instances
                spatial_fanout *= level.factors.fullProduct()
            elif isinstance(level, MemLevel):
                level.instances = spatial_fanout
                level.next_is_compute = isinstance(self[i+1], ComputeLevel) if i+1 < len(self) else False
            elif isinstance(level, ComputeLevel):
                level.instances = spatial_fanout*level.factors.fullProduct()

    """
    Clears the accumulators for incrementally updated tile sizes and
    prime factors products in the architecture.
    """
    def resetTilesAndFactors(self):
        for level in self:
            level.factors.clear()
            for dim in ['M', 'K', 'N']:
                level.tile_sizes[dim] = 1

    """
    Returns the overall utilization of spatial instances of a mapping.
    """
    def spatialUtilization(self):
        utilization = 1
        for level in self:
            if isinstance(level, SpatialLevel):
                utilization *= level.factors.fullProduct()/level.mesh
        return utilization

    """
    Hash unique for each factors allocation and dataflows pair.
    """
    def hashFromFactors(self):
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
    by the present computations.

    If enforce is False, in case of usatisfiable constraints an error is
    printed and the return value must be checked for failure. Otherwise,
    an assertion enforces constraints to comply.
    """
    def fitConstraintsToComp(self, comp, comp_name = None, enforce = False):
        failed = False
        for dim in ['M', 'K', 'N']:
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
                            if self.name and comp_name:
                                print(f"ERROR: failed to fit comp: {comp_name if comp_name else comp} to arch: {self.name} because the constraint on level: {level.name} and dimension: {dim} ({eq}{level.factors_constraints[dim + eq]}) cannot be satisfied by comp ({dim}: {comp[dim]})!")
                            else:
                                assert False, f"Failed to fit comp to arch because the constraint on level: {level.name} and dimension: {dim} ({eq}{level.factors_constraints[dim + eq]}) cannot be satisfied by comp ({dim}: {comp[dim]})!"
                            failed = True
                            break
                        print(f"WARNING: updating constraint ({dim}: {level.factors_constraints[dim + eq]}) on level \"{level.name}\" to ({dim}: {comp[dim] // total_constraint}) to fit the computation.")
                        level.factors_constraints[dim + eq] = comp[dim] // total_constraint
                    elif eq == '' and (comp[dim] // total_constraint) % level.factors_constraints[dim] != 0 and not Settings.PADDED_MAPPINGS:
                        assert False, f"Failed to fit comp to arch because the constraint on level {level.name} ({dim}: {level.factors_constraints[dim]}) does not divide comp dimension {dim} ({comp[dim]}) exactly. To compensate, consider setting 'Settings.PADDED_MAPPINGS' to True."
                    total_constraint *= level.factors_constraints[dim + eq]
            if failed:
                break
        return failed

    """
    Returns the overall estimated area for the architecture (in um^2).
    Returns None if any component is missing an area estimate.
    """
    def totalArea(self, verbose = False):
        area = 0
        physical_instances = 1
        for level in self:
            if level.area == None:
                if verbose:
                    print(f"WARNING: None value for area found on level {level.name}, area calculation aborted.")
                return None
            area += level.area*physical_instances
            if isinstance(level, SpatialLevel):
                physical_instances *= level.mesh
        return area

    def __repr__(self):
        return f"<object Arch: name: {self.name}, levels: {super().__repr__()}>"
    
    def __str__(self):
        return f"{self.name}: [" + ", ".join([level.__str__() for level in self]) + "]"