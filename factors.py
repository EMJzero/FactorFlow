from functools import reduce
from enum import Enum

"""
Possible dataflows for GEMMs.
They can be recognized according to the inner-most loop,
ignoring those with a single iteration:
- innermost L -> WS
- innermost E -> OS
- innermost D -> IS
"""
class Dataflow(Enum):
    WS = 0
    OS = 1
    IS = 2

"""
Shape of the Matrix Multiplication in the form Out = W*In.
Dimensions are:
- D = W and Out's height
- E = Inner dimension, W's width and In's height
- L = In and Out's width
"""
class Shape():
    def __init__(self, D, E, L):
        self.D = D # Weight/Out rows
        self.E = E # Inner dimension, Weight cols/In rows
        self.L = L # In/Out cols

    def mem_footprint(self):
        return self.D*self.E + self.E*self.L + self.D*self.L

    def FLOPs(self):
        return 2*self.D*self.E*self.L

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
        
    def __str__(self):
        return f"{{D: {self.D}, E: {self.E}, L: {self.L}}}"

"""
Factors constituting the number of iterations unfolding over all
dimensions of any level of the architecture.

All dimensions are represented by a dictionary, which associates
to every prime factor (key) the number of times it occurs (value)
over that dimension.
"""
class Factors():
    def __init__(self, D = None, E = None, L = None):
        self.D = D if D else {}
        self.E = E if E else {}
        self.L = L if L else {}
        self._dim_products = {
            'D': reduce(lambda tot, f_a : tot*(f_a[0]**f_a[1]), self.D.items(), 1),
            'E': reduce(lambda tot, f_a : tot*(f_a[0]**f_a[1]), self.E.items(), 1),
            'L': reduce(lambda tot, f_a : tot*(f_a[0]**f_a[1]), self.L.items(), 1)
            }

    """
    Add "amount" instances of the provided factor to those of
    "dimension" in the current set of factors.
    """
    def addFactor(self, dimension, factor, amount):
        if factor in self[dimension]:
            self[dimension][factor] += amount
        else:
            self[dimension][factor] = amount
        self._dim_products[dimension] *= factor**amount

    """
    Remove "amount" instances of the provided factor from those of
    "dimension" in the current set of factors.
    Return False if the removal failed because the current factors do
    not have at least "amount" instances of "factor" along "dimension".
    """
    def removeFactor(self, dimension, factor, amount):
        if factor not in self[dimension] or self[dimension][factor] < amount:
            return False
        self[dimension][factor] -= amount
        if self[dimension][factor] == 0:
            self[dimension].pop(factor)
        self._dim_products[dimension] //= factor**amount
        return True

    """
    Product of all prime factors along a dimension, equivalent
    to the actual number of iterations along said dimension.
    """
    def dimProduct(self, dimension):
        return self._dim_products[dimension]

    """Total number of iterations across all three dimensions."""
    def fullProduct(self):
        return self._dim_products['D']*self._dim_products['E']*self._dim_products['L']

    """
    Recomputes the correct values for the dimProducts as of the current
    factors. Must be called any time factors are updated directly, without
    going through the addFactor and removeFactor methods.
    
    Pass dimensions as an array of dimensions to only reset indicated ones.
    """
    def resetDimProducts(self, dimensions = None):
        dimensions = dimensions if dimensions else self._dim_products.keys()
        for dim in dimensions:
            res = 1
            for f, v in self[dim].items():
                res *= f**v
            self._dim_products[dim] = res
    
    """
    Checks whether "subset" has less or the same occurencies of prime
    factors along each dimension (independently) as this instance.
    False if "subset" has an additional factor or more occurrencies
    of one along any of the three dimensions.
    """
    def isSubset(self, subset):
        for k, v in subset.D.items():
            if k not in self.D or v > self.D[k]:
                return False
        for k, v in subset.E.items():
            if k not in self.E or v > self.E[k]:
                return False
        for k, v in subset.L.items():
            if k not in self.L or v > self.L[k]:
                return False
        return True

    """
    Give the tile sizes involved across iterations and, optionally,
    which operands are bypassed, returns the amount of memory required
    to store all tiles needed by all iterations.
    (It is assumed that a level always stores all data for the
    iterations unfolding over it)
    """
    def mem_footprint(self, tile_sizes, in_bp = 1, w_bp = 1, out_bp = 1):
        return (self.dimProduct('D')*self.dimProduct('E')*w_bp*tile_sizes.D*tile_sizes.E +
                self.dimProduct('E')*self.dimProduct('L')*in_bp*tile_sizes.E*tile_sizes.L +
                self.dimProduct('D')*self.dimProduct('L')*out_bp*tile_sizes.D*tile_sizes.L)

    """
    Returns the factors present on the specified dimension as a list rather
    than as a dictionary. Factors multiplicity in the list reflects arity.
    """
    def toList(self, dimension):
        return [k for k in self[dimension] for _ in range(self[dimension][k])]

    """
    Reset this "Factors" instance to no factors along any dimension.
    """
    def clear(self):
        self.D.clear()
        self.E.clear()
        self.L.clear()
        self._dim_products = {'D': 1, 'E': 1, 'L': 1}

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)