from functools import reduce
from enum import Enum

"""
Possible dataflows for GEMMs.
They can be recognized according to the inner-most loop,
ignoring those with a single iteration:
- innermost N -> WS
- innermost K -> OS
- innermost M -> IS
"""
class Dataflow(Enum):
    WS = 0
    OS = 1
    IS = 2

"""
Shape of the Matrix Multiplication in the form Out = W*In.
Dimensions are:
- M = W and Out's height
- K = Inner dimension, W's width and In's height
- N = In and Out's width
"""
class Shape():
    def __init__(self, M, K, N):
        self.M = M # Weight/Out rows
        self.K = K # Inner dimension, Weight cols/In rows
        self.N = N # In/Out cols

    def mem_footprint(self):
        return self.M*self.K + self.K*self.N + self.M*self.N

    def FLOPs(self):
        return 2*self.M*self.K*self.N

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def clear(self):
        self.M = 1
        self.K = 1
        self.N = 1

    def __str__(self):
        return f"{{M: {self.M}, K: {self.K}, N: {self.N}}}"

"""
Factors constituting the number of iterations unfolding over all
dimensions of any level of the architecture.

All dimensions are represented by a dictionary, which associates
to every prime factor (key) the number of times it occurs (value)
over that dimension.
"""
class Factors():
    def __init__(self, M = None, K = None, N = None):
        self.M = M if M else {}
        self.K = K if K else {}
        self.N = N if N else {}
        self._dim_products = {
            'M': reduce(lambda tot, f_a : tot*(f_a[0]**f_a[1]), self.M.items(), 1),
            'K': reduce(lambda tot, f_a : tot*(f_a[0]**f_a[1]), self.K.items(), 1),
            'N': reduce(lambda tot, f_a : tot*(f_a[0]**f_a[1]), self.N.items(), 1)
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
        return self._dim_products['M']*self._dim_products['K']*self._dim_products['N']

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
        for k, v in subset.M.items():
            if k not in self.M or v > self.M[k]:
                return False
        for k, v in subset.K.items():
            if k not in self.K or v > self.K[k]:
                return False
        for k, v in subset.N.items():
            if k not in self.N or v > self.N[k]:
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
        return (self.dimProduct('M')*self.dimProduct('K')*w_bp*tile_sizes.M*tile_sizes.K +
                self.dimProduct('K')*self.dimProduct('N')*in_bp*tile_sizes.K*tile_sizes.N +
                self.dimProduct('M')*self.dimProduct('N')*out_bp*tile_sizes.M*tile_sizes.N)

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
        self.M.clear()
        self.K.clear()
        self.N.clear()
        self._dim_products = {'M': 1, 'K': 1, 'N': 1}

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)