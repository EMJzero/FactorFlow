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

"""
Dictionary wrapper which keeps a valid flag updated w.r.t.
changes to its content, the flag is reset to False every time
that the dictionary is updated, and must be set manually
by invoking setValid(). Check validity with isValid() instead.
"""
class LazyDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._valid = False
    
    def __setitem__(self, key, value):
        if key not in self or self[key] != value:
            self._valid = False
        super().__setitem__(key, value)

    def isValid(self):
        return self._valid

    def setValid(self):
        self._valid = True

"""
Factors constituting the number of iterations unfolding over all
dimensions of any level of the architecture.

All dimensions are represented by a dictionary, which associates
to every prime factor (key) the number of times it occurs (value)
over that dimension.

The class is lazy, it does not re-evaluate the full product of
prime factos along a dimension unless such dimension is altered.
"""
class Factors():
    def __init__(self, D = None, E = None, L = None):
        self.D = LazyDict(D) if D else LazyDict()
        self.E = LazyDict(E) if E else LazyDict()
        self.L = LazyDict(L) if L else LazyDict()
        self._dim_products = {'D': 1, 'E': 1, 'L': 1}

    """
    Product of all prime factors along a dimension, equivalent
    to the actual number of iterations along said dimension.
    """
    def dimProduct(self, dimension):
        if self[dimension].isValid():
            return self._dim_products[dimension]
        res = 1
        for f, v in self[dimension].items():
            res *= f**v
        self._dim_products[dimension] = res
        self[dimension].setValid()
        return res

    """Total number of iterations across all three dimensions."""
    def fullProduct(self):
        return self.dimProduct('D')*self.dimProduct('E')*self.dimProduct('L')

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

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)