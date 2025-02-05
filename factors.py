from __future__ import annotations

from itertools import permutations
from functools import reduce
from math import prod

from utils import *

from typing import Optional, Union

"""
Coupling defines the relationship between the dimensions and the tensors
(operands) involved in a compute kernel. Tensors are 3: inputs, weights,
and outputs. The kernel iterates once entirely over each dimension, the
binding determins which operands are affected by such iterations. E.g.:

dims = ['x', 'y', 'z', ...]
# let the kernel be
for x in [0:12]:
  for y in [0:36]:
    for z in [0:16]:
      ...
        Out[<out_coupling>] += W[<w_coupling>] * In[<in_coupling>]

# where a coupling is expanded as
w_coupling = ['X', 'Y']
W[<w_coupling>] -> W[x][y]

Coupling lists directly indicate which dimensions index which tensor, the
order is irrelevant, but maintained for readability.
A round of nested lists is allowed in coupling. The outer list indicates
independently indicized operand dimensions, the inner one indicates
indices that affect the same operand dimension. E.g.:

w_coupling = ['X', ['Y', 'Z']]
# implies an indexing like
W[x][y + z]

NOTE: this resembles a "product of sums" notation.
NOTE: inner lists of one element are equivalent to the element by itself,
      for instance [['X'], ['Y', 'Z']] is the same as ['X', ['Y', 'Z']].
      For practical reasons, every coupling in Coupling is converted to
      the first of the two forms, strictly becoming a list of lists with,
      eventually, a single element.

Strides build on top of the nested list, whenever at least two indices sum
up to indicize an operand, each index can be given a coefficient, a stride,
which by default is 1. A stride is created by binding its name to a dimension:

w_stride = {'Y': 'Ystride', 'Z': 'Zstride'}
# continuing from above, imples an indexing like
W[x][ystride*y + zstride*z]

Constructor arguments:
- dims: list of dimensions used in (iterated in) the kernel
- in/w/out_coupling: list of dimensions indexing each operand
- in/w/out_strides: dictionary binding stride constant names to dimensions
"""
class Coupling:
    def __init__(self, dims : list[str], in_coupling : list[Union[str, list[str]]], w_coupling : list[Union[str, list[str]]], out_coupling : list[Union[str, list[str]]], in_strides : Optional[dict[str, str]] = None, w_strides : Optional[dict[str, str]] = None, out_strides : Optional[dict[str, str]] = None):
        assert all(dims.count(dim) == 1 for dim in dims), f"Invalid coupling: dims ({dims}) must not contain duplicates."
        assert is_two_levels_list(in_coupling) and is_two_levels_list(w_coupling) and is_two_levels_list(out_coupling), f"Invalid coupling: in_coupling ({in_coupling}), w_coupling ({w_coupling}), and out_coupling ({out_coupling}) must be one or two level lists, no more."
        
        # NOTE: due to less memory overhead, less cache misses, more effective prefetching, and lists-specific CPython optimizations, LISTs perform better than SETs and TUPLEs for all hereby data structures!
        # [tested in Python3.13 and up to convolutions, may not hold for larger tensor comprehensions]
        self.dims : list[str] = dims
        self.flat_in_coupling : list[str] = flatten_two_levels_list(in_coupling)
        self.flat_w_coupling : list[str] = flatten_two_levels_list(w_coupling)
        self.flat_out_coupling : list[str] = flatten_two_levels_list(out_coupling)
        # uncoupled dims are permitted, if there is a reason to model the whole kernel running multiple times
        assert all(dim in dims for dim in self.flat_in_coupling), f"Invalid coupling: in_coupling ({in_coupling}) must be a subset of dims ({dims})."
        assert all(dim in dims for dim in self.flat_w_coupling), f"Invalid coupling: w_coupling ({w_coupling}) must be a subset of dims ({dims})."
        assert all(dim in dims for dim in self.flat_out_coupling), f"Invalid coupling: out_coupling ({out_coupling}) must be a subset of dims ({dims})."
        assert all(self.flat_in_coupling.count(dim) == 1 for dim in self.flat_in_coupling), f"Invalid coupling: in_coupling ({in_coupling}) must not use a dimension more than once, not even between sublists."
        assert all(self.flat_w_coupling.count(dim) == 1 for dim in self.flat_w_coupling), f"Invalid coupling: w_coupling ({w_coupling}) must not use a dimension more than once, not even between sublists."
        assert all(self.flat_out_coupling.count(dim) == 1 for dim in self.flat_out_coupling), f"Invalid coupling: out_coupling ({out_coupling}) must not use a dimension more than once, not even between sublists."
        
        self.in_coupling : list[list[str]] = list(map(lambda x : x if isinstance(x, list) else [x], in_coupling))
        self.w_coupling : list[list[str]] = list(map(lambda x : x if isinstance(x, list) else [x], w_coupling))
        self.out_coupling : list[list[str]] = list(map(lambda x : x if isinstance(x, list) else [x], out_coupling))
        
        self.in_strides : dict[str, str] = in_strides if in_strides else {}
        self.w_strides : dict[str, str] = w_strides if w_strides else {}
        self.out_strides : dict[str, str] = out_strides if out_strides else {}
        assert all(dim in self.dims for dim in self.in_strides.keys()), f"Invalid coupling: all keys assigned in in_strides {self.in_strides.keys()} must be dimensions of the coupling ({self.dims})."
        assert all(dim in self.dims for dim in self.w_strides.keys()), f"Invalid coupling: all keys assigned for w_strides {self.w_strides.keys()} must be dimensions of the coupling ({self.dims})."
        assert all(dim in self.dims for dim in self.out_strides.keys()), f"Invalid coupling: all keys assigned for out_strides {self.out_strides.keys()} must be dimensions of the coupling ({self.dims})."
        assert all(dim not in self.dims for dim in self.in_strides.values()), f"Invalid coupling: all values assigned in in_strides {self.in_strides.values()} must not be dimensions of the coupling ({self.dims})."
        assert all(dim not in self.dims for dim in self.w_strides.values()), f"Invalid coupling: all values assigned for w_strides {self.w_strides.values()} must not be dimensions of the coupling ({self.dims})."
        assert all(dim not in self.dims for dim in self.out_strides.values()), f"Invalid coupling: all values assigned for out_strides {self.out_strides.values()} must not be dimensions of the coupling ({self.dims})."
        assert all(any(dim in dim_sum for dim_sum in self.in_coupling if len(dim_sum) > 1) for dim in self.in_strides.keys()), f"Invalid coupling: all dimensions referenced in in_strides {self.in_strides.keys()} must be part of a sum of at least two indices on the input (thus pick among {list(reduce(lambda x, y : x+y, filter(lambda dim_sum : len(dim_sum) > 1, self.in_coupling), []))})"
        assert all(any(dim in dim_sum for dim_sum in self.w_coupling if len(dim_sum) > 1) for dim in self.w_strides.keys()), f"Invalid coupling: all dimensions referenced in w_strides {self.w_strides.keys()} must be part of a sum of at least two indices on the weight (thus pick among {list(reduce(lambda x, y : x+y, filter(lambda dim_sum : len(dim_sum) > 1, self.w_coupling), []))})"
        assert all(any(dim in dim_sum for dim_sum in self.out_coupling if len(dim_sum) > 1) for dim in self.out_strides.keys()), f"Invalid coupling: all dimensions referenced in out_strides {self.out_strides.keys()} must be part of a sum of at least two indices on the output (thus pick among {list(reduce(lambda x, y : x+y, filter(lambda dim_sum : len(dim_sum) > 1, self.out_coupling), []))})"
        
        if any(len(dim_sum) > 2 and any(dim in self.in_strides for dim in dim_sum) for dim_sum in self.in_coupling): print(f"WARNING: coupling {self} has strides applied to a sum of more than 2 indices on its input tensor ({filter(lambda dim_sum: len(dim_sum) > 2 and any(dim in self.in_strides for dim in dim_sum), self.in_coupling)}), as a result a less-efficient enumeration algorithm will be used to compute the distinct values originating by such strided sums instead of the exact expression for distinct values which is available only for strided sums of maximum 2 indices.")
        if any(len(dim_sum) > 2 and any(dim in self.w_strides for dim in dim_sum) for dim_sum in self.w_coupling): print(f"WARNING: coupling {self} has strides applied to a sum of more than 2 indices on its weights tensor ({filter(lambda dim_sum: len(dim_sum) > 2 and any(dim in self.w_strides for dim in dim_sum), self.w_coupling)}), as a result a less-efficient enumeration algorithm will be used to compute the distinct values originating by such strided sums instead of the exact expression for distinct values which is available only for strided sums of maximum 2 indices.")
        if any(len(dim_sum) > 2 and any(dim in self.out_strides for dim in dim_sum) for dim_sum in self.out_coupling): print(f"WARNING: coupling {self} has strides applied to a sum of more than 2 indices on its output tensor ({filter(lambda dim_sum: len(dim_sum) > 2 and any(dim in self.out_strides for dim in dim_sum), self.out_coupling)}), as a result a less-efficient enumeration algorithm will be used to compute the distinct values originating by such strided sums instead of the exact expression for distinct values which is available only for strided sums of maximum 2 indices.")

    """
    If True, the provided comp's is valid for the present coupling.
    This means that all required dimensions are specified.
    """
    def isCompatibleComp(self, comp: Shape) -> bool:
        return (all(dim in comp for dim in self.dims) and
                all(dim in comp for dim in self.in_strides.values()) and
                all(dim in comp for dim in self.w_strides.values()) and
                all(dim in comp for dim in self.out_strides.values()))
    
    """
    If True, the current and provided coupling are compatible with the same
    architectural constraints and mappings. However, 'coupling' does not
    necessarily model a subset of the kernels modeled by the current coupling.
    """
    def isCompatibleCoupling(self, coupling: Coupling) -> bool:
        return (all(dim in self.dims for dim in coupling.dims) and
                all(dim in self.flat_in_coupling for dim in coupling.flat_in_coupling) and
                all(dim in self.flat_w_coupling for dim in coupling.flat_w_coupling) and
                all(dim in self.flat_out_coupling for dim in coupling.flat_out_coupling) and
                all(dim in self.in_strides and self.in_strides[dim] == stride for dim, stride in coupling.in_strides.items()) and
                all(dim in self.w_strides and self.w_strides[dim] == stride for dim, stride in coupling.w_strides.items()) and
                all(dim in self.out_strides and self.out_strides[dim] == stride for dim, stride in coupling.out_strides.items()))
    
    """
    If True, the provided coupling is equivalent to the current one without some
    dimensions (that is the same as those being of size 1). Therefore, 'coupling'
    can model a subset of the kernels modeled by the current coupling.
    """
    def isSubcoupling(self, coupling : Coupling) -> bool:
        return (self.isCompatibleCoupling(coupling) and
                len(self.in_coupling) == len(coupling.in_coupling) and len(self.w_coupling) == len(coupling.w_coupling) and len(self.out_coupling) == len(coupling.out_coupling) and
                any(all(set(dim_sum) <= set(self_dim_sum) for dim_sum, self_dim_sum in zip(coupling.in_coupling, perm)) for perm in permutations(self.in_coupling, len(coupling.in_coupling))) and
                any(all(set(dim_sum) <= set(self_dim_sum) for dim_sum, self_dim_sum in zip(coupling.w_coupling, perm)) for perm in permutations(self.w_coupling, len(coupling.w_coupling))) and
                any(all(set(dim_sum) <= set(self_dim_sum) for dim_sum, self_dim_sum in zip(coupling.out_coupling, perm)) for perm in permutations(self.out_coupling, len(coupling.out_coupling))))
    
    """
    Returns the list of indices that sum up to indicize 'operand' of which 'dim'
    is a part of. The list must be at least of length 'min_lenght', otherwise
    None is returned.
    Valid operand names are: 'in', 'w', and 'out'.
    """
    def getDimSum(self, operand : str, dim : str, min_lenght : int = 1) -> Optional[list[str]]:
        if operand == 'in':
            return next((dim_sum for dim_sum in self.in_coupling if dim in dim_sum and len(dim_sum) >= min_lenght), None)
        elif operand == 'w':
            return next((dim_sum for dim_sum in self.w_coupling if dim in dim_sum and len(dim_sum) >= min_lenght), None)
        elif operand == 'out':
            return next((dim_sum for dim_sum in self.out_coupling if dim in dim_sum and len(dim_sum) >= min_lenght), None)
        else:
            raise Exception(f"Unrecognized operand ({operand}) in coupling.")
    
    """
    Returns the flat coupling list for the provided operand.
    Valid operand names are: 'in', 'w', and 'out'.
    """
    def flatCouplingByOperand(self, operand : str) -> list[str]:
        if operand == 'in':
            return self.flat_in_coupling
        elif operand == 'w':
            return self.flat_w_coupling
        elif operand == 'out':
            return self.flat_out_coupling
        else:
            raise Exception(f"Unrecognized operand ({operand}) in coupling.")
    
    """
    Returns a compact string representing the coupling.
    """
    def compactStr(self) -> str:
        def coup2str(coupling, strides):
            return '[' + ']['.join(map(lambda dim_sum : '+'.join(map(lambda dim : strides[dim] + '*' + dim if dim in strides else dim, dim_sum)) if len(dim_sum) > 1 else dim_sum[0], coupling)) + ']'
        return f"dims: {''.join(self.dims)}, in_coupling: {coup2str(self.in_coupling, self.in_strides)}, w_coupling: {coup2str(self.w_coupling, self.w_strides)}, out_coupling: {coup2str(self.out_coupling, self.out_strides)}"
    
    def __str__(self) -> str:
        return "{" + f"dims: {self.dims}, in_coupling: {self.in_coupling}, w_coupling: {self.w_coupling}, out_coupling: {self.out_coupling}, in_strides: {self.in_strides}, w_strides: {self.w_strides}, out_strides: {self.out_strides}" + "}"

"""
Shape of a kernel in the form Out = W x In.
Where 'x' is a MAC-based linear algebra operation.
In other words, this class models the size of the kernel's dimensions.
"""
class Shape(dict[str, int]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def memFootprint(self, coupling : Coupling) -> int:
        return (prod(sum(self[dim] for dim in dim_sum) - len(dim_sum) + 1 for dim_sum in coupling.in_coupling) +
                prod(sum(self[dim] for dim in dim_sum) - len(dim_sum) + 1 for dim_sum in coupling.w_coupling) +
                prod(sum(self[dim] for dim in dim_sum) - len(dim_sum) + 1 for dim_sum in coupling.out_coupling))

    def FLOPs(self) -> int:
        return 2*prod(self.values())

    def fitToCoupling(self, coupling : Coupling) -> None:
        for dim in coupling.dims:
            if dim not in self:
                self[dim] = 1

    def clear(self) -> None:
        for dim in self.keys():
            self[dim] = 1

    def __str__(self) -> str:
        return "{" + ", ".join(f"{k}: {v}" for k, v in self.items()) + "}"

"""
Factors constituting the number of iterations unfolding over all
dimensions of any level of the architecture.

All dimensions are represented by the outer dictionary, which
associates to every prime factor (key) the number of times it
occurs (value) over that dimension.

Constructor:
- if a list of dimensions is provided, the class is initialized
  with no factors assigned to each of them.
- if a dictionary is provided, that is the instance's initial content,
  and shall be in the form "{dim: {prime_factor: arity, ...}, ...}".
- if arbitrary arguments are provided, each of them becomes a dimension
  and its value shall be a prime_factor->arity dictionary as above.
"""
class Factors(dict[str, dict[int, int]]):
    def __init__(self, iterable : Union[list[str], tuple[str], dict[str, dict[int, int]]] = None, *args, **kwargs):
        if isinstance(iterable, list) or isinstance(iterable, tuple):
            super().__init__()
            for dim in iterable:
                self[dim] = {}
        elif isinstance(iterable, dict):
            super().__init__(iterable)
        else:
            super().__init__(*args, **kwargs)
        
        self._dim_products : dict[str, int] = {k: reduce(lambda tot, f_a : tot*(f_a[0]**f_a[1]), v.items(), 1) for k, v in self.items()}

    """
    Add "amount" instances of the provided factor to those of
    "dimension" in the current set of factors.
    """
    def addFactor(self, dimension : str, factor : int, amount : int) -> None:
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
    def removeFactor(self, dimension : str, factor : int, amount : int) -> bool:
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
    def dimProduct(self, dimension : str) -> int:
        return self._dim_products[dimension]

    """Total number of iterations across all three dimensions."""
    def fullProduct(self) -> int:
        return prod(self._dim_products.values())

    """
    Recomputes the correct values for the dimProducts as of the current
    factors. Must be called any time factors are updated directly, without
    going through the addFactor and removeFactor methods.

    Pass dimensions as an array of dimensions to only reset indicated ones.
    """
    def resetDimProducts(self, dimensions : Optional[str] = None) -> None:
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
    def isSubset(self, subset : Factors) -> bool:
        for dim in subset.keys():
            for k, v in subset[dim].items():
                if k not in self[dim] or v > self[dim][k]:
                    return False
        return True

    """
    Give the tile sizes involved across iterations and, optionally,
    which operands are bypassed, returns the amount of memory required
    to store all tiles needed by all iterations.
    (It is assumed that a level always stores all data for the
    iterations unfolding over it)
    """
    def memFootprint(self, tile_sizes : Shape, coupling : Coupling, in_bp : bool = 1, w_bp : bool = 1, out_bp : bool = 1) -> int:
        return (prod(sum(tile_sizes[dim]*self._dim_products[dim] for dim in dim_sum) - len(dim_sum) + 1 for dim_sum in coupling.in_coupling)*in_bp +
                prod(sum(tile_sizes[dim]*self._dim_products[dim] for dim in dim_sum) - len(dim_sum) + 1 for dim_sum in coupling.w_coupling)*w_bp +
                prod(sum(tile_sizes[dim]*self._dim_products[dim] for dim in dim_sum) - len(dim_sum) + 1 for dim_sum in coupling.out_coupling)*out_bp)

    """
    Returns the factors present on the specified dimension as a list rather
    than as a dictionary. Factors multiplicity in the list reflects arity.
    """
    def toList(self, dimension : str) -> list[int]:
        return [k for k in self[dimension] for _ in range(self[dimension][k])]

    """
    Reset this "Factors" instance to no factors along any dimension.
    """
    def clear(self) -> None:
        for k in self.keys():
            self[k].clear()
            self._dim_products[k] = 1

    def __str__(self) -> str:
        return "{" + ", ".join(f"{k}: {v}" for k, v in self.items()) + "}"