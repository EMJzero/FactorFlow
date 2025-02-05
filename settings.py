import os

class Settings():
    # If True, enables logging of the MSE process. Note that such prints occur during the timed
    # section of the program, set to False for accurate timing results.
    VERBOSE = True

    # If False, FF only searches for better solutions at a one-factor distance from the current one,
    # if True, FF searches for solutions at a distance of multiple factors, all be it only arity is
    # varied, with the tried factor being just one. (tries different multiplicities)
    ITERATE_AMOUNTS = False
    # If True, factors allocated on fanout levels will not be optimized (as if they were constraints),
    # and this is done after factor allocation in the fanouts is maximized.
    # NOTE: automatically set to False in case of 2 dimensions on the same fanout.
    FREEZE_SA = True
    # Number of one-factor steps to try during local search after of which the best choice is picked.
    # NOTE: automatically raised to (at least) 2 in case of 2 dimensions on the same fanout.
    STEPS_TO_EXPLORE = 1
    # If True, any recursively explored step after the first one, will only attempt to move factors
    # into the destination level which was the source for the previous move.
    # NOTE: automatically set to True in case of 2 dimensions on the same fanout.
    LIMIT_NEXT_STEP_DST_TO_CURRENT_SRC = False
    # If True, any intermediate step of a multi-step exploration will not need to satisfy architectural
    # constraints, on the condition that the final step will satisfy them.
    # NOTE: can be True iif LIMIT_NEXT_STEP_DST_TO_CURRENT_SRC is True
    # NOTE: automatically set to True in case of 2 dimensions on the same fanout.
    NO_CONSTRAINTS_CHECK_DURING_MULTISTEP = LIMIT_NEXT_STEP_DST_TO_CURRENT_SRC and False
    # If True, in case of 2 dimensions on the same fanout only the FIRST dimension gets factors
    # allocation during fanout maximization. If False, both dimensions equally get factors.
    # NOTE: when this is True, optimizeDataflows also iterates over which dimension is maximized,
    #       in other words also fanout dimensions are permutated to pick the one to maximize.
    #       [Dimensions with a constraint are not iterated over]
    # >>> Play with this in case of 2 dimensions on the same fanout!!!
    # >>> Setting this to True costs Nx time, where N is the number of rotations of fanout dimensions.
    # >>> Henceforth, usage is suggested when MULTITHREADED is True.
    ONLY_MAXIMIZE_ONE_FANOUT_DIM = True
    # If True, saves time by assuming that any permutation differing from an optimal one by the order
    # of dimensions involving one with a single iteration can be optimized starting from where it
    # already is, and thus avoids a complete re-initialization.
    # NOTE: True is highly recommended to avoid prohibitive MSE times.
    PERM_SKIP = True
    # If True, saves time by assuming that any permutation differing from an optimal one by the order
    # of dimensions involving one with a single iteration cannot be optimized further, thus skips
    # it entirely. Setting it to False can slightly improve the best found mapping.
    # NOTE: unless PERM_SKIP is True, this setting is useless.
    HARD_PERM_SKIP = False
    # If True, the list of permutations to explore on each level is filtered to keep a single permutation
    # for each set of permutations that could reach the same reuse patterns (hence, those having the same
    # innermost iterated dimension coupled to each operand and the same dimensions above and below it)
    DISTINCT_REUSE_OPPORTUNITIES = True
    # If True, rules are built based on the best selected permutations throughout the exploration,
    # and these rules are subsequently enforced to prune the remaining permutations, speeding up
    # the exploration under the assumption that there is some consistency between optimal choices.
    PERM_PRUNING = False
    # When PERM_PRUNING is True, the following 3 settings determine the number of times a dimension
    # needs to (1) have a single iteration, (2) be in a certain relative order with another, and
    # (3) occupy a specific position in the ordering of the best mapping found for a subset of
    # permutations, before such occurrance becomes a rule that is enforced to prune later permutations.
    DIM_AT_1_COUNT_BEFORE_LOCK = 2
    RELATIVE_ORDER_COUNT_BEFORE_LOCK = 2
    POSITIONAL_COUNT_BEFORE_LOCK = 2
    # If True, the Wart will be multiplied by the utilization of the fanouts in the spatial architecture,
    # punishing mappings which underutilize fanouts.
    UTILIZATION_IN_WART = True
    # If True, drain reads will be assumed at zero energy cost for all levels, this is equivalent to
    # assuming that the last write bypasses the target level, going upward, and directly writes in the
    # above level, thus negating the need for a read to drain. As a result, drains also don't contribute
    # to the final latency and level stalls anymore.
    # NOTE: setting this to True is required to match Timeloop's results exactly.
    FREE_DRAINS = False
    # If True, GEMM dimensions might get padded to reach the least larger-than-current size which can
    # be allocated to the entirety of a fanout's instances.
    # This is performed as part of the fanout maximization.
    # NOTE: this is useless unless ONLY_MAXIMIZE_ONE_FANOUT_DIM is True.
    PADDED_MAPPINGS = ONLY_MAXIMIZE_ONE_FANOUT_DIM and False
    # If True, any padding applied due to PADDED_MAPPINGS will be logged.
    # NOTE: this is useless unless PADDED_MAPPINGS is True.
    VERBOSE_PADDED_MAPPINGS = PADDED_MAPPINGS and False
    # If True, the 'distinct_values' method in 'utils.py' uses an approximated formula to compute the
    # distinct values which can be produced by a linear combination like 'x_const*x+y_const*y'.
    # The resulting count, and the following reads/writes counts might thus be overestimated.
    # When this is False, the computation of 'distinct_values' MAY (depends on strides) become slower.
    # NOTE: setting this to True is needed to match Timeloop, as it uses the same approximation.
    OVERESTIMATE_DISTINCT_VALUES = False

    # If True, the exploration of permutations done in optimizeDataflows will run across multiple
    # threads (or better, processes, due to the GIL).
    MULTITHREADED = True
    # Number of threads to use if MULTITHREADED is True. If None, it is set to the number of
    # logical CPUs available on the system.
    THREADS_COUNT = 8
    
    # Path to the folder above Accelergy, for a normal installation in Ubuntu that is usually like:
    # "/home/<username>/.local/lib/python3.X/site-packages/"
    # FactorFlow has been tested with commit 'd1d199e571e621ce11168efe1af2583dec0c2c49' of Accelergy.
    # NOTE: this is NOT required if you have installed Accelergy as a python package and can import it.
    ACCELERGY_PATH = "\\\\wsl.localhost/Ubuntu-22.04/home/zero/.local/lib/python3.10/site-packages"

    @classmethod
    def toString(self) -> str:
        res = "Settings("
        for k, v in vars(self).items():
            if not k.startswith("__"):
                res += f"{k}={v},"
        return res[:-1] + ")"