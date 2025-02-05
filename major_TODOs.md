- improve latency modeling by dividing it per-operand (in updateStats)
- ~~maximize SA usage before MSE (in factorFlow)~~
- ~~prune some dataflow permutations via the work on derivatives~~
- ~~create the dump of a mapping functionality within the new Arch class! => NEW FILE arch.py containing two classes, Arch and (maybe) Mapping (which shall also store top metrics: EDP and Wart, shall also have the factorsAtOne method)~~
- in case of padding, ignore the MAC cost for padding values
- ~~decide whether to treat ComputeLevels as FanoutLevels or MemLevels in fanoutMaximization and optimizeDataflows!!!~~
- could unify part of level constructors (like the constraints asserts)!
- support for strides -> can only apply a stride to all but one of the dimensions in a sum of indices for an operand, the result is that each strided index for that operand will count as *stride itself when computing tile sizes and memory footprints, and the skip-read-in-indices-sum mechanic will apply not when innermost iterations are two, but when they are equal to the stride+1. Dimensions sizes themselves still correctly represents tensors, even when there is stride, BUT REMOVE FROM A DIMENSIONS'S PRIME FACTORS THE PRIME FACTORS OF THE STRIDE, because less iterations are needed when striding! Print the stride for each non-stride-1 dimension in printFactors, after the mapping! Add stride-dimensions among dimension names, then flag them as strides with a new class!
- make the flat couplings into sets -> NO NEED, lists are fine as long as they are short
- initialize Factors in levels with only the level dataflow's dimensions, and update Factors to return as if at 1 any missing dimension


```python
# >>>> THIS WAS LOCATED IN LEVEL.PY, LINE 122 <<<<
# >> MODEL VECTOR ENERGY FIX <<
# 
# ADD HERE (constructor):
# - mem_width: bits per wordline
# - datawidth: operands size in bits
# - a method called "padToDatawidth" which from a number of MOPs, returns the number of MOPs actually happening since you read only full wordlines (MOPs += mem_width - MOPs%mem_width)
# [Maybe call this "paddingForDatawidth" and make it return solely the required padding]
#
# While you are at it, add documentation for what a function returns down in the methods!!!
#
# ADD TO setMOPs method down here:
# - (NAH) extra arguments to store the padded to datawidth version of MOPs -> NAH, WHAT A WASTE OF MEMORY!!!!
#
# MORE INFO UPDATED IN MAIN!!!!
# -----------------------------
# CORRECT INFO FROM TELEGRAMMMM:
# 
# The best way is the following:
# - add a flag “consider_memwidth_padding”
# - when the flag is True, in MOPs, at the lines where you compute tilesize*dim*dim, obtaining the size of the tile stored on your level, add there the +(memwidth-fulltilesize%memwidth) to the fulltilesizes
# - updateStats received the flag too and passes it to MOPs, the flag is True always during MSE. On the final updateStats called by the main thread, first use the flag to print EDP, then remove it and updateStats again to print MOPs!
# - in the future, add an attribute to the Arch class signaling if it was evaluated last with or without the flag.
#
# => aggiorna anche i test per usare la flag su False, ed aggiungi test con la Flag true e che stimano l’access energy dalla vector access energy correttamente!
# => l’access energy rimane come attributo ed è calcolata, se non data, dalla vector access energy, aggiungi un metodo “printAccessEnergys” che le stampa assieme alle vector!
#
# Commit together with this the leak energy!
#
#
# MORE BAD IDEAS:
#
# When an operand is reused only once on a layer, don't even store it!
# In other words, if I am weight stationary, I should not store weights, as they are read only once from me! I could directly pass them through from the above layer!
# => It could make sense to not store the operand you are stationary with, modeling it as adding an artificial bypass!

# >>>> THIS WAS LOCATED IN MAIN, LINE 36<<<<
# MODEL VECTOR ENERGY FIX
# KEY OBSERVATION (leave this in the code): the only case in which a MemLevel is forced to read padding to reach full datawidth is when it exhausts the current tile it stores,
# therefore this only happens when it exhausts its own loops, and thus occurs only once per every OUTER iteration w.r.t. that level. While on-level iterations occur, the level
# simply buffers any extra part of a wordline that it reads in order to send it later one downward. Hencefort here we need to multiply the padding of MOPs by outer iterations!
# 
# ADD:
# - compute, for each in_reads, w_reads, ..., a variant called datawidth_in_reads, datawidth_w_reads, ..., by mapping them through the MemLevel.padToDatawidth method!
# - pass both datawidth and base versions all through the map here multiplying them by iterations
# - at last, use the datawidth versions to compute the full reads and writes for the WMOPs call!
# BETTER:
# - let "padToDatawidth"->"paddingForDatawidth" return just the padding, and accumulate it in just 2 variables (read_datawidth_padding, write_datawidth_padding)
#   - there is no need to distinguish between operands!
# - multiply the two variables by iterations, and add them to reads and writes before WMOPs
# - also add the two variables to setMOPs!
#
# DANGER: you need LATENCY for the energy leakage, therefore move this calculation in the third loop over the architecture!!!
```

```python
# CONCLUSIONS for MemLevel.MOPs with arbitrary Couplings:
# 1) if the next level is a fanout level (if there is a chain of fanouts, consider them all) and it has any of the dim in innermost_dim_sum among its dimensions with more than one iteration,
#    then this reuse (this IF case here) must be skipped, since the halo that would be reuse is stored in the wrong instance, which cannot reuse it, and is more efficient to re-read the
#    data rather than move it beween instances or flip the order with which instances are indexed (oh dear, this last option may be worth it if it can be cheaply implemented)!
#    => This caused the 20x error!
# 2) HERE IS WHERE THE 2.4 COME FROM, RATHER THAN 8+3-1=10 (tileQ+tileS-1) it is 8*3=24 (tileQ*tileS), so, simply, dimension that get unfolded spatially, get their iterations
#    multiplied, rather than summed, during the dim_sum's tile size calculation! But why is that? That is, if there is no multicast support!
#    WRONG FIX: if any dimension in a dim_sum is part of any fanout in a potential chain of fanouts following this level, then the dim_sum goes from sum to product!
#    FIX: if a dimension in a dim_sum is part of any fanout in a potential chain of fanouts following this level, then ONLY THAT DIM is multiplied by the rest of the sum of the dim_sum!
#    => each instance wants its part of the tile, which depends on the sum of Q and S, indicizing said tile, this result occurs if each instance reads its part of the tile on its own,
#       therefore we get a read for <spatial-iteration>*(dim_sum of the other dimensions not in the fanout)*remaining_tile_sizes.
#    => if the NoC has multicast capabilities, THIS should not occur...
#    => You don’t just commute the sum of tile sizes in a product! Multiply by the total spatially unrolled iterations along one of the dim_sum dimensions the reads, but still perform the
#       sum along the dim_sum, but divide each tile_size by the total iterations on the fanout!
# 3) In timeloop, if the stationarity condition triggers (the skip-if above), then the halo if (this IF case here) cannot trigger! This is the reason for the wrong DRAM input reads, since M
#    was the innermost dimension, and its stationarity prevents halo reuse on P! According to timeloop's article, it seems that it indeed does only once check for stationarity OR halo reuse,
#    and does not check subsequent cases...
```

```python
# ISSUE on multithreaded optimizeDataflows version with shared past_perms when the counter to delete a past_perm was set to len(permutations[i])*threads_per_range[i]:
# |||
# some threads ON THE SAME KEY see threads_per_range[i] at 1 some at 2, this happens if the threads are not a power of two (as on the server) because
# some partitions gets more threads. Therefore as soon as a thread seeing a 1 triggers this conditions, it removes the key too early. However even if
# that didn't happen, the key would never get removed, because the 2 would be too high of a count.
# The only solution is to increase the counter IFF the level below was completed and remove the key on len(permutations[i])...
# ISSUE:
# while the normal case where lower level permutations are explored is well handled by this logic which "pushes upward" the solution, the equi-datflow
# and pruning case are not well handled. Two threads which split among themselves level 'i', will share the same slice of level 'i-1', now when they
# push upward only one of them gets the following if as True and increases the above counter, but if they both then start to get equi-dataflow matches,
# both will at the same time trigger the counter increase on level 'i-1' since they don't bother to go down to level 'i'. This at the same time is a
# a waste of time since both analyze and skip the same mappings!
# NONONO: if all permutations on this level have been explored or the other threads ...
# Final solution:
# A FINITE STATE MACHINE, an Enum of 4 elements: (Pending, Exploring, Skipping, Done)
# - commit this state of the repo, just as a memo
# - ditch the set in ThreadSafeHeap
# - each ThreadSafeHeap gets a list containing one of those Enum values for each of the permutations on its level.
# - all permutations start at 'Pending'
# - from 'Pending' you can go to 'Exploring' or 'Skipping'
# - from 'Exploring' you can go to 'Skipping'
# - from 'Exploring' and 'Skipping' you can go to 'Done'
# - if a thread "goes down" for a permutation, it is marked from 'Pending' to 'Exploring', unless it was already 'Exploring', while if it was 'Skipping', no "going down", but skip too.
# - if you either get an equi-dataflow match or a pruning you go from 'Pending' or 'Exploring' to 'Skipped'
# - if you reach this point of the code, the check to "push up"
# - delete the past_perm iff all list entries are 'Done' FUUUUCK I still need another counter to know how many threads explored and how many skipped...
# NEVERMIND, we go back to the set, but make it a dictionary where each entry counts the number of skips!
```

```python
# FAILED IDEA -> TOO CLUNKY!
# FROM: engine.py - line 428 as of commit a5c37d11159b707fa7d03f86e87d80011e4ae1d1
    while True:
        choices = exploreOneStep(remaining_steps = Settings.STEPS_TO_EXPLORE)
        if len(choices) == 0:
            if verbose: print(f"No valid follow-up configuration, stopping, current Wart: {best_wart:.3e}")
            break
        # >>> GREEDY MOVE <<<
        best_choice = max(choices, key=choices.get)
        if choices[best_choice] < best_wart:
            if verbose: print(f"Stopping with current Wart: {best_wart:.3e}, while best choice is: {choices[best_choice]:.3e}")
            break
        multiple_moves = Settings.INITAL_MULTIPLE_MOVES[inital_multiple_moves_idx] if inital_multiple_moves_idx < len(Settings.INITAL_MULTIPLE_MOVES) else 1
        # apply the top 'multiple_moves' moves, without re-evaluating mappings in between and speculating that other moves retained their improvements after the first one got applied
        while multiple_moves > 0:
            # each individual choice is defined by 5 parameters, chained to another 5 for each nested exploration step
            multisteps = len(best_choice) // 5
            moves_count += multisteps
            for i in range(multisteps):
                if not arch.moveFactor(best_choice[5*i + 0], best_choice[5*i + 1], best_choice[5*i + 2], best_choice[5*i + 3], best_choice[5*i + 4], skip_src_constraints = Settings.NO_CONSTRAINTS_CHECK_DURING_MULTISTEP and i < multisteps - 1):
                    break
                    # TODO: ondo outer steps if breaking inside...
            best_wart = choices[best_choice]
            multiple_moves -= 1
            best_choice = max(choices, key=choices.get)
        inital_multiple_moves_idx += 1

# FROM: settings.py
    # Number of moves to apply all at once by selecting them from the best performing moves during
    # local search. This is a list since subsequent local search steps pop from the list's head
    # the number of moves to apply, that defaults to 1 on the empty list.
    # NOTE: if a move is invalidated by a previously selected once, the best next one is picked,
    # as a result of this and of the lack of re-evaluation of mappings between the application of
    # moves, using this may lead to a severe worsening of results.
    INITAL_MULTIPLE_MOVES = [3, 2, 2]
```