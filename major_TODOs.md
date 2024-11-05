- improve latency modeling by dividing it per-operand (in updateStats)
- ~~maximize SA usage before MSE (in factorFlow)~~
- ~~prune some dataflow permutations via the work on derivatives~~
- in case of padding, ignore the MAC cost for padding values
- decide whether to treat ComputeLevels as FanoutLevels or MemLevels in fanoutMaximization and optimizeDataflows!!!
- could unify part of level constructors (like the constraints asserts)!


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