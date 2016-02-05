Things for Future Development
-------------------------------------------------------------------------------------------------
1. This compiler does not support synchronization for code having overlapping boundaries in data
   partitions, a.k.a ghost regions.
2. We implemented a barrier based simplified synchronization scheme instead of the detailed highly
   decoupled scheme suggested in the intermediate representation. We did this just to save time.
   Once Andrew, my profe., get the time and proper mindset to develop better sync primitives for
   different synchronization types; we can replace the barrier based synchronization currently in
   place with a new one that will keep the signaler of updates separate from waiting computations
   and maximize the opportunity of avoiding waiting for sync signals by placing as many computa-
   tions as possible in between signals and waits.
3. Epoch handling and function calls need to be implemented 
