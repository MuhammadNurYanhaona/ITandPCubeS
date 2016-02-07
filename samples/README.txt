A single IT source should run in all PCubeS architectures, let it be a multicore machine
or a segmented memory supercomputer, and so on. Hence, the samples should not be different
for different back-end types -- but we have different directories for samples intended to 
run in different backends. 

This is done as at the current phase of the project, we have some features that are not
implemented for all back-end types. For example, IO support for the multicore backend and 
the segmented-memory backend compilers are totally disjoint.

Once we implements all features uniformly in for all backend types; we will, God willing,
simplify the samples directory.
