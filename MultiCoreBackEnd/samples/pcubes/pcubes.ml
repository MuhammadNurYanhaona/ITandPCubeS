// pcubes description of a 16-core AMD Opteron CPU
// Note that the numbering of spaces represents their
// orders in the hierarchy, with lowest number 
// representing the lowest space 

// syntax 'Space #Number : $Space-Name (#PPU-Count)'
Space 5: CPU (1)
Space 4: NUMA Node (2)
Space 3: Core Pair (4)
Space 2*: Core (2) // * after the space number means
		   // this is the space for CPU cores
		   // this is not part of original
                   // PCubeS definition. It is only
		   // been used for thread affinity.	 		
Space 1: Hyperthread (4)
