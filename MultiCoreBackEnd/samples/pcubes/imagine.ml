// pcubes description of an imaginary multi-core CPU
// having 5 cores and 2 hyperthreads per core

// syntax 'Space #Number : $Space-Name (#PPU-Count)'
Space 2*: Core (5) // * after the space number means
		   // this is the space for CPU cores
		   // this is not part of original
                   // PCubeS definition. It is only
		   // been used for thread affinity.	 		
Space 1: Hyperthread (2)
