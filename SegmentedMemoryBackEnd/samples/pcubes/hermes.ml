// PCubeS Description of a 64-Core Opteron Machine in Big Iron Cluster
// AMD Opteron 6276 Server Processor
// Total Shared Memory 256 GB
// Data width 64 bit

//--------------------------------------------------------------------------------------
//Space #Number : 	$Space-Name	(#PPU-Count)	// Comment
//--------------------------------------------------------------------------------------
Space   5:		Socket		(1)		// 256 GB RAM Total
Space 	4^:  		CPU 		(4)		// 64 GB RAM Per CPU (location of memory segmentation
							// indicated by the '^' marker)						
Space 	3: 		NUMA-Node 	(2) 		// 6 MB L-3 Cache		
Space 	2: 		Core-Pair 	(4)		// 2 MB L-2 Cache (1 floating point unit per core-pair)
Space 	1*:		Core		(2)		// 16 KB L-1 Cache ( OS numbers processors starting
							// from this space, indicated by the '*' marker. Don't
							// map anything in this space. It was not supposed to 
							// be there in the original PCubeS description.)
