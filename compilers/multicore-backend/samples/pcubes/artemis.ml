// PCubeS Description of a 32-Core Artemis machine
// AMD Opteron 6276 Server Processor
// Total Shared Memory 125 GB
// Data width 64 bit

//--------------------------------------------------------------------------------------
//Space #Number : 	$Space-Name	(#PPU-Count)	// Comment
//--------------------------------------------------------------------------------------
Space   5:		Socket		(1)		// 125 GB RAM Total
Space 	4:  		CPU 		(2)		// 64 GB RAM Per CPU			
Space 	3: 		NUMA-Node 	(2) 		// 6 MB L-3 Cache		
Space 	2: 		Core-Pair 	(4)		// 2 MB L-2 Cache (1 floating point unit per core-pair)
Space 	1<core>:	Core		(2)		// 16 KB L-1 Cache 
