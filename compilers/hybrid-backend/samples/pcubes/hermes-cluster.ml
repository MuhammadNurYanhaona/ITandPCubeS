//------------------------- PCubeS Description of a cluster of four 64-Core Opteron Machines -------------------------
//--------------------------------------------------------------------------------------------------------------------
//Space #Number : 		$Space-Name	(#PPU-Count)	// Comments
//--------------------------------------------------------------------------------------------------------------------
Space	6:			Node		(1)		
Space   5<unit>:		Socket		(4)		// multiple physical units at this level
Space 	4<segment>:  		CPU 		(4)		// 64 GB RAM Per CPU (location of memory segmentation)						
Space 	3: 			NUMA-Node 	(2) 		// 6 MB L-3 Cache		
Space 	2: 			Core-Pair 	(4)		// 2 MB L-2 Cache (1 floating point unit per core-pair)
Space 	1<core>:		Core		(2)		// 16 KB L-1 Cache (core numbering starts here)
