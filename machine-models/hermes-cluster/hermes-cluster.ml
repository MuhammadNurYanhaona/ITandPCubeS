//------------------------- PCubeS Description of a cluster of four 64-Core Opteron Machines -------------------------
//--------------------------------------------------------------------------------------------------------------------
//Space #Number : 		$Space-Name	(#PPU-Count)	// Comments
//--------------------------------------------------------------------------------------------------------------------
Space	6:			Cluster		(1)		// nodes are connected by 10 GB ethernet
Space   5<unit>:		Node		(4)		// four nodes in the cluster
Space 	4<segment>:  		CPU 		(4)		// 64 GB RAM Per CPU						
Space 	3: 			NUMA-Node 	(2) 		// 6 MB L-3 Cache		
Space 	2: 			Core-Pair 	(4)		// 2 MB L-2 Cache (1 floating point ALU unit)
Space 	1<core>:		Core		(2)		// 16 KB L-1 Cache (1 integer ALU unit)
