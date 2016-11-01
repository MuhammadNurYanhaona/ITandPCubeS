//--------------------- PCubeS Description of the CPU cluster of the big Red II Supercomputer  -----------------------
//--------------------------------------------------------------------------------------------------------------------
//Space #Number : 		$Space-Name	(#PPU-Count)	// Comments
//--------------------------------------------------------------------------------------------------------------------
Space	6:			Cluster		(1)		// the whole cluster		
Space   5<unit><segment>:	Node		(344)		// multiple physical units at this level
Space 	4:  			CPU 		(2)		// 63 GB RAM for 2 CPUs						
Space 	3: 			NUMA-Node 	(2) 		// 6 MB L-3 Cache		
Space 	2: 			Core-Pair 	(4)		// 2 MB L-2 Cache (1 floating point unit per core-pair)
Space 	1<core>:		Core		(2)		// 16 KB L-1 Cache (core numbering starts here)
