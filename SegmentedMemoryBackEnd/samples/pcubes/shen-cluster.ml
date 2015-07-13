//---------------- PCubeS Description of a cluster of four 20 Core Intel(R) Xeon(R) CPU E5-2670 ----------------------
//--------------------------------------------------------------------------------------------------------------------
//Space #Number : 		$Space-Name	(#PPU-Count)	// Comments
//--------------------------------------------------------------------------------------------------------------------
Space	5:			Node		(1)		// cluster level
Space 	4<unit><segment>:  	CPU 		(4)		// 4 CPUs (location of memory segmentation)
Space	3:			Bi-Section	(2)		// 2 groups of cores in each CPU				
Space 	2: 			Core-Group 	(10) 		// in each group, 25 MB L-3 cache for 10 cores		
Space 	1<core>:		Core		(1)		// 256 KB L-2 cache for individual core
