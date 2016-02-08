//---------------- PCubeS Description of a cluster of four 20 Core Intel(R) Xeon(R) CPU E5-2670 ----------------------
//--------------------------------------------------------------------------------------------------------------------
//Space #Number : 		$Space-Name	(#PPU-Count)	// Comments
//--------------------------------------------------------------------------------------------------------------------
Space	4:			Node		(1)		// cluster level
Space 	3<unit><segment>:  	CPU 		(4)		// 4 CPUs
Space	2:			Bi-Section	(2)		// 2 groups of 10 cores in each CPU, 
								// 25 MB L-3 cache per group				
Space 	1<core>:		Core		(10)		// 256 KB L-2 cache for individual cores
