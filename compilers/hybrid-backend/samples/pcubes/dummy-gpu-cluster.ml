//---------------------------- An imaginary cluster connecting four of my laptops ------------------------------------
//-------------------------------------- Each having an NVIDIA K-20 GPU ----------------------------------------------
//--------------------------------------------------------------------------------------------------------------------
//Space #Number : 			$Space-Name	(#PPU-Count)	// Comments
//--------------------------------------------------------------------------------------------------------------------
Model:	CPU-Only-Model
Space	4:				Node		(1)		
Space 	3<segment><unit>:  		CPU 		(4)						
Space 	2: 				Core 		(2)
Space 	1<core>:			Hyperthread	(2)

Model:	Hybrid-Model
Space	5:				Node		(1)		
Space 	4<segment><unit><core>:  	CPU 		(4)						
Space 	3<gpu>: 			GPU		(1)
Space 	2:				SM		(13)
Space	1:				Warp		(16)
