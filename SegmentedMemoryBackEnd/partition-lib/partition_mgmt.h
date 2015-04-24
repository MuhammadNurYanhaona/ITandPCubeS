#ifndef _H_partition_mgmt
#define _H_partition_mgmt

#include "../codegen/structure.h"

/* For each partition function, we need to know how many LPUs will be generated given 
   the user provided configuration and the length of the dimension been partitioned by
   the function. 
   The signature for these functions is as follows

   int $(FUNCTION)_partitionCount(Dimension d, int PPU_Count, ...)

   Here '...' stands for the arguments for the partition function, if there is any, in
   their respective order.
*/

int block_size_partitionCount(Dimension d, int ppuCount, int size);
int block_count_partitionCount(Dimension d, int ppuCount, int count);
int stride_partitionCount(Dimension d, int ppuCount);
int block_stride_partitionCount(Dimension d, int ppuCount, int size);
	
/*
   Furthermore, we need a mechanism to determine the portition of the dimension that 
   falls within a single LPU given its ID as input. Here the assumption is that, data 
   is stored in a manner that all indexes for a single LPU should be consecutive 
   regardless of the nature of the partitioning function. We will use associated index
   transformation functions to determine actual to transformed index mapping during
   index traversal.

   The signature for these functions are as follows

   Dimension $(FUNCTION)_getRange(Dimension d, int lpuCount, 
		int lpuId, bool copyMode, ...)

   The forth parameter is there to distinguish between LPU part range generation for
   data structure been allocated a new in the underlying LPS or referencing back to
   the allocation done for the data structure been allocated in some ancestor LPS.
   If the dimension is reordered by the partition function then the LPU description
   should be different for these two different situations.		

   Note that if a partition function supports overlapping boundary/ghost regions among
   adjacent LPUs then two padding parameters (for front and back ends) should be
   added in the range function to adjust the returned dimension range appropriately. 		
*/

Dimension block_size_getRange(Dimension d, int lpuCount, int lpuId, bool copyMode,
		int size, int frontPadding, int backPadding); 
Dimension block_count_getRange(Dimension d, int lpuCount, int lpuId, bool copyMode,
		int count, int frontPadding, int backPadding); 
Dimension stride_getRange(Dimension d, int lpuCount, int lpuId, bool copyMode); 
Dimension block_stride_getRange(Dimension d, int lpuCount, int lpuId, bool copyMode, 
		int size); 

/*
   An alternative to the above approach of data re-ordering is to provide index 
   iteration specifications for different partition functions. These specifications
   are written as composition of two basic three functions
   1. range
   2. stride, and
   3. filter
   TODO: need to decide about the signature and implementation of these functions	 
*/
 	
/*   
   Finally, we need a function definition to determine lower if a a particular index
   falls within the data range for a given LPU. This is to combine range expressions 
   that may be present in any activation condition and generate the list of LPU ID 
   that will be activated dynamically.

   Signature of this function is as follows
   1. LpuIdRange *$(FUNCTION)_getUpperRange(int index, Dimension d, int ppuCount, ...) 
   2. LpuIdRange *$(FUNCTION)_getLowerRange(int index, Dimension d, int ppuCount, ...)
   3. int $(FUNCTION)_getInclusiveLpuId(int index, Dimension d, int ppuCount, ...)

   Note that if any of the above functions bears no meaning for a partition function
   then it should return NULL in its implementation.	  	   
*/

LpuIdRange *block_size_getUpperRange(int index, Dimension d, int ppuCount, int size);
LpuIdRange *block_size_getLowerRange(int index, Dimension d, int ppuCount, int size);
int block_size_getInclusiveLpuId(int index, Dimension d, int ppuCount, int size);

LpuIdRange *block_count_getUpperRange(int index, Dimension d, int ppuCount, int count);
LpuIdRange *block_count_getLowerRange(int index, Dimension d, int ppuCount, int count);
int block_count_getInclusiveLpuId(int index, Dimension d, int ppuCount, int count);

LpuIdRange *stride_getUpperRange(int index, Dimension d, int ppuCount);
LpuIdRange *stride_getLowerRange(int index, Dimension d, int ppuCount);
int stride_getInclusiveLpuId(int index, Dimension d, int ppuCount);

inline LpuIdRange *block_stride_getUpperRange(int index, 
		Dimension d, int ppuCount, int size);
inline LpuIdRange *block_stride_getLowerRange(int index, 
		Dimension d, int ppuCount, int size);
int block_stride_getInclusiveLpuId(int index, Dimension d, int ppuCount, int size);


#endif
