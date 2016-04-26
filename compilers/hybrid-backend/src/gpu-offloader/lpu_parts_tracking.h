#ifndef _H_lpu_parts_tracking
#define _H_lpu_parts_tracking

// Our current strategy for offloading computation to GPUs involves the host CPU allocates and maintains the data parts, as done in
// the segmented-memory environment, and mediates movements of parts as needed by LPUs in and out of the GPU card memory. Further, 
// a batch of LPUs will be scheduled for execution in the GPU at a time necessitating stage in and out of a set of parts for a data  
// item -- as opposed to just one. This library holds all management structures and functions needed for tracking, synchronizing, 
// cleaning up LPU data parts for host and GPU interaction.

#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "../runtime/structure.h"

/* This class is used to extract any particular array and its associated information that is part of an LPU. We cannot retain the
 * LPU references directly as the current system uses just a single LPU object for a particular LPS and change the properties within
 * it as computation progresses from one LPU to the next. In addition, note that data parts are originally retrieved from the memory
 * management module and assigned to the properties of an LPU on demand. Thus, the information maintained by this class is already
 * available in the corresponding classes in the memory management module. Nevertheless, we added a new class as going back from LPUs
 * the data parts will require significant changes in the code that we want to avoid at the earlier phase of the development. Further,
 * that approach may add more overheads in the CPU to GPU transition and vice versa process. 
 */
class LpuDataPart {
  private:
	int dimensionality;
	PartDimension *dimensions;
	void *data;
	int elementSize;
	List<int*> *partId;
	bool readOnly;
	char *hashTag;
  public:
	bool isMatchingId(List<int*> *candidateId);
	int getSize();
};

/* In most likely cases, the memory capacity of the GPU will be far less than the memory capacity of the host machine. Therefore, 
 * the host cannot just pump in data for new LPUs in the GPU without taking into consideration how much GPU memory is remaining to
 * hold that data. This class is for tracking the memory consumption level in the GPU at an instance to better estimate the proper
 * LPU batch size for offloading computations.    
 */
class GpuMemoryConsumptionStat {
  private:
	long currSpaceConsumption;
	long consumptionLimit;
  public:
	void addDataPart(LpuDataPart *part);
	void reset();
	bool isOverfilled();
	float *getConsumptionLevel();	
}; 

/* This class keeps track of the data parts of different LPUs that are part of the current batch that is under execution or going
 * to be executed. 
 */
class LpuDataPartTracker {
  private:
	Hashtable<List<int>*> *partIndexListMap;
  	Hashtable<List<LpuDataPart*>*> *dataPartMap;
  public:
	void initialize(List<const char*> *varNames);
	void addDataPart(LpuDataPart *dataPart, const char *varName);
	void clear();	   	
};

/* Since data parts can be very small and numerous, there might be a significant cost in staging than in and out of GPU card memory. 
 * Rather, we combine all parts for a particular property in the LPUs into a single buffer and stage in and out the buffer to/of the 
 * GPU (note that staging out is only needed if the buffer has been updated by the offloaded kernel in the GPU). This class generates 
 * the CPU/GPU buffers for a single LPU property. Along with the data buffer, we need two additional buffers for storing the part 
 * indexes for different LPUs for that property and the beginning index of a part in the data buffer.   
 */
class PropertyBufferManager {
  private:
	long bufferCapacity;
	const char *propertyName;
	void *cpuBuffer;
	int *cpuPartIndexBuffer;
	int *cpuPartBeginningBuffer;
	void *gpuBuffer;
	int *gpuPartIndexBuffer;
	int *gpuPartBeginningBuffer;
  public:
	void allocateCpuBufferIfNeeded(long requiredCapacity);
	void copyPartsInCpuBuffer(List<LpuDataPart*> *dataPartsList, List<int> *partIndexList);
	void prepareAllGpuBuffers();
	void syncDataPartsFromBuffer(List<LpuDataPart*> *dataPartsList);
	void cleanupGpuBuffers();
};

/* This is a helper class introduced to reduce the number of parameters need to be passed to GPU kernels for LPU data references */
class GpuBufferReference {
  public:
	void *dataBuffer;
	int *partIndexBuffer;
	int *partBeginningBuffer;
};

/* This class manages the property buffers and access to those buffers for different LPU properties for all LPUs that are part of a 
 * single batch.    
 */
class LpuDataBufferManager {
  private:
	Hashtable<PropertyBufferManager*> *propertyBuffers;
  public:
	void initializeIfNeeded(List<const char*> *propertyNames);
	void copyPartsInGpu(List<LpuDataPart*> *dataPartsList, List<int> *partIndexList);
	void retrievePartsFromGpu(List<LpuDataPart*> *dataPartsList);
	GpuBufferReference *getGpuBufferReference(const char *propertyName);
	void cleanupGpuBuffers();
};

/* This class serves as a broker for data transfers between the host and the GPU. It maintains GPU and CPU buffer state information
 * using the earlier classes of this library and try to reuse those buffers, if possible, for multiple batch submissions to the GPU.
 */
class LpuBatchController {
  private:
	int batchLpuCountThreshold;
	LpuDataPartTracker *dataPartTracker;
	LpuDataBufferManager *bufferManager; 		
	GpuMemoryConsumptionStat *GpuMemStat;
  public:
	void initialize(int lpuCountThreshold, long memoryConsumptionLimit, List<const char*> *propertyNames);
	bool canHoldLpu(Lpu *lpu);
	void submitCurrentBatchToGpu();
	void updateBatchDataPartsFromGpuResults();
	void resetController();	

	// Task:LPS specific sub-classes of the batch controller should provide implementation for the following two functions
	virtual void calculateLpuMemoryRequirement(LPU *lpu);
	virtual void addLpuToTheCurrentBatch(LPU *lpu);
};

#endif
