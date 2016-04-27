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

/* For computation over valid transformed indexes we need to copy in the storage and partition dimension information of each data 
 * part in the GPU. A part-dimension object (a part of an LPU) is not suitable for directly being copied into the GPU memory due to
 * its hierarchical nature. So we have the following class to transform the part-dimension object into a GPU friendly format.
 */
class PartDimRanges {
  private:
	int size;
	int depth;
	Range *ranges;
  public:
	PartDimRanges(int dimensionality, PartDimension *partDimensions);
	~PartDimRanges();
	int getSize() { return size; }	
	int getDepth() { return depth; }
	void copyIntoBuffer(int *buffer);
};

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
	PartDimRanges *partDimRanges;
	void *data;
	// this tells the size of each element inside the data part
	int elementSize;
	// this tells the total element count in the data part
	int elementCount;
	// this is the part Id of the storage unit (not the partition unit) that host the data for this LPU part
	List<int*> *partId;
	// read only data parts need not be retrieved back from the GPU at the end of the kernel executions
	bool readOnly;
  public:
	LpuDataPart(int dimensionality, 
			PartDimension *dimensions, 
			void *data, 
			int elementSize, 
			List<int*> *partId);
	~LpuDataPart() { delete partDimRanges; }
	PartDimRanges *getPartDimRanges() { return partDimRanges; }
	void *getData() { return data; }
	List<int*> *getId() { return partId; }
	bool isMatchingId(List<int*> *candidateId);
	int getSize();
	void flagReadOnly() { readOnly = true; }
	bool isReadOnly() { return readOnly; }
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
	GpuMemoryConsumptionStat(long consumptionLimit); 
	void addPartConsumption(LpuDataPart *part);
	void reset() { currSpaceConsumption = 0l; }
	bool isOverfilled() { return currSpaceConsumption > consumptionLimit; }
	bool canHoldLpu(int lpuMemReq) { return currSpaceConsumption + lpuMemReq <= consumptionLimit; }
	
	// this determines what parcentage of the set consumption limit has already being filled 
	float getConsumptionLevel();	
}; 

/* This class keeps track of the data parts of different LPUs that are part of the current batch that is under execution or going
 * to be executed. 
 */
class LpuDataPartTracker {
  private:
	Hashtable<List<int>*> *partIndexMap;
  	Hashtable<List<LpuDataPart*>*> *dataPartMap;
  public:
	LpuDataPartTracker();
	void initialize(List<const char*> *varNames);
	List<int> *getPartIndexList(const char *varName) { return partIndexMap->Lookup(varName); }
	List<LpuDataPart*> *getDataPartList(const char *varName) { return dataPartMap->Lookup(varName); } 
	
	// An addition of a new LPU data part for a particular property may fail as that part may have been already included as part
	// of a previous LPU. The return value of this function indicates if the add operation was successful so that the caller can 
	// delete the data part if not needed.		
	bool addDataPart(LpuDataPart *dataPart, const char *varName);

	// delete all data parts of the current batch	
	void clear();	   	
};

/* This is a helper class introduced to reduce the number of parameters need to be passed to GPU kernels for LPU data references */
class GpuBufferReferences {
  public:
	void *dataBuffer;
	int *partIndexBuffer;
	int *partRangeBuffer;
	int *partBeginningBuffer;
};

/* Since data parts can be very small and numerous, there might be a significant cost in staging than in and out of GPU card memory. 
 * Rather, we combine all parts for a particular property in the LPUs into a single buffer and stage in and out the buffer to/of the 
 * GPU (note that staging out is only needed if the buffer has been updated by the offloaded kernel in the GPU). This class generates 
 * the CPU/GPU buffers for a single LPU property. Along with the data buffer, we need two additional buffers for storing the part 
 * indexes for different LPUs for that property and the beginning index of a part in the data buffer.   
 */
class PropertyBufferManager {
  private:
	int bufferSize;
	int bufferEntryCount;
	int bufferReferenceCount;
	int partRangeDepth;
	int partRangeBufferSize;

	char *cpuBuffer;
	int *cpuPartIndexBuffer;
	int *cpuPartRangeBuffer;
	int *cpuPartBeginningBuffer;

	char *gpuBuffer;
	int *gpuPartIndexBuffer;
	int *gpuPartBeginningBuffer;
	int *gpuPartRangeBuffer;
  public:
	PropertyBufferManager();
	~PropertyBufferManager();
	void prepareCpuBuffers(List<LpuDataPart*> *dataPartsList, List<int> *partIndexList);
	void prepareGpuBuffers();
	GpuBufferReferences getGpuBufferReferences();
	void syncDataPartsFromBuffer(List<LpuDataPart*> *dataPartsList);
	void cleanupBuffers();
};

/* This class manages the property buffers and access to those buffers for different LPU properties for all LPUs that are part of a 
 * single batch.    
 */
class LpuDataBufferManager {
  private:
	Hashtable<PropertyBufferManager*> *propertyBuffers;
  public:
	LpuDataBufferManager(List<const char*> *propertyNames);
	void copyPartsInGpu(const char *propertyName, 
			List<LpuDataPart*> *dataPartsList, 
			List<int> *partIndexList);
	GpuBufferReferences getGpuBufferReferences(const char *propertyName);
	void retrieveUpdatesFromGpu(const char *propertyName, List<LpuDataPart*> *dataPartsList);
	void reset();
};

/* This class serves as a broker for data transfers between the host and the GPU. It maintains GPU and CPU buffer state information
 * using the earlier classes of this library and try to reuse those buffers, if possible, for multiple batch submissions to the GPU.
 */
class LpuBatchController {
  private:
	int batchLpuCountThreshold;
	int currentBatchSize;
	List<const char*> *propertyNames;
	List<const char*> *toBeModifiedProperties;
	LpuDataPartTracker *dataPartTracker;
	LpuDataBufferManager *bufferManager; 		
	GpuMemoryConsumptionStat *gpuMemStat;
  public:
	LpuBatchController(int lpuCountThreshold, 
			long memoryConsumptionLimit, 
			List<const char*> *propertyNames,
			List<const char*> *toBeModifiedProperties);

	bool canAddNewLpu() { return currentBatchSize < batchLpuCountThreshold; }
	bool canHoldLpu(LPU *lpu);
	void submitCurrentBatchToGpu();
	bool isEmptyBatch() { return currentBatchSize > 0; }
	int getBatchLpuCountThreshold() { return batchLpuCountThreshold; }
	int getCurrentBatchSize() { return currentBatchSize; }
	GpuBufferReferences getGpuBufferReferences(const char *propertyName) { 
		return bufferManager->getGpuBufferReferences(propertyName); 
	}
	void updateBatchDataPartsFromGpuResults();
	void resetController();	

	// Task:LPS specific sub-classes of the batch controller should provide implementation for the following two functions
	virtual int calculateLpuMemoryRequirement(LPU *lpu) = 0;
	virtual void addLpuToTheCurrentBatch(LPU *lpu) { currentBatchSize++; }
};

#endif
