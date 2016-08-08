#ifndef _H_lpu_parts_tracking
#define _H_lpu_parts_tracking

// Our current strategy for offloading computation to GPUs involves the host CPU allocates and maintains the data parts, as done in
// the segmented-memory environment, and mediates movements of parts as needed by LPUs in and out of the GPU card memory. Further, 
// a batch of LPUs will be scheduled for execution in the GPU at a time necessitating stage in and out of a set of parts for a data  
// item -- as opposed to just one. This library holds all management structures and functions needed for tracking, synchronizing, 
// cleaning up LPU data parts for host and GPU interaction.

#include "part_id_tree.h"
#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "../runtime/structure.h"

#include <fstream>
#include <vector>

class LpuBatchController;
class DataPartitionConfig;

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
	void describe(std::ofstream &stream, int indentLevel);
	int getStorageDimLength(int dimNo);
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
  protected:
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
	virtual ~LpuDataPart() { delete partDimRanges; }
	PartDimRanges *getPartDimRanges() { return partDimRanges; }
	virtual void *getData() { return data; }
	List<int*> *getId() { return partId; }
	bool isMatchingId(List<int*> *candidateId);
	virtual int getSize();
	void flagReadOnly() { readOnly = true; }
	bool isReadOnly() { return readOnly; }
	int getDataDimensions() { return dimensionality; }
	virtual void describe(std::ofstream &stream, int indentLevel);
	
	// these two functions are provided to simplify code generation for determining the size of the largest subpart that may 
	// be copied into and out of the SM memory for a larger data part residing in the GPU card memory.
	void copyDimLengths(int *dimLengthArray);
	virtual int getSizeForSubpartsWithDimLengths(int *subpartLengthsArray);

};

/* This extension to the above LPU Data Part class is needed to stage-in/out properties that have multiple epoch versions. */
class VersionedLpuDataPart : public LpuDataPart {
  protected:
	short versionCount;
	int currVersionHead;
	List<void*> *dataVersions;
  public:
	VersionedLpuDataPart(int dimensionality, 
                        PartDimension *dimensions,
                        List<void*> *dataVersions, 
                        int elementSize, 
                        List<int*> *partId);
	~VersionedLpuDataPart() { delete dataVersions; }
	void *getData() { return dataVersions->Nth(currVersionHead); }
	short getVersionCount() { return versionCount; }
	void *getDataVersion(int version);
	void advanceEpochVersion();
	int getSize() { return LpuDataPart::getSize() * versionCount; }
	void describe(std::ofstream &stream, int indentLevel);
	int getSizeForSubpartsWithDimLengths(int *subpartLengthsArray);
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

/* This class keeps track of all parts for a particular data structure and provides efficient mechanism to store and search them. 
 */
class LpuDataPartLister {
  private:
	List<LpuDataPart*> *dataPartList;
	PartIdNode *dataPartTree;
  public:
	LpuDataPartLister();
	~LpuDataPartLister();
	void clear();
	
	// tells if a data part with the matching ID is already in the data part list
	bool isAlreadyIncluded(List<int*> *dataPartId);
	
	// returns the index in the data part list of the data part having the argument part-ID
	int getPartStorageIndex(List<int*> *dataPartId);

	// adds a new data part in the data part list and returns the index in the list the part has been put into; if a part with
	// the same ID is already in the list then it returns -1 to indicate the insertion has failed
	int addDataPart(LpuDataPart *dataPart);

	List<LpuDataPart*> *getDataPartList() { return dataPartList; }
};

/* This class keeps track of the data parts of different LPUs that are part of the current batch that is under execution or going
 * to be executed. 
 */
class LpuDataPartTracker {
  private:
	// When the GPU PPUs are distinguished from the host, specific LPUs are allocated for execution to specific PPUs. Then data
	// part tracking for those LPUs need to be done separately for those PPUs too. Therefore, a count of the number of PPUs in
	// the GPU and a vector having an index list entry per PPU is needed.		
	int distinctPpuCount;
	Hashtable<std::vector<List<int>*>*> *partIndexMap;
	// The actual data parts are, however, not distinguished by LPUs as multiple LPUs intended for different PPUs may share a
	// single data part
  	Hashtable<LpuDataPartLister*> *dataPartListerMap;
	// To be able to allocate and assign parts of dynamic SM memory properly, we need to know the maximum memory requirement for
	// a data part of individual variables used inside the GPU kernels. This property is used to keep track of the maximum memory 
	// consumption per data part per variable 
	Hashtable<int*> *maxDataPartSizes;	

	// a back pointer to the the lpu-batch-controller is maintained to get access to its part-size determining function
	LpuBatchController *batchController;

	std::ofstream *logFile;
  public:
	LpuDataPartTracker(int distinctPpuCount, LpuBatchController *batchController);
	void setLogFile(std::ofstream *logFile) { this->logFile = logFile; }
	void initialize(List<const char*> *varNames);
	std::vector<List<int>*> *getPartIndexListVector(const char *varName) { 
		return partIndexMap->Lookup(varName); 
	}
	List<LpuDataPart*> *getDataPartList(const char *varName); 
	int getMaxPartSize(const char *varName) { return *(maxDataPartSizes->Lookup(varName)); }
	
        // An addition of a new LPU data part for a particular property may fail as that part may have been already included as part
        // of a previous LPU. The return value of this function indicates if the add operation was successful so that the caller can 
        // delete the data part if not needed.
	// Note that even already included data parts must be attempted to be added through this function as the partIndexMap needs
	// to be updated even for redundant data parts.
	bool addDataPart(LpuDataPart *dataPart, const char *varName, int ppuIndex);

	// this tells if a candidate data part has already been included in the part tracker
	bool isAlreadyIncluded(List<int*> *dataPartId, const char *varName);

	// this function deletes all data parts of the current batch	
	void clear();	   	
};

/* This is a helper class introduced to reduce the number of parameters need to be passed to GPU kernels for LPU data references */
class GpuBufferReferences {
  public:
	char *dataBuffer;
	int *partIndexBuffer;
	int *partRangeBuffer;
	int *partBeginningBuffer;

	// this non-pointer variable is needed to determine how to read information from the part-range-buffer
	int partRangeDepth;
};

/* Data structures that have multiple versions need another buffer to be exchanged between the host and the GPU to keep track of the
 * most recent versions of different parts
 */
class VersionedGpuBufferReferences : public GpuBufferReferences {
  public:
	short versionCount;
	short *versionIndexBuffer;
};

/* Since data parts can be very small and numerous, there might be a significant cost in staging than in and out of GPU card memory. 
 * Rather, we combine all parts for a particular property in the LPUs into a single buffer and stage in and out the buffer to/of the 
 * GPU (note that staging out is only needed if the buffer has been updated by the offloaded kernel in the GPU). This class generates 
 * the CPU/GPU buffers for a single LPU property. Along with the data buffer, we need two additional buffers for storing the part 
 * indexes for different LPUs for that property and the beginning index of a part in the data buffer.   
 */
class PropertyBufferManager {
  protected:
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

	std::ofstream *logFile;
  public:
	PropertyBufferManager();
	~PropertyBufferManager();
	void setLogFile(std::ofstream *logFile) { this->logFile = logFile; }
	virtual void prepareCpuBuffers(List<LpuDataPart*> *dataPartsList, 
			std::vector<List<int>*> *partIndexListVector);
	virtual void prepareGpuBuffers();
	virtual GpuBufferReferences *getGpuBufferReferences();
	virtual void syncDataPartsFromBuffer(List<LpuDataPart*> *dataPartsList);
	virtual void cleanupBuffers();
  protected:
	virtual void copyDataIntoBuffer(LpuDataPart *dataPart, char *dataBuffer);
};

/* As we needed an additional host:gpu interaction buffer for data structures with multiple versions, we need enhancement to the 
 * property buffer manager class to accomodate for that extra buffer's management 
 */
class VersionedPropertyBufferManager : public PropertyBufferManager {
  private:
	short *cpuDataPartVersions;
	short *gpuDataPartVersions;
	short versionCount;
  public:
	void prepareCpuBuffers(List<LpuDataPart*> *dataPartsList,
                        std::vector<List<int>*> *partIndexListVector);
	void prepareGpuBuffers();
	GpuBufferReferences *getGpuBufferReferences();
	void syncDataPartsFromBuffer(List<LpuDataPart*> *dataPartsList);
	void cleanupBuffers();
  protected:
	void copyDataIntoBuffer(LpuDataPart *dataPart, char *dataBuffer);  
};

/* This class manages the property buffers and access to those buffers for different LPU properties for all LPUs that are part of a 
 * single batch.    
 */
class LpuDataBufferManager {
  private:
	Hashtable<PropertyBufferManager*> *propertyBuffers;
	std::ofstream *logFile;
  public:
	LpuDataBufferManager(List<const char*> *propertyNames);
	LpuDataBufferManager(List<const char*> *versionlessProperties, 
			List<const char*> *multiversionProperties);
	void setLogFile(std::ofstream *logFile);
	void copyPartsInGpu(const char *propertyName, 
			List<LpuDataPart*> *dataPartsList, 
			std::vector<List<int>*> *partIndexListVector);
	GpuBufferReferences *getGpuBufferReferences(const char *propertyName);
	void retrieveUpdatesFromGpu(const char *propertyName, 
			List<LpuDataPart*> *dataPartsList);
	void reset();
};

/* This class serves as a broker for data transfers between the host and the GPU. It maintains GPU and CPU buffer state information
 * using the earlier classes of this library and try to reuse those buffers, if possible, for multiple batch submissions to the GPU.
 */
class LpuBatchController {
  protected:
	int batchLpuCountThreshold;
	int currentBatchSize;
	int distinctPpuCount;
	List<const char*> *propertyNames;
	List<const char*> *toBeModifiedProperties;
	LpuDataPartTracker *dataPartTracker;
	LpuDataBufferManager *bufferManager; 		
	GpuMemoryConsumptionStat *gpuMemStat;

	// a reference to the map of partition configurations of different data structure is maintained so that subclasses can give
	// proper implementation of the calculateSmMemReqForDataPart function 
	Hashtable<DataPartitionConfig*> *partConfigMap;

	std::ofstream *logFile;
  public:
	LpuBatchController(Hashtable<DataPartitionConfig*> *partConfigMap);
	void setBufferManager(LpuDataBufferManager *manager) { this->bufferManager = manager; }
	void initialize(int lpuCountThreshold, 
			long memoryConsumptionLimit, 
			int distinctPpuCount,
			List<const char*> *propertyNames,
			List<const char*> *toBeModifiedProperties);
	void setLogFile(std::ofstream *logFile);
	bool canAddNewLpu() { return currentBatchSize < batchLpuCountThreshold; }
	bool canHoldLpu(LPU *lpu);
	void submitCurrentBatchToGpu();
	bool isEmptyBatch() { return currentBatchSize == 0; }
	int getBatchLpuCountThreshold() { return batchLpuCountThreshold; }
	int getCurrentBatchSize() { return currentBatchSize; }
	int getMaxPartSizeForProperty( const char *varName) { return dataPartTracker->getMaxPartSize(varName); }
	GpuBufferReferences *getGpuBufferReferences(const char *propertyName) { 
		return bufferManager->getGpuBufferReferences(propertyName); 
	}
	void updateBatchDataPartsFromGpuResults();
	void resetController();	

	// Task:LPS specific sub-classes of the batch controller should provide implementation for the following three functions.
	// In particular, the implementation of the third function is important to determine what LPU data parts should be copied
	// into the SMs's shared memory and what should be left on the GPU card memory and be accessed from there during kernel
	// computation. This is a critical analysis as SM memory is at the range of 100 times fater than the card memory but not
	// all data structures can be fit into it.
	virtual int calculateLpuMemoryRequirement(LPU *lpu) = 0;
	virtual void addLpuToTheCurrentBatch(LPU *lpu, int ppuIndex) { currentBatchSize++; }
	virtual int calculateSmMemReqForDataPart(const char *varName, LpuDataPart *dataPart) {
		return dataPart->getSize();
	}
};

#endif
