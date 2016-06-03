#include "lpu_parts_tracking.h"
#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "../runtime/structure.h"
#include "../gpu-utils/gpu_utils.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <fstream>

//--------------------------------------------------------------- Part Dim Ranges ---------------------------------------------------------------/

PartDimRanges::PartDimRanges(int dimensionality, PartDimension *partDimensions) {
	
	depth = partDimensions[0].getDepth();
	int entryCount = dimensionality * (depth + 1); // depth number of partition range and one storage range is needed
	ranges = new Range[entryCount];
	size = entryCount * 2; // each range has a min and max attributes	

	// first copy in the storage dimension range
	int currentIndex = 0;
	for (; currentIndex < dimensionality; currentIndex++) {
		ranges[currentIndex] = partDimensions[currentIndex].storage.range;
	}

	// then copy in the partition dimension in the bottom up fashion
	for (int i = 0; i < dimensionality; i++) {
		PartDimension *dimension = &(partDimensions[i]);
		for (int d = 0; d < depth; d++) {
			ranges[currentIndex] = dimension->partition.range;
			dimension = dimension->parent;
			currentIndex++;
		}
	}
}

PartDimRanges::~PartDimRanges() {
	delete[] ranges;
}

void PartDimRanges::copyIntoBuffer(int *buffer) {
	memcpy(buffer, ranges, size * sizeof(int));
}

//---------------------------------------------------------------- LPU Data Part ----------------------------------------------------------------/

LpuDataPart::LpuDataPart(int dimensionality,
		PartDimension *dimensions,
		void *data,
		int elementSize,
		List<int*> *partId) {
	
	this->dimensionality = dimensionality;
	this->partId = partId;
	this->elementSize = elementSize;
	this->readOnly = false;
	this->partDimRanges = new PartDimRanges(dimensionality, dimensions);
	this->data = data;
	
	elementCount = 1;
	for (int i = 0; i < dimensionality; i++) {
		elementCount *= dimensions[i].storage.getLength();
	}
}

bool LpuDataPart::isMatchingId(List<int*> *candidateId) {
	for (int i = 0; i < candidateId->NumElements(); i ++) {
		int *myId = partId->Nth(i);
		int *otherId = candidateId->Nth(i);
		for (int j = 0; j < dimensionality; j++) {
			if (myId[j] != otherId[j]) return false;
		}
	}
	return true;
}

int LpuDataPart::getSize() {
	return elementCount * elementSize;
}

//----------------------------------------------------------- Versioned LPU Data Part -----------------------------------------------------------/

VersionedLpuDataPart::VersionedLpuDataPart(int dimensionality,
		PartDimension *dimensions,
		List<void*> *dataVersions,
		int elementSize,
		List<int*> *partId) 
		: LpuDataPart(dimensionality, dimensions, NULL, elementSize, partId) {
	this->dataVersions = dataVersions;
	this->versionCount = dataVersions->NumElements();
	this->currVersionHead = 0;
}

void *VersionedLpuDataPart::getDataVersion(int version) {
	int index = (currVersionHead + version) % versionCount;
	return dataVersions->Nth(index);
}
        
void VersionedLpuDataPart::advanceEpochVersion() {
	currVersionHead = (currVersionHead + 1) % versionCount;
}

//--------------------------------------------------------- GPU Memory Consumption Stat ---------------------------------------------------------/

GpuMemoryConsumptionStat::GpuMemoryConsumptionStat(long consumptionLimit) {
	this->consumptionLimit = consumptionLimit;
	this->currSpaceConsumption = 0l;
}

void GpuMemoryConsumptionStat::addPartConsumption(LpuDataPart *part) {
	int partSize = part->getSize();
	currSpaceConsumption += partSize;
}

float GpuMemoryConsumptionStat::getConsumptionLevel() {
	return (100.0 * currSpaceConsumption) / consumptionLimit;
}

//------------------------------------------------------------ LPU Data Part Tracker ------------------------------------------------------------/

LpuDataPartTracker::LpuDataPartTracker() {
	partIndexMap = new Hashtable<List<int>*>;
	dataPartMap = new Hashtable<List<LpuDataPart*>*>;
}

void LpuDataPartTracker::initialize(List<const char*> *varNames) {
	for (int i = 0; i < varNames->NumElements(); i++) {
		const char *varName = varNames->Nth(i);
		partIndexMap->Enter(varName, new List<int>);
		dataPartMap->Enter(varName, new List<LpuDataPart*>);
	}
}

bool LpuDataPartTracker::addDataPart(LpuDataPart *dataPart, const char *varName) {
	
	List<int> *partIndexList = partIndexMap->Lookup(varName);
	List<LpuDataPart*> *dataPartList = dataPartMap->Lookup(varName);
	int matchingIndex = -1;
	List<int*> *dataPartId = dataPart->getId();
	for (int i = 0; i < dataPartList->NumElements(); i++) {
		LpuDataPart *includedPart = dataPartList->Nth(i);
		if (includedPart->isMatchingId(dataPartId)) {
			matchingIndex = i;
			break;
		}
	}
	if (matchingIndex == -1) {
		partIndexList->Append(dataPartList->NumElements());	
		dataPartList->Append(dataPart);
		return true;
	} else {
		partIndexList->Append(matchingIndex);
		return false;
	}
}

bool LpuDataPartTracker::isAlreadyIncluded(List<int*> *dataPartId, const char *varName) {
	List<LpuDataPart*> *dataPartList = dataPartMap->Lookup(varName);
	for (int i = 0; i < dataPartList->NumElements(); i++) {
		LpuDataPart *includedPart = dataPartList->Nth(i);
		if (includedPart->isMatchingId(dataPartId)) return true;
	}
	return false;	
}

void LpuDataPartTracker::clear() {
	
	List<int> *indexList = NULL;
	Iterator<List<int>*> indexListIterator = partIndexMap->GetIterator();
	while ((indexList = indexListIterator.GetNextValue()) != NULL) {
		indexList->clear();
	}

	List<LpuDataPart*> *partList = NULL;
	Iterator<List<LpuDataPart*>*> partListIterator = dataPartMap->GetIterator();
	while ((partList = partListIterator.GetNextValue()) != NULL) {
		while (partList->NumElements() > 0) {
			LpuDataPart *dataPart = partList->Nth(0);
			partList->RemoveAt(0);
			delete dataPart;
		}
	}
}

//----------------------------------------------------------- Property Buffer Manager -----------------------------------------------------------/

PropertyBufferManager::PropertyBufferManager() {
	bufferSize = 0;
	bufferEntryCount = 0;
	bufferReferenceCount = 0;
	partRangeDepth = 0;
	cpuBuffer = NULL;
        cpuPartIndexBuffer = NULL;
	cpuPartRangeBuffer = NULL;
        cpuPartBeginningBuffer = NULL;
        gpuBuffer = NULL;
        gpuPartIndexBuffer = NULL;
	gpuPartRangeBuffer = NULL;
        gpuPartBeginningBuffer = NULL;
}

PropertyBufferManager::~PropertyBufferManager() {
	cleanupBuffers();
}

void PropertyBufferManager::prepareCpuBuffers(List<LpuDataPart*> *dataPartsList, 
		List<int> *partIndexList, 
		std::ofstream &logFile) {

	bufferEntryCount = dataPartsList->NumElements();
	bufferReferenceCount = partIndexList->NumElements();
	bufferSize = 0;
	for (int i = 0; i < dataPartsList->NumElements(); i++) {
		bufferSize += dataPartsList->Nth(i)->getSize();
	}

	cpuBuffer = (char *) malloc(bufferSize * sizeof(char));
	cpuPartBeginningBuffer = (int *) malloc(bufferEntryCount * sizeof(int));

	PartDimRanges *dimRanges = dataPartsList->Nth(0)->getPartDimRanges();
	partRangeDepth = dimRanges->getDepth();
	int dimRangeInfoSize = dimRanges->getSize();
	partRangeBufferSize = bufferEntryCount * dimRangeInfoSize;
	cpuPartRangeBuffer = (int *) malloc(partRangeBufferSize * sizeof(int));

	int currentIndex = 0;
	for (int i = 0; i < dataPartsList->NumElements(); i++) {
		
		LpuDataPart *dataPart = dataPartsList->Nth(i);
		int size = dataPart->getSize();
		char *dataStart = cpuBuffer + currentIndex;
		copyDataIntoBuffer(dataPart, dataStart);
		
		int *infoRangeStart = cpuPartRangeBuffer + (dimRangeInfoSize * i);
		dataPart->getPartDimRanges()->copyIntoBuffer(infoRangeStart);

		cpuPartBeginningBuffer[i] = currentIndex;
		currentIndex += size;
	}

	cpuPartIndexBuffer = (int *) malloc(bufferReferenceCount * sizeof(int));
	for (int i = 0; i < partIndexList->NumElements(); i++) {
		int index = partIndexList->Nth(i);
		cpuPartIndexBuffer[i] = index;
	}
}

void PropertyBufferManager::prepareGpuBuffers(std::ofstream &logFile) {
	
	check_error(cudaMalloc((void **) &gpuBuffer, bufferSize), logFile);
	check_error(cudaMemcpyAsync(gpuBuffer, cpuBuffer, bufferSize, cudaMemcpyHostToDevice, 0), logFile);
	
	check_error(cudaMalloc((void **) &gpuPartIndexBuffer, bufferReferenceCount * sizeof(int)), logFile);
	check_error(cudaMemcpyAsync(gpuPartIndexBuffer, cpuPartIndexBuffer, 
			bufferReferenceCount * sizeof(int), cudaMemcpyHostToDevice, 0), logFile);
	
	check_error(cudaMalloc((void **) &gpuPartRangeBuffer, partRangeBufferSize * sizeof(int)), logFile);
	check_error(cudaMemcpyAsync(gpuPartRangeBuffer, cpuPartRangeBuffer, 
			partRangeBufferSize * sizeof(int), cudaMemcpyHostToDevice, 0), logFile);
	
	check_error(cudaMalloc((void **) &gpuPartBeginningBuffer, bufferEntryCount * sizeof(int)), logFile);
	check_error(cudaMemcpyAsync(gpuPartBeginningBuffer, cpuPartBeginningBuffer, 
			bufferEntryCount * sizeof(int), cudaMemcpyHostToDevice, 0), logFile);
}

GpuBufferReferences *PropertyBufferManager::getGpuBufferReferences() {
	GpuBufferReferences *bufferRef = new GpuBufferReferences();
	bufferRef->dataBuffer = gpuBuffer;
	bufferRef->partIndexBuffer = gpuPartIndexBuffer;
	bufferRef->partRangeBuffer = gpuPartRangeBuffer;
	bufferRef->partBeginningBuffer = gpuPartBeginningBuffer;
	bufferRef->partRangeDepth = partRangeDepth;
	return bufferRef;
}

void PropertyBufferManager::syncDataPartsFromBuffer(List<LpuDataPart*> *dataPartsList, std::ofstream &logFile) {
	
	check_error(cudaMemcpy(cpuBuffer, gpuBuffer, bufferSize, cudaMemcpyDeviceToHost), logFile);

	int currentIndex = 0;
	for (int i = 0; i < dataPartsList->NumElements(); i++) {
		LpuDataPart *dataPart = dataPartsList->Nth(i);
		char *dataStart = cpuBuffer + currentIndex;
		void *data = dataPart->getData();
		int size = dataPart->getSize();
		memcpy(data, dataStart, size);
		currentIndex += size;
	}
}

void PropertyBufferManager::cleanupBuffers() {
	
	bufferSize = 0;
	bufferEntryCount = 0;
	bufferReferenceCount = 0;
	partRangeDepth = 0;
	
	free(cpuBuffer);
	cpuBuffer = NULL;
	free(cpuPartIndexBuffer); 
        cpuPartIndexBuffer = NULL;
	free(cpuPartRangeBuffer); 
        cpuPartRangeBuffer = NULL;
	free(cpuPartBeginningBuffer);
        cpuPartBeginningBuffer = NULL;
       
	cudaFree(gpuBuffer); 
	gpuBuffer = NULL;
	cudaFree(gpuPartIndexBuffer);
        gpuPartIndexBuffer = NULL;
	cudaFree(gpuPartRangeBuffer);
        gpuPartRangeBuffer = NULL;
	cudaFree(gpuPartBeginningBuffer);
        gpuPartBeginningBuffer = NULL;
}

void PropertyBufferManager::copyDataIntoBuffer(LpuDataPart *dataPart, char *dataBuffer) {
	void *data = dataPart->getData();
	int size = dataPart->getSize();
	memcpy(dataBuffer, data, size);
}

//------------------------------------------------------- Versioned Property Buffer Manager -----------------------------------------------------/

void VersionedPropertyBufferManager::prepareCpuBuffers(List<LpuDataPart*> *dataPartsList,
                        List<int> *partIndexList, std::ofstream &logFile) {
	
	PropertyBufferManager::prepareCpuBuffers(dataPartsList, partIndexList, logFile);
	cpuDataPartVersions = new short[bufferEntryCount];
	for (int i = 0; i < bufferEntryCount; i++) {

		// at the beginning all parts being staged in to the GPU should have 0 as the current version as the most 
		// up-to-date version is copied to the beginning of the data buffer.
		cpuDataPartVersions[i] = 0;
	}
	VersionedLpuDataPart *firstPart = (VersionedLpuDataPart *) dataPartsList->Nth(0);
	versionCount = firstPart->getVersionCount();
}

void VersionedPropertyBufferManager::prepareGpuBuffers(std::ofstream &logFile) {
	PropertyBufferManager::prepareGpuBuffers(logFile);
	check_error(cudaMalloc((void **) &gpuDataPartVersions, bufferEntryCount * sizeof(short)), logFile);
	check_error(cudaMemcpyAsync(gpuDataPartVersions, cpuDataPartVersions, 
			bufferEntryCount * sizeof(short), cudaMemcpyHostToDevice, 0), logFile);
}

GpuBufferReferences *VersionedPropertyBufferManager::getGpuBufferReferences() {
	
	VersionedGpuBufferReferences *bufferRef = new VersionedGpuBufferReferences();
	
	bufferRef->dataBuffer = gpuBuffer;
	bufferRef->partIndexBuffer = gpuPartIndexBuffer;
	bufferRef->partRangeBuffer = gpuPartRangeBuffer;
	bufferRef->partBeginningBuffer = gpuPartBeginningBuffer;
	bufferRef->partRangeDepth = partRangeDepth;

	bufferRef->versionCount = versionCount;
	bufferRef->versionIndexBuffer = gpuDataPartVersions;

	return bufferRef;
}

void VersionedPropertyBufferManager::syncDataPartsFromBuffer(List<LpuDataPart*> *dataPartsList, std::ofstream &logFile) {

	check_error(cudaMemcpyAsync(cpuBuffer, gpuBuffer, bufferSize, cudaMemcpyDeviceToHost, 0), logFile);
	check_error(cudaMemcpy(cpuDataPartVersions, gpuDataPartVersions, 
			bufferEntryCount * sizeof(short), cudaMemcpyDeviceToHost), logFile);

	int currentIndex = 0;
	for (int i = 0; i < bufferEntryCount; i++) {
		VersionedLpuDataPart *dataPart = (VersionedLpuDataPart *) dataPartsList->Nth(i);
		short currVersion = cpuDataPartVersions[i];
		int sizePerVersion = dataPart->getSize() / versionCount;
		for (int j = 0; j < versionCount; j++) {
			int versionIndex = (currVersion + j) % versionCount;
			char *dataStart = cpuBuffer + currentIndex + versionIndex * sizePerVersion;
			void *data = dataPart->getDataVersion(versionIndex);
			memcpy(data, dataStart, sizePerVersion);
		}
		currentIndex += dataPart->getSize();
	}
}

void VersionedPropertyBufferManager::cleanupBuffers() {
	
	PropertyBufferManager::cleanupBuffers();

	free(cpuDataPartVersions);
        cpuDataPartVersions = NULL;       
	cudaFree(gpuDataPartVersions); 
	gpuDataPartVersions = NULL;
}

void VersionedPropertyBufferManager::copyDataIntoBuffer(LpuDataPart *dataPart, char *dataBuffer) {
	VersionedLpuDataPart *versionedPart = (VersionedLpuDataPart *) dataPart;
	int versionCount = versionedPart->getVersionCount();
	int sizePerVersion = versionedPart->getSize() / versionCount;
	for (int i = 0; i < versionCount; i++) {
		void *data = versionedPart->getDataVersion(i);
		char *dataStart = dataBuffer + i * sizePerVersion;
		memcpy(dataStart, data, sizePerVersion);
	}
}

//------------------------------------------------------------ LPU Data Buffer Manager ----------------------------------------------------------/

LpuDataBufferManager::LpuDataBufferManager(List<const char*> *propertyNames) {
	propertyBuffers = new Hashtable<PropertyBufferManager*>;
	for (int i = 0; i < propertyNames->NumElements(); i++) {
		const char *propertyName = propertyNames->Nth(i);
		propertyBuffers->Enter(propertyName, new PropertyBufferManager());
	}
}

LpuDataBufferManager::LpuDataBufferManager(List<const char*> *versionlessProperties, 
                        List<const char*> *multiversionProperties) {
	
	propertyBuffers = new Hashtable<PropertyBufferManager*>;

	for (int i = 0; i < versionlessProperties->NumElements(); i++) {
		const char *propertyName = versionlessProperties->Nth(i);
		propertyBuffers->Enter(propertyName, new PropertyBufferManager());
	}

	for (int i = 0; i < multiversionProperties->NumElements(); i++) {
		const char *propertyName = multiversionProperties->Nth(i);
		propertyBuffers->Enter(propertyName, new VersionedPropertyBufferManager());
	}
}

void LpuDataBufferManager::copyPartsInGpu(const char *propertyName, 
		List<LpuDataPart*> *dataPartsList, 
		List<int> *partIndexList,
		std::ofstream &logFile) {
	PropertyBufferManager *propertyManager = propertyBuffers->Lookup(propertyName);
	propertyManager->prepareCpuBuffers(dataPartsList, partIndexList, logFile);
	propertyManager->prepareGpuBuffers(logFile);
}

GpuBufferReferences *LpuDataBufferManager::getGpuBufferReferences(const char *propertyName) {
	PropertyBufferManager *propertyManager = propertyBuffers->Lookup(propertyName);
	return propertyManager->getGpuBufferReferences();	
}

void LpuDataBufferManager::retrieveUpdatesFromGpu(const char *propertyName, 
		List<LpuDataPart*> *dataPartsList, 
		std::ofstream &logFile) {
	PropertyBufferManager *propertyManager = propertyBuffers->Lookup(propertyName);
	propertyManager->syncDataPartsFromBuffer(dataPartsList, logFile);
}

void LpuDataBufferManager::reset() {
	PropertyBufferManager *propertyManager = NULL;
	Iterator<PropertyBufferManager*> iterator = propertyBuffers->GetIterator();
	while ((propertyManager = iterator.GetNextValue()) != NULL) {
		propertyManager->cleanupBuffers();
	}
}

//------------------------------------------------------------- LPU Batch Controller ------------------------------------------------------------/

LpuBatchController::LpuBatchController() {
	propertyNames = NULL;
	toBeModifiedProperties = NULL;
	batchLpuCountThreshold = 1;
	currentBatchSize = 0;
	gpuMemStat = NULL;
	dataPartTracker = NULL;
	bufferManager = NULL;
	this->logFile = NULL;
}

void LpuBatchController::initialize(int lpuCountThreshold, 
		long memoryConsumptionLimit, 
		List<const char*> *propertyNames,
		List<const char*> *toBeModifiedProperties) {

	this->propertyNames = propertyNames;
	this->toBeModifiedProperties = toBeModifiedProperties;
	batchLpuCountThreshold = lpuCountThreshold;
	currentBatchSize = 0;
	gpuMemStat = new GpuMemoryConsumptionStat(memoryConsumptionLimit);
	dataPartTracker = new LpuDataPartTracker();
	dataPartTracker->initialize(propertyNames);
}

bool LpuBatchController::canHoldLpu(LPU *lpu) {
	int moreMemoryNeeded = calculateLpuMemoryRequirement(lpu);
	return gpuMemStat->canHoldLpu(moreMemoryNeeded);
}

void LpuBatchController::submitCurrentBatchToGpu() {
	if (currentBatchSize > 0) {
		for (int i = 0; i < propertyNames->NumElements(); i++) {
			const char *varName = propertyNames->Nth(i);
			List<int> *partIndexes = dataPartTracker->getPartIndexList(varName);
			List<LpuDataPart*> *parts = dataPartTracker->getDataPartList(varName);
			bufferManager->copyPartsInGpu(varName, parts, partIndexes, *logFile);
		}		
	}
}

void LpuBatchController::updateBatchDataPartsFromGpuResults() {
	if (currentBatchSize > 0) {
		for (int i = 0; i < toBeModifiedProperties->NumElements(); i++) {
			const char *property = toBeModifiedProperties->Nth(i);
			List<LpuDataPart*> *parts = dataPartTracker->getDataPartList(property);
			bufferManager->retrieveUpdatesFromGpu(property, parts, *logFile);
		}
	}
}

void LpuBatchController::resetController() {
	if (currentBatchSize > 0) {
		currentBatchSize = 0;
		dataPartTracker->clear();
		bufferManager->reset();
		gpuMemStat->reset();
	}
}
