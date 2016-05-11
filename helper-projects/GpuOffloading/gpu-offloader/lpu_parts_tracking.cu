#include "lpu_parts_tracking.h"
#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "../runtime/structure.h"

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

	logFile << "Preparing the CPU buffers for the batch\n";
	logFile.flush();
	
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

	logFile << "Allocated the part data, beginning index, and range buffers\n";
	logFile.flush();

	int currentIndex = 0;
	for (int i = 0; i < dataPartsList->NumElements(); i++) {
		
		LpuDataPart *dataPart = dataPartsList->Nth(i);
		void *data = dataPart->getData();
		int size = dataPart->getSize();
		char *dataStart = cpuBuffer + currentIndex;
		memcpy(dataStart, data, size);

		int *infoRangeStart = cpuPartRangeBuffer + (dimRangeInfoSize * i);
		dataPart->getPartDimRanges()->copyIntoBuffer(infoRangeStart);

		cpuPartBeginningBuffer[i] = currentIndex;
		currentIndex += size;
	}

	logFile << "Copied data into the previously allocated buffers; going to prepared the part index buffer\n";
	logFile.flush();

	cpuPartIndexBuffer = (int *) malloc(bufferReferenceCount * sizeof(int));
	for (int i = 0; i < partIndexList->NumElements(); i++) {
		int index = partIndexList->Nth(i);
		int partStartsAt = cpuPartBeginningBuffer[index];
		cpuPartIndexBuffer[i] = partStartsAt;
	}
}

void PropertyBufferManager::prepareGpuBuffers(std::ofstream &logFile) {
	
	logFile << "Copying buffers from the CPU to the GPU\n";
	logFile.flush();

	cudaMalloc((void **) &gpuBuffer, bufferSize);
	cudaMemcpy(gpuBuffer, cpuBuffer, bufferSize, cudaMemcpyHostToDevice);
	
	cudaMalloc((void **) &gpuPartIndexBuffer, bufferReferenceCount * sizeof(int));
	cudaMemcpy(gpuPartIndexBuffer, cpuPartIndexBuffer, 
			bufferReferenceCount * sizeof(int), cudaMemcpyHostToDevice);
	
	cudaMalloc((void **) &gpuPartRangeBuffer, partRangeBufferSize);
	cudaMemcpy(gpuPartRangeBuffer, cpuPartRangeBuffer, partRangeBufferSize, cudaMemcpyHostToDevice);
	
	cudaMalloc((void **) &gpuPartBeginningBuffer, bufferEntryCount * sizeof(int));
	cudaMemcpy(gpuPartBeginningBuffer, cpuPartBeginningBuffer, 
			bufferEntryCount * sizeof(int), cudaMemcpyHostToDevice);
}

GpuBufferReferences PropertyBufferManager::getGpuBufferReferences() {
	GpuBufferReferences bufferRef;
	bufferRef.dataBuffer = gpuBuffer;
	bufferRef.partIndexBuffer = gpuPartIndexBuffer;
	bufferRef.partRangeBuffer = gpuPartRangeBuffer;
	bufferRef.partBeginningBuffer = gpuPartBeginningBuffer;
	bufferRef.partRangeDepth = partRangeDepth;
	return bufferRef;
}

void PropertyBufferManager::syncDataPartsFromBuffer(List<LpuDataPart*> *dataPartsList, std::ofstream &logFile) {
	
	logFile << "Retrieving updated data from the GPU to synchronize data parts in the CPU\n";
	logFile.flush();

	cudaMemcpy(cpuBuffer, gpuBuffer, bufferSize, cudaMemcpyDeviceToHost);

	int currentIndex = 0;
	for (int i = 0; i < dataPartsList->NumElements(); i++) {
		LpuDataPart *dataPart = dataPartsList->Nth(i);
		void *data = dataPart->getData();
		int size = dataPart->getSize();
		char *start = cpuBuffer + currentIndex;
		memcpy(data, start, size);
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

//------------------------------------------------------------ LPU Data Buffer Manager ----------------------------------------------------------/

LpuDataBufferManager::LpuDataBufferManager(List<const char*> *propertyNames) {
	propertyBuffers = new Hashtable<PropertyBufferManager*>;
	for (int i = 0; i < propertyNames->NumElements(); i++) {
		const char *propertyName = propertyNames->Nth(i);
		propertyBuffers->Enter(propertyName, new PropertyBufferManager());
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

GpuBufferReferences LpuDataBufferManager::getGpuBufferReferences(const char *propertyName) {
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
	bufferManager = new LpuDataBufferManager(propertyNames);
}

bool LpuBatchController::canHoldLpu(LPU *lpu) {
	int moreMemoryNeeded = calculateLpuMemoryRequirement(lpu);
	return gpuMemStat->canHoldLpu(moreMemoryNeeded);
}

void LpuBatchController::submitCurrentBatchToGpu() {
	if (currentBatchSize > 0) {
		for (int i = 0; i < propertyNames->NumElements(); i++) {
			const char *varName = propertyNames->Nth(i);
			
			*logFile << "Preparing buffers for property '" << varName << "'\n";
			logFile->flush();
			
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
