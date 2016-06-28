#include "lpu_parts_tracking.h"
#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "../runtime/structure.h"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>

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

void PartDimRanges::describe(std::ofstream &stream, int indentLevel) {
	std::ostringstream indent;
	for (int i = 0; i < indentLevel; i++) indent << '\t';
	stream << indent.str() << "Part Dimension Range: [";
	for (int i = 0; i < size; i++) {
		stream << "(" << ranges[i].min << ", " << ranges[i].max << ")";
	}
	stream << "]\n";	
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
	for (int i = 0; i < candidateId->NumElements(); i++) {
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

void LpuDataPart::describe(std::ofstream &stream, int indentLevel) {
	std::ostringstream indent;
	for (int i = 0; i < indentLevel; i++) indent << '\t';
	stream << indent.str() << "Data Part Reference: " << this << "\n";
	indent << '\t';
	stream << indent.str() << "Dimensionality: " << dimensionality << "\n";
	stream << indent.str() << "Element Count: " << elementCount << "\n";
	stream << indent.str() << "Element Size: " << elementSize << "\n";
	stream << indent.str() << "Total Size: " << getSize() << "\n"; 	
	partDimRanges->describe(stream, indentLevel + 1);
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

void VersionedLpuDataPart::describe(std::ofstream &stream, int indentLevel) {
	LpuDataPart::describe(stream, indentLevel);
	std::ostringstream indent;
        for (int i = 0; i <= indentLevel; i++) indent << '\t';
	stream << indent.str() << "Version Count: " << versionCount << "\n";
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

LpuDataPartTracker::LpuDataPartTracker(int distinctPpuCount) {
	this->distinctPpuCount = distinctPpuCount;
	partIndexMap = new Hashtable<std::vector<List<int>*>*>;
	dataPartMap = new Hashtable<List<LpuDataPart*>*>;
	this->logFile = NULL;
}

void LpuDataPartTracker::initialize(List<const char*> *varNames) {
	for (int i = 0; i < varNames->NumElements(); i++) {
		const char *varName = varNames->Nth(i);
		std::vector<List<int>*> *partIndexVector = new std::vector<List<int>*>;
		partIndexVector->reserve(distinctPpuCount);
		for (int j = 0; j < distinctPpuCount; j++) {
			partIndexVector->push_back(new List<int>);	
		}
		partIndexMap->Enter(varName, partIndexVector);
		dataPartMap->Enter(varName, new List<LpuDataPart*>);
	}
}

bool LpuDataPartTracker::addDataPart(LpuDataPart *dataPart, const char *varName, int ppuIndex) {
	
	std::vector<List<int>*> *partIndexListVector = partIndexMap->Lookup(varName);
	List<int> *partIndexList = partIndexListVector->at(ppuIndex);
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
	
	std::vector<List<int>*> *indexListVector = NULL;
	Iterator<std::vector<List<int>*>*> indexListIterator = partIndexMap->GetIterator();
	
	while ((indexListVector = indexListIterator.GetNextValue()) != NULL) {
		for (int i = 0; i < distinctPpuCount; i++) {
			List<int> *indexList = indexListVector->at(i);
			indexList->clear();
		}
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
	logFile = NULL;
}

PropertyBufferManager::~PropertyBufferManager() {
	cleanupBuffers();
}

void PropertyBufferManager::prepareCpuBuffers(List<LpuDataPart*> *dataPartsList, 
		std::vector<List<int>*> *partIndexListVector) {

	bufferEntryCount = dataPartsList->NumElements();
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

	bufferReferenceCount = 0;
	for (unsigned int i = 0; i < partIndexListVector->size(); i++) {
		List<int> *partIndexList = partIndexListVector->at(i);
		bufferReferenceCount += partIndexList->NumElements();
	}
	cpuPartIndexBuffer = (int *) malloc(bufferReferenceCount * sizeof(int));
	currentIndex = 0;
	for (unsigned int l = 0; l < partIndexListVector->size(); l++) {
		List<int> *partIndexList = partIndexListVector->at(l);
		for (int i = 0; i < partIndexList->NumElements(); i++) {
			int index = partIndexList->Nth(i);
			cpuPartIndexBuffer[currentIndex] = index;
			currentIndex++;
		}
	}
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

void PropertyBufferManager::copyDataIntoBuffer(LpuDataPart *dataPart, char *dataBuffer) {
	void *data = dataPart->getData();
	int size = dataPart->getSize();
	memcpy(dataBuffer, data, size);
}

//------------------------------------------------------- Versioned Property Buffer Manager -----------------------------------------------------/

void VersionedPropertyBufferManager::prepareCpuBuffers(List<LpuDataPart*> *dataPartsList,
                        std::vector<List<int>*> *partIndexListVector) {
	
	PropertyBufferManager::prepareCpuBuffers(dataPartsList, partIndexListVector);
	cpuDataPartVersions = new short[bufferEntryCount];
	for (int i = 0; i < bufferEntryCount; i++) {

		// at the beginning all parts being staged in to the GPU should have 0 as the current version as the most 
		// up-to-date version is copied to the beginning of the data buffer.
		cpuDataPartVersions[i] = 0;
	}
	VersionedLpuDataPart *firstPart = (VersionedLpuDataPart *) dataPartsList->Nth(0);
	versionCount = firstPart->getVersionCount();
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
	logFile = NULL;
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

void LpuDataBufferManager::setLogFile(std::ofstream *logFile) { 
	this->logFile = logFile;
	PropertyBufferManager *propertyManager = NULL;
        Iterator<PropertyBufferManager*> iterator = propertyBuffers->GetIterator();
        while ((propertyManager = iterator.GetNextValue()) != NULL) {
                propertyManager->setLogFile(logFile);
        } 
}

void LpuDataBufferManager::copyPartsInGpu(const char *propertyName, 
		List<LpuDataPart*> *dataPartsList, 
		std::vector<List<int>*> *partIndexListVector) {
	PropertyBufferManager *propertyManager = propertyBuffers->Lookup(propertyName);
	propertyManager->prepareCpuBuffers(dataPartsList, partIndexListVector);
	propertyManager->prepareGpuBuffers();
}

GpuBufferReferences *LpuDataBufferManager::getGpuBufferReferences(const char *propertyName) {
	PropertyBufferManager *propertyManager = propertyBuffers->Lookup(propertyName);
	return propertyManager->getGpuBufferReferences();	
}

void LpuDataBufferManager::retrieveUpdatesFromGpu(const char *propertyName, 
		List<LpuDataPart*> *dataPartsList) {
	PropertyBufferManager *propertyManager = propertyBuffers->Lookup(propertyName);
	propertyManager->syncDataPartsFromBuffer(dataPartsList);
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
		int distinctPpuCount,
		List<const char*> *propertyNames,
		List<const char*> *toBeModifiedProperties) {

	this->propertyNames = propertyNames;
	this->toBeModifiedProperties = toBeModifiedProperties;
	batchLpuCountThreshold = lpuCountThreshold;
	currentBatchSize = 0;
	this->distinctPpuCount = distinctPpuCount;
	gpuMemStat = new GpuMemoryConsumptionStat(memoryConsumptionLimit);
	dataPartTracker = new LpuDataPartTracker(distinctPpuCount);
	dataPartTracker->initialize(propertyNames);
}

void LpuBatchController::setLogFile(std::ofstream *logFile) { 
	this->logFile = logFile;
	dataPartTracker->setLogFile(logFile);
	bufferManager->setLogFile(logFile); 
} 

bool LpuBatchController::canHoldLpu(LPU *lpu) {
	int moreMemoryNeeded = calculateLpuMemoryRequirement(lpu);
	return gpuMemStat->canHoldLpu(moreMemoryNeeded);
}

void LpuBatchController::submitCurrentBatchToGpu() {
	if (currentBatchSize > 0) {
		for (int i = 0; i < propertyNames->NumElements(); i++) {
			const char *varName = propertyNames->Nth(i);
			std::vector<List<int>*> *partIndexVector 
					= dataPartTracker->getPartIndexListVector(varName);
			List<LpuDataPart*> *parts = dataPartTracker->getDataPartList(varName);
			bufferManager->copyPartsInGpu(varName, parts, partIndexVector);
		}		
	}
}

void LpuBatchController::updateBatchDataPartsFromGpuResults() {
	if (currentBatchSize > 0) {
		for (int i = 0; i < toBeModifiedProperties->NumElements(); i++) {
			const char *property = toBeModifiedProperties->Nth(i);
			List<LpuDataPart*> *parts = dataPartTracker->getDataPartList(property);
			bufferManager->retrieveUpdatesFromGpu(property, parts);
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
