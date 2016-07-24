#include "comm_buffer.h"
#include "confinement_mgmt.h"
#include "data_transfer.h"
#include "part_config.h"
#include "../memory-management/allocation.h"
#include "../memory-management/part_tracking.h"
#include "../utils/list.h"
#include "../utils/binary_search.h"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdlib>
#include <vector>
#include <cstring>

using namespace std;

//---------------------------------------------------- Communication Buffer ------------------------------------------------------/

CommBuffer::CommBuffer(DataExchange *exchange, SyncConfig *syncConfig) {

	dataExchange = exchange;
	senderPartList = syncConfig->getSenderDataParts();
	receiverPartList = syncConfig->getReceiverDataParts();

	ConfinementConstructionConfig *confinementConfig = syncConfig->getConfinementConfig();
	dataDimensions = confinementConfig->getDataDimensions();
	elementCount = exchange->getTotalElementsCount();
	elementSize = syncConfig->getElementSize();
	localSegmentTag = confinementConfig->getLocalSegmentTag();
	senderTree = confinementConfig->getSenderPartTree();
	receiverTree = confinementConfig->getReceiverPartTree();
	senderDataConfig = confinementConfig->getSenderConfig();
	receiverDataConfig = confinementConfig->getReceiverConfig();

	bufferTag = 0;
}

bool CommBuffer::isSendActivated() {
	return dataExchange->getSender()->hasSegmentTag(localSegmentTag);
}

bool CommBuffer::isReceiveActivated() {
	return dataExchange->getReceiver()->hasSegmentTag(localSegmentTag);
}

int CommBuffer::compareTo(CommBuffer *other, bool forReceive) { 
	return dataExchange->compareTo(other->dataExchange, forReceive); 
}

void CommBuffer::describe(std::ostream &stream, int indentation) {
	std::ostringstream indent;
	for (int i = 0; i < indentation; i++) indent << '\t';
	stream << indent.str() << "Tag: " << bufferTag << "\n";
	stream << indent.str() << "Element Count: " << elementCount << "\n";
	dataExchange->describe(indentation, stream);	
}

char *CommBuffer::getData() {
	std::cout << "buffer content reference is not valid for all communication buffer types";
	std::exit(EXIT_FAILURE);
}

void CommBuffer::setData(char *data) {
	std::cout << "buffer content reference cannot be set in all communication buffer types";
	std::exit(EXIT_FAILURE);
}

void CommBuffer::setBufferTag(int prefix, int digitsForSegment) {
	
	int firstSenderTag = dataExchange->getSender()->getSegmentTags().at(0);
	int firstReceiverTag = dataExchange->getReceiver()->getSegmentTags().at(0);

	ostringstream ostream;
	ostream << prefix;
	ostream << std::setfill('0') << std::setw(digitsForSegment) << firstSenderTag;
	ostream << std::setfill('0') << std::setw(digitsForSegment) << firstReceiverTag;
	istringstream istream(ostream.str());
	istream >> this->bufferTag;
}

//---------------------------------------------- Pre-processed Communication Buffer ----------------------------------------------/

PreprocessedCommBuffer::PreprocessedCommBuffer(DataExchange *ex, SyncConfig *sC) : CommBuffer(ex, sC) {
	senderTransferMapping = NULL;
	receiverTransferMapping = NULL;
	if (isSendActivated()) {
		senderTransferMapping = new char*[elementCount];
		setupMappingBuffer(senderTransferMapping, senderPartList, senderTree, senderDataConfig);
	}
	if (isReceiveActivated()) {
		receiverTransferMapping = new char*[elementCount];
		setupMappingBuffer(receiverTransferMapping, receiverPartList, receiverTree, receiverDataConfig);
	}
}

PreprocessedCommBuffer::~PreprocessedCommBuffer() {
	if (senderTransferMapping != NULL) delete[] senderTransferMapping;
	if (receiverTransferMapping != NULL) delete[] receiverTransferMapping;
}

void PreprocessedCommBuffer::setupMappingBuffer(char **buffer,
		DataPartsList *dataPartList,
		PartIdContainer *partContainerTree,
		DataItemConfig *dataConfig) {

	DataPartSpec *dataPartSpec = new DataPartSpec(dataPartList->getPartList(), dataConfig);
	vector<XformedIndexInfo*> *transformVector = new vector<XformedIndexInfo*>;
	transformVector->reserve(dataDimensions);
	for (int i = 0; i < dataDimensions; i++) {
		transformVector->push_back(new XformedIndexInfo());
	}

	ExchangeIterator *iterator = getIterator();
	int elementIndex = 0;
	TransferLocationSpec *transferSpec = new TransferLocationSpec(elementSize);
	while (iterator->hasMoreElements()) {
		vector<int> *dataItemIndex = iterator->getNextElement();
		dataPartSpec->initPartTraversalReference(dataItemIndex, transformVector);
		transferSpec->setBufferLocation(&buffer[elementIndex]);
		partContainerTree->transferData(transformVector, 
				transferSpec, dataPartSpec, false, std::cout);
		elementIndex++;
	}

	delete dataPartSpec;
	delete transformVector;
	delete transferSpec;
}

//----------------------------------------------- Index-Mapped Communication Buffer ----------------------------------------------/

IndexMappedCommBuffer::IndexMappedCommBuffer(DataExchange *ex, SyncConfig *sC) : CommBuffer(ex, sC) {
	senderTransferIndexMapping = NULL;
	receiverTransferIndexMapping = NULL;
	if (isSendActivated()) {
		senderTransferIndexMapping = new DataPartIndex[elementCount];
		setupMappingBuffer(senderTransferIndexMapping, senderPartList, senderTree, senderDataConfig);
	}
	if (isReceiveActivated()) {
		receiverTransferIndexMapping = new DataPartIndex[elementCount];
		setupMappingBuffer(receiverTransferIndexMapping, 
				receiverPartList, receiverTree, receiverDataConfig);
	}
}
        
IndexMappedCommBuffer::~IndexMappedCommBuffer() {
	if (senderTransferIndexMapping != NULL) delete[] senderTransferIndexMapping;
	if (receiverTransferIndexMapping != NULL) delete[] receiverTransferIndexMapping;
}

void IndexMappedCommBuffer::setupMappingBuffer(DataPartIndex *indexMappingBuffer,
                        DataPartsList *dataPartList,
                        PartIdContainer *partContainerTree,
                        DataItemConfig *dataConfig) {
	
	DataPartSpec *dataPartSpec = new DataPartSpec(dataPartList->getPartList(), dataConfig);
	vector<XformedIndexInfo*> *transformVector = new vector<XformedIndexInfo*>;
	transformVector->reserve(dataDimensions);
	for (int i = 0; i < dataDimensions; i++) {
		transformVector->push_back(new XformedIndexInfo());
	}

	ExchangeIterator *iterator = getIterator();
	int elementIndex = 0;
	TransferIndexSpec *transferSpec = new TransferIndexSpec(elementSize);
	while (iterator->hasMoreElements()) {
		vector<int> *dataItemIndex = iterator->getNextElement();
		dataPartSpec->initPartTraversalReference(dataItemIndex, transformVector);
		transferSpec->setPartIndexReference(&indexMappingBuffer[elementIndex]);
		partContainerTree->transferData(transformVector, 
				transferSpec, dataPartSpec, false, std::cout);
		elementIndex++;
	}

	delete dataPartSpec;
	delete transformVector;
	delete transferSpec;
}

//------------------------------------------------ Physical Communication Buffer -------------------------------------------------/

PhysicalCommBuffer::PhysicalCommBuffer(DataExchange *e, SyncConfig *s) : CommBuffer(e, s) {
	int bufferSize = elementCount * elementSize;
	data = new char[bufferSize];
	for (int i = 0; i < bufferSize; i++) {
		data[i] = 0;
	}
}

void PhysicalCommBuffer::readData(bool loggingEnabled, std::ostream &logFile) {
	if (isSendActivated()) {
		if (loggingEnabled) {
			logFile << "start reading data for communication buffer: " << bufferTag << "\n";
		}
		
		DataPartSpec *partSpec = new DataPartSpec(senderPartList->getPartList(), senderDataConfig);
		TransferSpec *readSpec = new TransferSpec(DATA_PART_TO_COMM_BUFFER, elementSize);
		transferData(readSpec, partSpec, senderTree, loggingEnabled, logFile);
		delete partSpec;
		delete readSpec;
		
		if (loggingEnabled) {
			logFile << "reading done for communication buffer: " << bufferTag << "\n";
		}
	}
}

void PhysicalCommBuffer::writeData(bool loggingEnabled, std::ostream &logFile) {
	if (isReceiveActivated()) {
		if (loggingEnabled) {
			logFile << "start writing data from communication buffer: " << bufferTag << "\n";
		}
		
		DataPartSpec *partSpec = new DataPartSpec(receiverPartList->getPartList(), receiverDataConfig);
		TransferSpec *writeSpec = new TransferSpec(COMM_BUFFER_TO_DATA_PART, elementSize);
		transferData(writeSpec, partSpec, receiverTree, loggingEnabled, logFile);
		delete partSpec;
		delete writeSpec;

		if (loggingEnabled) {
			logFile << "writing done for communication buffer: " << bufferTag << "\n";
		}
	}
}

void PhysicalCommBuffer::transferData(TransferSpec *transferSpec,
		DataPartSpec *dataPartSpec,
		PartIdContainer *partTree, 
		bool loggingEnabled, std::ostream &logFile) {

	vector<XformedIndexInfo*> *transformVector = new vector<XformedIndexInfo*>;
	transformVector->reserve(dataDimensions);
	for (int i = 0; i < dataDimensions; i++) {
		transformVector->push_back(new XformedIndexInfo());
	}
	ExchangeIterator *iterator = getIterator();
	int elementIndex = 0;
	int transferRequests = 0;
	int transferCount = 0;
	while (iterator->hasMoreElements()) {
		vector<int> *dataItemIndex = iterator->getNextElement();
		if (loggingEnabled) {
			logFile << "\t\tTransfer Request for: (";
			for (int i = 0; i < dataItemIndex->size(); i++) {
				if (i > 0) logFile << ',';
				logFile << dataItemIndex->at(i);
			}
			logFile << ")\n";
		}
		dataPartSpec->initPartTraversalReference(dataItemIndex, transformVector);
		char *dataLocation = data + elementIndex * elementSize;
		transferSpec->setBufferEntry(dataLocation, dataItemIndex);
		int elementTransfers = partTree->transferData(transformVector, 
				transferSpec, 
				dataPartSpec, loggingEnabled, logFile, 3);
		if (loggingEnabled) {
			logFile << "\t\tData Transfers for Request: ";
			logFile << elementTransfers << "\n";
		}
		elementIndex++;
		transferRequests++;
		transferCount += elementTransfers;
	}
	delete transformVector;
	
	if (loggingEnabled) {
		logFile << "\tTotal transfer requests: " << transferRequests << "\n";
		logFile << "\tTotal data transfers for those requests: " << transferCount << "\n";
	}
}

//------------------------------------------ Pre-processed Physical Communication Buffer -----------------------------------------/

PreprocessedPhysicalCommBuffer::PreprocessedPhysicalCommBuffer(DataExchange *exchange,
		SyncConfig *syncConfig) : PreprocessedCommBuffer(exchange, syncConfig) {
	
	int bufferSize = elementCount * elementSize;
	data = new char[bufferSize];
	for (int i = 0; i < bufferSize; i++) {
		data[i] = 0;
	}
}

void PreprocessedPhysicalCommBuffer::readData(bool loggingEnabled, std::ostream &logFile) {
	for (int i = 0; i < elementCount; i++) {
		char *readLocation = senderTransferMapping[i];
		char *writeLocation = data + i * elementSize;
		memcpy(writeLocation, readLocation, elementSize);
	}
}

void PreprocessedPhysicalCommBuffer::writeData(bool loggingEnabled, std::ostream &logFile) {
	for (int i = 0; i < elementCount; i++) {
		char *readLocation = data + i * elementSize;
		char *writeLocation = receiverTransferMapping[i];
		memcpy(writeLocation, readLocation, elementSize);
	}
}

//------------------------------------------- Index-mapped Physical Communication Buffer -----------------------------------------/

IndexMappedPhysicalCommBuffer::IndexMappedPhysicalCommBuffer(DataExchange *exchange, 
		SyncConfig *syncConfig) : IndexMappedCommBuffer(exchange, syncConfig) {
	
	int bufferSize = elementCount * elementSize;
        data = new char[bufferSize];
        for (int i = 0; i < bufferSize; i++) {
                data[i] = 0;
        }
}

void IndexMappedPhysicalCommBuffer::readData(bool loggingEnabled, std::ostream &logFile) {
	for (int i = 0; i < elementCount; i++) {
                char *readLocation = senderTransferIndexMapping[i].getLocation();
                char *writeLocation = data + i * elementSize;
                memcpy(writeLocation, readLocation, elementSize);
        }
}
        
void IndexMappedPhysicalCommBuffer::writeData(bool loggingEnabled, std::ostream &logFile) {
	for (int i = 0; i < elementCount; i++) {
                char *readLocation = data + i * elementSize;
                char *writeLocation = receiverTransferIndexMapping[i].getLocation();
                memcpy(writeLocation, readLocation, elementSize);
        }
}

//------------------------------------------------- Virtual Communication Buffer -------------------------------------------------/

void VirtualCommBuffer::readData(bool loggingEnabled, std::ostream &logFile) {
	if (isReceiveActivated()) {
		if (loggingEnabled) {
			logFile << "start transferring data for communication buffer ";
			logFile << bufferTag << "\n";
		}
	
		DataPartSpec *readPartSpec = new DataPartSpec(senderPartList->getPartList(), senderDataConfig);
		TransferSpec *readTransferSpec = new TransferSpec(DATA_PART_TO_COMM_BUFFER, elementSize);

		DataPartSpec *writePartSpec = new DataPartSpec(receiverPartList->getPartList(), receiverDataConfig);
		TransferSpec *writeTransferSpec = new TransferSpec(COMM_BUFFER_TO_DATA_PART, elementSize);

		Assert(senderDataConfig != receiverDataConfig);

		char *dataEntry = new char[elementSize];

		vector<XformedIndexInfo*> *transformVector = new vector<XformedIndexInfo*>;
		transformVector->reserve(dataDimensions);
		for (int i = 0; i < dataDimensions; i++) {
			transformVector->push_back(new XformedIndexInfo());
		}

		int transferRequests = 0;
		int readCount = 0;
		int writeCount = 0;

		ExchangeIterator *iterator = getIterator();
		while (iterator->hasMoreElements()) {

			// get the next data item index from the iterator
			vector<int> *dataItemIndex = iterator->getNextElement();
			if (loggingEnabled) {
				logFile << "\t\tTransfer Request for: (";
				for (int i = 0; i < dataItemIndex->size(); i++) {
					if (i > 0) logFile << ',';
					logFile << dataItemIndex->at(i);
				}
				logFile << ")\n";
			}

			// traverse the sender tree and copy data from appropriate location in the the data-entry
			readTransferSpec->setBufferEntry(dataEntry, dataItemIndex);
			readPartSpec->initPartTraversalReference(dataItemIndex, transformVector);
			int elementsRead = senderTree->transferData(transformVector, 
					readTransferSpec, 
					readPartSpec, loggingEnabled, logFile, 3);
			if (loggingEnabled) {
				logFile << "\t\tElements Read: " << elementsRead << "\n";
			}
			readCount += elementsRead;

			// then write the data-entry in the appropriate location on the other side the same way
			writeTransferSpec->setBufferEntry(dataEntry, dataItemIndex);
			writePartSpec->initPartTraversalReference(dataItemIndex, transformVector);
			int elementsWritten = receiverTree->transferData(transformVector, 
					writeTransferSpec, 
					writePartSpec, loggingEnabled, logFile, 3);
			if (loggingEnabled) {
				logFile << "\t\tElements written: " << elementsWritten << "\n";
			}
			writeCount += elementsWritten;

			transferRequests++;
		}
		
		if (loggingEnabled) {
			logFile << "\tTotal data transfer requests: " << transferRequests << "\n";
			logFile << "\tData reads: " << readCount << "\n";
			logFile << "\tData writes: " << writeCount << "\n";
			logFile << "data transferring done for communication buffer " << bufferTag << "\n";
		}

		delete readPartSpec;
		delete readTransferSpec;
		delete writePartSpec;
		delete writeTransferSpec;
		delete transformVector;
		delete dataEntry;
	}
}

//------------------------------------------- Pre-processed Virtual Communication Buffer -----------------------------------------/

void PreprocessedVirtualCommBuffer::readData(bool loggingEnabled, std::ostream &logFile) {
	for (int i = 0; i < elementCount; i++) {
		char *readLocation = senderTransferMapping[i];
		char *writeLocation = receiverTransferMapping[i];
		memcpy(writeLocation, readLocation, elementSize);
	}
}

//-------------------------------------------- Index-mapped Virtual Communication Buffer -----------------------------------------/

void IndexMappedVirtualCommBuffer::readData(bool loggingEnabled, std::ostream &logFile) {
	for (int i = 0; i < elementCount; i++) {
                char *readLocation = senderTransferIndexMapping[i].getLocation();
                char *writeLocation = receiverTransferIndexMapping[i].getLocation();
                memcpy(writeLocation, readLocation, elementSize);
        }
}

//-------------------------------------------------- Communication Buffer Manager ------------------------------------------------/

CommBufferManager::CommBufferManager(const char *dependencyName) {
	this->dependencyName = dependencyName;
	commBufferList = new List<CommBuffer*>;
}

CommBufferManager::~CommBufferManager() {
	while (commBufferList->NumElements() > 0) {
		CommBuffer *buffer = commBufferList->Nth(0);
		commBufferList->RemoveAt(0);
		delete buffer;
	}
	delete commBufferList;
}

List<CommBuffer*> *CommBufferManager::getSortedList(bool sortForReceive, List<CommBuffer*> *bufferList) {
	
	List<CommBuffer*> *originalList = commBufferList;
	if (bufferList != NULL) {
		originalList = bufferList;
	}
	List<CommBuffer*> *sortedList = new List<CommBuffer*>;
	for (int i = 0; i < originalList->NumElements(); i++) {
		CommBuffer *buffer = originalList->Nth(i);
		if (sortForReceive && !buffer->isReceiveActivated()) continue;
		else if (!sortForReceive && !buffer->isSendActivated()) continue;
		int j = 0;
		for (; j < sortedList->NumElements(); j++) {
			if (sortedList->Nth(j)->compareTo(buffer, sortForReceive) > 0) break;
		}
		sortedList->InsertAt(buffer, j);
	}
	return sortedList;
}

void CommBufferManager::seperateLocalAndRemoteBuffers(int localSegmentTag, 
		List<CommBuffer*> *localBufferList, 
		List<CommBuffer*> *remoteBufferList) {
	
	for (int i = 0; i < commBufferList->NumElements(); i++) {
		CommBuffer *buffer = commBufferList->Nth(i);
		DataExchange *exchange = buffer->getExchange();
		if (exchange->isIntraSegmentExchange(localSegmentTag)) {
			localBufferList->Append(buffer);
		} else remoteBufferList->Append(buffer);
	}
}

std::vector<int> *CommBufferManager::getParticipantsTags() {
	std::vector<int> *participantTags = new std::vector<int>;
	for (int i = 0; i < commBufferList->NumElements(); i++) {
		CommBuffer *buffer = commBufferList->Nth(i);
		DataExchange *exchange = buffer->getExchange();
		std::vector<int> senderTags = exchange->getSender()->getSegmentTags();
		binsearch::addThoseNotExist(participantTags, &senderTags);
		std::vector<int> receiverTags = exchange->getReceiver()->getSegmentTags();
		binsearch::addThoseNotExist(participantTags, &receiverTags);
	}
	return participantTags;
}
