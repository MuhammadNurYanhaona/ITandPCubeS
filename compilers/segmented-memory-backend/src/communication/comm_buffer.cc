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
	long int elementIndex = 0;
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
		senderTransferIndexMapping = new DataPartIndexList[elementCount];
		setupMappingBuffer(senderTransferIndexMapping, senderPartList, senderTree, senderDataConfig);
	}
	if (isReceiveActivated()) {
		receiverTransferIndexMapping = new DataPartIndexList[elementCount];
		setupMappingBuffer(receiverTransferIndexMapping, 
				receiverPartList, receiverTree, receiverDataConfig);
	}
}
        
IndexMappedCommBuffer::~IndexMappedCommBuffer() {
	if (senderTransferIndexMapping != NULL) {
		delete[] senderTransferIndexMapping;
	}
	if (receiverTransferIndexMapping != NULL) {
		delete[] receiverTransferIndexMapping;
	}
}

void IndexMappedCommBuffer::setupMappingBuffer(DataPartIndexList *indexMappingBuffer,
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
	long int elementIndex = 0;
	TransferIndexSpec *transferSpec = new TransferIndexSpec(elementSize);
	while (iterator->hasMoreElements()) {
		vector<int> *dataItemIndex = iterator->getNextElement();
		dataPartSpec->initPartTraversalReference(dataItemIndex, transformVector);
		transferSpec->setPartIndexListReference(&indexMappingBuffer[elementIndex]);
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
	long int bufferSize = elementCount * elementSize;
	data = new char[bufferSize];
	for (long int i = 0; i < bufferSize; i++) {
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
	long int elementIndex = 0;
	long int transferRequests = 0;
	long int transferCount = 0;
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
	
	long int bufferSize = elementCount * elementSize;
	data = new char[bufferSize];
	for (long int i = 0; i < bufferSize; i++) {
		data[i] = 0;
	}
}

void PreprocessedPhysicalCommBuffer::readData(bool loggingEnabled, std::ostream &logFile) {
	for (long int i = 0; i < elementCount; i++) {
		char *readLocation = senderTransferMapping[i];
		char *writeLocation = data + i * elementSize;
		memcpy(writeLocation, readLocation, elementSize);
	}
}

void PreprocessedPhysicalCommBuffer::writeData(bool loggingEnabled, std::ostream &logFile) {
	for (long int i = 0; i < elementCount; i++) {
		char *readLocation = data + i * elementSize;
		char *writeLocation = receiverTransferMapping[i];
		memcpy(writeLocation, readLocation, elementSize);
	}
}

//------------------------------------------- Index-mapped Physical Communication Buffer -----------------------------------------/

IndexMappedPhysicalCommBuffer::IndexMappedPhysicalCommBuffer(DataExchange *exchange, 
		SyncConfig *syncConfig) : IndexMappedCommBuffer(exchange, syncConfig) {
	
	long int bufferSize = elementCount * elementSize;
        data = new char[bufferSize];
        for (long int i = 0; i < bufferSize; i++) {
                data[i] = 0;
        }
}

void IndexMappedPhysicalCommBuffer::readData(bool loggingEnabled, std::ostream &logFile) {

	for (long int i = 0; i < elementCount; i++) {

		
		// Notice that data is read from only the first matched location in the operating memory data part
		// storage. This is because, all locations matching a single entry in the communication buffer 
		// should have identical content.
		List<DataPartIndex> *partIndexList = senderTransferIndexMapping[i].getPartIndexList();
                char *readLocation = partIndexList->Nth(0).getLocation();
                char *writeLocation = data + i * elementSize;
                memcpy(writeLocation, readLocation, elementSize);
        }
}
        
void IndexMappedPhysicalCommBuffer::writeData(bool loggingEnabled, std::ostream &logFile) {
	
	for (long int i = 0; i < elementCount; i++) {
                char *readLocation = data + i * elementSize;
		List<DataPartIndex> *partIndexList = receiverTransferIndexMapping[i].getPartIndexList();

		// Unlike in the case for readData, writing should access every single matched location in the data
		// parts as we need to ensure that all locations matching a single entry in the communication buffer
		// are synchronized with the same update.	
		for (int j = 0; j < partIndexList->NumElements(); j++) {
                	char *writeLocation = partIndexList->Nth(j).getLocation();
                	memcpy(writeLocation, readLocation, elementSize);
		}
        }
}

//---------------------------------------- Swift-Index-mapped Physical Communication Buffer --------------------------------------/

SwiftIndexMappedPhysicalCommBuffer::SwiftIndexMappedPhysicalCommBuffer(
		DataExchange *exchange, SyncConfig *syncConfig) 
		: IndexMappedPhysicalCommBuffer(exchange, syncConfig) {
	
	senderSwiftIndexMapping = new List<DataPartIndexList*>;
	receiverSwiftIndexMapping = new List<DataPartIndexList*>;
	if (isSendActivated()) {
		setupSwiftIndexMapping(senderTransferIndexMapping, senderSwiftIndexMapping, false);
	}
	if (isReceiveActivated()) {
		setupSwiftIndexMapping(receiverTransferIndexMapping, receiverSwiftIndexMapping, true);
	}
}

SwiftIndexMappedPhysicalCommBuffer::~SwiftIndexMappedPhysicalCommBuffer() {

	while (senderSwiftIndexMapping->NumElements() > 0) {
		DataPartIndexList *entry = senderSwiftIndexMapping->Nth(0);
		senderSwiftIndexMapping->RemoveAt(0);
		delete entry;
	}
	delete senderSwiftIndexMapping;

	while (receiverSwiftIndexMapping->NumElements() > 0) {
		DataPartIndexList *entry = receiverSwiftIndexMapping->Nth(0);
		receiverSwiftIndexMapping->RemoveAt(0);
		delete entry;
	}
	delete receiverSwiftIndexMapping;
}

void SwiftIndexMappedPhysicalCommBuffer::readData(bool loggingEnabled, std::ostream &logFile) {
	int index = 0;
	for (int i = 0; i < senderSwiftIndexMapping->NumElements(); i++) {
		DataPartIndexList *partIndexList = senderSwiftIndexMapping->Nth(i);
		char *bufferIndex = data + index * elementSize;
		index += partIndexList->read(bufferIndex, elementSize);
	}
}

void SwiftIndexMappedPhysicalCommBuffer::writeData(bool loggingEnabled, std::ostream &logFile) {
	int index = 0;
	for (int i = 0; i < receiverSwiftIndexMapping->NumElements(); i++) {
		DataPartIndexList *partIndexList = receiverSwiftIndexMapping->Nth(i);
		char *bufferIndex = data + index * elementSize;
		index += partIndexList->write(bufferIndex, elementSize);
	}
}

void SwiftIndexMappedPhysicalCommBuffer::setupSwiftIndexMapping(DataPartIndexList *transferIndexMapping,
			List<DataPartIndexList*> *swiftIndexMapping,
                        bool allowMultPartIndexesForSameBufferIndex) {

	DataPartSwiftIndexList *currSwiftIndex = NULL;
	DataPartIndexList *currEntry = new DataPartIndexList();	
	for (long int i = 0; i < elementCount; i++) {
		currEntry->clone(&transferIndexMapping[i]);
		List<DataPartIndex> *indexList = currEntry->getPartIndexList();
		if (!allowMultPartIndexesForSameBufferIndex || indexList->NumElements() == 1) {
			DataPartIndex partIndex = indexList->Nth(0);
			if (currSwiftIndex == NULL) {
				currSwiftIndex = new DataPartSwiftIndexList(partIndex.getDataPart());
				currSwiftIndex->addIndex(partIndex.getIndex());
			} else {
				if (currSwiftIndex->getDataPart() == partIndex.getDataPart()) {
					currSwiftIndex->addIndex(partIndex.getIndex());
				} else {
					currSwiftIndex->setupIndexArray();
					swiftIndexMapping->Append(currSwiftIndex);
					currSwiftIndex = new DataPartSwiftIndexList(partIndex.getDataPart());
					currSwiftIndex->addIndex(partIndex.getIndex());
				}
			}
		} else {
			if (currSwiftIndex != NULL) {
				currSwiftIndex->setupIndexArray();
				swiftIndexMapping->Append(currSwiftIndex);
				DataPartIndexList *cloneEntry = new DataPartIndexList();
				cloneEntry->clone(currEntry);
				swiftIndexMapping->Append(cloneEntry);
				currSwiftIndex = NULL;	
			} else {
				DataPartIndexList *cloneEntry = new DataPartIndexList();
				cloneEntry->clone(currEntry);
				swiftIndexMapping->Append(cloneEntry);
			}
		} 
	}
	if (currSwiftIndex != NULL) {
		currSwiftIndex->setupIndexArray();
		swiftIndexMapping->Append(currSwiftIndex);
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

		long int transferRequests = 0;
		long int readCount = 0;
		long int writeCount = 0;

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
	for (long int i = 0; i < elementCount; i++) {
		char *readLocation = senderTransferMapping[i];
		char *writeLocation = receiverTransferMapping[i];
		memcpy(writeLocation, readLocation, elementSize);
	}
}

//-------------------------------------------- Index-mapped Virtual Communication Buffer -----------------------------------------/

void IndexMappedVirtualCommBuffer::readData(bool loggingEnabled, std::ostream &logFile) {

	for (long int i = 0; i < elementCount; i++) {

		// data is read from only one location as all matching locations should have the same content
		List<DataPartIndex> *sourcePartIndexList = senderTransferIndexMapping[i].getPartIndexList();
                char *readLocation = sourcePartIndexList->Nth(0).getLocation();
			
		// update is propagated to all matching locations as they need have equivalent in data content
		List<DataPartIndex> *destPartIndexList = receiverTransferIndexMapping[i].getPartIndexList();
		for (int j = 0; j < destPartIndexList->NumElements(); j++) {
                	char *writeLocation = destPartIndexList->Nth(j).getLocation();
                	memcpy(writeLocation, readLocation, elementSize);
		}
        }
}

//----------------------------------------- Swift-Index-mapped Virtual Communication Buffer --------------------------------------/

SwiftIndexMappedVirtualCommBuffer::SwiftIndexMappedVirtualCommBuffer(
		DataExchange *exchange, SyncConfig *syncConfig) 
		: IndexMappedVirtualCommBuffer(exchange, syncConfig) {

	long int bufferSize = elementCount * elementSize;
	data = new char[bufferSize];
	for (long int i = 0; i < bufferSize; i++) {
		data[i] = 0;
	}

	generateSwiftIndexMappings();
}

SwiftIndexMappedVirtualCommBuffer::~SwiftIndexMappedVirtualCommBuffer() {
	
	delete[] data;

	while (senderSwiftIndexMapping->NumElements() > 0) {
                DataPartIndexList *entry = senderSwiftIndexMapping->Nth(0);
                senderSwiftIndexMapping->RemoveAt(0);
                delete entry;
        }
        delete senderSwiftIndexMapping;

        while (receiverSwiftIndexMapping->NumElements() > 0) {
                DataPartIndexList *entry = receiverSwiftIndexMapping->Nth(0);
                receiverSwiftIndexMapping->RemoveAt(0);
                delete entry;
        }
        delete receiverSwiftIndexMapping;
}

void SwiftIndexMappedVirtualCommBuffer::readData(bool loggingEnabled, std::ostream &logFile) {

	long int index = 0;
        for (int i = 0; i < senderSwiftIndexMapping->NumElements(); i++) {
                DataPartIndexList *partIndexList = senderSwiftIndexMapping->Nth(i);
                char *bufferIndex = data + index * elementSize;
                index += partIndexList->read(bufferIndex, elementSize);
        }

	index = 0;
        for (int i = 0; i < receiverSwiftIndexMapping->NumElements(); i++) {
                DataPartIndexList *partIndexList = receiverSwiftIndexMapping->Nth(i);
                char *bufferIndex = data + index * elementSize;
                index += partIndexList->write(bufferIndex, elementSize);
        }
}

void SwiftIndexMappedVirtualCommBuffer::generateSwiftIndexMappings() {
	
	// prepare the sender-side swift index mapping
	senderSwiftIndexMapping = new List<DataPartIndexList*>;
	DataPartSwiftIndexList *currSwiftIndex = NULL;
	DataPartIndexList *currSendEntry = new DataPartIndexList();	
	for (long int i = 0; i < elementCount; i++) {
		currSendEntry->clone(&senderTransferIndexMapping[i]);
		List<DataPartIndex> *indexList = currSendEntry->getPartIndexList();
		DataPartIndex partIndex = indexList->Nth(0);
		if (currSwiftIndex == NULL) {
			currSwiftIndex = new DataPartSwiftIndexList(partIndex.getDataPart());
			currSwiftIndex->addIndex(partIndex.getIndex());
		} else {
			if (currSwiftIndex->getDataPart() == partIndex.getDataPart()) {
				currSwiftIndex->addIndex(partIndex.getIndex());
			} else {
				currSwiftIndex->setupIndexArray();
				senderSwiftIndexMapping->Append(currSwiftIndex);
				currSwiftIndex = new DataPartSwiftIndexList(partIndex.getDataPart());
				currSwiftIndex->addIndex(partIndex.getIndex());
			}
		}
	}
	if (currSwiftIndex != NULL) {
		currSwiftIndex->setupIndexArray();
		senderSwiftIndexMapping->Append(currSwiftIndex);
	}

	// prepare the receiver-side swift index mapping
	receiverSwiftIndexMapping = new List<DataPartIndexList*>;
	currSwiftIndex = NULL;
	DataPartIndexList *currRecvEntry = new DataPartIndexList();
	List<DataPartIndex> *refIndexList = new List<DataPartIndex>;	
	for (long int i = 0; i < elementCount; i++) {

		currRecvEntry->clone(&receiverTransferIndexMapping[i]);
		List<DataPartIndex> *recvIndexList = currRecvEntry->getPartIndexList();

		// If there is only one receive index for this communication buffer entry then augment existing or start 
		// a new swift index mapping.
		if (recvIndexList->NumElements() == 1) {
			DataPartIndex partIndex = recvIndexList->Nth(0);
			DataPart *dataPart = partIndex.getDataPart();
			long int index = partIndex.getIndex();
			if (currSwiftIndex == NULL) { 
				currSwiftIndex = new DataPartSwiftIndexList(dataPart);
				currSwiftIndex->addIndex(index);
			} else {
				if (currSwiftIndex->getDataPart() == dataPart) {
					currSwiftIndex->addIndex(index);
				} else {
					currSwiftIndex->setupIndexArray();
					receiverSwiftIndexMapping->Append(currSwiftIndex);
					currSwiftIndex = new DataPartSwiftIndexList(dataPart);
					currSwiftIndex->addIndex(index);
				}
			}
		// If there are more than one receive indices then first we try to avoid sender index being unwittingly
		// repeated in the receiver side. After eliminating such redundancy, we may be able to create a new swift 
		// mapping. Otherwise, just store the original transfer mapping 
		} else {
			
			// filering receive part indexes
			currSendEntry->clone(&senderTransferIndexMapping[i]);
			DataPartIndex sendPartIndex = currSendEntry->getPartIndexList()->Nth(0);
			refIndexList->clear();
			for (int j = 0; j < recvIndexList->NumElements(); j++) {
				DataPartIndex recvPartIndex = recvIndexList->Nth(j);
				if (sendPartIndex.getDataPart() != recvPartIndex.getDataPart()) {
					refIndexList->Append(recvPartIndex);
				}	
			}

			// attempt to augment existing or start a new swift index mapping
			if (refIndexList->NumElements() == 1) {
				DataPartIndex partIndex = refIndexList->Nth(0);
				DataPart *dataPart = partIndex.getDataPart();
				long int index = partIndex.getIndex();	
				if (currSwiftIndex == NULL) { 
					currSwiftIndex = new DataPartSwiftIndexList(dataPart);
					currSwiftIndex->addIndex(index);
				} else {
					if (currSwiftIndex->getDataPart() == dataPart) {
						currSwiftIndex->addIndex(index);
					} else {
						currSwiftIndex->setupIndexArray();
						receiverSwiftIndexMapping->Append(currSwiftIndex);
						currSwiftIndex = new DataPartSwiftIndexList(dataPart);
						currSwiftIndex->addIndex(index);
					}
				}
			// if failed then store the normal index mapping
			} else {
				if (currSwiftIndex != NULL) {
					currSwiftIndex->setupIndexArray();
					receiverSwiftIndexMapping->Append(currSwiftIndex);
					DataPartIndexList *cloneEntry = new DataPartIndexList();
					cloneEntry->clonePartIndexList(refIndexList);
					receiverSwiftIndexMapping->Append(cloneEntry);
					currSwiftIndex = NULL;	
				} else {
					DataPartIndexList *cloneEntry = new DataPartIndexList();
					cloneEntry->clonePartIndexList(refIndexList);
					receiverSwiftIndexMapping->Append(cloneEntry);
				}
			}
		}
	}
	if (currSwiftIndex != NULL) {
		currSwiftIndex->setupIndexArray();
		receiverSwiftIndexMapping->Append(currSwiftIndex);
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
