#include "comm_buffer.h"
#include "confinement_mgmt.h"
#include "data_transfer.h"
#include "part_config.h"
#include "../memory-management/allocation.h"
#include "../memory-management/part_tracking.h"
#include "../utils/list.h"

#include <iostream>
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
}

bool CommBuffer::isSendActivated() {
	return dataExchange->getSender()->hasSegmentTag(localSegmentTag);
}

bool CommBuffer::isReceiveActivated() {
	return dataExchange->getReceiver()->hasSegmentTag(localSegmentTag);
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
		List<DataPart*> *dataPartList,
		PartIdContainer *partContainerTree,
		DataItemConfig *dataConfig) {

	DataPartSpec *dataPartSpec = new DataPartSpec(dataPartList, dataConfig);
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
		partContainerTree->transferData(transformVector, transferSpec, dataPartSpec);
		elementIndex++;
	}

	delete dataPartSpec;
	delete transformVector;
	delete transferSpec;
}

//------------------------------------------------ Physical Communication Buffer -------------------------------------------------/

PhysicalCommBuffer::PhysicalCommBuffer(DataExchange *e, SyncConfig *s) : CommBuffer(e, s) {
	data = new char[elementCount * elementSize];
}

void PhysicalCommBuffer::readData() {
	if (isSendActivated()) {
		DataPartSpec *partSpec = new DataPartSpec(senderPartList, senderDataConfig);
		TransferSpec *readSpec = new TransferSpec(DATA_PART_TO_COMM_BUFFER, elementSize);
		transferData(readSpec, partSpec, senderTree);
		delete partSpec;
		delete readSpec;
	}
}

void PhysicalCommBuffer::writeData() {
	if (isReceiveActivated()) {
		DataPartSpec *partSpec = new DataPartSpec(receiverPartList, receiverDataConfig);
		TransferSpec *writeSpec = new TransferSpec(COMM_BUFFER_TO_DATA_PART, elementSize);
		transferData(writeSpec, partSpec, receiverTree);
		delete partSpec;
		delete writeSpec;
	}
}

void PhysicalCommBuffer::transferData(TransferSpec *transferSpec,
		DataPartSpec *dataPartSpec,
		PartIdContainer *partTree) {

	vector<XformedIndexInfo*> *transformVector = new vector<XformedIndexInfo*>;
	transformVector->reserve(dataDimensions);
	for (int i = 0; i < dataDimensions; i++) {
		transformVector->push_back(new XformedIndexInfo());
	}
	ExchangeIterator *iterator = getIterator();
	int elementIndex = 0;
	while (iterator->hasMoreElements()) {
		vector<int> *dataItemIndex = iterator->getNextElement();
		dataPartSpec->initPartTraversalReference(dataItemIndex, transformVector);
		char *dataLocation = data + elementIndex * elementSize;
		transferSpec->setBufferEntry(dataLocation, dataItemIndex);
		partTree->transferData(transformVector, transferSpec, dataPartSpec);
		elementIndex++;
	}
	delete transformVector;
}

//------------------------------------------ Pre-processed Physical Communication Buffer -----------------------------------------/

PreprocessedPhysicalCommBuffer::PreprocessedPhysicalCommBuffer(DataExchange *exchange,
		SyncConfig *syncConfig) : PreprocessedCommBuffer(exchange, syncConfig) {
	data = new char[elementCount * elementSize];
}

void PreprocessedPhysicalCommBuffer::readData() {
	for (int i = 0; i < elementCount; i++) {
		char *readLocation = senderTransferMapping[i];
		char *writeLocation = data + i * elementSize;
		memcpy(writeLocation, readLocation, elementSize);
	}
}

void PreprocessedPhysicalCommBuffer::writeData() {
	for (int i = 0; i < elementCount; i++) {
		char *readLocation = data + i * elementSize;
		char *writeLocation = receiverTransferMapping[i];
		memcpy(writeLocation, readLocation, elementSize);
	}
}

//------------------------------------------------- Virtual Communication Buffer -------------------------------------------------/

void VirtualCommBuffer::readData() {
	if (isReceiveActivated()) {

		DataPartSpec *readPartSpec = new DataPartSpec(senderPartList, senderDataConfig);
		TransferSpec *readTransferSpec = new TransferSpec(DATA_PART_TO_COMM_BUFFER, elementSize);

		DataPartSpec *writePartSpec = new DataPartSpec(receiverPartList, receiverDataConfig);
		TransferSpec *writeTransferSpec = new TransferSpec(COMM_BUFFER_TO_DATA_PART, elementSize);

		char *dataEntry = new char[elementSize];

		vector<XformedIndexInfo*> *transformVector = new vector<XformedIndexInfo*>;
		transformVector->reserve(dataDimensions);
		for (int i = 0; i < dataDimensions; i++) {
			transformVector->push_back(new XformedIndexInfo());
		}

		ExchangeIterator *iterator = getIterator();
		while (iterator->hasMoreElements()) {

			// get the next data item index from the iterator
			vector<int> *dataItemIndex = iterator->getNextElement();

			// traverse the sender tree and copy data from appropriate location in the the data-entry
			readTransferSpec->setBufferEntry(dataEntry, dataItemIndex);
			readPartSpec->initPartTraversalReference(dataItemIndex, transformVector);
			senderTree->transferData(transformVector, readTransferSpec, readPartSpec);

			// then write the data-entry in the appropriate location on the other side the same way
			writeTransferSpec->setBufferEntry(dataEntry, dataItemIndex);
			writePartSpec->initPartTraversalReference(dataItemIndex, transformVector);
			receiverTree->transferData(transformVector, writeTransferSpec, writePartSpec);
		}
	}
}

//------------------------------------------- Pre-processed Virtual Communication Buffer -----------------------------------------/

void PreprocessedVirtualCommBuffer::readData() {
	for (int i = 0; i < elementCount; i++) {
		char *readLocation = senderTransferMapping[i];
		char *writeLocation = receiverTransferMapping[i];
		memcpy(writeLocation, readLocation, elementSize);
	}
}
