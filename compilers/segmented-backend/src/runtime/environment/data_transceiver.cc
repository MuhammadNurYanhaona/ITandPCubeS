#include "data_transceiver.h"
#include "environment.h"

#include "../memory-management/allocation.h"
#include "../memory-management/part_generation.h"
#include "../memory-management/part_tracking.h"
#include "../communication/data_transfer.h"
#include "../communication/confinement_mgmt.h"
#include "../communication/part_config.h"

#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/interval.h"
#include "../../../../common-libs/utils/common_utils.h"
#include "../../../../common-libs/utils/id_generation.h"

#include <mpi.h>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string.h>

using namespace std;

//-------------------------------------------------------- Transfer Config ------------------------------------------------------------

TransferConfig::TransferConfig(ProgramEnvironment *progEnv, 
		char *dataItemId, 
		ListReferenceKey *svk, 
		LpsAllocation *recv, TaskItem *rtk) {
	this->progEnv = progEnv;
	this->dataItemId = dataItemId;
	this->sourceVersionKey = svk;
	this->receiver = recv;
	this->receiverTaskItem = rtk;
	this->elementSize = rtk->getElementSize();
	this->logFile = NULL;
}

PartsListReference *TransferConfig::getSourceReference() {
	char *stringKey = sourceVersionKey->generateKey();
	PartsListReference *reference = progEnv->getVersionManager(dataItemId)->getVersion(stringKey);
	free(stringKey);
	return reference;
}

List<MultidimensionalIntervalSeq*> *TransferConfig::getLocalDestinationFold() {
	DataPartitionConfig *partConfig = receiver->getPartitionConfig();
	PartIdContainer *containerTree = receiver->getContainerTree();
	return ListReferenceAttributes::computeSegmentFold(partConfig, containerTree, logFile);	
}

char *TransferConfig::generateTargetVersionkey() {
	int envId = receiverTaskItem->getEnvironment()->getEnvId();
	const char *itemName = receiverTaskItem->getEnvLinkKey()->getVarName();
	ListReferenceKey *receiverKey = receiver->generatePartsListReferenceKey(envId, itemName);
	return receiverKey->generateKey();
}

ListReferenceAttributes *TransferConfig::prepareTargetVersionAttributes() {
	
	int dimensionality = receiverTaskItem->getDimensionality();
        List<Dimension> *rootDimensions = new List<Dimension>;
        for (int i = 0; i < dimensionality; i++) {
                rootDimensions->Append(receiverTaskItem->getDimension(i));
        }

	DataPartitionConfig *partConfig = receiver->getPartitionConfig();
	PartIdContainer *containerTree = receiver->getContainerTree();
	ListReferenceAttributes *attr = new ListReferenceAttributes(partConfig, rootDimensions);
	attr->setPartContainerTree(containerTree);
	attr->computeSegmentFold();
	return attr;
}

//------------------------------------------------- Communication Requirement Finder --------------------------------------------------

CommunicationReqFinder::CommunicationReqFinder(List<MultidimensionalIntervalSeq*> *sourceFold,
		List<MultidimensionalIntervalSeq*> *targetFold) {
	this->localSourceFold = sourceFold;
	this->localTargetFold = targetFold;
}

bool CommunicationReqFinder::isCrossSegmentCommRequired(std::ofstream &logFile) {

	bool localDataSufficient = ListReferenceAttributes::isSuperFold(localSourceFold, localTargetFold);
	int transferReqValue = localDataSufficient ? 0 : 1;
	int sum = -1;

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        int status = MPI_Allreduce(&transferReqValue, &sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if (status != MPI_SUCCESS) {
                cout << rank << ": could not participate in the all to all reduction to determine communication need\n";
                exit(EXIT_FAILURE);
        }

	return (sum > 0);
}

//---------------------------------------------------- Segment Mapping Preparer -------------------------------------------------------

SegmentMappingPreparer::SegmentMappingPreparer(List<MultidimensionalIntervalSeq*> *localSegmentContent) {
	this->localSegmentContent = localSegmentContent;
}

List<SegmentDataContent*> *SegmentMappingPreparer::shareSegmentsContents(std::ofstream &logFile) {
	
	int foldSize = 0;
	char *foldString = NULL;
	if (localSegmentContent != NULL) {
		foldString = MultidimensionalIntervalSeq::convertSetToString(localSegmentContent);
		foldSize = strlen(foldString);
	}
	
	int rank, segmentCount;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &segmentCount);

	int *foldSizes = new int[segmentCount];
	int status = MPI_Allgather(&foldSize, 1, MPI_INT, foldSizes, 1, MPI_INT, MPI_COMM_WORLD);
        if (status != MPI_SUCCESS) {
                cout << rank << ": could not determine the data content sizes of different segments\n";
                exit(EXIT_FAILURE);
        }
	
	int *displacements = new int[segmentCount];
	int currentIndex = 0;
	for (int i = 0; i < segmentCount; i++) {
		displacements[i] = currentIndex;
		currentIndex += foldSizes[i];
	}
	char *foldDescBuffer = new char[currentIndex];

	status = MPI_Allgatherv(foldString, foldSize, MPI_CHAR, 
			foldDescBuffer, foldSizes, displacements, MPI_CHAR, MPI_COMM_WORLD);
        if (status != MPI_SUCCESS) {
                cout << rank << ": could not gather fold descriptions from all segments\n";
                exit(EXIT_FAILURE);
        }
	
	List<SegmentDataContent*> *segmentContentMap = new List<SegmentDataContent*>;
	currentIndex = 0;
	for (int i = 0; i < segmentCount; i++) {
		int length = foldSizes[i];
		if (length == 0) continue;
		int beginIndex = currentIndex;
		int endIndex = currentIndex + length - 1;
		if (i != rank) {
			char *contentStr = new char[length + 1];
			char *indexStart = foldDescBuffer + beginIndex;
			strncpy(contentStr, indexStart, length);
			contentStr[length] = '\0';
			segmentContentMap->Append(new SegmentDataContent(i, contentStr));
		}
		currentIndex += length;
	}

	delete[] foldSizes;
	delete[] displacements;
	delete[] foldDescBuffer;
	free(foldString);
	
	return segmentContentMap;
}

//------------------------------------------------------- Local Transferrer -----------------------------------------------------------

LocalTransferrer::LocalTransferrer(TransferConfig *transferConfig) {
	
	PartsListReference *srcReference = transferConfig->getSourceReference();
	ListReferenceAttributes *srcAttrs = srcReference->getAttributes();
	this->sourceConfig = srcAttrs->getPartitionConfig();
	this->sourcePartsTree = srcAttrs->getPartContainerTree();
	this->sourceFold = srcAttrs->getSegmentFold();
	this->sourcePartList = srcReference->getPartsList()->getDataParts();

	LpsAllocation *receiverAlloc = transferConfig->getReceiver();
	this->targetConfig = receiverAlloc->getPartitionConfig();
	this->targetPartsTree = receiverAlloc->getContainerTree();
	this->targetFold = transferConfig->getLocalDestinationFold();
	this->targetPartList = receiverAlloc->getPartsList()->getDataParts();

	TaskItem *receiverItem = transferConfig->getReceiverTaskItem();
	this->elementSize = receiverItem->getElementSize();
}

void LocalTransferrer::transferData(std::ofstream &logFile) {
	
	if (sourceFold == NULL || targetFold == NULL) return;

	Participant *sender = new Participant(SEND, NULL, sourceFold);
	Participant *receiver = new Participant(RECEIVE, NULL, targetFold);
	List<MultidimensionalIntervalSeq*> *intersect = DataExchange::getCommonRegion(sender, receiver);
	if (intersect == NULL) return;
	
	DataExchange *exchange = new DataExchange(sender, receiver, intersect);
	ExchangeIterator *iterator = new ExchangeIterator(exchange);

	DataItemConfig *sendConfig = sourceConfig->generateStateFulVersion();	
	DataPartSpec *readPartSpec = new DataPartSpec(sourcePartList, sendConfig);
	TransferSpec *readTransferSpec = new TransferSpec(DATA_PART_TO_COMM_BUFFER, elementSize);
	DataItemConfig *recvConfig = targetConfig->generateStateFulVersion();
	DataPartSpec *writePartSpec = new DataPartSpec(targetPartList, recvConfig);
	TransferSpec *writeTransferSpec = new TransferSpec(COMM_BUFFER_TO_DATA_PART, elementSize);		

	int dataDimensions = readPartSpec->getDimensionality();	
	vector<XformedIndexInfo*> *transformVector = new vector<XformedIndexInfo*>;
	transformVector->reserve(dataDimensions);
	for (int i = 0; i < dataDimensions; i++) {
		transformVector->push_back(new XformedIndexInfo());
	}

	char *dataEntry = new char[elementSize];
	while (iterator->hasMoreElements()) {

		vector<int> *dataItemIndex = iterator->getNextElement();

		readTransferSpec->setBufferEntry(dataEntry, dataItemIndex);
		readPartSpec->initPartTraversalReference(dataItemIndex, transformVector);
		sourcePartsTree->transferData(transformVector, readTransferSpec, readPartSpec, false, cout);

		writeTransferSpec->setBufferEntry(dataEntry, dataItemIndex);
                writePartSpec->initPartTraversalReference(dataItemIndex, transformVector);
                targetPartsTree->transferData(transformVector, writeTransferSpec, writePartSpec, false, cout);
	}

	delete exchange;
	delete iterator;
	delete sendConfig;
	delete readPartSpec;
	delete readTransferSpec;
	delete recvConfig;
	delete writePartSpec;
	delete writeTransferSpec;
	delete transformVector;
	delete[] dataEntry;
}

//-------------------------------------------------------- Transfer Buffer ------------------------------------------------------------

TransferBuffer::TransferBuffer(int sender, int receiver, 
		DataExchange *exchange,
		DataPartSpec *partListSpec, TransferSpec *transferSpec,
		PartIdContainer *partContainerTree) {
	this->sender = sender;
	this->receiver = receiver;
	this->exchange = exchange;
	this->partListSpec = partListSpec;
	this->transferSpec = transferSpec;
	this->partContainerTree = partContainerTree;
	int elementSize = transferSpec->getStepSize();
	this->data = new char[exchange->getTotalElementsCount() * elementSize];
}

TransferBuffer::~TransferBuffer() {
	delete[] data;
	delete exchange;
	delete transferSpec;
}

void TransferBuffer::preprocessBuffer(std::ofstream &logFile) {
	TransferDirection direction = transferSpec->getDirection();
	if (direction == DATA_PART_TO_COMM_BUFFER) {
		processBuffer(logFile);
	}
}

void TransferBuffer::postProcessBuffer(std::ofstream &logFile) {
	TransferDirection direction = transferSpec->getDirection();
	if (direction == COMM_BUFFER_TO_DATA_PART) {
		processBuffer(logFile);
	}
}

int TransferBuffer::compareTo(TransferBuffer *other) {
	if (this->bufferTag > other->bufferTag) return 1;
	if (this->bufferTag < other->bufferTag) return -1;
	return 0;
}

void TransferBuffer::processBuffer(std::ofstream &logFile) {
	
	int dataDimensions = partListSpec->getDimensionality();
	int elementSize = transferSpec->getStepSize();

	vector<XformedIndexInfo*> *transformVector = new vector<XformedIndexInfo*>;
        transformVector->reserve(dataDimensions);
        for (int i = 0; i < dataDimensions; i++) {
                transformVector->push_back(new XformedIndexInfo());
        }

	ExchangeIterator iterator = ExchangeIterator(exchange);
        long int elementIndex = 0;
        long int transferRequests = 0;
        long int transferCount = 0;
        while (iterator.hasMoreElements()) {
                vector<int> *dataItemIndex = iterator.getNextElement();
               	partListSpec->initPartTraversalReference(dataItemIndex, transformVector);
                char *dataLocation = data + elementIndex * elementSize;
                transferSpec->setBufferEntry(dataLocation, dataItemIndex);
                partContainerTree->transferData(transformVector, transferSpec, partListSpec, false, cout);
                elementIndex++;
        }
        delete transformVector;
}

long int TransferBuffer::getSize() {
	int elementSize = transferSpec->getStepSize();
        return exchange->getTotalElementsCount() * elementSize;
}	

//--------------------------------------------------- Transfer Buffers Preparer -------------------------------------------------------

TransferBuffersPreparer::TransferBuffersPreparer(int elementSize,
		List<MultidimensionalIntervalSeq*> *localSourceContent,
		DataPartSpec *sourcePartsListSpec,
		List<MultidimensionalIntervalSeq*> *localTargetContent,
		DataPartSpec *targetPartsListSpec) {

	this->elementSize = elementSize;
	this->localSourceContent = localSourceContent;
	this->sourcePartsListSpec = sourcePartsListSpec;
	this->localTargetContent = localTargetContent;
	this->targetPartsListSpec = targetPartsListSpec;
}

List<TransferBuffer*> *TransferBuffersPreparer::createBuffersForOutgoingTransfers(PartIdContainer *sourceContainer,
		List<SegmentDataContent*> *targetContentMap, std::ofstream &logFile) {
	
	int senderId, segmentCount;
	MPI_Comm_rank(MPI_COMM_WORLD, &senderId);
	MPI_Comm_size(MPI_COMM_WORLD, &segmentCount);
	int digits = countDigits(segmentCount); 
	
	if (localSourceContent == NULL) return NULL;
	if (targetContentMap == NULL || targetContentMap->NumElements() == 0) return NULL;

	Participant *sender = new Participant(SEND, NULL, localSourceContent);
	List<TransferBuffer*> *transferBufferList = new List<TransferBuffer*>;
	for (int i = 0; i < targetContentMap->NumElements(); i++) {
		SegmentDataContent *segmentContent = targetContentMap->Nth(i);
		List<MultidimensionalIntervalSeq*> *segmentFold = segmentContent->generateFold();
		
		Participant *receiver = new Participant(RECEIVE, NULL, segmentFold);
		List<MultidimensionalIntervalSeq*> *intersect = DataExchange::getCommonRegion(sender, receiver);
		if (intersect == NULL) {
			// deletion of the receiver will also delete the segment fold being generated for it
			delete receiver;
			continue;
		}
		int receiverId = segmentContent->getSegmentId();
		DataExchange *exchange = new DataExchange(sender, receiver, intersect);
		DataPartSpec *partListSpec = sourcePartsListSpec;
		PartIdContainer *containerTree = sourceContainer;
		TransferSpec *transferSpec = new TransferSpec(DATA_PART_TO_COMM_BUFFER, elementSize);
		TransferBuffer *buffer = new TransferBuffer(senderId, receiverId, 
				exchange, partListSpec, transferSpec, containerTree);
		int bufferTag = idutils::concateIds(senderId, receiverId, digits);
		buffer->setBufferTag(bufferTag);
		transferBufferList->Append(buffer);
	}
	
	return transferBufferList;
}

List<TransferBuffer*> *TransferBuffersPreparer::createBuffersForIncomingTransfers(PartIdContainer *targetContainer,
                        List<SegmentDataContent*> *sourceContentMap, std::ofstream &logFile) {

	if (localTargetContent == NULL) return NULL;
	if (sourceContentMap == NULL || sourceContentMap->NumElements() == 0) return NULL;
	
	int receiverId, segmentCount;
	MPI_Comm_rank(MPI_COMM_WORLD, &receiverId);
	MPI_Comm_size(MPI_COMM_WORLD, &segmentCount);
	int digits = countDigits(segmentCount); 

	Participant *receiver = new Participant(RECEIVE, NULL, localTargetContent);
	List<TransferBuffer*> *transferBufferList = new List<TransferBuffer*>;
	for (int i = 0; i < sourceContentMap->NumElements(); i++) {
		SegmentDataContent *segmentContent = sourceContentMap->Nth(i);
		List<MultidimensionalIntervalSeq*> *segmentFold = segmentContent->generateFold();
		
		Participant *sender = new Participant(SEND, NULL, segmentFold);
		List<MultidimensionalIntervalSeq*> *intersect = DataExchange::getCommonRegion(sender, receiver);
		if (intersect == NULL) {
			// deletion of the sender will also delete the segment fold being generated for it
			delete sender;
			continue;
		}
		int senderId = segmentContent->getSegmentId();
		DataExchange *exchange = new DataExchange(sender, receiver, intersect);
		DataPartSpec *partListSpec = targetPartsListSpec;
		PartIdContainer *containerTree = targetContainer;
		TransferSpec *transferSpec = new TransferSpec(COMM_BUFFER_TO_DATA_PART, elementSize);
		TransferBuffer *buffer = new TransferBuffer(senderId, receiverId, 
				exchange, partListSpec, transferSpec, containerTree);
		int bufferTag = idutils::concateIds(senderId, receiverId, digits);
		buffer->setBufferTag(bufferTag);
		transferBufferList->Append(buffer);
	}

	return transferBufferList;
}

//------------------------------------------------------ Buffer Transferrer -----------------------------------------------------------

BufferTransferrer::BufferTransferrer(bool mode, List<TransferBuffer*> *bufferList) {
	this->sendMode = mode;
	this->bufferList = bufferList;
	this->transferRequests = NULL;
}

BufferTransferrer::~BufferTransferrer() {
	if (transferRequests != NULL) {
		delete[] transferRequests;
	}
}

void BufferTransferrer::sendDataAsync(std::ofstream &logFile) {
	
	if (!sendMode) {
		cout << "cannot use a buffer transferrer that is configured for receive to send data\n";
		exit(EXIT_FAILURE);
	}
	if (bufferList == NULL || bufferList->NumElements() == 0) return;
	
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	transferRequests = new MPI_Request[bufferList->NumElements()];

	for (int i = 0; i < bufferList->NumElements(); i++) {
		TransferBuffer *buffer = bufferList->Nth(i);
		buffer->preprocessBuffer(logFile);
		long int size = buffer->getSize();
		char *data = buffer->getData();
		int receiver = buffer->getReceiver();
		int tag = buffer->getBufferTag();
		int status = MPI_Isend(data, size, MPI_CHAR, receiver, tag, MPI_COMM_WORLD, &transferRequests[i]);
		if (status != MPI_SUCCESS) {
			cout << "Segment " << rank << ": could not issue asynchronous send\n";
			exit(EXIT_FAILURE);
		}
	}
}
        
void BufferTransferrer::receiveDataAsync(std::ofstream &logFile) {
	
	if (sendMode) {
		cout << "cannot use a buffer transferrer that is configured for send to receive data\n";
		exit(EXIT_FAILURE);
	}
	if (bufferList == NULL || bufferList->NumElements() == 0) return;
	
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	transferRequests = new MPI_Request[bufferList->NumElements()];

	for (int i = 0; i < bufferList->NumElements(); i++) {
		TransferBuffer *buffer = bufferList->Nth(i);
		long int size = buffer->getSize();
		char *data = buffer->getData();
		int sender = buffer->getSender();
		int tag = buffer->getBufferTag();
		int status = MPI_Irecv(data, size, MPI_CHAR, sender, tag, MPI_COMM_WORLD, &transferRequests[i]);
		if (status != MPI_SUCCESS) {
			cout << "Segment " << rank << ": could not issue asynchronous receive\n";
			exit(EXIT_FAILURE);
		}
	}
}

void BufferTransferrer::waitForTransferComplete(std::ofstream &logFile) {

	if (bufferList != NULL && bufferList->NumElements() > 0) {
		
		int transferCount = bufferList->NumElements();
		int status = MPI_Waitall(transferCount, transferRequests, MPI_STATUSES_IGNORE);
        	if (status != MPI_SUCCESS) {
                	cout << "some of the transfer requests failed to finish\n";
                	exit(EXIT_FAILURE);
        	}
		if (!sendMode) {
			for (int i = 0; i < bufferList->NumElements(); i++) {
				TransferBuffer *buffer = bufferList->Nth(i);
				buffer->postProcessBuffer(logFile);
			}
		}
	}
}	

//----------------------------------------------------- Data Transfer Manager ---------------------------------------------------------

DataTransferManager::DataTransferManager(TransferConfig *transferConfig, bool computeTargetFoldAfresh) {
	this->transferConfig = transferConfig;
	this->sourceContentMap = NULL;
	this->targetContentMap = NULL;
	this->computeTargetFoldAfresh = computeTargetFoldAfresh;
}

void DataTransferManager::handleTransfer() {

	// First determine if there is a need for locally moving data from the source parts list version in the program
	// environment to the target LPS allocation. If there is a need then do the local transfer first
	transferConfig->setLogFile(logFile);
	LocalTransferrer localTransferrer = LocalTransferrer(transferConfig);
	localTransferrer.transferData(*logFile);

	// Then determine if there is a need for cross-segment communications for the transfer configuration. If the local
	// transfer suffices then exit
	List<MultidimensionalIntervalSeq*> *localSourceFold = localTransferrer.getSourceFold();
	List<MultidimensionalIntervalSeq*> *localTargetFold = localTransferrer.getTargetFold();
	CommunicationReqFinder commReqFinder = CommunicationReqFinder(localSourceFold, localTargetFold);
	if (!commReqFinder.isCrossSegmentCommRequired(*logFile)) return;

	// When there is a need for cross-segment communications, each segment needs to know what other segments have for
	// the source and target parts lists. A checking is first made if that information is already available. If not 
	// then the information is collected by collective segment-fold gathering by all segments.
	PartsListReference *sourceRef = transferConfig->getSourceReference();
	PartsListAttributes *sourceAttrs = sourceRef->getPartsList()->getAttributes();
	if (sourceAttrs->isSegmentMappingKnown()) {
		sourceContentMap = sourceAttrs->getSegmentMapping();
	} else {
		SegmentMappingPreparer mappingPreparer = SegmentMappingPreparer(localSourceFold);
		sourceContentMap = mappingPreparer.shareSegmentsContents(*logFile);
		sourceAttrs->setSegmentsContents(sourceContentMap);
	}
	char *dataItemId = transferConfig->getDataItemId();
	char *targetKey = transferConfig->generateTargetVersionkey();
	ProgramEnvironment *progEnv = transferConfig->getProgEnv();
	PartsListReference *targetRef = progEnv->getVersionManager(dataItemId)->getVersion(targetKey);
	bool targetMappingRetrieved = false;
	if (targetRef != NULL && !computeTargetFoldAfresh) {
		PartsListAttributes *targetAttrs = targetRef->getPartsList()->getAttributes();
		if (targetAttrs->isSegmentMappingKnown()) {
			targetContentMap = targetAttrs->getSegmentMapping();
			targetMappingRetrieved = true;
		}
	}
	if (!targetMappingRetrieved) {
		SegmentMappingPreparer mappingPreparer = SegmentMappingPreparer(localTargetFold);
		targetContentMap = mappingPreparer.shareSegmentsContents(*logFile);
	}

	// then prepare transfer buffers for all the incoming and outgoing messages; note that the way segment content map
	// is organized, it is ensured that buffers are placed in proper order (by tags) in the resulting lists
	int elementSize = transferConfig->getElementSize();
	DataItemConfig *sourceConfig = localTransferrer.getSourceConfig()->generateStateFulVersion();
	List<DataPart*> *sourceParts = localTransferrer.getSourcePartList();
	DataPartSpec *sourcePartSpec = new DataPartSpec(sourceParts, sourceConfig);
	DataItemConfig *targetConfig = localTransferrer.getTargetConfig()->generateStateFulVersion();
	List<DataPart*> *targetParts = localTransferrer.getTargetPartList();
	DataPartSpec *targetPartSpec = new DataPartSpec(targetParts, targetConfig);
	TransferBuffersPreparer buffersPreparer = TransferBuffersPreparer(elementSize, localSourceFold,
                        sourcePartSpec, localTargetFold, targetPartSpec);
	PartIdContainer *sourceContainer = localTransferrer.getSourcePartsTree();
	PartIdContainer *targetContainer = localTransferrer.getTargetPartsTree();
	List<TransferBuffer*> *incomingBuffers 
			= buffersPreparer.createBuffersForIncomingTransfers(targetContainer, 
					sourceContentMap, *logFile);
	List<TransferBuffer*> *outgoingBuffers 
			= buffersPreparer.createBuffersForOutgoingTransfers(sourceContainer, 
					targetContentMap, *logFile);

	// then first issue asynchronous receives for all incoming buffers then issue asynchronous sends for all outgoing
	// buffers; then wait for all transfers to finish
	BufferTransferrer *buffReceiver = new BufferTransferrer(false, incomingBuffers);
	buffReceiver->receiveDataAsync(*logFile);
	BufferTransferrer *buffSender = new BufferTransferrer(true, outgoingBuffers);
	buffSender->sendDataAsync(*logFile);
	buffSender->waitForTransferComplete(*logFile);
	buffReceiver->waitForTransferComplete(*logFile);

	// delete all the transfer buffers
	if (incomingBuffers != NULL) {
		while (incomingBuffers->NumElements() > 0) {
			TransferBuffer *buffer = incomingBuffers->Nth(0);
			incomingBuffers->RemoveAt(0);
			delete buffer;
		}
		delete incomingBuffers;
	}
	if (outgoingBuffers != NULL) {
		while (outgoingBuffers->NumElements() > 0) {
			TransferBuffer *buffer = outgoingBuffers->Nth(0);
			outgoingBuffers->RemoveAt(0);
			delete buffer;
		}
		delete outgoingBuffers;
	}
	
	// release the memories consumed for locally created dynamic objects
	delete sourceConfig;
	delete targetConfig;
	delete sourcePartSpec;
	delete targetPartSpec;
	delete buffSender;
	delete buffReceiver; 			
}
