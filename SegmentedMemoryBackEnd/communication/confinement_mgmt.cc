#include "confinement_mgmt.h"
#include "part_distribution.h"
#include "part_folding.h"
#include "part_config.h"
#include "../utils/list.h"
#include "../utils/interval.h"
#include "../utils/id_generation.h"
#include "../utils/binary_search.h"
#include "../partition-lib/partition.h"
#include "../memory-management/part_tracking.h"
#include <iostream>
#include <vector>
#include <sstream>
#include <algorithm>

using namespace std;

//--------------------------------------- Confinement Construction Configuration ---------------------------------------------/

ConfinementConstructionConfig::	ConfinementConstructionConfig(int lST,
		int sLps, DataItemConfig *sCon,
		int rLps, DataItemConfig *rCon,
		int cLps,
		PartIdContainer *sPTree, PartIdContainer *rPTree,
		BranchingContainer *pDTree) {
	localSegmentTag = lST;
	senderLps = sLps;
	senderConfig = sCon;
	receiverLps = rLps;
	receiverConfig = rCon;
	confinementLps = cLps;
	senderPartTree = sPTree;
	receiverPartTree = rPTree;
	partDistributionTree = pDTree;
}

vector<LpsDimConfig> *ConfinementConstructionConfig::getSendPathInDistributionTree() {
	vector<LpsDimConfig> *configVector = senderConfig->generateDimOrderVector();
	int confinementLevel = getConfinementLevel(configVector);
	if (senderLps == confinementLps) {
		return forwardTruncateDimVector(confinementLevel, configVector);
	} else {
		return forwardTruncateDimVector(confinementLevel + 1, configVector);
	}
}

vector<LpsDimConfig> *ConfinementConstructionConfig::getReceivePathInDistributionTree() {
	vector<LpsDimConfig> *configVector = receiverConfig->generateDimOrderVector();
	int confinementLevel = getConfinementLevel(configVector);
	if (receiverLps == confinementLps) {
		return forwardTruncateDimVector(confinementLevel, configVector);
	} else {
		return forwardTruncateDimVector(confinementLevel + 1, configVector);
	}
}

vector<LpsDimConfig> *ConfinementConstructionConfig::forwardTruncateDimVector(int truncateLevel,
		std::vector<LpsDimConfig> *originalVector) {
	vector<LpsDimConfig> *truncatedVector = new vector<LpsDimConfig>;
	unsigned int index = 0;
	while (originalVector->at(index).getLevel() != truncateLevel) index++;
	truncatedVector->insert(truncatedVector->begin(),
			originalVector->begin() + index, originalVector->end());
	return truncatedVector;
}

vector<LpsDimConfig> *ConfinementConstructionConfig::backwardTruncateDimVector(int truncateLevel,
		std::vector<LpsDimConfig> *originalVector) {
	vector<LpsDimConfig> *truncatedVector = new vector<LpsDimConfig>;
	unsigned int index = originalVector->size() - 1;
	while (originalVector->at(index).getLevel() != truncateLevel) index--;
	truncatedVector->insert(truncatedVector->begin(),
			originalVector->begin(), originalVector->begin() + index + 1);
	return truncatedVector;
}

int ConfinementConstructionConfig::getConfinementLevel(vector<LpsDimConfig> *configVector) {
	if (configVector == NULL) {
		configVector = senderConfig->generateDimOrderVector();
	}
	int level = -1;
	for (unsigned int i = 0; i < configVector->size(); i++) {
		LpsDimConfig dimConfig = configVector->at(i);
		if (dimConfig.getLpsId() == confinementLps) return dimConfig.getLevel();
	}
	return level;
}

List<List<int*>*> *ConfinementConstructionConfig::getConfinementIds(int confinementLevel) {

	if (senderPartTree == NULL && receiverPartTree == NULL) return NULL;

	int dimensions = senderConfig->getDimensionality();
	List<List<int*>*> *confinementList = new List<List<int*>*>;
	if (senderPartTree != NULL) {
		List<List<int*>*> *sendConfinments = senderPartTree->getAllPartIdsAtLevel(confinementLevel, dimensions);
		if (sendConfinments != NULL) {
			confinementList->AppendAll(sendConfinments);
			delete sendConfinments;
		}
	}
	if (senderLps != receiverLps && receiverPartTree != NULL) {
		List<List<int*>*> *receiveConfinments = receiverPartTree->getAllPartIdsAtLevel(confinementLevel, dimensions);
		if (receiveConfinments != NULL) {
			int existingConfinements = confinementList->NumElements();
			for (int i = 0; i < receiveConfinments->NumElements(); i++) {
				List<int*> *candidate = receiveConfinments->Nth(i);
				bool found = false;
				for (int j = 0; j < existingConfinements; j++) {
					List<int*> *reference = confinementList->Nth(j);
					if (idutils::areIdsEqual(candidate, reference, dimensions)) {
						found = true;
						break;
					}
				}
				if (!found) {
					confinementList->Append(candidate);
				} else {
					while (candidate->NumElements() > 0) {
						int *idAtLevel = candidate->Nth(0);
						candidate->RemoveAt(0);
						delete[] idAtLevel;
					}
					delete candidate;
				}
			}
			delete receiveConfinments;
		}
	}
	if (confinementList->NumElements() == 0) {
		delete confinementList;
		return NULL;
	}
	return confinementList;
}

void ConfinementConstructionConfig::configurePaddingInPartitionConfigs() {

	// On and above the confinement level all paddings should be included in interval description generation.
	// Below confinement, sender paddings should not be considered as we want to avoid same date being sent
	// from multiple participants.
	setPaddingThresholdInDataConfig(senderConfig, confinementLps);

	// For the receiver side, padding down to the receiver LPS must be included as each receiver branch needs
	// to fill its content, regardless of that being part of it being ghost boundary region or not. Below the
	// receiver LPS level, paddings need not be included as each receiver can supply the data to its lower
	// level for ghost boundary regions.
	setPaddingThresholdInDataConfig(receiverConfig, receiverLps);
}

void ConfinementConstructionConfig::setPaddingThresholdInDataConfig(DataItemConfig *dataConfig,
		int thresholdLps) {

	int levels = dataConfig->getLevels();
	int dataDimensions = getDataDimensions();
	int level = 0;
	for (; level < levels; level++) {
		PartitionConfig *config = dataConfig->getConfigForLevel(level);
		// this logic work because hierarchically related LPSes are numbered in increasing order
		if (config->getLpsId() <= thresholdLps) {
			for (int dimNo = 0; dimNo < dataDimensions; dimNo++) {
				PartitionInstr *instr = config->getInstruction(dimNo);
				instr->setExcludePaddingFlag(false);
			}
		} else break;
	}
	for (; level < levels; level++) {
		PartitionConfig *config = dataConfig->getConfigForLevel(level);
		for (int dimNo = 0; dimNo < dataDimensions; dimNo++) {
			PartitionInstr *instr = config->getInstruction(dimNo);
			instr->setExcludePaddingFlag(true);
		}
	}
}

//----------------------------------------------------- Participant ----------------------------------------------------------/

Participant::Participant(CommRole r, std::vector<int*> *c, List<MultidimensionalIntervalSeq*> *d) {
	role = r;
	containerId = c;
	dataDescription = d;
	id = 0;
}

Participant::~Participant() {
	delete containerId;
	while (dataDescription->NumElements() > 0) {
		MultidimensionalIntervalSeq *seq = dataDescription->Nth(0);
		dataDescription->RemoveAt(0);
		delete seq;
	}
	delete dataDescription;
}

void Participant::addSegmentTag(int segmentTag) {
	int location = binsearch::locateKey(segmentTags, segmentTag);
	if (location == KEY_NOT_FOUND) {
		location = binsearch::locatePointOfInsert(segmentTags, segmentTag);
		segmentTags.insert(segmentTags.begin() + location, segmentTag);
	}
}

bool Participant::hasSegmentTag(int segmentTag) {
	return binsearch::locateKey(segmentTags, segmentTag) != KEY_NOT_FOUND;
}

bool Participant::isEqual(Participant *other) {
	List<MultidimensionalIntervalSeq*> *otherDesc = other->dataDescription;
	if (otherDesc->NumElements() != dataDescription->NumElements()) return false;
	for (int i = 0; i < dataDescription->NumElements(); i++) {
		MultidimensionalIntervalSeq *seq1 = dataDescription->Nth(i);
		bool found = false;
		for (int j = 0; j < otherDesc->NumElements(); j++) {
			MultidimensionalIntervalSeq *seq2 = otherDesc->Nth(j);
			if (seq1->isEqual(seq2)) {
				found = true;
				break;
			}
		}
		if (!found) return false;
	}
	return true;
}

//---------------------------------------------------- Data Exchange ---------------------------------------------------------/

DataExchange::DataExchange(Participant *sender, Participant *receiver) {
	this->sender = sender;
	this->receiver = receiver;
	exchangeDesc = sender->getDataDescription();
	fullOverlap = true;
}

DataExchange::DataExchange(Participant *sender,
		Participant *receiver, List<MultidimensionalIntervalSeq*> *exchangeDesc) {
	this->sender = sender;
	this->receiver = receiver;
	List<MultidimensionalIntervalSeq*> *orderedSeqList = new List<MultidimensionalIntervalSeq*>;
	orderedSeqList->Append(exchangeDesc->Nth(0));
	for (int i = 1; i < exchangeDesc->NumElements(); i++) {
		MultidimensionalIntervalSeq *currentSeq = exchangeDesc->Nth(i);
		int insertLocation = orderedSeqList->NumElements();
		for (int j = 0; j < orderedSeqList->NumElements(); j++) {
			MultidimensionalIntervalSeq *referenceSeq = orderedSeqList->Nth(j);
			if (currentSeq->compareTo(referenceSeq) < 0) {
				insertLocation = j;
				break;
			}
		}
		orderedSeqList->InsertAt(currentSeq, insertLocation);
	}
	this->exchangeDesc = orderedSeqList;
	fullOverlap = false;
}

DataExchange::~DataExchange() {
	if (!fullOverlap) {
		while (exchangeDesc->NumElements() > 0) {
			MultidimensionalIntervalSeq *seq = exchangeDesc->Nth(0);
			exchangeDesc->RemoveAt(0);
			delete seq;
		}
		delete exchangeDesc;
	}
}

void DataExchange::describe(int indentLevel, ostream &stream) {
	ostringstream indent;
	for (int i = 0; i < indentLevel; i++) indent << '\t';
	stream << indent.str() << "Sender No: " << sender->getId() << "\n";
	stream << indent.str() << "Sender Segments: ";
	vector<int> sendTags = sender->getSegmentTags();
	for (unsigned int i = 0; i < sendTags.size(); i++) {
		stream << sendTags[i] << ' ';
	}
	stream << indent.str() << "\nReceiver No: " << receiver->getId() << "\n";
	stream << indent.str() << "Receiver Segments: ";
	vector<int> receiveTags = receiver->getSegmentTags();
	for (unsigned int i = 0; i < receiveTags.size(); i++) {
		stream << receiveTags[i] << ' ';
	}
	stream << '\n' << indent.str() << "Sender Data Content:\n";
	drawDataDescription(sender->getDataDescription());
	stream << '\n' << indent.str() << "Receiver Data Content:\n";
	drawDataDescription(receiver->getDataDescription());
	stream << "\n" << indent.str() << "Total data items: " << getTotalElementsCount() << "\n";
	stream << '\n' << indent.str() << "To be Exchanged Data: \n";
	drawDataDescription(exchangeDesc);
}

int DataExchange::getTotalElementsCount() {
	int count = exchangeDesc->Nth(0)->getNumOfElements();
	for (int i = 1; i < exchangeDesc->NumElements(); i++) {
		count += exchangeDesc->Nth(i)->getNumOfElements();
	}
	return count;
}

List<MultidimensionalIntervalSeq*> *DataExchange::getCommonRegion(Participant *sender, Participant *receiver) {

	List<MultidimensionalIntervalSeq*> *overlap = new List<MultidimensionalIntervalSeq*>;
	List<MultidimensionalIntervalSeq*> *senderData = sender->getDataDescription();
	List<MultidimensionalIntervalSeq*> *receiverData = receiver->getDataDescription();

	for (int i = 0; i < senderData->NumElements(); i++) {
		MultidimensionalIntervalSeq *seq1 = senderData->Nth(i);
		for (int j = 0; j < receiverData->NumElements(); j++) {
			MultidimensionalIntervalSeq *seq2 = receiverData->Nth(j);
			List<MultidimensionalIntervalSeq*> *intersection = seq1->computeIntersection(seq2);
			if (intersection != NULL) {
				overlap->AppendAll(intersection);
				delete intersection;
			}
		}
	}
	if (overlap->NumElements() == 0) {
		delete overlap;
		return NULL;
	}
	return overlap;
}

int DataExchange::compareTo(DataExchange *other, bool forReceive) {
	
	vector<int> mySegmentTags;
	vector<int> othersSegmentTags;
	if (forReceive) {
		mySegmentTags = sender->getSegmentTags();
		othersSegmentTags = other->sender->getSegmentTags();
	} else {
		mySegmentTags = receiver->getSegmentTags();
		othersSegmentTags = other->receiver->getSegmentTags();
	}
	
	// try to prioritize the exchange having participants with lower PPU IDs in the other side
	int smallestTagVectorSize = min(mySegmentTags.size(), othersSegmentTags.size());
	for (int i = 0; i < smallestTagVectorSize; i++) {
		if (mySegmentTags.at(i) < othersSegmentTags.at(i)) return -1;
		else if (mySegmentTags.at(i) > othersSegmentTags.at(i)) return 1;
	}

	// then try to prioritize the exchange that have more PPUs waiting-on/signaling-from the other side
	if (mySegmentTags.size() > othersSegmentTags.size()) return -1;
	else if (mySegmentTags.size() < othersSegmentTags.size()) return 1;

	// if the exchanges are equivalent in both cases then prioritize the one that communicate more data
	int myEntries = getTotalElementsCount();
	int othersEntries = other->getTotalElementsCount();
	if (myEntries < othersEntries) return -1;
	else if (myEntries > othersEntries) return 1;

	return 0;
}

bool DataExchange::isIntraSegmentExchange(int localSegmentTag) {
	vector<int> sendersTags = sender->getSegmentTags();
	if (sendersTags.size() > 1 || sendersTags.at(0) != localSegmentTag) return false;
	vector<int> receiversTags = receiver->getSegmentTags();
	if (receiversTags.size() > 1 
		|| receiversTags.at(0) != localSegmentTag) return false;
	return true;
}

bool DataExchange::involvesLocalSegment(int localSegmentTag) {
	return sender->hasSegmentTag(localSegmentTag) || receiver->hasSegmentTag(localSegmentTag);
}

void DataExchange::drawDataDescription(List<MultidimensionalIntervalSeq*> *seqList) {
	for (int i = 0; i < seqList->NumElements(); i++) {
		cout << "Sequence #" << i << ":\n";
		MultidimensionalIntervalSeq *seq = seqList->Nth(i);
		seq->draw();
	}
}

bool DataExchange::contentsEqual(DataExchange *other) {
	if (this->exchangeDesc->NumElements() != other->exchangeDesc->NumElements()) return false;
	for (int i = 0; i < this->exchangeDesc->NumElements(); i++) {
		MultidimensionalIntervalSeq *mySeq = this->exchangeDesc->Nth(i);
		bool seqFound = false;
		for (int j = 0; j < other->exchangeDesc->NumElements(); j++) {
			MultidimensionalIntervalSeq *otherSeq = other->exchangeDesc->Nth(j);
			if (mySeq->isEqual(otherSeq)) {
				seqFound = true;
				break;
			}
		}
		if (!seqFound) return false;
	}
	return true;
}

void DataExchange::mergeWithOther(DataExchange *other) {
	vector<int> senderTagsInOther = other->getSender()->getSegmentTags();
	for (int i = 0; i < senderTagsInOther.size(); i++) {
		sender->addSegmentTag(senderTagsInOther.at(i));
	}
	vector<int> receiverTagsInOther = other->receiver->getSegmentTags();
	for (int i = 0; i < receiverTagsInOther.size(); i++) {
		receiver->addSegmentTag(receiverTagsInOther.at(i));
	}
}

int DataExchange::getTotalParticipantsCount(List<DataExchange*> *exchangeList, bool sendingSide) {
	std::vector<int> participantVector;
	for (int i = 0; i < exchangeList->NumElements(); i++) {
		DataExchange *exchange = exchangeList->Nth(i);
		std::vector<int> exchangeTags;
		if (sendingSide) {
			exchangeTags = exchange->getSender()->getSegmentTags();
		} else {
			exchangeTags = exchange->getReceiver()->getSegmentTags();
		}
		for (int j = 0; j < exchangeTags.size(); j++) {
			int currentTag = exchangeTags.at(j);
			binsearch::insertIfNotExist(&participantVector, currentTag);
		}
	}
	return participantVector.size();
}

//-------------------------------------- Cross Segment Interaction Specification ---------------------------------------------/

CrossSegmentInteractionSpec::CrossSegmentInteractionSpec(Container *container, ConfinementConstructionConfig *config) {

	this->localSegmentTag = config->getLocalSegmentTag();

	DataItemConfig *senderConfig = config->getSenderConfig();
	vector<LpsDimConfig> *senderPath = senderConfig->generateDimOrderVector();
	int confinementLevel = config->getConfinementLevel(senderPath);
	int dataDimensions = senderConfig->getDimensionality();
	generateParticipantList(container,
			dataDimensions, SEND, *senderPath, senderConfig, confinementLevel);

	// even when the sender and the receivers are from the same LPS, we need to generate separate participants lists 
	// for them as the data part configuration may be different, e.g., one can have padding enabled in some levels 
	// but other not
	DataItemConfig *receiverConfig = config->getReceiverConfig();
	vector<LpsDimConfig> *receiverPath = receiverConfig->generateDimOrderVector();
	generateParticipantList(container,
				dataDimensions, RECEIVE, *receiverPath, receiverConfig, confinementLevel);

	this->localInteractionAllowed = !config->isIntraContrainerSync();
}

List<DataExchange*> *CrossSegmentInteractionSpec::generateSendExchanges() {
	if (localSender == NULL) return NULL;
	List<Participant*> *localSenderList = new List<Participant*>;
	localSenderList->Append(localSender);
	if (localReceiver == NULL || !localInteractionAllowed) {
		return Confinement::generateDataExchangeList(localSenderList, remoteReceivers);
	} else {
		List<Participant*> *receiverList = new List<Participant*>;
		receiverList->AppendAll(remoteReceivers);
		receiverList->Append(localReceiver);
		return Confinement::generateDataExchangeList(localSenderList, receiverList);
	}
}

List<DataExchange*> *CrossSegmentInteractionSpec::generateReceiveExchanges() {
	if (localReceiver == NULL) return NULL;
	List<Participant*> *localReceiverList = new List<Participant*>;
	localReceiverList->Append(localReceiver);
	return Confinement::generateDataExchangeList(remoteSenders, localReceiverList);
}

List<DataExchange*> *CrossSegmentInteractionSpec::generateRemoteExchanges() {
	return Confinement::generateDataExchangeList(remoteSenders, remoteReceivers);
}

void CrossSegmentInteractionSpec::describe(int indentLevel, std::ostream &stream) {
	ostringstream indent;
	for (int i = 0; i < indentLevel; i++) indent << '\t';
	if (localSender != NULL && remoteReceivers->NumElements() > 0) {
		stream << indent.str() << "Send is activated from local operating memory\n";
		stream << indent.str() << "Remote Receivers: ";
		stream << remoteReceivers->NumElements() << "\n";
	} else {
		stream << indent.str() << "Current segment will not send any data\n";
	}
	if (localReceiver != NULL && remoteSenders->NumElements() > 0) {
		stream << indent.str() << "Receive is activated from local operating memory\n";
		stream << indent.str() << "Remote Senders: ";
		stream << remoteSenders->NumElements() << "\n";
	} else {
		stream << indent.str() << "Current segment will not receive any data\n";
	}
	if (localInteractionAllowed && localSender != NULL && localReceiver != NULL) {
		stream << indent.str() << "Involves local transfer between part hierarchies\n";
	}
}

void CrossSegmentInteractionSpec::generateParticipantList(Container *confinementContainer,
		int dataDimensions,
		CommRole role,
		std::vector<LpsDimConfig> pathOrder,
		DataItemConfig *dataConfig,
		int pruningLevel) {

	List<Participant*> *remoteParticipants = new List<Participant*>;
	Participant *localParticipant = NULL;
	List<PartFolding*> *foldingList = new List<PartFolding*>;
	vector<int> segmentTags = confinementContainer->getSegmentTags();
	vector<int*> *containerId = confinementContainer->getPartId(dataDimensions);

	// Sometimes the confinement level may be the root of the distribution hierarchy. We cannot prune the root in the
	// folded partition description of a segment's data content. The best that can be done is to prune up to the first
	// LPS level along the data structure's partition configuration hierarchy. Therefore an adjustment has been made
	// here for the active pruning level.
	int activePruningLevel = max(0, pruningLevel);

	for (unsigned int i = 0; i < segmentTags.size(); i++) {

		int segmentTag = segmentTags.at(i);
		PartFolding *folding = confinementContainer->foldContainerForSegment(segmentTag, pathOrder, true);
		if (folding == NULL) continue;

		// Note that individual dimension folds get pruned before we generate the interval description for a fold. 
		// Still we attempt an initial pruning of the entire multidimensional fold before to reduce the number of 
		// strands that we will need to consider later. This pruning below is not essential for the correctness of 
		// the final interval description but it improves the chance of that being more compact.
		folding->pruneFolding(activePruningLevel, dataConfig);

		if (segmentTag == localSegmentTag) {
			List<MultidimensionalIntervalSeq*> *dataDesc = folding->generateIntervalDesc(dataConfig);
			if (dataDesc != NULL) {
				localParticipant = new Participant(role, containerId, dataDesc);
				localParticipant->addSegmentTag(localSegmentTag);
			}
			continue;
		}

		bool foldingFound = false;
		int matchingIndex = -1;
		for (int k = 0; k < foldingList->NumElements(); k++) {
			PartFolding *reference = foldingList->Nth(k);
			if (reference->isEqualInContent(folding)) {
				foldingFound = true;
				matchingIndex = k;
				break;
			}
		}
		if (!foldingFound) {
			List<MultidimensionalIntervalSeq*> *dataDesc = folding->generateIntervalDesc(dataConfig);
			if (dataDesc != NULL) {
				Participant *participant = new Participant(role, containerId, dataDesc);
				int senderCountSoFar = remoteParticipants->NumElements();
				participant->setId(senderCountSoFar);
				participant->addSegmentTag(segmentTag);
				remoteParticipants->Append(participant);
			}
		} else {
			Participant *participant = remoteParticipants->Nth(matchingIndex);
			participant->addSegmentTag(segmentTag);
			delete folding;
		}
	}

	while (foldingList->NumElements() > 0) {
		PartFolding *folding = foldingList->Nth(0);
		foldingList->RemoveAt(0);
		delete folding;
	}

	if (role == SEND) {
		this->localSender = localParticipant;
		this->remoteSenders = remoteParticipants;
	} else {
		this->localReceiver = localParticipant;
		this->remoteReceivers = remoteParticipants;
	}
}

//------------------------------------- Within Container Interaction Specification -------------------------------------------/

IntraContainerInteractionSpec::IntraContainerInteractionSpec(BranchingContainer *container,
		ConfinementConstructionConfig *config) {

	this->localSegmentTag = config->getLocalSegmentTag();
	this->dataDimensions = config->getDataDimensions();

	vector<LpsDimConfig> *senderPath = config->getSendPathInDistributionTree();
	int senderBranchHeaderLps = senderPath->at(0).getLpsId();
	List<Container*> *senderBranches = container->listDescendantContainersForLps(senderBranchHeaderLps, 0, false);
	vector<LpsDimConfig> *receiverPath = config->getReceivePathInDistributionTree();
	int receiverBranchHeaderLps = receiverPath->at(0).getLpsId();
	List<Container*> *receiverBranches = NULL;
	if (!config->isIntraContrainerSync()) {
		receiverBranches = container->listDescendantContainersForLps(receiverBranchHeaderLps, 0, false);
	} else receiverBranches = senderBranches;

	int senderBranchLevel = senderPath->at(0).getLevel();
	senderList = generateParticipantList(senderBranches,
			SEND, *senderPath, config->getSenderConfig(), senderBranchLevel);
	int receiverBranchLevel = receiverPath->at(0).getLevel();
	receiverList = generateParticipantList(receiverBranches,
			RECEIVE, *receiverPath, config->getReceiverConfig(), receiverBranchLevel);
}

List<Participant*> *IntraContainerInteractionSpec::generateParticipantList(List<Container*> *participantBranches,
		CommRole role,
		vector<LpsDimConfig> pathOrder,
		DataItemConfig *dataConfig, int pruningLevel) {

	List<Participant*> *participantList = new List<Participant*>;
	List<PartFolding*> *foldingList = new List<PartFolding*>;

	for (int j = 0; j < participantBranches->NumElements(); j++) {
		Container *branch = participantBranches->Nth(j);
		vector<int> segmentTags = branch->getSegmentTags();
		PartFolding *folding = branch->foldContainerForSegment(localSegmentTag, pathOrder, true);
		if (folding == NULL) continue;
		folding->pruneFolding(pruningLevel, dataConfig);
		vector<int*> *containerId = branch->getPartId(dataDimensions);
		List<MultidimensionalIntervalSeq*> *dataDesc = folding->generateIntervalDesc(dataConfig);
		if (dataDesc != NULL) {
			Participant *participant = new Participant(role, containerId, dataDesc);
			int senderCountSoFar = participantList->NumElements();
			participant->setId(senderCountSoFar);
			participant->addSegmentTag(localSegmentTag);
			participantList->Append(participant);
		}
	}

	while (foldingList->NumElements() > 0) {
		PartFolding *folding = foldingList->Nth(0);
		foldingList->RemoveAt(0);
		delete folding;
	}

	return participantList;
}

List<DataExchange*> *IntraContainerInteractionSpec::generateExchanges() {
	List<DataExchange*> *exchangeList = Confinement::generateDataExchangeList(senderList, receiverList);
	if (exchangeList != NULL) {
		List<DataExchange*> *filteredList = new List<DataExchange*>;
		// keep only those exchanges that are not circular in nature
		for (int i = 0; i < exchangeList->NumElements(); i++) {
			DataExchange *exchange = exchangeList->Nth(i);
			vector<int*> *senderId = exchange->getSender()->getContainerId();
			vector<int*> *receiverId = exchange->getReceiver()->getContainerId();
			if (!idutils::areIdsEqual(senderId, receiverId, dataDimensions)) {
				filteredList->Append(exchange);
			}
		}
		delete exchangeList;
		return filteredList;
	}
	return exchangeList;
}

void IntraContainerInteractionSpec::describe(int indentLevel, ostream &stream) {
	ostringstream indent;
	for (int i = 0; i < indentLevel; i++) indent << '\t';
	stream << indent.str() << "Participating branch count: " << senderList->NumElements();
	for (int i = 0; i < senderList->NumElements(); i++) {
		stream << '\n' << indent.str() << "\tParticipant #" << i << ": ";
		Participant *participant = senderList->Nth(i);
		vector<int*> *containerId = participant->getContainerId();
		for (unsigned int level = 0; level < containerId->size(); level++) {
			stream << "[";
			int *idAtLevel = containerId->at(level);
			for (int dim = 0; dim < dataDimensions; dim++) {
				if (dim > 0) stream << ", ";
				stream << idAtLevel[dim];
			}
			stream << "]";
		}
	}
}

//----------------------------------------------------- Confinement ----------------------------------------------------------/

Confinement::Confinement(int dd, BranchingContainer *cC, ConfinementConstructionConfig *config) {

	dataDimensions = dd;
	confinementContainer = cC;
	participantSegments = cC->getSegmentTags();
	remoteInteractions = new CrossSegmentInteractionSpec(confinementContainer, config);
	bool localInterchangeApplicable = config->isIntraContrainerSync();
	if (localInterchangeApplicable) {
		localInteractions = new IntraContainerInteractionSpec(confinementContainer, config);
	} else localInteractions = NULL;
}

List<DataExchange*> *Confinement::getAllDataExchanges() {

	List<DataExchange*> *dataExchangeList = new List<DataExchange*>;
	List<DataExchange*> *remoteSends = remoteInteractions->generateSendExchanges();
	if (remoteSends != NULL) {
		dataExchangeList->AppendAll(remoteSends);
		delete remoteSends;
	}
	List<DataExchange*> *remoteReceives = remoteInteractions->generateReceiveExchanges();
	if (remoteReceives != NULL) {
		dataExchangeList->AppendAll(remoteReceives);
		delete remoteReceives;
	}

	List<DataExchange*> *nonLocalTransfers = remoteInteractions->generateRemoteExchanges();
	if (nonLocalTransfers != NULL) {
		dataExchangeList->AppendAll(nonLocalTransfers);
		delete nonLocalTransfers;
	}

	// Compacting is done only for non-local data exchanges as there should be no two data local data exchanges that
	// have the same data content. If they do then there is an error in the data exchange generation process.
	List<DataExchange*> *compactList = new List<DataExchange*>;
	while (dataExchangeList->NumElements() > 0) {
		
		DataExchange *exchange = dataExchangeList->Nth(0);
		dataExchangeList->RemoveAt(0);

		DataExchange *matchingExchange = NULL;
		for (int j = 0; j < compactList->NumElements(); j++) {
			if (exchange->contentsEqual(compactList->Nth(j))) {
				matchingExchange = compactList->Nth(j);
				break;
			}
		}
		if (matchingExchange != NULL) {
			matchingExchange->mergeWithOther(exchange);
			delete exchange;
		} else {
			compactList->Append(exchange);
		}
	}
	dataExchangeList->AppendAll(compactList);
	delete compactList;

	if (localInteractions != NULL) {
		List<DataExchange*> *localExchanges = localInteractions->generateExchanges();
		if (localExchanges != NULL) {
			dataExchangeList->AppendAll(localExchanges);
			delete localExchanges;
		}
	}
	if (dataExchangeList->NumElements() == 0) {
		delete dataExchangeList;
		return NULL;
	}
	return dataExchangeList;
}

List<Confinement*> *Confinement::generateAllConfinements(ConfinementConstructionConfig *config, int rootLps) {

	// setup the data partition configurations of sender and receiver sides properly before they been used in any
	// interval description construction process
	config->configurePaddingInPartitionConfigs();

	// If the confinement level is the root LPS (that is there is no sharing between the sender and the receiver
	// LPS paths) then there will be a single confinement root for the underlying data synchronization. So we create
	// the confinement using the part-distribution-tree and return it
	BranchingContainer *partDistributionTree = config->getDistributionTree();
	int dataDimensions = config->getDataDimensions();
	if (config->isRootConfinement(rootLps)) {
		List<Confinement*> *confinementList = new List<Confinement*>;
		Confinement *rootConfinement = new Confinement(dataDimensions, partDistributionTree, config);
		confinementList->Append(rootConfinement);
		return confinementList;
	}

	// determine the confinement level to retrieve all confinement roots' id from part-tracking hierarchies and the
	// confinement vector to locate the corresponding containers in the distribution tree
	vector<LpsDimConfig> *senderVector = config->getSenderConfig()->generateDimOrderVector();
	int confinementLevel = config->getConfinementLevel(senderVector);
	vector<LpsDimConfig> *confinementVector = config->backwardTruncateDimVector(confinementLevel, senderVector);
	List<Confinement*> *confinementList = new List<Confinement*>;

	// any of the part-tracking-tree can be NULL indicating that the executing segment has no portion of the data
	// structure
	PartIdContainer *senderTree = config->getSenderPartTree();
	if (senderTree != NULL) {
		List<List<int*>*> *partIdList = senderTree->getAllPartIdsAtLevel(confinementLevel, dataDimensions);
		for (int i = 0; i < partIdList->NumElements(); i++) {
			List<int*> *containerId = partIdList->Nth(i);
			Container *container = partDistributionTree->getContainer(containerId, *confinementVector);
			BranchingContainer *branchContainer = reinterpret_cast<BranchingContainer*>(container);
			Confinement *confinement = new Confinement(dataDimensions, branchContainer, config);
			confinementList->Append(confinement);
		}
	}

	// if the underlying synchronization is intended for the parts of the same part-tracking-tree then the sender
	// tree processing has already returned all confinement roots
	PartIdContainer *receiverTree = config->getReceiverPartTree();
	int confinementsFoundBefore = confinementList->NumElements();
	if (!config->isIntraContrainerSync() && receiverTree != NULL) {
		List<List<int*>*> *partIdList = receiverTree->getAllPartIdsAtLevel(confinementLevel, dataDimensions);
		for (int i = 0; i < partIdList->NumElements(); i++) {
			List<int*> *containerId = partIdList->Nth(i);
			Container *container = partDistributionTree->getContainer(containerId, *confinementVector);
			BranchingContainer *branchContainer = reinterpret_cast<BranchingContainer*>(container);

			// before adding the second set of confinement roots for the receiver paths, we need to do 
			// redundancy checking against the confinements found by searching for the senders as some of 
			// the confinements in two sides of the communication may be common
			bool confinementExists = false;
			for (int j = 0; j < confinementsFoundBefore; j++) {
				Confinement *sendConfinement = confinementList->Nth(j);
				if (sendConfinement->getContainer() == branchContainer) {
					confinementExists = true;
					break;
				}
			}
			if (!confinementExists) {
				Confinement *confinement = new Confinement(dataDimensions, branchContainer, config);
				confinementList->Append(confinement);
			}
		}
	}

	return confinementList;
}

List<DataExchange*> *Confinement::generateDataExchangeList(List<Participant*> *senderList,
		List<Participant*> *receiverList) {

	List<DataExchange*> *dataExchangeList = new List<DataExchange*>;
	for (int i = 0; i < senderList->NumElements(); i++) {
		Participant *sender = senderList->Nth(i);
		for (int j = 0; j < receiverList->NumElements(); j++) {
			Participant *receiver = receiverList->Nth(j);
			if (sender->isEqual(receiver)) {
				DataExchange *exchange = new DataExchange(sender, receiver);
				dataExchangeList->Append(exchange);
				continue;
			}
			List<MultidimensionalIntervalSeq*> *overlap = DataExchange::getCommonRegion(sender, receiver);
			if (overlap != NULL) {
				DataExchange *exchange = new DataExchange(sender, receiver, overlap);
				dataExchangeList->Append(exchange);
			}
		}
	}
	if (dataExchangeList == NULL) {
		delete dataExchangeList;
		return NULL;
	}
	return dataExchangeList;
}

//-------------------------------------------------- Exchange Iterator -------------------------------------------------------/

ExchangeIterator::ExchangeIterator(DataExchange *exchange) {
	this->totalElementsCount = exchange->getTotalElementsCount();
	this->sequences = exchange->getExchangeDesc();
	this->iterator = new SequenceIterator(sequences->Nth(0));
	this->currentElement = 0;
	this->currentSequence = 0;
}

ExchangeIterator::~ExchangeIterator() {
	if (iterator != NULL) delete iterator;
}

vector<int> *ExchangeIterator::getNextElement() {
	if (!iterator->hasMoreElements()) {
		delete iterator;
		currentSequence++;
		if (currentSequence < sequences->NumElements()) {
			iterator = new SequenceIterator(sequences->Nth(currentSequence));
		} else return NULL;
	}
	currentElement++;
	return iterator->getNextElement();
}

void ExchangeIterator::printNextElement(std::ostream &stream) {
	vector<int> *element = getNextElement();
	int dimensionality = sequences->Nth(0)->getDimensionality();
	for (int i = 0; i < dimensionality; i++) {
		stream << element->at(i);
		if (i < dimensionality - 1) {
			stream << ',';
		}
	}
	stream << '\n';
}
