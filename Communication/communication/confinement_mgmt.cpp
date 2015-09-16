#include "confinement_mgmt.h"
#include "../utils/list.h"
#include "../utils/interval.h"
#include "../utils/partition.h"
#include "../utils/id_generation.h"
#include "../utils/binary_search.h"
#include "part_distribution.h"
#include "../part-management/part_folding.h"
#include "../part-management/part_config.h"
#include "../part-management/part_tracking.h"
#include <iostream>
#include <vector>

using namespace std;

//--------------------------------------- Confinement Construction Configuration ---------------------------------------------/

ConfinementConstructionConfig::ConfinementConstructionConfig(int sLps, DataItemConfig *sCon,
		int rLps, DataItemConfig *rCon,
		int cLps,
		PartIdContainer *sPTree, PartIdContainer *rPTree,
		BranchingContainer *pDTree) {
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
		return forwardTruncatedDimVector(confinementLevel, configVector);
	} else {
		return forwardTruncatedDimVector(confinementLevel + 1, configVector);
	}
}

vector<LpsDimConfig> *ConfinementConstructionConfig::getReceivePathInDistributionTree() {
	vector<LpsDimConfig> *configVector = receiverConfig->generateDimOrderVector();
	int confinementLevel = getConfinementLevel(configVector);
	if (receiverLps == confinementLps) {
		return forwardTruncatedDimVector(confinementLevel, configVector);
	} else {
		return forwardTruncatedDimVector(confinementLevel + 1, configVector);
	}
}

vector<LpsDimConfig> *ConfinementConstructionConfig::forwardTruncatedDimVector(int truncateLevel,
		std::vector<LpsDimConfig> *originalVector) {
	vector<LpsDimConfig> *truncatedVector = new vector<LpsDimConfig>;
	unsigned int index = 0;
	while (originalVector->at(index).getLevel() != truncateLevel) index++;
	truncatedVector->insert(truncatedVector->begin(),
			originalVector->begin() + index, originalVector->end());
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
					if (isIdsEqual(candidate, reference, dimensions)) {
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

DataExchange::DataExchange(int senderId, int receiverId) {
	this->senderId = senderId;
	this->receiverId = receiverId;
	exchangeDesc = NULL;
	fullOverlap = true;
}

DataExchange::DataExchange(int senderId, int receiverId, List<MultidimensionalIntervalSeq*> *exchangeDesc) {
	this->senderId = senderId;
	this->receiverId = receiverId;
	this->exchangeDesc = exchangeDesc;
	fullOverlap = false;
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

//----------------------------------------------------- Confinement ----------------------------------------------------------/

Confinement::Confinement(int dd, BranchingContainer *cC, ConfinementConstructionConfig *config) {

	dataDimensions = dd;
	confinementContainer = cC;
	participantSegments = cC->getSegmentTags();

	vector<LpsDimConfig> *senderPath = config->getSendPathInDistributionTree();
	int senderBranchHeaderLps = senderPath->at(0).getLpsId();
	List<Container*> *senderBranches = confinementContainer->listDescendantContainersForLps(
			senderBranchHeaderLps, 0, false);

	vector<LpsDimConfig> *receiverPath = config->getReceivePathInDistributionTree();
	int receiverBranchHeaderLps = receiverPath->at(0).getLpsId();
	List<Container*> *receiverBranches = NULL;
	if (!config->isIntraContrainerSync()) {
		receiverBranches = confinementContainer->listDescendantContainersForLps(receiverBranchHeaderLps, 0, false);
	} else receiverBranches = senderBranches;

	int senderBranchLevel = senderPath->at(0).getLevel();
	senderList = generateParticipantList(senderBranches,
			SEND, *senderPath, config->getSenderConfig(), senderBranchLevel);

	if (!config->isIntraContrainerSync()) {
		int receiverBranchLevel = receiverPath->at(0).getLevel();
		receiverList = generateParticipantList(receiverBranches,
				RECEIVE, *receiverPath, config->getReceiverConfig(), receiverBranchLevel);
	} else {
		receiverList = senderList;
		for (int i = 0; i < senderList->NumElements(); i++) {
			senderList->Nth(i)->setRole(SEND_OR_RECEIVE);
		}
	}
}

List<DataExchange*> *Confinement::generateDataExchangeList() {
	List<DataExchange*> *dataExchangeList = new List<DataExchange*>;
	for (int i = 0; i < senderList->NumElements(); i++) {
		Participant *sender = senderList->Nth(i);
		for (int j = 0; j < receiverList->NumElements(); j++) {
			Participant *receiver = receiverList->Nth(j);
			if (sender->isEqual(receiver)) {
				DataExchange *exchange = new DataExchange(i, j);
				dataExchangeList->Append(exchange);
				continue;
			}
			List<MultidimensionalIntervalSeq*> *overlap = DataExchange::getCommonRegion(sender, receiver);
			if (overlap != NULL) {
				DataExchange *exchange = new DataExchange(i, j, overlap);
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

List<Participant*> *Confinement::generateParticipantList(List<Container*> *participantBranches,
		CommRole role, vector<LpsDimConfig> pathOrder, DataItemConfig *dataConfig, int pruningLevel) {

	List<Participant*> *participantList = new List<Participant*>;
	List<PartFolding*> *foldingList = new List<PartFolding*>;

	for (unsigned int i = 0; i < participantSegments.size(); i++) {
		int segmentTag = participantSegments.at(i);
		for (int j = 0; j < participantBranches->NumElements(); j++) {
			Container *branch = participantBranches->Nth(j);
			PartFolding *folding = branch->foldContainerForSegment(segmentTag, pathOrder, true);
			folding->pruneFolding(pruningLevel, dataConfig);
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
				vector<int*> *containerId = branch->getPartId(dataDimensions);
				List<MultidimensionalIntervalSeq*> *dataDesc = folding->generateIntervalDesc(dataConfig);
				if (dataDesc != NULL) {
					Participant *participant = new Participant(role, containerId, dataDesc);
					int senderCountSoFar = participantList->NumElements();
					participant->setId(senderCountSoFar);
					participant->addSegmentTag(segmentTag);
					participantList->Append(participant);
				}
			} else {
				Participant *participant = participantList->Nth(matchingIndex);
				participant->addSegmentTag(segmentTag);
				delete folding;
			}
		}
	}

	while (foldingList->NumElements() > 0) {
		PartFolding *folding = foldingList->Nth(0);
		foldingList->RemoveAt(0);
		delete folding;
	}

	return participantList;
}
