#include "sync_stat.h"
#include "data_flow.h"
#include "data_access.h"
#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "../utils/binary_search.h"
#include "../utils/string_utils.h"
#include "../semantics/task_space.h"

#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>

//---------------------------------------------------- Communication Characteristics -------------------------------------------------------/

CommunicationCharacteristics::CommunicationCharacteristics(const char *varName) {
	this->varName = varName;
	communicationRequired = false;
	confinementSpace = NULL;
	senderDataAllocatorSpace = NULL;
	senderSyncSpace = NULL;
	receiverDataAllocatorSpace = NULL;
	receiverSyncSpace = NULL;
	syncRequirement = NULL;
	waitSignalMayDifferFromSyncLpses = false;
	signalerSpace = NULL;
	waitingSpace = NULL;
}

void CommunicationCharacteristics::setSyncRequirement(SyncRequirement *syncRequirement) {
	this->syncRequirement = syncRequirement;
}

SyncRequirement *CommunicationCharacteristics::getSyncRequirement() { 
	return syncRequirement; 
}

bool CommunicationCharacteristics::shouldAllocateGroupResources() {

	// For ghost sync and cross-propagation sync, each segment may have a different group of other segments
	// that it will be communicating with. So it is not possible to divide the segments into smaller groups
	// where communications are restricted to happen within individual groups. The remaining dependency types 
	// may be benefited from having a group communication resource set.
	GhostRegionSync *ghostSync = dynamic_cast<GhostRegionSync*>(syncRequirement);
	CrossPropagationSync *crossSync = dynamic_cast<CrossPropagationSync*>(syncRequirement);
	return (ghostSync == NULL && crossSync == NULL); 	
}

//----------------------------------------------------------- Sync Requirement -------------------------------------------------------------/

SyncRequirement::SyncRequirement(const char *syncTypeName) {
	this->syncTypeName = syncTypeName;
	this->variableName = NULL;
	this->dependentLps = NULL;
	this->waitingComputation = NULL;
	this->arc = NULL;
	this->index = -1;
	counterRequirement = true;
	replacementSync = NULL;
}

void SyncRequirement::print(int indent) {
	std::ostringstream indentStr;
	for (int i = 0; i < indent; i++) indentStr << '\t';
	std::cout << indentStr.str() << "Dependency on: Space " << dependentLps->getName() << std::endl;
	std::cout << indentStr.str() << "Waiting Computation: " << waitingComputation->getName() << std::endl; 
}

void SyncRequirement::writeDescriptiveComment(std::ofstream &stream, bool forDependent) {
	stream << "// ";
	if (forDependent) {
		stream << syncTypeName << " dependency on Space " << dependentLps->getName(); 
		stream <<" due to update on \"";
		stream << variableName << "\"";
		stream << '\n'; 
	} else {
		stream << "Need to signal update done on \"";
		stream << variableName << "\" to synchronize \"";
		stream << syncTypeName << "\" dependency on Space ";
		stream << dependentLps->getName(); 
		stream << '\n'; 
	}
}

Space *SyncRequirement::getSyncOwner() {
	if (arc->getSyncRoot() == NULL) return arc->getCommRoot();
	return arc->getSyncRoot();
}

const char *SyncRequirement::getSyncName() {
	std::ostringstream stream;
	stream << arc->getArcName() << syncTypeName;
	return strdup(stream.str().c_str());
}

const char *SyncRequirement::getReverseSyncName() {
	std::ostringstream stream;
	stream << arc->getArcName() << "ReverseSync";
	return strdup(stream.str().c_str());
}

int SyncRequirement::compareTo(SyncRequirement *other) {
	DependencyArc *otherArc = other->getDependencyArc();
	int srcIndex = arc->getSource()->getIndex();
	int otherSrcIndex = otherArc->getSource()->getIndex();
	if (srcIndex < otherSrcIndex) return -1;
	else if (srcIndex > otherSrcIndex) return 1;
	int destIndex = arc->getDestination()->getIndex();
	int otherDestIndex = otherArc->getDestination()->getIndex();
	if (destIndex < otherDestIndex) return -1;
	else if (destIndex > otherDestIndex) return 1;
	int arcId = arc->getArcId();
	int otherArcId = otherArc->getArcId();
	if (arcId < otherArcId) return -1;
	else if (arcId > otherArcId) return 1;
	return 0;
}

CommunicationCharacteristics *SyncRequirement::getCommunicationInfo(int segmentationPPS) {

	CommunicationCharacteristics *commCharacter = new CommunicationCharacteristics(variableName);
	Space *commRoot = arc->getCommRoot();
	Space *syncRoot = arc->getSyncRoot();
	Space *confinementSpace = (syncRoot != NULL) ? syncRoot : commRoot;
	Space *senderSyncSpace = arc->getSource()->getSpace();	
	Space *receiverSyncSpace = arc->getDestination()->getSpace();
	DataStructure *senderStruct = senderSyncSpace->getStructure(variableName);
	DataStructure *receiverStruct = receiverSyncSpace->getStructure(variableName);
	
	// in the general case, to involve communication, either the confinement of the synchronization must be
	// above the PPS level where memory segmentation occurs, or there must be two different allocations for 
	// the data structure in the sender and receiver sides of the synchronization
	bool communicationNeeded = (confinementSpace->getPpsId() > segmentationPPS) 
			|| (senderStruct->getAllocator() != receiverStruct->getAllocator());

	commCharacter->setCommunicationRequired(communicationNeeded);
	commCharacter->setConfinementSpace(confinementSpace);
	commCharacter->setSenderSyncSpace(senderSyncSpace);
	commCharacter->setReceiverSyncSpace(receiverSyncSpace);
	commCharacter->setSenderDataAllocatorSpace(senderStruct->getAllocator());
	commCharacter->setReceiverDataAllocatorSpace(receiverStruct->getAllocator());
	commCharacter->setSyncRequirement(this);

	return commCharacter;	
}

List<SyncRequirement*> *SyncRequirement::sortList(List<SyncRequirement*> *reqList) {
	
	if (reqList == NULL || reqList->NumElements() <= 1) return reqList;
	
	List<SyncRequirement*> *sortedList = new List<SyncRequirement*>;
	sortedList->Append(reqList->Nth(0));
	for (int i = 1; i < reqList->NumElements(); i++) {
		SyncRequirement *sync = reqList->Nth(i);
		bool found = false;
		for (int j = 0; j < sortedList->NumElements(); j++) {
			SyncRequirement *sortSync = sortedList->Nth(j);
			if (sync->compareTo(sortSync) <= 0) {
				found = true;
				sortedList->InsertAt(sync, j);
				break;
			}
		}
		if (!found) sortedList->Append(sync);
	}
	return sortedList;
}

void SyncRequirement::separateCommunicationFromSynchronizations(int segmentedPPS, 
		List<SyncRequirement*> *sourceList,
                List<SyncRequirement*> *commList, List<SyncRequirement*> *syncList) {

	if (sourceList == NULL) return;
	for (int i = 0; i < sourceList->NumElements(); i++) {
		SyncRequirement *req = sourceList->Nth(i);
		CommunicationCharacteristics *commCharacter = req->getCommunicationInfo(segmentedPPS);
		if (commCharacter->isCommunicationRequired()) commList->Append(req);
		else syncList->Append(req);
		delete commCharacter;
	}
}

//----------------------------------------------------------- Replication Sync -------------------------------------------------------------/

void ReplicationSync::print(int indent) {
	for (int i = 0; i < indent; i++) std::cout << '\t';
	std::cout << "Type: " << "replication sync" << std::endl;
	SyncRequirement::print(indent);
}

//---------------------------------------------------------- Ghost Region Sync -------------------------------------------------------------/

GhostRegionSync::GhostRegionSync() : SyncRequirement("GSync") {
	overlappingDirections = NULL;
}

void GhostRegionSync::setOverlappingDirections(List<int> *overlappingDirections) {
	this->overlappingDirections = overlappingDirections;
}

void GhostRegionSync::print(int indent) {
	std::ostringstream indentStr;
	for (int i = 0; i < indent; i++) indentStr << '\t';
	std::cout << indentStr.str() << "Type: " << "ghost region sync" << std::endl;
	std::cout << indentStr.str() << "Overlapping Dimensions:";
	for (int i = 0; i < overlappingDirections->NumElements(); i++) {
		std::cout << " " << overlappingDirections->Nth(i);
	}
	std::cout << std::endl;
	SyncRequirement::print(indent);
}

CommunicationCharacteristics *GhostRegionSync::getCommunicationInfo(int segmentationPPS) {

	CommunicationCharacteristics *commCharacter = new CommunicationCharacteristics(variableName);
	SyncStage *updaterSyncStage = dynamic_cast<SyncStage*>(arc->getSource());
	FlowStage *actualUpdater = updaterSyncStage->getUltimateModifier(variableName);
	Space *syncSpace = updaterSyncStage->getSpace();
	Space *confinementSpace = syncSpace->getParent();
	DataStructure *structure = actualUpdater->getSpace()->getStructure(variableName);
	Space *allocatorSpace = structure->getAllocator(); 

	commCharacter->setCommunicationRequired(true);
	commCharacter->setConfinementSpace(confinementSpace);
	commCharacter->setSenderSyncSpace(syncSpace);
	commCharacter->setReceiverSyncSpace(syncSpace);
	commCharacter->setSenderDataAllocatorSpace(allocatorSpace);
	commCharacter->setReceiverDataAllocatorSpace(allocatorSpace);
	commCharacter->setSyncRequirement(this);
	
	commCharacter->setDifferentWaitSignalerFromSync();
	commCharacter->setSignalerSpace(syncSpace);
	commCharacter->setWaitingSpace(arc->getDestination()->getSpace());

	return commCharacter;	
}

//--------------------------------------------------------- Up Propagation Sync ------------------------------------------------------------/

void UpPropagationSync::print(int indent) {
	for (int i = 0; i < indent; i++) std::cout << '\t';
	std::cout << "Type: " << "up propagation sync" << std::endl;
	SyncRequirement::print(indent);
}

Space *UpPropagationSync::getSyncSpan() {
	return arc->getSource()->getSpace();	
}

//-------------------------------------------------------- Down Propagation Sync -----------------------------------------------------------/

void DownPropagationSync::print(int indent) {
	for (int i = 0; i < indent; i++) std::cout << '\t';
	std::cout << "Type: " << "down propagation sync" << std::endl;
	SyncRequirement::print(indent);
}

//-------------------------------------------------------- Cross Propagation Sync ----------------------------------------------------------/

void CrossPropagationSync::print(int indent) {
	for (int i = 0; i < indent; i++) std::cout << '\t';
	std::cout << "Type: " << "cross propagation sync" << std::endl;
	SyncRequirement::print(indent);
}

//------------------------------------------------------ Variable Sync Requirement ---------------------------------------------------------/

VariableSyncReqs::VariableSyncReqs(const char *varName) {
	this->varName = varName;
	syncList = new List<SyncRequirement*>;	
}

void VariableSyncReqs::addSyncRequirement(SyncRequirement *syncReq) {
	syncList->Append(syncReq);
}

void VariableSyncReqs::deactivateRedundantSyncReqs() {
	
	FlowStage *updater = syncList->Nth(0)->getDependencyArc()->getSource();
	int updaterIndex = updater->getIndex();
	
	std::vector<int> precedingDependentIndexes;
	std::vector<DependencyArc*> precedingDependents;
	std::vector<int> succeedingDependentIndexes;
	std::vector<DependencyArc*> succeedingDependents;

	// first separate the dependencies into two groups: those that goes back from the updater stage and those that goes forward
	// to some future stage
	for (int i = 0; i < syncList->NumElements(); i++) {
		DependencyArc* arc = syncList->Nth(i)->getDependencyArc();
		FlowStage *dependent = arc->getDestination();
		int dependentIndex = dependent->getIndex();

		if (dependentIndex < updaterIndex) {
			int location = binsearch::locatePointOfInsert(precedingDependentIndexes, dependentIndex);
			precedingDependents.insert(precedingDependents.begin() + location, arc);
			precedingDependentIndexes.insert(precedingDependentIndexes.begin() 
					+ location, dependentIndex);
		} else {
			int location = binsearch::locatePointOfInsert(succeedingDependentIndexes, dependentIndex);
			succeedingDependents.insert(succeedingDependents.begin() + location, arc);
			succeedingDependentIndexes.insert(succeedingDependentIndexes.begin() 
					+ location, dependentIndex);
		}
	}

	// for list of successor dependents, if there are two of them that execute in the same LPS, deactivate the second without
	// further investigation
	List<const char*> *consideredLpsList = new List<const char*>;
	std::vector<DependencyArc*> filteredSuccessors;
	for (int i = 0; i < succeedingDependents.size(); i++) {
		DependencyArc *arc = succeedingDependents.at(i);
		FlowStage *dependent = arc->getDestination();
		const char *lpsOfDependent = dependent->getSpace()->getName();
		if (string_utils::contains(consideredLpsList, lpsOfDependent)) {
			arc->deactivate();
		} else {
			consideredLpsList->Append(lpsOfDependent);
			filteredSuccessors.push_back(arc);
		}
	}

	// For list of predecessor dependents, if there are two of them that execute in the same LPS, the second one can be 
	// deactivated if the common container stage of <updater, second-dependent> and that of <updaer, first-dependent> is within
	// the same repeat cycle. Otherwise, the first one needs not be signaled but both dependencies should be active.
	consideredLpsList->clear();
	std::vector<DependencyArc*> candidatePredecessors;
	List<DependencyArc*> *nearestDependentsByLpses = new List<DependencyArc*>;
	for (int i = 0; i < precedingDependents.size(); i++) {
		DependencyArc *arc = precedingDependents.at(i);
		FlowStage *dependent = arc->getDestination();
		Space *lpsOfDependent = dependent->getSpace();
		const char *lpsName = lpsOfDependent->getName();
		if (string_utils::contains(consideredLpsList, lpsName)) {
			DependencyArc *lastArc = NULL;
			int lastArcIndex = 0;
			for (int j = 0; j < nearestDependentsByLpses->NumElements(); j++) {
				DependencyArc *candidate = nearestDependentsByLpses->Nth(j);
				if (candidate->getDestination()->getSpace() == lpsOfDependent) {
					lastArc = candidate;
					lastArcIndex = j;
					break;
				}
			}
			FlowStage *earlierDependent = lastArc->getDestination();
			FlowStage *commonParent1 = updater->getNearestCommonAncestor(dependent);
			FlowStage *commonParent2 = updater->getNearestCommonAncestor(earlierDependent);
			if (commonParent1 == commonParent2 
					|| commonParent1->getRepeatIndex() == commonParent2->getRepeatIndex()) {
				arc->deactivate();
			} else {
				lastArc->signal();
				lastArc->disableSignaling();
				lastArc->setSignalingReplacement(arc);
				nearestDependentsByLpses->RemoveAt(lastArcIndex);
				nearestDependentsByLpses->InsertAt(arc, lastArcIndex);
				candidatePredecessors.push_back(arc);
			}
		} else {
			consideredLpsList->Append(lpsName);
			nearestDependentsByLpses->Append(arc);
			candidatePredecessors.push_back(arc);
		}
	}

	// Finally, a successor dependent can lead to the annulling of a predecessor dependent if the former is in a closer or the
	// same repeat nesting group with the updater than the latter is (again both must execute in the same LPS). Otherwise, the 
	// successor dependency needs not be signaled. 	
	std::vector<DependencyArc*> filteredPredecessors;
	for (int i = 0; i < candidatePredecessors.size(); i++) {
		DependencyArc *arc = candidatePredecessors.at(i);
		FlowStage *dependent = arc->getDestination();
		Space *lpsOfDependent = dependent->getSpace();
		DependencyArc *successorArc = NULL;
		for (int j = 0; j < filteredSuccessors.size(); j++) {
			if (filteredSuccessors.at(j)->getDestination()->getSpace() == lpsOfDependent) {
				successorArc = filteredSuccessors.at(j);
				break;
			}
		}
		if (successorArc != NULL) {
			FlowStage *laterDependent = successorArc->getDestination();
			FlowStage *commonParent1 = updater->getNearestCommonAncestor(dependent);
			FlowStage *commonParent2 = updater->getNearestCommonAncestor(laterDependent);
			if (commonParent1 == commonParent2 
					|| commonParent1->getRepeatIndex() <= commonParent2->getRepeatIndex()) {
				arc->deactivate();
			} else {
				successorArc->signal();
				successorArc->disableSignaling();
				successorArc->setSignalingReplacement(arc);
				filteredPredecessors.push_back(arc);
			}		
		} else {
			filteredPredecessors.push_back(arc);
		}
	}

	// Keep only the active sync requirements in the updated sync list
	List<SyncRequirement*> *filteredList = new List<SyncRequirement*>;
	for (int i = 0; i < syncList->NumElements(); i++) {
		SyncRequirement *syncReq = syncList->Nth(i);
		if (syncReq->isActive()) filteredList->Append(syncReq);
	}
	syncList = filteredList;

	// any sync requirement for which signaling is disabled should get a reference to the sync requirements that does
	// signaling for it
	for (int i = 0; i < syncList->NumElements(); i++) {
		SyncRequirement *syncReq = syncList->Nth(i);
		DependencyArc *arc = syncReq->getDependencyArc();
		if (!arc->doesRequireSignal()) {
			DependencyArc *replacementArc = arc->getSignalingReplacement();
			for (int j = 0; j < syncList->NumElements(); j++) {
				SyncRequirement *otherReq = syncList->Nth(j);
				if (otherReq->getDependencyArc() == replacementArc) {
					syncReq->setReplacementSync(otherReq);
					break;
				}
			}
		}	
	}
}

void VariableSyncReqs::print(int indent) {
	for (int i = 0; i < indent; i++) std::cout << '\t';
	std::cout << "Variable: " << varName << std::endl;
	for (int i = 0; i < syncList->NumElements(); i++) {
		if (i > 0) std::cout << std::endl;
		syncList->Nth(i)->print(indent + 1);
	}	
}

//-------------------------------------------------------- Stage Sync Requirement ----------------------------------------------------------/

StageSyncReqs::StageSyncReqs(FlowStage *computation) {
	this->computation = computation;
	this->updaterLps = computation->getSpace();
	varSyncMap = new Hashtable<VariableSyncReqs*>;
}

void StageSyncReqs::addVariableSyncReq(const char *varName, SyncRequirement *syncReq, bool addDependency) {
	VariableSyncReqs *varSyncReqs = varSyncMap->Lookup(varName);
	if (varSyncReqs == NULL) {
		varSyncReqs = new VariableSyncReqs(varName);
		varSyncMap->Enter(varName, varSyncReqs, true);
	}
	varSyncReqs->addSyncRequirement(syncReq);
	if (addDependency) {
		FlowStage *dependentStage = syncReq->getWaitingComputation();
		dependentStage->getAllSyncDependencies()->addDependency(syncReq);
	}
}

List<VariableSyncReqs*> *StageSyncReqs::getVarSyncList() {
	List<VariableSyncReqs*> *syncList = new List<VariableSyncReqs*>;
	VariableSyncReqs *syncReq;
	Iterator<VariableSyncReqs*> iterator = varSyncMap->GetIterator();
	while ((syncReq = iterator.GetNextValue()) != NULL) {
		syncList->Append(syncReq);
	}
	return syncList;
}

bool StageSyncReqs::isDependentStage(FlowStage *suspectedDependentStage) {
	List<VariableSyncReqs*> *varSyncList = getVarSyncList();
	for (int i = 0; i < varSyncList->NumElements(); i++) {
		VariableSyncReqs *varSync = varSyncList->Nth(i);
		List<SyncRequirement*> *syncList = varSync->getSyncList();
		for (int j = 0; j < syncList->NumElements(); j++) {
			SyncRequirement *syncReq = syncList->Nth(j);
			DependencyArc *arc = syncReq->getDependencyArc();
			if (suspectedDependentStage == syncReq->getWaitingComputation()
					|| suspectedDependentStage == arc->getSignalSink()) {
				return true;
			}	
		}	
	}
	return false;
}

void StageSyncReqs::removeRedundencies() {
	List<VariableSyncReqs*> *varSyncList = getVarSyncList();
	for (int i = 0; i < varSyncList->NumElements(); i++) {
		varSyncList->Nth(i)->deactivateRedundantSyncReqs();
	}
}

void StageSyncReqs::print(int indent) {
	List<VariableSyncReqs*> *varSyncList = getVarSyncList();
	if (varSyncList->NumElements() > 0) {
		std::ostringstream indentStr;
		for (int i = 0; i < indent; i++) indentStr << '\t';
		std::cout << indentStr.str() << "Computation: " << computation->getName() << std::endl;
		std::cout << indentStr.str() << "Update on: Space " << updaterLps->getName() << std::endl;
		for (int i = 0; i < varSyncList->NumElements(); i++) {
			if (i > 0) std::cout << std::endl;
			varSyncList->Nth(i)->print(indent + 1);
		}
	}
}

List<SyncRequirement*> *StageSyncReqs::getAllSyncRequirements() {
	
	List<SyncRequirement*> *finalSyncList = new List<SyncRequirement*>;
	List<VariableSyncReqs*> *varSyncList = getVarSyncList();
	
	for (int i = 0; i < varSyncList->NumElements(); i++) {
		VariableSyncReqs *varSyncs = varSyncList->Nth(i);
		List<SyncRequirement*> *syncList = varSyncs->getSyncList();
		finalSyncList->AppendAll(syncList);
	}
	return finalSyncList;
}

List<SyncRequirement*> *StageSyncReqs::getAllNonSignaledSyncReqs() {
	
	List<SyncRequirement*> *activeSyncList = new List<SyncRequirement*>;
	List<VariableSyncReqs*> *varSyncList = getVarSyncList();
	
	for (int i = 0; i < varSyncList->NumElements(); i++) {
		VariableSyncReqs *varSyncs = varSyncList->Nth(i);
		List<SyncRequirement*> *syncList = varSyncs->getSyncList();
		for (int j = 0; j < syncList->NumElements(); j++) {
			SyncRequirement *sync = syncList->Nth(j);
			if (!sync->getDependencyArc()->isSignaled()) {
				activeSyncList->Append(sync);
			}	
		} 
	}
	return activeSyncList;
}

//------------------------------------------------------- Stage Sync Dependencies ----------------------------------------------------------/

StageSyncDependencies::StageSyncDependencies(FlowStage *computation) {
	this->computation = computation;
	this->dependentLps = computation->getSpace();
	syncList = new List<SyncRequirement*>;
}

void StageSyncDependencies::addDependency(SyncRequirement *syncReq) {
	syncList->Append(syncReq);
}
        
void StageSyncDependencies::addAllDependencies(List<SyncRequirement*> *syncReqList) {
	syncList->AppendAll(syncReqList);
}
        
List<SyncRequirement*> *StageSyncDependencies::getActiveDependencies() {
	List<SyncRequirement*> *activeDependencies = new List<SyncRequirement*>;
	for (int i = 0; i < syncList->NumElements(); i++) {
		SyncRequirement *currentDependency = syncList->Nth(i);
		if (currentDependency->isActive()) activeDependencies->Append(currentDependency);	
	}
	return activeDependencies;
}


