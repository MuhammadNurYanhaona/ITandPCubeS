#include "sync_stat.h"
#include "data_flow.h"
#include "data_access.h"
#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "../semantics/task_space.h"

#include <iostream>
#include <sstream>
#include <fstream>


//----------------------------------------------------------- Sync Requirement -------------------------------------------------------------/

SyncRequirement::SyncRequirement(const char *syncTypeName) {
	this->syncTypeName = syncTypeName;
	this->variableName = NULL;
	this->dependentLps = NULL;
	this->waitingComputation = NULL;
	this->arc = NULL;
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
	Space *receiverSyncSpace = arc->getSource()->getSpace();
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


