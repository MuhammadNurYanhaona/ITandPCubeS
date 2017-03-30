#include "data_dependency.h"
#include "sync_stage_implantation.h"
#include "../syntax/ast_task.h"
#include "../semantics/scope.h"
#include "../semantics/symbol.h"
#include "../semantics/task_space.h"
#include "../semantics/computation_flow.h"
#include "../../../common-libs/utils/list.h"
#include "../../../common-libs/utils/hashtable.h"
#include "../../../common-libs/utils/string_utils.h"

#include <algorithm>
#include <sstream>

//------------------------------------------- Last Modifier Panel -----------------------------------------------/

LastModifierPanel *LastModifierPanel::panel = new LastModifierPanel();

LastModifierPanel::LastModifierPanel() {
	stageList = new Hashtable<FlowStage*>;
}

FlowStage *LastModifierPanel::getLastModifierOfVar(const char *varName) {
	return stageList->Lookup(varName);
}

void LastModifierPanel::setLastModifierOfVar(FlowStage *stage, const char *varName) {
	stageList->Enter(varName, stage, true);
}

//--------------------------------------------- Dependency Arc --------------------------------------------------/

DependencyArc::DependencyArc(FlowStage *source, FlowStage *destination, const char *varName) {
	
	this->source = source;
        this->destination = destination;
        this->varName = varName;
	
	signalSrc = source;
	signalSink = destination;
	reactivator = false;
	active = true;
	signaled = false;
	signalingNotRequired = false;
	signalingReplacement = NULL;
	nestingIndex = -1;
	arcName = NULL;
	DataDependencies *sourceDependencies = source->getDataDependencies();
	sourceDependencies->addOutgoingArcIfNotExists(this);
}

int DependencyArc::getNestingIndex() {
	if (nestingIndex == -1) {
		int sourceNesting = signalSrc->getRepeatIndex();
		int destinationNesting = signalSink->getRepeatIndex();
		nestingIndex = std::max(sourceNesting, destinationNesting);
	}
	return nestingIndex;
}

void DependencyArc::print(int indent, bool displaySource, bool displayDestination) {
	for (int i = 0; i < indent; i++) printf("\t");
	printf("move %s", varName);
	if (displaySource) printf(" from %s", source->getName());
	if (displayDestination) printf(" to %s", destination->getName());
	printf(" comm-root %s", communicationRoot->getName());
	if (syncRoot != NULL) printf(" sync-root %s", syncRoot->getName());
	printf("\n");
}

void DependencyArc::deriveSyncAndCommunicationRoots(PartitionHierarchy *hierarchy) {
	
	Space *sourceSpace = source->getSpace();
	Space *destinationSpace = destination->getSpace();
	communicationRoot = hierarchy->getCommonAncestor(sourceSpace, destinationSpace);
	while (communicationRoot->getLocalStructure(varName) == NULL) {
		communicationRoot = communicationRoot->getParent();
	}
	
	syncRoot = NULL;
	if ((dynamic_cast<SyncStage*>(source) != NULL) 
		&& (sourceSpace == destinationSpace 
			|| destinationSpace->isParentSpace(sourceSpace))) {
		return;
	}
	DataStructure *structure = sourceSpace->getStructure(varName);
	if (dynamic_cast<ArrayDataStructure*>(structure) == NULL) {
		syncRoot = hierarchy->getRootSpace();
	} else {
		Space *syncRootCandidate = communicationRoot;
		do {
			if (syncRootCandidate->isReplicatedInCurrentSpace(varName)) {
				syncRoot = syncRootCandidate;	
			}
		} while ((syncRootCandidate = syncRootCandidate->getParent()) != NULL);

		if (syncRoot != NULL) {
			syncRoot = syncRoot->getParent();
			while (syncRoot->getLocalStructure(varName) == NULL) {
				syncRoot = syncRoot->getParent();	
			}
		}
	}
}

const char *DependencyArc::getArcName() {
	if (arcName == NULL) {
		std::ostringstream nameStr;
		nameStr << varName << "Stage";
		nameStr << source->getIndex() << "No";
		nameStr << arcId;
		arcName = strdup(nameStr.str().c_str());
	}
	return arcName;
}

//------------------------------------------- Data Dependencies -------------------------------------------------/

DataDependencies::DataDependencies() {
	incomingArcs = new List<DependencyArc*>;
	outgoingArcs = new List<DependencyArc*>;
}

void DataDependencies::addIncomingArcIfNotExists(DependencyArc *arc) {
	FlowStage *source = arc->getSource();
	const char *data = arc->getVarName();
	for (int i = 0; i < incomingArcs->NumElements(); i++) {
		DependencyArc *arc = incomingArcs->Nth(i);
		if (arc->getSource() != source) continue;
		if (strcmp(arc->getVarName(), data) != 0) continue;
		return; 
	}
	incomingArcs->Append(arc); 
}

void DataDependencies::addOutgoingArcIfNotExists(DependencyArc *arc) {
	FlowStage *destination = arc->getDestination();
	const char *data = arc->getVarName();
	for (int i = 0; i < outgoingArcs->NumElements(); i++) {
		DependencyArc *arc = outgoingArcs->Nth(i);
		if (arc->getDestination() != destination) continue;
		if (strcmp(arc->getVarName(), data) != 0) continue;
		return; 
	}
	outgoingArcs->Append(arc); 
	arc->setArcId(outgoingArcs->NumElements());
}

List<DependencyArc*> *DataDependencies::getActiveDependencies() {
	List<DependencyArc*> *activeList = new List<DependencyArc*>;
	for (int i = 0; i < incomingArcs->NumElements(); i++) {
		DependencyArc *arc = incomingArcs->Nth(i);
		if (arc->isActive()) activeList->Append(arc);
	}
	return activeList;
}

void DataDependencies::print(int indent) {
	if (outgoingArcs->NumElements() == 0) return;
	for (int i = 0; i < indent; i++) printf("\t");
	printf("Data dependency arcs:\n");
	for (int i = 0; i < outgoingArcs->NumElements(); i++) {
		outgoingArcs->Nth(i)->print(indent + 1, false, true);
	}
}
