#include "../utils/hashtable.h"
#include "../utils/list.h"
#include "../syntax/ast_task.h"
#include "../semantics/scope.h"
#include "../semantics/symbol.h"
#include "data_access.h"
#include "../semantics/task_space.h"
#include "../syntax/ast_task.h"

//---------------------------------------------- Access Flags ---------------------------------------------------/

void AccessFlags::mergeFlags(AccessFlags *other) {
	if (other == NULL) return;
	if (other->read) this->read = true;
	if (other->write) this->write = true;
	if (other->reduce) this->reduce = true;
	if (other->redirect) this->redirect = true;
}

void AccessFlags::printFlags() {
	if (read) printf("-R-");
	if (write) printf("-W-");
	if (reduce) printf("-A-");
	if (redirect) printf("-C-");
}

//-------------------------------------------- Variable Access --------------------------------------------------/

VariableAccess::VariableAccess(const char *varName) {
	this->varName = varName;
	contentAccess = false;
	metadataAccess = false;
	contentAccessFlags = NULL;
	metadataAccessFlags = NULL;
}

void VariableAccess::markContentAccess() {
	if (contentAccess) return;	
	contentAccess = true;
	contentAccessFlags = new AccessFlags;
}

void VariableAccess::markMetadataAccess() {
	if (metadataAccess) return;	
	metadataAccess = true;
	metadataAccessFlags = new AccessFlags;
}

void VariableAccess::mergeAccessInfo(VariableAccess *other) {
	if (other->contentAccess && !this->contentAccess) {
		markContentAccess();
	}
	if (this->contentAccess) {
		contentAccessFlags->mergeFlags(other->contentAccessFlags);
	} 

	if (other->metadataAccess && !this->metadataAccess) {
		markMetadataAccess();
	}
	if (this->metadataAccess) {
		metadataAccessFlags->mergeFlags(other->metadataAccessFlags);
	} 
}

void VariableAccess::printAccessDetail(int indent) {
	for (int i = 0; i < indent; i++) printf("\t");
	printf("%s:", varName);
	if (contentAccess) {
		printf(" content-");
		contentAccessFlags->printFlags();
	}
	if (metadataAccess) {
		printf(" metadata-");
		metadataAccessFlags->printFlags();
	}
	printf("\n");
}

//-----------------------------------------Task Global References -----------------------------------------------/

TaskGlobalReferences::TaskGlobalReferences(Scope *taskGlobalScope) {
	this->globalScope = taskGlobalScope;
	referencesToGlobals = new Hashtable<const char*>;
}

void TaskGlobalReferences::setNewReference(const char *localVarName, const char *globalVarName) {
	Assert(globalScope->lookup(globalVarName) != NULL);
	referencesToGlobals->Enter(localVarName, globalVarName, true);
}

bool TaskGlobalReferences::doesReferToGlobal(const char *localVarName) {
	if (globalScope->lookup(localVarName) != NULL) return true;
	return (referencesToGlobals->Lookup(localVarName) != NULL);
}

bool TaskGlobalReferences::isGlobalVariable(const char *name) {
	return globalScope->lookup(name) != NULL;
}

VariableSymbol *TaskGlobalReferences::getGlobalRoot(const char *localVarName) {
	if (globalScope->lookup(localVarName) != NULL) { 
		return (VariableSymbol*) globalScope->lookup(localVarName);
	}
	if (referencesToGlobals->Lookup(localVarName) == NULL) return NULL;
	const char *globalVarName = referencesToGlobals->Lookup(localVarName);
	return (VariableSymbol*) globalScope->lookup(globalVarName);
}

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
	active = true;
	DataDependencies *sourceDependencies = source->getDataDependencies();
	sourceDependencies->addOutgoingArcIfNotExists(this);
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
	}
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
