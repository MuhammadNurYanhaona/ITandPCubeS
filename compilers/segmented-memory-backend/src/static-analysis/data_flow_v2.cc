#include "data_flow.h"
#include "data_access.h"
#include "sync_stat.h"
#include "../syntax/ast.h"
#include "../syntax/ast_expr.h"
#include "../syntax/ast_stmt.h"
#include "../syntax/ast_task.h"
#include "../syntax/ast_type.h"
#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "../utils/string_utils.h"
#include "../utils/code_constant.h"
#include "../semantics/task_space.h"
#include "../semantics/scope.h"
#include "../semantics/symbol.h"
#include "../codegen/name_transformer.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

//-------------------------------------------------- Flow Stage ----------------------------------------------------------/

FlowStage *FlowStage::CurrentFlowStage = NULL;

FlowStage::FlowStage(int index, Space *space, Expr *executeCond) {
	this->index = index;
	this->space = space;
	this->executeCond = executeCond;
	this->accessMap = new Hashtable<VariableAccess*>;
	this->dataDependencies = new DataDependencies();
	this->name = NULL;
	this->synchronizationReqs = NULL;
	syncDependencies = new StageSyncDependencies(this);
	this->parent = NULL;
	this->epochDependentVarList = new List<const char*>;
}

void FlowStage::mergeAccessMapTo(Hashtable<VariableAccess*> *destinationMap) {
	Stmt::mergeAccessedVariables(destinationMap, accessMap);
}

void FlowStage::addAccessInfo(VariableAccess *accessLog) {
	const char *varName = accessLog->getName();
	if (accessMap->Lookup(varName) != NULL) accessMap->Lookup(varName)->mergeAccessInfo(accessLog);
	else accessMap->Enter(varName, accessLog, true);
}

Hashtable<VariableAccess*> *FlowStage::getAccessLogsForSpaceInIndexLimit(Space *space,
                        List<FlowStage*> *stageList, int startIndex, int endIndex,
                        bool includeMentionedSpace) {
	Hashtable<VariableAccess*> *accessLogs = new Hashtable<VariableAccess*>;
	for (int i = startIndex; i <= endIndex; i++) {
		FlowStage *stage = stageList->Nth(i);
		Space *stageSpace = stage->getSpace();
		if (stageSpace->isParentSpace(space)) {
			Stmt::mergeAccessedVariables(accessLogs, stage->getAccessMap());	
		} else if (stageSpace == space && includeMentionedSpace) {
			Stmt::mergeAccessedVariables(accessLogs, stage->getAccessMap());	
		}
	}
	return accessLogs;
}

Hashtable<VariableAccess*> *FlowStage::getAccessLogsForReturnToSpace(Space *space,
                        List<FlowStage*> *stageList, int endIndex) {
	Hashtable<VariableAccess*> *accessLogs = new Hashtable<VariableAccess*>;
	for (int i = endIndex; i >= 0; i--) {
		FlowStage *stage = stageList->Nth(i);
		Space *stageSpace = stage->getSpace();
		if (!stageSpace->isParentSpace(space)) break;
		Stmt::mergeAccessedVariables(accessLogs, stage->getAccessMap());	
	}
	return accessLogs;	
}

void FlowStage::print(int indent) {
	for (int i = 0; i < indent; i++) printf("\t");
	printf("Flow Stage: %s (Space: %s)", name, space->getName());
	if (executeCond != NULL) printf(" Conditionally Executed");
	printf("\n");
	Iterator<VariableAccess*> iter = accessMap->GetIterator();
        VariableAccess* accessLog;
        while ((accessLog = iter.GetNextValue()) != NULL) {
        	accessLog->printAccessDetail(indent + 1);
        }
	dataDependencies->print(indent + 1);
}

void FlowStage::performDependencyAnalysis(Hashtable<VariableAccess*> *accessLogs, PartitionHierarchy *hierarchy) {
	
	LastModifierPanel *modifierPanel = LastModifierPanel::getPanel();
        Iterator<VariableAccess*> iter = accessLogs->GetIterator();
        VariableAccess *accessLog;

        while ((accessLog = iter.GetNextValue()) != NULL) {
                if (accessLog->isRead()) {
                        FlowStage *modifier = modifierPanel->getLastModifierOfVar(accessLog->getName());
                        if (modifier != NULL && modifier != this && isDataModifierRelevant(modifier)) {
                                DependencyArc *arc = new DependencyArc(modifier, this, accessLog->getName());
				arc->deriveSyncAndCommunicationRoots(hierarchy);

				// Note that setting up outgoing arc on the source of the dependency happens inside
				// the constructor of the DepedencyArc class. So here we have to consider the 
				// destination of the arc only.  
                                dataDependencies->addIncomingArcIfNotExists(arc);
                        }
                }
                if (accessLog->isModified()) {
                        modifierPanel->setLastModifierOfVar(this, accessLog->getName());
                }
        }
}

bool FlowStage::isDataModifierRelevant(FlowStage *modifier) {
	SyncStage *syncStage = dynamic_cast<SyncStage*>(modifier);
	if (syncStage == NULL || !syncStage->isLoaderSync()) return true;
	return (this->space == syncStage->space || this->space->isParentSpace(syncStage->space));
}

int FlowStage::assignIndexAndGroupNo(int currentIndex, int currentGroupNo, int currentRepeatCycle) {
	this->index = currentIndex;
	this->groupNo = currentGroupNo;
	this->repeatIndex = currentRepeatCycle;
	return currentIndex + 1;
}

FlowStage *FlowStage::getNearestCommonAncestor(FlowStage *other) {
	FlowStage *first = this;
	FlowStage *second = other;
	while (first->index != second->index) {
		if (first->index > second->index) {
			first = first->parent;
		} else if (second->index > first->index) {
			second = second->parent;
		}
	}
	return first;
}

List<const char*> *FlowStage::filterInArraysFromAccessMap(Hashtable<VariableAccess*> *accessMap) {
	if (accessMap == NULL) {
		accessMap = this->accessMap;
	}
	List<const char*> *arrayList = new List<const char*>;
	Iterator<VariableAccess*> iterator = accessMap->GetIterator();
	VariableAccess *accessLog;
	while ((accessLog = iterator.GetNextValue()) != NULL) {
		const char *varName = accessLog->getName();
		DataStructure *structure = space->getLocalStructure(varName);
		if (structure == NULL) continue;
		ArrayDataStructure *array = dynamic_cast<ArrayDataStructure*>(structure);
		if (array != NULL) {
			arrayList->Append(varName);
		}
	}
	return arrayList;
}

void FlowStage::fillInTaskEnvAccessList(List<VariableAccess*> *envAccessList) {
	for (int i = 0; i < envAccessList->NumElements(); i++) {
		VariableAccess *envAccessLog = envAccessList->Nth(i);
		const char *varName = envAccessLog->getName();
		VariableAccess *stageAccessLog = accessMap->Lookup(varName);
		if (stageAccessLog != NULL) {
			envAccessLog->mergeAccessInfo(stageAccessLog);
		}
	}
}

void FlowStage::calculateLPSUsageStatistics() {
	if (space->isDynamic() && executeCond != NULL) {
		List<const char*> *localStructures = space->getLocalDataStructureNames();
		for (int i = 0; i < localStructures->NumElements(); i++) {
			const char *varName = localStructures->Nth(i);
			space->getLocalStructure(varName)->getUsageStat()->resetAccessCount();
		}
	}
	Iterator<VariableAccess*> iterator = accessMap->GetIterator();
	VariableAccess *accessLog;
	while ((accessLog = iterator.GetNextValue()) != NULL) {
		if (!accessLog->isContentAccessed()) continue;
		const char *varName = accessLog->getName();
		DataStructure *structure = space->getLocalStructure(varName);
		AccessFlags *accessFlags = accessLog->getContentAccessFlags();
		LPSVarUsageStat *usageStat = structure->getUsageStat();
		if (accessFlags->isRead() || accessFlags->isWritten()) {
			usageStat->addAccess();
		}
		if (accessFlags->isReduced()) {
			usageStat->flagReduced();
		}
	}
}

void FlowStage::analyzeSynchronizationNeeds() {
	
	List<DependencyArc*> *outgoingArcList = dataDependencies->getOutgoingArcs();
	synchronizationReqs = new StageSyncReqs(this);
	for (int i = 0; i < outgoingArcList->NumElements(); i++) {
		DependencyArc *arc = outgoingArcList->Nth(i);
		const char *varName = arc->getVarName();
		FlowStage *destination = arc->getDestination();
		Space *destLps = destination->getSpace();

		// If the destination and current flow stage's LPSes are the same then two scenarios are there
		// for us to consider: either the variable is replicated or it has overlapping partitions among
		// adjacent LPUs. The former case is handled here. The latter is taken care of by the sync
		// stage's overriding implementation; therefore we ignore it. 
		if (destLps == space) {
			if (space->isReplicatedInCurrentSpace(varName)) {
				ReplicationSync *replication = new ReplicationSync();
				replication->setVariableName(varName);
				replication->setDependentLps(destLps);
				replication->setWaitingComputation(destination);
				replication->setDependencyArc(arc);
				synchronizationReqs->addVariableSyncReq(varName, replication, true);
			}
		// If the destination and current flow stage's LPSes are not the same then there is definitely
		// a synchronization need. The specific type of synchronization needed depends on the relative
		// position of these two LPSes in the partition hierarchy.
		} else {
			SyncRequirement *syncReq = NULL;
			if (space->isParentSpace(destLps)) {
				syncReq = new UpPropagationSync();
			} else if (destLps->isParentSpace(space)) {
				syncReq = new DownPropagationSync();
			} else {
				syncReq = new CrossPropagationSync();
			}
			syncReq->setVariableName(varName);
			syncReq->setDependentLps(destLps);
			syncReq->setWaitingComputation(destination);
			syncReq->setDependencyArc(arc);
			synchronizationReqs->addVariableSyncReq(varName, syncReq, true);
		}
	}
	synchronizationReqs->removeRedundencies(); 	
}

StageSyncReqs *FlowStage::getAllSyncRequirements() { return synchronizationReqs; }

StageSyncDependencies *FlowStage::getAllSyncDependencies() { return syncDependencies; }

void FlowStage::printSyncRequirements(int indentLevel) {
	for (int i = 0; i < indentLevel; i++) std::cout << '\t';
	std::cout << "Stage: " << name << "\n"; 
	synchronizationReqs->print(indentLevel + 1); 
}

bool FlowStage::isDependentStage(FlowStage *suspectedDependent) {
	if (synchronizationReqs == NULL) return false;
	return synchronizationReqs->isDependentStage(suspectedDependent); 
}

List<const char*> *FlowStage::getAllOutgoingDependencyNamesAtNestingLevel(int nestingLevel) {
	if (this->repeatIndex != nestingLevel) return NULL;
	List<const char*> *arcNameList = new List<const char*>;
	List<SyncRequirement*> *syncList = synchronizationReqs->getAllNonSignaledSyncReqs();
	for (int i = 0; i < syncList->NumElements(); i++) {
		DependencyArc *arc = syncList->Nth(i)->getDependencyArc();
		if (arc->getNestingIndex() == nestingLevel) {
			arcNameList->Append(arc->getArcName());
		}
	}
	return arcNameList;
}

void FlowStage::setReactivatorFlagsForSyncReqs() {
	
	List<SyncRequirement*> *syncList = synchronizationReqs->getAllSyncRequirements();
	List<SyncRequirement*> *filteredList = new List<SyncRequirement*>;
	if (syncList != NULL && syncList->NumElements() > 0) {
		for (int i = 0; i < syncList->NumElements(); i++) {
			SyncRequirement *sync = syncList->Nth(i);
			DependencyArc *arc = sync->getDependencyArc();
			
			// if this is not the signal source then we can skip this arc as the signal source will
			// take care of the reactivation flag processing
			if (this != arc->getSignalSrc()) continue;

			filteredList->Append(sync);	
		}
	}

	// if the filtered list is not empty then there is a need for reactivation signaling
	if (filteredList->NumElements() > 0) {

		// reactivation signals should be received from PPUs of all LPSes that have computations dependent
    		// on changes made by the current stage under concern. So we need to partition the list into 
		// smaller lists for individual LPSes
		Hashtable<List<SyncRequirement*>*> *syncMap = new Hashtable<List<SyncRequirement*>*>;
		for (int i = 0; i < filteredList->NumElements(); i++) {
			SyncRequirement *sync = filteredList->Nth(i);
			Space *dependentLps = sync->getDependentLps();
			List<SyncRequirement*> *lpsSyncList = syncMap->Lookup(dependentLps->getName());
			if (lpsSyncList == NULL) {
				lpsSyncList = new List<SyncRequirement*>;
			}
			lpsSyncList->Append(sync);
			syncMap->Enter(dependentLps->getName(), lpsSyncList, true);	
		}  
		
		// iterate over the sync list correspond to each LPS and select a reactivating sync for that LPS 
		Iterator<List<SyncRequirement*>*> iterator = syncMap->GetIterator();
		List<SyncRequirement*> *lpsSyncList;
		while ((lpsSyncList = iterator.GetNextValue()) != NULL) {		

			DependencyArc *closestPreArc = NULL;
			DependencyArc *furthestSuccArc = NULL;	
		
			for (int i = 0; i < lpsSyncList->NumElements(); i++) {
				SyncRequirement *sync = lpsSyncList->Nth(i);
				DependencyArc *arc = sync->getDependencyArc();
				
				FlowStage *syncStage = arc->getSignalSink();
				int syncIndex = syncStage->getIndex();
				if (syncIndex < this->index 
						&& (closestPreArc == NULL 
						|| syncIndex > closestPreArc->getSignalSink()->getIndex())) {
					closestPreArc = arc;
				} else if (syncIndex >= this->index 
						&& (furthestSuccArc == NULL
						|| syncIndex > furthestSuccArc->getSignalSink()->getIndex())) {
					furthestSuccArc = arc;
				}
			}

			// The reactivator should be the last stage that will read change made by this stage 
			// before this stage can execute again. That will be the closest predecessor, if exists, 
			// or the furthest successor
			if (closestPreArc != NULL) {
				closestPreArc->setReactivator(true);
			} else if (furthestSuccArc != NULL) {
				furthestSuccArc->setReactivator(true);
			} else {
				std::cout << "This is strange!!! No reactivator is found for stage: ";
				std::cout << this->getName() << std::endl;
				std::exit(EXIT_FAILURE);
			} 
		}
	}
}

List<DependencyArc*> *FlowStage::getAllTaskDependencies() {
	if (dataDependencies == NULL) return new List<DependencyArc*>;
	return dataDependencies->getOutgoingArcs();
}

List<const char*> *FlowStage::getVariablesNeedingCommunication(int segmentedPPS) {

	if (synchronizationReqs == NULL) return NULL;
	List<SyncRequirement*> *syncList = synchronizationReqs->getAllSyncRequirements();
	List<const char*> *varList = new List<const char*>;
	for (int i = 0; i < syncList->NumElements(); i++) {
		SyncRequirement *syncReq = syncList->Nth(i);
		bool commNeeded = syncReq->getCommunicationInfo(segmentedPPS)->isCommunicationRequired();
		const char *varName = syncReq->getVariableName();
		if (commNeeded && !string_utils::contains(varList, varName)) {
			varList->Append(varName);
		}
	}
	return varList;
}

List<CommunicationCharacteristics*> *FlowStage::getCommCharacteristicsForSyncReqs(int segmentedPPS) {
	if (synchronizationReqs == NULL) return NULL;
	List<SyncRequirement*> *syncList = synchronizationReqs->getAllSyncRequirements();
	List<CommunicationCharacteristics*> *commCharList = new List<CommunicationCharacteristics*>;
	for (int i = 0; i < syncList->NumElements(); i++) {
		SyncRequirement *syncReq = syncList->Nth(i);
		commCharList->Append(syncReq->getCommunicationInfo(segmentedPPS));
	}	
	return commCharList;
}

//-------------------------------------------------- Sync Stage ---------------------------------------------------------/

SyncStage::SyncStage(Space *space, SyncMode mode, SyncStageType type) : FlowStage(0, space, NULL) {
	
	this->mode = mode;	
	this->type = type;
	
	if (mode == Load || mode == Load_And_Configure) {
		this->name = "\"Data Loader Sync\"";
	} else if (mode == Ghost_Region_Update) {
		this->name = "\"Ghost Region Updater Sync\"";
	} else {
		this->name = "\"Data Restorer Sync\"";
	}
	
	this->prevDataModifiers = new Hashtable<FlowStage*>;
}

int SyncStage::populateAccessMap(List<VariableAccess*> *accessLogs, 
		bool filterOutNonReads, bool filterOutNonWritten) {
	int count = 0;
	for (int i = 0; i < accessLogs->NumElements(); i++) {
                VariableAccess *accessLog = accessLogs->Nth(i);
                if (accessLog->isContentAccessed()) {
			if (filterOutNonReads 
				&& !(accessLog->getContentAccessFlags()->isRead()
					|| accessLog->getContentAccessFlags()->isReduced())) continue;
			if (filterOutNonWritten && !accessLog->getContentAccessFlags()->isWritten()) continue;
			
			DataStructure *structure = space->getLocalStructure(accessLog->getName());
			ArrayDataStructure *arrayStruct = dynamic_cast<ArrayDataStructure*>(structure);
			bool scalar = (arrayStruct == NULL);
			
                        VariableAccess *syncLog = new VariableAccess(accessLog->getName());
                        syncLog->markContentAccess();
                        if (accessLog->getContentAccessFlags()->isWritten()) {
				syncLog->getContentAccessFlags()->flagAsWritten();
			} else if (!scalar) syncLog->getContentAccessFlags()->flagAsWritten();

			if (mode == Load_And_Configure) {
				syncLog->markMetadataAccess();
				syncLog->getMetadataAccessFlags()->flagAsWritten();
			}

                        syncLog->getContentAccessFlags()->flagAsRead();
                        addAccessInfo(syncLog);
                        count++;
                }
	}
	return count;
}

void SyncStage::performDependencyAnalysis(PartitionHierarchy *hierarchy) {
	
	LastModifierPanel *modifierPanel = LastModifierPanel::getPanel();
        Iterator<VariableAccess*> iter = accessMap->GetIterator();
        VariableAccess *accessLog;

	// just save a reference to the last modifier of any variable this sync stage is interested in
        while ((accessLog = iter.GetNextValue()) != NULL) {
                if (accessLog->isRead() || accessLog->isModified()) {
                        FlowStage *modifier = modifierPanel->getLastModifierOfVar(accessLog->getName());
                        if (modifier != NULL && modifier != this && isDataModifierRelevant(modifier)) {
				prevDataModifiers->Enter(accessLog->getName(), modifier);
			}
		}
	}
	
	// then call the superclass's dependency analysis function
	FlowStage::performDependencyAnalysis(hierarchy);
}

void SyncStage::analyzeSynchronizationNeeds() {

	// the general logic for setting up synchronization requirements works for all sync stages except the
	// ghost region sync stages
	if (mode != Ghost_Region_Update) {
		FlowStage::analyzeSynchronizationNeeds();
	} else {
		List<DependencyArc*> *outgoingArcList = dataDependencies->getOutgoingArcs();
		synchronizationReqs = new StageSyncReqs(this);
		for (int i = 0; i < outgoingArcList->NumElements(); i++) {
			
			DependencyArc *arc = outgoingArcList->Nth(i);
			const char *varName = arc->getVarName();
			FlowStage *destination = arc->getDestination();

			// if the destination is another ghost region sync stage then we ignore this dependency
			// as the segmented-memory communication model ensures that such dependencies will be
			// resolved automatically
			SyncStage *destSync = dynamic_cast<SyncStage*>(destination);
			if (destSync != NULL && destSync->mode == Ghost_Region_Update) continue;

			Space *destLps = destination->getSpace();
			DataStructure *structure = space->getLocalStructure(varName);
			ArrayDataStructure *array = dynamic_cast<ArrayDataStructure*>(structure);
			if (array == NULL) continue;
			if (!array->hasOverlappingsAmongPartitions()) continue;
			List<int> *overlappingDims = array->getOverlappingPartitionDims();
			GhostRegionSync *ghostRegion = new GhostRegionSync();
			ghostRegion->setVariableName(varName);
			ghostRegion->setDependentLps(destLps);
			ghostRegion->setWaitingComputation(destination);
			ghostRegion->setDependencyArc(arc);
			ghostRegion->setOverlappingDirections(overlappingDims);
			synchronizationReqs->addVariableSyncReq(varName, ghostRegion, true);
		}
	}

	// indicates that the synchronization requirements registered by the stage do not need any execution 
	// counter to be activated
	if (synchronizationReqs != NULL) {
		List<SyncRequirement*> *syncList = synchronizationReqs->getAllSyncRequirements();
		for (int i = 0; i < syncList->NumElements(); i++) {
			syncList->Nth(i)->setCounterRequirement(false);
		}
	}
}

void SyncStage::generateInvocationCode(std::ofstream &stream, int indentation, Space *containerSpace) {
	// for multicore back-ends sync stages do nothing 
}

FlowStage *SyncStage::getUltimateModifier(const char *varName) {
	FlowStage *lastModifier = prevDataModifiers->Lookup(varName);
	if (lastModifier == NULL) return NULL;
	SyncStage *previousSync = dynamic_cast<SyncStage*>(lastModifier);
	if (previousSync == NULL) return lastModifier;
	return previousSync->getUltimateModifier(varName);
}

//------------------------------------------------ Execution Stage -------------------------------------------------------/

ExecutionStage::ExecutionStage(int index, Space *space, Expr *executeCond) 
		: FlowStage(index, space, executeCond) {
	this->code = NULL;
	this->scope = NULL;
}

void ExecutionStage::setCode(List<Stmt*> *stmtList) {
	this->code = new StmtBlock(stmtList);
}

void ExecutionStage::performEpochUsageAnalysis() {
	FlowStage::CurrentFlowStage = this;
	code->analyseEpochDependencies(space);
}

void ExecutionStage::print(int indent) {
	FlowStage::print(indent);
	if (epochDependentVarList->NumElements() != 0) {
		for (int i = 0; i <= indent; i++) {
			std::cout << '\t';
		}
		std::cout << "Epoch Dependent Variables: ";
		for (int i = 0; i < epochDependentVarList->NumElements(); i++) {
			if (i != 0) std::cout << ", ";
			std::cout << epochDependentVarList->Nth(i);
		}
		std::cout << "\n";
	}
}

void ExecutionStage::translateCode(std::ofstream &stream) {

	std::string activateHd = 	"\n\t//---------------------- Activating Condition -------------------------------\n\n";
	std::string localMdHd =  	"\n\t//-------------------- Local Copies of Metadata -----------------------------\n\n";
	std::string localVarDclHd = 	"\n\t//------------------- Local Variable Declarations ---------------------------\n\n";
	std::string computeHd = 	"\n\t//----------------------- Computation Begins --------------------------------\n\n";
	std::string returnHd =  	"\n\t//------------------------- Returning Flag ----------------------------------\n\n";
	
	// reset the name transformer to user common "lpu." prefix for array access in case it is been modified
	ntransform::NameTransformer::transformer->setLpuPrefix("lpu->");

	// create local variables for all array dimensions so that later on name-transformer that add 
	// prefix/suffix to accessed global variables can work properly
	stream <<  localMdHd;
	stream << "\t// create local copies of partition and storage dimension configs of all arrays\n";
	std::string stmtIndent = "\t";
        List<const char*> *localArrays = filterInArraysFromAccessMap();
	for (int i = 0; i < localArrays->NumElements(); i++) {
        	const char *arrayName = localArrays->Nth(i);
        	ArrayDataStructure *array = (ArrayDataStructure*) space->getStructure(arrayName);
        	int dimensions = array->getDimensionality();
        	stream << stmtIndent << "Dimension ";
                stream  << arrayName << "PartDims[" << dimensions << "];\n";
                for (int j = 0; j < dimensions; j++) {
                	stream << stmtIndent;
                	stream << arrayName << "PartDims[" << j << "] = lpu->";
                        stream << arrayName << "PartDims[" << j << "].partition;\n";
                }
                stream << stmtIndent << "Dimension ";
                stream  << arrayName << "StoreDims[" << dimensions << "];\n";
                for (int j = 0; j < dimensions; j++) {
                	stream << stmtIndent;
               		stream << arrayName << "StoreDims[" << j << "] = lpu->";
                        stream << arrayName << "PartDims[" << j << "].storage;\n";
        	}
        }

	// create a local part-dimension object for probable array dimension based range or assignment expressions
	stream << "\n\t// create a local part-dimension object for later use\n";
        stream << "\tPartDimension partConfig;\n";
	
	// create a local integer for holding intermediate values of transformed index during inclusion testing
	stream << "\n\t// create a local transformed index variable for later use\n";
        stream << "\tint xformIndex;\n";
	
	// if the execution block has an activation condition on it then we need to evaluate it before proceeding
	// into executing anything further
	if (executeCond != NULL) {
		stream <<  activateHd;
		stream << "\tif(!(";
		std::ostringstream conditionStream;
		executeCond->translate(conditionStream, 1, 0, space);
		stream << conditionStream.str();
		stream << ")) {\n";
		stream << "\t\treturn FAILURE_RUN;\n";
		stream << "\t}\n";	
	}

	// declare any local variables found in the computation	
	std::ostringstream localVars;
        scope->declareVariables(localVars, 1);
	if (localVars.str().length() > 0) {
		stream <<  localVarDclHd;
		stream << localVars.str();
	}

        // translate statements into C++ code
	stream <<  computeHd;
	std::ostringstream codeStream;
	code->generateCode(codeStream, 1, space);
	stream << codeStream.str();

        // finally return a successfull run indicator
	stream <<  returnHd;
	stream << "\treturn SUCCESS_RUN;\n";	
}

void ExecutionStage::generateInvocationCode(std::ofstream &stream, int indentation, Space *containerSpace) {
	
	// write the indent
	std::ostringstream indentStr;
	for (int i = 0; i < indentation; i++) indentStr << '\t';
	std::ostringstream nextIndent;
	nextIndent << indentStr.str();

	// TODO logic for handling stages with reduction operation need to be investigated further. For now we are 
	// not using the group entry restriction as reductions are converted into sequential traversal of underlying 
	// array(s). Therefore the if block is commented out below.
	//
	// if this is not a group entry execution stage then add a condition ensure that a PPU with only valid ID 
	// corresponding to this stage's LPS should go on executing the code
	// if (!isGroupEntry()) {
		nextIndent << indent;
		stream << indentStr.str() << "if (threadState->isValidPpu(Space_" << space->getName();
		stream << ")) {\n";
	// }
	
	// if the flow-stage's body involves epoch expressions then we have to advance the epoch versions of related
	// data structures   
	if (epochDependentVarList->NumElements() > 0) {
		stream << nextIndent.str() << "{ // advancing epoch versions of relevant data structures\n";

		// retrieve common properties that will be needed to do data structure version updates and later LPU
		// reference refreshing
		const char *lpsName = space->getName();
		stream << nextIndent.str() << "TaskData *taskData = ";
		stream << "threadState->getTaskData()" << stmtSeparator;
		stream << nextIndent.str() << "Hashtable<DataPartitionConfig*> ";
		stream << "*partConfigMap = threadState->getPartConfigMap()" << stmtSeparator;

		for (int i = 0; i < epochDependentVarList->NumElements(); i++) {
			const char *varName = epochDependentVarList->Nth(i);
			DataStructure *structure = space->getStructure(varName);
			ArrayDataStructure *array = dynamic_cast<ArrayDataStructure*>(structure);

			// version count is by default 0; thus we add 1 here
			int versions = structure->getLocalVersionCount() + 1;
			if (array == NULL) {
				// for a scalar variable's version update we need to swap values with older versions
				// of the same variable directly inside the thread-globals object
				stream << nextIndent.str() << indent;
				stream << "{ // epoch advancement for " << varName << "\n";

				// swapping must be done from the rear end; otherwise, intermediate values will get
				// corrupted in the process
				for (int v = versions - 1; v > 0; v--) {
					stream << nextIndent.str() << indent << "taskGlobals->" << varName;
					stream << "_lag_" << v << " = taskGlobals->" << varName;
					if (v > 1) { stream << "_lag_" << (v - 1); }
					stream << stmtSeparator;
				}				

				stream << nextIndent.str() << indent << "}\n";
			} else {
				// for an array, we need to identify the data part Id from the LPU Id then retrieve 
				// the data-part object and rotate its memory-allocation cylinder to do the version 
				// update
				stream << nextIndent.str() << indent;
				stream << "{ // epoch advancement for " << varName << "\n";

				stream << nextIndent.str() << indent << "List<int*> *lpuIdChain = ";
				stream << "threadState->getLpuIdChainWithoutCopy(";
				stream << '\n' << nextIndent.str() << tripleIndent;
				stream << "Space_" << lpsName;
				Space *rootSpace = space->getRoot(); 
				stream << paramSeparator << "Space_" << rootSpace->getName() << ")";
				stream << stmtSeparator;
				stream << nextIndent.str() << indent << "DataPartitionConfig *config = ";
				stream << "partConfigMap->Lookup(\"";
				stream << varName << "Space" << lpsName << "Config\")" << stmtSeparator;
				stream << nextIndent.str() << indent << "PartIterator *iterator = ";
				stream << "threadState->getIterator(Space_" << lpsName << paramSeparator;  
				stream << '\"' << varName << "\")" << stmtSeparator;
				stream << nextIndent.str() << indent << "List<int*> *partId = ";
				stream << "iterator->getPartIdTemplate()" << stmtSeparator;
				stream << nextIndent.str() << indent << "config->generatePartId(lpuIdChain";
				stream << paramSeparator << "partId)" << stmtSeparator;
				stream << nextIndent.str() << indent << "DataItems *items = ";
				stream << "taskData->getDataItemsOfLps(\"" << lpsName; 
				stream << "\"" << paramSeparator << "\"" << varName << "\")" << stmtSeparator;
				stream << nextIndent.str() << indent << "DataPart *dataPart = ";
				stream << "items->getDataPart(partId" << paramSeparator;
				stream << "iterator)" << stmtSeparator;
				stream << nextIndent.str() << indent << "dataPart->advanceEpoch()" << stmtSeparator;
	
				stream << nextIndent.str() << indent << "}\n";
			}	
		}
		// after epoch advancement, the LPU reference needs to be updated so that different data structure 
		// versions are in appropriate properties of the LPU
		stream << nextIndent.str() << "generateSpace" << lpsName << "Lpu(";
		stream << "threadState" << paramSeparator << "partConfigMap" << paramSeparator;
		stream << "taskData)" << stmtSeparator;
		stream << nextIndent.str() << "space" << lpsName << "Lpu = (Space" << lpsName << "_LPU*) ";
		stream << "threadState->getCurrentLpu(Space_" << lpsName << ")" << stmtSeparator;
		
		stream << nextIndent.str() << "}\n";	
	}

	// invoke the related method with current LPU parameter
	stream << nextIndent.str() << "// invoking user computation\n";
	stream << nextIndent.str();
	stream << "int stage" << index << "Executed = ";
	stream << name << "(space" << space->getName() << "Lpu" << paramSeparator;
	// along with other default arguments
	stream << '\n' << nextIndent.str() << doubleIndent << "arrayMetadata" << paramSeparator;
	stream << '\n' << nextIndent.str() << doubleIndent << "taskGlobals" << paramSeparator;
	stream << '\n' << nextIndent.str() << doubleIndent << "threadLocals" << paramSeparator;
	stream << "partition" << paramSeparator << '\n' << nextIndent.str();
	stream << doubleIndent << "threadState->threadLog)" << stmtSeparator;

	// then update all synchronization counters that depend on the execution of this stage for their activation
	List<SyncRequirement*> *syncList = synchronizationReqs->getAllSyncRequirements();
	for (int i = 0; i < syncList->NumElements(); i++) {
		stream << nextIndent.str() << syncList->Nth(i)->getDependencyArc()->getArcName();
		stream << " += stage" << index << "Executed;\n";
	}

	/*----------------------------------------------- Turned off	
	// write a log entry for the stage executed
	stream << nextIndent.str() << "threadState->logExecution(\"";
	stream << name << "\", Space_" << space->getName() << ");\n";
	----------------------------------------------------------*/ 	
	
	// Again the condition below is commented out as we are not using the group-entrance logic properly close the 
	// if condition if applicable
	// if (!isGroupEntry()) {
		stream << indentStr.str() << "}\n";
	// }
}

// An execution stage is a group entry if it contains any reduction operation
bool ExecutionStage::isGroupEntry() {
	Iterator<VariableAccess*> iterator = accessMap->GetIterator();
	VariableAccess *access;
	while ((access = iterator.GetNextValue()) != NULL) {
		if (access->isContentAccessed() && access->getContentAccessFlags()->isReduced()) {
			return true;
		}
	}
	return false;
}

void ExecutionStage::setLpsExecutionFlags() {
	space->flagToExecuteCode();
}
