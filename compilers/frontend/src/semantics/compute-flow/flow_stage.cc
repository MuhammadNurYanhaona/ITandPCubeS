#include "../computation_flow.h"
#include "../scope.h"
#include "../symbol.h"
#include "../task_space.h"
#include "../data_access.h"
#include "../../common/errors.h"
#include "../../common/location.h"
#include "../../syntax/ast.h"
#include "../../syntax/ast_expr.h"
#include "../../syntax/ast_stmt.h"
#include "../../syntax/ast_task.h"
#include "../../static-analysis/usage_statistic.h"
#include "../../static-analysis/task_env_stat.h"
#include "../../static-analysis/reduction_info.h"
#include "../../static-analysis/data_dependency.h"
#include "../../static-analysis/sync_stage_implantation.h"
#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/hashtable.h"

#include <iostream>
#include <fstream>
#include <sstream>

//---------------------------------------------------------- Flow Stage ---------------------------------------------------------/

FlowStage *FlowStage::CurrentFlowStage = NULL;

FlowStage::FlowStage(Space *space) {
        this->space = space;
        this->parent = NULL;
	this->location = NULL;
	this->accessMap = new Hashtable<VariableAccess*>;
	this->epochDependentVarList = new List<const char*>;
	this->dataDependencies = new DataDependencies();
}

Hashtable<VariableAccess*> *FlowStage::getAccessMap() { return accessMap; }

Hashtable<VariableAccess*> *FlowStage::validateDataAccess(Scope *taskScope, Expr *activationCond, Stmt *code) {
	
	//------------------------------------------------------------------------first gather the access information

	TaskGlobalReferences *references = new TaskGlobalReferences(taskScope);
	Hashtable<VariableAccess*> *accessMap = new Hashtable<VariableAccess*>;
	if (activationCond != NULL) { 
		accessMap = activationCond->getAccessedGlobalVariables(references);
	}
	if (code != NULL) {
		Hashtable<VariableAccess*> *codeMap = code->getAccessedGlobalVariables(references);
		Stmt::mergeAccessedVariables(accessMap, codeMap);
	}

	//------------------------------------------------------------------------then validate the variable accesses
	
	// partitionable data structures (currently only arrays are partitionable) must be explicitly
        // mentioned in the partition section correspond to the LPS within which a computation stage executes. 
	// Scalar or non partitionable structures, on the other hand, are not specified explicitly. They get 
	// added to the LPS as they have been found to be used within an LPS's flow stages.
        Iterator<VariableAccess*> iter = accessMap->GetIterator();
        VariableAccess *accessLog;
        while ((accessLog = iter.GetNextValue()) != NULL) {
                const char *name = accessLog->getName();
                if (space->isInSpace(name)) continue;
                VariableSymbol *symbol = (VariableSymbol*) taskScope->lookup(name);
                Type *type = symbol->getType();
                ArrayType *arrayType = dynamic_cast<ArrayType*>(type);
                StaticArrayType *staticArray = dynamic_cast<StaticArrayType*>(type);
                if (arrayType != NULL && staticArray == NULL) {
                        ReportError::ArrayPartitionUnknown(location, name, space->getName());
                } else {
                        DataStructure *source = space->getStructure(name);
                        DataStructure *structure = new DataStructure(source);
                        space->addDataStructure(structure);
                }
        }

	return accessMap;
}

Space *FlowStage::getCommonIntermediateLps(FlowStage *container, FlowStage *contained) {
	
	Space *ancestorLps = container->getSpace();
        Space *descendentLps = contained->getSpace();
	if (ancestorLps == descendentLps 
			|| descendentLps->getParent() == ancestorLps) {
		return NULL;
	}

	Space *nextLevelLps = NULL;
	Space *currentLps = descendentLps->getParent();
	while (currentLps != ancestorLps) {
		nextLevelLps = currentLps;
		currentLps = currentLps->getParent();
	}
	return nextLevelLps;
}

int FlowStage::assignIndexAndGroupNo(int currentIndex, int currentGroupNo, int currentRepeatCycle) {
        this->index = currentIndex;
        this->groupNo = currentGroupNo;
        this->repeatIndex = currentRepeatCycle;
        return currentIndex + 1;
}

const char *FlowStage::getName() {
	std::ostringstream name;
	name << "Flow Stage: [" << index;
	name << ", " << groupNo << ", " << repeatIndex<< "]";
	return strdup(name.str().c_str());
}

void FlowStage::implantSyncStagesInFlow(CompositeStage *containerStage, List<FlowStage*> *currStageList) {
	
	// before re-inserting the current flow stage in the container stage that was originally holding it, 
	// check if execution of earlier flow stages within the container necessitates some sync stage implant
	// before the current stage
	containerStage->addSyncStagesBeforeExecution(this, currStageList);
	
	// re-insert the current flow stage in the container 
	containerStage->addStageAtEnd(this);

	// add the current stage in the stage list to keep the reconstruction process rolling
	currStageList->Append(this);
}

Hashtable<VariableAccess*> *FlowStage::getAccessLogsForSpaceInIndexLimit(Space *space,
                        List<FlowStage*> *stageList, 
			int startIndex, 
			int endIndex,
                        bool includeMentionedSpace) {

        Hashtable<VariableAccess*> *accessLogs = new Hashtable<VariableAccess*>;
        for (int i = startIndex; i <= endIndex; i++) {
                FlowStage *stage = stageList->Nth(i);
		stage->populateAccessMapForSpaceLimit(accessLogs, space, includeMentionedSpace);	
        }
        return accessLogs;
}

Hashtable<VariableAccess*> *FlowStage::getAccessLogsForReturnToSpace(Space *space, 
		List<FlowStage*> *stageList, 
		int endIndex) {
        
	Hashtable<VariableAccess*> *accessLogs = new Hashtable<VariableAccess*>;
        for (int i = endIndex; i >= 0; i--) {
                FlowStage *stage = stageList->Nth(i);
                Space *stageSpace = stage->getSpace();
                if (!stageSpace->isParentSpace(space)) break;
		stage->populateAccessMapForSpaceLimit(accessLogs, space, false);
        }
        return accessLogs;
}

void FlowStage::calculateLPSUsageStatistics() {
        
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



bool FlowStage::isLpsDependent() {
	VariableAccess *accessLog;
        Iterator<VariableAccess*> iterator = accessMap->GetIterator();
        while ((accessLog = iterator.GetNextValue()) != NULL) {
                if (!(accessLog->isContentAccessed()
                        || (accessLog->isMetadataAccessed()
                                && accessLog->isLocalAccess()))) continue;
                const char *varName = accessLog->getName();
                DataStructure *structure = space->getStructure(varName);
                ArrayDataStructure *array = dynamic_cast<ArrayDataStructure*>(structure);
                if (array != NULL) return true;
        }
        return false;
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

void FlowStage::prepareTaskEnvStat(TaskEnvStat *taskStat) {

        Iterator<VariableAccess*> iterator = accessMap->GetIterator();
        VariableAccess *accessLog;
        while ((accessLog = iterator.GetNextValue()) != NULL) {

                // if only the metadata of the object has been accessed then there is no need to track it
                if (!(accessLog->isRead() || accessLog->isModified())) continue;

                const char *varName = accessLog->getName();
                EnvVarStat *varStat = taskStat->getVariableStat(varName);

                // if there is no stat object for the variable access in the environment then the variable is not an 
		// environmental data structure
                if (varStat == NULL) continue;

                if (accessLog->isModified()) {
                        varStat->flagWriteOnLps(space);
                } else {
                        varStat->flagReadOnLps(space);
                }
        }
}

DataDependencies *FlowStage::getDataDependencies() { return dataDependencies; }

bool FlowStage::isDataModifierRelevant(FlowStage *modifier) {
        SyncStage *syncStage = dynamic_cast<SyncStage*>(modifier);
        if (syncStage == NULL || !syncStage->isLoaderSync()) return true;
        return (this->space == syncStage->space || this->space->isParentSpace(syncStage->space));
}

void FlowStage::performDependencyAnalysis(PartitionHierarchy *hierarchy) {

        LastModifierPanel *modifierPanel = LastModifierPanel::getPanel();
        Iterator<VariableAccess*> iter = accessMap->GetIterator();
        VariableAccess *accessLog;

        while ((accessLog = iter.GetNextValue()) != NULL) {
                if (accessLog->isRead()) {
                        FlowStage *modifier = modifierPanel->getLastModifierOfVar(accessLog->getName());
                        if (modifier != NULL && modifier != this && isDataModifierRelevant(modifier)) {
                                DependencyArc *arc = new DependencyArc(modifier, this, accessLog->getName());
                                arc->deriveSyncAndCommunicationRoots(hierarchy);

                                // Note that setting up outgoing arc on the source of the dependency happens inside
                                // the constructor of the DepedencyArc class. So here we consider the destination of 
				// the arc only.  
                                dataDependencies->addIncomingArcIfNotExists(arc);
                        }
                }
                if (accessLog->isModified()) {
                        modifierPanel->setLastModifierOfVar(this, accessLog->getName());
                }
        }
}
