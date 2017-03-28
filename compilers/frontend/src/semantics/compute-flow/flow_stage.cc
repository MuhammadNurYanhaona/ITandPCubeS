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
#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/hashtable.h"

#include <iostream>
#include <fstream>

//---------------------------------------------------------- Flow Stage ---------------------------------------------------------/

FlowStage::FlowStage(Space *space) {
        this->space = space;
        this->parent = NULL;
	this->accessMap = new Hashtable<VariableAccess*>;
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

