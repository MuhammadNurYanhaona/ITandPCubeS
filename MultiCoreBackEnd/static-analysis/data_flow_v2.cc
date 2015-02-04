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
#include "../semantics/task_space.h"
#include "../semantics/scope.h"
#include "../semantics/symbol.h"
#include "../codegen/name_transformer.h"

#include <iostream>
#include <fstream>
#include <sstream>

//-------------------------------------------------- Flow Stage ----------------------------------------------------------/

FlowStage::FlowStage(int index, Space *space, Expr *executeCond) {
	this->index = index;
	this->space = space;
	this->executeCond = executeCond;
	this->accessMap = new Hashtable<VariableAccess*>;
	this->dataDependencies = new DataDependencies();
	this->name = NULL;
	this->synchronizationReqs = NULL;
	syncDependencies = new StageSyncDependencies(this);
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
		// if the destination and current flow stage's LPSes are the same then two scenarios are there
		// for us to consider: either the variable is replicated or it has overlapping partitions among
		// adjacent LPUs
		if (destLps == space) {
			if (space->isReplicatedInCurrentSpace(varName)) {
				ReplicationSync *replication = new ReplicationSync();
				replication->setVariableName(varName);
				replication->setDependentLps(destLps);
				replication->setWaitingComputation(destination);
				replication->setDependencyArc(arc);
				synchronizationReqs->addVariableSyncReq(varName, replication, true);
			} else {
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
}

StageSyncReqs *FlowStage::getAllSyncRequirements() { return synchronizationReqs; }

StageSyncDependencies *FlowStage::getAllSyncDependencies() { return syncDependencies; }

void FlowStage::printSyncRequirements() { synchronizationReqs->print(0); }

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

void SyncStage::generateInvocationCode(std::ofstream &stream, int indentation, Space *containerSpace) {
	// for multicore back-ends sync stages do nothing 
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
	
	// if the execution block has an activation condition on it then we need to evaluate it before proceeding
	// into executing anything further
	if (executeCond != NULL) {
		stream <<  activateHd;
		stream << "\tif(!(";
		std::ostringstream conditionStream;
		executeCond->translate(conditionStream, 1, 0);
		stream << conditionStream.str();
		stream << ")) {\n";
		stream << "\t\treturn FAILURE_RUN;\n";
		stream << "\t}\n";	
	}

	// declare any local variables found in the computation	
	std::ostringstream localVars;
        Iterator<Symbol*> iterator = scope->get_local_symbols();
       	Symbol *symbol;
	bool symbolFound = false;
        while ((symbol = iterator.GetNextValue()) != NULL) {
                VariableSymbol *variable = dynamic_cast<VariableSymbol*>(symbol);
                if (variable == NULL) continue;
		symbolFound = true;
                Type *type = variable->getType();
                const char *name = variable->getName();
        	localVars << stmtIndent << type->getCppDeclaration(name) << ";\n";
        }
	if (symbolFound) {
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
	std::ostringstream indent;
	for (int i = 0; i < indentation; i++) indent << '\t';
	std::ostringstream nextIndent;
	nextIndent << indent.str();

	// if this is not a group entry execution stage then add a condition ensure that a PPU with only
	// valid ID corresponding to this stage's LPS should go on executing the code
	if (!isGroupEntry()) {
		nextIndent << '\t';
		stream << indent.str() << "if (threadState->isValidPpu(Space_" << space->getName();
		stream << ")) {\n";
	}

	// invoke the related method with current LPU parameter
	stream << nextIndent.str() << "// invoking user computation\n";
	stream << nextIndent.str();
	stream << "int stage" << index << "Executed = ";
	stream << name << "(space" << space->getName() << "Lpu, ";
	// along with other default arguments
	stream << '\n' << nextIndent.str() << "\t\tarrayMetadata,";
	stream << '\n' << nextIndent.str() << "\t\ttaskGlobals,";
	stream << '\n' << nextIndent.str() << "\t\tthreadLocals, partition);\n";

	// then update all synchronization counters that depend on the execution of this stage for their 
	// activation
	List<SyncRequirement*> *syncList = synchronizationReqs->getAllSyncReqirements();
	for (int i = 0; i < syncList->NumElements(); i++) {
		stream << nextIndent.str() << syncList->Nth(i)->getDependencyArc()->getArcName();
		stream << " += stage" << index << "Executed;\n";
	}
	
	// write a log entry for the stage executed
	stream << nextIndent.str() << "threadState->logExecution(\"";
	stream << name << "\", Space_" << space->getName() << ");\n"; 	
	
	// close the if condition if applicable
	if (!isGroupEntry()) {
		stream << indent.str() << "}\n";
	}
}

// An execution stage is a group entry if it contain any reduction operation
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
