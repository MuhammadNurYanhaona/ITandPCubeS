#include "../ast.h"
#include "../ast_type.h"
#include "../ast_def.h"
#include "../ast_stmt.h"
#include "../ast_type.h"
#include "../ast_task.h"
#include "../ast_partition.h"
#include "../../common/errors.h"
#include "../../common/location.h"
#include "../../common/constant.h"
#include "../../semantics/scope.h"
#include "../../semantics/symbol.h"
#include "../../semantics/helper.h"
#include "../../semantics/computation_flow.h"
#include "../../semantics/data_access.h"
#include "../../static-analysis/task_env_stat.h"
#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/hashtable.h"
#include "../../../../common-libs/utils/string_utils.h"

#include <sstream>

//------------------------------------------------------ Task Definition ----------------------------------------------------------/

TaskDef *TaskDef::currentTask = NULL;

TaskDef::TaskDef(Identifier *i,
                DefineSection *d,
                EnvironmentSection *e,
                InitializeSection *init,
		StagesSection *s,
                ComputationSection *c,
                PartitionSection *p): Definition(*i->GetLocation()) {

        Assert(i != NULL && d != NULL && e != NULL && s != NULL && c != NULL && p != NULL);
        id = i;
        id->SetParent(this);
        define = d;
        define->SetParent(this);
        environment = e;
        environment->SetParent(this);
        initialize = init;
        if (initialize != NULL) {
                initialize->SetParent(this);
        }
	stages = s;
	stages->SetParent(this);
        compute = c;
        compute->SetParent(this);
        partition = p;
        partition->SetParent(this);
}

void TaskDef::PrintChildren(int indentLevel) {
        id->Print(indentLevel + 1);
        define->Print(indentLevel + 1);
        environment->Print(indentLevel + 1);
        if (initialize != NULL) initialize->Print(indentLevel + 1);
	stages->Print(indentLevel + 1);
        compute->Print(indentLevel + 1);
        partition->Print(indentLevel + 1);
}

void TaskDef::analyzeStageDefinitions() {
	List<StageDefinition*> *stageDefs = stages->getStageDefinitions();
	for (int i =0; i < stageDefs->NumElements(); i++) {
		StageDefinition *stageDef = stageDefs->Nth(i);
		stageDef->determineArrayDimensions();
	}
}

List<Type*> *TaskDef::getInitArgTypes() {
	if (initialize == NULL) return new List<Type*>;
        else return initialize->getArgumentTypes();
}

int TaskDef::getPartitionArgsCount() { 
        return partition->getArgumentsCount(); 
}

void TaskDef::attachScope(Scope *parentScope) {

        //--------------------------------create a scope with all the variables declared in the define section
        
	Scope *scope = new Scope(TaskScope);
        List<VariableDef*> *varList = define->getDefinitions();
        for (int i = 0; i < varList->NumElements(); i++) {
                VariableDef *var = varList->Nth(i);
                VariableSymbol *varSym = new VariableSymbol(var);
                scope->insert_symbol(varSym);
                if (var->isReduction()) {
                        varSym->flagAsReduction();
                }
        }

        //------------------------------------create the environment scope and at the same time a tuple for it

        List<VariableDef*> *envDef = new List<VariableDef*>;
        List<Type*> *envElementTypes = new List<Type*>;
        Scope *envScope = new Scope(TaskScope);
        List<EnvironmentLink*> *envLinks = environment->getLinks();
        for (int i = 0; i < envLinks->NumElements(); i++) {
                Identifier *var = envLinks->Nth(i)->getVariable();
                Symbol *symbol = scope->lookup(var->getName());
                if (symbol == NULL) {
                        ReportError::UndefinedSymbol(var, false);
                } else {
                        envScope->copy_symbol(symbol);
                        VariableSymbol *varSym = (VariableSymbol*) symbol;
                        envDef->Append(new VariableDef(var, varSym->getType()));
                        envElementTypes->Append(varSym->getType());
                }
        }
        envDef->Append(new VariableDef(new Identifier(*GetLocation(), "name"), Type::stringType));
        envElementTypes->Append(Type::stringType);

        const char *initials = string_utils::getInitials(id->getName());
        char *envTupleName = (char *) malloc(strlen(initials) + 12);
        strcpy(envTupleName, initials);
        strcat(envTupleName, "Environment");

	Identifier *envId = new Identifier(*GetLocation(), envTupleName);
        envTuple = new TupleDef(envId, envDef);
        envTuple->flagAsEnvironment();
        envTuple->setSymbol(new TupleSymbol(envId, envTuple, envElementTypes));
        envTuple->getSymbol()->setNestedScope(envScope);
        parentScope->insert_symbol(envTuple->getSymbol());

	//-------------------------------------------------------create the partition scope and a tuple for it
        
	List<Identifier*> *partitionArgs = partition->getArguments();
        List<VariableDef*> *partitionDef = new List<VariableDef*>;
        List<Type*> *partElementTypes = new List<Type*>;
        char *partTupleName = (char *) malloc(strlen(initials) + 10);
        strcpy(partTupleName, initials);
        strcat(partTupleName, "Partition");
        Identifier *partitionId = new Identifier(*GetLocation(), partTupleName);
        Scope *partScope = new Scope(TaskPartitionScope);
        for (int i = 0; i < partitionArgs->NumElements(); i++) {
                Identifier *arg = partitionArgs->Nth(i);
                VariableDef *var = new VariableDef(arg, Type::intType);
                partitionDef->Append(var);
                partScope->insert_symbol(new VariableSymbol(var));
                partElementTypes->Append(Type::intType);
        }
        partitionTuple = new TupleDef(partitionId, partitionDef);
        partitionTuple->setSymbol(new TupleSymbol(partitionId, partitionTuple, partElementTypes));
        partitionTuple->getSymbol()->setNestedScope(partScope);
        parentScope->insert_symbol(partitionTuple->getSymbol());

        //----------------------------------------------------set the symbol for the task and its nested scope
        
	symbol = new TaskSymbol(id->getName(), this);
        symbol->setNestedScope(scope);
        ((TaskSymbol *) symbol)->setEnvScope(envScope);
        ((TaskSymbol *) symbol)->setPartitionScope(partScope);
        parentScope->insert_symbol(symbol);
}

void TaskDef::typeCheckInitializeSection(Scope *scope) {
	if (initialize != NULL) {
                Scope *executionScope = scope->enter_scope(new Scope(ExecutionScope));
                NamedType *partitionType = new NamedType(partitionTuple->getId());
                executionScope->insert_symbol(new VariableSymbol("partition", partitionType));
                initialize->performScopeAndTypeChecking(executionScope);
        }
}

void TaskDef::constructPartitionHierarchy() {
         partition->constructPartitionHierarchy(this);
}

void TaskDef::constructComputationFlow(Scope *programScope) {

	// set up a static reference of the current task to be accessible during the constrution process
	TaskDef::currentTask = this;

	// retrieve the root of the partition hierarchy
	PartitionHierarchy *lpsHierarchy = partition->getPartitionHierarchy();
	Space *rootLps = lpsHierarchy->getRootSpace();

	// prepare the task scope
	Scope *taskScope = programScope->enter_scope(symbol->getNestedScope());
	Scope *executionScope = taskScope->enter_scope(new Scope(ExecutionScope)); 
	NamedType *partitionType = new NamedType(partitionTuple->getId());
	executionScope->insert_symbol(new VariableSymbol("partition", partitionType));

	// pass control to the Computation Section to prepare the computation flow
	FlowStageConstrInfo cnstrInfo = FlowStageConstrInfo(rootLps, 
			executionScope, lpsHierarchy);
	computation = compute->generateComputeFlow(&cnstrInfo);
}

void TaskDef::validateScope(Scope *parentScope) {
	
	// check if the variables in the Define Section have valid types
	List<VariableDef*> *varList = define->getDefinitions();
        for (int i = 0; i < varList->NumElements(); i++) {
                VariableDef *var = varList->Nth(i);
                var->validateScope(parentScope);
        }

	// check access characteristics of task-global variables in the initialize section
	Scope *scope = symbol->getNestedScope();
	if (initialize != NULL) {
		initialize->performVariableAccessAnalysis(scope);
	}

	// check access characteristics of task-global variables in the computation flow
	computation->performDataAccessChecking(scope);
} 

// Note that the ordering of the function calls within the whole analysis procedure is important as 
// there are internal depdendency among these analyses through their input/output data structures. 
void TaskDef::performStaticAnalysis() {

        Scope *scope = symbol->getNestedScope();

        //------------------------------------------------------------------- Sync Stage Implantation
	
	// sync stages are implanted in-between LPS transition and compute stage execution to
	// facilitate data dependency identification and resolution as the flow of control moves
	// from stages to stages	
	List<FlowStage*> *inProgressStageList = new List<FlowStage*>;
	computation->implantSyncStagesInFlow(NULL, inProgressStageList);

	//------------------------------------------------------------- Detailed Data Access Analyses
        
	// transfer variable access information to partition hierarchy to aid memory management
        computation->calculateLPSUsageStatistics();
	// determine how many versions of different data structures need to be maintained at runtime
        computation->performEpochUsageAnalysis();
	// flag those LPSes that have computation stages in them to decide about LPU generation
        computation->setLpsExecutionFlags();

	//------------------------------------------------ Stage Augmentation for LPS Synchronization
        
	// generate after execution environment statistics that tells about stale/fresh data items
        TaskEnvStat *taskEnvStat = getAfterExecutionEnvStat();
        // create list of sync stages for LPSes that have their copy of some data being stale
        List<FlowStage*> *syncStagesForStaleLpses = taskEnvStat->generateSyncStagesForStaleLpses();
        // append the sync stages to the computation flow
        for (int i = 0; i < syncStagesForStaleLpses->NumElements(); i++) {
                computation->addStageAtEnd(syncStagesForStaleLpses->Nth(i));
        }

	//------------------------------------------------------------- Flow Expansion for Reductions
        
	// generate metadata about reduction statements found in the computation 
	PartitionHierarchy *lpsHierarchy = partition->getPartitionHierarchy();
        computation->populateReductionMetadata(lpsHierarchy);
	// insert reduction boundary stages in the computation flow
	computation->upliftReductionInstrs();
	// again populate reduction metadata in compute stage instances as the uplifting process
	// remove the previous metadata
        computation->populateReductionMetadata(lpsHierarchy);
	// presumably new stages have been added to the flow; so correct different indexes of the
	// flow stages
	computation->assignIndexAndGroupNo(0, -1, -1);
	// then validate the usage of reduction results; validation needs correct stage indexing
	computation->validateReductions();

	//--------------------------------------------------------------- Filling LPS Transition Gaps

	// In some back-ends it is difficult to generate code for implicit LPS transitions. By
	// implicit transitions we mean a direct transition from an ancestor LPS to a descendent 
	// LPS skipping any intermediate LPSes. So after all stage augmentations are done, we
	// make all LPS transitions explicit
	computation->makeAllLpsTransitionsExplicit();
	// presumably new stages have been added to the flow; so again assign correct indexes to 
	// the flow stages
	computation->assignIndexAndGroupNo(0, -1, -1);
	
	//------------------------------------------------------------------ Data Dependency Analysis

	// determine the read-write dependencies that occur as flow of computation moves along stages   
        computation->performDependencyAnalysis(lpsHierarchy);
	// determine what dependency relationships should be translated into synchronization require-
        // ments and recursively mark the sources of these synchronization signals      
        computation->analyzeSynchronizationNeeds();
	// uplift the destinations of composite-stage boundary crossing synchronization dependencies
        computation->upliftSynchronizationDependencies();
        // then uplift the sources of the composite-stage boundary crossing dependencies
        computation->upliftSynchronizationNeeds();
	// finally determine which sync stage for a synchronization will reset the sync primitive so 
	// that it can be reused (e.g., for latter iterations)
        computation->setReactivatorFlagsForSyncReqs();
}

List<VariableAccess*> *TaskDef::getAccessLogOfEnvVariables() {
        List<VariableAccess*> *accessList = new List<VariableAccess*>;
        List<EnvironmentLink*> *links = environment->getLinks();
        for (int i = 0; i < links->NumElements(); i++) {
                EnvironmentLink *link = links->Nth(i);
                const char *varName = link->getVariable()->getName();
                accessList->Append(new VariableAccess(varName));
        }
        computation->fillInTaskEnvAccessList(accessList);
        return accessList;
}

TaskEnvStat *TaskDef::getAfterExecutionEnvStat() {
        List<VariableAccess*> *envAccessList = getAccessLogOfEnvVariables();
        Space *rootLps = partition->getPartitionHierarchy()->getRootSpace();
        TaskEnvStat *taskStat = new TaskEnvStat(envAccessList, rootLps);
        computation->prepareTaskEnvStat(taskStat);
        return taskStat;
}

CompositeStage *TaskDef::getComputation() { return computation; }
