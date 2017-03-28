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

//------------------------------------------------------- Repeat Control Block --------------------------------------------------/

RepeatControlBlock::RepeatControlBlock(Space *space, 
		RepeatCycleType type, Expr *executeCond) : CompositeStage(space) {
	this->type = type;
	this->condition = executeCond;
}

void RepeatControlBlock::performDataAccessChecking(Scope *taskScope) {
	accessMap = validateDataAccess(taskScope, condition, NULL);
	CompositeStage::performDataAccessChecking(taskScope);
}

void RepeatControlBlock::print(int indentLevel) {
	std::ostringstream indent;
	for (int i = 0; i < indentLevel; i++) indent << '\t';
	std::cout << indent.str() << "Repition: ";
	std::cout << "(Space " << space->getName() << ")\n";
	CompositeStage::print(indentLevel);
}

void RepeatControlBlock::calculateLPSUsageStatistics() {
	
	Iterator<VariableAccess*> iterator = accessMap->GetIterator();
        VariableAccess *accessLog;
        
	while ((accessLog = iterator.GetNextValue()) != NULL) {
                if (!accessLog->isContentAccessed()) continue;
                const char *varName = accessLog->getName();
                DataStructure *structure = space->getLocalStructure(varName);
                AccessFlags *accessFlags = accessLog->getContentAccessFlags();
                LPSVarUsageStat *usageStat = structure->getUsageStat();

                // As repeat condition is supposed to be evaluated multiple times, any data structure been used
                // here should be marked as multiple access. Thus we add access information twice.
                if (accessFlags->isRead() || accessFlags->isWritten()) {
                        usageStat->addAccess();
                        usageStat->addAccess();
                }
        }

        // Two calls have been made for evaluating the usage statistics in nested computation stages assuming 
	// that a repeat cycle will executes at least twice, and two calls are sufficient to flag multiple 
	// accesses to data structures nested within the repeat block.
        CompositeStage::calculateLPSUsageStatistics();
        CompositeStage::calculateLPSUsageStatistics();
}

void RepeatControlBlock::performEpochUsageAnalysis() {
        CompositeStage::performEpochUsageAnalysis();
        FlowStage::CurrentFlowStage = this;
        condition->setEpochVersions(space, 0);
}

//-------------------------------------------------- Condititional Execution Block ----------------------------------------------/

ConditionalExecutionBlock::ConditionalExecutionBlock(Space *space, 
		Expr *executeCond) : CompositeStage(space) {
	this->condition = executeCond;
}

void ConditionalExecutionBlock::performDataAccessChecking(Scope *taskScope) {
        accessMap = validateDataAccess(taskScope, condition, NULL);
        CompositeStage::performDataAccessChecking(taskScope);
}

void ConditionalExecutionBlock::print(int indentLevel) {
        std::ostringstream indent;
        for (int i = 0; i < indentLevel; i++) indent << '\t';
        std::cout << indent.str() << "Conditional Execution: ";
	std::cout << "(Space " << space->getName() << ")\n";
        CompositeStage::print(indentLevel);
}

void ConditionalExecutionBlock::calculateLPSUsageStatistics() {
	
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
        }
        CompositeStage::calculateLPSUsageStatistics();
}

void ConditionalExecutionBlock::performEpochUsageAnalysis() {
        CompositeStage::performEpochUsageAnalysis();
        FlowStage::CurrentFlowStage = this;
        condition->setEpochVersions(space, 0);
}

//------------------------------------------------------ LPS Transition Block ---------------------------------------------------/

LpsTransitionBlock::LpsTransitionBlock(Space *space, Space *ancestorSpace) : CompositeStage(space) {
	this->ancestorSpace = ancestorSpace;	
}

void LpsTransitionBlock::print(int indentLevel) {
        std::ostringstream indent;
        CompositeStage::print(indentLevel - 1);
}

//-----------------------------------------------------  Epoch Boundary Block ---------------------------------------------------/

EpochBoundaryBlock::EpochBoundaryBlock(Space *space) : CompositeStage(space) {}

void EpochBoundaryBlock::print(int indentLevel) {
        std::ostringstream indent;
        for (int i = 0; i < indentLevel; i++) indent << '\t';
        std::cout << indent.str() << "Epoch Boundary: ";
	std::cout << "(Space " << space->getName() << ")\n";
        CompositeStage::print(indentLevel);
}
