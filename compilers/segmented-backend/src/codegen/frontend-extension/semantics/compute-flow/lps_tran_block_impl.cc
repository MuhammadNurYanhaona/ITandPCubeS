#include "../../../utils/code_constant.h"
#include "../../../utils/name_transformer.h"
#include "../../../../../../common-libs/utils/list.h"
#include "../../../../../../common-libs/domain-obj/constant.h"
#include "../../../../../../frontend/src/semantics/task_space.h"
#include "../../../../../../frontend/src/semantics/computation_flow.h"
#include "../../../../../../frontend/src/static-analysis/reduction_info.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

void LpsTransitionBlock::genReductionResultPreprocessingCode(std::ofstream &stream, int indentation) {
	
	std::ostringstream indentStream;
        for (int i = 0; i < indentation; i++) indentStream << indent;
        std::string indentStr = indentStream.str();
	List<ReductionMetadata*> *reductionInfoSet = space->getAllReductionConfigs();
	
	stream << std::endl;
	stream << indentStr << "{ // scope entrance for reduction result preparation\n\n";

	// retrieve the task data reference	
	stream << indentStr << "TaskData *taskData = threadState->getTaskData()" << stmtSeparator;

	// get a hold of the hierarchical LPU ID to locate data parts based on LPU ID
        stream << indentStr << "List<int*> *lpuIdChain = threadState->getLpuIdChainWithoutCopy(";
        stream << "Space_" << space->getName();
        Space *rootSpace = space->getRoot();
        stream << paramSeparator << "Space_" << rootSpace->getName() << ")" << stmtSeparator;

	for (int i = 0; i < reductionInfoSet->NumElements(); i++) {
		ReductionMetadata *metadata = reductionInfoSet->Nth(i);
		
		const char *resultVar = metadata->getResultVar();
		DataStructure *structure = space->getStructure(resultVar);
		Type *resultType = structure->getType();
		
		stream << std::endl << indentStr << "// processing for '" << resultVar << "'\n";

		// retrieve the result variable from the reduction results access container
		stream << indentStr << "reduction::Result *" << resultVar << " = ";
		stream << "taskData->getResultVar(\"" << resultVar << "\"" << paramSeparator;
		stream << "lpuIdChain)" << stmtSeparator;

		// assign the value of the reduction variable to proper thread-state property
		ReductionOperator reductionOp = metadata->getOpCode();
		if (reductionOp == MAX_ENTRY || reductionOp == MIN_ENTRY) {
			std::cout << "Sorry, index reductions haven't been implemented yet.\n";
			std::exit(EXIT_FAILURE);
		}
		std::ostringstream resultPropertyStr;
                resultPropertyStr << "data." << resultType->getCType() << "Value";
                std::string resultProperty = resultPropertyStr.str();
                std::ostringstream outputFieldStream;
                outputFieldStream << resultVar << "->" << resultProperty;
                std::string outputField = outputFieldStream.str();
		ntransform::NameTransformer *transformer = ntransform::NameTransformer::transformer;
                const char *propertyName = transformer->getTransformedName(resultVar, false, false);
                stream << indentStr << propertyName << " = " << outputField << stmtSeparator;	
	}

	stream << std::endl << indentStr << "} // scope exit for reduction result preparation\n";
}

void LpsTransitionBlock::generateInvocationCode(std::ofstream &stream, int indentation, Space *containerSpace) {

	const char *spaceName = space->getName();
	std::ostringstream indentStream;
        for (int i = 0; i < indentation; i++) indentStream << indent;
        std::string indentStr = indentStream.str();

	// create a new local scope for traversing LPUs of this new scope
	stream << std::endl;
	stream << indentStr << "{ // scope entrance for iterating LPUs of Space ";
	stream << spaceName << "\n";
	// declare a new variable for tracking the last LPU ID
	stream << indentStr << "int space" << spaceName << "LpuId = INVALID_ID" << stmtSeparator;
	// declare another variable to track the iteration number of the while loop
	stream << indentStr << "int space" << spaceName << "Iteration = 0" << stmtSeparator;
	// declare a new variable to hold on to current LPU of this LPS
	stream << indentStr << "Space" << spaceName << "_LPU *space" << spaceName << "Lpu = NULL";
	stream << stmtSeparator;
	// declare another variable to assign the value of get-Next-LPU call
	stream << indentStr << "LPU *lpu = NULL" << stmtSeparator;
	stream << indentStr << "while((lpu = threadState->getNextLpu(";
	stream << "Space_" << spaceName << paramSeparator << "Space_" << containerSpace->getName();
	stream << paramSeparator << "space" << spaceName << "LpuId)) != NULL) {\n";
	// cast the common LPU variable to LPS specific LPU             
	stream << indentStr << indent << "space" << spaceName << "Lpu = (Space" << spaceName;
	stream  << "_LPU*) lpu" << stmtSeparator;

	// if the current LPS is the root of the reduction range for some non-task-global reductions then prepare
	// the result variables of those reduction operations
	if (!space->isSingletonLps() && space->isRootOfSomeReduction()) {
		genReductionResultPreprocessingCode(stream, indentation + 1);		
	}

	// invoke the subflow nested within the LPS transition block
	CompositeStage::generateInvocationCode(stream, indentation + 1, this->space);

	// update the iteration number and next LPU ID
	stream << indentStr << indent << "space" << spaceName << "LpuId = space" << spaceName;
	stream << "Lpu->id" << stmtSeparator;
	stream << indentStr << indent << "space" << spaceName << "Iteration++" << stmtSeparator;
	stream << indentStr << "}\n";

	// at the end, remove LPS entry checkpoint checkpoint if the container LPS is not the root LPS
	if (!containerSpace->isRoot()) {
		stream << indentStr << "threadState->removeIterationBound(Space_";
		stream << containerSpace->getName() << ')' << stmtSeparator;
	}

	// exit from the scope
	stream << indentStr << "} // scope exit for iterating LPUs of Space ";
	stream << space->getName() << "\n";	
}
