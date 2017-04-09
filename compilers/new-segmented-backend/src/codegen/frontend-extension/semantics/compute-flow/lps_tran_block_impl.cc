#include "../../../utils/code_constant.h"
#include "../../../../../../frontend/src/semantics/task_space.h"
#include "../../../../../../frontend/src/semantics/computation_flow.h"

#include <fstream>
#include <sstream>

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
