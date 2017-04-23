#include "../../../utils/code_constant.h"
#include "../../../utils/name_transformer.h"
#include "../../../../../../frontend/src/syntax/ast_expr.h"
#include "../../../../../../frontend/src/semantics/task_space.h"
#include "../../../../../../frontend/src/semantics/computation_flow.h"

#include <fstream>
#include <sstream>

void ConditionalExecutionBlock::generateInvocationCode(std::ofstream &stream, 
		int indentation, Space *containerSpace) {

	// get the name of the lpu for the executing LPS
	std::ostringstream lpuName;
	lpuName << "space" << space->getName() << "Lpu->";

	std::ostringstream indentStream;
	for (int i = 0; i < indentation; i++) indentStream << indent;
	std::string indentStr = indentStream.str();

	stream << indentStr << "{ // scope entrance for conditional subflow\n";

	if (isLpsDependent()) {
		// if the condition involves accessing metadata of some task global array then we need to 
		// create local copies of its metadata so that name transformer can work properly 
		List<const char*> *localArrays = filterInArraysFromAccessMap();
		for (int i = 0; i < localArrays->NumElements(); i++) {
			const char *arrayName = localArrays->Nth(i);
			ArrayDataStructure *array = (ArrayDataStructure*) space->getStructure(arrayName);
			int dimensions = array->getDimensionality();
			stream << indentStr << "Dimension ";
			stream  << arrayName << "PartDims[" << dimensions << "]" << stmtSeparator;
			stream << indentStr << "Dimension ";
			stream  << arrayName << "StoreDims[" << dimensions << "]" << stmtSeparator;
			for (int j = 0; j < dimensions; j++) {
				stream << indentStr;
				stream << arrayName << "PartDims[" << j << "] = " << lpuName.str();
				stream << arrayName << "PartDims[" << j << "].partition" << stmtSeparator;
				stream << indentStr;
				stream << arrayName << "StoreDims[" << j << "] = " << lpuName.str();
				stream << arrayName << "PartDims[" << j << "].storage" << stmtSeparator;
			}
		}
	}
	
	// update the name transformer's LPU accesss prefix for probable array access within the condition
	ntransform::NameTransformer::transformer->setLpuPrefix(lpuName.str().c_str());

	// generate condition statement
	std::ostringstream conditionStream;
	condition->translate(conditionStream, indentation, 0, space);
	stream << indentStr << "if(" << conditionStream.str() << ") {\n";
	
	// generate code for the sub-flow nested within the condition block
	CompositeStage::generateInvocationCode(stream, indentation + 1, containerSpace);

	stream << indentStr << "} // end of condition checking block\n";
	stream << indentStr << "} // scope exit for conditional subflow\n";
}
