#include "../../../utils/code_constant.h"
#include "../../../utils/name_transformer.h"
#include "../../../../../../common-libs/domain-obj/constant.h"
#include "../../../../../../frontend/src/syntax/ast_expr.h"
#include "../../../../../../frontend/src/semantics/task_space.h"
#include "../../../../../../frontend/src/semantics/computation_flow.h"

#include <fstream>
#include <sstream>

void RepeatControlBlock::generateInvocationCode(std::ofstream &stream, int indentation, Space *containerSpace) {

	std::ostringstream indentStream;
	for (int i = 0; i < indentation; i++) indentStream << indent;
	std::string indentStr = indentStream.str();

	// create a scope for repeat loop
	stream << std::endl << indentStr << "{ // scope entrance for repeat loop\n";

	// declare a repeat iteration number tracking variable
	stream << indentStr << "int repeatIteration = 0" << stmtSeparator;

	// get the name of the lpu for the execution LPS
	std::ostringstream lpuName;
	lpuName << "space" << space->getName() << "Lpu->";

	// If the repeat condition involves accessing metadata of some task global array then we need to create 
	// local copies of its metadata so that name transformer can work properly 
	if (isLpsDependent()) {
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

	// update the name transformer for probable array access within repeat condition
	ntransform::NameTransformer::transformer->setLpuPrefix(lpuName.str().c_str());

	if (this->type == Condition_Repeat) {
		std::ostringstream condStream;
		condition->translate(condStream, indentation, 0, space);
		stream << indentStr << "while (" << condStream.str() << ") {\n";
	} else {
		// translate the range expression into a for loop
		RangeExpr *rangeExpr = dynamic_cast<RangeExpr*>(condition);
		std::ostringstream rangeLoop;
		rangeExpr->generateLoopForRangeExpr(rangeLoop, indentation, space);
		stream << rangeLoop.str();
	}
	// declare all synchronization counter variables here that will be updated inside repeat loop 
	declareSynchronizationCounters(stream, indentation + 1, this->repeatIndex + 1);
	// translate the repeat body
	CompositeStage::generateInvocationCode(stream, indentation + 1, containerSpace);
	// increase the loop iteration counter
	stream << indentStr << "\trepeatIteration++" << stmtSeparator;
	// close the range loop
	stream << indentStr << "}\n";

	// exit the scope created for the repeat loop 
	stream << indentStr << "} // scope exit for repeat loop\n";
}
