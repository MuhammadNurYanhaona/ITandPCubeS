#include "../../../utils/task_global.h"
#include "../../../utils/code_constant.h"
#include "../../../../../../common-libs/utils/list.h"
#include "../../../../../../frontend/src/syntax/ast_stmt.h"
#include "../../../../../../frontend/src/syntax/ast_task.h"
#include "../../../../../../frontend/src/semantics/task_space.h"

#include <sstream>

// The current code generation process for external code blocks assumes that the code block is written
// in C or C++. So we can just expand the code blocks within the stream for the .cc file for the task.
// Later we have to change this strategy and expand external code blocks in separate files based on the
// underlying languages they have been written on. Then a function call type expansion semantics should
// be better. To elaborate, we will have the call being generated in our .cc file for the task and 
// expend the actual code block as a function in a separate file. That separate file can be compiled
// with the appropriate compiler for the language of the extern block. Afterwords, we will just link the
// object files as we do now.


void ExternCodeBlock::declareReplacementVars(std::ostringstream &stream, 
		std::string indents, Space *space) {
		
	TaskDef *taskDef = TaskDef::currentTask;
	List<TaskGlobalScalar*> *globalScalars = TaskGlobalCalculator::calculateTaskGlobals(taskDef);
	Space *rootLps = space->getRoot();
	
	// generate copies of scalar variables that matches the name of the source code
	if (globalScalars->NumElements() > 0) {
		stream << '\n' << indents << "// generating local variables for global scalars\n";
	}
	for (int i = 0; i < globalScalars->NumElements(); i++) {
		TaskGlobalScalar *scalar = globalScalars->Nth(i);
		Type *scalarType = scalar->getType();
		const char *varName = scalar->getName();
		stream << indents << scalarType->getCType() << " " << varName << " = ";
		if (scalar->isLocallyManageable()) {
			stream << "threadLocals->" << varName << stmtSeparator;
		} else {
			stream << "taskGlobals->" << varName << stmtSeparator;
		}
	}
	
	// if the LPS is partitioned into LPU then extract the ID from the LPU and make it directly
	// accessible within the external block
	if (space->getDimensionCount() > 0) {
		stream << '\n' << indents << "// generating a local version of the LPU ID\n";
		int dimensions = space->getDimensionCount();
		stream << indents << "int lpuId[" << dimensions << "]" << stmtSeparator;
		for (int i = 0; i < dimensions; i++) {
			stream << indents << "lpuId[" << i << "] = ";
			stream << "lpu->lpuId[" << i << ']' << stmtSeparator;
		}
	}

	// generating local versions of all array dimension metadata
	List<const char*> *arrays = rootLps->getLocallyUsedArrayNames();
	stream << '\n' << indents << "// generating local variables for array dimension metadata\n";
	for (int i = 0; i < arrays->NumElements(); i++) {
		const char *arrayName = arrays->Nth(i);
		ArrayDataStructure *array = (ArrayDataStructure *) rootLps->getStructure(arrayName);
		int arrayDims = array->getDimensionality();
		stream << indents << "Dimension " << arrayName << "_dimension";
		stream << "[" << arrayDims << "]" << stmtSeparator;
		for (int j = 0; j < arrayDims; j++) {
			stream << indents << arrayName << "_dimension[" << j << "] = arrayMetadata->";
			stream << arrayName << "Dims[" << j << "]" << stmtSeparator;
		}
	}
}

void ExternCodeBlock::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {

	std::ostringstream indentStr;
	for (int i = 0; i < indentLevel; i++) indentStr << indent;
	std::string earlyIndents = indentStr.str();
	indentStr <<  indent;
	std::string indents = indentStr.str();

	// start a new scope for the external code block
	stream << '\n' << earlyIndents << "{ // starting scope for an external code block\n";

	// if the extern code block is a part of a compute stage of a task then make scalar task global
	// variables accessible to the code block
	TaskDef *taskDef = TaskDef::currentTask;
	if (taskDef != NULL) {
		declareReplacementVars(stream, indents, space);
	}

	// jumping into the external code block within a further nested block
	stream << '\n' << indents << "{ // external code block starts\n";
	stream << codeBlock;
	stream << '\n' << indents << "} // external code block ends\n\n";

	// end the scope for the external code block
	stream << earlyIndents << "} // ending scope for the external code block\n\n";
}
