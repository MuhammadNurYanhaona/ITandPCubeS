#include "../../../utils/code_constant.h"
#include "../../../../../../common-libs/utils/list.h"
#include "../../../../../../frontend/src/syntax/ast_stmt.h"
#include "../../../../../../frontend/src/syntax/ast_expr.h"
#include "../../../../../../frontend/src/syntax/ast_type.h"
#include "../../../../../../frontend/src/semantics/task_space.h"

#include <iostream>
#include <sstream>
#include <string>
#include <cstdlib>

void ReductionStmt::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {

	const char *outputField;
	if (reductionVar != NULL) {
		// the case of a cross-LPU reduction on some task-global data structure 
		
		const char *resultName = reductionVar->getName();
		DataStructure *resultStruct = space->getStructure(resultName);
		Type *resultType = resultStruct->getType();

		// This naming strategy to find the appropriate property in the union holding reduction result 
		// is incomplete. Currently this is sufficient as we do not have the unsigned primitive types yet 
		// that have a space in their C type names. TODO we need to make change in the property naming 
		// convension when we will add those types in IT.
		std::ostringstream resultPropertyStr;
		resultPropertyStr << "data." << resultType->getCType() << "Value";
		std::string resultProperty = resultPropertyStr.str();

		std::ostringstream outputFieldStream;
		outputFieldStream << resultName << "->" << resultProperty;
		outputField = strdup(outputFieldStream.str().c_str());
	} else {
		// the case of a local reduction within the stage computation

		outputField = left->getName();
	}

	std::ostringstream indents;
	for (int i = 0; i < indentLevel; i++) indents << indent;
	
	if (op == SUM) {
		stream << indents.str() << outputField << " += ";
                right->translate(stream, indentLevel, 0, space);
                stream << stmtSeparator;
	} else if (op == PRODUCT) {
		stream << indents.str() << outputField << " *= ";
                right->translate(stream, indentLevel, 0, space);
                stream << stmtSeparator;
	} else if (op == MAX) {
		stream << indents.str() << "if (" << outputField;
		stream << " < ";
                right->translate(stream, indentLevel, 0, space);
		stream << ") {\n";
		stream << indents.str() << indent;
		stream << outputField << " = ";
                right->translate(stream, indentLevel, 0, space);
                stream << stmtSeparator;
		stream << indents.str() << "}\n";	
	} else if (op == MIN) {
		stream << indents.str() << "if (" << outputField;
		stream << " > ";
                right->translate(stream, indentLevel, 0, space);
		stream << ") {\n";
		stream << indents.str() << indent;
		stream << outputField << " = ";
                right->translate(stream, indentLevel, 0, space);
                stream << stmtSeparator;
		stream << indents.str() << "}\n";
	} else if (op == LAND) {
		stream << indents.str() << outputField << " = ";
		stream << outputField << " && ";
                right->translate(stream, indentLevel, 0, space);
                stream << stmtSeparator;
	} else if (op == LOR) {
		stream << indents.str() << outputField << " = ";
		stream << outputField << " || ";
                right->translate(stream, indentLevel, 0, space);
                stream << stmtSeparator;
	} else if (op == BAND) {
		stream << indents.str() << outputField << " = ";
		stream << outputField << " & ";
                right->translate(stream, indentLevel, 0, space);
                stream << stmtSeparator;
	} else if (op == BOR) {
		stream << indents.str() << outputField << " = ";
		stream << outputField << " | ";
                right->translate(stream, indentLevel, 0, space);
                stream << stmtSeparator;
	} else {
		std::cout << "Average, Max-entry, and Min-entry reductions haven't been implemented yet";
		std::exit(EXIT_FAILURE);
	}
}
