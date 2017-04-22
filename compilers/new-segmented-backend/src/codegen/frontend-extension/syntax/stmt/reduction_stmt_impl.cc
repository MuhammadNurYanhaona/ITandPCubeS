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

	if (reductionVar == NULL) {
		std::cout << "Code-generation for Local reduction inside compute stages is not supported yet";
		std::exit(EXIT_FAILURE);
	}

	//----------------------------------- the case of a cross-LPU reduction on some task-global data structure 
	
	const char *outputField;
	const char *resultName = reductionVar->getName();
	DataStructure *resultStruct = space->getStructure(resultName);
	Type *resultType = resultStruct->getType();

	// This naming strategy to find the appropriate property in the union holding reduction result is 
	// incomplete. Currently this is sufficient as we do not have the unsigned primitive types yet that have 
	// a space in their C type names. TODO we need to make change in the property naming convension when we 
	// will add those types in IT.
	std::string resultProperty;
	if (op == MIN_ENTRY || op == MAX_ENTRY) {
		std::ostringstream resultPropertyStr;
		Type *exprType = right->getType();
		resultPropertyStr << "data." << exprType->getCType() << "Value";
		resultProperty = resultPropertyStr.str();
	} else {
		std::ostringstream resultPropertyStr;
		resultPropertyStr << "data." << resultType->getCType() << "Value";
		resultProperty = resultPropertyStr.str();
	}

	std::ostringstream outputFieldStream;
	outputFieldStream << resultName << "->" << resultProperty;
	outputField = strdup(outputFieldStream.str().c_str());

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
	} else if (op == MAX || op == MAX_ENTRY) {
		stream << indents.str() << "if (" << outputField;
		stream << " < ";
                right->translate(stream, indentLevel, 0, space);
		stream << ") {\n";
		stream << indents.str() << indent;
		stream << outputField << " = ";
                right->translate(stream, indentLevel, 0, space);
                stream << stmtSeparator;
		if (op == MAX_ENTRY) {
			List<const char*> *indexFields = enclosingLoop->getAllIndexNames();
			stream << indents.str() << indent;
			stream << resultName << "->index = " << indexFields->Nth(0) << stmtSeparator;
		}
		stream << indents.str() << "}\n";	
	} else if (op == MIN || op == MIN_ENTRY) {
		stream << indents.str() << "if (" << outputField;
		stream << " > ";
                right->translate(stream, indentLevel, 0, space);
		stream << ") {\n";
		stream << indents.str() << indent;
		stream << outputField << " = ";
                right->translate(stream, indentLevel, 0, space);
                stream << stmtSeparator;
		if (op == MIN_ENTRY) {
			List<const char*> *indexFields = enclosingLoop->getAllIndexNames();
                        stream << indents.str() << indent;
                        stream << resultName << "->index = " << indexFields->Nth(0) << stmtSeparator;
		}
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
		std::cout << "User defined reduction primitives are not being supported yet";
		std::exit(EXIT_FAILURE);
	}
}
