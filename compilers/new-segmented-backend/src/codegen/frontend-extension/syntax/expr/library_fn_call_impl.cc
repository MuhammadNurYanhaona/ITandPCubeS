#include "../../../../../../frontend/src/syntax/ast_library_fn.h"
#include "../../../../../../frontend/src/syntax/ast_expr.h"
#include "../../../../../../frontend/src/semantics/task_space.h"
#include "../../../utils/code_constant.h"

#include <sstream>
#include <cstdlib>

void Root::translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space) {
	
	// argument 1 is the number
	Expr *arg1 = arguments->Nth(0);
	// argument 2 is the root power
	Expr *arg2 = arguments->Nth(1);

	stream << "pow(";
	arg1->translate(stream, indentLevel);
	stream << paramSeparator;
	stream << "1.0 / ";	
	arg2->translate(stream, indentLevel);
	stream << ")";
}

void Random::translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space) {
	stream << "rand()";
}

void LoadArray::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {

	Expr *arg1 = arguments->Nth(0);
	Expr *arg2 = arguments->Nth(1);
	ArrayType *array = (ArrayType*) arg1->getType();
	Type *elemType = array->getTerminalElementType();
	int dimensions = array->getDimensions();

	std::ostringstream indentStr;
	for (int i = 0; i < indentLevel; i++) indentStr << indent;

	stream << indentStr.str() << "{ // scope starts for load-array operation\n";
	std::ostringstream arrayExpr;
	
	// get the translated C++ expression for the argument array
	arg1->translate(arrayExpr, 0);
	
	// create a dimension array to hold metadata about the array
	stream << indentStr.str() << "Dimension arrayDims[" << dimensions << "]" << stmtSeparator;

	// generate a prompt to decide if to read the array from file or not
	stream << indentStr.str() << "if (outprompt::getYesNoAnswer(\"Want to read array";
        stream << " \\\"" << arrayExpr.str() << "\\\" from a file?\"";
        stream << ")) {\n";

        // if the response is yes then generate a prompt for reading the array from a file 
        stream << indentStr.str() << indent;
        stream << arrayExpr.str() << " = ";
	stream << "inprompt::readArrayFromFile ";
	stream << '<' << elemType->getName() << "> ";
	stream << "(\"" << arrayExpr.str() << "\"" << paramSeparator;
	stream << std::endl << indentStr.str() << doubleIndent;
	stream << dimensions << paramSeparator;
	stream << "arrayDims" << paramSeparator;
	arg2->translate(stream, 0);
	stream << ")" << stmtSeparator;

	// otherwise, generate code for randomly initialize the array
	stream << indentStr.str() << "} else {\n";
	// create a prompt to get the dimensions information for the variable under concern
	stream << indentStr.str() << indent;
	stream << "inprompt::readArrayDimensionInfo(\"" << arrayExpr.str() << "\"" << paramSeparator;
	stream << dimensions << paramSeparator << "arrayDims)" << stmtSeparator;
	// then allocate an array for the variable
	stream << indentStr.str() << indent;
	stream << arrayExpr.str() << " = allocate::allocateArray ";
	stream << '<' << elemType->getName() << "> ";
	stream << '(' << dimensions << paramSeparator;
	stream << "arrayDims)" << stmtSeparator;
	// finally randomly initialize the array
	stream << indentStr.str() << indent;
	stream << "allocate::randomFillPrimitiveArray ";
	stream << '<' << elemType->getName() << "> ";
	stream << "(" << arrayExpr.str() << paramSeparator;
	stream << std::endl << indentStr.str() << doubleIndent;
	stream << dimensions << paramSeparator;
	stream << "arrayDims)" << stmtSeparator;
	stream << indentStr.str() << "}\n";
	
	// populate partition dimension objects of the array based on the dimension been updated by the library function
	for (int d = 0; d < dimensions; d++) {
		stream << indentStr.str() << arrayExpr.str() << "Dims[" << d << "].partition = "; 
		stream << "arrayDims[" << d << "]" << stmtSeparator; 
		stream << indentStr.str() << arrayExpr.str() << "Dims[" << d << "].storage = "; 
		stream << "arrayDims[" << d << "].getNormalizedDimension()" << stmtSeparator; 
	}

	stream << indentStr.str() << "} // scope ends for load-array operation\n";
}

void StoreArray::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {
		
	Expr *arg1 = arguments->Nth(0);
	Expr *arg2 = arguments->Nth(1);
	ArrayType *array = (ArrayType*) arg1->getType();
	Type *elemType = array->getTerminalElementType();
	int dimensions = array->getDimensions();

	std::ostringstream indentStr;
	for (int i = 0; i < indentLevel; i++) indentStr << indent;
	
	// get the translated C++ expression for the argument array
	std::ostringstream arrayExpr;
	arg1->translate(arrayExpr, 0);

	// first generate a prompt that will ask the user if he wants to write this array to a file
	stream << indentStr.str() << "if (outprompt::getYesNoAnswer(\"Want to save array";
	stream << " \\\"" << arrayExpr.str() << "\\\" in a file?\"";
	stream << ")) {\n";

	// create a dimension object and copy storage information from array metadata object into the former
	stream << indentStr.str() << indent;
	stream << "Dimension arrayDims[" << dimensions << "]" << stmtSeparator;
	for (int d = 0; d < dimensions; d++) {
		stream << indentStr.str() << indent; 
		stream << "arrayDims[" << d << "] = "; 
		stream << arrayExpr.str() << "Dims[" << d << "].storage"; 
		stream << stmtSeparator; 
	}

	// then generate the prompt for writing the array to the file specified by the user
	stream << indentStr.str() << "\toutprompt::writeArrayToFile ";
	stream << '<' << elemType->getName() << '>';
	stream << " (\"" << arrayExpr.str() << "\"" << paramSeparator;
	stream << std::endl << indentStr.str() << doubleIndent;
	stream << arrayExpr.str() << paramSeparator;
	stream << dimensions << paramSeparator;
	stream << "arrayDims" << paramSeparator;
	arg2->translate(stream, 0);
	stream << ")" << stmtSeparator;

	// close the if block at the end        
	stream << indentStr.str() << "}\n";
}

void BindInput::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {

	std::ostringstream indentStr;
	for (int i = 0; i < indentLevel; i++) indentStr << indent;
	std::string indents = indentStr.str();
	
	Expr *arg1 = arguments->Nth(0);
	Expr *arg2 = arguments->Nth(1);
	Expr *arg3 = arguments->Nth(2);

	stream << '\n' << indents << "{ // scope starts for binding input\n";
	stream << indents << "TaskItem *item = ";
	arg1->translate(stream, indentLevel);
	stream << "->getItem(";
	arg2->translate(stream, indentLevel);
	stream << ")" << stmtSeparator;
	stream << indents << "ReadFromFileInstruction *instr = new ReadFromFileInstruction(item)";
	stream << stmtSeparator;
	stream << indents << "instr->setFileName(";
	arg3->translate(stream, indentLevel);
	stream << ")" << stmtSeparator;
	stream << indents; 
	arg1->translate(stream, indentLevel);
	stream << "->addInitEnvInstruction(instr)" << stmtSeparator;
	stream << indents << "} // scope ends for binding input\n\n";
}

void BindOutput::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {
		
	std::ostringstream indentStr;
	for (int i = 0; i < indentLevel; i++) indentStr << indent;
	std::string indents = indentStr.str();
	
	Expr *arg1 = arguments->Nth(0);
	Expr *arg2 = arguments->Nth(1);
	Expr *arg3 = arguments->Nth(2);

	stream << indents;
	arg1->translate(stream, indentLevel);
	stream << "->writeItemToFile(";
	arg2->translate(stream, indentLevel);
	stream << paramSeparator;	
	arg3->translate(stream, indentLevel);
	stream << ")" << stmtSeparator;
}
