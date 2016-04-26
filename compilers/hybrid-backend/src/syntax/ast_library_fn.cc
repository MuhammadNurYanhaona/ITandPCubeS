#include "ast.h"
#include "ast_expr.h"
#include "ast_type.h"
#include "ast_library_fn.h"
#include "../utils/list.h"
#include "../utils/code_constant.h"
#include "../utils/decorator_utils.h"
#include "../semantics/scope.h"
#include "../semantics/symbol.h"
#include "errors.h"

#include <iostream>

//-------------------------------------------------------- Static Constants -----------------------------------------------------/

const char *Root::Name = "root";
const char *Random::Name = "random";
const char *LoadArray::Name = "load_array";
const char *LoadListOfArrays::Name = "load_list_of_arrays";
const char *StoreArray::Name = "store_array";
const char *StoreListOfArrays::Name = "store_list_of_arrays";
const char *BindInput::Name = "bind_input";
const char *BindOutput::Name = "bind_output";

//-------------------------------------------------------- Library Function -----------------------------------------------------/

LibraryFunction::LibraryFunction(int argumentCount, Identifier *functionName, 
		List<Expr*> *arguments, yyltype loc) : Expr(loc) {
	this->argumentCount = argumentCount;
	this->functionName = functionName;
	this->arguments = arguments;
}

LibraryFunction::LibraryFunction(int argumentCount, Identifier *functionName, List<Expr*> *arguments) : Expr() {
	this->argumentCount = argumentCount;
	this->functionName = functionName;
	this->arguments = arguments;
}

bool LibraryFunction::isLibraryFunction(Identifier *id) {
	const char* name = id->getName();
	return (strcmp(name, Root::Name) == 0 || strcmp(name, Random::Name) == 0
		|| strcmp(name, LoadArray::Name) == 0 || strcmp(name, LoadListOfArrays::Name) == 0
		|| strcmp(name, StoreArray::Name) == 0 
		|| strcmp(name, StoreListOfArrays::Name) == 0
		|| strcmp(name, BindInput::Name) == 0
		|| strcmp(name, BindOutput::Name) == 0);
}

void LibraryFunction::PrintChildren(int indentLevel) {
	PrintLabel(indentLevel + 1, "Arguments");
        arguments->PrintAll(indentLevel + 2);
}

void LibraryFunction::resolveType(Scope *scope, bool ignoreFailure) {
	if (argumentCount != arguments->NumElements()) {
		std::cout << "argument count problem\n";
		ReportError::TooFewOrTooManyParameters(functionName, arguments->NumElements(),
                		argumentCount, ignoreFailure);
	} else {
		validateArguments(scope, ignoreFailure);
	}
}

LibraryFunction *LibraryFunction::getFunctionExpr(Identifier *id, List<Expr*> *arguments, yyltype loc) {
	
	const char* name = id->getName();
	LibraryFunction *function = NULL;

	// note that there should never be a default 'else' block here; then the system will fail to find user defined functions
	if (strcmp(name, Root::Name) == 0) {
		function = new Root(id, arguments, loc);
	} else if (strcmp(name, Random::Name) == 0) {
		function = new Random(id, arguments, loc);
	} else if (strcmp(name, LoadArray::Name) == 0) {
		function = new LoadArray(id, arguments, loc);
	} else if (strcmp(name, LoadListOfArrays::Name) == 0) {
		function = new LoadListOfArrays(id, arguments, loc);
	} else if (strcmp(name, StoreArray::Name) == 0) {
		function = new StoreArray(id, arguments, loc);
	} else if (strcmp(name, StoreListOfArrays::Name) == 0) {
		function = new StoreListOfArrays(id, arguments, loc);
	} else if (strcmp(name, BindInput::Name) == 0) {
		function = new BindInput(id, arguments, loc);
	} else if (strcmp(name, BindOutput::Name) == 0) {
		function = new BindOutput(id, arguments, loc);
	}

	return function;	
}

//------------------------------------------------------------ Root ------------------------------------------------------------/

void Root::validateArguments(Scope *scope, bool ignoreFailure) {

	Expr *arg1 = arguments->Nth(0);
	Expr *arg2 = arguments->Nth(1);
	arg1->resolveType(scope, ignoreFailure);
	arg2->resolveType(scope, ignoreFailure);

	Type *arg1Type = arg1->getType();
	if (arg1Type == NULL) {
		ReportError::UnknownExpressionType(arg1, ignoreFailure);	
	} else if (arg1Type != Type::intType 
			&& arg1Type != Type::floatType 
			&& arg1Type != Type::doubleType 
			&& arg1Type != Type::errorType) {
		ReportError::InvalidExprType(arg1, arg1Type, ignoreFailure);
	}

	Type *arg2Type = arg2->getType();
	if (arg2Type == NULL) {
		ReportError::UnknownExpressionType(arg2, ignoreFailure);	
	} else if (arg2Type != Type::intType && arg2Type != Type::errorType) {
		ReportError::IncompatibleTypes(arg2->GetLocation(), arg2Type, Type::intType, ignoreFailure);
	}
	
	this->type = arg1Type;
}

void Root::inferType(Scope *scope, Type *rootType) {
	if (this->type == NULL) {
		this->type = rootType;
	}
	if (arguments->NumElements() == 2) {
		arguments->Nth(0)->inferType(scope, this->type);
		arguments->Nth(1)->inferType(scope, Type::intType);
	}
}

//--------------------------------------------------- Array Operation -----------------------------------------------------/

void ArrayOperation::validateArguments(Scope *scope, bool ignoreFailure) {
	
	Expr *arg1 = arguments->Nth(0);
	arg1->resolveType(scope, ignoreFailure);
	Type *arg1Type = arg1->getType();
	if (arg1Type == NULL) {
		ReportError::UnknownExpressionType(arg1, ignoreFailure);	
	} else {
		ArrayType *arrayType = dynamic_cast<ArrayType*>(arg1Type);
		if (arrayType == NULL) {
			ReportError::InvalidArrayAccess(arg1->GetLocation(), arg1Type, ignoreFailure);
		}
	}

	Expr *arg2 = arguments->Nth(1);
	arg2->resolveType(scope, ignoreFailure);
	Type *arg2Type = arg2->getType();
	if (arg2Type == NULL) {
		ReportError::UnknownExpressionType(arg2, ignoreFailure);	
	} else if (arg2Type != Type::stringType && arg2Type != Type::errorType) {
		ReportError::IncompatibleTypes(arg2->GetLocation(), arg2Type, Type::stringType, ignoreFailure);
	}
	this->type = Type::voidType;	
}

void ArrayOperation::inferType(Scope *scope, Type *rootType) {
	if (arguments->NumElements() == 2) {
		arguments->Nth(1)->inferType(scope, Type::stringType);
		arguments->Nth(1)->resolveType(scope, false);
	}
}

//------------------------------------------------------- Load Array ------------------------------------------------------/

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

//------------------------------------------------------- Store Array -----------------------------------------------------/

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

//----------------------------------------------------- Bind Operation ----------------------------------------------------/

void BindOperation::validateArguments(Scope *scope, bool ignoreFailure) {

	NamedType *envType = NULL;	
	Expr *arg1 = arguments->Nth(0);
	arg1->resolveType(scope, ignoreFailure);
	Type *arg1Type = arg1->getType();
	if (arg1Type == NULL) {
		ReportError::UnknownExpressionType(arg1, ignoreFailure);	
	} else {
		NamedType *objectType = dynamic_cast<NamedType*>(arg1Type);
		if (objectType == NULL || !objectType->isEnvironmentType()) {
			ReportError::NotAnEnvironment(arg1->GetLocation(), arg1Type, ignoreFailure);
		} else {
			envType = objectType;
		}
	}
	
	Expr *arg2 = arguments->Nth(1);
	StringConstant *varName = dynamic_cast<StringConstant*>(arg2);
	if (varName == NULL) {
		ReportError::NotAConstant(arg2->GetLocation(), "string", ignoreFailure);
	} else if (envType != NULL) {
		const char *arrayName = varName->getValue();
		Symbol *symbol = scope->lookup(envType->getTaskName());
		TaskSymbol *task = dynamic_cast<TaskSymbol*>(symbol);
		TaskDef *taskDef = (TaskDef*) task->getNode();
                TupleDef *envTuple = taskDef->getEnvTuple();
		if (envTuple->getComponent(arrayName) == NULL) {
			ReportError::InvalidInitArg(arg2->GetLocation(), envType->getName(), arrayName, ignoreFailure);
		}		
	}

	Expr *arg3 = arguments->Nth(2);
	arg3->resolveType(scope, ignoreFailure);
	Type *arg3Type = arg3->getType();
	if (arg3Type == NULL) {
		ReportError::UnknownExpressionType(arg3, ignoreFailure);	
	} else if (arg3Type != Type::stringType && arg3Type != Type::errorType) {
		ReportError::IncompatibleTypes(arg3->GetLocation(), arg3Type, Type::stringType, ignoreFailure);
	}

	this->type = Type::voidType;	
}

void BindOperation::inferType(Scope *scope, Type *type) {
	if (arguments->NumElements() == 3) {
		arguments->Nth(2)->inferType(scope, Type::stringType);
		arguments->Nth(2)->resolveType(scope, false);
	}
}

//------------------------------------------------------- Bind Input ------------------------------------------------------/

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

//------------------------------------------------------- Bind Output -----------------------------------------------------/

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
