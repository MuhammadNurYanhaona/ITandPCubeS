#include "../../../../../../common-libs/domain-obj/constant.h"
#include "../../../../../../frontend/src/syntax/ast_expr.h"
#include "../../../../../../frontend/src/syntax/ast_def.h"
#include "../../../../../../frontend/src/semantics/task_space.h"
#include "../../../utils/code_constant.h"

#include <sstream>
#include <iostream>
#include <cstdlib>

void FunctionCall::translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space) {

	const char *functionName = base->getName();
        FunctionDef *fnDef = FunctionDef::fnDefMap->Lookup(functionName);

	// generate an expression for arguments
	List<FunctionArg*> *fnArgs = fnDef->getArguments();
	List<Type*> *paramTypes = new List<Type*>;
	std::ostringstream argStream;
	for (int i = 0; i < fnArgs->NumElements(); i++) {
		
		if (i > 0) argStream << paramSeparator;
		
		FunctionArg *formalArg = fnArgs->Nth(i);
		if (formalArg->getType() == REFERENCE_TYPE) {
			argStream << "&(";
		}

		// record the argument type for later use in function instance identification
		Expr *actualArg = arguments->Nth(i);
		paramTypes->Append(actualArg->getType());

		// translate the argument expression
		actualArg->translate(argStream, 0, 0, space);

		if (formalArg->getType() == REFERENCE_TYPE) {
			argStream << ")";
		}
	}

	// locate the function instance for the argument type
	FunctionInstance *fnInstance = fnDef->getInstanceForParamTypes(paramTypes);
	
	// generate the function call
	stream << fnInstance->getName() << "(";
	stream << argStream.str() << ")";
}
