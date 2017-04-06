#include "../../../utils/code_constant.h"
#include "../../../utils/name_transformer.h"
#include "../../../../../../common-libs/utils/decorator_utils.h"
#include "../../../../../../common-libs/domain-obj/constant.h"
#include "../../../../../../frontend/src/syntax/ast_def.h"
#include "../../../../../../frontend/src/syntax/ast_stmt.h"
#include "../../../../../../frontend/src/semantics/scope.h"
#include "../../../../../../frontend/src/semantics/data_access.h"

#include <fstream>
#include <sstream>

void FunctionInstance::generateCode(std::ofstream &headerFile, std::ofstream &programFile) {

	//----------------------------------------------------------generate function header

	std::ostringstream fnHeader;
	fnHeader << returnType->getCType() << " " << fnName << "(";
	for (int i = 0; i < arguments->NumElements(); i++) {
		if (i > 0) fnHeader << paramSeparator;
		FunctionArg *arg = arguments->Nth(i);
		Type *argType = argumentTypes->Nth(i);
		fnHeader << argType->getCType() << " ";
		if (arg->getType() == REFERENCE_TYPE) fnHeader << "*";
		fnHeader << arg->getName()->getName();
	}
	fnHeader << ")";	

	//------------------------------------------------------------generate function body

	// reset the name transformer to avoid spill over of logic from task generation
        ntransform::NameTransformer::transformer->reset();
	
	std::ostringstream fnBody;
	fnBody << "{\n";

	// declare local variables
	decorator::writeCommentHeader(1, &fnBody, "Local Variable Declarations");
	fnBodyScope->declareVariables(fnBody, 1);
	fnBody << "\n";

	// translate the code section
	decorator::writeCommentHeader(1, &fnBody, "Function Body");
	code->generateCode(fnBody, 1);

	fnBody << "}\n";
	
	//----------------------------------Write declaration and definition in output files

	headerFile << fnHeader.str() << stmtSeparator;
	programFile << fnHeader.str() << " " << fnBody.str() << "\n";
}

