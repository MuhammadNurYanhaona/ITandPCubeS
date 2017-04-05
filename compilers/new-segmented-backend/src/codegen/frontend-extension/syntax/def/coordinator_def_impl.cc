#include "../../../utils/array_assignment.h"
#include "../../../../../../frontend/src/syntax/ast_def.h"
#include "../../../../../../frontend/src/syntax/ast_stmt.h"
#include "../../../../../../frontend/src/semantics/scope.h"
#include "../../../../../../frontend/src/semantics/data_access.h"

void CoordinatorDef::declareVariablesInScope(std::ostringstream &stream, int indent) {
        executionScope->declareVariables(stream, indent);
}

void CoordinatorDef::generateCode(std::ostringstream &stream, Scope *scope) {

        Scope *execScope = scope->enter_scope(executionScope);

        // First perform an variable access analysis to set up the flags on fields properly 
        // concerning whether they represent accessing metadata or data of a variable. TODO 
        // this is a patch solution to use variable access analysis here in this manner. We 
        // should refactor the logic in the future.
        TaskGlobalReferences *references = new TaskGlobalReferences(execScope);
	code->getAccessedGlobalVariables(references);

        // set the context for code translation to coordinator function
        codecntx::enterCoordinatorContext();

        // Then generate code.
        stream << "\t//------------------------------------------ Coordinator Program\n\n";
	code->generateCode(stream, 1);
        stream << "\n\t//--------------------------------------------------------------\n";	
}

