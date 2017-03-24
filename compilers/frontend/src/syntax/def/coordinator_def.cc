#include "../ast.h"
#include "../ast_type.h"
#include "../ast_def.h"
#include "../ast_stmt.h"
#include "../../semantics/scope.h"
#include "../../semantics/symbol.h"
#include "../../common/errors.h"
#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/string_utils.h"

#include <fstream>
#include <sstream>
#include <cstdlib>
#include <iostream>

//----------------------------------- Coordinator/Main Definition -------------------------------------/

CoordinatorDef::CoordinatorDef(Identifier *a, List<Stmt*> *c, yyltype loc) : Definition(loc) {
        
	Assert(a != NULL && c != NULL);
        argument = a;
        code = new StmtBlock(c);
	code->SetParent(this);
        argument->SetParent(this);

	argumentTuple = NULL;
	executionScope = NULL;
}

void CoordinatorDef::PrintChildren(int indentLevel) {
        if (argument != NULL) argument->Print(indentLevel + 1, "(Argument) ");
        PrintLabel(indentLevel + 1, "Code");
        code->Print(indentLevel + 2);
}

Node *CoordinatorDef::clone() {
	Identifier *newArg = (Identifier*) argument->clone();
	Stmt *newCode = (Stmt*) code->clone();
	List<Stmt*> *codeList = new List<Stmt*>;
	codeList->Append(newCode);
	return new CoordinatorDef(newArg, codeList, *GetLocation());
}

void CoordinatorDef::validateScope(Scope *parentScope) {

        // Create a scope for holding the command line arguments
        Scope *scope = parentScope->enter_scope(new Scope(ProgramScope));
        VariableDef *commandLineArg = new VariableDef(argument, new MapType(*argument->GetLocation()));
        scope->insert_symbol(new VariableSymbol(commandLineArg));

        // Create the execution scope attached to the command line arguments' scope
        executionScope = scope->enter_scope(new Scope(CoordinatorScope));

	// the scope and type analysis should repeat as long as we resolve new expression types
        int iteration = 0;
        int resolvedTypes = 0;
        do {    resolvedTypes = code->resolveExprTypesAndScopes(executionScope, iteration);
                iteration++;
        } while (resolvedTypes != 0);

        // emit all scope and type errors, if exist
        code->emitScopeAndTypeErrors(executionScope);	

        // generate the definition for a structure to hold the arguments for the coordinator function
        MapType *argumentMap = (MapType*) commandLineArg->getType();
        List<VariableDef*> *argumentList = argumentMap->getElementList();
        Identifier *tupleId = new Identifier(*argument->GetLocation(), "ProgramArgs");
        argumentTuple = new TupleDef(tupleId, argumentList);
}
