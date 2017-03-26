#include "../ast.h"
#include "../ast_type.h"
#include "../ast_def.h"
#include "../ast_stmt.h"
#include "../ast_type.h"
#include "../ast_task.h"
#include "../ast_partition.h"
#include "../../common/errors.h"
#include "../../common/location.h"
#include "../../common/constant.h"
#include "../../semantics/scope.h"
#include "../../semantics/symbol.h"
#include "../../semantics/helper.h"
#include "../../semantics/data_access.h"
#include "../../semantics/computation_flow.h"
#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/hashtable.h"
#include "../../../../common-libs/utils/string_utils.h"

#include <sstream>

//----------------------------------------------------- Initialize Section --------------------------------------------------------/

InitializeSection::InitializeSection(List<Identifier*> *a, List<Stmt*> *c, yyltype loc) : Node(loc) {
        Assert(a != NULL && c != NULL);
        arguments = a;
        for (int i = 0; i < arguments->NumElements(); i++) {
                arguments->Nth(i)->SetParent(this);
        }
        argumentTypes = NULL;
        code = new StmtBlock(c);
	code->SetParent(this);
}

void InitializeSection::PrintChildren(int indentLevel) {
        if (arguments->NumElements() > 0) {
                PrintLabel(indentLevel + 1, "Arguments");
                arguments->PrintAll(indentLevel + 2);
        }
        PrintLabel(indentLevel + 1, "Code");
        code->Print(indentLevel + 2);
}

void InitializeSection::performScopeAndTypeChecking(Scope *parentScope) {
	
	// Generate a parameter scope for initialize arguments
        Scope *parameterScope = new Scope(TaskInitScope);
        TaskDef *taskDef = (TaskDef*) this->parent;
        Scope *taskDefineScope = taskDef->getSymbol()->getNestedScope();
        for (int i = 0; i < arguments->NumElements(); i++) {
                Identifier *id = arguments->Nth(i);
                if (taskDefineScope->lookup(id->getName()) != NULL) {
                        parameterScope->copy_symbol(taskDefineScope->lookup(id->getName()));
                } else {
                        parameterScope->insert_symbol(new VariableSymbol(id->getName(), NULL));
                }
        }

	// enter to the nested scopes for the task and the init section
        Scope *executionScope  = parentScope->enter_scope(taskDefineScope);
        executionScope = executionScope->enter_scope(parameterScope);

        // create a new scope for the init body (code section)
        Scope *initBodyScope = executionScope->enter_scope(new Scope(TaskInitBodyScope));

	// the scope and type analysis should repeat as long as we resolve new expression types
        int iteration = 0;
        int resolvedTypes = 0;
        do {	resolvedTypes = code->resolveExprTypesAndScopes(initBodyScope, iteration);
                iteration++;
        } while (resolvedTypes != 0);

	// emit all scope and type errors, if exist
	code->emitScopeAndTypeErrors(initBodyScope);
	
	// prepare the scopes for storage
	taskDefineScope->detach_from_parent();
        parameterScope->detach_from_parent();
        initBodyScope->detach_from_parent();

        // save parameter and init body scopes
        TaskSymbol *taskSymbol = (TaskSymbol*) taskDef->getSymbol();
        taskSymbol->setInitScope(parameterScope);
        this->scope = initBodyScope;

        // store the argument types for actual to formal parameter matching
        argumentTypes = new List<Type*>;
        for (int i = 0; i < arguments->NumElements(); i++) {
                Identifier *id = arguments->Nth(i);
                VariableSymbol *symbol = (VariableSymbol*) parameterScope->lookup(id->getName());
                if (symbol->getType() == NULL) {
                        ReportError::TypeInferenceError(id, false);
                } else {
                        argumentTypes->Append(symbol->getType());
                }
        }	
}

void InitializeSection::performVariableAccessAnalysis(Scope *taskGlobalScope) {
        accessMap = new Hashtable<VariableAccess*>;
        for (int i = 0; i < arguments->NumElements(); i++) {
                Identifier *id = arguments->Nth(i);
                if (taskGlobalScope->lookup(id->getName()) != NULL) {
                        VariableAccess *accessLog = new VariableAccess(id->getName());
                        accessLog->markContentAccess();
                        accessLog->getContentAccessFlags()->flagAsWritten();
                        accessMap->Enter(id->getName(), accessLog, true);
                }
        }
        TaskGlobalReferences *references = new TaskGlobalReferences(taskGlobalScope);
	Hashtable<VariableAccess*> *table = code->getAccessedGlobalVariables(references);
	Stmt::mergeAccessedVariables(accessMap, table);
}


