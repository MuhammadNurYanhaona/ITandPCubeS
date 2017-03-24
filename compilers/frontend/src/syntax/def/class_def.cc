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

//----------------------------------------- Tuple Definition -------------------------------------------/

TupleDef::TupleDef(Identifier *i, List<VariableDef*> *c) : Definition(*i->GetLocation()) {
        Assert(i != NULL && c != NULL);
        id = i;
        components = c;
        id->SetParent(this);
        for (int j = 0; j < components->NumElements(); j++) {
                VariableDef *var = components->Nth(j);
                var->SetParent(this);
        }
	environment = false;
}

void TupleDef::PrintChildren(int indentLevel) {
        id->Print(indentLevel);
        components->PrintAll(indentLevel + 1);
        printf("\n");
}

Node *TupleDef::clone() {
	Identifier *newId = (Identifier*) id->clone();
	List<VariableDef*> *newCompList = new List<VariableDef*>;
        for (int j = 0; j < components->NumElements(); j++) {
                VariableDef *var = components->Nth(j);
		VariableDef *newVar = (VariableDef*) var->clone();
                newCompList->Append(newVar);
        }
	TupleDef *newDef = new TupleDef(newId, newCompList);
	if (environment) newDef->flagAsEnvironment();
	return newDef;
}

void TupleDef::attachScope(Scope *parentScope) {

        Scope *scope = new Scope(TupleScope);
        List<Type*> *elementTypes = new List<Type*>;

        for (int i = 0; i < components->NumElements(); i++) {
                VariableDef *element = components->Nth(i);
                VariableSymbol *varSym = new VariableSymbol(element);
                if (scope->lookup(element->getId()->getName()) != NULL) {
                        ReportError::ConflictingDefinition(element->getId(), false);
                }
                scope->insert_symbol(varSym);
                elementTypes->Append(varSym->getType());
        }

        symbol = new TupleSymbol(id, this, elementTypes);
        symbol->setNestedScope(scope);
        parentScope->insert_symbol(symbol);
}

void TupleDef::validateScope(Scope *parentScope) {
        for (int i = 0; i < components->NumElements(); i++) {
                VariableDef *element = components->Nth(i);
                element->validateScope(parentScope);
        }
}

VariableDef *TupleDef::getComponent(const char *name) {
        for (int i = 0; i < components->NumElements(); i++) {
                VariableDef *element = components->Nth(i);
                if (strcmp(element->getId()->getName(), name) == 0) return element;
        }
        return NULL;
}

