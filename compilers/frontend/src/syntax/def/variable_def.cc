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

//----------------------------------------- Variable Definition ------------------------------------------/

VariableDef::VariableDef(Identifier *i, Type *t) : Definition(*i->GetLocation()) {
        Assert(i != NULL && t != NULL);
        id = i; type = t;
        id->SetParent(this);
        type->SetParent(this);
        reduction = false;
}

VariableDef::VariableDef(Identifier *i) : Definition(*i->GetLocation()) {
        Assert(i != NULL);
        id = i; type = NULL;
        id->SetParent(this);
        reduction = false;
}

void VariableDef::PrintChildren(int indentLevel) {
        id->Print(indentLevel + 1);
        if (type != NULL) type->Print(indentLevel + 1);
}

List<VariableDef*> *VariableDef::DecomposeDefs(List<Identifier*> *idList, Type *type) {
        List<VariableDef*> *varList = new List<VariableDef*>;
        for (int i = 0; i < idList->NumElements(); i++) {
           varList->Append(new VariableDef(idList->Nth(i), type));
        }
        return varList;
}

Node *VariableDef::clone() {
	Identifier *newId = (Identifier*) id->clone();
	if (type == NULL) {
		VariableDef *newDef = new VariableDef(newId);
		if (reduction) {
			newDef->flagAsReduction();
		}
		return newDef;
	}
	Type *newType = (Type*) type->clone();
	VariableDef *newDef = new VariableDef(newId, newType);
	if (reduction) {
		newDef->flagAsReduction();
	}
	return newDef;
}

void VariableDef::validateScope(Scope *parentScope) {

        ArrayType *arrayType = dynamic_cast<ArrayType*>(type);
        ListType *listType = dynamic_cast<ListType*>(type);
        if (arrayType != NULL) {
                Type *elementType = arrayType->getTerminalElementType();
                if (parentScope->lookup(elementType->getName()) == NULL) {
                        ReportError::UndeclaredTypeError(id, elementType, "array element ", false);
                }
        } else if (listType != NULL) {
                Type *elementType = listType->getTerminalElementType();
                if (parentScope->lookup(elementType->getName()) == NULL) {
                        ReportError::UndeclaredTypeError(id, elementType, "list element ", false);
                }
        } else if (parentScope->lookup(type->getName()) == NULL) {
                ReportError::UndeclaredTypeError(id, type, NULL, false);
        }
}

