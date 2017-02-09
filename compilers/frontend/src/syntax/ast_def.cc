#include "ast.h"
#include "ast_type.h"
#include "ast_def.h"
#include "ast_stmt.h"
#include "../common/errors.h"
#include "../../../common-libs/utils/list.h"

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


//----------------------------------------- Program Definition ------------------------------------------/

ProgramDef *ProgramDef::program = NULL;

ProgramDef::ProgramDef(List<Node*> *c) : Definition() {
        Assert(c != NULL);
        components = c;
        for (int i = 0; i < components->NumElements(); i++) {
                Node *node = components->Nth(i);
                node->SetParent(this);
        }
}

void ProgramDef::PrintChildren(int indentLevel) {
        components->PrintAll(indentLevel+1);
        printf("\n");
}

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
}

void TupleDef::PrintChildren(int indentLevel) {
        id->Print(indentLevel);
        components->PrintAll(indentLevel + 1);
        printf("\n");
}

//----------------------------------- Coordinator/Main Definition -------------------------------------/

CoordinatorDef::CoordinatorDef(Identifier *a, List<Stmt*> *c, yyltype loc) : Definition(loc) {
        Assert(a != NULL && c != NULL);
        argument = a;
        code = c;
        argument->SetParent(this);
        for (int i = 0; i < code->NumElements(); i++) {
                Stmt *stmt = code->Nth(i);
                stmt->SetParent(this);
        }
}

void CoordinatorDef::PrintChildren(int indentLevel) {
        if (argument != NULL) argument->Print(indentLevel + 1, "(Argument) ");
        PrintLabel(indentLevel + 1, "Code");
        code->PrintAll(indentLevel + 2);
}

//--------------------------------------- Function Definition -----------------------------------------/

FunctionArg::FunctionArg(Identifier *name, ArgumentType type) {
	Assert(name != NULL);
	this->name = name;
	name->SetParent(this);
	this->type = type;
}
        
void FunctionArg::PrintChildren(int indentLevel) {
	if (type = VALUE_TYPE) {
		PrintLabel(indentLevel + 1, "Reference Arg:");
	} else {
        	PrintLabel(indentLevel + 1, "Value Arg: ");
	}
	name->Print(0);
}

FunctionDef::FunctionDef(Identifier *id, List<FunctionArg*> *arguments, Stmt *code) {
	Assert(id != NULL && code != NULL);
        this->id = id;
        id->SetParent(this);
        this->code = code;
	code->SetParent(this);
	this->arguments = arguments;
	if (arguments != NULL && arguments->NumElements() > 0) {
		for (int i = 0; i < arguments->NumElements(); i++) {
			arguments->Nth(i)->SetParent(this);
		}
	}
}
        
void FunctionDef::PrintChildren(int indentLevel) {
	id->Print(indentLevel + 1, "(Name) ");
	if (arguments != NULL && arguments->NumElements() > 0) {
		PrintLabel(indentLevel + 1, "Arguments: ");
        	arguments->PrintAll(indentLevel + 2);
	}
        code->Print(indentLevel + 1, "(Code) ");
}
