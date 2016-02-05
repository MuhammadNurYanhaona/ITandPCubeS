#include "ast.h"
#include "ast_type.h"
#include "ast_def.h"
#include "ast_stmt.h"
#include "list.h"

VariableDef::VariableDef(Identifier *i, Type *t) : Node(*i->GetLocation()) {
	Assert(i != NULL && t != NULL);
	id = i; type = t;
}

void VariableDef::PrintChildren(int indentLevel) {
	id->Print(indentLevel + 1);
        type->Print(indentLevel + 1);
}

List<VariableDef*> *VariableDef::DecomposeDefs(List<Identifier*> *idList, Type *type) {
	List<VariableDef*> *varList = new List<VariableDef*>;
	for (int i = 0; i < idList->NumElements(); i++) {
           varList->Append(new VariableDef(idList->Nth(i), type));
       	}
	return varList;
}

ProgramDef::ProgramDef(List<Node*> *c) {
	Assert(c != NULL);
	components = c;
} 

void ProgramDef::PrintChildren(int indentLevel) {
	components->PrintAll(indentLevel+1);
    	printf("\n");
}

TupleDef::TupleDef(Identifier *i, List<VariableDef*> *c) : Node(*i->GetLocation()) {
	Assert(i != NULL && c != NULL);
	id = i;
	components = c;
}

void TupleDef::PrintChildren(int indentLevel) {
	id->Print(indentLevel);
	components->PrintAll(indentLevel+1);
    	printf("\n");
}

CoordinatorDef::CoordinatorDef(Identifier *a, List<Stmt*> *c, yyltype loc) : Node(loc) {
	Assert(a != NULL && c != NULL);
	argument = a;
	code = c;
}

void CoordinatorDef::PrintChildren(int indentLevel) {
	if (argument != NULL) argument->Print(indentLevel + 1, "(Argument) ");
	PrintLabel(indentLevel + 1, "Code");
	code->PrintAll(indentLevel + 2);
}

FunctionHeader::FunctionHeader(List<VariableDef*> *a, List<VariableDef*> *r) : Node() {
	Assert(a != NULL && r != NULL);
	arguments = a;
	results = r;	
}

void FunctionHeader::PrintChildren(int indentLevel) {
	PrintLabel(indentLevel + 1, "Input Arguments");
	arguments->PrintAll(indentLevel + 2);
	PrintLabel(indentLevel + 1, "Result Elements");	
	results->PrintAll(indentLevel + 2);
}

FunctionDef::FunctionDef(Identifier *i, FunctionHeader *h, List<Stmt*> *c) : Node(*i->GetLocation()) {
	Assert(i != NULL && h != NULL && c != NULL);
	id = i;
	header = h;
	code = c;
}

void FunctionDef::PrintChildren(int indentLevel) {
	id->Print(indentLevel + 1, "(Name) ");
	header->Print(indentLevel + 1);
	PrintLabel(indentLevel + 1, "(Code) ");
	code->PrintAll(indentLevel + 2);
}
