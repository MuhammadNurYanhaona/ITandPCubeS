#include "ast.h"
#include "ast_type.h"
#include "ast_def.h"
#include "ast_stmt.h"
#include "../semantics/scope.h"
#include "../semantics/symbol.h"
#include "../common/errors.h"
#include "../../../common-libs/utils/list.h"
#include "../../../common-libs/utils/string_utils.h"

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

Node *ProgramDef::clone() {	
	List<Node*> *newCompList = new List<Node*>;
	for (int i = 0; i < components->NumElements(); i++) {
                Node *node = components->Nth(i);
		Node *newNode = node->clone();
		newCompList->Append(newNode);
	}
	ProgramDef *newDef = new ProgramDef(newCompList);
	return newDef;
}

// note that the sequencing of some supporting function calls within this function is important 
// for the correct operation of semantic analysis
void ProgramDef::performScopeAndTypeChecking() {
	
	//-----------------------------------------define an overall scope for the program and
 
	Scope *scope = new Scope(ProgramScope);

	//--------------------determine the element types and properties of user defined types

        Type::storeBuiltInTypesInScope(scope);
	List<Definition*> *classDefs = getComponentsByType(CLASS_DEF);
	for (int i = 0; i < classDefs->NumElements(); i++) {
		TupleDef *classDef = (TupleDef*) classDefs->Nth(i);
		Identifier *defId = classDef->getId();
                if (scope->lookup(defId->getName()) != NULL) {
                        ReportError::ConflictingDefinition(defId, false);
                } else {
			classDef->attachScope(scope);
		}	
	}
	for (int i = 0; i < classDefs->NumElements(); i++) {
		Definition *classDef = classDefs->Nth(i);
		classDef->validateScope(scope);	
	}

	//------------annotate the use of function arguments inside function bodies with value
	//-----------------------------------------------value or reference type specification
	List<Definition*> *fnDefs = getComponentsByType(FN_DEF);
	for (int i = 0; i < fnDefs->NumElements(); i++) {
		FunctionDef *fnDef = (FunctionDef*) fnDefs->Nth(i);
		fnDef->annotateArgAccessesByType();
	}

	//---------------------create a static function definition map to retrieve definitions
	//----------------------by name for type-inference analysis for any function-call made 
	//--------------------------------------------------------from the rest of the program
	FunctionDef::fnDefMap = new Hashtable<FunctionDef*>;
	for (int i = 0; i < fnDefs->NumElements(); i++) {
		FunctionDef *fnDef = (FunctionDef*) fnDefs->Nth(i);
		const char *fnName = fnDef->getId()->getName();
                if (FunctionDef::fnDefMap->Lookup(fnName) != NULL) {
                        ReportError::ConflictingDefinition(fnDef->getId(), false);
                } else {
			FunctionDef::fnDefMap->Enter(fnName, fnDef);
		}
	}
 
	//-----------------analyze the stages section of the tasks before stage instanciations
	List<Definition*> *taskDefs = getComponentsByType(TASK_DEF);
        for (int i = 0; i < taskDefs->NumElements(); i++) {
                TaskDef *taskDef = (TaskDef*) taskDefs->Nth(i);
		taskDef->analyzeStageDefinitions();
        }

	//-------------------------------------associate the program scope with the definition

	symbol = new Symbol("Program", this);
        symbol->setNestedScope(scope);
}

List<Definition*> *ProgramDef::getComponentsByType(DefTypeId typeId) {

	List<Definition*> *filteredList = new List<Definition*>;
	for (int i = 0; i < components->NumElements(); i++) {
                Definition *def = (Definition*) components->Nth(i);
		if (def->getDefTypeId() == typeId) {
			filteredList->Append(def);
		}
	}
	return filteredList;
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

Node *TupleDef::clone() {
	Identifier *newId = (Identifier*) id->clone();
	List<VariableDef*> *newCompList = new List<VariableDef*>;
        for (int j = 0; j < components->NumElements(); j++) {
                VariableDef *var = components->Nth(j);
		VariableDef *newVar = (VariableDef*) var->clone();
                newCompList->Append(newVar);
        }
	TupleDef *newDef = new TupleDef(newId, newCompList);
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

Node *CoordinatorDef::clone() {
	Identifier *newArg = (Identifier*) argument->clone();
	List<Stmt*> *newCode = new List<Stmt*>;
	for (int i = 0; i < code->NumElements(); i++) {
                Stmt *stmt = code->Nth(i);
		newCode->Append((Stmt*) stmt->clone());
	}
	return new CoordinatorDef(newArg, newCode, *GetLocation());
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

Node *FunctionArg::clone() {
	Identifier *newName = (Identifier*) name->clone();
	return new FunctionArg(newName, type);
}
	
Hashtable<FunctionDef*> *FunctionDef::fnDefMap = NULL;

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

Node *FunctionDef::clone() {
	Identifier *newId = (Identifier*) id->clone();
	List<FunctionArg*> *newArgsList = new List<FunctionArg*>;
	if (arguments != NULL && arguments->NumElements() > 0) {
                for (int i = 0; i < arguments->NumElements(); i++) {
			FunctionArg *arg = arguments->Nth(i);
			newArgsList->Append((FunctionArg*) arg->clone());
		}
	}
	Stmt *newCode = (Stmt*) code->clone();
	return new FunctionDef(newId, newArgsList, newCode);
}

void FunctionDef::annotateArgAccessesByType() {

	// filter the reference type arguments arguments
	List<const char*> *refArgNames = new List<const char*>;
	if (arguments != NULL && arguments->NumElements() > 0) {
                for (int i = 0; i < arguments->NumElements(); i++) {
                        FunctionArg *arg = arguments->Nth(i);
			if (arg->getType() == REFERENCE_TYPE) {
				Identifier *argId = arg->getName();
				refArgNames->Append(argId->getName());
			}
		}
	}

	// identify all field accesses within function body
	List<Expr*> *fieldAccesses = new List<Expr*>;
	code->retrieveExprByType(fieldAccesses, FIELD_ACC);

	// locate the base fields within the field access expressions and match them with ref-arg-list
	for (int i = 0; i < fieldAccesses->NumElements(); i++) {
		FieldAccess *exprField = (FieldAccess*) fieldAccesses->Nth(i);
		FieldAccess *rootField = exprField->getTerminalField();
		if (rootField == NULL) continue;

		// if the field name is found in the reference argument list then flag the field as a 
		// reference access
		const char *fieldName = rootField->getField()->getName();
		if (string_utils::contains(refArgNames, fieldName)) {
			rootField->flagAsReferenceField();
		}
	}	

	delete fieldAccesses;
	delete refArgNames;
}
