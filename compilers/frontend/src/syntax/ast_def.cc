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
	if (type == VALUE_TYPE) {
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
	instanceList = new List<FunctionInstance*>;
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

Type *FunctionDef::resolveFnInstanceForParameterTypes(Scope *programScope, 
		List<Type*> *paramTypes, Identifier *callerId) {

	if (arguments->NumElements() != paramTypes->NumElements()) {
		ReportError::TooFewOrTooManyParameters(callerId, paramTypes->NumElements(),
				arguments->NumElements(), false);
		return Type::errorType;
	}
	
	// check for cycle in the function type resolution process
	const char *fnName = id->getName();
	for (int i = 0; i < FunctionInstance::fnNameStack->NumElements(); i++) {
		const char *nameInStack = FunctionInstance::fnNameStack->Nth(i);
		if (strcmp(fnName, nameInStack) == 0) {
			ReportError::CyclicalFnCalls(callerId->GetLocation(), fnName, false);	
			return Type::errorType;
		}
	}

	// check if the function has been resolved already for the specific parameter type combination
	FunctionInstance *fnInstance = getInstanceForParamTypes(paramTypes);
	if (fnInstance != NULL) return fnInstance->getReturnType();

	// try to create a new function instance for the argument types
	int instanceId = instanceList->NumElements();
	fnInstance = new FunctionInstance(this, instanceId, paramTypes, programScope);

	// put the function instance in function call stack then do the scope-and-type checking
	FunctionInstance::fnNameStack->Append(fnName);
	FunctionInstance::fnInstanceStack->Append(fnInstance);
	fnInstance->performScopeAndTypeChecking(programScope);

	// remove the function instance from the stack after scope-and-type checking is complete
	int stackSize = FunctionInstance::fnNameStack->NumElements();
	FunctionInstance::fnNameStack->RemoveAt(stackSize - 1);
	FunctionInstance::fnInstanceStack->RemoveAt(stackSize - 1);

	// only store the function instance if the return type is not in error
	if (fnInstance->getReturnType() != Type::errorType) {
		instanceList->Append(fnInstance);
	} else {
		ReportError::CouldNotResolveFnForArgs(callerId->GetLocation(), fnName, false);	
	}

	return fnInstance->getReturnType();		
}

FunctionInstance *FunctionDef::getInstanceForParamTypes(List<Type*> *paramTypes) {
	for (int i = 0; i < instanceList->NumElements(); i++) {
		FunctionInstance *instance = instanceList->Nth(i);
		if (instance->isMatchingArguments(paramTypes)) return instance;
	}
	return NULL;	
}

//---------------------------------------- Function Instance ------------------------------------------/

List<const char*> *FunctionInstance::fnNameStack = new List<const char*>;
List<FunctionInstance*> *FunctionInstance::fnInstanceStack = new List<FunctionInstance*>;

FunctionInstance::FunctionInstance(FunctionDef *fnDef, 
		int instanceId, List<Type*> *argTypes, Scope *programScope) {

	// generate a unique name for the function instance
	std::ostringstream stream;
	stream << fnDef->getId()->getName() << "_" << instanceId;
	this->fnName = strdup(stream.str().c_str());
	
	// get the argument names from the original function definition and assign them types
	this->arguments = fnDef->getArguments();
	this->argumentTypes = argTypes;
	
	this->returnType = NULL;

	// create a complete clone of the code for instance specific analysis later
	this->code = (Stmt*) fnDef->getCode()->clone();

	// do scope and type checking to resolve types of expressions in the body of the code
	performScopeAndTypeChecking(programScope);	
}

void FunctionInstance::performScopeAndTypeChecking(Scope *programScope) {
	
	// create a new parameter scope for the function and insert arguments as symbols in it
	fnHeaderScope = new Scope(FunctionScope);
        for (int i = 0; i < arguments->NumElements(); i++) {
		Identifier *argName = arguments->Nth(i)->getName();
		Type *argType = argumentTypes->Nth(i);	
		VariableDef *var = new VariableDef(argName, argType);
                VariableSymbol *varSym = new VariableSymbol(var);
                fnHeaderScope->insert_symbol(varSym);
        }

	// do a scope-and-type analysis of the function body
	Scope *parameterScope = programScope->enter_scope(fnHeaderScope);
	Scope *executionScope = parameterScope->enter_scope(new Scope(FunctionBodyScope));

	// the scope and type analysis should repeat as long as we resolve new expression types
	int iteration = 0;
	int resolvedTypes = 0;
	do {
		resolvedTypes = code->resolveExprTypesAndScopes(executionScope, iteration);
		iteration++;
	} while (resolvedTypes != 0);


	// check if the function has any un-resolved or erroneous expression; if there is any
	// expression in error then the return type should be invalid
	int errorCount = code->emitScopeAndTypeErrors(executionScope);
	if (errorCount > 0) {
		this->returnType = Type::errorType;
	} else {
		// if the function instance return type is NULL after scope-and-type resolution
		// then the function definition has no return statement in it
		if (this->returnType == NULL) {
			this->returnType = Type::voidType;
		}
	}
	
	// store the function body scope in the instance (after breacking the parent scope links)
	parameterScope->detach_from_parent();
	executionScope->detach_from_parent();
	fnBodyScope = executionScope;
}

bool FunctionInstance::isMatchingArguments(List<Type*> *argTypeList) {
	for (int i = 0; i < argumentTypes->NumElements(); i++) {
		Type *actualType = argTypeList->Nth(i);
		Type *expectedType = argumentTypes->Nth(i);
		if (!actualType->isEqual(expectedType)) return false;
	}
	return true;
}

FunctionInstance *FunctionInstance::getMostRecentFunction() {
	int stackSize = FunctionInstance::fnInstanceStack->NumElements();
	if (stackSize == 0) return NULL;
	return FunctionInstance::fnInstanceStack->Nth(stackSize - 1);
}
