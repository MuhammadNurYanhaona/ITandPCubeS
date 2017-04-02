#include "../ast.h"
#include "../ast_type.h"
#include "../ast_def.h"
#include "../ast_stmt.h"
#include "../../common/errors.h"
#include "../../semantics/scope.h"
#include "../../semantics/symbol.h"
#include "../../codegen-helper/extern_config.h"
#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/hashtable.h"
#include "../../../../common-libs/utils/string_utils.h"

#include <fstream>
#include <sstream>
#include <cstdlib>
#include <iostream>

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

IncludesAndLinksMap *FunctionDef::getExternBlocksHeadersAndLibraries() {
	IncludesAndLinksMap *configMap = new IncludesAndLinksMap();
	code->retrieveExternHeaderAndLibraries(configMap);
	return configMap;		
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
