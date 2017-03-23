#include "../ast.h"
#include "../ast_expr.h"
#include "../ast_type.h"
#include "../ast_library_fn.h"
#include "../../common/errors.h"
#include "../../common/constant.h"
#include "../../semantics/scope.h"
#include "../../semantics/symbol.h"
#include "../../../../common-libs/utils/list.h"

#include <iostream>

//-------------------------------------------------------- Static Constants -----------------------------------------------------/

const char *Root::Name = "root";
const char *Random::Name = "random";
const char *LoadArray::Name = "load_array";
const char *StoreArray::Name = "store_array";
const char *BindInput::Name = "bind_input";
const char *BindOutput::Name = "bind_output";

//-------------------------------------------------------- Library Function -----------------------------------------------------/

LibraryFunction::LibraryFunction(int argumentCount, Identifier *functionName,
                List<Expr*> *arguments, yyltype loc) : Expr(loc) {
        this->argumentCount = argumentCount;
        this->functionName = functionName;
        this->arguments = arguments;
}

LibraryFunction::LibraryFunction(int argumentCount, Identifier *functionName, List<Expr*> *arguments) : Expr() {
        this->argumentCount = argumentCount;
        this->functionName = functionName;
        this->arguments = arguments;
}

bool LibraryFunction::isLibraryFunction(Identifier *id) {
        const char* name = id->getName();
        return (strcmp(name, Root::Name) == 0 
		|| strcmp(name, Random::Name) == 0
                || strcmp(name, LoadArray::Name) == 0 
                || strcmp(name, StoreArray::Name) == 0
                || strcmp(name, BindInput::Name) == 0
                || strcmp(name, BindOutput::Name) == 0);
}

LibraryFunction *LibraryFunction::getFunctionExpr(Identifier *id, List<Expr*> *arguments, yyltype loc) {

        const char* name = id->getName();
        LibraryFunction *function = NULL;

        // note that there should never be a default 'else' block here; then the system will fail to find user 
	// defined functions
        if (strcmp(name, Root::Name) == 0) {
                function = new Root(id, arguments, loc);
        } else if (strcmp(name, Random::Name) == 0) {
                function = new Random(id, arguments, loc);
        } else if (strcmp(name, LoadArray::Name) == 0) {
                function = new LoadArray(id, arguments, loc);
        } else if (strcmp(name, StoreArray::Name) == 0) {
                function = new StoreArray(id, arguments, loc);
        } else if (strcmp(name, BindInput::Name) == 0) {
                function = new BindInput(id, arguments, loc);
        } else if (strcmp(name, BindOutput::Name) == 0) {
                function = new BindOutput(id, arguments, loc);
        }

        return function;
}

void LibraryFunction::PrintChildren(int indentLevel) {
        PrintLabel(indentLevel + 1, "Arguments");
        arguments->PrintAll(indentLevel + 2);
}

Node *LibraryFunction::clone() {
	Identifier *newId = (Identifier*) functionName->clone();
	List<Expr*> *newArgsList = new List<Expr*>;
	for (int i = 0; i < arguments->NumElements(); i++) {
		Expr *arg = arguments->Nth(i);
		Expr *newArg = (Expr*) arg->clone();
		newArgsList->Append(newArg);
	}
	return getFunctionExpr(newId, newArgsList, *GetLocation());
}

void LibraryFunction::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	Expr::retrieveExprByType(exprList, typeId);
	for (int i = 0; i < arguments->NumElements(); i++) {
                Expr *arg = arguments->Nth(i);
		arg->retrieveExprByType(exprList, typeId);
	}
}

int LibraryFunction::emitSemanticErrors(Scope *scope) {
	if (argumentCount != arguments->NumElements()) {
                std::cout << "argument count problem\n";
                ReportError::TooFewOrTooManyParameters(functionName, arguments->NumElements(),
                                argumentCount, false);
		return 1;
        } else {
                return emitErrorsInArguments(scope);
        }
}

void LibraryFunction::performStageParamReplacement(
		Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap,
		Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap) {
	
	for (int i = 0; i < arguments->NumElements(); i++) {
                Expr *arg = arguments->Nth(i);
		arg->performStageParamReplacement(nameAdjustmentInstrMap, arrayAccXformInstrMap);
	}
}

//-------------------------------------------------------------- Root -----------------------------------------------------------/

int Root::resolveExprTypes(Scope *scope) {

	int resolvedExprs = 0;
	Expr *arg1 = arguments->Nth(0);
        Expr *arg2 = arguments->Nth(1);
        resolvedExprs += arg1->resolveExprTypesAndScopes(scope);
        resolvedExprs += arg2->resolveExprTypesAndScopes(scope);
	resolvedExprs += arg2->performTypeInference(scope, Type::intType);

	Type *arg1Type = arg1->getType();
	if (arg1Type != NULL && arg2->getType() != NULL) {
		this->type = arg1Type;
		resolvedExprs++;
	}
	return resolvedExprs;
}

int Root::inferExprTypes(Scope *scope, Type *assignedType) {
	this->type = assignedType;
	int resolvedExprs = 1;
	Expr *arg1 = arguments->Nth(0);
	resolvedExprs += arg1->performTypeInference(scope, assignedType);
	return resolvedExprs;
}

int Root::emitErrorsInArguments(Scope *scope) {

	int errors = 0;
	Expr *arg1 = arguments->Nth(0);
	errors += arg1->emitScopeAndTypeErrors(scope);
        Type *arg1Type = arg1->getType();
        if (arg1Type != NULL && arg1Type != Type::intType
                        && arg1Type != Type::floatType
                        && arg1Type != Type::doubleType
                        && arg1Type != Type::errorType) {
                ReportError::InvalidExprType(arg1, arg1Type, false);
		errors++;
        }

        Expr *arg2 = arguments->Nth(1);
	errors += arg2->emitScopeAndTypeErrors(scope);
        Type *arg2Type = arg2->getType();
        if (arg2Type != NULL && arg2Type != Type::intType 
			&& arg2Type != Type::errorType) {
                ReportError::IncompatibleTypes(arg2->GetLocation(), 
			arg2Type, Type::intType, false);
		errors++;
        }

	return errors;
}

//--------------------------------------------------------- Array Operation -----------------------------------------------------/

int ArrayOperation::resolveExprTypes(Scope *scope) {

	int resolvedExprs = 0;
	Expr *arg1 = arguments->Nth(0);
        resolvedExprs += arg1->resolveExprTypesAndScopes(scope);
        Type *arg1Type = arg1->getType();

	Expr *arg2 = arguments->Nth(1);
        resolvedExprs += arg2->resolveExprTypesAndScopes(scope);
	resolvedExprs += arg2->performTypeInference(scope, Type::stringType);
        Type *arg2Type = arg2->getType();

	if (arg1Type != NULL && arg2Type != NULL) {
		this->type = Type::voidType;
		resolvedExprs++;
	}
	return resolvedExprs;
}

int ArrayOperation::emitErrorsInArguments(Scope *scope) {
	
	int errors = 0;
	Expr *arg1 = arguments->Nth(0);
	errors += arg1->emitScopeAndTypeErrors(scope);
        Type *arg1Type = arg1->getType();
        if (arg1Type != NULL) {
                ArrayType *arrayType = dynamic_cast<ArrayType*>(arg1Type);
                if (arrayType == NULL) {
                        ReportError::InvalidArrayAccess(arg1->GetLocation(), arg1Type, false);
			errors++;
                }
        }

        Expr *arg2 = arguments->Nth(1);
	errors += arg2->emitScopeAndTypeErrors(scope);
        Type *arg2Type = arg2->getType();
        if (arg2Type != NULL && arg2Type != Type::stringType && arg2Type != Type::errorType) {
                ReportError::IncompatibleTypes(arg2->GetLocation(), 
			arg2Type, Type::stringType, false);
		errors++;
        }
        return errors;
}

//--------------------------------------------------------- Bind Operation ------------------------------------------------------/

int BindOperation::resolveExprTypes(Scope *scope) {

	int resolvedExprs = 0;
	Expr *arg1 = arguments->Nth(0);	
	resolvedExprs += arg1->resolveExprTypesAndScopes(scope);

	// identify the task environment type from the first argument
        Type *arg1Type = arg1->getType();
	NamedType *envType = NULL;
	if (arg1Type != NULL) {
                NamedType *objectType = dynamic_cast<NamedType*>(arg1Type);
                if (objectType != NULL && !objectType->isEnvironmentType()) {
                        envType = objectType;
                }
        }

	Expr *arg2 = arguments->Nth(1);
	resolvedExprs += arg2->resolveExprTypesAndScopes(scope);
	resolvedExprs += arg2->performTypeInference(scope, Type::stringType);
	Type *arg2Type = arg2->getType();

	Expr *arg3 = arguments->Nth(2);
	resolvedExprs += arg3->resolveExprTypesAndScopes(scope);
	resolvedExprs += arg3->performTypeInference(scope, Type::stringType);
	Type *arg3Type = arg3->getType();

	if (arg1Type != NULL && arg2Type != NULL && arg3Type != NULL) {
		this->type = Type::voidType;
		resolvedExprs++;
	}

	return resolvedExprs;
}

int BindOperation::emitErrorsInArguments(Scope *scope) {
	
	int errors = 0;
	NamedType *envType = NULL;
        Expr *arg1 = arguments->Nth(0);
        errors += arg1->emitScopeAndTypeErrors(scope);
        Type *arg1Type = arg1->getType();
        if (arg1Type != NULL) {
                NamedType *objectType = dynamic_cast<NamedType*>(arg1Type);
                if (objectType == NULL || !objectType->isEnvironmentType()) {
                        ReportError::NotAnEnvironment(arg1->GetLocation(), arg1Type, false);
			errors++;
                } else {
                        envType = objectType;
                }
        }

        Expr *arg2 = arguments->Nth(1);
        StringConstant *varName = dynamic_cast<StringConstant*>(arg2);
        if (varName == NULL) {
                ReportError::NotAConstant(arg2->GetLocation(), "string", false);
		errors++;
        } else if (envType != NULL) {
                const char *arrayName = varName->getValue();
                Symbol *symbol = scope->lookup(envType->getTaskName());
                TaskSymbol *task = dynamic_cast<TaskSymbol*>(symbol);
                TaskDef *taskDef = (TaskDef*) task->getNode();
                TupleDef *envTuple = taskDef->getEnvTuple();
                if (envTuple->getComponent(arrayName) == NULL) {
                        ReportError::InvalidInitArg(arg2->GetLocation(), 
					envType->getName(), arrayName, false);
			errors++;
                }
        }

        Expr *arg3 = arguments->Nth(2);
	errors += arg3->emitScopeAndTypeErrors(scope);
        Type *arg3Type = arg3->getType();
        if (arg3Type != NULL && arg3Type != Type::stringType && arg3Type != Type::errorType) {
                ReportError::IncompatibleTypes(arg3->GetLocation(), 
				arg3Type, Type::stringType, false);
		errors++;
        }

	return errors;
}
