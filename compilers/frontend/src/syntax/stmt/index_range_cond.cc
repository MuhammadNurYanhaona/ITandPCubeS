#include "../ast.h"
#include "../ast_stmt.h"
#include "../ast_expr.h"
#include "../ast_def.h"
#include "../ast_task.h"
#include "../../common/errors.h"
#include "../../common/location.h"
#include "../../semantics/scope.h"
#include "../../semantics/symbol.h"
#include "../../semantics/helper.h"
#include "../../semantics/array_acc_transfrom.h"
#include "../../../../common-libs/utils/list.h"

#include <iostream>
#include <sstream>
#include <cstdlib>

//-------------------------------------------------------- Index Range Condition -------------------------------------------------------/

IndexRangeCondition::IndexRangeCondition(List<Identifier*> *i, Identifier *c,
                int dim, Expr *rs, yyltype loc) : Node(loc) {
        Assert(i != NULL && c != NULL);
        indexes = i;
        for (int j = 0; j < indexes->NumElements(); j++) {
                indexes->Nth(j)->SetParent(this);
        }
        collection = c;
        collection->SetParent(this);
        restrictions = rs;
        if (restrictions != NULL) {
                restrictions->SetParent(this);
        }
        this->dimensionNo = dim - 1;
}

void IndexRangeCondition::PrintChildren(int indentLevel) {
        indexes->PrintAll(indentLevel + 1, "(Index) ");
        collection->Print(indentLevel + 1, "(Array/List) ");
        if (restrictions != NULL) restrictions->Print(indentLevel + 1, "(Restrictions) ");
}

Node *IndexRangeCondition::clone() {
	List<Identifier*> *newIndexes = new List<Identifier*>;
	for (int j = 0; j < indexes->NumElements(); j++) {
                Identifier *index = indexes->Nth(j);
		newIndexes->Append((Identifier*) index->clone());
        }
	Identifier *newColl = (Identifier*) collection->clone();
	Expr *newRestr = NULL;
	if (restrictions != NULL) {
		newRestr = (Expr*) restrictions->clone();
	}
	return new IndexRangeCondition(newIndexes, newColl, dimensionNo, newRestr, *GetLocation());
}

void IndexRangeCondition::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	if (restrictions != NULL) restrictions->retrieveExprByType(exprList, typeId);
}

int IndexRangeCondition::resolveExprTypesAndScopes(Scope *executionScope, int iteration) {
	if (iteration == 0) {
		for (int i = 0; i < indexes->NumElements(); i++) {
			Identifier *ind = indexes->Nth(i);
			const char* indexName = ind->getName();
			if (executionScope->lookup(indexName) != NULL) {
				ReportError::ConflictingDefinition(ind, false);
			} else {
				// The current version of the compiler resolves indexes as integer types as 
				// opposed to IndexType that support non-unit stepping and wrapped around
				// index range traversal. This is so since we have not enabled those features
				// in the language yet.
				VariableDef *variable = new VariableDef(ind, Type::intType);
				executionScope->insert_symbol(new VariableSymbol(variable));
			}
		}
	}
	if (restrictions != NULL) {
		int resolvedExprs = restrictions->resolveExprTypesAndScopes(executionScope, iteration);
		int inferredTypes = restrictions->performTypeInference(executionScope, Type::boolType);
		resolvedExprs += inferredTypes;
		return resolvedExprs;
	}
	return 0;		
}

int IndexRangeCondition::emitScopeAndTypeErrors(Scope *scope) {

	int errors = 0;

	// make sure that the iteration is happening over the indices an array
	Symbol *colSymbol = scope->lookup(collection->getName());
        if (colSymbol == NULL) {
                ReportError::UndefinedSymbol(collection, false);
		errors++;
        } else {
                VariableSymbol *varSym = (VariableSymbol*) colSymbol;
                Type *varType = varSym->getType();
                ArrayType *arrayType = dynamic_cast<ArrayType*>(varType);
                if (arrayType == NULL) {
                        ReportError::NonArrayInIndexedIteration(collection, varType, false);
			errors++;
                }
        }

	// if there is any additional iteration restriction, it must be of boolean type
	if (restrictions != NULL) {
		Type *condType = restrictions->getType();
		if (condType != NULL && condType != Type::boolType) {
			ReportError::InvalidExprType(restrictions, Type::boolType, false);
			errors++;
		} else {
			errors += restrictions->emitScopeAndTypeErrors(scope);
		}
	}

	return errors;
}

void IndexRangeCondition::performStageParamReplacement(
		Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap,
		Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap) {

	if (restrictions != NULL) {
		restrictions->performStageParamReplacement(nameAdjustmentInstrMap, arrayAccXformInstrMap);
	}
	
	const char *collectionName = collection->getName();
	ParamReplacementConfig *nameReplConfig = nameAdjustmentInstrMap->Lookup(collectionName);
	if (nameReplConfig != NULL) {
		Expr *argument = nameReplConfig->getInvokingArg();

		// get argument type and check if it is a dynamic array
                Type *type = argument->getType();
                ArrayType *array = dynamic_cast<ArrayType*>(type);
                StaticArrayType *staticArray = dynamic_cast<StaticArrayType*>(type);
                bool dynamicType = (array != NULL) && (staticArray == NULL);

		if (!dynamicType) {
			ReportError::NonArrayInIndexedIteration(collection, type, false);
			return;
		}
		
		FieldAccess *field = (FieldAccess*) argument;
		const char *argArrayName = field->getField()->getName();
		collection->changeName(argArrayName);
		return;		
	}

	ParamReplacementConfig *arrayAccReplConfig = arrayAccXformInstrMap->Lookup(collectionName);
	if (arrayAccReplConfig != NULL) {
		ArrayPartConfig *arrayPartConfig = arrayAccReplConfig->getArrayPartConfig();
		FieldAccess *baseArrayAcc = arrayPartConfig->getBaseArrayAccess();
		const char *argArrayName = baseArrayAcc->getField()->getName();
		collection->changeName(argArrayName);

		// if there is any explicit mentioning of dimension then the part dimension number should be
		// updated to point to the proper dimension in the original array 
		if (dimensionNo >= 0) {
			dimensionNo = arrayPartConfig->getOrigDimension(dimensionNo);
		}	
	} 
}

