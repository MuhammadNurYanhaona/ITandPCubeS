#include "../ast.h"
#include "../ast_stmt.h"
#include "../ast_expr.h"
#include "../ast_type.h"
#include "../../common/errors.h"
#include "../../common/constant.h"
#include "../../semantics/scope.h"
#include "../../semantics/symbol.h"
#include "../../semantics/helper.h"
#include "../../semantics/loop_index.h"
#include "../../semantics/data_access.h"
#include "../../semantics/array_acc_transfrom.h"
#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/hashtable.h"

#include <iostream>
#include <sstream>
#include <cstdlib>

//--------------------------------------------------- Array Access ----------------------------------------------------/

ArrayAccess::ArrayAccess(Expr *b, Expr *i, yyltype loc) : Expr(loc) {
        Assert(b != NULL && i != NULL);
        base = b;
        base->SetParent(this);
        index = i;
        index->SetParent(this);
	fillerIndexAccess = false;
}

void ArrayAccess::PrintChildren(int indentLevel) {
        base->Print(indentLevel + 1, "(Base) ");
        index->Print(indentLevel + 1, "(Index) ");
}

Node *ArrayAccess::clone() {
	Expr *newBase = (Expr*) base->clone();
	Expr *newIndex = (Expr*) index->clone();
	return new ArrayAccess(newBase, newIndex, *GetLocation());
}

void ArrayAccess::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	if (typeId == getExprTypeId()) {
		Expr::retrieveExprByType(exprList, typeId);
	} else {
		base->retrieveExprByType(exprList, typeId);
		index->retrieveExprByType(exprList, typeId);
	}
}

int ArrayAccess::getIndexPosition() {
        ArrayAccess *precedingAccess = dynamic_cast<ArrayAccess*>(base);
        if (precedingAccess != NULL) return precedingAccess->getIndexPosition() + 1;
        return 0;
}

Expr *ArrayAccess::getEndpointOfArrayAccess() {
        ArrayAccess *precedingAccess = dynamic_cast<ArrayAccess*>(base);
        if (precedingAccess != NULL) {
                return precedingAccess->getEndpointOfArrayAccess();
        } else return base;
}

int ArrayAccess::resolveExprTypes(Scope *scope) {

	int resolvedExprs = 0;
	resolvedExprs += base->resolveExprTypesAndScopes(scope);
        Type *baseType = base->getType();
	if (baseType == NULL) return resolvedExprs;

	ArrayType *arrayType = dynamic_cast<ArrayType*>(baseType);
	if (arrayType == NULL) {
		this->type = Type::errorType;
		return resolvedExprs;
	}

	IndexRange *indexRange = dynamic_cast<IndexRange*>(index);
        if (indexRange != NULL) {
		this->type = arrayType;
		resolvedExprs += indexRange->resolveExprTypesAndScopes(scope);
	} else {
		this->type = arrayType->reduceADimension();
		resolvedExprs += index->resolveExprTypesAndScopes(scope);
		resolvedExprs += index->performTypeInference(scope, Type::intType);

		// record the association of the index with any encircling looping range, if applicable
		int position = getIndexPosition();
                FieldAccess *indexField = dynamic_cast<FieldAccess*>(index);
		if (indexField != NULL && indexField->isTerminalField()) {
			const char *indexName = indexField->getBaseVarName();
			const char *arrayName = base->getBaseVarName();
			IndexScope *indexScope = IndexScope::currentScope->getScopeForAssociation(indexName);
			if (indexScope != NULL) {
				IndexArrayAssociation *association = new IndexArrayAssociation(indexName,
						arrayName, position);
				indexScope->saveAssociation(association);
				indexField->markAsIndex();
			}
		}	
	}
	resolvedExprs++;
	return resolvedExprs;
}

int ArrayAccess::emitSemanticErrors(Scope *scope) {
	
	int errors = 0;
	errors += base->emitScopeAndTypeErrors(scope);
        Type *baseType = base->getType();
        if (baseType == NULL) {
		ReportError::InvalidArrayAccess(GetLocation(), NULL, false);
        } else {
                ArrayType *arrayType = dynamic_cast<ArrayType*>(baseType);
                if (arrayType == NULL) {
			ReportError::InvalidArrayAccess(base->GetLocation(), baseType, false);
		}
	}
	errors += index->emitScopeAndTypeErrors(scope);
	return errors;
}

void ArrayAccess::retrieveTerminalFieldAccesses(List<FieldAccess*> *fieldList) {
	base->retrieveTerminalFieldAccesses(fieldList);
	index->retrieveTerminalFieldAccesses(fieldList);
}

void ArrayAccess::performStageParamReplacement(
		Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap,
		Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap) {

	//-------------------------------if this is not an access to a parameter then just move forward
	
	Expr *rootArray = getEndpointOfArrayAccess();
	FieldAccess *rootField = dynamic_cast<FieldAccess*>(rootArray);
	// if the base expression is not a terminal field access of array type then it is definitely
	// not an access to a parameter array part (this can still be an access to a whole array passed
	// as an argument)
	if (rootField == NULL || !rootField->isTerminalField()) {
		base->performStageParamReplacement(nameAdjustmentInstrMap, arrayAccXformInstrMap);	 
		index->performStageParamReplacement(nameAdjustmentInstrMap, arrayAccXformInstrMap);
		return;	
	}
	const char *arrayName = rootField->getField()->getName();
	ParamReplacementConfig *arrayPartReplConfig = arrayAccXformInstrMap->Lookup(arrayName);
	// this is the more likely whole array being passed as an argument case; in that situation, 
	// just name change application on field-access expression will suffice as the transformation
	if (arrayPartReplConfig == NULL) {
		base->performStageParamReplacement(nameAdjustmentInstrMap, arrayAccXformInstrMap);	 
		index->performStageParamReplacement(nameAdjustmentInstrMap, arrayAccXformInstrMap);
		return;	
	}

	//---------------------------------------------------------end of skipping transformation logic 

	// If the current array access is not the final index access of the array then the replacement
	// process has been already completed for the base expression. We only need to do any necessary
	// parameter replacement for the elements in the index expression
	if (!isFinalIndexAccess() && !fillerIndexAccess) {
		index->performStageParamReplacement(nameAdjustmentInstrMap, arrayAccXformInstrMap);
	
	} else {
		// transform the array access expression chain
		ArrayPartConfig *arrayPartConfig = arrayPartReplConfig->getArrayPartConfig();
		arrayPartConfig->transformedAccessToArrayPart(this);		

		// Apply any transformation needed in the index expression. This is done after the 
		// transformation of the array access expression chain as the transformation may change 
		// the index expression and flag the new index as a filler index being added during the 
		// transformation process.
		if (!fillerIndexAccess) {
			index->performStageParamReplacement(nameAdjustmentInstrMap, 
					arrayAccXformInstrMap);
		}
	}

	// Continue the transformation towards the base array. Note that we do this last to have a tail 
	// to head progression in the transformation process.
	base->performStageParamReplacement(nameAdjustmentInstrMap, arrayAccXformInstrMap);	 
}

Hashtable<VariableAccess*> *ArrayAccess::getAccessedGlobalVariables(TaskGlobalReferences *globalReferences) {

        Hashtable<VariableAccess*> *table = base->getAccessedGlobalVariables(globalReferences);
        const char *baseVarName = getBaseVarName();
        FieldAccess *baseField = dynamic_cast<FieldAccess*>(base);
        if (baseField != NULL && baseField->isTerminalField()) {
                VariableAccess *accessLog = table->Lookup(baseVarName);
                if (accessLog != NULL) {
                        accessLog->markContentAccess();
                }
        }
        Hashtable<VariableAccess*> *indexTable = index->getAccessedGlobalVariables(globalReferences);
        Iterator<VariableAccess*> iter = indexTable->GetIterator();
        VariableAccess *indexAccess;
        while ((indexAccess = iter.GetNextValue()) != NULL) {
                if (indexAccess->isMetadataAccessed()) indexAccess->getMetadataAccessFlags()->flagAsRead();
                if(indexAccess->isContentAccessed()) indexAccess->getContentAccessFlags()->flagAsRead();
                if (table->Lookup(indexAccess->getName()) != NULL) {
                        VariableAccess *accessLog = table->Lookup(indexAccess->getName());
                        accessLog->mergeAccessInfo(indexAccess);
                } else {
                        table->Enter(indexAccess->getName(), indexAccess, true);
                }
        }
        return table;
}

bool ArrayAccess::isFinalIndexAccess() {
	if (parent == NULL) return true;
	ArrayAccess *parentArrayAcc = dynamic_cast<ArrayAccess*>(parent);
	return (parentArrayAcc != NULL);
}

void ArrayAccess::setEpochVersions(Space *space, int epoch) {
        base->setEpochVersions(space, epoch);
        index->setEpochVersions(space, 0);
}

