#include "../ast.h"
#include "../ast_stmt.h"
#include "../ast_expr.h"
#include "../ast_type.h"
#include "../../common/errors.h"
#include "../../common/constant.h"
#include "../../semantics/scope.h"
#include "../../semantics/symbol.h"
#include "../../semantics/data_access.h"
#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/hashtable.h"

#include <iostream>
#include <sstream>
#include <cstdlib>

//------------------------------------------------- Index Range -------------------------------------------------------/

IndexRange::IndexRange(Expr *b, Expr *e, bool p, yyltype loc) : Expr(loc) {
        begin = b;
        if (begin != NULL) {
                begin->SetParent(this);
        }
        end = e;
        if (end != NULL) {
                end->SetParent(this);
        }
        fullRange = (b == NULL && e == NULL);
	this->partOfArray = p;
}

void IndexRange::PrintChildren(int indentLevel) {
        if (begin != NULL) begin->Print(indentLevel + 1);
        if (end != NULL) end->Print(indentLevel + 1);
}

Node *IndexRange::clone() {
	Expr *newBegin = NULL;
	Expr *newEnd = NULL;
	if (begin != NULL) newBegin = (Expr*) begin->clone();
	if (end != NULL) newEnd = (Expr*) end->clone();
	return new IndexRange(newBegin, newEnd, partOfArray, *GetLocation());
}

void IndexRange::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	if (typeId == getExprTypeId()) {
		Expr::retrieveExprByType(exprList, typeId);
	} else {
		if (begin != NULL) begin->retrieveExprByType(exprList, typeId);
		if (end != NULL) end->retrieveExprByType(exprList, typeId);
	}
}

int IndexRange::resolveExprTypes(Scope *scope) {
	int resolvedExprs = 0;
	if (begin != NULL) {
		resolvedExprs += begin->resolveExprTypesAndScopes(scope);
		resolvedExprs += begin->performTypeInference(scope, Type::intType);
	}
	if (end != NULL) {
		resolvedExprs += end->resolveExprTypesAndScopes(scope);
		resolvedExprs += end->performTypeInference(scope, Type::intType);
	}
	this->type = (partOfArray) ? Type::voidType : Type::rangeType;
	resolvedExprs++;
	return resolvedExprs;
}

int IndexRange::emitSemanticErrors(Scope *scope) {
	int errors = 0;
	if (begin != NULL) {
		errors += begin->emitScopeAndTypeErrors(scope);
		Type *beginType = begin->getType();
                if (beginType != NULL && !Type::intType->isAssignableFrom(beginType)) {
                        ReportError::IncompatibleTypes(begin->GetLocation(), 
					beginType, Type::intType, false);
		}
	}
	if (end != NULL) {
		errors += end->emitScopeAndTypeErrors(scope);
		Type *endType = end->getType();
                if (endType != NULL && !Type::intType->isAssignableFrom(endType)) {
                        ReportError::IncompatibleTypes(end->GetLocation(), 
					endType, Type::intType, false);
		}
	}
	return errors;
}

void IndexRange::retrieveTerminalFieldAccesses(List<FieldAccess*> *fieldList) {
	if (begin != NULL) {
		begin->retrieveTerminalFieldAccesses(fieldList);
	}
	if (end != NULL) {
		end->retrieveTerminalFieldAccesses(fieldList);
	}
}

void IndexRange::performStageParamReplacement(
		Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap,
		Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap) {
	if (begin != NULL) {
		begin->performStageParamReplacement(nameAdjustmentInstrMap, arrayAccXformInstrMap);
	}
	if (end != NULL) {
		end->performStageParamReplacement(nameAdjustmentInstrMap, arrayAccXformInstrMap);
	}
}

Hashtable<VariableAccess*> *IndexRange::getAccessedGlobalVariables(TaskGlobalReferences *globalReferences) {
        Hashtable<VariableAccess*> *table = new Hashtable<VariableAccess*>;
        if (begin != NULL) {
                table = begin->getAccessedGlobalVariables(globalReferences);
        }
        if (end != NULL) {
                Hashtable<VariableAccess*> *eTable = end->getAccessedGlobalVariables(globalReferences);
		Stmt::Stmt::mergeAccessedVariables(table, eTable);
        }
	Iterator<VariableAccess*> iter = table->GetIterator();
        VariableAccess *accessLog;
        while((accessLog = iter.GetNextValue()) != NULL) {
                if(accessLog->isContentAccessed())
                        accessLog->getContentAccessFlags()->flagAsRead();
                if (accessLog->isMetadataAccessed())
                        accessLog->getMetadataAccessFlags()->flagAsRead();
        }
        return table;
}

void IndexRange::setEpochVersions(Space *space, int epoch) {
        if (begin != NULL) begin->setEpochVersions(space, epoch);
        if (end != NULL) end->setEpochVersions(space, epoch);
}

