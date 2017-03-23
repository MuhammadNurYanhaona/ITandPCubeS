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
#include "../../../../common-libs/utils/list.h"

#include <iostream>
#include <sstream>
#include <cstdlib>

//------------------------------------------------------- Sequential For Loop --------------------------------------------------------/

SLoopAttribute::SLoopAttribute(Expr *range, Expr *step, Expr *restriction) {
        Assert(range != NULL);
        this->range = range;
        this->step = step;
        this->restriction = restriction;
}

SLoopAttribute *SLoopAttribute::clone() {
	Expr *newRange = (Expr*) range->clone();
	Expr *newStep = NULL;
	if (step != NULL) newStep = (Expr*) step->clone();
	Expr *newRestr = NULL;
	if (restriction != NULL) newRestr = (Expr*) restriction->clone();
	return new SLoopAttribute(newRange, newStep, newRestr);
}

SLoopStmt::SLoopStmt(Identifier *i, SLoopAttribute *attr, Stmt *b, yyltype loc) : LoopStmt(b, loc) {
        
	Assert(i != NULL && attr != NULL);
        
	id = i;
        id->SetParent(this);
	attrRef = attr;
        
	rangeExpr = attr->getRange();
        rangeExpr->SetParent(this);
        stepExpr = attr->getStep();
        if (stepExpr != NULL) {
                stepExpr->SetParent(this);
        }
        restriction = attr->getRestriction();
        if (restriction != NULL) {
                restriction->SetParent(this);
        }
}

void SLoopStmt::PrintChildren(int indentLevel) {
        id->Print(indentLevel + 1, "(Index) ");
        rangeExpr->Print(indentLevel + 1, "(Range) ");
        if (stepExpr != NULL) stepExpr->Print(indentLevel + 1, "(Step) ");
        if (restriction != NULL) restriction->Print(indentLevel + 1, "(Index Restriction) ");
        body->Print(indentLevel + 1);
}

Node *SLoopStmt::clone() {
	Identifier *newId = (Identifier*) id->clone();
	SLoopAttribute *newAttr = (SLoopAttribute*) attrRef->clone();
	Stmt *newBody = (Stmt*) body->clone();
	return new SLoopStmt(newId, newAttr, newBody, *GetLocation());
}

void SLoopStmt::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	rangeExpr->retrieveExprByType(exprList, typeId);
	if (stepExpr != NULL) stepExpr->retrieveExprByType(exprList, typeId);
	if (restriction != NULL) restriction->retrieveExprByType(exprList, typeId);
	body->retrieveExprByType(exprList, typeId);
}

int SLoopStmt::resolveExprTypesAndScopes(Scope *executionScope, int iteration) {
	
	// create a new scope for the loop and enter it
	Scope *loopScope = NULL;
	if (iteration == 0) {
		Scope *loopScope = executionScope->enter_scope(new Scope(StatementBlockScope));
		
		// enter the loop iterator in the scope
		if (loopScope->lookup(id->getName()) != NULL) {
                	ReportError::ConflictingDefinition(id, false);
		} else {
			VariableSymbol *var = new VariableSymbol(new VariableDef(id, Type::intType));
			loopScope->insert_symbol(var);
		}
	} else {
		loopScope = executionScope->enter_scope(this->scope);
	}

	int resolvedExprs = 0;
	
	// Type inference process only makes progress if the type of the expression is currently unknown 
	// or erroneously resolved. So it is okay to just invoke the inference process after an attempt 
	// is made to resolve the expression using the normal process. 
	resolvedExprs += rangeExpr->resolveExprTypesAndScopes(loopScope, iteration);
	resolvedExprs += rangeExpr->performTypeInference(loopScope, Type::rangeType);
	if (stepExpr != NULL) {
		resolvedExprs += stepExpr->resolveExprTypesAndScopes(loopScope, iteration);
		resolvedExprs += stepExpr->performTypeInference(loopScope, Type::intType);
	}
	if (restriction != NULL) {
		resolvedExprs += restriction->resolveExprTypesAndScopes(loopScope, iteration);
                resolvedExprs += restriction->performTypeInference(loopScope, Type::boolType);
	}
	
	// try to resolve the body after evaluating the iteration expression to maximize type discovery
	resolvedExprs += body->resolveExprTypesAndScopes(loopScope, iteration);

	// exit the scope
	loopScope->detach_from_parent();
        this->scope = loopScope;

	return resolvedExprs;
}

int SLoopStmt::emitScopeAndTypeErrors(Scope *scope) {
	int errors = 0;
	
	// range expression must be of range type
	errors += rangeExpr->emitScopeAndTypeErrors(scope);
	Type *type = rangeExpr->getType();
	if (type != NULL && type != Type::rangeType) {
		ReportError::InvalidExprType(rangeExpr, Type::rangeType, false);
		errors++;
	}
	
	// if exists, step expression must of integer type
        if (stepExpr != NULL) {
		errors += stepExpr->emitScopeAndTypeErrors(scope);
		Type *type = stepExpr->getType();
		if (type != NULL && type != Type::intType) {
			ReportError::InvalidExprType(stepExpr, Type::intType, false);
			errors++;
		}
	}

	// if exists, restriction must be of boolean type
        if (restriction != NULL) {
		errors += restriction->emitScopeAndTypeErrors(scope);
		Type *type = restriction->getType();
		if (type != NULL && type != Type::boolType) {
			ReportError::InvalidExprType(restriction, Type::boolType, false);
			errors++;
		}
	}

        errors += body->emitScopeAndTypeErrors(scope);
	return errors;
}

void SLoopStmt::performStageParamReplacement(
		Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap,
		Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap) {
	
	rangeExpr->performStageParamReplacement(nameAdjustmentInstrMap, arrayAccXformInstrMap);
	if (stepExpr != NULL) {
		rangeExpr->performStageParamReplacement(
				nameAdjustmentInstrMap, arrayAccXformInstrMap);
	}
	if (restriction != NULL) {
		restriction->performStageParamReplacement(
				nameAdjustmentInstrMap, arrayAccXformInstrMap);
	}
}

