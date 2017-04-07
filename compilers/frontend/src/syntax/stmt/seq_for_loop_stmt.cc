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
#include "../../semantics/loop_index.h"
#include "../../semantics/data_access.h"
#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/hashtable.h"

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
		loopScope = executionScope->enter_scope(new Scope(StatementBlockScope));
		
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

	// if needed create and enter a loop index to array dimension association scope
	if (iteration == 0) {
		prepareIndexScope(executionScope);
        } else {
                IndexScope::currentScope->enterScope(this->indexScope);
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

	// exit the index to array dimension association scope
	this->indexScope = IndexScope::currentScope;
        IndexScope::currentScope->goBackToOldScope();

	return resolvedExprs;
}

int SLoopStmt::emitScopeAndTypeErrors(Scope *executionScope) {

	int errors = 0;
	Scope *loopScope = executionScope->enter_scope(this->scope);
	
	// range expression must be of range type
	errors += rangeExpr->emitScopeAndTypeErrors(loopScope);
	Type *type = rangeExpr->getType();
	if (type != NULL && type != Type::rangeType) {
		ReportError::InvalidExprType(rangeExpr, Type::rangeType, false);
		errors++;
	}
	
	// if exists, step expression must of integer type
        if (stepExpr != NULL) {
		errors += stepExpr->emitScopeAndTypeErrors(loopScope);
		Type *type = stepExpr->getType();
		if (type != NULL && type != Type::intType) {
			ReportError::InvalidExprType(stepExpr, Type::intType, false);
			errors++;
		}
	}

	// if exists, restriction must be of boolean type
        if (restriction != NULL) {
		errors += restriction->emitScopeAndTypeErrors(loopScope);
		Type *type = restriction->getType();
		if (type != NULL && type != Type::boolType) {
			ReportError::InvalidExprType(restriction, Type::boolType, false);
			errors++;
		}
	}

        errors += body->emitScopeAndTypeErrors(loopScope);
	loopScope->detach_from_parent();
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

Hashtable<VariableAccess*> *SLoopStmt::getAccessedGlobalVariables(TaskGlobalReferences *globalReferences) {

        Hashtable<VariableAccess*> *table = body->getAccessedGlobalVariables(globalReferences);
        mergeAccessedVariables(table, rangeExpr->getAccessedGlobalVariables(globalReferences));
        if (stepExpr != NULL) mergeAccessedVariables(table,
                        stepExpr->getAccessedGlobalVariables(globalReferences));
        if (restriction != NULL) mergeAccessedVariables(table,
                        restriction->getAccessedGlobalVariables(globalReferences));

        Iterator<VariableAccess*> iter = table->GetIterator();
        VariableAccess *accessLog;
        while ((accessLog = iter.GetNextValue()) != NULL) {
                if(accessLog->isContentAccessed())
                        accessLog->getContentAccessFlags()->flagAsRead();
                if (accessLog->isMetadataAccessed())
                        accessLog->getMetadataAccessFlags()->flagAsRead();
        }
        return table;
}

void SLoopStmt::prepareIndexScope(Scope *executionScope) {

	// create an empty index association scope and enter it
	IndexScope::currentScope->deriveNewScope();

	// Try to find out if the range corresponding to a dimension of some global array. If it is so 
        // then create an entry in the index scope.
        const char *potentialArray = rangeExpr->getBaseVarName();
        Scope *taskScope = executionScope->get_nearest_scope(TaskScope);
        if (potentialArray != NULL && taskScope != NULL) {
                Symbol *symbol = taskScope->local_lookup(potentialArray);
                bool attemptResolve = false;
                if (symbol != NULL) {
                        VariableSymbol *variable = dynamic_cast<VariableSymbol*>(symbol);
                        if (variable != NULL) { 
                                Type *varType = variable->getType();
                                if (dynamic_cast<ArrayType*>(varType) != NULL &&
                                                dynamic_cast<StaticArrayType*>(varType) == NULL) {
                                        attemptResolve = true;
                                }
                        }
                }
		// It seems finding out if an expression is a dimension access of a task global array is
                // a messy effort. The expression is expected  to look like array.dimension#No.range. So
                // there is a need to do a three level unfolding of expression. It would be nice if we 
                // could generalize this procedure somewhere. TODO may be worth attempting in the future.
                bool dimensionFound = false;
                int dimension = 0;
                if (attemptResolve) {
                        FieldAccess *rangeField = dynamic_cast<FieldAccess*>(rangeExpr);
                        if (rangeField != NULL) {
                                Expr *base = rangeField->getBase();
                                FieldAccess *baseField = dynamic_cast<FieldAccess*>(base);
                                if (baseField != NULL) {
                                        Expr *arrayExpr = baseField->getBase();
                                        FieldAccess *arrayAccess = dynamic_cast<FieldAccess*>(arrayExpr);
                                        Identifier *field = baseField->getField();
                                        DimensionIdentifier *dimensionId =
                                                        dynamic_cast<DimensionIdentifier*>(field);
                                        if (arrayAccess != NULL
                                                        && arrayAccess->isTerminalField()
                                                        && dimensionId != NULL) {
                                                dimensionFound = true;
                                                // this is a coversion between 1 based to 0 based indexing
                                                dimension = dimensionId->getDimensionNo() - 1;
                                        }
                                }
                        }
                }
		if (dimensionFound) {
                        const char *indexName = id->getName();
                        IndexScope::currentScope->initiateAssociationList(indexName);
                        List<IndexArrayAssociation*> *list
                                        = IndexScope::currentScope->getAssociationsForIndex(indexName);
                        list->Append(new IndexArrayAssociation(indexName, potentialArray, dimension));
                        
			// if an association from the index to an array dimension is found then this loop
			// should be flagged as an array traversal loop
			arrayIndexTraversal = true;
                }
        }
}

void SLoopStmt::analyseEpochDependencies(Space *space) {
        rangeExpr->setEpochVersions(space, 0);
        if (stepExpr != NULL) stepExpr->setEpochVersions(space, 0);
        if (restriction != NULL) restriction->setEpochVersions(space, 0);
	body->analyseEpochDependencies(space);
}
