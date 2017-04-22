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
#include "../../semantics/data_access.h"
#include "../../static-analysis/reduction_info.h"
#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/hashtable.h"

#include <iostream>
#include <sstream>
#include <cstdlib>

//-------------------------------------------------------- Reduction Statement -------------------------------------------------------/

ReductionStmt::ReductionStmt(Identifier *l, char *o, Expr *r, yyltype loc) : Stmt(loc) {

        Assert(l != NULL && r != NULL && o != NULL);

        left = l;
        left->SetParent(this);

        if (strcmp(o, "sum") == 0) op = SUM;
        else if (strcmp(o, "product") == 0) op = PRODUCT;
        else if (strcmp(o, "max") == 0) op = MAX;
        else if (strcmp(o, "max_entry") == 0) op = MAX_ENTRY;
        else if (strcmp(o, "min") == 0) op = MIN;
        else if (strcmp(o, "min_entry") == 0) op = MIN_ENTRY;
        else if (strcmp(o, "land") == 0) op = LAND;
        else if (strcmp(o, "lor") == 0) op = LOR;
        else if (strcmp(o, "band") == 0) op = BAND;
        else if (strcmp(o, "bor") == 0) op = BOR;
        else {
                std::cout << "Currently the compiler does not support user defined reduction functions";
                Assert(0 == 1);
        }

        right = r;
        right->SetParent(this);
	reductionVar = NULL;
	enclosingLoop = NULL;
}

ReductionStmt::ReductionStmt(Identifier *l, ReductionOperator o, Expr *r, yyltype loc) : Stmt(loc) {
        left = l;
        left->SetParent(this);
	op = o;
        right = r;
        right->SetParent(this);
	reductionVar = NULL;
	enclosingLoop = NULL;
}	

void ReductionStmt::PrintChildren(int indentLevel) {
        left->Print(indentLevel + 1);
        PrintLabel(indentLevel + 1, "Operator");
        switch (op) {
                case SUM: printf("Sum"); break;
                case PRODUCT: printf("Product"); break;
                case MAX: printf("Maximum"); break;
                case MIN: printf("Minimum"); break;
                case MIN_ENTRY: printf("Minimum Entry"); break;
                case MAX_ENTRY: printf("Maximum Entry"); break;
                case LOR: printf("Logical OR"); break;
                case LAND: printf("Logical AND"); break;
                case BOR: printf("Bitwise OR"); break;
                case BAND: printf("Bitwise AND"); break;
        }
        right->Print(indentLevel + 1);
}

Node *ReductionStmt::clone() {
	Identifier *newLeft = (Identifier*) left->clone();
	Expr *newRight = (Expr*) right->clone();
	return new ReductionStmt(newLeft, op, newRight, *GetLocation());	
}

void ReductionStmt::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	right->retrieveExprByType(exprList, typeId);
}

int ReductionStmt::resolveExprTypesAndScopes(Scope *executionScope, int iteration) {

	int resolvedExprs = right->resolveExprTypesAndScopes(executionScope, iteration);
	if (op == LOR || op == LAND) {
		resolvedExprs += right->performTypeInference(executionScope, Type::boolType);
	}

	// resolve the result type from the reduced expression type
	Type *rightType = right->getType();
	Type *resultType = inferResultTypeFromOpAndExprType(rightType);	
	if (resultType != NULL) {
		const char *resultVar = left->getName();
		if (executionScope->lookup(resultVar) == NULL) {
			VariableSymbol *var = new VariableSymbol(new VariableDef(left, resultType));
			executionScope->insert_inferred_symbol(var);
			resolvedExprs++;	
		}
	}

	enclosingLoop = getEnclosingLoop();

	return resolvedExprs;
}

Type *ReductionStmt::inferResultTypeFromOpAndExprType(Type *exprType) {

	if (exprType == NULL || exprType == Type::errorType) return NULL;
        
	switch (op) {
                case SUM: return exprType;
                case PRODUCT: return exprType;
                case MAX: return exprType;
                case MIN: return exprType;
                case MIN_ENTRY: return Type::intType;
                case MAX_ENTRY: return Type::intType;
                case LOR: return Type::boolType;
                case LAND: return Type::boolType;
                case BOR: return exprType;
                case BAND: return exprType;
        }

	return NULL;
}

int ReductionStmt::emitScopeAndTypeErrors(Scope *scope) {
	int errorCount = 0;
	if (enclosingLoop == NULL) {
		ReportError::ReductionOutsideForLoop(GetLocation(), false);
		errorCount++;
	} else if (op == MIN_ENTRY || op == MAX_ENTRY) {
	List<const char*> *indexList = enclosingLoop->getAllIndexNames();
		if (indexList->NumElements() != 1) {
			ReportError::IndexReductionOnMultiIndexLoop(GetLocation(), false);	
			errorCount++;
		}
	}
	errorCount += right->emitScopeAndTypeErrors(scope);
	return errorCount;
}

void ReductionStmt::performStageParamReplacement(
		Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap,
		Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap) {

	right->performStageParamReplacement(nameAdjustmentInstrMap, arrayAccXformInstrMap);

	const char *resultName = left->getName();
	ParamReplacementConfig *paramReplConfig = nameAdjustmentInstrMap->Lookup(resultName);
	if (paramReplConfig != NULL) {
		Expr *argument = paramReplConfig->getInvokingArg();
		ReductionVar *reductionVar = dynamic_cast<ReductionVar*>(argument);
		if (reductionVar == NULL) {
			ReportError::InvalidReductionExpr(argument, false);
			return;
		}
		this->reductionVar = (ReductionVar*) reductionVar->clone();
	}
}

Hashtable<VariableAccess*> *ReductionStmt::getAccessedGlobalVariables(TaskGlobalReferences *globalReferences) {

        Hashtable<VariableAccess*> *table = new Hashtable<VariableAccess*>;
	if (reductionVar != NULL) {
		const char *resultName = reductionVar->getName();
		VariableAccess *accessLog = new VariableAccess(resultName);
		accessLog->markContentAccess();
		accessLog->getContentAccessFlags()->flagAsReduced();
		table->Enter(resultName, accessLog, true);
	} else {
		const char *fieldName = left->getName();
		if (globalReferences->doesReferToGlobal(fieldName)) {	
			VariableAccess *accessLog = new VariableAccess(fieldName);
			accessLog->markContentAccess();
			accessLog->getContentAccessFlags()->flagAsWritten();
			table->Enter(fieldName, accessLog, true);
		}
	}

        Hashtable<VariableAccess*> *rTable = right->getAccessedGlobalVariables(globalReferences);
	List<Expr*> *rightFieldAccesses = new List<Expr*>;
        right->retrieveExprByType(rightFieldAccesses, FIELD_ACC);
        for (int i = 0; i < rightFieldAccesses->NumElements(); i++) {
		FieldAccess *exprField = (FieldAccess*) rightFieldAccesses->Nth(i);
                FieldAccess *rootField = exprField->getTerminalField();
                if (rootField == NULL) continue;

                const char *varName = rootField->getField()->getName();
                VariableAccess *accessLog = rTable->Lookup(varName);

                // if the field is not a task-global variable then we can ignore it 
                if (accessLog == NULL) continue;

                Type *fieldType = rootField->getType();
                ArrayType *array = dynamic_cast<ArrayType*>(fieldType);

                // if the field is not an array then its access flags are already set properly
                if (array == NULL) continue;

                // if the content of the array has been accessed then it should be flagged as read
                if (accessLog->isContentAccessed()) {
                        accessLog->getContentAccessFlags()->flagAsRead();
                }
        }

        mergeAccessedVariables(table, rTable);
        return table;
}

PLoopStmt *ReductionStmt::getEnclosingLoop() {
	Node *parent = this->GetParent();
	while (parent != NULL) {
		Stmt *parentStmt = dynamic_cast<Stmt*>(parent);
		if (parentStmt == NULL) break;
		LoopStmt *enclosingLoop = dynamic_cast<LoopStmt*>(parentStmt);
		if (enclosingLoop != NULL) {
			PLoopStmt *parallelLoop = dynamic_cast<PLoopStmt*>(enclosingLoop);
			if (parallelLoop != NULL) return parallelLoop;
			else break;
		}
		parent = parent->GetParent();
	}
	return NULL;
}

void ReductionStmt::analyseEpochDependencies(Space *space) {
        right->analyseEpochDependencies(space);
}

void ReductionStmt::extractReductionInfo(List<ReductionMetadata*> *infoSet,
                PartitionHierarchy *lpsHierarchy,
                Space *executingLps) {

	if (reductionVar == NULL) return;

	char spaceId = reductionVar->getSpaceId();
	const char *resultVar = reductionVar->getName();
        Space *reductionRootLps = lpsHierarchy->getSpace(spaceId);

	if (executingLps != reductionRootLps && !executingLps->isParentSpace(reductionRootLps)) {
		ReportError::InvalidReductionRange(GetLocation(), 
				executingLps->getName(), reductionRootLps->getName(), false);
		return;
	}

	Type *exprType = right->getType();
        ReductionMetadata *metadata = new ReductionMetadata(resultVar,
                        op, exprType, reductionRootLps, executingLps, GetLocation());
        infoSet->Append(metadata);
}
