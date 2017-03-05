#include "ast.h"
#include "ast_stmt.h"
#include "ast_expr.h"
#include "ast_def.h"
#include "ast_task.h"
#include "../common/errors.h"
#include "../common/location.h"
#include "../semantics/scope.h"
#include "../semantics/symbol.h"
#include "../../../common-libs/utils/list.h"

#include <iostream>
#include <sstream>
#include <cstdlib>

//------------------------------------------------------------ Statement Block ---------------------------------------------------------/

StmtBlock::StmtBlock(List<Stmt*> *s) : Stmt() {
        Assert(s != NULL);
        stmts = s;
        for (int i = 0; i < stmts->NumElements(); i++) {
                stmts->Nth(i)->SetParent(this);
        }
}

void StmtBlock::PrintChildren(int indentLevel) {
        stmts->PrintAll(indentLevel + 1);
}

Node *StmtBlock::clone() {
	List<Stmt*> *newStmtList = new List<Stmt*>;
	for (int i = 0; i < stmts->NumElements(); i++) {
                Stmt *stmt = stmts->Nth(i);
		Stmt *newStmt = (Stmt*) stmt->clone();
		newStmtList->Append(newStmt);
        }
	return new StmtBlock(newStmtList);
}

void StmtBlock::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	for (int i = 0; i < stmts->NumElements(); i++) {
                Stmt *stmt = stmts->Nth(i);
		stmt->retrieveExprByType(exprList, typeId);
	}
}

int StmtBlock::resolveExprTypesAndScopes(Scope *executionScope, int iteration) {
	int resolvedExprs = 0;
	for (int i = 0; i < stmts->NumElements(); i++) {
                Stmt *stmt = stmts->Nth(i);
		resolvedExprs += stmt->resolveExprTypesAndScopes(executionScope, iteration);
	}
	return resolvedExprs;
}

//-------------------------------------------------------- Conditional Statement -------------------------------------------------------/

ConditionalStmt::ConditionalStmt(Expr *c, Stmt *s, yyltype loc) : Stmt(loc) {
        Assert(s != NULL);
        condition = c;
        if (condition != NULL) {
                condition->SetParent(this);
        }
        stmt = s;
        stmt->SetParent(this);
}

void ConditionalStmt::PrintChildren(int indentLevel) {
        if (condition != NULL) condition->Print(indentLevel, "(If) ");
        stmt->Print(indentLevel);
}

Node *ConditionalStmt::clone() {
	Expr *newCond = (Expr*) condition->clone();
	Stmt *newStmt = (Stmt*) stmt->clone();
	return new ConditionalStmt(newCond, newStmt, *GetLocation());
}

void ConditionalStmt::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	if (condition != NULL) condition->retrieveExprByType(exprList, typeId);
	stmt->retrieveExprByType(exprList, typeId);
}

int ConditionalStmt::resolveExprTypesAndScopes(Scope *executionScope, int iteration) {
	int resolvedExprs = stmt->resolveExprTypesAndScopes(executionScope, iteration);
	if (condition != NULL) {
		resolvedExprs += condition->resolveExprTypesAndScopes(executionScope, iteration);
		int inferredTypes = condition->performTypeInference(executionScope, Type::boolType);
		resolvedExprs += inferredTypes;
	}
	return resolvedExprs;
}

//------------------------------------------------------------ If/Else Block -----------------------------------------------------------/

IfStmt::IfStmt(List<ConditionalStmt*> *ib, yyltype loc) : Stmt(loc) {
        Assert(ib != NULL);
        ifBlocks = ib;
        for (int i = 0; i < ifBlocks->NumElements(); i++) {
                ifBlocks->Nth(i)->SetParent(this);
        }
}

void IfStmt::PrintChildren(int indentLevel) {
        ifBlocks->PrintAll(indentLevel + 1);
}

Node *IfStmt::clone() {
	List<ConditionalStmt*> *newBlocks = new List<ConditionalStmt*>;
	for (int i = 0; i < ifBlocks->NumElements(); i++) {
                ConditionalStmt *stmt = ifBlocks->Nth(i);
		ConditionalStmt *newStmt = (ConditionalStmt*) stmt->clone();
		newBlocks->Append(newStmt);
        }
	return new IfStmt(newBlocks, *GetLocation());
}

void IfStmt::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	for (int i = 0; i < ifBlocks->NumElements(); i++) {
                ConditionalStmt *stmt = ifBlocks->Nth(i);
		stmt->retrieveExprByType(exprList, typeId);
	}
}

int IfStmt::resolveExprTypesAndScopes(Scope *excutionScope, int iteration) {
	int resolvedExprs = 0;
	for (int i = 0; i < ifBlocks->NumElements(); i++) {
                ConditionalStmt *stmt = ifBlocks->Nth(i);
		resolvedExprs += stmt->resolveExprTypesAndScopes(excutionScope, iteration);
	}
	return resolvedExprs;
}

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

//------------------------------------------------------------ Loop Statement ----------------------------------------------------------/

LoopStmt::LoopStmt() : Stmt() {
	this->body = NULL;
	this->scope = NULL;
}

LoopStmt::LoopStmt(Stmt *body, yyltype loc) : Stmt(loc) {
	Assert(body != NULL);
	this->body = body;
	this->body->SetParent(this);
	this->scope = NULL;
}

//------------------------------------------------------------ Parallel Loop ----------------------------------------------------------/

PLoopStmt::PLoopStmt(List<IndexRangeCondition*> *rc, Stmt *b, yyltype loc) : LoopStmt(b, loc) {
        Assert(rc != NULL);
        rangeConditions = rc;
        for (int i = 0; i < rangeConditions->NumElements(); i++) {
                rangeConditions->Nth(i)->SetParent(this);
        }
}

void PLoopStmt::PrintChildren(int indentLevel) {
        rangeConditions->PrintAll(indentLevel + 1);
        body->Print(indentLevel + 1);
}

Node *PLoopStmt::clone() {
	List<IndexRangeCondition*> *newCondList = new List<IndexRangeCondition*>;
	for (int i = 0; i < rangeConditions->NumElements(); i++) {
                IndexRangeCondition *condition = rangeConditions->Nth(i);
		IndexRangeCondition *newCond = (IndexRangeCondition*) condition->clone();
		newCondList->Append(newCond);
        }
	Stmt *newBody = (Stmt*) body->clone();
	return new PLoopStmt(newCondList, newBody, *GetLocation());
}

void PLoopStmt::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	for (int i = 0; i < rangeConditions->NumElements(); i++) {
                IndexRangeCondition *condition = rangeConditions->Nth(i);
		condition->retrieveExprByType(exprList, typeId);
	}
	body->retrieveExprByType(exprList, typeId);
}

int PLoopStmt::resolveExprTypesAndScopes(Scope *executionScope, int iteration) {

	// create a new scope for the loop and enter it
	Scope *loopScope = NULL;
	if (iteration == 0) {
		Scope *loopScope = executionScope->enter_scope(new Scope(StatementBlockScope));
	} else {
		loopScope = executionScope->enter_scope(this->scope);
	}

	int resolvedExprs = 0;
	for (int i = 0; i < rangeConditions->NumElements(); i++) {
                IndexRangeCondition *condition = rangeConditions->Nth(i);
		resolvedExprs += condition->resolveExprTypesAndScopes(loopScope, iteration);
	}
	
	// try to resolve the body after evaluating the iteration expression to maximize type discovery
	resolvedExprs += body->resolveExprTypesAndScopes(loopScope, iteration);

	// exit the scope
	loopScope->detach_from_parent();
        this->scope = loopScope;

	return resolvedExprs;
}

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
	
	// try to resolve the body after evaluating the iteration expression to maximize type discovery
	resolvedExprs += body->resolveExprTypesAndScopes(loopScope, iteration);

	// exit the scope
	loopScope->detach_from_parent();
        this->scope = loopScope;

	return resolvedExprs;
}

//------------------------------------------------------------ While Loop ------------------------------------------------------------/

WhileStmt::WhileStmt(Expr *c, Stmt *b, yyltype loc) : Stmt(loc) {
        Assert(c != NULL && b != NULL);
        condition = c;
        condition->SetParent(this);
        body = b;
        body->SetParent(this);
}

void WhileStmt::PrintChildren(int indentLevel) {
        condition->Print(indentLevel + 1, "(Condition) ");
        body->Print(indentLevel + 1);
}

Node *WhileStmt::clone() {
	Expr *newCond = (Expr*) condition->clone();
	Stmt *newBody = (Stmt*) body->clone();
	return new WhileStmt(newCond, newBody, *GetLocation());
}

void WhileStmt::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	condition->retrieveExprByType(exprList, typeId);
	body->retrieveExprByType(exprList, typeId);
}

int WhileStmt::resolveExprTypesAndScopes(Scope *executionScope, int iteration) {
	
	int resolvedExprs = 0;
	resolvedExprs += condition->resolveExprTypesAndScopes(executionScope, iteration);
	resolvedExprs += condition->performTypeInference(executionScope, Type::boolType);

	// try to resolve the body after evaluating the iteration expression to maximize type discovery
	resolvedExprs += body->resolveExprTypesAndScopes(executionScope, iteration);
	return resolvedExprs;
}

//-------------------------------------------------------- Reduction Statement -------------------------------------------------------/

ReductionStmt::ReductionStmt(Identifier *l, char *o, Expr *r, yyltype loc) : Stmt(loc) {

        Assert(l != NULL && r != NULL && o != NULL);

        left = l;
        left->SetParent(this);

        if (strcmp(o, "sum") == 0) op = SUM;
        else if (strcmp(o, "product") == 0) op = PRODUCT;
        else if (strcmp(o, "max") == 0) op = MAX;
        else if (strcmp(o, "maxEntry") == 0) op = MAX_ENTRY;
        else if (strcmp(o, "min") == 0) op = MIN;
        else if (strcmp(o, "minEntry") == 0) op = MIN_ENTRY;
        else if (strcmp(o, "avg") == 0) op = AVG;
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
}

ReductionStmt::ReductionStmt(Identifier *l, ReductionOperator o, Expr *r, yyltype loc) : Stmt(loc) {
        left = l;
        left->SetParent(this);
	op = o;
        right = r;
        right->SetParent(this);
}	

void ReductionStmt::PrintChildren(int indentLevel) {
        left->Print(indentLevel + 1);
        PrintLabel(indentLevel + 1, "Operator");
        switch (op) {
                case SUM: printf("Sum"); break;
                case PRODUCT: printf("Product"); break;
                case MAX: printf("Maximum"); break;
                case MIN: printf("Minimum"); break;
                case AVG: printf("Average"); break;
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
			VariableSymbol *var = new VariableSymbol(new VariableDef(left, Type::intType));
			executionScope->insert_inferred_symbol(var);
			resolvedExprs++;	
		}
	}

	return resolvedExprs;
}

Type *ReductionStmt::inferResultTypeFromOpAndExprType(Type *exprType) {

	if (exprType == NULL || exprType == Type::errorType) return NULL;
        
	switch (op) {
                case SUM: return exprType;
                case PRODUCT: return exprType;
                case MAX: return exprType;
                case MIN: return exprType;
                case AVG: return exprType;
                case MIN_ENTRY: return Type::intType;
                case MAX_ENTRY: return Type::intType;
                case LOR: return Type::boolType;
                case LAND: return Type::boolType;
                case BOR: return exprType;
                case BAND: return exprType;
        }

	return NULL;
}

//-------------------------------------------------------- External Code Block -------------------------------------------------------/

ExternCodeBlock::ExternCodeBlock(const char *language,
                        List<const char*> *headerIncludes,
                        List<const char*> *libraryLinks,
                        const char *codeBlock, yyltype loc) : Stmt(loc) {

        Assert(language != NULL && codeBlock != NULL);
        this->language = language;
        this->headerIncludes = headerIncludes;
        this->libraryLinks = libraryLinks;
        this->codeBlock = codeBlock;
}

void ExternCodeBlock::PrintChildren(int indentLevel) {
        std::ostringstream indent;
        for (int i = 0; i < indentLevel; i++) {
                indent << '\t';
        }
        std::cout << indent.str() << "Language: " << language << "\n";
        if (headerIncludes != NULL) {
                std::cout << indent.str() << "Included Headers:\n";
                for (int i = 0; i < headerIncludes->NumElements(); i++) {
                        std::cout << indent.str() << '\t' << headerIncludes->Nth(i) << "\n";
                }
        }
        if (libraryLinks != NULL) {
                std::cout << indent.str() << "Linked Libraries:\n";
                for (int i = 0; i < libraryLinks->NumElements(); i++) {
                        std::cout << indent.str() << '\t' << libraryLinks->Nth(i) << "\n";
                }
        }
        std::cout << indent.str() << "Code Block:" << codeBlock << "\n";
}

Node *ExternCodeBlock::clone() {
	const char *newLng = strdup(language);
	List<const char*> *newIncls = new List<const char*>;
	for (int i = 0; i < headerIncludes->NumElements(); i++) {
		newIncls->Append(strdup(headerIncludes->Nth(i)));
	}
	List<const char*> *newLibs = new List<const char*>;
	for (int i = 0; i < libraryLinks->NumElements(); i++) {
		newLibs->Append(strdup(libraryLinks->Nth(i)));
	}
	const char *newCode = strdup(codeBlock);
	return new ExternCodeBlock(newLng, newIncls, newLibs, newCode, *GetLocation());
}

//------------------------------------------------------------ Return Stmt -----------------------------------------------------------/

ReturnStmt::ReturnStmt(Expr *expr, yyltype loc) : Stmt(loc) {
	Assert(expr != NULL);
	this->expr = expr;
	this->expr->SetParent(this);
}

void ReturnStmt::PrintChildren(int indentLevel) {
	expr->Print(indentLevel + 1);
}

Node *ReturnStmt::clone() {
	Expr *newExpr = (Expr*) expr->clone();
	return new ReturnStmt(newExpr, *GetLocation());
}

void ReturnStmt::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	expr->retrieveExprByType(exprList, typeId);
}

int ReturnStmt::resolveExprTypesAndScopes(Scope *executionScope, int iteration) {
	return expr->resolveExprTypesAndScopes(executionScope, iteration);
}
