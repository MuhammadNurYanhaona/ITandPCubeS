#include "list.h"
#include "ast.h"
#include "ast_stmt.h"
#include "ast_expr.h"
#include "ast_def.h"
#include "symbol.h"

//------------------------------------------ Statement Block ----------------------------------------/
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

void StmtBlock::checkSemantics(Scope *executionScope, bool ignoreTypeFailures) {
	for (int i = 0; i < stmts->NumElements(); i++) {
		Stmt *stmt = stmts->Nth(i);
		stmt->checkSemantics(executionScope, ignoreTypeFailures);
	}
}

void StmtBlock::performTypeInference(Scope *executionScope) {
	for (int i = 0; i < stmts->NumElements(); i++) {
		Stmt *stmt = stmts->Nth(i);
		stmt->performTypeInference(executionScope);
	}
}
	
//-------------------------------------- Conditional Statement ---------------------------------------/
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

void ConditionalStmt::checkSemantics(Scope *executionScope, bool ignoreTypeFailures) {
	if (condition != NULL) condition->resolveType(executionScope, ignoreTypeFailures);
	stmt->checkSemantics(executionScope, ignoreTypeFailures);	
}

void ConditionalStmt::performTypeInference(Scope *executionScope) {
	if (condition != NULL) {
		condition->inferType(executionScope, Type::boolType);
	}
	stmt->performTypeInference(executionScope);	
}

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

void IfStmt::checkSemantics(Scope *executionScope, bool ignoreTypeFailures) {
	for (int i = 0; i < ifBlocks->NumElements(); i++) {
		ConditionalStmt *stmt = ifBlocks->Nth(i);
		stmt->checkSemantics(executionScope, ignoreTypeFailures);
	}
}

void IfStmt::performTypeInference(Scope *executionScope) {
	for (int i = 0; i < ifBlocks->NumElements(); i++) {
		ConditionalStmt *stmt = ifBlocks->Nth(i);
		stmt->performTypeInference(executionScope);
	}
}
	
//----------------------------------------- Parallel Loop -------------------------------------------/
IndexRangeCondition::IndexRangeCondition(List<Identifier*> *i, Identifier *c, Expr *rs, yyltype loc) : Node(loc) {
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
}

void IndexRangeCondition::PrintChildren(int indentLevel) {
	indexes->PrintAll(indentLevel + 1, "(Index) ");
	collection->Print(indentLevel + 1, "(Array/List) ");
	if (restrictions != NULL) restrictions->Print(indentLevel + 1, "(Restrictions) ");
}

void IndexRangeCondition::resolveTypes(Scope *executionScope, bool ignoreTypeFailures) {

	for (int i = 0; i < indexes->NumElements(); i++) {
		Identifier *ind = indexes->Nth(i);
		const char* indexName = ind->getName();
		if (executionScope->lookup(indexName) != NULL) {
			// report error
		} else {
			VariableDef *variable = new VariableDef(ind, Type::intType);
			executionScope->insert_symbol(new VariableSymbol(variable));
		}
	}

	Symbol *colSymbol = executionScope->lookup(collection->getName());
	if (colSymbol == NULL && !ignoreTypeFailures) {
		// report error
	} else if (colSymbol != NULL) {
		VariableSymbol *varSym = (VariableSymbol*) colSymbol;
		Type *varType = varSym->getType();
		ArrayType *arrayType = dynamic_cast<ArrayType*>(varType);
		if (arrayType == NULL) {
			// report error
		}
	}

	if (restrictions != NULL) {
		restrictions->resolveType(executionScope, ignoreTypeFailures);
	}
}

void IndexRangeCondition::inferTypes(Scope *executionScope) {
	if (restrictions != NULL) {
		restrictions->inferType(executionScope, Type::boolType);
	}
}

PLoopStmt::PLoopStmt(List<IndexRangeCondition*> *rc, Stmt *b, yyltype loc) : LoopStmt(loc) {
	Assert(rc != NULL && b != NULL);
	rangeConditions = rc;
	for (int i = 0; i < rangeConditions->NumElements(); i++) {
		rangeConditions->Nth(i)->SetParent(this);
	}
	body = b;
	body->SetParent(this);
}

void PLoopStmt::PrintChildren(int indentLevel) {
	rangeConditions->PrintAll(indentLevel + 1);
	body->Print(indentLevel + 1);
}

void PLoopStmt::performTypeInference(Scope *executionScope) {
	Scope *loopScope = executionScope->enter_scope(this->scope);
	for (int i = 0; i < rangeConditions->NumElements(); i++) {
		IndexRangeCondition *cond = rangeConditions->Nth(i);
		cond->inferTypes(loopScope);
	}
	body->performTypeInference(loopScope);
}

void PLoopStmt::checkSemantics(Scope *executionScope, bool ignoreTypeFailures) {
	Scope *loopScope = executionScope->enter_scope(new Scope(StatementBlockScope));
	for (int i = 0; i < rangeConditions->NumElements(); i++) {
		IndexRangeCondition *cond = rangeConditions->Nth(i);
		cond->resolveTypes(loopScope, ignoreTypeFailures);
	}
	body->checkSemantics(loopScope, ignoreTypeFailures);
	loopScope->detach_from_parent();
	this->scope = loopScope;
}

//--------------------------------------- Sequential Loop -------------------------------------------/
SLoopStmt::SLoopStmt(Identifier *i, Expr *re, Expr *se, Stmt *b, yyltype loc) : LoopStmt(loc) {
	Assert(i != NULL && re != NULL && b != NULL);
	id = i;
	id->SetParent(this);
	rangeExpr = re;
	rangeExpr->SetParent(this);
	stepExpr = se;
	if (stepExpr != NULL) {
		stepExpr->SetParent(this);
	}
	body = b;
	body->SetParent(this);
}
    	
void SLoopStmt::PrintChildren(int indentLevel) {
	id->Print(indentLevel + 1, "(Index) ");
	rangeExpr->Print(indentLevel + 1, "(Range) ");
	if (stepExpr != NULL) stepExpr->Print(indentLevel + 1, "(Step) ");
	body->Print(indentLevel + 1);
}

void SLoopStmt::performTypeInference(Scope *executionScope) {
	Scope *loopScope = executionScope->enter_scope(this->scope);
	rangeExpr->inferType(loopScope, Type::rangeType);
	if (stepExpr != NULL) stepExpr->inferType(loopScope, Type::intType);
	body->performTypeInference(loopScope);
}

void SLoopStmt::checkSemantics(Scope *executionScope, bool ignoreTypeFailures) {

	Scope *loopScope = executionScope->enter_scope(new Scope(StatementBlockScope));
	if (loopScope->lookup(id->getName()) != NULL) {
		// report error
	} else {
		VariableSymbol *var = new VariableSymbol(new VariableDef(id, Type::intType));
		loopScope->insert_symbol(var);
	}
	rangeExpr->resolveType(loopScope, ignoreTypeFailures);
	if (stepExpr != NULL) stepExpr->resolveType(loopScope, ignoreTypeFailures);

	body->checkSemantics(loopScope, ignoreTypeFailures);
	loopScope->detach_from_parent();
	this->scope = loopScope;
}

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

void WhileStmt::performTypeInference(Scope *executionScope) {
	condition->inferType(executionScope, Type::boolType);
	body->performTypeInference(executionScope);
}

void WhileStmt::checkSemantics(Scope *executionScope, bool ignoreTypeFailures) {
	condition->resolveType(executionScope, ignoreTypeFailures);
	body->checkSemantics(executionScope, ignoreTypeFailures);
}


