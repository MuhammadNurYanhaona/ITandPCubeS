#include "list.h"
#include "ast.h"
#include "ast_stmt.h"
#include "ast_expr.h"

StmtBlock::StmtBlock(List<Stmt*> *s) : Stmt() {
	Assert(s != NULL);
	stmts = s;	
}
    	
void StmtBlock::PrintChildren(int indentLevel) {
	stmts->PrintAll(indentLevel + 1);
}
	
ConditionalStmt::ConditionalStmt(Expr *c, Stmt *s, yyltype loc) : Stmt(loc) {
	Assert(s != NULL);
	condition = c;
	stmt = s;
}

void ConditionalStmt::PrintChildren(int indentLevel) {
	if (condition != NULL) condition->Print(indentLevel, "(If) ");
	stmt->Print(indentLevel);
}

IfStmt::IfStmt(List<ConditionalStmt*> *ib, yyltype loc) : Stmt(loc) {
	Assert(ib != NULL);
	ifBlocks = ib;
}

void IfStmt::PrintChildren(int indentLevel) {
	ifBlocks->PrintAll(indentLevel + 1);
}
	
IndexRangeCondition::IndexRangeCondition(List<Identifier*> *i, Identifier *c, Expr *rs, yyltype loc) : Node(loc) {
	Assert(i != NULL && c != NULL);
	indexes = i;
	collection = c;
	restrictions = rs;
}

void IndexRangeCondition::PrintChildren(int indentLevel) {
	indexes->PrintAll(indentLevel + 1, "(Index) ");
	collection->Print(indentLevel + 1, "(Array/List) ");
	if (restrictions != NULL) restrictions->Print(indentLevel + 1, "(Restrictions) ");
}

PLoopStmt::PLoopStmt(List<IndexRangeCondition*> *rc, Stmt *b, yyltype loc) : Stmt(loc) {
	Assert(rc != NULL && b != NULL);
	rangeConditions = rc;
	body = b;
}

void PLoopStmt::PrintChildren(int indentLevel) {
	rangeConditions->PrintAll(indentLevel + 1);
	body->Print(indentLevel + 1);
}

SLoopStmt::SLoopStmt(Identifier *i, Expr *re, Expr *se, Stmt *b, yyltype loc) : Stmt(loc) {
	Assert(i != NULL && re != NULL && b != NULL);
	id = i;
	rangeExpr = re;
	stepExpr = se;
	body = b;
}
    	
void SLoopStmt::PrintChildren(int indentLevel) {
	id->Print(indentLevel + 1, "(Index) ");
	rangeExpr->Print(indentLevel + 1, "(Range) ");
	if (stepExpr != NULL) stepExpr->Print(indentLevel + 1, "(Step) ");
	body->Print(indentLevel + 1);
}

WhileStmt::WhileStmt(Expr *c, Stmt *b, yyltype loc) : Stmt(loc) {
	Assert(c != NULL && b != NULL);
	condition = c;
	body = b;
}	
    	
void WhileStmt::PrintChildren(int indentLevel) {
	condition->Print(indentLevel + 1, "(Condition) ");
	body->Print(indentLevel + 1);
}

