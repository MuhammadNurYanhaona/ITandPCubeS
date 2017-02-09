#include "ast.h"
#include "ast_stmt.h"
#include "ast_expr.h"
#include "ast_def.h"
#include "ast_task.h"
#include "../common/location.h"
#include "../common/errors.h"
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

//------------------------------------------------------------ Loop Statement ----------------------------------------------------------/

LoopStmt::LoopStmt() : Stmt() {}

LoopStmt::LoopStmt(Stmt *body, yyltype loc) : Stmt(loc) {
	Assert(body != NULL);
	this->body = body;
	this->body->SetParent(this);
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

//------------------------------------------------------- Sequential For Loop --------------------------------------------------------/

SLoopAttribute::SLoopAttribute(Expr *range, Expr *step, Expr *restriction) {
        Assert(range != NULL);
        this->range = range;
        this->step = step;
        this->restriction = restriction;
}

SLoopStmt::SLoopStmt(Identifier *i, SLoopAttribute *attr, Stmt *b, yyltype loc) : LoopStmt(b, loc) {
        Assert(i != NULL && attr != NULL);
        id = i;
        id->SetParent(this);
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

