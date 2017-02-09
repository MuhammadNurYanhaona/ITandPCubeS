#include "ast.h"
#include "ast_stmt.h"
#include "ast_expr.h"
#include "ast_type.h"
#include "../../../common-libs/utils/list.h"
#include "../../../common-libs/utils/hashtable.h"

#include <iostream>
#include <sstream>
#include <cstdlib>

//----------------------------------------------- Constant Expression -------------------------------------------------/

IntConstant::IntConstant(yyltype loc, int val) : Expr(loc) {
        value = val;
	size = TWO_BYTES;
}

IntConstant::IntConstant(yyltype loc, int value, IntSize size) : Expr(loc) {
        this->value = value;
        this->size = size;
}

void IntConstant::PrintChildren(int indentLevel) {
        printf("%d", value);
}

FloatConstant::FloatConstant(yyltype loc, float val) : Expr(loc) {
        value = val;
}

void FloatConstant::PrintChildren(int indentLevel) {
        printf("%f", value);
}

DoubleConstant::DoubleConstant(yyltype loc, double val) : Expr(loc) {
        value = val;
}

void DoubleConstant::PrintChildren(int indentLevel) {
        printf("%g", value);
}

BoolConstant::BoolConstant(yyltype loc, bool val) : Expr(loc) {
        value = val;
}

void BoolConstant::PrintChildren(int indentLevel) {
        printf("%s", value ? "true" : "false");
}

StringConstant::StringConstant(yyltype loc, const char *val) : Expr(loc) {
        Assert(val != NULL);
        value = strdup(val);
}

void StringConstant::PrintChildren(int indentLevel) {
        printf("%s",value);
}

CharConstant::CharConstant(yyltype loc, char val) : Expr(loc) {
        value = val;
}

void CharConstant::PrintChildren(int indentLevel) {
        printf("%c",value);
}

ReductionVar::ReductionVar(char spaceId, const char *name, yyltype loc) : Expr(loc) {
	Assert(name != NULL);
	this->spaceId = spaceId;
	this->name = name;
}

void ReductionVar::PrintChildren(int indentLevel) {
	printf("Space %c:%s", spaceId, name);
}

//---------------------------------------------- Arithmatic Expression ------------------------------------------------/

ArithmaticExpr::ArithmaticExpr(Expr *l, ArithmaticOperator o, Expr *r, yyltype loc) : Expr(loc) {
        Assert(l != NULL && r != NULL);
        left = l;
        left->SetParent(this);
        op = o;
        right = r;
        right->SetParent(this);
}

void ArithmaticExpr::PrintChildren(int indentLevel) {
        PrintLabel(indentLevel + 1, "Operator");
        switch (op) {
                case ADD: printf("+"); break;
                case SUBTRACT: printf("-"); break;
                case MULTIPLY: printf("*"); break;
                case DIVIDE: printf("/"); break;
                case MODULUS: printf("%c", '%'); break;
                case LEFT_SHIFT: printf("<<"); break;
                case RIGHT_SHIFT: printf(">>"); break;
                case POWER: printf("**"); break;
                case BITWISE_AND: printf("&"); break;
                case BITWISE_XOR: printf("^"); break;
                case BITWISE_OR: printf("|"); break;
        }
        left->Print(indentLevel + 1);
        right->Print(indentLevel + 1);
}

//----------------------------------------------- Logical Expression -------------------------------------------------/

LogicalExpr::LogicalExpr(Expr *l, LogicalOperator o, Expr *r, yyltype loc) : Expr(loc) {
        Assert(r != NULL);
        left = l;
        if (left != NULL) {
                left->SetParent(this);
        }
        op = o;
        right = r;
        right->SetParent(this);
}

void LogicalExpr::PrintChildren(int indentLevel) {
        PrintLabel(indentLevel + 1, "Operator");
        switch (op) {
                case AND: printf("&&"); break;
                case OR: printf("||"); break;
                case NOT: printf("!"); break;
                case EQ: printf("=="); break;
                case NE: printf("!="); break;
                case GT: printf(">"); break;
                case LT: printf("<"); break;
                case GTE: printf(">="); break;
                case LTE: printf("<="); break;
        }
        if (left != NULL) left->Print(indentLevel + 1);
        right->Print(indentLevel + 1);
}


//----------------------------------------------- Reduction Expression ------------------------------------------------/

EpochExpr::EpochExpr(Expr *r, int lag) : Expr(*r->GetLocation()) {
        Assert(r != NULL && lag >= 0);
        root = r;
        root->SetParent(root);
        this->lag = lag;
}

void EpochExpr::PrintChildren(int indentLevel) {
        root->Print(indentLevel + 1, "(RootExpr) ");
        PrintLabel(indentLevel + 1, "Lag ");
	printf("%d", lag);
}

//-------------------------------------------------- Field Access -----------------------------------------------------/

FieldAccess::FieldAccess(Expr *b, Identifier *f, yyltype loc) : Expr(loc) {
        Assert(f != NULL);
        base = b;
        if (base != NULL) {
                base->SetParent(this);
        }
        field = f;
        field->SetParent(this);
}

void FieldAccess::PrintChildren(int indentLevel) {
        if(base != NULL) base->Print(indentLevel + 1);
        field->Print(indentLevel + 1);
}

//----------------------------------------------- Range Expressions --------------------------------------------------/

RangeExpr::RangeExpr(Identifier *i, Expr *r, Expr *s, bool l, yyltype loc) : Expr(loc) {
        Assert(i != NULL && r != NULL);
        index = i;
        index->SetParent(this);
        range = r;
        range->SetParent(this);
        step = s;
        if (step != NULL) {
                step->SetParent(this);
        }
        loopingRange = l;
}

void RangeExpr::PrintChildren(int indentLevel) {
        index->Print(indentLevel + 1, "(Index) ");
        range->Print(indentLevel + 1, "(Range) ");
        if (step != NULL) step->Print(indentLevel + 1, "(Step) ");
}

//--------------------------------------------- Assignment Expression ------------------------------------------------/

AssignmentExpr::AssignmentExpr(Expr *l, Expr *r, yyltype loc) : Expr(loc) {
        Assert(l != NULL && r != NULL);
        left = l;
        left->SetParent(this);
        right = r;
        right->SetParent(this);
}

void AssignmentExpr::PrintChildren(int indentLevel) {
        left->Print(indentLevel + 1);
        right->Print(indentLevel + 1);
}

//--------------------------------------------------- Array Access ----------------------------------------------------/

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

ArrayAccess::ArrayAccess(Expr *b, Expr *i, yyltype loc) : Expr(loc) {
        Assert(b != NULL && i != NULL);
        base = b;
        base->SetParent(this);
        index = i;
        index->SetParent(this);
}

void ArrayAccess::PrintChildren(int indentLevel) {
        base->Print(indentLevel + 1, "(Base) ");
        index->Print(indentLevel + 1, "(Index) ");
}

//------------------------------------------------- Function Call ----------------------------------------------------/

FunctionCall::FunctionCall(Identifier *b, List<Expr*> *a, yyltype loc) : Expr(loc) {
        Assert(b != NULL && a != NULL);
        base = b;
        base->SetParent(this);
        arguments = a;
        for (int i = 0; i <arguments->NumElements(); i++) {
                Expr *expr = arguments->Nth(i);
                expr->SetParent(this);
        }
}

void FunctionCall::PrintChildren(int indentLevel) {
        base->Print(indentLevel + 1, "(Name) ");
        PrintLabel(indentLevel + 1, "Arguments");
        arguments->PrintAll(indentLevel + 2);
}

//------------------------------------------------ Named Argument ----------------------------------------------------/

NamedArgument::NamedArgument(char *argName, Expr *argValue, yyltype loc) : Node(loc) {
	Assert(argName != NULL && argValue != NULL);
	this->argName = argName;
	this->argValue = argValue;
	this->argValue->SetParent(this);
}

void NamedArgument::PrintChildren(int indentLevel) {
	argValue->Print(indentLevel, argName);
}

//----------------------------------------------- Task Invocation ----------------------------------------------------/

TaskInvocation::TaskInvocation(List<NamedArgument*> *invocationArgs, yyltype loc) : Expr(loc) {
	Assert(invocationArgs != NULL);
	this->invocationArgs = invocationArgs;
}

void TaskInvocation::PrintChildren(int indentLevel) {
	PrintLabel(indentLevel + 1, "Arguments");
	invocationArgs->PrintAll(indentLevel + 2);
}

//------------------------------------------------ Object Create -----------------------------------------------------/

ObjectCreate::ObjectCreate(Type *o, List<NamedArgument*> *i, yyltype loc) : Expr(loc) {
        Assert(o != NULL && i != NULL);
        objectType = o;
        objectType->SetParent(this);
        initArgs = i;
        for (int j = 0; j < initArgs->NumElements(); j++) {
                initArgs->Nth(j)->SetParent(this);
        }
}

void ObjectCreate::PrintChildren(int indentLevel) {
        objectType->Print(indentLevel + 1);
        PrintLabel(indentLevel + 1, "Init-Arguments");
        initArgs->PrintAll(indentLevel + 2);
}
