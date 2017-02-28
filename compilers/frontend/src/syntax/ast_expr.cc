#include "ast.h"
#include "ast_stmt.h"
#include "ast_expr.h"
#include "ast_type.h"
#include "../common/constant.h"
#include "../../../common-libs/utils/list.h"
#include "../../../common-libs/utils/hashtable.h"

#include <iostream>
#include <sstream>
#include <cstdlib>

//--------------------------------------------------- Expression ------------------------------------------------------/

void Expr::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	if (this->getExprTypeId() == typeId) {
		exprList->Append(this);
	}
}

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

Node *ArithmaticExpr::clone() {
	Expr *newLeft = (Expr*) left->clone();
	Expr *newRight = (Expr*) right->clone();
	return new ArithmaticExpr(newLeft, op, newRight, *GetLocation());
}

void ArithmaticExpr::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	if (typeId == getExprTypeId()) {
		Expr::retrieveExprByType(exprList, typeId);
	} else {
		left->retrieveExprByType(exprList, typeId);
		right->retrieveExprByType(exprList, typeId);
	}
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

Node *LogicalExpr::clone() {
	Expr *newRight = (Expr*) right->clone();
	if (left == NULL) return new LogicalExpr(NULL, op, newRight, *GetLocation());
	Expr *newLeft = (Expr*) left->clone();
	return new LogicalExpr(newLeft, op, newRight, *GetLocation());
}

void LogicalExpr::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	if (typeId == getExprTypeId()) {
		Expr::retrieveExprByType(exprList, typeId);
	} else {
		right->retrieveExprByType(exprList, typeId);
		if (left != NULL) left->retrieveExprByType(exprList, typeId);
	}
}

//------------------------------------------------- Epoch Expression --------------------------------------------------/

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

Node *EpochExpr::clone() {
	Expr *newRoot = (Expr*) root->clone();
	return new EpochExpr(newRoot, lag);
}

void EpochExpr::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	if (typeId == getExprTypeId()) {
		Expr::retrieveExprByType(exprList, typeId);
	} else {
		root->retrieveExprByType(exprList, typeId);
	}
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
	referenceField = false;
}

void FieldAccess::PrintChildren(int indentLevel) {
        if(base != NULL) base->Print(indentLevel + 1);
        field->Print(indentLevel + 1);
}

Node *FieldAccess::clone() {
	Expr *newBase = (Expr*) base->clone();
	Identifier *newField = (Identifier*) field->clone();
	FieldAccess *newFieldAcc = new FieldAccess(newBase, newField, *GetLocation());
	if (referenceField) {
		newFieldAcc->flagAsReferenceField();
	}
	return newFieldAcc;
}

void FieldAccess::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	Expr::retrieveExprByType(exprList, typeId);
	if (base != NULL) base->retrieveExprByType(exprList, typeId);
}

FieldAccess *FieldAccess::getTerminalField() {
	if (base == NULL) return this;
	FieldAccess *baseField = dynamic_cast<FieldAccess*>(baseField);
	if (baseField == NULL) return NULL;
	return baseField->getTerminalField();
}

//----------------------------------------------- Range Expressions --------------------------------------------------/

RangeExpr::RangeExpr(Identifier *i, Expr *r, Expr *s, yyltype loc) : Expr(loc) {
        Assert(i != NULL && r != NULL);
        index = new FieldAccess(NULL, i, *i->GetLocation());
        index->SetParent(this);
        range = r;
        range->SetParent(this);
        step = s;
        if (step != NULL) {
                step->SetParent(this);
        }
        loopingRange = true;
}

RangeExpr::RangeExpr(Expr *i, Expr *r, yyltype loc) : Expr(loc) {
        Assert(i != NULL && r != NULL);
	Assert(dynamic_cast<FieldAccess*>(i) != NULL);
        index = (FieldAccess*) i;
        index->SetParent(this);
        range = r;
        range->SetParent(this);
        step = NULL;
        loopingRange = false;
}

void RangeExpr::PrintChildren(int indentLevel) {
        index->Print(indentLevel + 1, "(Index) ");
        range->Print(indentLevel + 1, "(Range) ");
        if (step != NULL) step->Print(indentLevel + 1, "(Step) ");
}

Node *RangeExpr::clone() {
	Identifier *newId = (Identifier*) index->getField()->clone();
	Expr *newRange = (Expr*) range->clone();
	if (loopingRange) {
		Expr *newStep = NULL;
		if (step != NULL) {
			newStep = (Expr*) step->clone();
		}
		return new RangeExpr(newId, newRange, newStep, *GetLocation());
	}
	FieldAccess *newIndex = (FieldAccess*) index->clone(); 
	return new RangeExpr(newIndex, newRange, *GetLocation());
}

void RangeExpr::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	if (typeId == getExprTypeId()) {
		Expr::retrieveExprByType(exprList, typeId);
	} else {
		index->retrieveExprByType(exprList, typeId);
		range->retrieveExprByType(exprList, typeId);
		if (step != NULL) step->retrieveExprByType(exprList, typeId);
	}	
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

Node *AssignmentExpr::clone() {
	Expr *newLeft = (Expr*) left->clone();
	Expr *newRight = (Expr*) right->clone();
	return new AssignmentExpr(newLeft, newRight, *GetLocation());
}

void AssignmentExpr::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	if (typeId == getExprTypeId()) {
		Expr::retrieveExprByType(exprList, typeId);
	} else {
		left->retrieveExprByType(exprList, typeId);
		right->retrieveExprByType(exprList, typeId);
	}
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

//------------------------------------------------- Function Call ----------------------------------------------------/

FunctionCall::FunctionCall(Identifier *b, List<Expr*> *a, yyltype loc) : Expr(loc) {
        Assert(b != NULL && a != NULL);
        base = b;
        base->SetParent(this);
        arguments = a;
        for (int i = 0; i < arguments->NumElements(); i++) {
                Expr *expr = arguments->Nth(i);
                expr->SetParent(this);
        }
}

void FunctionCall::PrintChildren(int indentLevel) {
        base->Print(indentLevel + 1, "(Name) ");
        PrintLabel(indentLevel + 1, "Arguments");
        arguments->PrintAll(indentLevel + 2);
}

Node *FunctionCall::clone() {
	Identifier *newBase = (Identifier*) base->clone();
	List<Expr*> *newArgs = new List<Expr*>;
	for (int i = 0; i < arguments->NumElements(); i++) {
                Expr *expr = arguments->Nth(i);
		newArgs->Append((Expr*) expr->clone());
	}
	return new FunctionCall(newBase, newArgs, *GetLocation());
}

void FunctionCall::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	Expr::retrieveExprByType(exprList, typeId);
	for (int i = 0; i < arguments->NumElements(); i++) {
                Expr *arg = arguments->Nth(i);
		arg->retrieveExprByType(exprList, typeId);
	}	
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

Node *NamedArgument::clone() {
	char *newName = strdup(argName);
	Expr *newValue = (Expr*) argValue->clone();
	return new NamedArgument(newName, newValue, *GetLocation());
}

void NamedArgument::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	argValue->retrieveExprByType(exprList, typeId);
}

//--------------------------------------------- Named Multi-Argument -------------------------------------------------/

NamedMultiArgument::NamedMultiArgument(char *argName, List<Expr*> *argList, yyltype loc) : Node(loc) {
	Assert(argName != NULL && argList != NULL && argList->NumElements() > 0);
	this->argName = argName;
	this->argList = argList;
	for (int i = 0; i < argList->NumElements(); i++) {
		this->argList->Nth(i)->SetParent(this);
	}
}

void NamedMultiArgument::PrintChildren(int indentLevel) {
	PrintLabel(indentLevel + 1, argName);
	argList->PrintAll(indentLevel + 2);
}

Node *NamedMultiArgument::clone() {
	char *newName = strdup(argName);
	List<Expr*> *newArgList = new List<Expr*>;
	for (int i = 0; i < argList->NumElements(); i++) {
		Expr *arg = argList->Nth(i);
                newArgList->Append((Expr*) arg->clone());
        }
	return new NamedMultiArgument(newName, newArgList, *GetLocation());
}

void NamedMultiArgument::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	for (int i = 0; i < argList->NumElements(); i++) {
                Expr *arg = argList->Nth(i);
		arg->retrieveExprByType(exprList, typeId);
        }
}

//----------------------------------------------- Task Invocation ----------------------------------------------------/

TaskInvocation::TaskInvocation(List<NamedMultiArgument*> *invocationArgs, yyltype loc) : Expr(loc) {
	Assert(invocationArgs != NULL);
	this->invocationArgs = invocationArgs;
	for (int i = 0; i < invocationArgs->NumElements(); i++) {
		NamedMultiArgument *arg = invocationArgs->Nth(i);
		arg->SetParent(this);
	}
}

void TaskInvocation::PrintChildren(int indentLevel) {
	PrintLabel(indentLevel + 1, "Arguments");
	invocationArgs->PrintAll(indentLevel + 2);
}

Node *TaskInvocation::clone() {
	List<NamedMultiArgument*> *newInvokeArgs = new List<NamedMultiArgument*>;
	for (int i = 0; i < invocationArgs->NumElements(); i++) {
		NamedMultiArgument *arg = invocationArgs->Nth(i);
		newInvokeArgs->Append((NamedMultiArgument*) arg->clone());
	}
	return new TaskInvocation(newInvokeArgs, *GetLocation());
}

void TaskInvocation::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	if (typeId == getExprTypeId()) {
		Expr::retrieveExprByType(exprList, typeId);
	} else {
		for (int i = 0; i < invocationArgs->NumElements(); i++) {
			NamedMultiArgument *arg = invocationArgs->Nth(i);
			arg->retrieveExprByType(exprList, typeId);
		}
	}
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

Node *ObjectCreate::clone() {
	Type *newType = (Type*) objectType->clone();
	List<NamedArgument*> *newArgsList = new List<NamedArgument*>;
	for (int j = 0; j < initArgs->NumElements(); j++) {
                NamedArgument *arg = initArgs->Nth(j);
		newArgsList->Append((NamedArgument*) arg->clone());
        }
	return new ObjectCreate(newType, newArgsList, *GetLocation());
}

void ObjectCreate::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	Expr::retrieveExprByType(exprList, typeId);
	for (int j = 0; j < initArgs->NumElements(); j++) {
                NamedArgument *arg = initArgs->Nth(j);
		arg->retrieveExprByType(exprList, typeId);
	}
}
