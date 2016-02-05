#include "ast.h"
#include "ast_stmt.h"
#include "ast_expr.h"
#include "ast_type.h"
#include "list.h"
#include "string.h"


IntConstant::IntConstant(yyltype loc, int val) : Expr(loc) {
    	value = val;
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

CharacterConstant::CharacterConstant(yyltype loc, char val) : Expr(loc) {
    	value = val;
}

void CharacterConstant::PrintChildren(int indentLevel) {
    	printf("%c",value);
}

IndexRangeExpr::IndexRangeExpr(Identifier *i, Identifier *a) : Expr(*i->GetLocation()) {
	Assert(i != NULL && a != NULL);
	index = i;
	array = a;
}

void IndexRangeExpr::PrintChildren(int indentLevel) {
	index->Print(indentLevel);
	array->Print(indentLevel);
}

SubRangeExpr::SubRangeExpr(Expr *b, Expr *e, yyltype loc) : Expr(loc) {
	begin = b;
	end = e;
	fullRange = (b == NULL && e == NULL);		
}

void SubRangeExpr::PrintChildren(int indentLevel) {
	if (begin != NULL) begin->Print(indentLevel + 1);
	if (end != NULL) end->Print(indentLevel + 1);
}

ArithmaticExpr::ArithmaticExpr(Expr *l, ArithmaticOperator o, Expr *r, yyltype loc) : Expr(loc) {
	Assert(l != NULL && r != NULL);
	left = l;
	op = o;
	right = r;
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
	}
	left->Print(indentLevel + 1);
	right->Print(indentLevel + 1);
}

LogicalExpr::LogicalExpr(Expr *l, LogicalOperator o, Expr *r, yyltype loc) : Expr(loc) {
	Assert(r != NULL);
	left = l;
	op = o;
	right = r;
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

ReductionExpr::ReductionExpr(Expr *l, ReductionOperator o, Expr *r, yyltype loc) : Expr(loc) {
	Assert(l != NULL && r != NULL);
	left = l;
	op = o;
	right = r;
}

void ReductionExpr::PrintChildren(int indentLevel) {
	PrintLabel(indentLevel + 1, "Operator");
	switch (op) {
		case SUM: printf("Sum"); break;
		case PRODUCT: printf("Product"); break;
		case MAX: printf("Maximum"); break;
		case MIN: printf("Minimum"); break;
		case AVG: printf("Average"); break;
		case MIN_ENTRY: printf("Minimum Entry"); break;
		case MAX_ENTRY: printf("Maximum Entry"); break;
	}
	left->Print(indentLevel + 1);
	right->Print(indentLevel + 1);
}

EpochValue::EpochValue(Identifier *e, int l) : Expr(*e->GetLocation()) {
	Assert(e != NULL);
	epoch = e;
	lag = l;
}

void EpochValue::PrintChildren(int indentLevel) {
	epoch->Print(indentLevel + 1);
	printf(" - %d", lag);
}

EpochExpr::EpochExpr(Expr *r, EpochValue *e) : Expr(*r->GetLocation()) {
	Assert(r != NULL && e != NULL);
	root = r;
	epoch = e;
}

void EpochExpr::PrintChildren(int indentLevel) {
	root->Print(indentLevel + 1, "(RootExpr) ");
	epoch->Print(indentLevel + 1, "(Epoch) ");
}

FieldAccess::FieldAccess(Expr *b, Identifier *f, yyltype loc) : Expr(loc) {
	Assert(f != NULL);
	base = b;
	field = f;
}

void FieldAccess::PrintChildren(int indentLevel) {
	if(base != NULL) base->Print(indentLevel + 1);
	field->Print(indentLevel + 1);
}

RangeExpr::RangeExpr(Identifier *i, Expr *r, Expr *s, yyltype loc) : Expr(loc) {
	Assert(i != NULL && r != NULL);
	index = i;
	range = r;
	step = s;
}

void RangeExpr::PrintChildren(int indentLevel) {
	index->Print(indentLevel + 1, "(Index) ");
	range->Print(indentLevel + 1, "(Range) ");
	if (step != NULL) step->Print(indentLevel + 1, "(Step) ");
}

SubpartitionRangeExpr::SubpartitionRangeExpr(char s, yyltype loc) : Expr(loc) {
	spaceId = s;
}

void SubpartitionRangeExpr::PrintChildren(int indentLevel) {
	printf("Space %c", spaceId);
}

AssignmentExpr::AssignmentExpr(Expr *l, Expr *r, yyltype loc) : Expr(loc) {
	Assert(l != NULL && r != NULL);
	left = l;
	right = r;
}

void AssignmentExpr::PrintChildren(int indentLevel) {
	left->Print(indentLevel + 1);
	right->Print(indentLevel + 1);
}

ArrayAccess::ArrayAccess(Expr *b, Expr *i, yyltype loc) : Expr(loc) {
	Assert(b != NULL && i != NULL);
	base = b;
	index = i;
}

void ArrayAccess::PrintChildren(int indentLevel) {
	base->Print(indentLevel + 1, "(Base) ");
	index->Print(indentLevel + 1, "(Index) ");
}

FunctionCall::FunctionCall(Identifier *b, List<Expr*> *a, yyltype loc) : Expr(loc) {
	Assert(b != NULL && a != NULL);
	base = b;
	arguments = a;
}

void FunctionCall::PrintChildren(int indentLevel) {
	base->Print(indentLevel + 1, "(Name) ");
	PrintLabel(indentLevel + 1, "Arguments");
	arguments->PrintAll(indentLevel + 2);
}

OptionalInvocationParams::OptionalInvocationParams(Identifier *s, List<Expr*> *a, yyltype loc) : Expr(loc) {
	Assert(s != NULL && a != NULL);
	section = s;
	arguments = a;
}

const char *OptionalInvocationParams::InitializeSection = "Initialize";
const char *OptionalInvocationParams::PartitionSection = "Partition";

void OptionalInvocationParams::PrintChildren(int indentLevel) {
	section->Print(indentLevel + 1, "(Section) ");
	PrintLabel(indentLevel + 1, "Arguments");
	arguments->PrintAll(indentLevel + 2);
}

TaskInvocation::TaskInvocation(Identifier *n, Identifier *e, 
		List<OptionalInvocationParams*> *o, yyltype loc) : Expr(loc) {
	Assert(n != NULL && e != NULL && o != NULL);
	taskName = n;
	environment = e;
	optionalArgs = o;
}

void TaskInvocation::PrintChildren(int indentLevel) {
	taskName->Print(indentLevel + 1, "(Task) ");
	environment->Print(indentLevel + 1, "(Environment) ");
	optionalArgs->PrintAll(indentLevel + 1);
}

ObjectCreate::ObjectCreate(Type *o, List<Expr*> *i, yyltype loc) : Expr(loc) {
	Assert(o != NULL && i != NULL);
	objectType = o;
	initArgs = i;
}

void ObjectCreate::PrintChildren(int indentLevel) {
	objectType->Print(indentLevel + 1);
	PrintLabel(indentLevel + 1, "InitializationArgs");
	initArgs->PrintAll(indentLevel + 2);
}

    	
