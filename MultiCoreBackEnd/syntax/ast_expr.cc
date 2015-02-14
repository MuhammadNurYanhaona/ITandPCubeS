#include "ast.h"
#include "ast_stmt.h"
#include "ast_expr.h"
#include "ast_type.h"

#include "../utils/list.h"
#include "string.h"
#include "../semantics/symbol.h"
#include "../semantics/task_space.h"
#include "errors.h"

#include "../utils/hashtable.h"
#include "../static-analysis/data_access.h"
#include "../static-analysis/loop_index.h"
#include "../codegen/name_transformer.h"

#include <iostream>
#include <sstream>
#include <cstdlib>

//----------------------------------------------------- Expression ----------------------------------------------------/

void Expr::performTypeInference(Scope *executionScope) {
	if (this->type == NULL) {
		resolveType(executionScope, true);
	}
	inferType(executionScope, this->type);	
}

void Expr::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {
        for (int i = 0; i < indentLevel; i++) stream << '\t';
        translate(stream, indentLevel, 0, space);
        stream << ";\n";
}

void Expr::translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space) {
	std::cout << "A sub-class of expression didn't implement the code generation method\n";
	std::exit(EXIT_FAILURE);
}

List<FieldAccess*> *Expr::getTerminalFieldAccesses() {
	return new List<FieldAccess*>;
}

void Expr::copyNewFields(List<FieldAccess*> *destination, List<FieldAccess*> *source) {
	for (int i = 0; i < source->NumElements(); i++) {
		FieldAccess *field = source->Nth(i);
		bool match = false;
		for (int j = 0; j < destination->NumElements(); j++) {
			if (destination->Nth(j)->isEqual(field)) {
				match = true;
				break;
			}
		}
		if (!match) destination->Append(field);
	}
}

//----------------------------------------------- Constant Expression -------------------------------------------------/

IntConstant::IntConstant(yyltype loc, int val) : Expr(loc) {
    	value = val;
}

void IntConstant::PrintChildren(int indentLevel) {
    	printf("%d", value);
}

void IntConstant::inferType(Scope *scope, Type *rootType) {
	if (!rootType->isAssignableFrom(Type::intType)) {
		ReportError::InferredAndActualTypeMismatch(
			this->GetLocation(), rootType, Type::intType, false);
	}
}

FloatConstant::FloatConstant(yyltype loc, float val) : Expr(loc) {
    	value = val;
}

void FloatConstant::PrintChildren(int indentLevel) {
    	printf("%f", value);
}

void FloatConstant::inferType(Scope *scope, Type *rootType) {
	if (!rootType->isAssignableFrom(Type::floatType)) {
		ReportError::InferredAndActualTypeMismatch(
			this->GetLocation(), rootType, Type::floatType, false);
	}
}

DoubleConstant::DoubleConstant(yyltype loc, double val) : Expr(loc) {
    	value = val;
}

void DoubleConstant::PrintChildren(int indentLevel) {
    	printf("%g", value);
}

void DoubleConstant::inferType(Scope *scope, Type *rootType) {
	if (!rootType->isAssignableFrom(Type::doubleType)) {
		ReportError::InferredAndActualTypeMismatch(
			this->GetLocation(), rootType, Type::doubleType, false);
	}
}

BoolConstant::BoolConstant(yyltype loc, bool val) : Expr(loc) {
    	value = val;
}

void BoolConstant::PrintChildren(int indentLevel) {
    	printf("%s", value ? "true" : "false");
}

void BoolConstant::inferType(Scope *scope, Type *rootType) {
	if (!rootType->isAssignableFrom(Type::boolType)) {
		ReportError::InferredAndActualTypeMismatch(
			this->GetLocation(), rootType, Type::boolType, false);
	}	
}

StringConstant::StringConstant(yyltype loc, const char *val) : Expr(loc) {
    	Assert(val != NULL);
    	value = strdup(val);
}

void StringConstant::PrintChildren(int indentLevel) {
    	printf("%s",value);
}

void StringConstant::inferType(Scope *scope, Type *rootType) {
	if (!rootType->isAssignableFrom(Type::stringType)) {
		ReportError::InferredAndActualTypeMismatch(
			this->GetLocation(), rootType, Type::stringType, false);
	}
}

CharacterConstant::CharacterConstant(yyltype loc, char val) : Expr(loc) {
    	value = val;
}

void CharacterConstant::PrintChildren(int indentLevel) {
    	printf("%c",value);
}

void CharacterConstant::inferType(Scope *scope, Type *rootType) {
	if (!rootType->isAssignableFrom(Type::charType)) {
		ReportError::InferredAndActualTypeMismatch(
			this->GetLocation(), rootType, Type::charType, false);
	}
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
	}
	left->Print(indentLevel + 1);
	right->Print(indentLevel + 1);
}

void ArithmaticExpr::resolveType(Scope *scope, bool ignoreFailure) {

	left->resolveType(scope, ignoreFailure);
	Type *leftType = left->getType();
	if (leftType == NULL) { 
		ReportError::UnknownExpressionType(left, ignoreFailure);
	}
	right->resolveType(scope, ignoreFailure);
	Type *rightType = right->getType();
	if (rightType == NULL) {
		ReportError::UnknownExpressionType(right, ignoreFailure);
	}

	if (leftType != NULL && !(leftType == Type::intType 
			|| leftType == Type::floatType 
			|| leftType == Type::doubleType 
			|| leftType == Type::charType
			|| leftType == Type::errorType)) {
		ReportError::UnsupportedOperand(left, leftType, "arithmatic expression", ignoreFailure);
	}
	if (rightType != NULL && !(rightType == Type::intType 
			|| rightType == Type::floatType 
			|| rightType == Type::doubleType 
			|| rightType == Type::charType
			|| rightType == Type::errorType)) {
		ReportError::UnsupportedOperand(right, rightType, "arithmatic expression", ignoreFailure);
	}

	if (leftType != NULL && rightType != NULL) {
		if (leftType->isAssignableFrom(rightType)) {
			this->type = leftType;
		} else if (rightType->isAssignableFrom(leftType)) {
			this->type = rightType;
		} else {
			ReportError::TypeMixingError(this, 
					leftType, rightType, "arithmatic expression", ignoreFailure);
			this->type = Type::errorType;
		}
	}
}

void ArithmaticExpr::inferType(Scope *scope, Type *rootType) {
	if (rootType == NULL) {		
		right->inferType(scope);
		left->inferType(scope);
		if (left->getType() != NULL && right->getType() == NULL) {
			right->inferType(scope, left->getType());
		} else if (right->getType() != NULL && left->getType() == NULL) {
			left->inferType(scope, right->getType());
		}
	} else if (this->type == NULL || rootType->isAssignableFrom(this->type)) {
		this->type = rootType;
		left->inferType(scope, rootType);
		right->inferType(scope, rootType);
	} else if (!rootType->isAssignableFrom(this->type)) {
		ReportError::InferredAndActualTypeMismatch(this->GetLocation(), 
				rootType, this->type, false);
	}
}

Hashtable<VariableAccess*> *ArithmaticExpr::getAccessedGlobalVariables(TaskGlobalReferences *globalReferences) {
	
	Hashtable<VariableAccess*> *table = right->getAccessedGlobalVariables(globalReferences);
	Hashtable<VariableAccess*> *lTable = left->getAccessedGlobalVariables(globalReferences);
	mergeAccessedVariables(table, lTable);	
	
	Iterator<VariableAccess*> iter = table->GetIterator();
	VariableAccess *accessLog;
	while((accessLog = iter.GetNextValue()) != NULL) {
		if(accessLog->isContentAccessed()) 
			accessLog->getContentAccessFlags()->flagAsRead();
		if (accessLog->isMetadataAccessed())
			accessLog->getMetadataAccessFlags()->flagAsRead();
	}
	return table;
}

void ArithmaticExpr::translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space) {
	if (op != POWER) {
		left->translate(stream, indentLevel, currentLineLength, space);
		switch (op) {
			case ADD: stream << " + "; break;
			case SUBTRACT: stream << " - "; break;
			case MULTIPLY: stream << " * "; break;
			case DIVIDE: stream << " / "; break;
			case MODULUS: stream << ' ' << '%' << ' '; break;
			case LEFT_SHIFT: stream <<" << "; break;
			case RIGHT_SHIFT: stream << " >> "; break;
			default: break;
		}
		right->translate(stream, indentLevel, currentLineLength, space);
	} else {
		stream << "pow(";
		left->translate(stream, indentLevel, currentLineLength, space);
		stream << ", ";
		right->translate(stream, indentLevel, currentLineLength, space);
	}
}

List<FieldAccess*> *ArithmaticExpr::getTerminalFieldAccesses() {
	List<FieldAccess*> *leftList = left->getTerminalFieldAccesses();
	Expr::copyNewFields(leftList, right->getTerminalFieldAccesses());
	return leftList;
}

//------------------------------------------------ Logical Expression -------------------------------------------------/

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

void LogicalExpr::resolveType(Scope *scope, bool ignoreFailure) {
	
	right->resolveType(scope, ignoreFailure);
	Type *rightType = right->getType();
	if (rightType == NULL) {
		ReportError::UnknownExpressionType(right, ignoreFailure);
	}

	Type *leftType = NULL;
	if (left != NULL) {
		left->resolveType(scope, ignoreFailure);
		leftType = left->getType();
		if (leftType == NULL) { 
			ReportError::UnknownExpressionType(left, ignoreFailure);
		} 
	}

	bool arithMaticOperator = (op == EQ || op == NE || op == GT || op == LT || op == GTE || op == LTE);
	if (arithMaticOperator) {
		if (leftType != NULL && !(leftType == Type::intType 
				|| leftType == Type::floatType 
				|| leftType == Type::doubleType 
				|| leftType == Type::charType
				|| leftType == Type::errorType)) {
			ReportError::UnsupportedOperand(left, leftType, 
					"logical expression", ignoreFailure);
		}
		if (rightType != NULL && !(rightType == Type::intType 
				|| rightType == Type::floatType 
				|| rightType == Type::doubleType 
				|| rightType == Type::charType
				|| rightType == Type::errorType)) {
			ReportError::UnsupportedOperand(right, rightType, 
					"logical expression", ignoreFailure);
		}
		if (leftType != NULL && rightType != NULL) {
			if (!leftType->isAssignableFrom(rightType) 
					&& !rightType->isAssignableFrom(leftType)) {
				ReportError::TypeMixingError(this, leftType, rightType, 
						"logical expression", ignoreFailure);
			}
		}		
	} else {
		if (!rightType->isAssignableFrom(Type::boolType)) {
			ReportError::IncompatibleTypes(right->GetLocation(), 
					rightType, Type::boolType, ignoreFailure);
		}
		if (leftType != NULL && !leftType->isAssignableFrom(Type::boolType)) {
			ReportError::IncompatibleTypes(left->GetLocation(), 
					leftType, Type::boolType, ignoreFailure);
		}
	}
	
	this->type = Type::boolType;
}

void LogicalExpr::inferType(Scope *scope, Type *rootType) {

	if (rootType != NULL && !rootType->isAssignableFrom(Type::boolType)) {
		ReportError::InferredAndActualTypeMismatch(this->GetLocation(), 
				rootType, Type::boolType, false);
	} else {
		bool arithMaticOperator = (op == EQ || op == NE || op == GT || op == LT || op == GTE || op == LTE);
		if (arithMaticOperator) {
			left->resolveType(scope, true);
			Type *leftType = left->getType();
			right->resolveType(scope, true);
			Type *rightType = right->getType();
			if (leftType == NULL && rightType != NULL) {
				left->inferType(scope, rightType);
			} else if (leftType != NULL && rightType == NULL) {
				right->inferType(scope, leftType);
			} else {
				left->inferType(scope);
				right->inferType(scope);
			}
		} else {
			right->inferType(scope, Type::boolType);
			if (left != NULL) left->inferType(scope, Type::boolType);
		}
	}
}

Hashtable<VariableAccess*> *LogicalExpr::getAccessedGlobalVariables(TaskGlobalReferences *globalReferences) {
	
	Hashtable<VariableAccess*> *table = right->getAccessedGlobalVariables(globalReferences);
	if (left == NULL) return table;
	Hashtable<VariableAccess*> *lTable = left->getAccessedGlobalVariables(globalReferences);
	mergeAccessedVariables(table, lTable);
	
	Iterator<VariableAccess*> iter = table->GetIterator();
	VariableAccess *accessLog;
	while((accessLog = iter.GetNextValue()) != NULL) {
		if(accessLog->isContentAccessed()) 
			accessLog->getContentAccessFlags()->flagAsRead();
		if (accessLog->isMetadataAccessed())
			accessLog->getMetadataAccessFlags()->flagAsRead();
	}
	return table;
}

void LogicalExpr::translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space) {
	if (left != NULL) {
		left->translate(stream, indentLevel, currentLineLength, space);
	}
	switch (op) {
		case AND: stream << " && "; break;
		case OR: stream << " || "; break;
		case NOT: stream << "!"; break;
		case EQ: stream << " == "; break;
		case NE: stream << " != "; break;
		case GT: stream << " > "; break;
		case LT: stream << " < "; break;
		case GTE: stream << " >= "; break;
		case LTE: stream << " <= "; break;
	}
	right->translate(stream, indentLevel, currentLineLength, space);
}

List<FieldAccess*> *LogicalExpr::getTerminalFieldAccesses() {
	if (left == NULL) return right->getTerminalFieldAccesses();
	List<FieldAccess*> *leftList = left->getTerminalFieldAccesses();
	Expr::copyNewFields(leftList, right->getTerminalFieldAccesses());
	return leftList;
}

// TODO: note that this is a simple implementation that does not appy any transformation to the logical expression
// that may allow it to express as a collective of AND operations. More elaborate implementation that does that may
// be attempted later if seems worthwhile.
List<LogicalExpr*> *LogicalExpr::getANDBreakDown() {
	List<LogicalExpr*> *breakDownList = new List<LogicalExpr*>;
	// it is a unary NOT statement and there is nothing to do here
	if (left == NULL) {
		breakDownList->Append(this); 
		return NULL;
	// if it is not an AND operation then we cannot break it down either
	} else if (op != AND) {
		breakDownList->Append(this);
	} else {
		LogicalExpr *leftLogic = dynamic_cast<LogicalExpr*>(left);
		breakDownList->AppendAll(leftLogic->getANDBreakDown());
		LogicalExpr *rightLogic = dynamic_cast<LogicalExpr*>(right);
		breakDownList->AppendAll(rightLogic->getANDBreakDown());
	}
	return breakDownList;
}

//----------------------------------------------- Reduction Expression ------------------------------------------------/

ReductionExpr::ReductionExpr(char *o, Expr *r, yyltype loc) : Expr(loc) {
	Assert(r != NULL && o != NULL);
	if (strcmp(o, "sum") == 0) op = SUM;
	else if (strcmp(o, "product") == 0) op = PRODUCT;
	else if (strcmp(o, "max") == 0) op = MAX;
	else if (strcmp(o, "maxEntry") == 0) op = MAX_ENTRY;
	else if (strcmp(o, "min") == 0) op = MIN;
	else if (strcmp(o, "minEntry") == 0) op = MIN_ENTRY;
	else if (strcmp(o, "avg") == 0) op = AVG;
	else {
		// Forcefully through a fault for now. Later we will add user defined reduction function, God willing.
		Assert(0 == 1);	
	}
	right = r;
	right->SetParent(this);
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
	right->Print(indentLevel + 1);
}

void ReductionExpr::resolveType(Scope *scope, bool ignoreFailure) {

	right->resolveType(scope, ignoreFailure);
	Type *rightType = right->getType();
	if (rightType == NULL) {
		ReportError::UnknownExpressionType(right, ignoreFailure);
	}

	if (rightType != NULL && rightType != Type::intType 
			&& rightType != Type::floatType
			&& rightType != Type::doubleType
			&& rightType != Type::errorType) {
		ReportError::UnsupportedOperand(right, rightType, 
				"reduction expression", ignoreFailure);
		this->type = Type::errorType;
	}

	if (op == MIN_ENTRY || op == MAX_ENTRY) {
			this->type = Type::intType;
	} else {
		this->type = rightType;
	}
}

void ReductionExpr::inferType(Scope *scope, Type *rootType) {

	if (op == MIN_ENTRY || op == MAX_ENTRY) {
		this->type = Type::intType;
		right->inferType(scope);
	} else {	
		if (rootType == NULL) { right->inferType(scope);
		} else if (this->type == NULL || rootType->isAssignableFrom(this->type)) {
			this->type = rootType;
			right->inferType(scope, rootType);
		} else if (!rootType->isAssignableFrom(this->type)) {
			ReportError::InferredAndActualTypeMismatch(this->GetLocation(), 
					rootType, this->type, false);
		}
	}
}

Hashtable<VariableAccess*> *ReductionExpr::getAccessedGlobalVariables(TaskGlobalReferences *globalReferences) {
	
	const char *rightVarName = right->getBaseVarName();
	ArrayAccess *rArray = dynamic_cast<ArrayAccess*>(right);
	if (rArray != NULL) {
		// Reduction of some array
		if (globalReferences->doesReferToGlobal(rightVarName)) {
			VariableSymbol *symbol = globalReferences->getGlobalRoot(rightVarName);
			rightVarName = symbol->getName();
		}
	}
	
	Hashtable<VariableAccess*> *table = right->getAccessedGlobalVariables(globalReferences);
	Iterator<VariableAccess*> iter = table->GetIterator();
	VariableAccess *accessLog;
	while ((accessLog = iter.GetNextValue()) != NULL) {
		if (rightVarName != NULL && strcmp(rightVarName, accessLog->getName()) == 0) {
			accessLog->getContentAccessFlags()->flagAsReduced();
		} else {
			if (accessLog->isMetadataAccessed()) 
				accessLog->getMetadataAccessFlags()->flagAsRead();
			if(accessLog->isContentAccessed()) 
				accessLog->getContentAccessFlags()->flagAsRead();
		}
	}
	return table;
}

List<FieldAccess*> *ReductionExpr::getTerminalFieldAccesses() { return right->getTerminalFieldAccesses(); }

//------------------------------------------------- Epoch Expression --------------------------------------------------/

EpochValue::EpochValue(Identifier *e, int l) : Expr(*e->GetLocation()) {
	Assert(e != NULL);
	epoch = e;
	epoch->SetParent(this);
	lag = l;
}

void EpochValue::PrintChildren(int indentLevel) {
	epoch->Print(indentLevel + 1);
	printf(" - %d", lag);
}

void EpochValue::resolveType(Scope *scope, bool ignoreFailure) {
	Symbol *symbol = scope->lookup(epoch->getName());
	if (symbol == NULL) {
		ReportError::UndefinedSymbol(epoch, ignoreFailure);
	} else {
		VariableSymbol *varSym = dynamic_cast<VariableSymbol*>(symbol);
		if (varSym == NULL) {
			ReportError::WrongSymbolType(epoch, "variable", ignoreFailure);
		} else {
			Type *symbolType = varSym->getType();
			if (symbolType != Type::epochType) {
				ReportError::IncompatibleTypes(GetLocation(), symbolType, 
						Type::epochType, ignoreFailure);
			}
		}
	}
	this->type = Type::epochType;	 
}

EpochExpr::EpochExpr(Expr *r, EpochValue *e) : Expr(*r->GetLocation()) {
	Assert(r != NULL && e != NULL);
	root = r;
	root->SetParent(root);
	epoch = e;
	epoch->SetParent(root);
}

void EpochExpr::PrintChildren(int indentLevel) {
	root->Print(indentLevel + 1, "(RootExpr) ");
	epoch->Print(indentLevel + 1, "(Epoch) ");
}

void EpochExpr::resolveType(Scope *scope, bool ignoreFailure) {
	root->resolveType(scope, ignoreFailure);
	this->type = root->getType();
	epoch->resolveType(scope, ignoreFailure);	
}

void EpochExpr::inferType(Scope *scope, Type *rootType) {
	if (rootType == NULL) {
		root->inferType(scope);
	} else if (this->type == NULL || rootType->isAssignableFrom(rootType)) {
		this->type = rootType;
		root->inferType(scope, rootType);	
	} else if (!rootType->isAssignableFrom(rootType)) {
		ReportError::InferredAndActualTypeMismatch(this->GetLocation(), rootType, 
				this->type, false);
	}
}

Hashtable<VariableAccess*> *EpochExpr::getAccessedGlobalVariables(TaskGlobalReferences *globalReferences) {
	Hashtable<VariableAccess*> *table = root->getAccessedGlobalVariables(globalReferences);
	Identifier *epochId = epoch->getId();
	if (globalReferences->doesReferToGlobal(epochId->getName())) {
		VariableSymbol *symbol = globalReferences->getGlobalRoot(epochId->getName());
		const char *globalName = symbol->getName();
		VariableAccess *accessLog = table->Lookup(globalName);
		if (accessLog != NULL) {
			accessLog->getContentAccessFlags()->flagAsRead();
		} else {
			accessLog = new VariableAccess(globalName);
			accessLog->markContentAccess();
			accessLog->getContentAccessFlags()->flagAsRead();
		}
		table->Enter(globalName, accessLog, true);
	}
	return table;
}

List<FieldAccess*> *EpochExpr::getTerminalFieldAccesses() { return root->getTerminalFieldAccesses(); }

//---------------------------------------------- Assignment Expression ------------------------------------------------/

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

void AssignmentExpr::resolveType(Scope *scope, bool ignoreFailure) {

	left->resolveType(scope, ignoreFailure);
	Type *leftType = left->getType();
	right->resolveType(scope, ignoreFailure);
	Type *rightType = right->getType();
	if (leftType == NULL && rightType == NULL) {
		ReportError::UnknownExpressionType(right, ignoreFailure);
	} 
	if (leftType != NULL && rightType != NULL && !leftType->isAssignableFrom(rightType)) {
		ReportError::TypeMixingError(this, leftType, rightType, 
				"assignment", ignoreFailure);
	}

	FieldAccess *fieldAccess = dynamic_cast<FieldAccess*>(left);
	ArrayAccess *arrayAccess = dynamic_cast<ArrayAccess*>(left);
	if (fieldAccess == NULL && arrayAccess == NULL) {
		EpochExpr *epochExpr = dynamic_cast<EpochExpr*>(left);
		if (epochExpr == NULL) {
			ReportError::NonLValueInAssignment(left, ignoreFailure);
		} else {
			Expr *epochRoot = epochExpr->getRootExpr();
			fieldAccess = dynamic_cast<FieldAccess*>(epochRoot);
			arrayAccess = dynamic_cast<ArrayAccess*>(epochRoot);
			if (fieldAccess == NULL && arrayAccess == NULL) {
				ReportError::NonLValueInAssignment(left, ignoreFailure);
			}
		}
	}
	
	if (!ignoreFailure) {
		if (leftType == NULL && rightType != NULL) {
			left->inferType(scope, rightType);
		} else if (leftType != NULL && rightType == NULL) {
			right->inferType(scope, leftType);
		}
	}

	if (leftType == NULL) {
		this->type = right->getType();
	} else {
		this->type = left->getType();
	}
}

void AssignmentExpr::inferType(Scope *scope, Type *rootType) {
	if (rootType != NULL) {
		left->inferType(scope, rootType);
		right->inferType(scope, rootType);
	} else {
		Type *leftType = left->getType();
		Type *rightType = right->getType();
		if (leftType == NULL && rightType != NULL) {
			left->inferType(scope, rightType);
		} else if (leftType != NULL && rightType == NULL) {
			right->inferType(scope, leftType);
		} else {
			left->inferType(scope);
			right->inferType(scope);
		}
	}
}

Hashtable<VariableAccess*> *AssignmentExpr::getAccessedGlobalVariables(TaskGlobalReferences *globalReferences) {
	const char* baseVarName = left->getBaseVarName();
	if (baseVarName == NULL) {
		ReportError::NotLValueinAssignment(GetLocation());
	}
	Hashtable<VariableAccess*> *table = left->getAccessedGlobalVariables(globalReferences);
	if (baseVarName != NULL && table->Lookup(baseVarName) != NULL) {
		VariableAccess *accessLog = table->Lookup(baseVarName);
		if(accessLog->isContentAccessed()) 
			accessLog->getContentAccessFlags()->flagAsWritten();
		if (accessLog->isMetadataAccessed())
			accessLog->getMetadataAccessFlags()->flagAsWritten();
	}

	Hashtable<VariableAccess*> *rTable = right->getAccessedGlobalVariables(globalReferences);
	Iterator<VariableAccess*> iter = rTable->GetIterator();
	VariableAccess *accessLog;
	bool reductionOnRight = (dynamic_cast<ReductionExpr*>(right) != NULL);
	while ((accessLog = iter.GetNextValue()) != NULL) {		
		if (!reductionOnRight) {
			if (accessLog->isMetadataAccessed()) accessLog->getMetadataAccessFlags()->flagAsRead();
			if(accessLog->isContentAccessed()) accessLog->getContentAccessFlags()->flagAsRead();
		}
		if (table->Lookup(accessLog->getName()) != NULL) {
			table->Lookup(accessLog->getName())->mergeAccessInfo(accessLog);
		} else {
			table->Enter(accessLog->getName(), accessLog, true);
		}
	}

	// check if any local/global reference is made to some global variable through the assignment expression and
	// take care of that
	FieldAccess *field = dynamic_cast<FieldAccess*>(left);
	if (field != NULL && field->isTerminalField()) {
		const char *rightSide = right->getBaseVarName();
		Type *rightType = right->getType();
		ArrayType *arrayType = dynamic_cast<ArrayType*>(rightType);
		if (rightSide != NULL && globalReferences->doesReferToGlobal(rightSide) && arrayType != NULL) {
			if (!globalReferences->isGlobalVariable(baseVarName)) {
				VariableSymbol *root = globalReferences->getGlobalRoot(rightSide);
				globalReferences->setNewReference(baseVarName, root->getName());
			} else {
				accessLog = table->Lookup(baseVarName);
				accessLog->markContentAccess();
				accessLog->getContentAccessFlags()->flagAsRedirected();
			}
		}
	}
	return table;
}

void AssignmentExpr::translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space) {
	left->translate(stream, indentLevel, currentLineLength, space);
	stream << " = ";
	right->translate(stream, indentLevel, currentLineLength, space);
}

void AssignmentExpr::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {

	// If the right is also an assignment expression then this is a compound statement. Break it up into
	// several simple statements.
	AssignmentExpr *rightExpr = dynamic_cast<AssignmentExpr*>(right);
	Expr *rightPart = right;
	if (rightExpr != NULL) {
		rightPart = rightExpr->left;
		rightExpr->generateCode(stream, indentLevel, space);	
	}

	// If this is a dimension assignment statement with multiple dimensions of an array assigned once to
	// another one then we need to handle it separately.
	Type *leftType = left->getType();
	FieldAccess *fieldAccess = dynamic_cast<FieldAccess*>(left);
	if (leftType == Type::dimensionType 
			&& fieldAccess != NULL 
			&& fieldAccess->getBase() != NULL) {
		
		ArrayType *array = dynamic_cast<ArrayType*>(fieldAccess->getBase()->getType());
		DimensionIdentifier *dimension = dynamic_cast<DimensionIdentifier*>(fieldAccess->getField());
		if (array != NULL && dimension != NULL 
				&& dimension->getDimensionNo() == 0
				&& array->getDimensions() > 1) {
			int dimensionCount = array->getDimensions();
			for (int i = 0; i < dimensionCount; i++) {
				for (int j = 0; j < indentLevel; j++) stream << '\t';
				left->translate(stream, indentLevel, 0, space);
				stream << '[' << i << "] = ";
				rightPart->translate(stream, indentLevel, 0, space);
				stream << '[' << i << "];\n";
			}
		// If the array is unidimensional then it follows the normal assignment procedure
		} else {
			for (int i = 0; i < indentLevel; i++) stream << '\t';
			left->translate(stream, indentLevel, 0, space);
			stream << " = ";
			rightPart->translate(stream, indentLevel, 0, space);
			stream << ";\n";
		}
	} else {
		for (int i = 0; i < indentLevel; i++) stream << '\t';
		left->translate(stream, indentLevel, 0, space);
		stream << " = ";
		rightPart->translate(stream, indentLevel, 0, space);
		stream << ";\n";
	}
}

List<FieldAccess*> *AssignmentExpr::getTerminalFieldAccesses() {
	List<FieldAccess*> *leftList = left->getTerminalFieldAccesses();
	Expr::copyNewFields(leftList, right->getTerminalFieldAccesses());
	return leftList;
}

//-------------------------------------------------- Function Call ----------------------------------------------------/

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

void FunctionCall::resolveType(Scope *scope, bool ignoreFailure) {
	Symbol *symbol = scope->lookup(base->getName());
	if (symbol == NULL) {
		ReportError::UndefinedSymbol(base, ignoreFailure);
		this->type = Type::errorType;
	} else {
		FunctionSymbol *fnSymbol = dynamic_cast<FunctionSymbol*>(symbol);
		if (fnSymbol == NULL) {
			ReportError::WrongSymbolType(base, "function", ignoreFailure);
			this->type = Type::errorType;
		} else {
			List<Type*> *argumentTypes = fnSymbol->getArgumentTypes();
			if (argumentTypes->NumElements() != arguments->NumElements() && !ignoreFailure) {
				ReportError::TooFewOrTooManyParameters(base, arguments->NumElements(), 
						argumentTypes->NumElements(), ignoreFailure);
			} else {
				for (int i = 0; i < arguments->NumElements(); i++) {
					Expr *currentArg = arguments->Nth(i);
					currentArg->resolveType(scope, ignoreFailure);
					if (currentArg->getType() == NULL) {
						currentArg->inferType(scope, argumentTypes->Nth(i));
					} else if (!currentArg->getType()->isAssignableFrom(argumentTypes->Nth(i)) 
								&& !ignoreFailure) {
						ReportError::IncompatibleTypes(currentArg->GetLocation(), 
								currentArg->getType(), 
								argumentTypes->Nth(i), ignoreFailure);
					}
				}
			}
			// We have an special treatment for functions with a single return value
			// There results can be directly used in expressions	
			TupleDef *tuple = fnSymbol->getReturnType();
			List<VariableDef*> *resultComponents = tuple->getComponents();
			if (resultComponents->NumElements() == 1) {
				this->type = resultComponents->Nth(0)->getType();
			} else {
				this->type = new NamedType(tuple->getId());
			}
		}
	}
}

Hashtable<VariableAccess*> *FunctionCall::getAccessedGlobalVariables(TaskGlobalReferences *globalReferences) {
	Hashtable<VariableAccess*> *table = new Hashtable<VariableAccess*>;
	for (int i = 0; i < arguments->NumElements(); i++) {
		Expr *expr = arguments->Nth(i);
		Hashtable<VariableAccess*> *argTable = expr->getAccessedGlobalVariables(globalReferences);
		Iterator<VariableAccess*> iter = argTable->GetIterator();
		VariableAccess *accessLog;
		while ((accessLog = iter.GetNextValue()) != NULL) {
			if (table->Lookup(accessLog->getName()) != NULL) {
				table->Lookup(accessLog->getName())->mergeAccessInfo(accessLog);
			} else {
				table->Enter(accessLog->getName(), accessLog, true);
			}
		}
	}
	return table;	
}

List<FieldAccess*> *FunctionCall::getTerminalFieldAccesses() {
	List<FieldAccess*> *list = Expr::getTerminalFieldAccesses();
	for (int i = 0; i < arguments->NumElements(); i++) {
		Expr *expr = arguments->Nth(i);
		Expr::copyNewFields(list, expr->getTerminalFieldAccesses());
	}
	return list;
}

//------------------------------------------------- Task Invocation ---------------------------------------------------/

OptionalInvocationParams::OptionalInvocationParams(Identifier *s, List<Expr*> *a, yyltype loc) : Node(loc) {
	Assert(s != NULL && a != NULL);
	section = s;
	section->SetParent(this);
	arguments = a;
	for (int i = 0; i < arguments->NumElements(); i++) {
		Expr *expr = arguments->Nth(i);
		expr->SetParent(this);
	}
}

const char *OptionalInvocationParams::InitializeSection = "Initialize";
const char *OptionalInvocationParams::PartitionSection = "Partition";

void OptionalInvocationParams::PrintChildren(int indentLevel) {
	section->Print(indentLevel + 1, "(Section) ");
	PrintLabel(indentLevel + 1, "Arguments");
	arguments->PrintAll(indentLevel + 2);
}

void OptionalInvocationParams::validateTypes(Scope *scope, TaskDef *taskDef, bool ignoreFailure) {
	
	if (section->getName() == InitializeSection) {
		List<Type*> *initArgTypes = taskDef->getInitArgTypes();
		if (initArgTypes->NumElements() != arguments->NumElements()) {
			ReportError::TooFewOrTooManyParameters(section, arguments->NumElements(), 
					initArgTypes->NumElements(), ignoreFailure);
		} else {
			for (int i = 0; i < arguments->NumElements(); i++) {
				Expr *expr = arguments->Nth(i);
				Type *type = initArgTypes->Nth(i);
				expr->resolveType(scope, true);
				if (expr->getType() == NULL) {
					expr->inferType(scope, type);
				} else if (!type->isAssignableFrom(expr->getType())) {
					ReportError::IncompatibleTypes(expr->GetLocation(), 
							expr->getType(), type, ignoreFailure);
				}
			}
		}		
	} else if (section->getName() == PartitionSection) {
		int argsCount = taskDef->getPartitionArgsCount();
		if (argsCount != arguments->NumElements()) {
			ReportError::TooFewOrTooManyParameters(section, arguments->NumElements(), 
					argsCount, ignoreFailure);
		}
		for (int i = 0; i < arguments->NumElements(); i++) {
			Expr *expr = arguments->Nth(i);
			expr->resolveType(scope, true);
			if (expr->getType() == NULL) {
				expr->inferType(scope, Type::intType);
			} else if (!Type::intType->isAssignableFrom(expr->getType())) {
				ReportError::IncompatibleTypes(expr->GetLocation(), 
						expr->getType(), Type::intType, ignoreFailure);
			}
		}
	}	
}

TaskInvocation::TaskInvocation(Identifier *n, Identifier *e, 
		List<OptionalInvocationParams*> *o, yyltype loc) : Expr(loc) {
	Assert(n != NULL && e != NULL && o != NULL);
	taskName = n;
	taskName->SetParent(this);
	environment = e;
	environment->SetParent(this);
	optionalArgs = o;
	for (int i = 0; i < optionalArgs->NumElements(); i++) {
		optionalArgs->Nth(i)->SetParent(this);
	}
}

void TaskInvocation::PrintChildren(int indentLevel) {
	taskName->Print(indentLevel + 1, "(Task) ");
	environment->Print(indentLevel + 1, "(Environment) ");
	optionalArgs->PrintAll(indentLevel + 1);
}

void TaskInvocation::resolveType(Scope *scope, bool ignoreFailure) {
	this->type = Type::errorType;
	Symbol *symbol = scope->lookup(taskName->getName());
	if (symbol == NULL) {
		ReportError::UndefinedSymbol(taskName, ignoreFailure);
	} else {
		TaskSymbol *task = dynamic_cast<TaskSymbol*>(symbol);
		if (task == NULL) {
			ReportError::WrongSymbolType(taskName, "Task", ignoreFailure);
		} else {
			TaskDef *taskDef = (TaskDef*) task->getNode();	
			TupleDef *envTuple = taskDef->getEnvTuple();
			Symbol *symbol = scope->lookup(environment->getName());
			VariableSymbol *varSym = dynamic_cast<VariableSymbol*>(symbol);
			if (varSym == NULL) {
				ReportError::WrongSymbolType(environment, "Variable", ignoreFailure);
			} else {
				Type *type = varSym->getType();
				Type *expected = new NamedType(envTuple->getId());
				if (!type->isEqual(expected)) {
					ReportError::IncompatibleTypes(environment->GetLocation(), type, 
							expected, ignoreFailure);
				}
			}
			for (int i = 0; i < optionalArgs->NumElements(); i++) {
				OptionalInvocationParams *params = optionalArgs->Nth(i);
				params->validateTypes(scope, taskDef, ignoreFailure);	
			}	
		}
	}
}

//------------------------------------------------- Object Creation ---------------------------------------------------/

ObjectCreate::ObjectCreate(Type *o, List<Expr*> *i, yyltype loc) : Expr(loc) {
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
	PrintLabel(indentLevel + 1, "InitializationArgs");
	initArgs->PrintAll(indentLevel + 2);
}

void ObjectCreate::resolveType(Scope *scope, bool ignoreFailure) {
	
	const char* typeName = objectType->getName();
	
	// first we check if the object corresponds to the task environment of some task; we need special logic here
	if (strcmp(typeName, "TaskEnvironment") == 0) {
		if (initArgs->NumElements() != 1) {
			ReportError::TaskNameRequiredInEnvironmentCreate(objectType->GetLocation(), ignoreFailure);
			this->type = Type::errorType;
		} else {
			StringConstant *str = dynamic_cast<StringConstant*>(initArgs->Nth(0));
			if (str == NULL) {
				Expr *arg = initArgs->Nth(0);
				Type *argType = arg->getType();
				if (argType != NULL) {
					ReportError::IncompatibleTypes(arg->GetLocation(), argType, 
							Type::stringType, ignoreFailure);
					this->type = Type::errorType;
				}
			} else {
				Symbol *symbol = scope->lookup(str->getValue());
				if (symbol == NULL) {
					ReportError::UndefinedTask(str->GetLocation(), str->getValue(), ignoreFailure);
					this->type = Type::errorType;
				} else {
					TaskSymbol *task = dynamic_cast<TaskSymbol*>(symbol);
					if (task == NULL) {
						ReportError::UndefinedTask(str->GetLocation(), str->getValue(), 
								ignoreFailure);
						this->type = Type::errorType;
					} else {
						TaskDef *taskDef = (TaskDef*) task->getNode();
						this->type = new NamedType(taskDef->getEnvTuple()->getId());
					}
				}
			} 
		}
	} else {
		// then we check if the object corresponds to an array or list type
		ArrayType *arrayType = dynamic_cast<ArrayType*>(objectType);
		ListType *listType = dynamic_cast<ListType*>(objectType);
		if (arrayType != NULL) {
			Type *terminalType = arrayType->getTerminalElementType();
			if (scope->lookup(terminalType->getName()) == NULL) {
				ReportError::InvalidObjectTypeInNew(terminalType, ignoreFailure);
			}
		} else if (listType != NULL) {
			Type *terminalType = arrayType->getTerminalElementType();
			if (scope->lookup(terminalType->getName()) == NULL) {
				ReportError::InvalidObjectTypeInNew(terminalType, ignoreFailure);
			}

		// now the object has to be an instance of a declared or built-in tuple type
		} else {
			Symbol *symbol = scope->lookup(typeName);
			if (symbol == NULL) {
				ReportError::UndefinedSymbol(objectType->GetLocation(), typeName, ignoreFailure);
				this->type = Type::errorType;
				return;
			}
			TupleSymbol *tuple = dynamic_cast<TupleSymbol*>(symbol);
			if (tuple == NULL) {
				ReportError::WrongSymbolType(objectType->GetLocation(), typeName, "Tuple", ignoreFailure);
				this->type = Type::errorType;
				return;
			} 
			List<Type*> *elementTypes = tuple->getElementTypes();
			if (initArgs->NumElements() > elementTypes->NumElements()) {
				ReportError::TooManyParametersInNew(GetLocation(), typeName, initArgs->NumElements(), 
						elementTypes->NumElements(), ignoreFailure);
			} else {
				for (int i = 0; i < initArgs->NumElements(); i++) {
					Expr *currentArg = initArgs->Nth(i);
					Type *currentType = elementTypes->Nth(i);	
					currentArg->resolveType(scope, ignoreFailure);
					if (currentArg->getType() == NULL) {
						currentArg->inferType(scope, currentType);
					} else if (!currentType->isAssignableFrom(currentArg->getType())) {
						ReportError::IncompatibleTypes(currentArg->GetLocation(), 
								currentArg->getType(), currentType, ignoreFailure);
					}
				}
			}
		}
		this->type = objectType;
	}
}
    	
