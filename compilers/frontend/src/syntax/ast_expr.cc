#include "ast.h"
#include "ast_stmt.h"
#include "ast_expr.h"
#include "ast_type.h"
#include "../common/errors.h"
#include "../common/constant.h"
#include "../semantics/scope.h"
#include "../semantics/symbol.h"
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

int Expr::resolveExprTypesAndScopes(Scope *executionScope, int iteration) {
	if (this->type == NULL) {
		return resolveExprTypes(executionScope);
	}
	return 0;
}

int Expr::performTypeInference(Scope *executionScope, Type *assumedType) {	
	
	if (assumedType == NULL || assumedType == Type::errorType) return 0;

	if (this->type == NULL || this->type == Type::errorType) {
		return inferExprTypes(executionScope, assumedType);
	}
	return 0;
}

int Expr::emitScopeAndTypeErrors(Scope *scope) {
	int errors = 0;
	if (type == NULL || type == Type::errorType) {
		ReportError::UnknownExpressionType(this, false);
		errors++;
	}
	return errors + emitSemanticErrors(scope);	
}

//----------------------------------------------- Constant Expression -------------------------------------------------/

IntConstant::IntConstant(yyltype loc, int val) : Expr(loc) {
        value = val;
	size = TWO_BYTES;
	type = Type::intType;
}

IntConstant::IntConstant(yyltype loc, int value, IntSize size) : Expr(loc) {
        this->value = value;
        this->size = size;
	type = Type::intType;
}

void IntConstant::PrintChildren(int indentLevel) {
        printf("%d", value);
}

FloatConstant::FloatConstant(yyltype loc, float val) : Expr(loc) {
        value = val;
	type = Type::floatType;
}

void FloatConstant::PrintChildren(int indentLevel) {
        printf("%f", value);
}

DoubleConstant::DoubleConstant(yyltype loc, double val) : Expr(loc) {
        value = val;
	type = Type::doubleType;
}

void DoubleConstant::PrintChildren(int indentLevel) {
        printf("%g", value);
}

BoolConstant::BoolConstant(yyltype loc, bool val) : Expr(loc) {
        value = val;
	type = Type::boolType;
}

void BoolConstant::PrintChildren(int indentLevel) {
        printf("%s", value ? "true" : "false");
}

StringConstant::StringConstant(yyltype loc, const char *val) : Expr(loc) {
        Assert(val != NULL);
        value = strdup(val);
	type = Type::stringType;
}

void StringConstant::PrintChildren(int indentLevel) {
        printf("%s",value);
}

CharConstant::CharConstant(yyltype loc, char val) : Expr(loc) {
        value = val;
	type = Type::charType;
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

int ReductionVar::resolveExprTypes(Scope *scope) {
	VariableSymbol *symbol = (VariableSymbol*) scope->lookup(name);
	if (symbol != NULL) {
		this->type = symbol->getType();
		return 1;
	}
	return 0;
}

int ReductionVar::emitSemanticErrors(Scope *scope) {

	Symbol *symbol = scope->lookup(name);
        if (symbol == NULL) {
		ReportError::UndefinedSymbol(GetLocation(), name, false);
		return 1;
	}
	VariableSymbol *varSymbol = dynamic_cast<VariableSymbol*>(symbol);
	if (varSymbol == NULL) {
		ReportError::WrongSymbolType(GetLocation(), name, "Reduction Variable", false);
		return 1;
	} else if (!varSymbol->isReduction()) {
		Identifier *varId = new Identifier(*GetLocation(), name);
		ReportError::NotReductionType(varId, false);
		return 1;
	}
	return 0;
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

int ArithmaticExpr::resolveExprTypes(Scope *scope) {
	
	int resolvedExprs = 0;
	resolvedExprs += left->resolveExprTypes(scope);
        Type *leftType = left->getType();
        resolvedExprs += right->resolveExprTypes(scope);
        Type *rightType = right->getType();

	if (leftType != NULL && rightType != NULL) {
                if (leftType->isAssignableFrom(rightType)) {
                        this->type = leftType;
			resolvedExprs++;
                } else if (rightType->isAssignableFrom(leftType)) {
                        this->type = rightType;
			resolvedExprs++;
		}
	}
	
	resolvedExprs += left->performTypeInference(scope, rightType);
	resolvedExprs += right->performTypeInference(scope, leftType);

	return resolvedExprs;
}

int ArithmaticExpr::emitSemanticErrors(Scope *scope) {
	
	int errors = 0;

	// check the validity of the left-hand-side expression and its use in the arithmatic expression
	errors += left->emitScopeAndTypeErrors(scope);
	Type *leftType = left->getType();
	if (leftType != NULL && !(leftType == Type::intType
                        || leftType == Type::floatType
                        || leftType == Type::doubleType
                        || leftType == Type::charType
                        || leftType == Type::errorType)) {
                ReportError::UnsupportedOperand(left, leftType, "arithmatic expression", false);
		errors++;
        }
	
	// check the validity of the right-hand-side expression and its use in the arithmatic expression
	errors += right->emitScopeAndTypeErrors(scope);
	Type *rightType = right->getType();
        if (rightType != NULL && !(rightType == Type::intType
                        || rightType == Type::floatType
                        || rightType == Type::doubleType
                        || rightType == Type::charType
                        || rightType == Type::errorType)) {
                ReportError::UnsupportedOperand(right, rightType, "arithmatic expression", false);
		errors++;
        }

	// check the validity of combining the left and right hand-side expressions in arithmatic
        if (op == BITWISE_AND || op == BITWISE_OR || op == BITWISE_XOR) {
                if ((leftType != NULL 
				&& !(leftType == Type::intType 
					|| leftType == Type::errorType))
                       		|| (rightType != NULL && !(rightType == Type::intType
                                        || rightType == Type::errorType))) {
                        ReportError::UnsupportedOperand(right,
                                        rightType, "arithmatic expression", false);
			errors++;
                }
        }
        if (leftType != NULL && rightType != NULL) {
                if (!leftType->isAssignableFrom(rightType) 
				&& !rightType->isAssignableFrom(leftType)) {
                        ReportError::TypeMixingError(this,
                                        leftType, rightType, "arithmatic expression", false);
			errors++;
                }
        }
	return errors;
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

int LogicalExpr::resolveExprTypes(Scope *scope) {

	int resolvedExprs = 0;
	if (left != NULL) {
		resolvedExprs += left->resolveExprTypes(scope);
	}
	resolvedExprs += right->resolveExprTypes(scope);

	bool arithmaticOp = (op == EQ || op == NE || op == GT || op == LT || op == GTE || op == LTE);
	if (arithmaticOp) {
		resolvedExprs += right->performTypeInference(scope, left->getType());
		resolvedExprs += left->performTypeInference(scope, right->getType());
	} else {
		resolvedExprs += right->performTypeInference(scope, Type::boolType);
		resolvedExprs += left->performTypeInference(scope, Type::boolType);
	}
	
	this->type = Type::boolType;
	resolvedExprs++;

	return resolvedExprs;
}

int LogicalExpr::emitSemanticErrors(Scope *scope) {

	int errors = 0;
	
	// check the validity of the component expressions
	errors += right->emitScopeAndTypeErrors(scope);
	Type *rightType = right->getType();
	Type *leftType = NULL;
	if (left != NULL) {
		errors += left->emitScopeAndTypeErrors(scope);
        	leftType = left->getType();
	}

	// check the validity of combining/using the component expression(s) in the spcecific way as done by
	// the current expression
	bool arithmaticOperator = (op == EQ || op == NE || op == GT || op == LT || op == GTE || op == LTE);
        if (arithmaticOperator) {
                if (leftType != NULL && !(leftType == Type::intType
                                || leftType == Type::floatType
                                || leftType == Type::doubleType
                                || leftType == Type::charType
                                || leftType == Type::errorType)) {
                        ReportError::UnsupportedOperand(left, leftType, "logical expression", false);
			errors++;
                }
                if (rightType != NULL && !(rightType == Type::intType
                                || rightType == Type::floatType
                                || rightType == Type::doubleType
                                || rightType == Type::charType
                                || rightType == Type::errorType)) {
                        ReportError::UnsupportedOperand(right, rightType, "logical expression", false);
			errors++;
                }
                if (leftType != NULL && rightType != NULL) {
                        if (!leftType->isAssignableFrom(rightType)
                                        && !rightType->isAssignableFrom(leftType)) {
                                ReportError::TypeMixingError(this, leftType, rightType,
                                                "logical expression", false);
				errors++;
                        }
                }
        } else {
                if (rightType != NULL && !rightType->isAssignableFrom(Type::boolType)) {
                        ReportError::IncompatibleTypes(right->GetLocation(), rightType, Type::boolType, false);
			errors++;
                }
                if (leftType != NULL && !leftType->isAssignableFrom(Type::boolType)) {
                        ReportError::IncompatibleTypes(left->GetLocation(), leftType, Type::boolType, false);
			errors++;
                }
        }
	return errors;
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

int EpochExpr::resolveExprTypes(Scope *scope) {

	int resolvedExprs = root->resolveExprTypes(scope);
	Type *rootType = root->getType();

	if (rootType != NULL && rootType != Type::errorType) {
		this->type = rootType;
		resolvedExprs++;
	}
	return resolvedExprs;
}

int EpochExpr::emitSemanticErrors(Scope *scope) {
	return root->emitScopeAndTypeErrors(scope);
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
	arrayField = false;
	arrayDimensions = -1;
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
	if (arrayField) newFieldAcc->flagAsArrayField(arrayDimensions);
	return newFieldAcc;
}

void FieldAccess::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	Expr::retrieveExprByType(exprList, typeId);
	if (base != NULL) base->retrieveExprByType(exprList, typeId);
}

void FieldAccess::flagAsArrayField(int arrayDimensions) {
	arrayField = true;
	this->arrayDimensions = arrayDimensions;
}

FieldAccess *FieldAccess::getTerminalField() {
	if (base == NULL) return this;
	FieldAccess *baseField = dynamic_cast<FieldAccess*>(base);
	if (baseField == NULL) return NULL;
	return baseField->getTerminalField();
}

int FieldAccess::resolveExprTypes(Scope *scope) {
	
	int resolvedExprs = 0;
	if (base == NULL) {			// consider the terminal case of accessing a variable first
		VariableSymbol *symbol = (VariableSymbol*) scope->lookup(field->getName());
                if (symbol != NULL) {
                        this->type = symbol->getType();
			resolvedExprs++;
		}
		return resolvedExprs;
	}

	resolvedExprs += base->resolveExprTypes(scope);
	Type *baseType = base->getType();
	if (baseType == NULL || baseType == Type::errorType) return resolvedExprs;

	ArrayType *arrayType = dynamic_cast<ArrayType*>(baseType);
	MapType *mapType = dynamic_cast<MapType*>(baseType);
	if (arrayType != NULL) {		// check for the field access to be a part of an array		
		if (strcmp(field->getName(), Identifier::LocalId) == 0) {
			this->type = arrayType;
			resolvedExprs++;
		} else if (dynamic_cast<DimensionIdentifier*>(field) != NULL) {
			this->type = Type::dimensionType;
			resolvedExprs++;
		}
	} else if (mapType != NULL) {		// check for the field access to be an item in a map
		if (mapType->hasElement(field->getName())) {
			Type *elemType = mapType->getElementType(field->getName());
			if (elemType != NULL) {
				this->type = elemType;
				resolvedExprs++;
			}
		}
	} else {				// check if the field access to a property of a custom type
		Symbol *symbol = scope->lookup(baseType->getName());
		if (symbol != NULL) {
			Scope *baseScope = symbol->getNestedScope();
			if (baseScope != NULL && baseScope->lookup(field->getName()) != NULL) {
				VariableSymbol *fieldSymbol
					= (VariableSymbol*) baseScope->lookup(field->getName());
				this->type = fieldSymbol->getType();
				resolvedExprs++;
			}
		}
	}

	return resolvedExprs;
}

int FieldAccess::emitSemanticErrors(Scope *scope) {
	
	// check for the case when the current field access is not corresponding to accessing a property of
	// a larger object
	if (base == NULL) {
		Symbol *symbol = scope->lookup(field->getName());
                if (symbol != NULL && dynamic_cast<VariableSymbol*>(symbol) != NULL) {
			
			VariableSymbol *varSym = dynamic_cast<VariableSymbol*>(symbol);
                        this->type = varSym->getType();

			// if the field is of some custom-type then that type must be defined
                        NamedType *tupleType = dynamic_cast<NamedType*>(this->type);
                        if (tupleType != NULL) {
                                Symbol *typeSymbol = scope->lookup(tupleType->getName());
                                if (typeSymbol == NULL) {
                                        ReportError::UndeclaredTypeError(field, this->type, NULL, false);
                                        return 1;
                                }
                        }
                } else {
                        ReportError::UndefinedSymbol(field, false);
                        return 1;
                }
		return 0;
	}

	// check for the alternative case where the field access is accessing a property of a larger object
	int errors = 0;
	errors += base->emitScopeAndTypeErrors(scope);
	Type *baseType = base->getType();
	if (baseType != NULL) {
		ArrayType *arrayType = dynamic_cast<ArrayType*>(baseType);
		MapType *mapType = dynamic_cast<MapType*>(baseType);
		ListType *listType = dynamic_cast<ListType*>(baseType);
		if (arrayType != NULL) {
			DimensionIdentifier *dimension = dynamic_cast<DimensionIdentifier*>(field);
			if (dimension != NULL) {
				int dimensionality = arrayType->getDimensions();
				int fieldDimension = dimension->getDimensionNo();
				if (fieldDimension > dimensionality) {
					ReportError::NonExistingDimensionInArray(field, 
							dimensionality, fieldDimension, false);
					errors++;
				}
			} else {
				ReportError::NoSuchFieldInBase(field, arrayType, false);
				errors++;
			} 
		} else if (mapType == NULL && listType == NULL) {
			Symbol *symbol = scope->lookup(baseType->getName());
			if (symbol != NULL) {
				Scope *baseScope = symbol->getNestedScope();
				if (baseScope == NULL || baseScope->lookup(field->getName()) == NULL) {
					ReportError::NoSuchFieldInBase(field, baseType, false);
					errors++;
				}
			}
		}
	}
	return errors;
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

int RangeExpr::resolveExprTypes(Scope *scope) {
	
	int resolvedExprs = 0;	

	Identifier *fieldId = index->getField();
	VariableSymbol *symbol = (VariableSymbol*) scope->lookup(fieldId->getName());
	if (symbol == NULL && loopingRange) {

		// The current version of the compiler resolves indexes as integer types as  opposed to 
		// IndexType that support non-unit stepping and wrapped around index range traversal. 
		// This is so since we have not enabled those features in the language yet.
		VariableDef *variable = new VariableDef(fieldId, Type::intType);
		scope->insert_inferred_symbol(new VariableSymbol(variable));
	}

	resolvedExprs += range->resolveExprTypes(scope);
	resolvedExprs += range->performTypeInference(scope, Type::rangeType);

	if (step != NULL) {
		resolvedExprs += step->resolveExprTypes(scope);
		resolvedExprs += step->performTypeInference(scope, Type::intType);
	}

	this->type = Type::boolType;
	resolvedExprs++;

	return resolvedExprs; 	
}

int RangeExpr::emitSemanticErrors(Scope *scope) {

	int errors = 0;
	errors += index->emitScopeAndTypeErrors(scope);
        Type *indexType = index->getType();
        if (indexType != NULL && indexType != Type::intType && indexType != Type::errorType) {
        	ReportError::IncompatibleTypes(index->GetLocation(), 
				indexType, Type::intType, false);
		errors++;
        }

	errors += emitScopeAndTypeErrors(scope);
	Type *rangeType = range->getType();
        if (rangeType != NULL && rangeType != Type::rangeType) {
                 ReportError::IncompatibleTypes(range->GetLocation(), 
				rangeType, Type::rangeType, false);
        }

	if (step != NULL) {
                errors += step->emitScopeAndTypeErrors(scope);
                Type *stepType = step->getType();
                if (stepType != NULL 
				&& !Type::intType->isAssignableFrom(stepType)) {
                        ReportError::IncompatibleTypes(step->GetLocation(), 
					stepType, Type::intType, false);
                }
        }

	return errors;
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

int AssignmentExpr::resolveExprTypes(Scope *scope) {
	int resolvedExprs = 0;
	resolvedExprs += right->resolveExprTypes(scope);
	Type *rightType = right->getType();
	resolvedExprs += left->resolveExprTypes(scope);
	Type *leftType = left->getType();
	resolvedExprs += left->performTypeInference(scope, rightType);
	resolvedExprs += right->performTypeInference(scope, leftType);
	if (leftType != NULL && rightType != NULL && leftType->isAssignableFrom(rightType)) {
		this->type = leftType;
		resolvedExprs++;
	}
	return resolvedExprs;
}

int AssignmentExpr::emitSemanticErrors(Scope *scope) {

	int errors = 0;
	errors += left->emitScopeAndTypeErrors(scope);
	errors += right->emitScopeAndTypeErrors(scope);

	// check if the two sides of the assignment are compatible with each other
	Type *leftType = left->getType();
	Type *rightType = right->getType();
	if (leftType != NULL && rightType != NULL && !leftType->isAssignableFrom(rightType)) {
                ReportError::TypeMixingError(this, leftType, rightType, "assignment", false);
		errors++;
        }

	// check if the left-hand side of the assignment is a valid receiver expression
	FieldAccess *fieldAccess = dynamic_cast<FieldAccess*>(left);
        ArrayAccess *arrayAccess = dynamic_cast<ArrayAccess*>(left);
        if (fieldAccess == NULL && arrayAccess == NULL) {
                EpochExpr *epochExpr = dynamic_cast<EpochExpr*>(left);
                if (epochExpr == NULL) {
                        ReportError::NonLValueInAssignment(left, false);
			errors++;
                } else {
                        Expr *epochRoot = epochExpr->getRootExpr();
                        fieldAccess = dynamic_cast<FieldAccess*>(epochRoot);
                        arrayAccess = dynamic_cast<ArrayAccess*>(epochRoot);
                        if (fieldAccess == NULL && arrayAccess == NULL) {
                                ReportError::NonLValueInAssignment(left, false);
				errors++;
                        }
                }
        }
	return errors;
}

//------------------------------------------------- Index Range -------------------------------------------------------/

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

int IndexRange::resolveExprTypes(Scope *scope) {
	int resolvedExprs = 0;
	if (begin != NULL) {
		resolvedExprs += begin->resolveExprTypes(scope);
		resolvedExprs += begin->performTypeInference(scope, Type::intType);
	}
	if (end != NULL) {
		resolvedExprs += end->resolveExprTypes(scope);
		resolvedExprs += end->performTypeInference(scope, Type::intType);
	}
	this->type = (partOfArray) ? Type::voidType : Type::rangeType;
	resolvedExprs++;
	return resolvedExprs;
}

int IndexRange::emitSemanticErrors(Scope *scope) {
	int errors = 0;
	if (begin != NULL) {
		errors += begin->emitScopeAndTypeErrors(scope);
		Type *beginType = begin->getType();
                if (beginType != NULL && !Type::intType->isAssignableFrom(beginType)) {
                        ReportError::IncompatibleTypes(begin->GetLocation(), 
					beginType, Type::intType, false);
		}
	}
	if (end != NULL) {
		errors += end->emitScopeAndTypeErrors(scope);
		Type *endType = end->getType();
                if (endType != NULL && !Type::intType->isAssignableFrom(endType)) {
                        ReportError::IncompatibleTypes(end->GetLocation(), 
					endType, Type::intType, false);
		}
	}
	return errors;
}

//--------------------------------------------------- Array Access ----------------------------------------------------/

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

int ArrayAccess::getIndexPosition() {
        ArrayAccess *precedingAccess = dynamic_cast<ArrayAccess*>(base);
        if (precedingAccess != NULL) return precedingAccess->getIndexPosition() + 1;
        return 0;
}

Expr *ArrayAccess::getEndpointOfArrayAccess() {
        ArrayAccess *precedingAccess = dynamic_cast<ArrayAccess*>(base);
        if (precedingAccess != NULL) {
                return precedingAccess->getEndpointOfArrayAccess();
        } else return base;
}

int ArrayAccess::resolveExprTypes(Scope *scope) {

	int resolvedExprs = 0;
	resolvedExprs += base->resolveExprTypes(scope);
        Type *baseType = base->getType();
	if (baseType == NULL) return resolvedExprs;

	ArrayType *arrayType = dynamic_cast<ArrayType*>(baseType);
	if (arrayType == NULL) {
		this->type = Type::errorType;
		return resolvedExprs;
	}

	IndexRange *indexRange = dynamic_cast<IndexRange*>(index);
        if (indexRange != NULL) {
		this->type = arrayType;
		resolvedExprs += indexRange->resolveExprTypes(scope);
	} else {
		this->type = arrayType->reduceADimension();
		resolvedExprs += index->resolveExprTypes(scope);
		resolvedExprs += index->performTypeInference(scope, Type::intType);	
	}
	resolvedExprs++;
	return resolvedExprs;
}

int ArrayAccess::emitSemanticErrors(Scope *scope) {
	
	int errors = 0;
	errors += base->emitScopeAndTypeErrors(scope);
        Type *baseType = base->getType();
        if (baseType == NULL) {
		ReportError::InvalidArrayAccess(GetLocation(), NULL, false);
        } else {
                ArrayType *arrayType = dynamic_cast<ArrayType*>(baseType);
                if (arrayType == NULL) {
			ReportError::InvalidArrayAccess(base->GetLocation(), baseType, false);
		}
	}
	errors += index->emitScopeAndTypeErrors(scope);
	return errors;
}

//------------------------------------------------- Function Call -----------------------------------------------------/

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

int FunctionCall::resolveExprTypes(Scope *scope) {
	
	int resolvedExprs = 0;
	bool allArgsResolved = true;
	List<Type*> *argTypeList = new List<Type*>;
	for (int i = 0; i < arguments->NumElements(); i++) {
                Expr *arg = arguments->Nth(i);
		resolvedExprs += arg->resolveExprTypes(scope);
		Type *argType = arg->getType();
		if (argType == NULL || argType == Type::errorType) {
			allArgsResolved = false;
		}
	}

	if (allArgsResolved) {
		const char *functionName = base->getName();
		FunctionDef *fnDef = FunctionDef::fnDefMap->Lookup(functionName);
		if (fnDef == NULL) {
			ReportError::UndefinedSymbol(base, false);
			this->type = Type::errorType;
		} else {
			// determine the specific function instance for the type polymorphic function for
			// the current parameter types
			Scope *programScope = scope->get_nearest_scope(ProgramScope);
			Type *returnType = fnDef->resolveFnInstanceForParameterTypes(
					programScope, argTypeList, base);
			this->type = returnType;
			resolvedExprs++;
		}
	}

	delete argTypeList;
	return resolvedExprs;
}

int FunctionCall::emitSemanticErrors(Scope *scope) {
	int errors = 0;
	for (int i = 0; i < arguments->NumElements(); i++) {
                Expr *arg = arguments->Nth(i);
		errors += arg->emitScopeAndTypeErrors(scope);
	}
	return errors;
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

int NamedMultiArgument::resolveExprTypes(Scope *scope) {
	int resolvedExprs = 0;
	for (int i = 0; i < argList->NumElements(); i++) {
                Expr *arg = argList->Nth(i);
		resolvedExprs += arg->resolveExprTypes(scope);
	}
	return resolvedExprs;
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

int TaskInvocation::resolveExprTypes(Scope *scope) {

	int resolvedExprs = 0;

	// check for the existance of a task and an environment argument in execute command
	const char *taskName = getTaskName();
	FieldAccess *environment = getEnvArgument();
	if (taskName == NULL || environment == NULL) {
		ReportError::UnspecifiedTaskToExecute(GetLocation(), false);
		this->type = Type::errorType;
		return resolvedExprs;
	}

	// check for a valid matching task definition to execute
	Symbol *symbol = scope->lookup(taskName);
        TaskSymbol *task = NULL;
        if (symbol == NULL) {
                ReportError::UndefinedSymbol(GetLocation(), taskName, false);
		this->type = Type::errorType;
		return resolvedExprs;
        } else {
                task = dynamic_cast<TaskSymbol*>(symbol);
                if (task == NULL) {
                        ReportError::WrongSymbolType(GetLocation(), taskName, "Task", false);
			this->type = Type::errorType;
			return resolvedExprs;
                }
	}

	// set up the type of the environment argument if it is not known already
	TaskDef *taskDef = (TaskDef*) task->getNode();
	TupleDef *envTuple = taskDef->getEnvTuple();
	resolvedExprs += environment->resolveExprTypes(scope);
	resolvedExprs += environment->performTypeInference(scope, new NamedType(envTuple->getId()));

	// if there are any initialization arguments then resolve them
	List<Type*> *initArgTypes = taskDef->getInitArgTypes();
	List<Expr*> *initArgs = getInitArguments();
	bool fullyResolved = true;
	if (initArgTypes->NumElements() == initArgs->NumElements()) {
		for (int i = 0; i < initArgs->NumElements(); i++) {
			Expr *expr = initArgs->Nth(i);
			Type *type = initArgTypes->Nth(i);
			resolvedExprs += expr->resolveExprTypes(scope);
			resolvedExprs += expr->performTypeInference(scope, type);
			if (expr->getType() == NULL) {
				fullyResolved = false;
			}
		}
	}

	// resolve the partition arguments as integers if exists
	List<Expr*> *partitionArgs = getPartitionArguments();
	for (int i =0; i < partitionArgs->NumElements(); i++) {
		Expr *arg = partitionArgs->Nth(i);
		resolvedExprs += arg->resolveExprTypes(scope);
		resolvedExprs += arg->performTypeInference(scope, Type::intType);
	}

	if (fullyResolved) {
		this->type = Type::voidType;
		resolvedExprs++;
	}
	return resolvedExprs;	
}

int TaskInvocation::emitSemanticErrors(Scope *scope) {

	if (type == NULL || type == Type::errorType) return 0;

	int errors = 0;
	const char *taskName = getTaskName();
	Symbol *task = (TaskSymbol*) scope->lookup(taskName);
	TaskDef *taskDef = (TaskDef*) task->getNode();
	
	// the environment argument should be of proper type specific to the task being invoked
	FieldAccess *environment = getEnvArgument();
	TupleDef *envTuple = taskDef->getEnvTuple();
	Type *expectedEnvType = new NamedType(envTuple->getId());
	Type *envType = environment->getType();
	if (!envType->isEqual(expectedEnvType)) {
		ReportError::IncompatibleTypes(environment->GetLocation(), envType,
					expectedEnvType, false);
		errors++;
	}

	// if there are any initialization arguments then the number should match the expected
	// arguments and the types should be appropriate
	List<Type*> *initArgTypes = taskDef->getInitArgTypes();
	List<Expr*> *initArgs = getInitArguments();
	if (initArgTypes->NumElements() != initArgs->NumElements()) {
		NamedMultiArgument *init = retrieveArgByName("initialize");
		Identifier *sectionId = new Identifier(*init->GetLocation(), "Initialize");
		ReportError::TooFewOrTooManyParameters(sectionId, initArgs->NumElements(),
				initArgTypes->NumElements(), false);
	} else {
		for (int i = 0; i < initArgTypes->NumElements(); i++) {
			Type *expected = initArgTypes->Nth(i);
			Expr *arg = initArgs->Nth(i);
			errors += arg->emitScopeAndTypeErrors(scope);
			Type *found = arg->getType();
			if (found != NULL && found != Type::errorType 
					&& !expected->isAssignableFrom(found)) {
				ReportError::IncompatibleTypes(arg->GetLocation(),
						found, expected, false);
				errors++;
			}
		}
	}
	
	// all partition arguments should be integers and their count should match expectation
	List<Expr*> *partitionArgs = getPartitionArguments();
	int argsCount = taskDef->getPartitionArgsCount();
	if (argsCount != partitionArgs->NumElements()) {
		NamedMultiArgument *partition = retrieveArgByName("partition");
		Identifier *sectionId = new Identifier(*partition->GetLocation(), "Partition");
		ReportError::TooFewOrTooManyParameters(sectionId, partitionArgs->NumElements(),
				argsCount, false);
	} else {
		for (int i = 0; i < partitionArgs->NumElements(); i++) {
			Expr *arg = partitionArgs->Nth(i);
			errors += arg->emitScopeAndTypeErrors(scope);
			Type *argType = arg->getType();
			if (argType != NULL && !Type::intType->isAssignableFrom(argType)) {
                                ReportError::IncompatibleTypes(arg->GetLocation(),
					argType, Type::intType, false);
                        }
		}
	}

	return errors;
}

const char *TaskInvocation::getTaskName() {
	NamedMultiArgument *nameArg = retrieveArgByName("task");
	if (nameArg == NULL) return NULL;
	List<Expr*> *argList = nameArg->getArgList();
	StringConstant *taskName = dynamic_cast<StringConstant*>(argList->Nth(0));
	if (taskName == NULL) return NULL;
	return taskName->getValue();
}

FieldAccess *TaskInvocation::getEnvArgument() {
	NamedMultiArgument *envArg = retrieveArgByName("environment");
	if (envArg == NULL) return NULL;
        List<Expr*> *argList = envArg->getArgList();
	Expr *firstArg = argList->Nth(0);
	return dynamic_cast<FieldAccess*>(firstArg);
}

List<Expr*> *TaskInvocation::getInitArguments() {
	NamedMultiArgument *initArg = retrieveArgByName("initialize");
	if (initArg == NULL) return new List<Expr*>;
        return initArg->getArgList();
}

List<Expr*> *TaskInvocation::getPartitionArguments() {
	NamedMultiArgument *partitionArg = retrieveArgByName("partition");
        if (partitionArg == NULL) return new List<Expr*>;
        return partitionArg->getArgList();
}

NamedMultiArgument *TaskInvocation::retrieveArgByName(const char *argName) {
	for (int i = 0; i < invocationArgs->NumElements(); i++) {
        	NamedMultiArgument *arg = invocationArgs->Nth(i);
		const char *currentArgName = arg->getName();
		if (strcmp(argName, currentArgName) == 0) return arg;
	}
	return NULL;
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

int ObjectCreate::resolveExprTypes(Scope *scope) {

	// check if the object corresponds to an array or a list type; then component elements must be of a known type
	ArrayType *arrayType = dynamic_cast<ArrayType*>(objectType);
	ListType *listType = dynamic_cast<ListType*>(objectType);
	Type *terminalType = NULL;
	if (arrayType != NULL) terminalType = arrayType->getTerminalElementType();
	else if (listType != NULL) terminalType = listType->getTerminalElementType();
	if (terminalType != NULL) {
		if (scope->lookup(terminalType->getName()) == NULL) {
			this->type = Type::errorType;
			return 0;
		} else {
			this->type = objectType;
			return 1;
		}
	}
	
	// check if the object correspond to some built-in or user defined type; then the arguments will be properties
	const char* typeName = objectType->getName();
	Symbol *symbol = scope->lookup(typeName);
	if (symbol != NULL) {
		int resolvedExprs = 0;
        	TupleSymbol *tuple = dynamic_cast<TupleSymbol*>(symbol);
		if (tuple == NULL) {
			this->type = Type::errorType;
                        return 0;
		}
		List<Type*> *elementTypes = tuple->getElementTypes();
		if (initArgs->NumElements() > elementTypes->NumElements()) {
			ReportError::TooManyParametersInNew(GetLocation(), typeName, 
					initArgs->NumElements(),
					elementTypes->NumElements(), false);
		} 
		TupleDef *tupleDef = tuple->getTupleDef();
		const char *typeName = tupleDef->getId()->getName();
		bool fullyResolved = true;
		for (int i = 0; i < initArgs->NumElements(); i++) {
			NamedArgument *currentArg = initArgs->Nth(i);
			const char *argName = currentArg->getName();
			Expr *argValue = currentArg->getValue();
			resolvedExprs += argValue->resolveExprTypes(scope);
			VariableDef *propertyDef = tupleDef->getComponent(argName);
        		if (propertyDef == NULL) {
				ReportError::InvalidInitArg(GetLocation(), typeName, argName, false);	
			} else {
				resolvedExprs += argValue->performTypeInference(scope, propertyDef->getType());
			}
			if (argValue->getType() == NULL) {
				fullyResolved = false;
			}
		}
		// We flag the current object creation expression to have a type only if all arguments are resolved
		// to allow type resolution process to make progress and the arguments to be resolved in some later
		// iteration of the scope-and-type checking.
		if (fullyResolved) {
			this->type = objectType;
			resolvedExprs++;
		}
		return resolvedExprs;
	}

	// the final option is that the object creation is correspond to the creation of an environment object for a
	// task
	if (strcmp(typeName, "TaskEnvironment") == 0) {
		if (initArgs->NumElements() != 1) {
                        ReportError::TaskNameRequiredInEnvironmentCreate(objectType->GetLocation(), false);
                        this->type = Type::errorType;
			return 0;
                }
		NamedArgument *arg = initArgs->Nth(0);
		StringConstant *str = dynamic_cast<StringConstant*>(arg->getValue());
		const char *argName = arg->getName();
		if (str == NULL || strcmp(argName, "name") != 0) {
                        ReportError::TaskNameRequiredInEnvironmentCreate(objectType->GetLocation(), false);
			this->type = Type::errorType;
			return 0;
		}
		Symbol *symbol = scope->lookup(str->getValue());
		if (symbol == NULL) {
			ReportError::UndefinedTask(str->GetLocation(), str->getValue(), false);
			this->type = Type::errorType;
			return 0;
		} else {
			TaskSymbol *task = dynamic_cast<TaskSymbol*>(symbol);
			if (task == NULL) {
				ReportError::UndefinedTask(str->GetLocation(), str->getValue(), false);
				this->type = Type::errorType;
				return 0;
			} else {
				TaskDef *taskDef = (TaskDef*) task->getNode();
				NamedType *envType = new NamedType(taskDef->getEnvTuple()->getId());
				envType->flagAsEnvironmentType();
				envType->setTaskName(str->getValue());
				this->type = envType;
				return 1;
			}
		}
	}
	
	// if none of the previous conditions matches then this object creation is in error and flaged as such
	this->type = Type::errorType;
	return 0;
}


