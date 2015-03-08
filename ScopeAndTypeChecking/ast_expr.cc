#include "ast.h"
#include "ast_stmt.h"
#include "ast_expr.h"
#include "ast_type.h"

#include "list.h"
#include "string.h"
#include "symbol.h"
#include "errors.h"
#include <iostream>

//----------------------------------------------------- Expression ----------------------------------------------------/

void Expr::performTypeInference(Scope *executionScope) {
	if (this->type == NULL) {
		resolveType(executionScope, true);
	}
	inferType(executionScope, this->type);	
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

//----------------------------------------------- Reduction Expression ------------------------------------------------/

ReductionExpr::ReductionExpr(Expr *l, ReductionOperator o, Expr *r, yyltype loc) : Expr(loc) {
	Assert(l != NULL && r != NULL);
	left = l;
	left->SetParent(this);
	op = o;
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
	left->Print(indentLevel + 1);
	right->Print(indentLevel + 1);
}

void ReductionExpr::resolveType(Scope *scope, bool ignoreFailure) {

	right->resolveType(scope, ignoreFailure);
	Type *rightType = right->getType();
	if (rightType == NULL) {
		ReportError::UnknownExpressionType(right, ignoreFailure);
	}
	left->resolveType(scope, ignoreFailure);
	Type *leftType = left->getType();
	if (leftType == NULL) {
		ReportError::UnknownExpressionType(left, ignoreFailure);
	}

	if (rightType != NULL && rightType != Type::intType 
			&& rightType != Type::floatType
			&& rightType != Type::doubleType
			&& rightType != Type::errorType) {
		ReportError::UnsupportedOperand(right, rightType, 
				"reduction expression", ignoreFailure);
		this->type = Type::errorType;
	}

	if ((op == MIN_ENTRY || op == MAX_ENTRY) 
			&& leftType != NULL) {
		if (leftType != Type::intType) {
			ReportError::IncompatibleTypes(left->GetLocation(), leftType, 
					Type::intType, ignoreFailure);
			this->type = Type::errorType;
		} else {
			this->type = Type::intType;
		}
	} else if (leftType != NULL && rightType != NULL 
			&& !leftType->isAssignableFrom(rightType)) {
		ReportError::TypeMixingError(this, leftType, rightType, 
				"reduction expression", ignoreFailure);
		this->type = Type::errorType;
	} else {
		this->type = leftType;
	}
}

void ReductionExpr::inferType(Scope *scope, Type *rootType) {

	if (op == MIN_ENTRY || op == MAX_ENTRY) {
		this->type = Type::intType;
		left->inferType(scope, Type::intType);
		right->inferType(scope);
	} else {	
		if (rootType == NULL) {
			left->inferType(scope);
			right->inferType(scope);
			if (left->getType() == NULL && right->getType() != NULL) {
				left->inferType(scope, right->getType());
			} else if (right->getType() == NULL && left->getType() != NULL) {
				right->inferType(scope, left->getType());
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
}

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

void FieldAccess::resolveType(Scope *scope, bool ignoreFailure) {

	// accessing a field from a tuple type (user-defined/built-in)
	if (base != NULL) {
		base->resolveType(scope, ignoreFailure);
		Type *baseType = base->getType();
		if (baseType != NULL) {
			ArrayType *arrayType = dynamic_cast<ArrayType*>(baseType);
			MapType *mapType = dynamic_cast<MapType*>(baseType);
			if (arrayType != NULL) {
				DimensionIdentifier *dimension = dynamic_cast<DimensionIdentifier*>(field);
				if (strcmp(field->getName(), Identifier::LocalId) == 0) {
					this->type = arrayType;
				} else if (dimension != NULL) {
					int dimensionality = arrayType->getDimensions();
					int fieldDimension = dimension->getDimensionNo();
					if (fieldDimension > dimensionality) {
						ReportError::NonExistingDimensionInArray(field, 
								dimensionality, fieldDimension, ignoreFailure);
					}
					this->type = Type::dimensionType;
				} else {
					ReportError::NoSuchFieldInBase(field, arrayType, ignoreFailure);
					this->type = Type::errorType;
				}
			} else if (mapType != NULL) {
				if (mapType->hasElement(field->getName())) {
					Type *elemType = mapType->getElementType(field->getName());
					if (elemType == NULL) {
						ReportError::UnknownExpressionType(this, ignoreFailure);
					} else {
						this->type = elemType;
					}
				} else {
					mapType->setElement(new VariableDef(field));	
				}
			} else {
				if (baseType == Type::errorType) {
					this->type = Type::errorType;
				} else {
					Symbol *symbol = scope->lookup(baseType->getName());
					if (symbol == NULL) {
						this->type = Type::errorType;
					} else {
						Scope *baseScope = symbol->getNestedScope();
						if (baseScope == NULL || baseScope->lookup(field->getName()) == NULL) {
							ReportError::NoSuchFieldInBase(field, baseType, ignoreFailure);
							this->type = Type::errorType;
						} else {
							VariableSymbol *fieldSymbol 
								= (VariableSymbol*) baseScope->lookup(field->getName());
							this->type = fieldSymbol->getType();
						}
					}
				}
			}
		}
	// accessing a variable directly (defined/undefined)
	} else {
		VariableSymbol *symbol = (VariableSymbol*) scope->lookup(field->getName());
		if (symbol != NULL) {
			this->type = symbol->getType();
			NamedType *tupleType = dynamic_cast<NamedType*>(this->type);
			if (tupleType != NULL) {
				Symbol *typeSymbol = scope->lookup(tupleType->getName());
				if (typeSymbol == NULL) {
					ReportError::UndeclaredTypeError(field, this->type, NULL, ignoreFailure);
					this->type = Type::errorType;
				}
			}
		} else if (!ignoreFailure) {
			ReportError::UndefinedSymbol(field, ignoreFailure);
			this->type = Type::errorType;
		}
	}
}

/* TODO For mapping between index and integer types, some additional logic need to be incorporated here. */
void FieldAccess::inferType(Scope *scope, Type *rootType) {
	
	if (rootType == NULL) return;

	if (this->type != NULL && !rootType->isAssignableFrom(this->type)) {
		ReportError::InferredAndActualTypeMismatch(this->GetLocation(), rootType, 
				this->type, false);

	} else if (base != NULL) {
		base->resolveType(scope, true);
		Type *baseType = base->getType();
		if (baseType == NULL) return;
		MapType *mapType = dynamic_cast<MapType*>(baseType);
		if (mapType == NULL) return;
		
		if (mapType->hasElement(field->getName())) {
			VariableDef *element = mapType->getElement(field->getName());
			if (element->getType() == NULL || rootType->isAssignableFrom(element->getType())) {
				this->type = rootType;
				element->setType(rootType);
			}	
		} else {
			this->type = rootType;
			mapType->setElement(new VariableDef(field, rootType));
		}		
		
	} else if (base == NULL && this->type == NULL) {
		this->type = rootType;
		VariableSymbol *symbol = (VariableSymbol*) scope->lookup(field->getName());
		if (symbol != NULL) {
			symbol->setType(this->type);
		} else {
			symbol = new VariableSymbol(field->getName(), this->type);
			bool success = scope->insert_inferred_symbol(symbol);
			if (!success) {
				ReportError::Formatted(GetLocation(), 
						"couldn't create symbol in the scope for %s", 
						field->getName());
			}			
		}
	}
}

//------------------------------------------------ Range Expressions --------------------------------------------------/

RangeExpr::RangeExpr(Identifier *i, Expr *r, Expr *s, yyltype loc) : Expr(loc) {
	Assert(i != NULL && r != NULL);
	index = i;
	index->SetParent(this);
	range = r;
	range->SetParent(this);
	step = s;
	if (step != NULL) {
		step->SetParent(this);
	}
}

void RangeExpr::PrintChildren(int indentLevel) {
	index->Print(indentLevel + 1, "(Index) ");
	range->Print(indentLevel + 1, "(Range) ");
	if (step != NULL) step->Print(indentLevel + 1, "(Step) ");
}

void RangeExpr::resolveType(Scope *scope, bool ignoreFailure) {
	
	VariableSymbol *symbol = (VariableSymbol*) scope->lookup(index->getName());
	Type *indexType = NULL;
	if (symbol != NULL) indexType = symbol->getType();
	if (indexType == NULL) {
		symbol = new VariableSymbol(index->getName(), Type::intType);
		index->setSymbol(symbol);	
		bool success = scope->insert_inferred_symbol(symbol);
		if (!success) {
			ReportError::UndefinedSymbol(index, ignoreFailure);
		}
	} else if (indexType != Type::intType && indexType != Type::errorType) {
		 ReportError::IncompatibleTypes(index->GetLocation(), indexType, 
				Type::intType, ignoreFailure);	
	}
	
	range->resolveType(scope, ignoreFailure);
	Type *rangeType = range->getType();
	if (rangeType == NULL && !ignoreFailure) {
		ReportError::UnknownExpressionType(range, ignoreFailure);
	} else if (rangeType != NULL && rangeType != Type::rangeType) {
		 ReportError::IncompatibleTypes(range->GetLocation(), rangeType, 
				Type::rangeType, ignoreFailure);	
	}

	if (step != NULL) {
		step->resolveType(scope, ignoreFailure);
		Type *stepType = step->getType();
		if (stepType == NULL) {
			step->inferType(scope, Type::intType);
		} else if (!Type::intType->isAssignableFrom(stepType)) {
		 	ReportError::IncompatibleTypes(step->GetLocation(), stepType, 
					Type::intType, ignoreFailure);	
		}
	}
	
	this->type = Type::boolType;
}

SubpartitionRangeExpr::SubpartitionRangeExpr(char s, yyltype loc) : Expr(loc) {
	spaceId = s;
}

void SubpartitionRangeExpr::PrintChildren(int indentLevel) {
	printf("Space %c", spaceId);
}

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

void ArrayAccess::resolveType(Scope *scope, bool ignoreFailure) {

	base->resolveType(scope, ignoreFailure);
	Type *baseType = base->getType();
	if (baseType == NULL) {
		if (!ignoreFailure) {
			ReportError::InvalidArrayAccess(GetLocation(), NULL, ignoreFailure);
			this->type = Type::errorType;
		}
	} else {
		ArrayType *arrayType = dynamic_cast<ArrayType*>(baseType);
		if (arrayType == NULL) {
			if (baseType != Type::errorType) {
				ReportError::InvalidArrayAccess(base->GetLocation(), baseType, ignoreFailure);
			}
			this->type = Type::errorType;
		} else {
			SubRangeExpr *subRange = dynamic_cast<SubRangeExpr*>(index);
			if (subRange != NULL) {
				this->type = arrayType;
				subRange->inferType(scope, NULL);
			} else {
				this->type = arrayType->reduceADimension();
				index->inferType(scope, Type::intType);
			}	
		}
	}	
}

SubRangeExpr::SubRangeExpr(Expr *b, Expr *e, yyltype loc) : Expr(loc) {
	begin = b;
	if (begin != NULL) {
		begin->SetParent(this);
	}
	end = e;
	if (end != NULL) {
		end->SetParent(this);
	}
	fullRange = (b == NULL && e == NULL);		
}

void SubRangeExpr::PrintChildren(int indentLevel) {
	if (begin != NULL) begin->Print(indentLevel + 1);
	if (end != NULL) end->Print(indentLevel + 1);
}

void SubRangeExpr::inferType(Scope *scope, Type *rootType) {
	if (begin != NULL) begin->inferType(scope, Type::intType);
	if (end != NULL) end->inferType(scope, Type::intType);	
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

	if (strcmp(section->getName(), InitializeSection) == 0) {
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
	} else if (strcmp(section->getName(), PartitionSection) == 0) {
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
    	
