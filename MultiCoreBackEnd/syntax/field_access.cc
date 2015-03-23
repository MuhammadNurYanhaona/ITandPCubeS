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

//-------------------------------------------------- Field Access -----------------------------------------------------/

FieldAccess::FieldAccess(Expr *b, Identifier *f, yyltype loc) : Expr(loc) {
	Assert(f != NULL);
	base = b;
	if (base != NULL) {
		base->SetParent(this);
	}
	field = f;
	field->SetParent(this);
	metadata = false;
	local = false;
	index = false;
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

const char *FieldAccess::getBaseVarName() {
	if (base == NULL) return field->getName();
	return base->getBaseVarName();
}

Hashtable<VariableAccess*> *FieldAccess::getAccessedGlobalVariables(TaskGlobalReferences *globalReferences) {
	if (base == NULL) {
		Hashtable<VariableAccess*> *table = new Hashtable<VariableAccess*>;
		const char *fieldName = field->getName();
		if (globalReferences->doesReferToGlobal(fieldName)) {
			VariableSymbol *global = globalReferences->getGlobalRoot(fieldName);
			Type *globalType = global->getType();
			const char *globalName = global->getName();
			VariableAccess *accessLog = new VariableAccess(globalName);
			if ((dynamic_cast<ArrayType*>(globalType)) == NULL) {
				accessLog->markContentAccess();
			}
			table->Enter(globalName, accessLog, true);
		}
		return table;
	} else {
		FieldAccess *baseField = dynamic_cast<FieldAccess*>(base);
		if (baseField != NULL) {
			ArrayType *array = dynamic_cast<ArrayType*>(baseField->getType());
			// set some flags in the base expression that will help us during code generation
			if (array != NULL) {
				baseField->setMetadata(true);
				if (this->isLocalTerminalField()) {
					baseField->markLocal();
				}
			}
		}
		Hashtable<VariableAccess*> *table = base->getAccessedGlobalVariables(globalReferences);
		if (baseField == NULL || !baseField->isTerminalField()) return table;
		const char *fieldName = baseField->field->getName();
		if (globalReferences->doesReferToGlobal(fieldName)) {
			VariableSymbol *global = globalReferences->getGlobalRoot(fieldName);
			Type *globalType = global->getType();
			const char *globalName = global->getName();
			if ((dynamic_cast<ArrayType*>(globalType)) != NULL) {
				table->Lookup(globalName)->markMetadataAccess(); 
			} else {
				table->Lookup(globalName)->markContentAccess();
			}	
		}
		return table;
	}
}

bool FieldAccess::isLocalTerminalField() {
	if (base == NULL) return false;
	FieldAccess *baseField = dynamic_cast<FieldAccess*>(base);
	if (baseField == NULL) return false;
	return (baseField->isTerminalField() && strcmp(field->getName(), Identifier::LocalId) == 0);
}

void FieldAccess::translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space) {

	if (base != NULL) {
		// call the translate function recursively on base if it is not null
		base->translate(stream, indentLevel, currentLineLength, space);
		
		// if it is an array dimension then access the appropriate index corresponding to that dimension
		ArrayType *arrayType = dynamic_cast<ArrayType*>(base->getType());
		NamedType *userDefinedType = dynamic_cast<NamedType*>(base->getType());
		if (arrayType != NULL) {
			// skip the field if it is a local flag for an array dimension
			if (!strcmp(field->getName(), Identifier::LocalId) == 0) {
				DimensionIdentifier *dimension = dynamic_cast<DimensionIdentifier*>(field);
				int fieldDimension = dimension->getDimensionNo();
				// If Dimension No is 0 then this is handled in assignment statement translation when 
				// there is a possibility of multiple dimensions been copied from an array to another
				if (fieldDimension > 0) {
					// One is subtracted as dimension no started from 1 in the source code
					stream << '[' << fieldDimension - 1 << ']';
				} else if (arrayType->getDimensions() == 1) {
					stream << "[0]";
				}
				//------------------------------------------------------------------- Patch Solution
				// This is a patch solution to cover quickly cover up the design difference between
				// environment object and other objects. TODO kindly do appropriate refactoring in the
				// future to do this in a better way.
				FieldAccess *baseField = dynamic_cast<FieldAccess*>(base);
				if (baseField != NULL && !baseField->isTerminalField()) {
					stream << ".partition";	
				} 	
				//----------------------------------------------------------------------------------
			}
		// if it is the length field of a dimension object then convert that into an equivalent function call
		} else if (base->getType() == Type::dimensionType && strcmp(field->getName(), "length") == 0) {
			stream << ".getLength()";
		// if the base is some type of environment then according the logic we use it is a pointer reference;
		// so this case need an especial treatment
		} else if (userDefinedType != NULL && userDefinedType->isEnvironmentType()) {
			ntransform::NameTransformer *transformer = ntransform::NameTransformer::transformer;
			stream << "->";
			// in addition, since an environment object may contain array references, we need to get trans-
			// formed names based on types of its properties. 
			const char *fieldName = field->getName();
			stream << transformer->getTransformedName(fieldName, metadata, local, type);
		// otherwise just write the field directly
		} else {
			stream << "." << field->getName();
		}

	// if this is a terminal field then there may be a need for name transformation; so we consult the transformer
	} else {
		ntransform::NameTransformer *transformer = ntransform::NameTransformer::transformer;
		const char *fieldName = field->getName();
		stream << transformer->getTransformedName(fieldName, metadata, local);
	}
}

bool FieldAccess::isEqual(FieldAccess *other) {
	if (strcmp(this->field->getName(), other->field->getName()) != 0) return false;
	if (this->base == NULL && other->base == NULL) return true;
	if (this->base == NULL && other->base != NULL) return false;
	if (this->base != NULL && other->base == NULL) return false;
	FieldAccess *baseField1 = dynamic_cast<FieldAccess*>(this->base);
	FieldAccess *baseField2 = dynamic_cast<FieldAccess*>(other->base);
	if (baseField1 != NULL && baseField2 != NULL) {
		return baseField1->isEqual(baseField2);
	}
	return false;
}

List<FieldAccess*> *FieldAccess::getTerminalFieldAccesses() { 
	if (base == NULL) {
		List<FieldAccess*> *list = new List<FieldAccess*>;
		list->Append(this);
		return list;
	}
	return base->getTerminalFieldAccesses();
}

void FieldAccess::translateIndex(std::ostringstream &stream, const char *array, int dimension) {
	stream << field->getName();
	stream << array;
	stream << dimension;	
}
