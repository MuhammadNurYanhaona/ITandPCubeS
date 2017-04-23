#include "../../../utils/name_transformer.h"
#include "../../../../../../frontend/src/syntax/ast.h"
#include "../../../../../../frontend/src/syntax/ast_type.h"
#include "../../../../../../frontend/src/syntax/ast_expr.h"
#include "../../../../../../frontend/src/semantics/task_space.h"

#include <sstream>
#include <string>
#include <cstdlib>


void FieldAccess::translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space) {

	if (base != NULL) {
		
		// if the current field access is accessing the dimension parameter of some environmental array then
		// we invoke a different translation process
		if (isEnvArrayDim()) {
			translateEnvArrayDim(stream, indentLevel, currentLineLength, space);
			return;
		}

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
				if (baseField != NULL 
						&& !baseField->isTerminalField() 
						&& strcmp(baseField->field->getName(), Identifier::LocalId) != 0) {
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
		const char *transformedName = transformer->getTransformedName(fieldName, metadata, local);
		stream << transformedName;
		if (epochVersion > 0) {
			stream << "_lag_" << epochVersion;
		}
	}
}

bool FieldAccess::isEnvArrayDim() {
	if (this->type != Type::dimensionType) return false;
	if (base == NULL) return false;
	FieldAccess *baseFieldAccess = dynamic_cast<FieldAccess*>(base);
	if (baseFieldAccess == NULL) return false;
	ArrayType *array = dynamic_cast<ArrayType*>(baseFieldAccess->getType());
	StaticArrayType *staticArray = dynamic_cast<StaticArrayType*>(baseFieldAccess->getType());
	if (array == NULL || staticArray != NULL) return false;
	Expr *superBase = baseFieldAccess->getBase();
	if (superBase == NULL) return false;
	NamedType *superBaseType = dynamic_cast<NamedType*>(superBase->getType());
	if (superBaseType == NULL || !superBaseType->isEnvironmentType()) return false;
	return true;
}

void FieldAccess::translateEnvArrayDim(std::ostringstream &stream,
		int indentLevel,
		int currentLineLength, Space *space) {
	
	DimensionIdentifier *dimension = dynamic_cast<DimensionIdentifier*>(field);
	int dimIndex = dimension->getDimensionNo() - 1;
	FieldAccess *baseFieldAccess = dynamic_cast<FieldAccess*>(base);
	const char *arrayName = baseFieldAccess->getField()->getName();
	FieldAccess *superBaseAccess = dynamic_cast<FieldAccess*>(baseFieldAccess->getBase());
	const char *envName = superBaseAccess->getField()->getName();
	
	stream << envName << "->getItem(\"" << arrayName << "\")->getDimension(" << dimIndex << ")";
}

void FieldAccess::translateIndex(std::ostringstream &stream, const char *array, int dimension) {
	stream << field->getName();
	stream << array;
	stream << dimension;	
}
