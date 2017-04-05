#include "../../../../../../frontend/src/syntax/ast_expr.h"
#include "../../../../../../frontend/src/semantics/task_space.h"
#include "../../../utils/array_assignment.h"
#include "../../../utils/code_constant.h"

#include <sstream>
#include <iostream>
#include <cstdlib>

void AssignmentExpr::translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space) {
	left->translate(stream, indentLevel, currentLineLength, space);
	stream << " = ";
	right->translate(stream, indentLevel, currentLineLength, space);
}

void AssignmentExpr::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {

	// First, check if the assignment expression is an assignment of an array to another. If that is the case 
	// then we have to use an alternative, specific translation of this expression.
	if (isArrayAssignment(this)) {
		AssignmentDirectiveList *directives = new AssignmentDirectiveList(this);
		directives->generateCode(stream, indentLevel, space);
	} else {
		// If the right is also an assignment expression then this is a compound statement. Break it up into
		// several simple statements.
		AssignmentExpr *rightExpr = dynamic_cast<AssignmentExpr*>(right);
		Expr *rightPart = right;
		if (rightExpr != NULL) {
			rightPart = rightExpr->left;
			rightExpr->generateCode(stream, indentLevel, space);	
		}

		// Check if the assignment is a new declaration of a dynamic array. If it is so, we can ignore it as
		// such a variable is only a placeholder for some other environment variable and is already declared 
		// at the time of scope translation.
		if (!ObjectCreate::isDynamicArrayCreate(right)) {	

			// If this is a dimension assignment statement with multiple dimensions of an array assigned 
			// once to another one then we need to handle it separately.
			Type *leftType = left->getType();
			FieldAccess *fieldAccess = dynamic_cast<FieldAccess*>(left);
			if (leftType == Type::dimensionType 
					&& fieldAccess != NULL 
					&& fieldAccess->getBase() != NULL) {
				
				ArrayType *array = dynamic_cast<ArrayType*>(fieldAccess->getBase()->getType());
				DimensionIdentifier *dimension 
						= dynamic_cast<DimensionIdentifier*>(fieldAccess->getField());
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

			// if the right side of the assignment is a new operation, i.e., an object instantiation then
			// there might be some arguments been passed to the constructor. Those arguments need to be 
			// assigned to appropriate properties of the object as subsequent statements.  
			ObjectCreate *objectCreate = dynamic_cast<ObjectCreate*>(rightPart);
			if (objectCreate != NULL) {
				objectCreate->generateCodeForProperties(left, stream, indentLevel);
			}
		}
	}
}

