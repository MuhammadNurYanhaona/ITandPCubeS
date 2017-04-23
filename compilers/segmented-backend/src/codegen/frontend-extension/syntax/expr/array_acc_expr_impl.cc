#include "../../../../../../frontend/src/syntax/ast_expr.h"
#include "../../../../../../frontend/src/syntax/ast_type.h"
#include "../../../../../../frontend/src/semantics/task_space.h"
#include "../../../utils/name_transformer.h"
#include "../../../utils/code_constant.h"

#include <sstream>
#include <iostream>
#include <cstdlib>
#include <stack>

void ArrayAccess::translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space) {
	Type *baseType = base->getType();
	if (dynamic_cast<StaticArrayType*>(baseType) != NULL) {
		base->translate(stream, indentLevel, currentLineLength, space);
		stream << "["; 
		index->translate(stream, indentLevel, currentLineLength, space);
		stream << "]";
	} else {
		// Note that since user defined objects cannot have dynamic arrays, any dynamic array access must be
		// an access to a dynamic task global variable. Therefore we can assume that the base of this array
		// access has the name of that variable and its type is an array type.
		Expr *endpoint = getEndpointOfArrayAccess();
		endpoint->translate(stream, indentLevel, currentLineLength, space);
		std::ostringstream indexStream;
		ArrayType *arrayType = (ArrayType*) endpoint->getType();
		const char *arrayName = endpoint->getBaseVarName();
		// we now translate a possibly multidimensional array access into a unidimensional access as arrays
		// are stored as 1D memory blocks in the back-end
		generate1DIndexAccess(indexStream, indentLevel, arrayName, arrayType, space);
		// then we write the index access logic in the array
		stream << '[' << indexStream.str() << ']';
	}
}

void ArrayAccess::generate1DIndexAccess(std::ostringstream &stream, int indentLevel, 
		const char *array, ArrayType *type, Space *space) {
	
	int dimension = getIndexPosition();
	// If this is not the beginning index access in the expression then pass the computation forward first to get 
	// the earlier part been translated.
	if (dimension > 0) {
		ArrayAccess *previous = dynamic_cast<ArrayAccess*>(base);
		previous->generate1DIndexAccess(stream, indentLevel, array, type, space);
		stream << " + ";
	}
	int dimensionCount = type->getDimensions();
	// If the index access corresponds to some index of a parallel or sequential loop then it is readily available 
	// some pre-translated expression holder variable. So write the name of the variable in the stream
	FieldAccess *indexAccess = dynamic_cast<FieldAccess*>(index);
	if (indexAccess != NULL && indexAccess->isIndex()) {
		indexAccess->translateIndex(stream, array, dimension);
	// Otherwise, there might be a need for translating the index
	} else {
		std::ostringstream indexStream;
		index->translate(indexStream, 0, 0, space);
		stream << "((long) (";
		// if the current dimension is reordered at any point then we need an elaborate original to transformed
		// index conversion to be able to locate the storage address for the array index
		ArrayDataStructure *structure = (ArrayDataStructure*) space->getLocalStructure(array);
		if (structure->isDimensionLocallyReordered(dimension + 1)) {
			generateXformedIndex(stream, indentLevel, 
					indexStream.str().c_str(), array, dimension + 1, space);
		} else { 
			stream << indexStream.str();
		}
                stream << " - " << array << "StoreDims[" << dimension << "].range.min";
		stream << "))";
		std::ostringstream xform;
                for (int i = dimensionCount - 1; i > dimension; i--) {
                        stream << " * ((long) (" << array << "StoreDims[" << i << "].length))";
                }
	}
}

void ArrayAccess::generateXformedIndex(std::ostringstream &stream, int indentLevel, 
		const char *indexExpr, 
		const char *arrayName, int dimensionNo, Space *space) {
	
	std::string stmtSeparator = ";\n";
	std::ostringstream indent;
	for (int i = 0; i < indentLevel; i++) indent << '\t';
	indent << "\t\t";
	ntransform::NameTransformer *transformer = ntransform::NameTransformer::transformer;
	std::string lpuPrefix = transformer->getLpuPrefix();
	
	// The array-data-structure configuration in the semantic analysis starts array indexing from 1 as opposed to 
	// from zero. Thus, we need to make an adjustment when identifying the part-dimension objects. 
	std::ostringstream partConfigVar;
	partConfigVar << "(&" << lpuPrefix << arrayName << "PartDims[" << dimensionNo - 1 << "])";

	Space *rootSpace = space->getRoot();
	ArrayDataStructure *array = (ArrayDataStructure*) space->getLocalStructure(arrayName);
	
	std::stack<const char*> partConfigsStack;
	std::stack<ArrayDataStructure*> parentStructsStack;
	
	/* steps of the algorithm:	
		1. skip structure references until the first point of reorder has been reached
		2. add a reference to the stack if it is a point of reorder, or
		3. it is a reference that first partition the concerned dimension above a point of reorder  	
	*/

	ArrayDataStructure *parentArray = array;
	std::ostringstream parentArrows;
	while (!parentArray->isDimensionLocallyReordered(dimensionNo)) {
		parentArrows << "->parent";
		parentArray = (ArrayDataStructure*) parentArray->getSource();
	}
	while (parentArray != NULL && parentArray->getSource() != NULL) {
		
		if (parentArray->isDimensionLocallyReordered(dimensionNo)) {
			partConfigsStack.push(strdup(parentArrows.str().c_str()));
			parentStructsStack.push(parentArray);

		} else if (parentArray->isPartitionedAlongDimension(dimensionNo)) {
			ArrayDataStructure *lastArray = parentStructsStack.top();
			if (lastArray->isDimensionLocallyReordered(dimensionNo)) {
				partConfigsStack.push(strdup(parentArrows.str().c_str()));
				parentStructsStack.push(parentArray);
			}
		}
		if (parentArray->isPartitionedAlongDimension(dimensionNo) 
				&& !parentArray->isDimensionReordered(dimensionNo, rootSpace)) {
			break;
		}
		parentArrows << "->parent";
		parentArray = (ArrayDataStructure*) parentArray->getSource();
	}

	/* steps of the algorithm
		1. start by putting the original expression on the transform variable, xformIndex
		2. start popping elements from the stack
		3. at each step update the partConfig object to refer to appropriate parent link
		4. retrieve the parent structure reference and check if it is a point of reorder
		5. if it is a point of reorder then get the transform expression over xformIndex and assign the result to it
		6. if it is not a point of reorder then adjust xformIndex to be a zero-based index
		7. finally, write the xformIndex in the stream  
	*/

	stream << "(xformIndex = (" << indexExpr << ")";
	while (!partConfigsStack.empty()) {
		const char *pointerLinks = partConfigsStack.top();
		ArrayDataStructure *parentArray = parentStructsStack.top();
		stream << '\n' << indent.str();
		stream << ", partConfig = *" << partConfigVar.str() << pointerLinks;
		stream << '\n' << indent.str();
		stream << ", xformIndex = ";
		if (parentArray->isDimensionLocallyReordered(dimensionNo)) {
			stream << parentArray->getIndexXfromExpr(dimensionNo, "xformIndex");
		} else {
			stream << "partConfig.normalizeIndex(xformIndex)";
		}
		partConfigsStack.pop();
		parentStructsStack.pop();
	}
	stream << '\n' << indent.str();
	stream << ", xformIndex)";	
}
