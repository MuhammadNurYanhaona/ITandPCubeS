#include "../../../utils/name_transformer.h"
#include "../../../../../../common-libs/utils/list.h"
#include "../../../../../../common-libs/domain-obj/constant.h"
#include "../../../../../../frontend/src/syntax/ast_expr.h"
#include "../../../../../../frontend/src/semantics/task_space.h"

#include <sstream>
#include <iostream>
#include <cstdlib>
#include <stack>

void LogicalExpr::translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space) {
        stream << "(";
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
        stream << ")";
}

// Local expressions sometimes are added to indexed based parallel loop statement blocks to further restrict the 
// range of indexes been traversed by the for loop. Given that there might be multiple nested for loops in the target 
// code correspond to a single loop in IT source, we need to know what loop is the best place for a restricting 
// logical expression to put into and then do that. For this to be done efficiently we may need to break restricting 
// conditions connected by AND operators and put different parts in different location. So the following method has 
// been added to break a collective of AND statements into a list of such statements
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

// This function generates, as its name suggests, a starting or ending condition for an index from logical expressions 
// that are used as restricting conditions in parallel loops based on that index. An example of such restriction can 
// be "do { ... } for k in matrix AND k > expr." Here we will like to begin the translated C++ loops to start from 
// "k = expr + 1." On the other hand, if the range to be traversed by the index is a decreasing range than the same 
// condition should be used to exit from the loop instead of as a starting condition. If none of the expressions in 
// the list can be used to restrict generated index loop this way due to the nature of those expressions, then it does 
// nothing. The last three parameters are for determining if the index under concern traverses a reordered array 
// dimension, and if it does then transform the index start or end restriction that may be applied to the added 
// restrictions. The function returns a filtered list of index restriction expressions if some of the expressions in
// the original list can be successfully and precisely applied on the loop. Otherwise, it returns the original list
List<LogicalExpr*> *LogicalExpr::getIndexRestrictExpr(List<LogicalExpr*> *exprList,
		std::ostringstream &stream,
		const char *indexVar, const char *rangeExpr,
		int indentLevel, Space *space,
		bool xformedArrayRange, const char *arrayName, int dimensionNo) {

	// we create two local references for possible restriction on beginning and ending conditions
	LogicalExpr *gteExpr = NULL;
	LogicalExpr *lteExpr = NULL;

	// two flags to determine if we need to investigate further for restricting conditions
	bool gteFound = false;
	bool lteFound = false;
	
	// finally two marker variables to keep track on which side of a logical expression the loop index 
	// lies if the expression is a restricting condition of the form we desire
	int gteMarker = 0;
	int lteMarker = 0;

	for (int i = 0; i < exprList->NumElements(); i++) {
		LogicalExpr *currentExpr = exprList->Nth(i);
		int eval = currentExpr->isLoopRestrictExpr(indexVar);
		if (eval == 0) continue;
	
		// Note that the logic below ensures that the first candidate expressions for the loop start
		// and end conditions are retained. We may not need this.		
		LogicalOperator op = currentExpr->getOp();
		if (op == GT || op == GTE) {
			if (eval == -1 && !lteFound) {
				lteFound = true;
				lteMarker = -1;
				lteExpr = currentExpr;
			} else if (eval == 1 && !gteFound) {
				gteFound = true;
				gteMarker = 1;
				gteExpr = currentExpr;
			}
		} else {
			if (eval == -1 && !gteFound) {
				gteFound = true;
				gteMarker = -1;
				gteExpr = currentExpr;
			} else if (eval == 1 && !lteFound) {
				lteFound = true;
				lteMarker = 1;
				lteExpr = currentExpr;
			}
		}
	}

	// declare a list that contains expressions (o to 2) that should be filtered out from the original 
	// list
	List<LogicalExpr*> *excludedList = new List<LogicalExpr*>;

	// if we have any restricting condition then we can generate if conditions that will further 
	// restrict the loop index beginning or terminal conditions from what they are set to by default
	if (gteFound || lteFound) {

		std::string stmtSeparator = ";\n";	

		// first generate the indent string
		std::ostringstream indent;
		for (int i = 0; i < indentLevel; i++) indent << "\t";

		// create a local scope in case index transforms are needed to apply the restrictions 
		stream << indent.str() << "{// scope entrance for applying index loop restrictions\n";

		// initially assign the index start and end bounds that are applied by default to some
		// local variables
		stream << indent.str() << "int localIterationStart = iterationStart" << stmtSeparator; 	
		stream << indent.str() << "int localIterationBound = iterationBound" << stmtSeparator;

		// generate an if else block that will check if the index is increasing or decreasing
		// and based on that apply appropriate restrictions
		stream << indent.str();
		stream << "if (" << rangeExpr << ".min > " << rangeExpr << ".max) {\n";
		// when range is decreasing then less than equal condition should apply to the beginning
		// and greater than equal condition should apply to the ending
		if (lteFound) {
			Expr *lteCond = (lteMarker == 1) ? lteExpr->getRight() : lteExpr->getLeft();
			stream << indent.str() << "\t";
			stream << "localIterationStart = ";
			lteCond->translate(stream, indentLevel + 1, 0, space);
			if ((lteMarker == 1 && lteExpr->getOp() == LT) 
					|| (lteMarker == -1 && lteExpr->getOp() == GT)) {
				stream << " - 1";
			}
			stream << stmtSeparator;
			// if the index is traversing an array dimension range and the dimension of the 
			// array has been reordered then apply equivalent transformation on the iteration 
			// start condition
			if (xformedArrayRange) {
				bool precise = transformIndexRestriction(stream, "localIterationStart", 
						arrayName, dimensionNo, 
						indentLevel + 1, space, true, false);
				if (precise) excludedList->Append(lteExpr);
			} else {
				excludedList->Append(lteExpr);
			}
		}
		if (gteFound) {
			Expr *gteCond = (gteMarker == 1) ? gteExpr->getRight() : gteExpr->getLeft();
			stream << indent.str() << "\t";
			stream << "localIterationBound = ";
			gteCond->translate(stream, indentLevel + 1, 0, space);
			if ((gteMarker == 1 && gteExpr->getOp() == GT) 
					|| (gteMarker == -1 && gteExpr->getOp() == LT)) {
				stream << " + 1";
			}
			stream << stmtSeparator;
			// similar to the above case of iteration start condition, appy a transformation
			// on the iteration end condition, if applicable. 
			if (xformedArrayRange) {
				bool precise = transformIndexRestriction(stream, "localIterationBound", 
						arrayName, dimensionNo, 
						indentLevel + 1, space, false, true);
				if (precise) excludedList->Append(gteExpr);
			} else {
				excludedList->Append(gteExpr);
			}
		}

		// on the other hand, when the range is increasing, which is the normal case, greater than
		// condition should be applied to the beginning and less than at the ending
		stream << indent.str() << "} else {\n";
		if (lteFound) {
			Expr *lteCond = (lteMarker == 1) ? lteExpr->getRight() : lteExpr->getLeft();
			stream << indent.str() << "\t";
			stream << "localIterationBound = ";
			lteCond->translate(stream, indentLevel + 1, 0, space);
			if ((lteMarker == 1 && lteExpr->getOp() == LT) 
					|| (lteMarker == -1 && lteExpr->getOp() == GT)) {
				stream << " - 1";
			}
			stream << stmtSeparator;
			if (xformedArrayRange) {
				bool precise = transformIndexRestriction(stream, "localIterationBound", 
						arrayName, dimensionNo, 
						indentLevel + 1, space, false, true);
				if (precise) excludedList->Append(lteExpr);
			} else {
				excludedList->Append(lteExpr);
			}
		}
		if (gteFound) {
			Expr *gteCond = (gteMarker == 1) ? gteExpr->getRight() : gteExpr->getLeft();
			stream << indent.str() << "\t";
			stream << "localIterationStart = ";
			gteCond->translate(stream, indentLevel + 1, 0, space);
			if ((gteMarker == 1 && gteExpr->getOp() == GT) 
					|| (gteMarker == -1 && gteExpr->getOp() == LT)) {
				stream << " + 1";
			}
			stream << stmtSeparator;
			if (xformedArrayRange) {
				bool precise = transformIndexRestriction(stream, "localIterationStart", 
						arrayName, dimensionNo, 
						indentLevel + 1, space, true, false);
				if (precise) excludedList->Append(gteExpr);
			} else {
				excludedList->Append(gteExpr);
			}
		}
		stream << indent.str() << "}\n";
		

		// Finally, before applying the updated conditions we need to validate that new conditions will
		// not expand the range been traversed as oppose to restrict it further. This can happen often
		// when the range corresponds to some array dimension.
		stream << indent.str();
		stream << "if (" << rangeExpr << ".min > " << rangeExpr << ".max) {\n";
		stream << indent.str() << '\t';
		stream << "iterationStart = min(iterationStart, localIterationStart)" << stmtSeparator;
		stream << indent.str() << '\t';
		stream << "iterationBound = max(iterationBound, localIterationBound)" << stmtSeparator;
		stream << indent.str() << "} else {\n";
		stream << indent.str() << '\t';
		stream << "iterationStart = max(iterationStart, localIterationStart)" << stmtSeparator;
		stream << indent.str() << '\t';
		stream << "iterationBound = min(iterationBound, localIterationBound)" << stmtSeparator;
		stream << indent.str() << "}\n";
		
		// close the local scope 
		stream << indent.str() << "}// scope exit for applying index loop restrictions\n"; 	
	}

	// finally, filter out the excluded expressions from the original list
 	List<LogicalExpr*> *filterList = new List<LogicalExpr*>;
	for (int i = 0; i < exprList->NumElements(); i++) {
		LogicalExpr *currentExpr = exprList->Nth(i);
		bool found = false;
		for (int j = 0; j < excludedList->NumElements(); j++) {
			if (currentExpr == excludedList->Nth(j)) {
				found = true;
				break;
			}
		}
		if (!found) filterList->Append(currentExpr);
	}
	return filterList;
}

// This is a supporting function for the function above to determine whether to consider of skip an expression. 
// Instead of a boolean value, it returns an integer as we need to know on which side of the expression the index 
// variable lies. So it returns -1 if the expression is a loop restrict condition and the loop index is on the 
// right of the expression, 1 if the loop index is on the left, and 0 if the expression is not a loop restrict 
// condition.
int LogicalExpr::isLoopRestrictExpr(const char *loopIndex) {
	
	// only greater or less than comparison can be used for loop index restrictions
	if (left == NULL) return 0;
	if (!(op == GT || op == LT || op == GTE || op == LTE)) return 0;

	// for a valid candidate loop restricting condition the loop index should lie only one side
	// of the expression	
	FieldAccess *leftField = dynamic_cast<FieldAccess*>(left);
	FieldAccess *rightField = dynamic_cast<FieldAccess*>(right);
	if (leftField != NULL && leftField->isTerminalField() 
			&& strcmp(leftField->getField()->getName(), loopIndex) == 0) {
		List<FieldAccess*> *rightAccessList = new List<FieldAccess*>;
		right->retrieveTerminalFieldAccesses(rightAccessList);
		for (int i = 0; i < rightAccessList->NumElements(); i++) {
			if (strcmp(loopIndex, rightAccessList->Nth(i)->getField()->getName()) == 0) {
				return 0;
			}
		}
		return 1;	
	} else if (rightField != NULL && rightField->isTerminalField() 
			&& strcmp(rightField->getField()->getName(), loopIndex) == 0) {
		List<FieldAccess*> *leftAccessList = new List<FieldAccess*>;
		left->retrieveTerminalFieldAccesses(leftAccessList);
		for (int i = 0; i < leftAccessList->NumElements(); i++) {
			if (strcmp(loopIndex, leftAccessList->Nth(i)->getField()->getName()) == 0) {
				return 0;
			}	
		}
		return -1;
	}

	return 0;
}

// This function transforms a variable holding the value of an index restricting expression based on the partitioning 
// of the array dimension that the index is traversing -- when the loop corresponds to a reordered array dimension 
// traversal, of course. Notice the second last parameter of this function. This is used to determine what to set the 
// value of the variable to if it falls outside the range of the dimension that falls within the LPU where the generated 
// code will execute. This is needed as this function is used in context where we do not know if the variable is within 
// the boundary of the LPU. The last parameter is used to determine if a lower bound or an upper bound should be 
// attempted by the transformation process when the index is not in within the boundary of the LPU and we are sure about 
// its position relative to the LPU boundary. TODO probably we can exclude the second last parameter if we do some 
// refactoring in the implementation. Varify the correctness of the new implementation if you attempt that.
// The function returns a boolean value indicating if it made a precise transformation of the given restriction or not. 
bool LogicalExpr::transformIndexRestriction(std::ostringstream &stream,
		const char *varName, const char *arrayName, int dimensionNo,
		int indentLevel, Space *space,
		bool normalizedToMinOfRange, bool lowerBound) {

	bool preciseTransformation = true;
	std::string stmtSeparator = ";\n";
	std::string paramSeparator = ", ";
        std::ostringstream indent;
        for (int i = 0; i < indentLevel; i++) indent << '\t';

	// get the reference name for the partition dimension configuration object for the concerned
	// array's concerned dimension
        ntransform::NameTransformer *transformer = ntransform::NameTransformer::transformer;
        std::string lpuPrefix = transformer->getLpuPrefix();
        std::ostringstream partConfigVar;
        partConfigVar << "(&" << lpuPrefix << arrayName << "PartDims[" << dimensionNo - 1 << "])";

        // get the Root Space reference to aid in determining the extend of reordering later
        Space *rootSpace = space->getRoot();

        // grab array configuration information for current LPS
        ArrayDataStructure *array = (ArrayDataStructure*) space->getLocalStructure(arrayName);

	// create two stacks for traversing parent pointers; stacks are needed as transformation should 
	// be done from top to bottom in the LPS hierarchy
        std::stack<const char*> partConfigsStack;
        std::stack<ArrayDataStructure*> parentStructsStack;

	// reach the first reference of the array in the part config stack where the array along this
	// dimension is reordered by the its partitioning functions
	std::ostringstream parentArrows;
	while (!array->isDimensionLocallyReordered(dimensionNo)) {
		array = (ArrayDataStructure*) array->getSource();
		parentArrows << "->parent";
	}

	// reverse track references up to one LPS up to foremost reorder point and store all points where
	// the array has been partitioned along concerned dimension
	while (array != NULL) {
                if (array->isPartitionedAlongDimension(dimensionNo)) {
                        partConfigsStack.push(strdup(parentArrows.str().c_str()));
                        parentStructsStack.push(array);
                }

                if (array->isPartitionedAlongDimension(dimensionNo)
                        && !array->isDimensionReordered(dimensionNo, rootSpace)) break;

                parentArrows << "->parent";
                array = (ArrayDataStructure*) array->getSource();
        }

	// then transform the variable in a top-down fasion from uppermost to the lowermost reorder point
	ArrayDataStructure *lastArray = NULL;
	const char *lastPointerLinks = NULL;
	while (!partConfigsStack.empty()) {
		const char *pointerLinks = partConfigsStack.top();
                ArrayDataStructure *parentArray = parentStructsStack.top();

		// if the dimension is reordered in current LPS then apply transformation expression here
		// over current value of the transformed variable
		if (parentArray->isDimensionLocallyReordered(dimensionNo)) {

			// first normalize the index first if the reoredering LPS is preceeded by an order 
			if (lastArray != NULL && !lastArray->isDimensionLocallyReordered(dimensionNo)) {
				stream << indent.str();
				stream << "partConfig = *" << partConfigVar.str() << lastPointerLinks;
				stream << stmtSeparator;
				stream << indent.str();
				stream << varName << " = ";
				// note that safe-normalization is used as we do not know if the transformed
				// variable's value falls inside the LPU boundary for the underlying array
				// dimension and we need to set the value to partitioned dimension's min or
				// max value to probable incorrect computation as a result of non-sensical 
				// transformation 	
				stream << "partConfig.safeNormalizeIndex(" << varName;
				stream << paramSeparator;
				if (normalizedToMinOfRange) stream << "true";
				else stream << "false";
				stream << ")" << stmtSeparator;
				preciseTransformation = false;
			}

			// get the reference of the partition dimension configuration object for current LPS
			stream << indent.str();
			stream << "partConfig = *" << partConfigVar.str() << pointerLinks;
			stream << stmtSeparator;

			// then transform the index	
			stream << indent.str();
			stream << varName << " = ";
                        stream << parentArray->getImpreciseBoundOnXformedIndex(dimensionNo, 
					varName, lowerBound, indentLevel);
			stream << stmtSeparator;
		}
		
		// keep track of the array and pointer links to get to its partition config of the last LPS 
		// regardless of its been order-preserving or reordering.
		lastArray = parentArray;
		lastPointerLinks = pointerLinks;
	
		partConfigsStack.pop();
                parentStructsStack.pop();
	}

	return preciseTransformation; 
}
