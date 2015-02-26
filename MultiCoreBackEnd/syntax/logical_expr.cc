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
#include <stack>
#include <deque>

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

void LogicalExpr::getIndexRestrictExpr(List<LogicalExpr*> *exprList, std::ostringstream &stream,
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
				transformIndexRestriction(stream, "localIterationStart", 
						arrayName, dimensionNo, indentLevel + 1, space, true);
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
				transformIndexRestriction(stream, "localIterationBound", 
						arrayName, dimensionNo, indentLevel + 1, space, false);
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
				transformIndexRestriction(stream, "localIterationBound", 
						arrayName, dimensionNo, indentLevel + 1, space, false);
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
				transformIndexRestriction(stream, "localIterationStart", 
						arrayName, dimensionNo, indentLevel + 1, space, true);
			}
		}
		stream << indent.str() << "}\n";
		

		// Finally, before applying the updated conditions we need to validate that new conditions will
		// not expand the range been traversed as oppose to restrict it further. This can happen often
		// when the range corresponds to some array dimension.
		stream << indent.str();
		stream << "if (" << rangeExpr << ".min > " << rangeExpr << ".max) {\n";
		stream << indent.str() << '\t';
		stream << "iterationStart = std::min(iterationStart, localIterationStart)" << stmtSeparator;
		stream << indent.str() << '\t';
		stream << "iterationBound = std::max(iterationBound, localIterationBound)" << stmtSeparator;
		stream << indent.str() << "} else {\n";
		stream << indent.str() << '\t';
		stream << "iterationStart = std::max(iterationStart, localIterationStart)" << stmtSeparator;
		stream << indent.str() << '\t';
		stream << "iterationBound = std::min(iterationBound, localIterationBound)" << stmtSeparator;
		stream << indent.str() << "}\n";
		
		// close the local scope 
		stream << indent.str() << "}// scope exit for applying index loop restrictions\n"; 	
	} 
}

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
		List<FieldAccess*> *rightAccessList = right->getTerminalFieldAccesses();
		for (int i = 0; i < rightAccessList->NumElements(); i++) {
			if (strcmp(loopIndex, rightAccessList->Nth(i)->getField()->getName()) == 0) {
				return 0;
			}
		}
		return 1;	
	} else if (rightField != NULL && rightField->isTerminalField() 
			&& strcmp(rightField->getField()->getName(), loopIndex) == 0) {
		List<FieldAccess*> *leftAccessList = left->getTerminalFieldAccesses();
		for (int i = 0; i < leftAccessList->NumElements(); i++) {
			if (strcmp(loopIndex, leftAccessList->Nth(i)->getField()->getName()) == 0) {
				return 0;
			}	
		}
		return -1;
	}

	return 0;
}

void LogicalExpr::transformIndexRestriction(std::ostringstream &stream,
                        const char *varName, const char *arrayName, int dimensionNo,
                        int indentLevel, Space *space, 
			bool normalizedToMinOfRange) {

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
			}

			// get the reference of the partition dimension configuration object for current LPS
			stream << indent.str();
			stream << "partConfig = *" << partConfigVar.str() << pointerLinks;
			stream << stmtSeparator;

			// then transform the index	
			stream << indent.str();
			stream << varName << " = ";
                        stream << parentArray->getImpreciseBoundOnXformedIndex(dimensionNo, 
					varName, normalizedToMinOfRange);
			stream << stmtSeparator;
		}
		
		// keep track of the array and pointer links to get to its partition config of the last LPS 
		// regardless of its been order-preserving or reordering.
		lastArray = parentArray;
		lastPointerLinks = pointerLinks;
	
		partConfigsStack.pop();
                parentStructsStack.pop();
	} 
}

