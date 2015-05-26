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

//------------------------------------------------ Range Expressions --------------------------------------------------/

RangeExpr::RangeExpr(Identifier *i, Expr *r, Expr *s, bool l, yyltype loc) : Expr(loc) {
	Assert(i != NULL && r != NULL);
	index = i;
	index->SetParent(this);
	range = r;
	range->SetParent(this);
	step = s;
	if (step != NULL) {
		step->SetParent(this);
	}
	loopingRange = l;
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

Hashtable<VariableAccess*> *RangeExpr::getAccessedGlobalVariables(TaskGlobalReferences *globalReferences) {
	
	Hashtable<VariableAccess*> *table = range->getAccessedGlobalVariables(globalReferences);
	Iterator<VariableAccess*> iter = table->GetIterator();
	VariableAccess *accessLog;
	while ((accessLog = iter.GetNextValue()) != NULL) {
		if (accessLog->isMetadataAccessed()) accessLog->getMetadataAccessFlags()->flagAsRead();
		if (accessLog->isContentAccessed()) accessLog->getContentAccessFlags()->flagAsRead();
	}

	const char *indexName = index->getName();
	if (globalReferences->isGlobalVariable(indexName)) {
		accessLog = new VariableAccess(indexName);
		accessLog->markContentAccess();
		if (loopingRange) { 
			accessLog->getContentAccessFlags()->flagAsWritten();
		}
		accessLog->getContentAccessFlags()->flagAsRead();
		if (table->Lookup(indexName) != NULL) {
			table->Lookup(indexName)->mergeAccessInfo(accessLog);
		} else table->Enter(indexName, accessLog, true);
	}
	
	if (step == NULL) return table;
	Hashtable<VariableAccess*> *sTable = step->getAccessedGlobalVariables(globalReferences);
	iter = sTable->GetIterator();
	while ((accessLog = iter.GetNextValue()) != NULL) {
		if (accessLog->isMetadataAccessed()) accessLog->getMetadataAccessFlags()->flagAsRead();
		if (accessLog->isContentAccessed()) accessLog->getContentAccessFlags()->flagAsRead();
	}
	mergeAccessedVariables(table, sTable);

	return table;
}

void RangeExpr::setEpochVersions(Space *space, int epoch) {
	range->setEpochVersions(space, epoch);
	if (step != NULL) step->setEpochVersions(space, epoch);
}

void RangeExpr::translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space) {

	// get the appropriate back-end name for the index variable
	ntransform::NameTransformer *transformer = ntransform::NameTransformer::transformer;
        const char *indexVar = transformer->getTransformedName(index->getName(), false, false);

	// Check if the range expression is correspond to a local dimension range of any global array. If it
	// is so then we translate the range condition differently as there are chances of index range 
	// reordering due to partition function configuration.
	bool arrayAccessRange = false;
	const char *baseVar = range->getBaseVarName();
	if (baseVar != NULL) {
		List<FieldAccess*> *fieldAccessList = range->getTerminalFieldAccesses();
		bool localArray = false;
		for (int i = 0; i < fieldAccessList->NumElements(); i++) {
			FieldAccess *fieldAccess = fieldAccessList->Nth(i);
			if (strcmp(fieldAccess->getField()->getName(), baseVar) == 0) {
				localArray = fieldAccess->isLocal();
				break;
			}
		}
		DataStructure *structure = space->getLocalStructure(baseVar);
		if (structure != NULL) {
			ArrayDataStructure *array = dynamic_cast<ArrayDataStructure*>(structure);
			arrayAccessRange = (array != NULL) && localArray;
		}
	}

	// if this is an array access range then invoke the function that takes care of that case exclusively
	if (arrayAccessRange) {
		translateArrayRangeExprCheck(stream, indentLevel, space);
	// otherwise, translate the range condition in a simple way
	} else {	
		// translate the range condition
		std::ostringstream rangeStr;
		range->translate(rangeStr, indentLevel, 0, space);
		std::string rangeCond = rangeStr.str();

		std::ostringstream indent;
		for (int i = 0; i < indentLevel; i++) indent << '\t';
		indent << "\t\t";

		// by default a range expression is translated as a boolean condition
		
		// check if the index variable is larger than or equal to the min value of range
		stream << '(' << indexVar << " >= " << rangeCond << ".min"; 
		stream << '\n' << indent.str() << " && ";
		// and less than or equal to the max value of range
		stream << indexVar << " <= "<< rangeCond << ".max"; 
		stream << '\n' << indent.str() << " && ";
		// and range mean is less than range max
		stream  << rangeCond << ".min < " << rangeCond << ".max)";
		// otherwise, do the exact inverse calculation
		stream << '\n' << indent.str() << " || ";	
		stream << '(' << indexVar << " <= " << rangeCond << ".min"; 
		stream << '\n' << indent.str() << " && ";
		stream << indexVar << " >= " << rangeCond << ".max)";
	}
}

const char *RangeExpr::getIndexExpr() {
	ntransform::NameTransformer *transformer = ntransform::NameTransformer::transformer;
        return transformer->getTransformedName(index->getName(), false, false);
}

const char *RangeExpr::getRangeExpr(Space *space) {
	std::ostringstream rangeStr;
	range->translate(rangeStr, 0, 0, space);
	return strdup(rangeStr.str().c_str());
}

const char *RangeExpr::getStepExpr(Space *space) {
	std::ostringstream stepStr;
	if (step == NULL) stepStr << "1";
	else step->translate(stepStr, 0, 0, space);
	std::string stepCond = stepStr.str();
	return strdup(stepCond.c_str());
}

List<FieldAccess*> *RangeExpr::getTerminalFieldAccesses() {
	if (indexField == NULL) {
		indexField = new FieldAccess(NULL, index, *index->GetLocation());
	}
	List<FieldAccess*> *list = new List<FieldAccess*>;
	list->Append(indexField);
	Expr::copyNewFields(list, range->getTerminalFieldAccesses());
	if (step != NULL) Expr::copyNewFields(list, step->getTerminalFieldAccesses());
	return list;
}

const char *RangeExpr::getBaseArrayForRange(Space *executionSpace) {
	FieldAccess *rangeField = dynamic_cast<FieldAccess*>(range);
	if (rangeField == NULL) return NULL;
	if (rangeField->isTerminalField()) return NULL;
	const char *baseVar = rangeField->getBaseVarName();
	DataStructure *structure = executionSpace->getLocalStructure(baseVar);
	if (structure == NULL) return NULL;
	ArrayDataStructure *array = dynamic_cast<ArrayDataStructure*>(structure);
	if (array == NULL) return NULL;
	return baseVar;
}

// This implementation assumes that we know the range is related to some dimension of an array
int RangeExpr::getDimensionForRange(Space *executionSpace) {
	FieldAccess *rangeField = dynamic_cast<FieldAccess*>(range);
	Expr *base = rangeField->getBase();
	FieldAccess *baseField = dynamic_cast<FieldAccess*>(base);
	Identifier *field = baseField->getField();
	DimensionIdentifier *dimensionId = dynamic_cast<DimensionIdentifier*>(field);
	return dimensionId->getDimensionNo();	
}

void RangeExpr::generateLoopForRangeExpr(std::ostringstream &stream, 
		int indentation, Space *space, const char *loopBoundsRestrictCond) {
	
	std::string stmtSeparator = ";\n";
	std::ostringstream indent;
	for (int i = 0; i < indentation; i++) indent << '\t';

	const char *indexVar = this->getIndexExpr();                
	const char *rangeCond = this->getRangeExpr(space);
        const char *stepCond = this->getStepExpr(space);

	// find out if the used index need some index transformation before been used inside        
	bool involveIndexXform = false;
        const char *baseArray = this->getBaseArrayForRange(space);
        if (baseArray != NULL) {
		int dimension = this->getDimensionForRange(space);
		Space *rootSpace = space->getRoot();
		ArrayDataStructure *array = (ArrayDataStructure*) space->getLocalStructure(baseArray);
		if (array->isDimensionReordered(dimension, rootSpace)) {
        		involveIndexXform = true;
        	}
        }

        // create three new variables for setting appropriate loop  condition checking and index 
        // increment, and one variable to multiply index properly during looping
        stream << indent.str() << "int iterationStart = " << rangeCond << ".min";
        stream << stmtSeparator;
        stream << indent.str() << "int iterationBound = " << rangeCond << ".max";
        stream << stmtSeparator;
        stream << indent.str() << "int indexIncrement = " << stepCond << stmtSeparator;
        stream << indent.str() << "int indexMultiplier = 1" << stmtSeparator;
        stream << indent.str() << "if (" << rangeCond << ".min > " << rangeCond << ".max) {\n";
        stream << indent.str() << "\titerationBound *= -1" << stmtSeparator;
        stream << indent.str() << "\tindexIncrement *= -1" << stmtSeparator;
        stream << indent.str() << "\tindexMultiplier = -1" << stmtSeparator;
        stream << indent.str() << "}\n";

	// if index transformation is needed then declare transformed index variable
        std::ostringstream indexVarUsed;
        if (involveIndexXform) {
        	indexVarUsed << this->getIndexName() << "Xformed";
                stream << indent.str() << "int " << indexVarUsed.str() << stmtSeparator;
        } else indexVarUsed << indexVar;

	// if there is a loop restriction condition passed by the caller then apply it before creating the for loop
	if (loopBoundsRestrictCond != NULL) stream << loopBoundsRestrictCond;

        // write the for loop corresponding to the repeat instruction
        stream << indent.str() << "for (" << indexVarUsed.str() << " = " << "iterationStart; \n";
        stream << indent.str() << "\t\tindexMultiplier * " << indexVarUsed.str() << " <= iterationBound; \n";
        stream << indent.str() << "\t\t" << indexVarUsed.str() << " += indexIncrement) {\n";

	// if index transformation is used then do a reverse transformation to get to the original index
        if (involveIndexXform) {
		generateAssignmentExprForXformedIndex(stream, indentation + 1, space);
        }
	
	delete indexVar;
        delete rangeCond;
        delete stepCond;
}

void RangeExpr::translateArrayRangeExprCheck(std::ostringstream &exprStream, int indentLevel, Space *space) {
	
	std::string stmtSeparator = ";\n";
	std::ostringstream indent;
	for (int i = 0; i <= indentLevel; i++) indent << '\t';

	ntransform::NameTransformer *transformer = ntransform::NameTransformer::transformer;
        const char *indexVar = transformer->getTransformedName(index->getName(), false, false);
	std::string lpuPrefix = transformer->getLpuPrefix();
	
	const char *arrayName = getBaseArrayForRange(space);
	int dimensionNo = getDimensionForRange(space);

	std::ostringstream partConfigVar;
	partConfigVar << "(&" << lpuPrefix << arrayName << "PartDims[" << dimensionNo - 1 << "])";

	// get the Root Space reference to aid in determining the extend of reordering later
	Space *rootSpace = space->getRoot();
	
	// grab array configuration information for current LPS
	ArrayDataStructure *array = (ArrayDataStructure*) space->getLocalStructure(arrayName);
	
	// create two stacks for traversing parent pointers; stacks are needed as inclusion check should be 
	// done from top to bottom in LPS hierarchy
	std::stack<const char*> partConfigsStack;
	std::stack<ArrayDataStructure*> parentStructsStack;
	
	// put current array reference in stack so that it will be automatically processed with its
	// sources in higher LPSes
	partConfigsStack.push("");
	parentStructsStack.push(array);

	// if the array is reordered anywhere in the partition hierarchy starting from the root LPS down to 
	// current LPS then we have to recursively do inclusion testing on higher LPSes. Note that this 
	// implementation can be improved by considering only those points of reordering, but we do not do
	// that in the prototype implementation.	
	if (array->isDimensionReordered(dimensionNo, rootSpace)) {
		
		DataStructure *parent = array->getSource();
		std::ostringstream parentArrows;

		// Second condition in the while loop excludes the configuration object for root LPU as the 
		// array is not partitioned there. TODO we need to evaluate if this exclusion can lead to 
		// index out of bound errors in the generated code.
		while (parent != NULL && parent->getSource() != NULL) {
			parentArrows << "->parent";
			ArrayDataStructure *parentArray = (ArrayDataStructure*) parent;
			if (parentArray->isPartitionedAlongDimension(dimensionNo)) {
				partConfigsStack.push(strdup(parentArrows.str().c_str()));
				parentStructsStack.push(parentArray);
			}
			
			// we do not have to traverse past one partitioning LPS up to the foremost reorder 
			// point
			if (parentArray->isPartitionedAlongDimension(dimensionNo) 
					&& !parentArray->isDimensionReordered(dimensionNo, rootSpace)) {
				break;
			}

			parent = parent->getSource();
		}
	}
	
	// iterate over parent structure references and do index inclusion check in each in sequence
	ArrayDataStructure *lastArray = NULL;	 
	// a variable for tracking the number of inclusion checks has been made so far
	int clauseAdded = 0;
	while (!partConfigsStack.empty()) {
		const char *pointerLinks = partConfigsStack.top();
		ArrayDataStructure *parentArray = parentStructsStack.top();
		if (clauseAdded > 0) {
			exprStream << std::endl << indent.str() << '\t';
			exprStream << "&& (";
			// If the dimension has been reordered by previous LPS then we need to get the 
			// transformed index before we can do inclusion check for current LPS.	
			if (lastArray->isDimensionLocallyReordered(dimensionNo)) {
				exprStream << "xformIndex = ";
				exprStream << lastArray->getIndexXfromExpr(dimensionNo, "xformIndex");
				exprStream << ',' << std::endl << indent.str() << '\t';

			// Otherwise we have to check array has been reordered in current LPS; if it has
			// then we have to adjust the partition beginning to zero before doing inclusion
			// check. TODO this is a limitation of current prototype implementation that
			// LPU definition for reordering partition functions always start at zero. in
			// future when we will optimize the compiler then we need to find a way to avoid
			// adjusting indexes from order preserving partition functions during inclusion
			// check by redesigning the LPU configuration of reordering partition functions
			// appropriately.
			} else if (parentArray->isDimensionLocallyReordered(dimensionNo)) {
				exprStream << "xformIndex = partConfig.normalizeIndex(xformIndex)";
				exprStream << ',' << std::endl << indent.str() << '\t';
			}
		} else {
			exprStream << "(xformIndex = " << indexVar;
			exprStream << ',' << std::endl << indent.str() << '\t';
		}
		exprStream << "partConfig = *" << partConfigVar.str() << pointerLinks;
		exprStream << ',' << std::endl << indent.str() << '\t';
		
		// if the dimension is reordered in current LPS then we need to add the specific reorder 
		// expression applicable for the underlying partition function
		if (parentArray->isDimensionLocallyReordered(dimensionNo)) {
			exprStream << parentArray->getReorderedInclusionCheckExpr(
					dimensionNo, "xformIndex");
		// otherwise we can invoke the standard inline inclusion function on PartDimension object
		} else {
			exprStream << "partConfig.isIncluded(xformIndex)";
		}

		exprStream << ")";
		
		clauseAdded++; 
		lastArray = parentArray;
		partConfigsStack.pop();
		parentStructsStack.pop();
	}	
}

void RangeExpr::generateAssignmentExprForXformedIndex(std::ostringstream &stream, int indentLevel, Space *space) {
	
	std::string stmtSeparator = ";\n";
	std::ostringstream indent;
	for (int i = 0; i < indentLevel; i++) indent << '\t';

	ntransform::NameTransformer *transformer = ntransform::NameTransformer::transformer;
        const char *indexVar = transformer->getTransformedName(index->getName(), false, false);
	std::string lpuPrefix = transformer->getLpuPrefix();

	std::ostringstream xformIndexVar;
	xformIndexVar << index->getName() << "Xformed";
	
	const char *arrayName = getBaseArrayForRange(space);
	int dimensionNo = getDimensionForRange(space);

	// get the prefix that is common to reach any part-dimension object that will be used during reverse
	// transformation of transformed index
	std::ostringstream partConfigVar;
	partConfigVar << "(&" << lpuPrefix << arrayName << "PartDims[" << dimensionNo - 1 << "])";

	// get the Root Space reference to aid in determining the extend of reordering later
	Space *rootSpace = space->getRoot();
	
	// grab array configuration information for current LPS
	ArrayDataStructure *array = (ArrayDataStructure*) space->getLocalStructure(arrayName);

	// skip LPSes untill we get to the first LPS where the dimension has been reordered
	std::ostringstream parentArrows;
	while (!array->isDimensionLocallyReordered(dimensionNo)) {
		array = (ArrayDataStructure*) array->getSource();
		// we need to keep track how far we need to traverse the parent pointers to get to the first
		// index reorder point
		parentArrows << "->parent";
	}

	// Reverse transformation to the original storage index should take place in a bottom up fasion
	// from current LPS upto last reordered ancestor LPS. So we create a queue of ancestor partConfig
	// references and another for the pointers to get to them in generated code.
	std::deque<const char*> partConfigsQueue;
	std::deque<ArrayDataStructure*> parentStructsQueue;

	// iterate over parent structure references and put in queue all references that involves index
	// reordering along the concerned dimension; note that we do not have to traverse part the foremost
	// LPS where the index has been reordered.
	while (array != NULL) {
		if (array->isPartitionedAlongDimension(dimensionNo)) {
			partConfigsQueue.push_back(strdup(parentArrows.str().c_str()));
			parentStructsQueue.push_back(array);
		}
		// we do not have to traverse past one partitioning LPS up to the foremost reorder point
		if (array->isPartitionedAlongDimension(dimensionNo) 
			&& !array->isDimensionReordered(dimensionNo, rootSpace)) break;

		parentArrows << "->parent";
		array = (ArrayDataStructure*) array->getSource();
	}
	
	// create a local scope for the transformation process
	stream << indent.str() << "{ // scope start for index retransformation\n";
	
	// assign the transformed index to the default local variable created for index transform and 
	// retransform
	stream << indent.str() << "xformIndex = " << xformIndexVar.str() << stmtSeparator;

	// iterate over the queue and apply reverse transformation at each point to get final value for the
	// index variable
	while (!partConfigsQueue.empty()) {
		const char *pointer = partConfigsQueue.front();
		ArrayDataStructure *parentArray = parentStructsQueue.front();
		
		// get the part config reference to make it available for reverse index transformation 
		stream << indent.str() << "partConfig = *" << partConfigVar.str();
		stream << pointer << stmtSeparator;
		// if the dimension has been reordered in this LPS then assign the value of the reverse 
		// transformation to current value of xformIndex
		if (parentArray->isDimensionLocallyReordered(dimensionNo)) {
			stream << indent.str() << "xformIndex = ";
			stream << parentArray->getReverseXformExpr(dimensionNo, "xformIndex");
			stream << stmtSeparator;

		// Otherwise do an index adjustment as mixing of reordering and order preserving partition
		// functions may result in lost of the partition beginnings at the points where reordering
		// takes place. TODO in the future we have to develop a different mechanism for reordering
		// partition functions' LPU configuration that will save us from index adjustment on order
		// preserving partition functions.
		} else {
			stream << indent.str() << "xformIndex = partConfig.adjustIndex(xformIndex)";
			stream << stmtSeparator;
		}	
		
		partConfigsQueue.pop_front();
		parentStructsQueue.pop_front();
	}

	// finally assign the value of the reverse transformed index to the intended variable
	stream << indent.str() << indexVar << " = xformIndex" << stmtSeparator;

	// end the local scope
	stream << indent.str() << "} // scope end for index retransformation\n";
}

//----------------------------------------------- Subpartition Range --------------------------------------------------/

SubpartitionRangeExpr::SubpartitionRangeExpr(char s, yyltype loc) : Expr(loc) {
	spaceId = s;
}

void SubpartitionRangeExpr::PrintChildren(int indentLevel) {
	printf("Space %c", spaceId);
}

