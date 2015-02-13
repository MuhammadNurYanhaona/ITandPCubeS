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

void RangeExpr::translate(std::ostringstream &stream, int indentLevel, int currentLineLength) {

	// get the appropriate back-end name for the index variable
	ntransform::NameTransformer *transformer = ntransform::NameTransformer::transformer;
        const char *indexVar = transformer->getTransformedName(index->getName(), false, false);
	// translate the range condition
	std::ostringstream rangeStr;
	range->translate(rangeStr, indentLevel, 0);
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

const char *RangeExpr::getIndexExpr() {
	ntransform::NameTransformer *transformer = ntransform::NameTransformer::transformer;
        return transformer->getTransformedName(index->getName(), false, false);
}

const char *RangeExpr::getRangeExpr() {
	std::ostringstream rangeStr;
	range->translate(rangeStr, 0, 0);
	return strdup(rangeStr.str().c_str());
}

const char *RangeExpr::getStepExpr() {
	std::ostringstream stepStr;
	if (step == NULL) stepStr << "1";
	else step->translate(stepStr, 0, 0);
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

void RangeExpr::generateLoopForRangeExpr(std::ostringstream &stream, int indentation, Space *space) {
	
	std::string stmtSeparator = ";\n";
	std::ostringstream indent;
	for (int i = 0; i < indentation; i++) indent << '\t';

	const char *indexVar = this->getIndexExpr();                
	const char *rangeCond = this->getRangeExpr();
        const char *stepCond = this->getStepExpr();

	// find out if the used index need some index transformation before been used inside        
	bool involveIndexXform = false;
        const char *baseArray = this->getBaseArrayForRange(space);
        if (baseArray != NULL) {
        int dimension = this->getDimensionForRange(space);
        ArrayDataStructure *array = (ArrayDataStructure*) space->getLocalStructure(baseArray);
        	if (array->isDimensionLocallyReordered(dimension)) {
        		involveIndexXform = true;
        	}
        }

        // create two new variables for setting appropriate loop  condition checking and index 
        // increment, and one variable to multiply index properly during looping
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

        // write the for loop corresponding to the repeat instruction
        stream << indent.str() << "for (" << indexVarUsed.str() << " = " << rangeCond << ".min; \n";
        stream << indent.str() << "\t\tindexMultiplier * " << indexVarUsed.str() << " <= iterationBound; \n";
        stream << indent.str() << "\t\t" << indexVarUsed.str() << " += indexIncrement) {\n";
        if (involveIndexXform) {
        	stream << indent.str() << "\t// this should be an index transformation statement in future\n";
                stream << indent.str() << "\t" << indexVar << " = " << indexVarUsed.str();
                stream << stmtSeparator;
        }
	
	delete indexVar;
        delete rangeCond;
        delete stepCond;
}

void RangeExpr::translateArrayRangeExprCheck(std::ostringstream &stream, int indentLevel, Space *space) {
	
	std::string stmtSeparator = ";\n";
	std::ostringstream indent;
	for (int i = 0; i < indentLevel; i++) indent << '\t';
	indent << "\t\t";

	ntransform::NameTransformer *transformer = ntransform::NameTransformer::transformer;
        const char *indexVar = transformer->getTransformedName(index->getName(), false, false);
	std::string lpuPrefix = transformer->getLpuPrefix();
	
	const char *arrayName = getBaseArrayForRange(space);
	int dimensionNo = getDimensionForRange(space);

	std::ostringstream partConfigVar;
	partConfigVar << lpuPrefix << arrayName << "PartDims[" << dimensionNo - 1 << "]";

	// get the Root Space reference to aid in determining the extend of reordering later
	Space *rootSpace = space->getRoot();
	
	// grab array configuration information for current LPS
	ArrayDataStructure *array = (ArrayDataStructure*) space->getLocalStructure(arrayName);

	std::ostringstream exprStream;

	// if the array is reordered anywhere in the partition hierarchy starting from the root LPS down to 
	// current LPS then we have to recursively do inclusion testing on higher LPSes. Note that this 
	// implementation can be improved by considering only those points of reordering, but we do not do
	// that in the prototype implementation.	
	if (array->isReordered(rootSpace)) {
		
		// create two stacks for traversing parent pointers
		// stacks are needed as inclusion check should be done from top to bottom in LPS hierarchy
		std::stack<const char*> partConfigsStack;
		std::stack<ArrayDataStructure*> parentStructsStack;

		// put current array reference in stack so that it will be automatically processed with its
		// sources in higher LPSes
		partConfigsStack.push("");
		parentStructsStack.push(array);

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
			// we don't have to traverse beyond one LPS up beyond the foremost reorder point
			if (!parentArray->isReordered(rootSpace)) break;
		}
	
		// a variable for tracking the number of inclusion checks has been made so far
		int clauseAdded = 0;

		// iterate over parent structure references and do index inclusion check in each in sequence
		ArrayDataStructure *lastArray = NULL;	 
		while (!partConfigsStack.empty()) {
			const char *pointerLinks = partConfigsStack.top();
			ArrayDataStructure *parentArray = parentStructsStack.top();
			if (clauseAdded > 0) {
				exprStream << std::endl << indent.str();
				exprStream << "&& (";
				// if the dimension has been reordered by previous LPS then we need to get
				// the transformed index before we can do inclusion check for current LPS	
				if (lastArray->isDimensionLocallyReordered(dimensionNo)) {
					exprStream << "xformIndex = ";
					exprStream << lastArray->getIndexXfromExpr(dimensionNo, "xformIndex");
					exprStream << ',' << std::endl << indent.str() << '\t';
				}
			} else {
				exprStream << "(xformIndex = " << indexVar;
				exprStream << ',' << std::endl << indent.str() << '\t';
			}
			exprStream << "partConfig = *" << partConfigVar.str() << pointerLinks;
			exprStream << ',' << std::endl << indent.str() << '\t';
			
			// if the dimension is reordered in current LPS then we need to add the specific
			// reorder expression applicable for the underlying partition function
			if (parentArray->isDimensionLocallyReordered(dimensionNo)) {
				exprStream << parentArray->getReorderedInclusionCheckExpr(
						dimensionNo, "xformIndex");
			// otherwise we can invoke the standard inline inclusion function on PartDimension
			// object
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
}

//----------------------------------------------- Subpartition Range --------------------------------------------------/

SubpartitionRangeExpr::SubpartitionRangeExpr(char s, yyltype loc) : Expr(loc) {
	spaceId = s;
}

void SubpartitionRangeExpr::PrintChildren(int indentLevel) {
	printf("Space %c", spaceId);
}

