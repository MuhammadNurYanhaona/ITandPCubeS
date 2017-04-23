#include "../../../utils/name_transformer.h"
#include "../../../../../../common-libs/utils/list.h"
#include "../../../../../../frontend/src/syntax/ast_stmt.h"
#include "../../../../../../frontend/src/syntax/ast_expr.h"
#include "../../../../../../frontend/src/semantics/task_space.h"
#include "../../../../../../frontend/src/semantics/loop_index.h"

#include <sstream>

void LoopStmt::declareVariablesInScope(std::ostringstream &stream, int indentLevel) { 
	scope->declareVariables(stream, indentLevel);	
}

void LoopStmt::generateIndexLoops(std::ostringstream &stream, int indentLevel, 
			Space *space, Stmt *body, 
			List<LogicalExpr*> *indexRestrictions) {
	
	IndexScope::currentScope->enterScope(indexScope);

	// create two helper lists to keep track of the index restrictions that remains to be examined as we
	// put different restrictions in appropriate index traversal loops
	List<LogicalExpr*> *allRestrictions = indexRestrictions;
	List<LogicalExpr*> *remainingRestrictions = new List<LogicalExpr*>;

	Space *rootSpace = space->getRoot();

	// create an array for indexes that are single entry in the execution space so that we do not have to
	// create loops for them
	List<const char*> *forbiddenIndexes = new List<const char*>;
	
	List<IndexArrayAssociation*> *associateList = indexScope->getAllPreferredAssociations();
	int indentIncrease = 0;
	for (int i = 0; i < associateList->NumElements(); i++) {
		
		IndexArrayAssociation *association = associateList->Nth(i);
		const char *index = association->getIndex();
		const char *arrayName = association->getArray();
		int dimensionNo = association->getDimensionNo();
		int newIndent = indentLevel + indentIncrease;
		
		std::ostringstream indent;
		for (int j = 0; j < newIndent; j++) indent << '\t';
		// create a scope for the for loop corresponding to this association
		stream << indent.str();
		stream << "{// scope entrance for parallel loop on index " << index << "\n";
		// check if the index is a single entry
		bool forbidden = false;
		ArrayDataStructure *array = (ArrayDataStructure*) space->getLocalStructure(arrayName);
		// Dimension No starts from 1 in Data Structures
		if (array->isSingleEntryInDimension(dimensionNo + 1)) {
			forbiddenIndexes->Append(index);
			forbidden = true;
			// declare the initialized index variable
			stream << indent.str() << "int " << association->getIndex() << " = ";
			ntransform::NameTransformer *transformer = ntransform::NameTransformer::transformer;
			stream << transformer->getTransformedName(arrayName, true, true);
			stream << '[' << dimensionNo << "].range.min;\n"; 
		}
		
		// check any additional restrictions been provided that can be used as restricting conditions to 
		// further limit iteration range from original partition dimension's minimum to maximum value
		List<LogicalExpr*> *applicableRestrictions = NULL;
		if (allRestrictions != NULL && allRestrictions->NumElements() > 0) {
			remainingRestrictions = new List<LogicalExpr*>;
			Hashtable<const char*> *invisibleIndexes = new Hashtable<const char*>;
			for (int j = i + 1; j < associateList->NumElements(); j++) {
				IndexArrayAssociation *otherAssoc = associateList->Nth(j);
				const char *otherIndex = otherAssoc->getIndex();
				invisibleIndexes->Enter(otherIndex, otherIndex, true);
			}
			applicableRestrictions = getApplicableExprs(invisibleIndexes, allRestrictions, 
					remainingRestrictions);
			allRestrictions = remainingRestrictions;
		}  	

		if (!forbidden) {
			// declare the uninitialized index variable
			stream << indent.str() << "int " << index << ";\n"; 
			// convert the index access to a range loop iteration and generate code for that
			DataStructure *structure = space->getLocalStructure(association->getArray());
			RangeExpr *rangeExpr = association->convertToRangeExpr(structure->getType());

			// if there are index restricting conditions applicable to this loop index then try to extract
			// those conditions that can be used as limit the boundaries of the loop iteration
			const char *rangeCond = rangeExpr->getRangeExpr(space);
			std::ostringstream restrictStream;
			if (applicableRestrictions != NULL) {
				applicableRestrictions = LogicalExpr::getIndexRestrictExpr(applicableRestrictions, 
						restrictStream, index, rangeCond, newIndent, space,
						array->isDimensionReordered(dimensionNo + 1, rootSpace),
						arrayName, dimensionNo + 1);
			}

			rangeExpr->generateLoopForRangeExpr(stream, newIndent, space, restrictStream.str().c_str());
			indentIncrease++;
			newIndent++;	
		}
		
		// Apply any restrictions applicable for escaping some loop iterations. Note that this process reuse
		// some of the expressions that may be already used to limit the index start and end boundaries. None-
		// theless we apply them here again as our boundary setting restrictions may be imprecise. 
		// TODO note that this implementation is assuming that iteration restrictions are only applicable for 
		// non-single entry loops. This assumption should not be made. Including restrictions checking for 
		// single-entry for loop would require some code refactoring that we are avoiding at this point to save 
		// time, but this is not hard. More generic and appropriate mechanism is to put the body of the loop 
		// inside a composite if statement covering all additional restriction. If we do that than the 
		// restrictions should work for both mode of loop traversal.
		if (!forbidden && applicableRestrictions != NULL && applicableRestrictions->NumElements() > 0) {
			for (int k = 0; k < applicableRestrictions->NumElements(); k++) {	
				for (int in = 0; in < newIndent; in++) stream << '\t';
				stream << "if (!(";
				applicableRestrictions->Nth(k)->translate(stream, newIndent, 0, space);
				stream << ")) continue;\n";
			}
		}

		// generate auxiliary code for multi to unidimensional array indexing transformations
		// for all array accesses that use this index
		List<IndexArrayAssociation*> *list = indexScope->getAssociationsForIndex(index);
		list = IndexArrayAssociation::filterList(list);
		for (int j = 0; j < list->NumElements(); j++) {
			IndexArrayAssociation *otherAssoc = list->Nth(j);
			otherAssoc->generateTransform(stream, newIndent, space);
		}

	}

	// translate the body of the for loop
	body->generateCode(stream, indentLevel + indentIncrease, space);

	// close the for loops and the scopes
	for (int i = associateList->NumElements() - 1; i >= 0; i--) {
		IndexArrayAssociation *association = associateList->Nth(i);
		const char *index = association->getIndex();
		bool permitted = true;
		for (int j = 0; j < forbiddenIndexes->NumElements(); j++) {
			if (strcmp(forbiddenIndexes->Nth(j), index) == 0) {
				permitted = false;
				break;
			}
		}
		int newIndent = indentLevel + indentIncrease;
		if (permitted) {
			indentIncrease--;	
			newIndent = indentLevel + indentIncrease;
			for (int i = 0; i < newIndent; i++) stream << '\t';
			stream << "}\n";
		} 
		for (int i = 0; i < newIndent; i++) stream << '\t';
		stream << "}// scope exit for parallel loop on index " << association->getIndex() << "\n"; 
	}	

	IndexScope::currentScope->goBackToOldScope();
}

List<LogicalExpr*> *LoopStmt::getApplicableExprs(Hashtable<const char*> *indexesInvisible, 
                        List<LogicalExpr*> *currentExprList, 
                        List<LogicalExpr*> *remainingExprList) {

	List<LogicalExpr*> *includedExprList = new List<LogicalExpr*>;
	for (int i = 0; i < currentExprList->NumElements(); i++) {
		LogicalExpr *expr = currentExprList->Nth(i);
		List<FieldAccess*> *fieldAccessesInExpr = new List<FieldAccess*>;
		expr->retrieveTerminalFieldAccesses(fieldAccessesInExpr);
		bool invisibleIndexUsed = false;
		for (int j = 0; j < fieldAccessesInExpr->NumElements(); j++) {
			const char *fieldName = fieldAccessesInExpr->Nth(j)->getField()->getName();
			if (indexesInvisible->Lookup(fieldName) != NULL) {
				invisibleIndexUsed = true;
				break;
			}
		}
		if (invisibleIndexUsed) remainingExprList->Append(expr);
		else includedExprList->Append(expr);	
	}
	
	return includedExprList;
}
