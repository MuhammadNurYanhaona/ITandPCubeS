#include "../utils/list.h"
#include "ast.h"
#include "ast_stmt.h"
#include "ast_expr.h"
#include "ast_def.h"
#include "ast_task.h"
#include "location.h"
#include "../semantics/scope.h"
#include "../semantics/symbol.h"
#include "../semantics/task_space.h"
#include "../utils/hashtable.h"
#include "../utils/code_constant.h"
#include "errors.h"
#include "../static-analysis/loop_index.h"
#include "../static-analysis/extern_config.h"
#include "../static-analysis/task_global.h"
#include "../codegen/vectorization.h"
#include "../codegen/name_transformer.h"

#include <iostream>
#include <sstream>
#include <cstdlib>

//-------------------------------------------------------------- Statement -------------------------------------------------------------/

void Stmt::mergeAccessedVariables(Hashtable<VariableAccess*> *first, 
			Hashtable<VariableAccess*> *second) {
        if (second == NULL) return;
        Iterator<VariableAccess*> iter = second->GetIterator();
        VariableAccess *accessLog;
        while ((accessLog = iter.GetNextValue()) != NULL) {
                if (first->Lookup(accessLog->getName()) == NULL) {
			first->Enter(accessLog->getName(), new VariableAccess(accessLog->getName()), true);
		}
               	first->Lookup(accessLog->getName())->mergeAccessInfo(accessLog);
        }
}

//------------------------------------------------------------ Loop Statement ----------------------------------------------------------/

LoopStmt *LoopStmt::currentLoop = NULL;

LoopStmt::LoopStmt() : Stmt() { 
	scope = NULL; 
	reductionLoop = false;
	reduction = NULL; 
}

LoopStmt::LoopStmt(yyltype loc) : Stmt(loc) { 
	scope == NULL; 
	reductionLoop = false;
	reduction = NULL; 
}

void LoopStmt::setIndexScope(IndexScope *indexScope) { this->indexScope = indexScope; }

IndexScope *LoopStmt::getIndexScope() { return indexScope; }

void LoopStmt::setReductionExpr(ReductionExpr *reduction) { this->reduction = reduction; }

void LoopStmt::checkSemantics(Scope *excutionScope, bool ignoreTypeFailures) {
	if (reductionLoop) {
		StmtBlock *stmtBlock = (StmtBlock*) body;
		List<Stmt*> *stmtList = stmtBlock->getStmtList();
		if (stmtList->NumElements() != 1) {
			ReportError::CouplingOfReductionWithOtherExpr(GetLocation(), ignoreTypeFailures);
		} else {
			AssignmentExpr *assignment = dynamic_cast<AssignmentExpr*>(stmtList->Nth(0));
			if (assignment == NULL) {
				ReportError::CouplingOfReductionWithOtherExpr(GetLocation(), 
						ignoreTypeFailures);
			}
		}
	}
}

void LoopStmt::retrieveExternHeaderAndLibraries(IncludesAndLinksMap *includesAndLinksMap) {
        body->retrieveExternHeaderAndLibraries(includesAndLinksMap);
}

void LoopStmt::extractReductionInfo(List<ReductionMetadata*> *infoSet,
                PartitionHierarchy *lpsHierarchy, Space *executingLps) {
        body->extractReductionInfo(infoSet, lpsHierarchy, executingLps);
}

void LoopStmt::declareVariablesInScope(std::ostringstream &stream, int indentLevel) { 
	scope->declareVariables(stream, indentLevel);	
}

void LoopStmt::generateIndexLoops(std::ostringstream &stream, int indentLevel, 
			Space *space, Stmt *body, List<LogicalExpr*> *indexRestrictions) {
	
	IndexScope::currentScope->enterScope(indexScope);

	// check if an attempt can be made to vectorize the index loop
	bool vectorizable = false;
	PLoopStmt *parallelLoop = dynamic_cast<PLoopStmt*>(this);
	if (parallelLoop != NULL) {
		vectorizable = parallelLoop->isInnermostLoop();
	}

	// check if the loop is a reduction loop; if it is then do initialization for reduction
	if (reductionLoop) initializeReductionLoop(stream, indentLevel, space); 

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
	std::ostringstream loopHeaders;
	for (int i = 0; i < associateList->NumElements(); i++) {
		
		IndexArrayAssociation *association = associateList->Nth(i);
		const char *index = association->getIndex();
		const char *arrayName = association->getArray();
		int dimensionNo = association->getDimensionNo();
		int newIndent = indentLevel + indentIncrease;
		
		std::ostringstream indent;
		for (int j = 0; j < newIndent; j++) indent << '\t';
		// create a scope for the for loop corresponding to this association
		loopHeaders << indent.str();
		loopHeaders << "{// scope entrance for parallel loop on index " << index << "\n";
		// check if the index is a single entry
		bool forbidden = false;
		ArrayDataStructure *array = (ArrayDataStructure*) space->getLocalStructure(arrayName);
		// Dimension No starts from 1 in Data Structures
		if (array->isSingleEntryInDimension(dimensionNo + 1)) {
			forbiddenIndexes->Append(index);
			forbidden = true;
			// declare the initialized index variable
			loopHeaders << indent.str() << "int " << association->getIndex() << " = ";
			ntransform::NameTransformer *transformer = ntransform::NameTransformer::transformer;
			loopHeaders << transformer->getTransformedName(arrayName, true, true);
			loopHeaders << '[' << dimensionNo << "].range.min;\n"; 
		}
		
		// check any additional restrictions been provided that can be used as restricting conditions to 
		// further limit the iteration range from original partition dimension's minimum to maximum value
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
			loopHeaders << indent.str() << "int " << index << ";\n"; 
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

			rangeExpr->generateLoopForRangeExpr(loopHeaders, newIndent, 
					space, strdup(restrictStream.str().c_str()));
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
				for (int in = 0; in < newIndent; in++) loopHeaders << '\t';
				loopHeaders << "if (!(";
				applicableRestrictions->Nth(k)->translate(loopHeaders, newIndent, 0, space);
				loopHeaders << ")) continue;\n";
			}
		}

		// generate auxiliary code for multi to unidimensional array indexing transformations
		// for all array accesses that use this index
		List<IndexArrayAssociation*> *list = indexScope->getAssociationsForIndex(index);
		list = IndexArrayAssociation::filterList(list);
		for (int j = 0; j < list->NumElements(); j++) {
			IndexArrayAssociation *otherAssoc = list->Nth(j);
			otherAssoc->generateTransform(loopHeaders, newIndent, space);
		}
	}

	// If the loops are flagged as vectorizable then try to vectorize them by converting the headers to equivalent
	// loop headers but with embedded vectorization instructions. Otherwise, just write the loop headers into the
	// output stream.
	if (vectorizable) {
		const char *oldHeaders = strdup(loopHeaders.str().c_str());
		const char *newHeaders = vectorizeLoops(oldHeaders, indentLevel);
		stream << newHeaders;
	} else {
		stream << loopHeaders.str();
	}

	// translate the body of the for loop if it is not a reduction loop; otherwise do reduce iteration
	if (!reductionLoop) {
		body->generateCode(stream, indentLevel + indentIncrease, space);
	} else {
		performReduction(stream, indentLevel + indentIncrease, space);
	}

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

	// if this is a reduction loop then do any finalization needed for the reduction
	if (reductionLoop) finalizeReductionLoop(stream, indentLevel, space);

	IndexScope::currentScope->goBackToOldScope();
}

List<LogicalExpr*> *LoopStmt::getApplicableExprs(Hashtable<const char*> *indexesInvisible, 
                        List<LogicalExpr*> *currentExprList, 
                        List<LogicalExpr*> *remainingExprList) {

	List<LogicalExpr*> *includedExprList = new List<LogicalExpr*>;
	for (int i = 0; i < currentExprList->NumElements(); i++) {
		LogicalExpr *expr = currentExprList->Nth(i);
		List<FieldAccess*> *fieldAccessesInExpr = expr->getTerminalFieldAccesses();
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

void LoopStmt::initializeReductionLoop(std::ostringstream &stream, int indentLevel, Space *space) {
	std::ostringstream indent;
	for (int i = 0; i < indentLevel; i++) indent << '\t';
	stream << indent.str() << "{ //scope starts for reduction\n";
	reduction->setupForReduction(stream, indentLevel);
}

void LoopStmt::performReduction(std::ostringstream &stream, int indentLevel, Space *space) {
	reduction->generateCode(stream, indentLevel, space);
}

void LoopStmt::finalizeReductionLoop(std::ostringstream &stream, int indentLevel, Space *space) {
	reduction->finalizeReduction(stream, indentLevel);
	body->generateCode(stream, indentLevel, space);
	std::ostringstream indent;
	for (int i = 0; i < indentLevel; i++) indent << '\t';
	stream << indent.str() << "} //scope ends for reduction\n";
}

//------------------------------------------------------------ Statement Block ---------------------------------------------------------/

StmtBlock::StmtBlock(List<Stmt*> *s) : Stmt() {
	Assert(s != NULL);
	stmts = s;
	for (int i = 0; i < stmts->NumElements(); i++) {
		stmts->Nth(i)->SetParent(this);
	}	
}
    	
void StmtBlock::PrintChildren(int indentLevel) {
	stmts->PrintAll(indentLevel + 1);
}

void StmtBlock::checkSemantics(Scope *executionScope, bool ignoreTypeFailures) {
	for (int i = 0; i < stmts->NumElements(); i++) {
		Stmt *stmt = stmts->Nth(i);
		stmt->checkSemantics(executionScope, ignoreTypeFailures);
	}
}

void StmtBlock::performTypeInference(Scope *executionScope) {
	for (int i = 0; i < stmts->NumElements(); i++) {
		Stmt *stmt = stmts->Nth(i);
		stmt->performTypeInference(executionScope);
	}
}

Hashtable<VariableAccess*> *StmtBlock::getAccessedGlobalVariables(TaskGlobalReferences *globalReferences) {
	Hashtable<VariableAccess*> *table = Stmt::getAccessedGlobalVariables(NULL);
	for (int i = 0; i < stmts->NumElements(); i++) {
		Stmt *stmt = stmts->Nth(i);
		mergeAccessedVariables(table, stmt->getAccessedGlobalVariables(globalReferences));
	}
	return table;
}

void StmtBlock::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {
	for (int i = 0; i < stmts->NumElements(); i++) {
		Stmt *stmt = stmts->Nth(i);
		stmt->generateCode(stream, indentLevel, space);
	}	
}

void StmtBlock::analyseEpochDependencies(Space *space) {
	for (int i = 0; i < stmts->NumElements(); i++) {
		Stmt *stmt = stmts->Nth(i);
		stmt->analyseEpochDependencies(space);
	}	
}

void StmtBlock::retrieveExternHeaderAndLibraries(IncludesAndLinksMap *includesAndLinksMap) {
        for (int i = 0; i < stmts->NumElements(); i++) {
                Stmt *stmt = stmts->Nth(i);
                stmt->retrieveExternHeaderAndLibraries(includesAndLinksMap);
        }
}

void StmtBlock::extractReductionInfo(List<ReductionMetadata*> *infoSet,
                PartitionHierarchy *lpsHierarchy,
                Space *executingLps) {
        for (int i = 0; i < stmts->NumElements(); i++) {
                Stmt *stmt = stmts->Nth(i);
                stmt->extractReductionInfo(infoSet, lpsHierarchy, executingLps);
        }
}

bool StmtBlock::flagInnermostParallelForLoops() {
	bool innerLoopFound = false;
	for (int i = 0; i < stmts->NumElements(); i++) {
		Stmt *stmt = stmts->Nth(i);
		innerLoopFound = innerLoopFound || stmt->flagInnermostParallelForLoops();
	}
	return innerLoopFound;
}
	
//-------------------------------------------------------- Conditional Statement -------------------------------------------------------/

ConditionalStmt::ConditionalStmt(Expr *c, Stmt *s, yyltype loc) : Stmt(loc) {
	Assert(s != NULL);
	condition = c;
	if (condition != NULL) {
		condition->SetParent(this);
	}
	stmt = s;
	stmt->SetParent(this);
}

void ConditionalStmt::PrintChildren(int indentLevel) {
	if (condition != NULL) condition->Print(indentLevel, "(If) ");
	stmt->Print(indentLevel);
}

void ConditionalStmt::checkSemantics(Scope *executionScope, bool ignoreTypeFailures) {
	if (condition != NULL) condition->resolveType(executionScope, ignoreTypeFailures);
	stmt->checkSemantics(executionScope, ignoreTypeFailures);	
}

void ConditionalStmt::performTypeInference(Scope *executionScope) {
	if (condition != NULL) {
		condition->inferType(executionScope, Type::boolType);
	}
	stmt->performTypeInference(executionScope);	
}

Hashtable<VariableAccess*> *ConditionalStmt::getAccessedGlobalVariables(TaskGlobalReferences *globalReferences) {
	Hashtable<VariableAccess*> *table = stmt->getAccessedGlobalVariables(globalReferences);
	if (condition != NULL) mergeAccessedVariables(table,
					condition->getAccessedGlobalVariables(globalReferences));
	return table;
}

void ConditionalStmt::generateCode(std::ostringstream &stream, int indentLevel, bool first, Space *space) {
	if (first) {
		for (int i = 0; i < indentLevel; i++) stream << '\t';	
		stream << "if (";
		if (condition != NULL) {
			condition->translate(stream, indentLevel, 0, space);
		} else {
			stream << "true";
		}
		stream << ") {\n";
		stmt->generateCode(stream, indentLevel + 1, space);	
		for (int i = 0; i < indentLevel; i++) stream << '\t';	
		stream << "}";
	} else {
		if (condition != NULL) {
			stream << " else if (";
			condition->translate(stream, indentLevel, 0, space);
			stream << ") {\n";
		} else {
			stream << " else {\n";
		}
		stmt->generateCode(stream, indentLevel + 1, space);	
		for (int i = 0; i < indentLevel; i++) stream << '\t';	
		stream << "}";
	}
}

void ConditionalStmt::analyseEpochDependencies(Space *space) {
	if (condition != NULL) {
		condition->setEpochVersions(space, 0);
	}
	stmt->analyseEpochDependencies(space);
}

void ConditionalStmt::retrieveExternHeaderAndLibraries(IncludesAndLinksMap *includesAndLinksMap) {
        stmt->retrieveExternHeaderAndLibraries(includesAndLinksMap);
}

void ConditionalStmt::extractReductionInfo(List<ReductionMetadata*> *infoSet,
                PartitionHierarchy *lpsHierarchy,
                Space *executingLps) {
        stmt->extractReductionInfo(infoSet, lpsHierarchy, executingLps);
}

IfStmt::IfStmt(List<ConditionalStmt*> *ib, yyltype loc) : Stmt(loc) {
	Assert(ib != NULL);
	ifBlocks = ib;
	for (int i = 0; i < ifBlocks->NumElements(); i++) {
		ifBlocks->Nth(i)->SetParent(this);
	}
}

void IfStmt::PrintChildren(int indentLevel) {
	ifBlocks->PrintAll(indentLevel + 1);
}

void IfStmt::checkSemantics(Scope *executionScope, bool ignoreTypeFailures) {
	for (int i = 0; i < ifBlocks->NumElements(); i++) {
		ConditionalStmt *stmt = ifBlocks->Nth(i);
		stmt->checkSemantics(executionScope, ignoreTypeFailures);
	}
}

void IfStmt::performTypeInference(Scope *executionScope) {
	for (int i = 0; i < ifBlocks->NumElements(); i++) {
		ConditionalStmt *stmt = ifBlocks->Nth(i);
		stmt->performTypeInference(executionScope);
	}
}

Hashtable<VariableAccess*> *IfStmt::getAccessedGlobalVariables(TaskGlobalReferences *globalReferences) {
	Hashtable<VariableAccess*> *table = Stmt::getAccessedGlobalVariables(NULL);
	for (int i = 0; i < ifBlocks->NumElements(); i++) {
		ConditionalStmt *stmt = ifBlocks->Nth(i);
		mergeAccessedVariables(table, stmt->getAccessedGlobalVariables(globalReferences));
	}
	return table;
}

void IfStmt::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {
	for (int i = 0; i < ifBlocks->NumElements(); i++) {
		ConditionalStmt *stmt = ifBlocks->Nth(i);
		stmt->generateCode(stream, indentLevel, i == 0, space);
	}
	stream << '\n';
}

void IfStmt::analyseEpochDependencies(Space *space) {
	for (int i = 0; i < ifBlocks->NumElements(); i++) {
		ConditionalStmt *stmt = ifBlocks->Nth(i);
		stmt->analyseEpochDependencies(space);
	}
}

void IfStmt::retrieveExternHeaderAndLibraries(IncludesAndLinksMap *includesAndLinksMap) {
        for (int i = 0; i < ifBlocks->NumElements(); i++) {
                ConditionalStmt *stmt = ifBlocks->Nth(i);
                stmt->retrieveExternHeaderAndLibraries(includesAndLinksMap);
        }
}

void IfStmt::extractReductionInfo(List<ReductionMetadata*> *infoSet,
                PartitionHierarchy *lpsHierarchy,
                Space *executingLps) {
        for (int i = 0; i < ifBlocks->NumElements(); i++) {
                ConditionalStmt *stmt = ifBlocks->Nth(i);
                stmt->extractReductionInfo(infoSet, lpsHierarchy, executingLps);
        }
}

bool IfStmt::flagInnermostParallelForLoops() {
	bool innerLoopFound = false;
	for (int i = 0; i < ifBlocks->NumElements(); i++) {
		ConditionalStmt *stmt = ifBlocks->Nth(i);
		innerLoopFound = innerLoopFound || stmt->flagInnermostParallelForLoops();
	}
	return innerLoopFound;
}

//------------------------------------------------------------ Parallel Loop -----------------------------------------------------------/

IndexRangeCondition::IndexRangeCondition(List<Identifier*> *i, Identifier *c, 
		int dim, Expr *rs, yyltype loc) : Node(loc) {
	Assert(i != NULL && c != NULL);
	indexes = i;
	for (int j = 0; j < indexes->NumElements(); j++) {
		indexes->Nth(j)->SetParent(this);
	}
	collection = c;
	collection->SetParent(this);
	restrictions = rs;
	if (restrictions != NULL) {
		restrictions->SetParent(this);
	}
	this->dimensionNo = dim - 1;
}

void IndexRangeCondition::PrintChildren(int indentLevel) {
	indexes->PrintAll(indentLevel + 1, "(Index) ");
	collection->Print(indentLevel + 1, "(Array/List) ");
	if (restrictions != NULL) restrictions->Print(indentLevel + 1, "(Restrictions) ");
}

void IndexRangeCondition::resolveTypes(Scope *executionScope, bool ignoreTypeFailures) {

	for (int i = 0; i < indexes->NumElements(); i++) {
		Identifier *ind = indexes->Nth(i);
		const char* indexName = ind->getName();
		if (executionScope->lookup(indexName) != NULL) {
			ReportError::ConflictingDefinition(ind, ignoreTypeFailures);
		} else {
			VariableDef *variable = new VariableDef(ind, Type::intType);
			executionScope->insert_symbol(new VariableSymbol(variable));
		}
	}

	Symbol *colSymbol = executionScope->lookup(collection->getName());
	if (colSymbol == NULL) {
		ReportError::UndefinedSymbol(collection, ignoreTypeFailures);
	} else {
		VariableSymbol *varSym = (VariableSymbol*) colSymbol;
		Type *varType = varSym->getType();
		ArrayType *arrayType = dynamic_cast<ArrayType*>(varType);
		if (arrayType == NULL) {
			ReportError::NonArrayInIndexedIteration(collection, varType, ignoreTypeFailures);
		}
	}

	if (restrictions != NULL) {
		restrictions->resolveType(executionScope, ignoreTypeFailures);
	}
}

void IndexRangeCondition::inferTypes(Scope *executionScope) {
	if (restrictions != NULL) {
		restrictions->inferType(executionScope, Type::boolType);
	}
}

Hashtable<VariableAccess*> *IndexRangeCondition::getAccessedGlobalVariables(
		TaskGlobalReferences *globalReferences) {
	Hashtable<VariableAccess*> *table = new Hashtable<VariableAccess*>;
	if (globalReferences->doesReferToGlobal(collection->getName())) {
		const char *referenceName = collection->getName();
		const char *globalVar = globalReferences->getGlobalRoot(referenceName)->getName();
		VariableAccess *accessLog = new VariableAccess(globalVar);
		accessLog->markMetadataAccess();
		accessLog->getMetadataAccessFlags()->flagAsRead();
		table->Enter(globalVar, accessLog, true);
	}
	if (restrictions != NULL) {
		Hashtable<VariableAccess*> *rTable = 
				restrictions->getAccessedGlobalVariables(globalReferences);
		Iterator<VariableAccess*> iter = rTable->GetIterator();
		VariableAccess *accessLog;
		while ((accessLog = iter.GetNextValue()) != NULL) {
			if (table->Lookup(accessLog->getName()) != NULL) {
				table->Lookup(accessLog->getName())->mergeAccessInfo(accessLog);
			} else table->Enter(accessLog->getName(), accessLog, true);
		}
	}
	return table;
}

void IndexRangeCondition::putIndexesInIndexScope() {
	IndexScope *indexScope = IndexScope::currentScope;
	for (int i = 0; i < indexes->NumElements(); i++) {
		Identifier *index = indexes->Nth(i);
		const char *indexName = index->getName();
		const char *arrayName = collection->getName();
		indexScope->initiateAssociationList(indexName);
		indexScope->setPreferredArrayForIndex(indexName, arrayName);
		if (dimensionNo >= 0) {
			List<IndexArrayAssociation*> *list = indexScope->getAssociationsForIndex(indexName);
			list->Append(new IndexArrayAssociation(indexName, arrayName, dimensionNo));
		}
	}
}

void IndexRangeCondition::validateIndexAssociations(Scope *scope, bool ignoreFailure) {
	
	const char *collectionName = collection->getName();
	VariableSymbol *symbol = (VariableSymbol*) scope->lookup(collectionName);
	if (symbol == NULL) return;
	ArrayType *array = dynamic_cast<ArrayType*>(symbol->getType());
	if (array == NULL) return;
	int dimensions = array->getDimensions();
	if (dimensions == 1) return;
	
	for (int i = 0; i < indexes->NumElements(); i++) {
		Identifier *index = indexes->Nth(i);
		List<IndexArrayAssociation*> *associationList 
				= IndexScope::currentScope->getAssociationsForIndex(index->getName());
		bool mappingKnown = false;
		if (associationList != NULL) {
			for (int j = 0; j < associationList->NumElements(); j++) {
				IndexArrayAssociation *association = associationList->Nth(j);
				if (strcmp(association->getArray(), collectionName) == 0) {
					mappingKnown = true;
					int dimensionNo = association->getDimensionNo();
					break;
				}
			}
		}
		if (!mappingKnown) {
			ReportError::UnknownIndexToArrayAssociation(index, collection, ignoreFailure);
		}
	}
}

void IndexRangeCondition::analyseEpochDependencies(Space *space) {
	if (restrictions != NULL) {
		restrictions->setEpochVersions(space, 0);
	}
}

LogicalExpr *IndexRangeCondition::getRestrictions() { 
	if (restrictions == NULL) return NULL;
	return dynamic_cast<LogicalExpr*>(restrictions); 
}

PLoopStmt::PLoopStmt(List<IndexRangeCondition*> *rc, Stmt *b, yyltype loc) : LoopStmt(loc) {
	Assert(rc != NULL && b != NULL);
	rangeConditions = rc;
	for (int i = 0; i < rangeConditions->NumElements(); i++) {
		rangeConditions->Nth(i)->SetParent(this);
	}
	body = b;
	body->SetParent(this);
	innermostLoop = false;
}

void PLoopStmt::PrintChildren(int indentLevel) {
	rangeConditions->PrintAll(indentLevel + 1);
	body->Print(indentLevel + 1);
}

void PLoopStmt::performTypeInference(Scope *executionScope) {
	
	this->previousLoop = LoopStmt::currentLoop;
	LoopStmt::currentLoop = this;

	Scope *loopScope = executionScope->enter_scope(this->scope);
	for (int i = 0; i < rangeConditions->NumElements(); i++) {
		IndexRangeCondition *cond = rangeConditions->Nth(i);
		cond->inferTypes(loopScope);
	}
	body->performTypeInference(loopScope);

	LoopStmt::currentLoop = this->previousLoop;
	this->previousLoop = NULL;
}

void PLoopStmt::checkSemantics(Scope *executionScope, bool ignoreTypeFailures) {
	
	this->previousLoop = LoopStmt::currentLoop;
	LoopStmt::currentLoop = this;
	
	Scope *loopScope = executionScope->enter_scope(new Scope(StatementBlockScope));
	
	IndexScope::currentScope->deriveNewScope();
	for (int i = 0; i < rangeConditions->NumElements(); i++) {
		IndexRangeCondition *cond = rangeConditions->Nth(i);
		cond->putIndexesInIndexScope();
	}

	for (int i = 0; i < rangeConditions->NumElements(); i++) {
		IndexRangeCondition *cond = rangeConditions->Nth(i);
		cond->resolveTypes(loopScope, ignoreTypeFailures);
	}
	body->checkSemantics(loopScope, ignoreTypeFailures);

	for (int i = 0; i < rangeConditions->NumElements(); i++) {
		IndexRangeCondition *cond = rangeConditions->Nth(i);
		cond->validateIndexAssociations(loopScope, ignoreTypeFailures);
	}

	this->indexScope = IndexScope::currentScope;
	IndexScope::currentScope->goBackToOldScope();

	loopScope->detach_from_parent();
	this->scope = loopScope;
	
	LoopStmt::currentLoop = this->previousLoop;
	this->previousLoop = NULL;
	LoopStmt::checkSemantics(loopScope, ignoreTypeFailures);
}

Hashtable<VariableAccess*> *PLoopStmt::getAccessedGlobalVariables(TaskGlobalReferences *globalReferences) {
	Hashtable<VariableAccess*> *table = body->getAccessedGlobalVariables(globalReferences);
	for (int i = 0; i < rangeConditions->NumElements(); i++) {
		IndexRangeCondition *cond = rangeConditions->Nth(i);
		mergeAccessedVariables(table, cond->getAccessedGlobalVariables(globalReferences));
	}
	return table;	
}

List<const char*> *PLoopStmt::getIndexNames() {
	List<const char*> *indexNames = new List<const char*>;
	for (int i = 0; i < rangeConditions->NumElements(); i++) {
		IndexRangeCondition *cond = rangeConditions->Nth(i);
		List<Identifier*> *indexes = cond->getIndexes();
		for (int j = 0 ; j < indexes->NumElements(); j++) {
			indexNames->Append(indexes->Nth(j)->getName());
		}
	}
	return indexNames;
}

void PLoopStmt::analyseEpochDependencies(Space *space) {
	LoopStmt::analyseEpochDependencies(space);
	for (int i = 0; i < rangeConditions->NumElements(); i++) {
		IndexRangeCondition *cond = rangeConditions->Nth(i);
		cond->analyseEpochDependencies(space);
	}
}

bool PLoopStmt::flagInnermostParallelForLoops() {
	bool innerLoopFound = LoopStmt::flagInnermostParallelForLoops();
	innermostLoop = !(innerLoopFound);
	return true;
}

void PLoopStmt::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {
	LoopStmt::generateIndexLoops(stream, indentLevel, space, body, getIndexRestrictions());
}

List<LogicalExpr*> *PLoopStmt::getIndexRestrictions() {
	List<LogicalExpr*> *restrictionList = new List<LogicalExpr*>;
	for (int i = 0; i < rangeConditions->NumElements(); i++) {
		IndexRangeCondition *cond = rangeConditions->Nth(i);
		LogicalExpr *restriction = cond->getRestrictions();
		if (restriction != NULL) {
			List<LogicalExpr*> *containedExprList = restriction->getANDBreakDown();
			for (int j = 0; j < containedExprList->NumElements(); j++)
			restrictionList->Append(containedExprList->Nth(j));
		}
	}
	return restrictionList;
}

//--------------------------------------------------------- Sequential For Loop --------------------------------------------------------/

SLoopAttribute::SLoopAttribute(Expr *range, Expr *step, Expr *restriction) {
        Assert(range != NULL);
        this->range = range;
        this->step = step;
        this->restriction = restriction;
}

SLoopStmt::SLoopStmt(Identifier *i, SLoopAttribute *attr, Stmt *b, yyltype loc) : LoopStmt(loc) {
        Assert(i != NULL && attr != NULL && b != NULL);
        id = i;
        id->SetParent(this);
        rangeExpr = attr->getRange();
        rangeExpr->SetParent(this);
        stepExpr = attr->getStep();
        if (stepExpr != NULL) {
                stepExpr->SetParent(this);
        }
        restriction = attr->getRestriction();
        if (restriction != NULL) {
                restriction->SetParent(this);
        }
        body = b;
        body->SetParent(this);
        isArrayIndexTraversal = false;
}
    	
void SLoopStmt::PrintChildren(int indentLevel) {
	id->Print(indentLevel + 1, "(Index) ");
	rangeExpr->Print(indentLevel + 1, "(Range) ");
	if (stepExpr != NULL) stepExpr->Print(indentLevel + 1, "(Step) ");
	if (restriction != NULL) restriction->Print(indentLevel + 1, "(Index Restriction) ");
	body->Print(indentLevel + 1);
}

void SLoopStmt::performTypeInference(Scope *executionScope) {
	
	this->previousLoop = LoopStmt::currentLoop;
	LoopStmt::currentLoop = this;
	
	Scope *loopScope = executionScope->enter_scope(this->scope);
	rangeExpr->inferType(loopScope, Type::rangeType);
	if (stepExpr != NULL) stepExpr->inferType(loopScope, Type::intType);
	if (restriction != NULL) restriction->inferType(loopScope, Type::boolType);
	body->performTypeInference(loopScope);
	
	LoopStmt::currentLoop = this->previousLoop;
	this->previousLoop = NULL;
}

void SLoopStmt::checkSemantics(Scope *executionScope, bool ignoreTypeFailures) {

	this->previousLoop = LoopStmt::currentLoop;
	LoopStmt::currentLoop = this;
	
	// create a loop scope for normal variables and an index scope for index variables and enter both
	Scope *loopScope = executionScope->enter_scope(new Scope(StatementBlockScope));
	IndexScope::currentScope->deriveNewScope();
	
	if (loopScope->lookup(id->getName()) != NULL) {
		ReportError::ConflictingDefinition(id, ignoreTypeFailures);
	} else {
		VariableSymbol *var = new VariableSymbol(new VariableDef(id, Type::intType));
		loopScope->insert_symbol(var);
	}

	// Try to find out if the range corresponding to a dimension of some global array. If it is so 
	// then create an entry in the index scope.
	const char *potentialArray = rangeExpr->getBaseVarName();
	Scope *taskScope = executionScope->get_nearest_scope(TaskScope);
	if (potentialArray != NULL && taskScope != NULL) {
		Symbol *symbol = taskScope->local_lookup(potentialArray);
		bool attemptResolve = false;
		if (symbol != NULL) {
			VariableSymbol *variable = dynamic_cast<VariableSymbol*>(symbol);
			if (variable != NULL) {
				Type *varType = variable->getType();
				if (dynamic_cast<ArrayType*>(varType) != NULL &&
						dynamic_cast<StaticArrayType*>(varType) == NULL) {
					attemptResolve = true;
				}
			}
		}
		// It seems finding out if an expression is a dimension access of a task global array is
		// a messy effort. The expression is expected  to look like array.dimension#No.range. So
		// there is a need to do a three level unfolding of expression. It would be nice if we 
		// could generalize this procedure somewhere. TODO may be worth attempting in the future.
		bool dimensionFound = false;
		int dimension = 0;
		if (attemptResolve) {
			FieldAccess *rangeField = dynamic_cast<FieldAccess*>(rangeExpr);
			if (rangeField != NULL) {
        			Expr *base = rangeField->getBase();	
        			FieldAccess *baseField = dynamic_cast<FieldAccess*>(base);
				if (baseField != NULL) {
					Expr *arrayExpr = baseField->getBase();
					FieldAccess *arrayAccess = dynamic_cast<FieldAccess*>(arrayExpr);
        				Identifier *field = baseField->getField();
        				DimensionIdentifier *dimensionId = 
							dynamic_cast<DimensionIdentifier*>(field);
					if (arrayAccess != NULL 
							&& arrayAccess->isTerminalField() 
							&& dimensionId != NULL) {
						dimensionFound = true;
						// this is a coversion between 1 based to 0 based indexing
						dimension = dimensionId->getDimensionNo() - 1;
					}
				}
			}
		}
		if (dimensionFound) {
			const char *indexName = id->getName();
			IndexScope::currentScope->initiateAssociationList(indexName);
			List<IndexArrayAssociation*> *list 
					= IndexScope::currentScope->getAssociationsForIndex(indexName);
			list->Append(new IndexArrayAssociation(indexName, potentialArray, dimension));
			isArrayIndexTraversal = true;
		}
	}

	rangeExpr->resolveType(loopScope, ignoreTypeFailures);
	if (stepExpr != NULL) stepExpr->resolveType(loopScope, ignoreTypeFailures);
	if (restriction != NULL) restriction->resolveType(loopScope, ignoreTypeFailures);

	body->checkSemantics(loopScope, ignoreTypeFailures);
	
	this->indexScope = IndexScope::currentScope;
	IndexScope::currentScope->goBackToOldScope();

	loopScope->detach_from_parent();
	this->scope = loopScope;
	
	LoopStmt::currentLoop = this->previousLoop;
	this->previousLoop = NULL;
	LoopStmt::checkSemantics(loopScope, ignoreTypeFailures);
}

Hashtable<VariableAccess*> *SLoopStmt::getAccessedGlobalVariables(TaskGlobalReferences *globalReferences) {
	
	Hashtable<VariableAccess*> *table = body->getAccessedGlobalVariables(globalReferences);
	mergeAccessedVariables(table, rangeExpr->getAccessedGlobalVariables(globalReferences));
	if (stepExpr != NULL) mergeAccessedVariables(table, 
			stepExpr->getAccessedGlobalVariables(globalReferences));
	if (restriction != NULL) mergeAccessedVariables(table, 
			restriction->getAccessedGlobalVariables(globalReferences));
	
	Iterator<VariableAccess*> iter = table->GetIterator();
	VariableAccess *accessLog;
	while ((accessLog = iter.GetNextValue()) != NULL) {
		if(accessLog->isContentAccessed())
                        accessLog->getContentAccessFlags()->flagAsRead();
                if (accessLog->isMetadataAccessed())
                        accessLog->getMetadataAccessFlags()->flagAsRead();
	}
	return table; 
}

List<const char*> *SLoopStmt::getIndexNames() { 
	List<const char*> *indexNames = new List<const char*>;
	indexNames->Append(id->getName());
	return indexNames; 
}

void SLoopStmt::analyseEpochDependencies(Space *space) {
	LoopStmt::analyseEpochDependencies(space);
	rangeExpr->setEpochVersions(space, 0);
	if (stepExpr != NULL) stepExpr->setEpochVersions(space, 0);
	if (restriction != NULL) restriction->setEpochVersions(space, 0);
}

void SLoopStmt::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {

	if(isArrayIndexTraversal) {
		List<LogicalExpr*> *indexRestrictions = NULL;
                if (restriction != NULL) {
                        indexRestrictions = new List<LogicalExpr*>;
                        indexRestrictions->Append((LogicalExpr*) restriction);
                }
                LoopStmt::generateIndexLoops(stream, indentLevel, space, body, indexRestrictions);
	} else {
		IndexScope::currentScope->enterScope(indexScope);

		// check if the loop is a reduction loop; if it is then do initialization for reduction
		if (reductionLoop) initializeReductionLoop(stream, indentLevel, space); 

		std::ostringstream indent;
		for (int i = 0; i < indentLevel; i++) indent << '\t';	
		// create a scope for loop
        	stream << indent.str() << "{ // scope entrance for sequential loop\n";
		// declares any variable created in the nested scope of this loop
		declareVariablesInScope(stream, indentLevel);
		// create a range expression representing the sequential loop	
		RangeExpr *range = new RangeExpr(id, rangeExpr, stepExpr, true, *id->GetLocation());
        	// translate the range expression into a for loop
        	std::ostringstream rangeLoop;
        	range->generateLoopForRangeExpr(rangeLoop, indentLevel, space);
        	stream << rangeLoop.str();
	
		// if there is an additional restriction that says what values within the range should be skipped
                // then apply the restriction here as a continue block inside the loop
                if (restriction != NULL) {
                        stream << indent.str() << '\t';
                        stream << "if (!(";
                        restriction->translate(stream, indentLevel + 1, 0, space);
                        stream << ")) continue;\n";
                }
	
		// translate the body of the for loop if it is not a reduction loop; otherwise do reduce iteration
		if (!reductionLoop) {
			body->generateCode(stream, indentLevel + 1, space);
		} else {
			performReduction(stream, indentLevel + 1, space);
		}
	
        	// close the range loop
        	stream << indent.str() << "}\n";
        	// exit the scope created for the loop 
        	stream << indent.str() << "} // scope exit for sequential loop\n";

		// if this is a reduction loop then do any finalization needed for the reduction
		if (reductionLoop) finalizeReductionLoop(stream, indentLevel, space);
	
		IndexScope::currentScope->goBackToOldScope();
	}
}

//-------------------------------------------------------------- While Loop ------------------------------------------------------------/

WhileStmt::WhileStmt(Expr *c, Stmt *b, yyltype loc) : Stmt(loc) {
	Assert(c != NULL && b != NULL);
	condition = c;
	condition->SetParent(this);
	body = b;
	body->SetParent(this);
}	
    	
void WhileStmt::PrintChildren(int indentLevel) {
	condition->Print(indentLevel + 1, "(Condition) ");
	body->Print(indentLevel + 1);
}

void WhileStmt::performTypeInference(Scope *executionScope) {
	condition->inferType(executionScope, Type::boolType);
	body->performTypeInference(executionScope);
}

void WhileStmt::checkSemantics(Scope *executionScope, bool ignoreTypeFailures) {
	condition->resolveType(executionScope, ignoreTypeFailures);
	body->checkSemantics(executionScope, ignoreTypeFailures);
}

Hashtable<VariableAccess*> *WhileStmt::getAccessedGlobalVariables(
		TaskGlobalReferences *globalReferences) {
	Hashtable<VariableAccess*> *table = condition->getAccessedGlobalVariables(globalReferences);
	mergeAccessedVariables(table, body->getAccessedGlobalVariables(globalReferences));
	return table;
}

void WhileStmt::analyseEpochDependencies(Space *space) {
	body->analyseEpochDependencies(space);
	condition->setEpochVersions(space, 0);
}

void WhileStmt::retrieveExternHeaderAndLibraries(IncludesAndLinksMap *includesAndLinksMap) {
        body->retrieveExternHeaderAndLibraries(includesAndLinksMap);
}

void WhileStmt::extractReductionInfo(List<ReductionMetadata*> *infoSet,
                PartitionHierarchy *lpsHierarchy,
                Space *executingLps) {
        body->extractReductionInfo(infoSet, lpsHierarchy, executingLps);
}

void WhileStmt::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {
	for (int i = 0; i < indentLevel; i++) stream << '\t';
	stream << "do {\n";
	body->generateCode(stream, indentLevel + 1, space);
	for (int i = 0; i < indentLevel; i++) stream << '\t';
	stream << "} while(";
	if (condition != NULL) {
		condition->translate(stream, indentLevel, 0, space);
	} else {
		stream << "true";
	}
	stream << ");\n";
}

//---------------------------------------------------------- External Code Block -------------------------------------------------------/

ExternCodeBlock::ExternCodeBlock(const char *language,
                        List<const char*> *headerIncludes,
                        List<const char*> *libraryLinks,
                        const char *codeBlock, yyltype loc) : Stmt(loc) {

        Assert(language != NULL && codeBlock != NULL);
        this->language = language;
        this->headerIncludes = headerIncludes;
        this->libraryLinks = libraryLinks;
        this->codeBlock = codeBlock;
}

void ExternCodeBlock::PrintChildren(int indentLevel) {
        std::ostringstream indent;
        for (int i = 0; i < indentLevel; i++) {
                indent << '\t';
        }
        std::cout << indent.str() << "Language: " << language << "\n";
        if (headerIncludes != NULL) {
                std::cout << indent.str() << "Included Headers:\n";
                for (int i = 0; i < headerIncludes->NumElements(); i++) {
                        std::cout << indent.str() << '\t' << headerIncludes->Nth(i) << "\n";
                }
        }
        if (libraryLinks != NULL) {
                std::cout << indent.str() << "Linked Libraries:\n";
                for (int i = 0; i < libraryLinks->NumElements(); i++) {
                        std::cout << indent.str() << '\t' << libraryLinks->Nth(i) << "\n";
                }
        }
        std::cout << indent.str() << "Code Block:" << codeBlock << "\n";
}

void ExternCodeBlock::retrieveExternHeaderAndLibraries(IncludesAndLinksMap *includesAndLinksMap) {
        includesAndLinksMap->addIncludesAndLinksForLanguage(language,
                        headerIncludes, libraryLinks);
}

// The current code generation process for external code blocks assumes that the code block is written
// in C or C++. So we can just expand the code blocks within the stream for the .cc file for the task.
// Later we have to change this strategy and expand external code blocks in separate files based on the
// underlying languages they have been written on. Then a function call type expansion semantics should
// be better. To elaborate, we will have the call being generated in our .cc file for the task and 
// expend the actual code block as a function in a separate file. That separate file can be compiled
// with the appropriate compiler for the language of the extern block. Afterwords, we will just link the
// object files as we do now.
void ExternCodeBlock::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {
        
	TaskDef *taskDef = TaskDef::currentTask;
        List<TaskGlobalScalar*> *globalScalars = TaskGlobalCalculator::calculateTaskGlobals(taskDef);
        Space *rootLps = space->getRoot();

        std::ostringstream indentStr;
        for (int i = 0; i < indentLevel; i++) indentStr << '\t';
        std::string earlyIndents = indentStr.str();
        indentStr <<  '\t';
        std::string indents = indentStr.str();

        // start a new scope for the external code block
        stream << '\n' << earlyIndents << "{ // starting scope for an external code block\n";

        // generate copies of scalar variables that matches the name of the source code
        if (globalScalars->NumElements() > 0) {
                stream << '\n' << indents << "// generating local variables for global scalars\n";
        }
        for (int i = 0; i < globalScalars->NumElements(); i++) {
                TaskGlobalScalar *scalar = globalScalars->Nth(i);
                Type *scalarType = scalar->getType();
                const char *varName = scalar->getName();
                stream << indents << scalarType->getCType() << " " << varName << " = ";
                if (scalar->isLocallyManageable()) {
                        stream << "threadLocals->" << varName << stmtSeparator;
                } else {
                        stream << "taskGlobals->" << varName << stmtSeparator;
                }
        }

	// We need to discuss before enabling this properties in the hybrid compiler; if the stage
	// holding the extern code block is executing inside the GPU then creating too many variables
	// for the extern code block may result in register spills within the GPU SMs.	
	/*--------------------------------------------------------------------------------------------- 
        // if the LPS is partitioned into LPU then extract the ID from the LPU and make it directly
        // accessible within the external block
        if (space->getDimensionCount() > 0) {
                stream << '\n' << indents << "// generating a local version of the LPU ID\n";
                int dimensions = space->getDimensionCount();
                stream << indents << "int lpuId[" << dimensions << "]" << stmtSeparator;
                for (int i = 0; i < dimensions; i++) {
                        stream << indents << "lpuId[" << i << "] = ";
                        stream << "lpu->lpuId[" << i << ']' << stmtSeparator;
                }
        }

        // generating local versions of all array dimension metadata
        List<const char*> *arrays = rootLps->getLocallyUsedArrayNames();
        stream << '\n' << indents << "// generating local variables for array dimension metadata\n";
        for (int i = 0; i < arrays->NumElements(); i++) {
                const char *arrayName = arrays->Nth(i);
                ArrayDataStructure *array = (ArrayDataStructure *) rootLps->getStructure(arrayName);
                int arrayDims = array->getDimensionality();
                stream << indents << "Dimension " << arrayName << "_dimension";
                stream << "[" << arrayDims << "]" << stmtSeparator;
                for (int j = 0; j < arrayDims; j++) {
                        stream << indents << arrayName << "_dimension[" << j << "] = arrayMetadata->";
                        stream << arrayName << "Dims[" << j << "]" << stmtSeparator;
                }
        } ------------------------------------------------------------------------------------------*/

        // jumping into the external code block within a further nested block
        stream << '\n' << indents << "{ // external code block starts\n";
        stream << codeBlock;
        stream << '\n' << indents << "} // external code block ends\n\n";

        // end the scope for the external code block
        stream << earlyIndents << "} // ending scope for the external code block\n\n";
}

//--------------------------------------------------------- Reduction Statement --------------------------------------------------------/

ReductionStmt::ReductionStmt(char s, Identifier *l, char *o, Expr *r, yyltype loc) : Stmt(loc) {

        Assert(l != NULL && r != NULL && o != NULL);

        spaceId = s;
        resultVar = l;
        resultVar->SetParent(this);

        if (strcmp(o, "sum") == 0) op = SUM;
        else if (strcmp(o, "product") == 0) op = PRODUCT;
        else if (strcmp(o, "max") == 0) op = MAX;
        else if (strcmp(o, "maxEntry") == 0) op = MAX_ENTRY;
        else if (strcmp(o, "min") == 0) op = MIN;
        else if (strcmp(o, "minEntry") == 0) op = MIN_ENTRY;
        else if (strcmp(o, "avg") == 0) op = AVG;
        else if (strcmp(o, "land") == 0) op = LAND;
        else if (strcmp(o, "lor") == 0) op = LOR;
        else if (strcmp(o, "band") == 0) op = BAND;
        else if (strcmp(o, "bor") == 0) op = BOR;
        else {
                std::cout << "Currently the compiler does not support user defined reduction functions";
                Assert(0 == 1);
        }

        right = r;
        right->SetParent(this);
}

void ReductionStmt::PrintChildren(int indentLevel) {
        resultVar->Print(indentLevel + 1);
        PrintLabel(indentLevel + 1, "Operator");
        switch (op) {
                case SUM: printf("Sum"); break;
                case PRODUCT: printf("Product"); break;
                case MAX: printf("Maximum"); break;
                case MIN: printf("Minimum"); break;
                case AVG: printf("Average"); break;
                case MIN_ENTRY: printf("Minimum Entry"); break;
                case MAX_ENTRY: printf("Maximum Entry"); break;
                case LOR: printf("Logical OR"); break;
                case LAND: printf("Logical AND"); break;
                case BOR: printf("Bitwise OR"); break;
                case BAND: printf("Bitwise AND"); break;
        }
        right->Print(indentLevel + 1);
}

void ReductionStmt::performTypeInference(Scope *executionScope) {

        if (op == MIN_ENTRY || op == MAX_ENTRY) {
                right->inferType(executionScope, Type::intType);
        } else if (op == LOR || op == LAND) {
                right->inferType(executionScope, Type::boolType);
        } else {
                Symbol *symbol = executionScope->lookup(resultVar->getName());
                VariableSymbol *varSym = dynamic_cast<VariableSymbol*>(symbol);
                if (symbol == NULL || varSym == NULL) {
                        Type *resultType = varSym->getType();
                        right->inferType(executionScope, resultType);
                }
        }
}

void ReductionStmt::checkSemantics(Scope *executionScope, bool ignoreTypeFailures) {

        Symbol *symbol = executionScope->lookup(resultVar->getName());
        VariableSymbol *varSym = dynamic_cast<VariableSymbol*>(symbol);
        if (symbol == NULL || varSym == NULL) {
                ReportError::UndefinedSymbol(resultVar, ignoreTypeFailures);
        } else {
                if (!varSym->isReduction()) {
                        ReportError::NotReductionType(resultVar, ignoreTypeFailures);
                }
        }
        right->checkSemantics(executionScope, ignoreTypeFailures);
}

Hashtable<VariableAccess*> *ReductionStmt::getAccessedGlobalVariables(TaskGlobalReferences *globalReferences) {

        Hashtable<VariableAccess*> *table = Stmt::getAccessedGlobalVariables(NULL);

        const char *resultName = resultVar->getName();
        VariableAccess *accessLog = new VariableAccess(resultName);
        accessLog->markContentAccess();
        accessLog->getContentAccessFlags()->flagAsReduced();
        table->Enter(resultName, accessLog, true);

        Hashtable<VariableAccess*> *rTable = right->getAccessedGlobalVariables(globalReferences);
        List<FieldAccess*> *rightFieldAccesses = right->getTerminalFieldAccesses();
        for (int i = 0; i < rightFieldAccesses->NumElements(); i++) {
                FieldAccess *field = rightFieldAccesses->Nth(i);
                const char *varName = field->getField()->getName();
                accessLog = rTable->Lookup(varName);

                // if the field is not a task-global variable then we can ignore it 
                if (accessLog == NULL) continue;

                Type *fieldType = field->getType();
                ArrayType *array = dynamic_cast<ArrayType*>(fieldType);

                // if the field is not an array then its access flags are already set properly
                if (array == NULL) continue;

                // if the content of the array has been accessed then it should be flagged as read
                if (accessLog->isContentAccessed()) {
                        accessLog->getContentAccessFlags()->flagAsRead();
                }
        }

        mergeAccessedVariables(table, rTable);
        return table;
}

void ReductionStmt::analyseEpochDependencies(Space *space) {
        right->analyseEpochDependencies(space);
}

void ReductionStmt::extractReductionInfo(List<ReductionMetadata*> *infoSet,
                PartitionHierarchy *lpsHierarchy,
                Space *executingLps) {

        Space *reductionRootLps = lpsHierarchy->getSpace(spaceId);
        ReductionMetadata *metadata = new ReductionMetadata(resultVar->getName(),
                        op, reductionRootLps, executingLps, GetLocation());
        infoSet->Append(metadata);
}

void ReductionStmt::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {

        const char *resultName = resultVar->getName();
        DataStructure *resultStruct = space->getStructure(resultName);
        Type *resultType = resultStruct->getType();

        // This naming strategy to find the appropriate property in the union holding reduction result is incomplete.
        // Currently this is sufficient as we do not have the unsigned primitive types yet that have a space in their
        // C type names. TODO we need to make change in the property naming convension when we will add those types
        // in IT.
        std::ostringstream resultPropertyStr;
        resultPropertyStr << "data." << resultType->getCType() << "Value";
        std::string resultProperty = resultPropertyStr.str();

        std::ostringstream indents;
        for (int i = 0; i < indentLevel; i++) indents << indent;

        if (op == SUM) {
                stream << indents.str() << resultName << "->" << resultProperty << " += ";
                right->translate(stream, indentLevel, 0, space);
                stream << stmtSeparator;
        } else if (op == PRODUCT) {
                stream << indents.str() << resultName << "->" << resultProperty << " *= ";
                right->translate(stream, indentLevel, 0, space);
                stream << stmtSeparator;
        } else if (op == MAX) {
                stream << indents.str() << "if (" << resultName << "->" << resultProperty;
                stream << " < ";
                right->translate(stream, indentLevel, 0, space);
                stream << ") {\n";
                stream << indents.str() << indent;
                stream << resultName << "->" << resultProperty << " = ";
                right->translate(stream, indentLevel, 0, space);
                stream << stmtSeparator;
                stream << indents.str() << "}\n";
        } else if (op == MIN) {
                stream << indents.str() << "if (" << resultName << "->" << resultProperty;
                stream << " > ";
                right->translate(stream, indentLevel, 0, space);
                stream << ") {\n";
                stream << indents.str() << indent;
                stream << resultName << "->" << resultProperty << " = ";
                right->translate(stream, indentLevel, 0, space);
                stream << stmtSeparator;
                stream << indents.str() << "}\n";
        } else if (op == LAND) {
                stream << indents.str() << resultName << "->" << resultProperty << " = ";
                stream << resultName << "->" << resultProperty << " && ";
                right->translate(stream, indentLevel, 0, space);
                stream << stmtSeparator;
        } else if (op == LOR) {
                stream << indents.str() << resultName << "->" << resultProperty << " = ";
                stream << resultName << "->" << resultProperty << " || ";
                right->translate(stream, indentLevel, 0, space);
                stream << stmtSeparator;
        } else if (op == BAND) {
                stream << indents.str() << resultName << "->" << resultProperty << " = ";
                stream << resultName << "->" << resultProperty << " & ";
                right->translate(stream, indentLevel, 0, space);
                stream << stmtSeparator;
        } else if (op == BOR) {
                stream << indents.str() << resultName << "->" << resultProperty << " = ";
                stream << resultName << "->" << resultProperty << " | ";
                right->translate(stream, indentLevel, 0, space);
                stream << stmtSeparator;
        } else {
                std::cout << "Average, Max-entry, and Min-entry reductions haven't been implemented yet";
                std::exit(EXIT_FAILURE);
        }
}

