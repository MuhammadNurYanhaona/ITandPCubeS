#include "../utils/list.h"
#include "ast.h"
#include "ast_stmt.h"
#include "ast_expr.h"
#include "ast_def.h"
#include "../semantics/symbol.h"
#include "../utils/hashtable.h"
#include "errors.h"
#include "../static-analysis/loop_index.h"

//--------------------------------------------- Statement -------------------------------------------/

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

//------------------------------------------ Statement Block ----------------------------------------/

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
	
//-------------------------------------- Conditional Statement ---------------------------------------/

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
	
//----------------------------------------- Parallel Loop -------------------------------------------/

IndexRangeCondition::IndexRangeCondition(List<Identifier*> *i, Identifier *c, Expr *rs, yyltype loc) : Node(loc) {
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
	for (int i = 0; i < indexes->NumElements(); i++) {
		Identifier *index = indexes->Nth(i);
		IndexScope::currentScope->initiateAssociationList(index->getName());
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

PLoopStmt::PLoopStmt(List<IndexRangeCondition*> *rc, Stmt *b, yyltype loc) : LoopStmt(loc) {
	Assert(rc != NULL && b != NULL);
	rangeConditions = rc;
	for (int i = 0; i < rangeConditions->NumElements(); i++) {
		rangeConditions->Nth(i)->SetParent(this);
	}
	body = b;
	body->SetParent(this);
}

void PLoopStmt::PrintChildren(int indentLevel) {
	rangeConditions->PrintAll(indentLevel + 1);
	body->Print(indentLevel + 1);
}

void PLoopStmt::performTypeInference(Scope *executionScope) {
	Scope *loopScope = executionScope->enter_scope(this->scope);
	for (int i = 0; i < rangeConditions->NumElements(); i++) {
		IndexRangeCondition *cond = rangeConditions->Nth(i);
		cond->inferTypes(loopScope);
	}
	body->performTypeInference(loopScope);
}

void PLoopStmt::checkSemantics(Scope *executionScope, bool ignoreTypeFailures) {
	
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
	IndexScope::currentScope->goBackToOldScope();

	loopScope->detach_from_parent();
	this->scope = loopScope;
}

Hashtable<VariableAccess*> *PLoopStmt::getAccessedGlobalVariables(TaskGlobalReferences *globalReferences) {
	Hashtable<VariableAccess*> *table = body->getAccessedGlobalVariables(globalReferences);
	for (int i = 0; i < rangeConditions->NumElements(); i++) {
		IndexRangeCondition *cond = rangeConditions->Nth(i);
		mergeAccessedVariables(table, cond->getAccessedGlobalVariables(globalReferences));
	}
	return table;	
}

//--------------------------------------- Sequential Loop -------------------------------------------/

SLoopStmt::SLoopStmt(Identifier *i, Expr *re, Expr *se, Stmt *b, yyltype loc) : LoopStmt(loc) {
	Assert(i != NULL && re != NULL && b != NULL);
	id = i;
	id->SetParent(this);
	rangeExpr = re;
	rangeExpr->SetParent(this);
	stepExpr = se;
	if (stepExpr != NULL) {
		stepExpr->SetParent(this);
	}
	body = b;
	body->SetParent(this);
}
    	
void SLoopStmt::PrintChildren(int indentLevel) {
	id->Print(indentLevel + 1, "(Index) ");
	rangeExpr->Print(indentLevel + 1, "(Range) ");
	if (stepExpr != NULL) stepExpr->Print(indentLevel + 1, "(Step) ");
	body->Print(indentLevel + 1);
}

void SLoopStmt::performTypeInference(Scope *executionScope) {
	Scope *loopScope = executionScope->enter_scope(this->scope);
	rangeExpr->inferType(loopScope, Type::rangeType);
	if (stepExpr != NULL) stepExpr->inferType(loopScope, Type::intType);
	body->performTypeInference(loopScope);
}

void SLoopStmt::checkSemantics(Scope *executionScope, bool ignoreTypeFailures) {

	Scope *loopScope = executionScope->enter_scope(new Scope(StatementBlockScope));
	if (loopScope->lookup(id->getName()) != NULL) {
		ReportError::ConflictingDefinition(id, ignoreTypeFailures);
	} else {
		VariableSymbol *var = new VariableSymbol(new VariableDef(id, Type::intType));
		loopScope->insert_symbol(var);
	}
	rangeExpr->resolveType(loopScope, ignoreTypeFailures);
	if (stepExpr != NULL) stepExpr->resolveType(loopScope, ignoreTypeFailures);

	body->checkSemantics(loopScope, ignoreTypeFailures);
	loopScope->detach_from_parent();
	this->scope = loopScope;
}

Hashtable<VariableAccess*> *SLoopStmt::getAccessedGlobalVariables(TaskGlobalReferences *globalReferences) {
	
	Hashtable<VariableAccess*> *table = body->getAccessedGlobalVariables(globalReferences);
	mergeAccessedVariables(table, rangeExpr->getAccessedGlobalVariables(globalReferences));
	if (stepExpr != NULL) mergeAccessedVariables(table, 
			stepExpr->getAccessedGlobalVariables(globalReferences));
	
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

Hashtable<VariableAccess*> *WhileStmt::getAccessedGlobalVariables(TaskGlobalReferences *globalReferences) {
	Hashtable<VariableAccess*> *table = condition->getAccessedGlobalVariables(globalReferences);
	mergeAccessedVariables(table, body->getAccessedGlobalVariables(globalReferences));
	return table;
}
