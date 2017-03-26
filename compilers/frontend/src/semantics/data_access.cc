#include "data_access.h"
#include "scope.h"
#include "symbol.h"
#include "task_space.h"
#include "../syntax/ast_task.h"
#include "../syntax/ast_task.h"
#include "../../../common-libs/utils/list.h"
#include "../../../common-libs/utils/hashtable.h"
#include "../../../common-libs/utils/string_utils.h"

#include <algorithm>
#include <sstream>

//---------------------------------------------- Access Flags ---------------------------------------------------/

void AccessFlags::mergeFlags(AccessFlags *other) {
	if (other == NULL) return;
	if (other->read) this->read = true;
	if (other->write) this->write = true;
	if (other->reduce) this->reduce = true;
	if (other->redirect) this->redirect = true;
}

void AccessFlags::printFlags() {
	if (read) printf("-R-");
	if (write) printf("-W-");
	if (reduce) printf("-A-");
	if (redirect) printf("-C-");
}

//-------------------------------------------- Variable Access --------------------------------------------------/

VariableAccess::VariableAccess(const char *varName) {
	this->varName = varName;
	contentAccess = false;
	metadataAccess = false;
	contentAccessFlags = NULL;
	metadataAccessFlags = NULL;
	localAccess = false;
}

void VariableAccess::markContentAccess() {
	if (contentAccess) return;	
	contentAccess = true;
	contentAccessFlags = new AccessFlags;
}

void VariableAccess::markMetadataAccess() {
	if (metadataAccess) return;	
	metadataAccess = true;
	metadataAccessFlags = new AccessFlags;
}

void VariableAccess::mergeAccessInfo(VariableAccess *other) {
	if (other->contentAccess && !this->contentAccess) {
		markContentAccess();
	}
	if (this->contentAccess) {
		contentAccessFlags->mergeFlags(other->contentAccessFlags);
	} 

	if (other->metadataAccess && !this->metadataAccess) {
		markMetadataAccess();
	}
	if (this->metadataAccess) {
		metadataAccessFlags->mergeFlags(other->metadataAccessFlags);
	} 
}

void VariableAccess::printAccessDetail(int indent) {
	for (int i = 0; i < indent; i++) printf("\t");
	printf("%s:", varName);
	if (contentAccess) {
		printf(" content-");
		contentAccessFlags->printFlags();
	}
	if (metadataAccess) {
		printf(" metadata-");
		metadataAccessFlags->printFlags();
	}
	printf("\n");
}

//-----------------------------------------Task Global References -----------------------------------------------/

TaskGlobalReferences::TaskGlobalReferences(Scope *taskGlobalScope) {
	this->globalScope = taskGlobalScope;
	referencesToGlobals = new Hashtable<const char*>;
}

void TaskGlobalReferences::setNewReference(const char *localVarName, const char *globalVarName) {
	Assert(globalScope->lookup(globalVarName) != NULL);
	referencesToGlobals->Enter(localVarName, globalVarName, true);
}

bool TaskGlobalReferences::doesReferToGlobal(const char *localVarName) {
	if (globalScope->lookup(localVarName) != NULL) return true;
	return (referencesToGlobals->Lookup(localVarName) != NULL);
}

bool TaskGlobalReferences::isGlobalVariable(const char *name) {
	return globalScope->lookup(name) != NULL;
}

VariableSymbol *TaskGlobalReferences::getGlobalRoot(const char *localVarName) {
	if (globalScope->lookup(localVarName) != NULL) { 
		return (VariableSymbol*) globalScope->lookup(localVarName);
	}
	if (referencesToGlobals->Lookup(localVarName) == NULL) return NULL;
	const char *globalVarName = referencesToGlobals->Lookup(localVarName);
	return (VariableSymbol*) globalScope->lookup(globalVarName);
}
