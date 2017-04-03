#include "../ast.h"
#include "../ast_type.h"
#include "../ast_def.h"
#include "../ast_stmt.h"
#include "../../semantics/scope.h"
#include "../../semantics/symbol.h"
#include "../../common/errors.h"
#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/string_utils.h"

#include <fstream>
#include <sstream>
#include <cstdlib>
#include <iostream>

//----------------------------------------- Program Definition ------------------------------------------/

ProgramDef *ProgramDef::program = NULL;

ProgramDef::ProgramDef(List<Node*> *c) : Definition() {
        Assert(c != NULL);
        components = c;
        for (int i = 0; i < components->NumElements(); i++) {
                Node *node = components->Nth(i);
                node->SetParent(this);
        }
}

void ProgramDef::PrintChildren(int indentLevel) {
        components->PrintAll(indentLevel+1);
        printf("\n");
}

Node *ProgramDef::clone() {	
	List<Node*> *newCompList = new List<Node*>;
	for (int i = 0; i < components->NumElements(); i++) {
                Node *node = components->Nth(i);
		Node *newNode = node->clone();
		newCompList->Append(newNode);
	}
	ProgramDef *newDef = new ProgramDef(newCompList);
	return newDef;
}

// note that the sequencing of some supporting function calls within this function is important  for 
// the correct operation of semantic analysis
void ProgramDef::performScopeAndTypeChecking() {
	
	//------------------------------------------------define an overall scope for the program and
 
	Scope *scope = new Scope(ProgramScope);

	//---------------------------determine the element types and properties of user defined types

        Type::storeBuiltInTypesInScope(scope);
	List<Definition*> *classDefs = getComponentsByType(CLASS_DEF);
	for (int i = 0; i < classDefs->NumElements(); i++) {
		TupleDef *classDef = (TupleDef*) classDefs->Nth(i);
		Identifier *defId = classDef->getId();
                if (scope->lookup(defId->getName()) != NULL) {
                        ReportError::ConflictingDefinition(defId, false);
                } else {
			classDef->attachScope(scope);
		}	
	}
	for (int i = 0; i < classDefs->NumElements(); i++) {
		Definition *classDef = classDefs->Nth(i);
		classDef->validateScope(scope);	
	}

	//------------------annotate the use of function arguments inside function bodies with value
	//-----------------------------------------------------value or reference type specification
	List<Definition*> *fnDefs = getComponentsByType(FN_DEF);
	for (int i = 0; i < fnDefs->NumElements(); i++) {
		FunctionDef *fnDef = (FunctionDef*) fnDefs->Nth(i);
		fnDef->annotateArgAccessesByType();
	}

	//---------------------------create a static function definition map to retrieve definitions
	//----------------------------by name for type-inference analysis for any function-call made 
	//--------------------------------------------------------------from the rest of the program
	FunctionDef::fnDefMap = new Hashtable<FunctionDef*>;
	for (int i = 0; i < fnDefs->NumElements(); i++) {
		FunctionDef *fnDef = (FunctionDef*) fnDefs->Nth(i);
		const char *fnName = fnDef->getId()->getName();
                if (FunctionDef::fnDefMap->Lookup(fnName) != NULL) {
                        ReportError::ConflictingDefinition(fnDef->getId(), false);
                } else {
			FunctionDef::fnDefMap->Enter(fnName, fnDef);
		}
	}
 
	//-----------------------analyze the stages section of the tasks before stage instanciations
	List<Definition*> *taskDefs = getComponentsByType(TASK_DEF);
        for (int i = 0; i < taskDefs->NumElements(); i++) {
                TaskDef *taskDef = (TaskDef*) taskDefs->Nth(i);
		taskDef->analyzeStageDefinitions();
        }

	//---------------------------------------perform scope preparations for all task definitions
	for (int i = 0; i < taskDefs->NumElements(); i++) {
		TaskDef *task = (TaskDef*) taskDefs->Nth(i);
		task->attachScope(scope);
		task->typeCheckInitializeSection(scope);
	}
	//-------------------------------------if there are errors in some tasks' Initialize Section 
	//---------------------------------------then there is not any point progressing any further
	if (ReportError::NumErrors() > 0) return;

	//----------------------------------------------Construct the partition hierarchies of tasks
	for (int i = 0; i < taskDefs->NumElements(); i++) {
                TaskDef *task = (TaskDef*) taskDefs->Nth(i);
		task->constructPartitionHierarchy();
	}
	
	//--------------------------------------------------Construct the computation flows of tasks
	for (int i = 0; i < taskDefs->NumElements(); i++) {
                TaskDef *task = (TaskDef*) taskDefs->Nth(i);
		task->constructComputationFlow(scope);
	}

	//--------------------------------if the flow construction process failed for some task then
	//-------------------------------------------------there is no point progressing any further 
	if (ReportError::NumErrors() > 0) return;

	//--------------------------------------------------------------Now validate the task scopes
	for (int i = 0; i < taskDefs->NumElements(); i++) {
                TaskDef *task = (TaskDef*) taskDefs->Nth(i);
		task->validateScope(scope);
	}	

	//--------------------------------------------------process the program coordinator function
	List<Definition*> *defList = getComponentsByType(COORD_DEF);
	if (defList->NumElements() != 1) {
		std::cout << "Error: There must be one and only one program coordinator function\n";
		std::exit(EXIT_FAILURE);
	}
	CoordinatorDef *coordinator = (CoordinatorDef*) defList->Nth(0);
	coordinator->attachScope(scope);
	coordinator->validateScope(scope);
	
	//-------------------------------------------associate the program scope with the definition

	symbol = new Symbol("Program", this);
        symbol->setNestedScope(scope);
}

List<Definition*> *ProgramDef::getComponentsByType(DefTypeId typeId) {

	List<Definition*> *filteredList = new List<Definition*>;
	for (int i = 0; i < components->NumElements(); i++) {
                Definition *def = (Definition*) components->Nth(i);
		if (def->getDefTypeId() == typeId) {
			filteredList->Append(def);
		}
	}
	return filteredList;
}

// Note that currently the static analysis phase is restricted to analyses of the component tasks.
// We need to add static analyses of other components and also inter-component interactions when we
// implement more advanced features, e.g., concurrent task executions, promised by the language. 
void ProgramDef::performStaticAnalysis() {
	List<Definition*> *taskDefs = getComponentsByType(TASK_DEF);
        for (int i = 0; i < taskDefs->NumElements(); i++) {
                TaskDef *taskDef = (TaskDef*) taskDefs->Nth(i);
		taskDef->performStaticAnalysis();
	}	
}

Scope *ProgramDef::getScope() { return symbol->getNestedScope(); }
