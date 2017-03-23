#include "../ast.h"
#include "../ast_type.h"
#include "../ast_def.h"
#include "../ast_stmt.h"
#include "../ast_type.h"
#include "../ast_task.h"
#include "../ast_partition.h"
#include "../../common/errors.h"
#include "../../common/location.h"
#include "../../common/constant.h"
#include "../../semantics/scope.h"
#include "../../semantics/symbol.h"
#include "../../semantics/helper.h"
#include "../../semantics/computation_flow.h"
#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/hashtable.h"
#include "../../../../common-libs/utils/string_utils.h"

#include <sstream>

//------------------------------------------------------- Stages Section ----------------------------------------------------------/

//------------------------------------------------------------------------------------------------------------------Stage Definition
StageDefinition::StageDefinition(Identifier *name, List<Identifier*> *parameters, Stmt *codeBody) {
	Assert(name != NULL 
		&& parameters != NULL && parameters->NumElements() > 0 
		&& codeBody != NULL);
	this->name = name;
	this->name->SetParent(this);
	this->parameters = parameters;
	for (int i = 0; i < parameters->NumElements(); i++) {
		this->parameters->Nth(i)->SetParent(this);
	}
	this->codeBody = codeBody;
	this->codeBody->SetParent(this);
}

void StageDefinition::PrintChildren(int indentLevel) {
	name->Print(indentLevel + 1, "Name");
        PrintLabel(indentLevel + 1, "Parameters");
        parameters->PrintAll(indentLevel + 2);
        PrintLabel(indentLevel + 1, "Code Body");
	codeBody->Print(indentLevel + 2);
}

void StageDefinition::determineArrayDimensions() {

	Hashtable<ArrayDimConfig*> *resolvedArrays = new Hashtable<ArrayDimConfig*>;
	List<Expr*> *arrayAccesses = new List<Expr*>;
	codeBody->retrieveExprByType(arrayAccesses, ARRAY_ACC);

	for (int i = 0; i < arrayAccesses->NumElements(); i++) {
		ArrayAccess *arrayAcc = (ArrayAccess*) arrayAccesses->Nth(i);
		int dimensionality = arrayAcc->getIndexPosition() + 1;
		Expr *baseExpr = arrayAcc->getEndpointOfArrayAccess();
		FieldAccess *baseField = dynamic_cast<FieldAccess*>(baseExpr);

		// This condition should not be true in the general case as arrays cannot hold other arrays 
		// as elements in IT. The only time this will be true is when a static array property of a
		// user defined type instance has been accessed. Those array's dimensions are already known
		// from the type definition.
		if (baseField == NULL || !baseField->isTerminalField()) continue;

		const char *arrayName = baseField->getField()->getName();
		
		// check if the use of the array in the particular expression confirms with any earlier use
		// of the same array
		ArrayDimConfig	*arrayConfig = NULL;
		if ((arrayConfig = resolvedArrays->Lookup(arrayName)) != NULL) {

			// raise an error that array access dimensions are not matching any previous access
			if (arrayConfig->getDimensions() != dimensionality) {
				ReportError::ConflictingArrayDimensionCounts(arrayAcc->GetLocation(), 
					arrayName, arrayConfig->getDimensions(), dimensionality, false);
			}

		// if this is the first time use of the array then just store it as resolved
		} else {
			arrayConfig = new ArrayDimConfig(arrayName, dimensionality);
			resolvedArrays->Enter(arrayName, arrayConfig);
		}

		// annotate the field access as accessing an array of a particular dimensionality for later
		// validation against actual invocation arguments
		baseField->flagAsArrayField(dimensionality);			
	}

	delete resolvedArrays;
	delete arrayAccesses;
}

//--------------------------------------------------------------------------------------------------------------------Stages Section

StagesSection::StagesSection(List<StageDefinition*> *stages, yyltype loc) : Node(loc) {
	Assert(stages != NULL && stages->NumElements() > 0);
	this->stages = stages;
	for (int i = 0; i < stages->NumElements(); i++) {
		stages->Nth(i)->SetParent(this);
	}
}

void StagesSection::PrintChildren(int indentLevel) {
	stages->PrintAll(indentLevel + 1);
}

StageDefinition *StagesSection::retrieveStage(const char *stageName) {
	for (int i = 0; i < stages->NumElements(); i++) {
		StageDefinition *stage = stages->Nth(i);
		if (strcmp(stage->getName(), stageName) == 0) {
			return stage;
		}
	}
	return NULL;
}

