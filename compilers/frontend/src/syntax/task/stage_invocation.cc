#include "../ast.h"
#include "../ast_type.h"
#include "../ast_def.h"
#include "../ast_expr.h"
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

//----------------------------------------------------- Computation Section -------------------------------------------------------/

//------------------------------------------------------------------------------------------------------------------Stage Invocation

StageInvocation::StageInvocation(Identifier *stageName, 
		List<Expr*> *arguments, yyltype loc) : FlowPart(loc) {
	Assert(stageName != NULL && arguments != NULL && arguments->NumElements() > 0);
	this->stageName = stageName;
	this->stageName->SetParent(this);
	this->arguments = arguments;
	for (int i = 0; i < arguments->NumElements(); i++) {
		this->arguments->Nth(i)->SetParent(this);
	}
}

void StageInvocation::PrintChildren(int indentLevel) {
	stageName->Print(indentLevel + 1, "Stage ");
	PrintLabel(indentLevel + 1, "Arguments");
	arguments->PrintAll(indentLevel + 2);
}

void StageInvocation::constructComputeFlow(CompositeStage *currCompStage,
		semantic_helper::FlowStageConstrInfo *cnstrInfo) {

	Space *currLps = cnstrInfo->getCurrSpace();
	int group = cnstrInfo->getCurrGroupIndex();
	int repeatBlock = cnstrInfo->getCurrRepeatBlockIndex();

	// creates a new scope for the stage invocation
	Scope *parentScope = cnstrInfo->getScope();
	Scope *stageScope = parentScope->enter_scope(new Scope(ComputationStageScope));
	
	// validate that the stage being invoked does exist
	const char *nameOfStage = stageName->getName();
	StageDefinition *stageDef = TaskDef::currentTask->getStagesSection()->retrieveStage(nameOfStage);
	if (stageDef == NULL) {
		ReportError::UndefinedSymbol(stageName, false);
		return;
	}

	//-----------------------------------------------------------------------------------validate the arguments

	// ensure the argument and parameter counts matches
	List<Identifier*> *parameters = stageDef->getParameters();
	int paramCount = parameters->NumElements();
	int argCount = arguments->NumElements();
	if (paramCount != argCount) {
		ReportError::TooFewOrTooManyParameters(stageName, argCount, paramCount, false);
		return;
	}

	// ensure that the arguments form valid expressions
	int errorCount = 0;
	List<FieldAccess*> *argFields = new List<FieldAccess*>;
	for (int i = 0; i < arguments->NumElements(); i++) {
		Expr *arg = arguments->Nth(i);
		arg->resolveExprTypesAndScopes(stageScope);
		errorCount += arg->emitScopeAndTypeErrors(stageScope);
		arg->retrieveTerminalFieldAccesses(argFields);
	}
	if (errorCount > 0) return;

	// ensure that the argument fields are accessible from the LPS the stage has been invoked in
	for (int i = 0; i < argFields->NumElements(); i++) {
		FieldAccess *field = argFields->Nth(i);
		const char *fieldName = field->getField()->getName();
		VariableSymbol *symbol = (VariableSymbol*) stageScope->lookup(fieldName);
		Type *type = symbol->getType();
		ArrayType *array = dynamic_cast<ArrayType*>(type);
		StaticArrayType *staticArray = dynamic_cast<StaticArrayType*>(type);
		bool dynamicType = (array != NULL) && (staticArray == NULL);
		if (dynamicType) {
			DataStructure *structure = currLps->getLocalStructure(fieldName);
			if (structure == NULL) {
				ReportError::ArrayPartitionUnknown(field->GetLocation(), 
						fieldName, 
						stageName->getName(), 
						currLps->getName());
				errorCount++;
			}
		}
	}
	if (errorCount > 0) return;
	
	//------------------------------------------------------------------------------ end of argument validation

	// assign the scope to the stage instanciation
}

List<semantic_helper::ParamReplacementConfig*> *StageInvocation::generateParamReplacementConfigs() {
	
	List<semantic_helper::ParamReplacementConfig*> *replacementList 
			= new List<semantic_helper::ParamReplacementConfig*>;	
	const char *nameOfStage = stageName->getName();
	StageDefinition *stageDef = TaskDef::currentTask->getStagesSection()->retrieveStage(nameOfStage);
	List<Identifier*> *parameters = stageDef->getParameters();

	for (int i = 0; i < parameters->NumElements(); i++) {
		
		Identifier *param = parameters->Nth(i);
		const char *paramName = param->getName();
		Expr *argument = arguments->Nth(i);
		semantic_helper::ParamReplacementConfig *config = NULL;
		
		// get argument type and check if it is a dynamic array
		Type *type = argument->getType();
		ArrayType *array = dynamic_cast<ArrayType*>(type);
		StaticArrayType *staticArray = dynamic_cast<StaticArrayType*>(type);
		bool dynamicType = (array != NULL) && (staticArray == NULL);

		//--------------------------------------check if the argument is any of the three specific types of
		//---------------------------------------------expressions that are treated differently from others

		bool isReductionVar = (dynamic_cast<ReductionVar*>(argument) != NULL);

		FieldAccess *fieldAcc = dynamic_cast<FieldAccess*>(argument);
		bool isTerminalField = (fieldAcc != NULL && fieldAcc->isTerminalField());

		ArrayAccess *arrayAcc = dynamic_cast<ArrayAccess*>(argument);
		bool isArrayPart = (arrayAcc != NULL && dynamicType);

		//-----------------------------------------------------------------done argument type determination

		// if the argument is none of the above three types then it must be evaluated before and a local
		// variable matching the parameter name must be created in the compute stage body
		if (!(isReductionVar || isTerminalField || isArrayPart)) {
			config = new semantic_helper::ParamReplacementConfig(param, 
					argument, Evaluate_Before);
			replacementList->Append(config);
			continue;
		}

		// if the argument is a terminal field whose name matches that of the parameter then no replacement
		// is needed
		if (isTerminalField) {
			const char *argName = fieldAcc->getField()->getName();
			if (strcmp(argName, paramName) == 0) {
				config = new semantic_helper::ParamReplacementConfig(param, 
						argument, No_Replacement);
			// if the names are not the same then any access to the parameter field can be just renamed
			// to access the argument
			} else {
				config = new semantic_helper::ParamReplacementConfig(param,
                                                argument, Change_Name);
			}
			replacementList->Append(config);
			continue;
		}

		// a reduction variable as an argument will always demand a change of name from parameter access to
		// access to the argument variable
		if (isReductionVar) {
			config = new semantic_helper::ParamReplacementConfig(param,
					argument, Change_Name);
			replacementList->Append(config);
			continue;
		}
  
		// the last remaing case is that the argument is a part of an array; this is the most complicated 
		// case; we need to update expressions that use the corresponding parameter in many ways to produce
		// valid and intended accesses to the argument array's element and metadata
		config = new semantic_helper::ParamReplacementConfig(param, argument, Update_Expr);
		// set up an array-part-cofig for this replacement type to facilitate expression replacements
		config->setArrayPartConfig(new ArrayPartConfig(arrayAcc));
		replacementList->Append(config);
	}

	return replacementList;		
}

List<Stmt*> *StageInvocation::produceParamGeneratorCode(Scope *stageScope,
		List<semantic_helper::ParamReplacementConfig*> *paramReplConfigList) {

	List<Stmt*> *stmtList = new List<Stmt*>;
	for (int i = 0; i < paramReplConfigList->NumElements(); i++) {
		semantic_helper::ParamReplacementConfig *config = paramReplConfigList->Nth(i);
		if (config->getReplacementType() != Evaluate_Before) continue;

		Identifier *param = config->getParameter();
		Expr *argument = config->getInvokingArg();
		Type *type = argument->getType();

		// insert a symbol in the scope so a local variable with proper name is declared at the beginning
		VariableSymbol *symbol = new VariableSymbol(param->getName(), type);
		stageScope->insert_inferred_symbol(symbol);

		// then generate an assignment expression from the argument to the parameter
		FieldAccess *left = new FieldAccess(NULL, param, *argument->GetLocation());
		AssignmentExpr *assignment = new AssignmentExpr(left, argument, *GetLocation());
		stmtList->Append(assignment);
	}
	return stmtList;
}

