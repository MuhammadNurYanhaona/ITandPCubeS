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
#include "../../semantics/array_acc_transfrom.h"
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

void StageInvocation::constructComputeFlow(CompositeStage *currCompStage, FlowStageConstrInfo *cnstrInfo) {

	Space *currLps = cnstrInfo->getCurrSpace();
	int group = cnstrInfo->getCurrGroupIndex();
	int repeatBlock = cnstrInfo->getCurrRepeatBlockIndex();

	// creates a new scope for the stage invocation
	Scope *parentScope = cnstrInfo->getScope();
	Scope *stageScope = parentScope->enter_scope(new Scope(ComputationStageScope));
	Symbol *lpuIdSymbol = currLps->getLpuIdSymbol();
        if (lpuIdSymbol != NULL) { stageScope->insert_symbol(lpuIdSymbol); }
	
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

	//--------------------------------------------------------------- resolve the stage definition body for the
	//------------------------------------------------------------- arguments being used for current invocation

	// cloning is a must here as the same compute stages may be invoked from multiple places
	Stmt *codeBody = stageDef->getCode();
	Stmt *code = (Stmt*) codeBody->clone();

	// get metadata for parameter replacement
	List<ParamReplacementConfig*> *paramReplConfs = generateParamReplacementConfigs();
	
	// produce code for parameters that should be generated from the argument
	List<Stmt*> *paramGeneratorCode = produceParamGeneratorCode(stageScope, paramReplConfs);
	
	// create a map of parameter to replacement config for field accesses needing only name change in the code 
	Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap = new Hashtable<ParamReplacementConfig*>;
	for (int i = 0; i < paramReplConfs->NumElements(); i++) {
		ParamReplacementConfig *conf = paramReplConfs->Nth(i);
		if (conf->getReplacementType() != Change_Name) continue;

		const char *paramName = conf->getParameter()->getName();
		nameAdjustmentInstrMap->Enter(paramName, conf);	
	}

	// create another map of parameter to replacement config for arguments that are array parts, consequently
	// requiring a more complex transformation of expressions involving the parameter	
	Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap  = new Hashtable<ParamReplacementConfig*>;
	for (int i = 0; i < paramReplConfs->NumElements(); i++) {
		ParamReplacementConfig *conf = paramReplConfs->Nth(i);
		if (conf->getReplacementType() != Update_Expr) continue;

		const char *paramName = conf->getParameter()->getName();
		arrayAccXformInstrMap->Enter(paramName, conf);	
	}

	// apply the parameter replacement strategy on the code
	code->performStageParamReplacement(nameAdjustmentInstrMap, arrayAccXformInstrMap);
	
	// then do scope-and-type analysis in the updated code
	// the scope and type analysis should repeat as long as we resolve new expression types
        int iteration = 0;
        int resolvedTypes = 0;
        do {
                resolvedTypes = code->resolveExprTypesAndScopes(stageScope, iteration);
                iteration++;
        } while (resolvedTypes != 0);

	// check if the stage instance has any un-resolved or erroneous expression; if there is any expression in 
	// error then we can stop and exit
        errorCount = code->emitScopeAndTypeErrors(stageScope);
	if (errorCount > 0) {
		ReportError::CouldNotResolveStageForArgs(GetLocation(), nameOfStage, false);
		std::exit(EXIT_FAILURE);
	}

	// generate the final code by combining a1ny param generator statements with the resolved original code
	Stmt *finalCode = NULL;
	if (paramGeneratorCode->NumElements() != 0) {
		paramGeneratorCode->Append(code);
		finalCode = new StmtBlock(paramGeneratorCode);
	} else {
		finalCode = code;
	}

	//----------------------------------------------------------------------end of polymorphic stage resolution 

	// create a stage instanciation
	StageInstanciation *stageInstance = new StageInstanciation(currLps);
	int index = cnstrInfo->getLastStageIndex();
        stageInstance->setIndex(index);
        stageInstance->setGroupNo(group);
        stageInstance->setRepeatIndex(repeatBlock);
        cnstrInfo->advanceLastStageIndex();

	// assign the code, the scope, and a name to the stage instance
	stageInstance->setCode(finalCode);
	stageScope->detach_from_parent();
	stageInstance->setScope(stageScope);
	std::ostringstream nameStream;
	nameStream << nameOfStage << "_stage_" << index;
	stageInstance->setName(strdup(nameStream.str().c_str())); 

	// retrieve information about array-part arguments and store that in the stage instance
	List<ArrayPartConfig*> *arrayPartConfigList = getIndexRangeLimitedArrayArgConfigs(paramReplConfs);
	stageInstance->setArrayPartArgConfList(arrayPartConfigList);

	// assign the location of the current AST element to the flow-stage for later error checking purpose
	stageInstance->assignLocation(GetLocation());

	// add the newly created stage to the parent composite stage    
        currCompStage->addStageAtEnd(stageInstance);
}

List<ParamReplacementConfig*> *StageInvocation::generateParamReplacementConfigs() {
	
	List<ParamReplacementConfig*> *replacementList = new List<ParamReplacementConfig*>;	
	const char *nameOfStage = stageName->getName();
	StageDefinition *stageDef = TaskDef::currentTask->getStagesSection()->retrieveStage(nameOfStage);
	List<Identifier*> *parameters = stageDef->getParameters();

	for (int i = 0; i < parameters->NumElements(); i++) {
		
		Identifier *param = parameters->Nth(i);
		const char *paramName = param->getName();
		Expr *argument = arguments->Nth(i);
		ParamReplacementConfig *config = NULL;
		
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
			config = new ParamReplacementConfig(param, argument, Evaluate_Before);
			replacementList->Append(config);
			continue;
		}

		// if the argument is a terminal field whose name matches that of the parameter then no replacement
		// is needed
		if (isTerminalField) {
			const char *argName = fieldAcc->getField()->getName();
			if (strcmp(argName, paramName) == 0) {
				config = new ParamReplacementConfig(param, argument, No_Replacement);
			// if the names are not the same then any access to the parameter field can be just renamed
			// to access the argument
			} else {
				config = new ParamReplacementConfig(param, argument, Change_Name);
			}
			replacementList->Append(config);
			continue;
		}

		// a reduction variable as an argument will always demand a change of name from parameter access to
		// access to the argument variable
		if (isReductionVar) {
			config = new ParamReplacementConfig(param, argument, Change_Name);
			replacementList->Append(config);
			continue;
		}
  
		// the last remaing case is that the argument is a part of an array; this is the most complicated 
		// case; we need to update expressions that use the corresponding parameter in many ways to produce
		// valid and intended accesses to the argument array's element and metadata
		config = new ParamReplacementConfig(param, argument, Update_Expr);
		// set up an array-part-cofig for this replacement type to facilitate expression replacements
		config->setArrayPartConfig(new ArrayPartConfig(arrayAcc));
		replacementList->Append(config);
	}

	return replacementList;		
}

List<Stmt*> *StageInvocation::produceParamGeneratorCode(Scope *stageScope,
		List<ParamReplacementConfig*> *paramReplConfigList) {

	List<Stmt*> *stmtList = new List<Stmt*>;
	for (int i = 0; i < paramReplConfigList->NumElements(); i++) {
		ParamReplacementConfig *config = paramReplConfigList->Nth(i);
		if (config->getReplacementType() != Evaluate_Before) {
			continue;
		}

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

List<ArrayPartConfig*> *StageInvocation::getIndexRangeLimitedArrayArgConfigs(
		List<ParamReplacementConfig*> *paramReplConfigList) {

	List<ArrayPartConfig*> *configList = new List<ArrayPartConfig*>;
	for (int i = 0; i < paramReplConfigList->NumElements(); i++) {
		ArrayPartConfig *partConfig = paramReplConfigList->Nth(i)->getArrayPartConfig();
		if (partConfig == NULL) continue;

		FieldAccess *baseArray = partConfig->getBaseArrayAccess();
		ArrayType *type = (ArrayType*) baseArray->getType();
		int dimensions = type->getDimensions();
		
		for (int j = 0; j < dimensions; j++) {
			if (partConfig->isLimitedIndexRange(j)) {
				configList->Append(partConfig);
				break;
			}
		}		
	}

	return configList;
}

