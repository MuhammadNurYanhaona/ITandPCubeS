#include "../../../utils/code_constant.h"
#include "../../../utils/name_transformer.h"
#include "../../../../../../frontend/src/syntax/ast_expr.h"
#include "../../../../../../frontend/src/syntax/ast_stmt.h"
#include "../../../../../../frontend/src/semantics/task_space.h"
#include "../../../../../../frontend/src/semantics/computation_flow.h"
#include "../../../../../../frontend/src/semantics/array_acc_transfrom.h"
#include "../../../../../../frontend/src/static-analysis/sync_stat.h"
#include "../../../../../../frontend/src/static-analysis/reduction_info.h"
#include "../../../../../../frontend/src/static-analysis/data_dependency.h"

#include <fstream>
#include <sstream>

void StageInstanciation::translateCode(std::ofstream &stream) {

	std::string localMdHd =  	"\n\t//-------------------- Local Copies of Metadata -----------------------------\n\n";
	std::string localMdAd =  	"\n\t//---------------- Metadata Adjustment for Array Parts ----------------------\n\n";
	std::string redRstHd =  	"\n\t//------------------ Partial Results of Reductions  -------------------------\n\n";
	std::string localVarDclHd = 	"\n\t//------------------- Local Variable Declarations ---------------------------\n\n";
	std::string computeHd = 	"\n\t//----------------------- Computation Begins --------------------------------\n\n";
	std::string returnHd =  	"\n\t//------------------------- Returning Flag ----------------------------------\n\n";
	
	// reset the name transformer to user common "lpu." prefix for array access in case it is been modified
	ntransform::NameTransformer::transformer->setLpuPrefix("lpu->");

	// create local variables for all array dimensions so that later on the name-transformer, that adds 
	// prefix/suffix to accessed global variables, can work properly
	stream <<  localMdHd;
	stream << "\t// create local copies of partition and storage dimension configs of all arrays\n";
	std::string stmtIndent = "\t";
        List<const char*> *localArrays = filterInArraysFromAccessMap();
	for (int i = 0; i < localArrays->NumElements(); i++) {
        	const char *arrayName = localArrays->Nth(i);
        	ArrayDataStructure *array = (ArrayDataStructure*) space->getStructure(arrayName);
        	int dimensions = array->getDimensionality();
        	stream << stmtIndent << "Dimension ";
                stream  << arrayName << "PartDims[" << dimensions << "]" << stmtSeparator;
                for (int j = 0; j < dimensions; j++) {
                	stream << stmtIndent;
                	stream << arrayName << "PartDims[" << j << "] = lpu->";
                        stream << arrayName << "PartDims[" << j << "].partition" << stmtSeparator;
                }
                stream << stmtIndent << "Dimension ";
                stream  << arrayName << "StoreDims[" << dimensions << "];\n";
                for (int j = 0; j < dimensions; j++) {
                	stream << stmtIndent;
               		stream << arrayName << "StoreDims[" << j << "] = lpu->";
                        stream << arrayName << "PartDims[" << j << "].storage" << stmtSeparator;
        	}
        }

	// create a local part-dimension object for probable array dimension based range or assignment expressions
	stream << "\n\t// create a local part-dimension object for later use\n";
        stream << indent << "PartDimension partConfig" << stmtSeparator;
	
	// create a local integer for holding intermediate values of transformed index during inclusion testing
	stream << "\n\t// create a local transformed index variable for later use\n";
        stream << indent << "int xformIndex" << stmtSeparator;

	// if there is any array part argument with dimension range being limited with an index-range expression
	// then adjust the metadata for the array
	if (arrayPartArgConfList != NULL && arrayPartArgConfList->NumElements() > 0) {
		stream << localMdAd;
		for (int i = 0; i < arrayPartArgConfList->NumElements(); i++) {
			ArrayPartConfig *partConfig = arrayPartArgConfList->Nth(i);
			FieldAccess *baseArray = partConfig->getBaseArrayAccess();
			const char *arrayName = baseArray->getField()->getName();
			ArrayType *type = (ArrayType*) baseArray->getType();
			int dimensions = type->getDimensions();
			for (int dimNo = 0; dimNo < dimensions; dimNo++) {
				if (!partConfig->isLimitedIndexRange(dimNo)) continue;

				std::ostringstream rangeStream;
				IndexRange *activeRange = partConfig->getAccessibleIndexRange(dimNo);
				activeRange->translate(rangeStream, 0, 0, space);
				stream << indent <<  arrayName << "PartDims[" << dimNo << "].adjust(";
				stream << rangeStream.str() << ")" << stmtSeparator;
			}
		}
	}
	
	// if the compute stage involves any reduction then extract the references of local partial results from
	// the reduction result map
	if (nestedReductions->NumElements() > 0) {
		stream << redRstHd;
		for (int i = 0; i < nestedReductions->NumElements(); i++) {
			ReductionMetadata *reduction = nestedReductions->Nth(i);
			const char *varName = reduction->getResultVar();
			stream << indent << "reduction::Result *" << varName << " = ";
			stream << "localReductionResultMap->Lookup(\"" << varName << "\")" << stmtSeparator;
		}
	}

	// declare any local variables found in the computation	
	std::ostringstream localVars;
        scope->declareVariables(localVars, 1);
	if (localVars.str().length() > 0) {
		stream <<  localVarDclHd;
		stream << localVars.str();
	}

        // translate statements into C++ code
	stream <<  computeHd;
	std::ostringstream codeStream;
	code->generateCode(codeStream, 1, space);
	stream << codeStream.str();

        // finally return a successfull run indicator
	stream <<  returnHd;
	stream << indent << "return SUCCESS_RUN" << stmtSeparator;	
}

void StageInstanciation::generateInvocationCode(std::ofstream &stream, int indentation, Space *containerSpace) {
	
	// write the indent
	std::ostringstream indentStr;
	for (int i = 0; i < indentation; i++) indentStr << indent;
	std::ostringstream nextIndent;
	nextIndent << indentStr.str();

	// do an entry checking of the executing PPU's ability to execute any code in the current LPS
	nextIndent << indent;
	stream << indentStr.str() << "if (threadState->isValidPpu(Space_" << space->getName();
	stream << ")) {\n";
	
	// invoke the related method with current LPU parameter ...
	stream << nextIndent.str() << "// invoking user computation\n";
	stream << nextIndent.str();
	stream << "int stage" << index << "Executed = ";
	stream << name << "(space" << space->getName() << "Lpu" << paramSeparator;
	// along with other default arguments
	stream << '\n' << nextIndent.str() << doubleIndent << "arrayMetadata" << paramSeparator;
	stream << '\n' << nextIndent.str() << doubleIndent << "taskGlobals" << paramSeparator;
	stream << '\n' << nextIndent.str() << doubleIndent << "threadLocals" << paramSeparator;
	if (this->hasNestedReductions()) {
		stream << '\n' << nextIndent.str() << doubleIndent << "reductionResultsMap" << paramSeparator;
	}
	stream << '\n' << nextIndent.str() << doubleIndent << "partition" << paramSeparator;
	stream << '\n' << nextIndent.str() << doubleIndent << "threadState->threadLog)" << stmtSeparator;

	// then update all synchronization counters that depend on the execution of this stage for their activation
	List<SyncRequirement*> *syncList = synchronizationReqs->getAllSyncRequirements();
	for (int i = 0; i < syncList->NumElements(); i++) {
		DependencyArc *arc = syncList->Nth(i)->getDependencyArc();
		if (arc->doesRequireSignal()) {
			stream << nextIndent.str() << arc->getArcName();
			stream << " += stage" << index << "Executed;\n";
		}
	}

	/*----------------------------------------------- Turned off	
	// write a log entry for the stage executed
	stream << nextIndent.str() << "threadState->logExecution(\"";
	stream << name << "\", Space_" << space->getName() << ");\n";
	----------------------------------------------------------*/ 	
	
	// end of the entry checking condition block
	stream << indentStr.str() << "}\n";
}

