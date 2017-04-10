#include "../../../utils/code_constant.h"
#include "../../../utils/name_transformer.h"
#include "../../../../../../common-libs/utils/list.h"
#include "../../../../../../common-libs/utils/hashtable.h"
#include "../../../../../../common-libs/domain-obj/constant.h"
#include "../../../../../../frontend/src/syntax/ast_expr.h"
#include "../../../../../../frontend/src/semantics/task_space.h"
#include "../../../../../../frontend/src/semantics/computation_flow.h"

#include <fstream>
#include <sstream>

void EpochBoundaryBlock::genCodeForScalarVarEpochUpdates(std::ofstream &stream,
                int indentation,
                List<const char*> *scalarVarList) {

	if (scalarVarList == NULL || scalarVarList->NumElements()  == 0) return;
	
	std::ostringstream indentStream;
	for (int i = 0; i < indentation; i++) indentStream << indent;
	std::string indentStr = indentStream.str();

	stream << indentStr << "{ // scope starts for version updates of scalar variables\n";

	for (int i = 0; i < scalarVarList->NumElements(); i++) {
		const char *varName = scalarVarList->Nth(i);
		DataStructure *structure = space->getStructure(varName);

		// version count is by default 0; thus we add 1 here
		int versions = structure->getLocalVersionCount() + 1;
                                
		// for a scalar variable's version update we need to swap values with older versions of the 
		// same variable directly inside the thread-globals object
		// swapping must be done from the rear end; otherwise, intermediate values will get corrupted 
		// in the process
		for (int v = versions - 1; v > 0; v--) {
			stream << indentStr << "taskGlobals->" << varName;
			stream << "_lag_" << v << " = taskGlobals->" << varName;
			if (v > 1) { stream << "_lag_" << (v - 1); }
			stream << stmtSeparator;
		}
	}

	stream << indentStr << "} // scope ends for version updates of scalar variables\n";
}

void EpochBoundaryBlock::genCodeForArrayVarEpochUpdates(std::ofstream &stream,
		const char *lpsName,
		int indentation,
		List<const char*> *arrayVarList) {

	if (arrayVarList == NULL || arrayVarList->NumElements() == 0) return;

	std::ostringstream indentStream;
	for (int i = 0; i < indentation; i++) indentStream << indent;
	std::string indentStr = indentStream.str();
	
	stream << indentStr << "{ // scope starts for version updates of LPU data from Space ";
	stream << lpsName << "\n";

	// get a hold of the hierarchical LPU ID to locate data parts based on LPU ID
	stream << indentStr << "List<int*> *lpuIdChain = threadState->getLpuIdChainWithoutCopy(";
	stream << '\n' << indentStr << doubleIndent << "Space_" << lpsName;
	Space *rootSpace = space->getRoot();
	stream << paramSeparator << "Space_" << rootSpace->getName() << ")" << stmtSeparator;

	for (int i = 0; i < arrayVarList->NumElements(); i++) {
		const char *varName = arrayVarList->Nth(i);
		stream << indentStr << "DataPartitionConfig *config = partConfigMap->Lookup(\"";
		stream << varName << "Space" << lpsName << "Config\")" << stmtSeparator;
		stream << indentStr << "PartIterator *iterator = ";
		stream << "threadState->getIterator(Space_" << lpsName << paramSeparator;
		stream << '\"' << varName << "\")" << stmtSeparator;
		stream << indentStr << "List<int*> *partId = iterator->getPartIdTemplate()" << stmtSeparator;
		stream << indentStr << "config->generatePartId(lpuIdChain";
		stream << paramSeparator << "partId)" << stmtSeparator;
		stream << indentStr << "DataItems *items = taskData->getDataItemsOfLps(\"" << lpsName;
		stream << "\"" << paramSeparator << "\"" << varName << "\")" << stmtSeparator;
		stream << indentStr << "DataPart *dataPart = items->getDataPart(partId" << paramSeparator;
		stream << "iterator)" << stmtSeparator;
		stream << indentStr << "dataPart->advanceEpoch()" << stmtSeparator;
	}
	
	stream << indentStr << "} // scope ends for version updates of LPU data from Space ";
	stream << lpsName << "\n";
}

void EpochBoundaryBlock::genLpuTraversalLoopBegin(std::ofstream &stream, const char *lpsName, int indentation) {
	
	const char *containerLpsName = this->space->getName();
	
	std::ostringstream indentStream;
	for (int i = 0; i < indentation; i++) indentStream << indent;
	std::string indentStr = indentStream.str();

        stream << indentStr << "{ // scope entrance for iterating LPUs of Space ";
        stream << lpsName << "\n";
        // declare a new variable for tracking the last LPU ID
        stream << indentStr << "int space" << lpsName << "LpuId = INVALID_ID" << stmtSeparator;
        // declare another variable to assign the value of get-Next-LPU call
        stream << indentStr << "LPU *lpu = NULL" << stmtSeparator;
        stream << indentStr << "while((lpu = threadState->getNextLpu(";
        stream << "Space_" << lpsName << paramSeparator << "Space_" << containerLpsName;
        stream << paramSeparator << "space" << lpsName << "LpuId)) != NULL) {\n";
}

void EpochBoundaryBlock::genLpuTraversalLoopEnd(std::ofstream &stream, const char *lpsName, int indentation) {
	
	const char *containerLpsName = this->space->getName();

        std::ostringstream indentStream; 
        for (int i = 0; i < indentation; i++) indentStream << indent;
        std::string indentStr = indentStream.str();

	// update the next LPU ID
        stream << indentStr << indent << "space" << lpsName << "LpuId = lpu->id" << stmtSeparator;
        stream << indentStr << "}\n"; 

        // at the end, remove LPS entry checkpoint checkpoint if the Epoch block holder LPS is not the root LPS
        if (!space->isRoot()) {
                stream << indentStr << "threadState->removeIterationBound(Space_";
                stream << containerLpsName << ')' << stmtSeparator;
        }

        // exit from the scope
        stream << indentStr << "} // scope exit for iterating LPUs of Space " << lpsName << "\n";     
}

void EpochBoundaryBlock::generateInvocationCode(std::ofstream &stream, int indentation, Space *containerSpace) {
	
	const char *currLpsName = this->space->getName();

	std::ostringstream indentStream;
	for (int i = 0; i < indentation; i++) indentStream << indent;
	std::string indentStr = indentStream.str();

	// start a new scope for all epoch advancement
	stream << indentStr << "{ // scope starts for epoch boundary\n";
	
	// retrieve common properties that will be needed to do data structure version updates and, if needed, LPU
	// reference refreshing
	stream << indentStr << indent << "TaskData *taskData = threadState->getTaskData()" << stmtSeparator;
	stream << indentStr << indent << "Hashtable<DataPartitionConfig*> ";
	stream << "*partConfigMap = threadState->getPartConfigMap()" << stmtSeparator;

	List<const char*> *scalarVarList = new List<const char*>;
	for (int i = 0; i < lpsList->NumElements(); i++) {
		
		const char *lpsName = lpsList->Nth(i);
		List<const char*> *epochUpdateList = lpsToVarMap->Lookup(lpsName);
		List<const char*> *arrayVarList = filterArrayVariables(epochUpdateList, scalarVarList);
		if (arrayVarList->NumElements() == 0) continue;
		
		if (strcmp(currLpsName, lpsName) == 0) {
			// In this case, the current LPU's data parts' versions must be updated. Afterwards, we
			// have to recreate the LPU object so that the internal pointers are updated properly.
			
			genCodeForArrayVarEpochUpdates(stream, lpsName, indentation + 1, arrayVarList);
                
			stream << indentStr << indent << "generateSpace" << lpsName << "Lpu(";
			stream << "threadState" << paramSeparator << "partConfigMap" << paramSeparator;
			stream << "taskData)" << stmtSeparator;
			stream << indentStr << indent << "space" << lpsName << "Lpu = (Space";
			stream << lpsName << "_LPU*) ";
			stream << "threadState->getCurrentLpu(Space_" << lpsName << ")" << stmtSeparator;
 
		} else {
			// In the other case, the LPS under concern must be a descendent LPS and epoch versions
			// of all LPU data parts must be updated. So we iterate over the LPUs and do the epoch
			// update on LPUs one by one.
			
			genLpuTraversalLoopBegin(stream, lpsName, indentation + 1);
			genCodeForArrayVarEpochUpdates(stream, lpsName, indentation + 2, arrayVarList);
			genLpuTraversalLoopEnd(stream, lpsName, indentation + 1);
		}
	}

	// advance scalar variables' versions all at once at the end
	if (scalarVarList->NumElements() > 0) {
		genCodeForScalarVarEpochUpdates(stream, indentation + 1, scalarVarList);
	}
	
	// invoke the nested subflow
	CompositeStage::generateInvocationCode(stream, indentation + 1, containerSpace);

	// end the scope
	stream << indentStr << "} // scope ends for epoch boundary\n";
}

