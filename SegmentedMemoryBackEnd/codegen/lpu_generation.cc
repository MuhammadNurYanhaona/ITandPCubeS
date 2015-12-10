#include "lpu_generation.h"
#include "space_mapping.h"
#include "structure.h"
#include "../semantics/task_space.h"
#include "../utils/list.h"
#include "string.h"
#include "../utils/string_utils.h"
#include "../utils/hashtable.h"
#include "../utils/decorator_utils.h"
#include "../utils/code_constant.h"	
#include "../syntax/ast.h"
#include "../syntax/ast_expr.h"
#include "../static-analysis/usage_statistic.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <deque>

List<PartitionParameterConfig*> *generateLPUCountFunction(std::ofstream &headerFile, 
                std::ofstream &programFile, 
                const char *initials, Space *space) {

	List<PartitionParameterConfig*> *paramConfigList = new List<PartitionParameterConfig*>;
	CoordinateSystem *coordSys = space->getCoordinateSystem();
	int dimensionality = space->getDimensionCount();

	std::ostringstream functionHeader;
	functionHeader << "getLPUsCountOfSpace" << space->getName() << "(";
	functionHeader << "Hashtable<DataPartitionConfig*> *partConfigMap";
	
	std::ostringstream functionBody;
	functionBody << " {\n";
	functionBody << indent << "int *count = new int[" << dimensionality << "]" << stmtSeparator;

	Hashtable<const char*> *configRetrieveMap = new Hashtable<const char*>;

	// iterate over the dimension of the space and for each dimension pick an array partition to be used for 
	// partition-count calculation
	for (int i = 1; i <= dimensionality; i++) {
		Coordinate *coord = coordSys->getCoordinate(i);
		List<Token*> *tokenList = coord->getTokenList();

		for (int j = 0; j < tokenList->NumElements(); j++) {
			// if it is a replicated token then skip this data structure
			Token *token = tokenList->Nth(j);
			if (token->isWildcard()) continue;

			// otherwise use the data structure partition information to determine LPU count along
			// current dimension then break the loop
			ArrayDataStructure *array = (ArrayDataStructure*) token->getData();
			int arrayDim = token->getDimensionId();
			std::ostringstream dimensionParamName;
			dimensionParamName << array->getName() << "Dim" << arrayDim;
			const char *varName = array->getName();
			const char *lpsName = space->getName();

			// check if the array has been partitioned before in any other LPSes
			bool dimPartitionedBefore = array->isPartitionedAlongDimensionEarlier(arrayDim);
			if (dimPartitionedBefore) {
				// add the dimension parameter in the function header and keep track of the 
				// usage in the argument list for later use
				functionHeader << paramSeparator << "Dimension " << dimensionParamName.str();
				PartitionParameterConfig *paramConfig = new PartitionParameterConfig;
				paramConfig->arrayName = varName;
				paramConfig->dimensionNo = arrayDim;
				paramConfigList->Append(paramConfig);
			}

			// retrieve the data configuration object for the concerned array
			if (configRetrieveMap->Lookup(varName) == NULL) {
				functionBody << indent << "DataPartitionConfig *" << varName << "Config = ";
                        	functionBody << "partConfigMap->Lookup(";
                        	functionBody << '"' << varName << "Space" << lpsName << "Config" << '"' << ")";
                        	functionBody << stmtSeparator;
				configRetrieveMap->Enter(varName, varName);
			}

			// call function in the configuration object to determine the number of partition along
			// current dimension and add that in the LPU count
			functionBody << indent << "count[" << i - 1 << "] = ";
			functionBody << varName << "Config->getPartsCountAlongDimension(";
			functionBody << arrayDim - 1;
			if (dimPartitionedBefore) {
				functionBody << paramSeparator << "&" << dimensionParamName.str();
			}
			functionBody << ")";	
			functionBody << stmtSeparator;	
			break;
		}
	}
	functionBody << indent << "return count" << stmtSeparator;
	functionBody << "}\n";
	functionHeader << ")";

	// write the function signature in the header file
	headerFile << "int *" << functionHeader.str() << stmtSeparator; 

	// write the function specification in the program file
	programFile << "int *" << initials << "::" << functionHeader.str(); 
	programFile << functionBody.str();
	
	return paramConfigList;
}

Hashtable<List<PartitionParameterConfig*>*> *generateLPUCountFunctions(const char *headerFileName,
                const char *programFileName, 
                const char *initials, MappingNode *mappingRoot) {

	std::cout << "Generating LPU count founctions" << std::endl;

	// if the output files cannot be opened then return
	std::ofstream programFile, headerFile;
	programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
	headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
  	if (!programFile.is_open() || !headerFile.is_open()) {
		std::cout << "Unable to open output header/program file";
		std::exit(EXIT_FAILURE);
	}

	// add a common comments for all these functions
	const char *message = "functions for retrieving partition counts in different LPSes";
	decorator::writeSectionHeader(headerFile, message);
	headerFile << std::endl;
	decorator::writeSectionHeader(programFile, message);

	// iterate over all LPSes and generate a function for each partitioned LPS
	Hashtable<List<PartitionParameterConfig*>*> *paramTable 
			= new Hashtable<List<PartitionParameterConfig*>*>;		
	std::deque<MappingNode*> nodeQueue;
	nodeQueue.push_back(mappingRoot);
	while (!nodeQueue.empty()) {
		MappingNode *node = nodeQueue.front();
		nodeQueue.pop_front();
		for (int i = 0; i < node->children->NumElements(); i++) {
			nodeQueue.push_back(node->children->Nth(i));
		}
		Space *lps = node->mappingConfig->LPS;
		if (lps->getDimensionCount() == 0) continue;
		
		programFile << std::endl;
		List<PartitionParameterConfig*> *paramConfigList 
			= generateLPUCountFunction(headerFile, programFile, initials, lps);
		paramTable->Enter(lps->getName(), paramConfigList, true);
	}
	
	headerFile.close();
	programFile.close(); 
	return paramTable;
}

void generateLpuConstructionFunction(std::ofstream &headerFile, 
                std::ofstream &programFile, 
                const char *initials, Space *lps) {

	// create a function header that takes three default arguments
	const char *lpsName = lps->getName();
	std::ostringstream functionHeader;
	functionHeader << "generateSpace" << lpsName << "Lpu(";
	functionHeader << "ThreadState *threadState" << paramSeparator;
	functionHeader << '\n' << indent << doubleIndent;	
	functionHeader << "Hashtable<DataPartitionConfig*> *partConfigMap" << paramSeparator;
	functionHeader << '\n' <<  indent << doubleIndent;	
	functionHeader << "TaskData *taskData)";

	headerFile << "void " << functionHeader.str() << stmtSeparator;
	programFile << std::endl;
	programFile << "void " << initials << "::" << functionHeader.str() << " {\n";

	// retrieve the LPU correspond to the LPS under concern from the thread state; also retrieve its Id and
	// and LPU count for LPS
	programFile << std::endl;
	programFile << indent << "int *lpuId = threadState->getCurrentLpuId(Space_";
	programFile << lpsName << ")" << stmtSeparator;
	programFile << indent << "int *lpuCounts = threadState->getLpuCounts(Space_";
	programFile << lpsName << ")" << stmtSeparator;
	programFile << indent << "Space" << lpsName << "_LPU *lpu = (Space" << lpsName << "_LPU*) ";
        programFile << "threadState->getCurrentLpu(Space_" << lpsName; 
	programFile << paramSeparator <<  "true" << ")" << stmtSeparator;

	// retrieve LPU Id chain that shows how the LPU for the current LPS has been reached hierarchically from its
	// ancestor LPSes; this information will be needed to identify data parts of the LPU
	programFile << indent << "List<int*> *lpuIdChain = threadState->getLpuIdChainWithoutCopy(";
	programFile << "Space_" << lpsName << paramSeparator;
	programFile << "Space_" << lps->getRoot()->getName() << ")" << stmtSeparator;  

	// update LPU Id if the current LPS is partitioned 
	if (lps->getDimensionCount() > 0) {
		programFile << std::endl;	
		for (int i = 0; i < lps->getDimensionCount(); i++) {
			programFile << indent;
			programFile << "lpu->lpuId[" << i << "] = lpuId[" << i << "]";
			programFile << stmtSeparator;
		}
	}
	
	// keep a list of LPS names for whose LPUs will be retrieved in the LPU generation proceess for current LPS
	List<const char*> *parentLpsNames = new List<const char*>;
	
	List<const char*> *localArrays = lps->getLocallyUsedArrayNames();
	for (int i = 0; i < localArrays->NumElements(); i++) {
		ArrayDataStructure *array = (ArrayDataStructure*) lps->getLocalStructure(localArrays->Nth(i));
		ArrayType *arrayType = (ArrayType*) array->getType();
		Type *elemType = arrayType->getTerminalElementType();
		const char *varName = array->getName();

		// if the current LPS is a subpartition and the parent LPS is the holder of current array reference
		// then just assign every information from the parent to the current LPU regarding this array
		if (array->getSpace() != lps && lps->isSubpartitionSpace()) {
			programFile << std::endl;
			const char *parentLpsName = array->getSpace()->getName();
			if (!string_utils::contains(parentLpsNames, parentLpsName)) {
				parentLpsNames->Append(parentLpsName);
				programFile << indent << "Space" << parentLpsName;
				programFile << "_LPU *space" << parentLpsName << "Lpu = (Space" << parentLpsName; 
				programFile << "_LPU*) threadState->getCurrentLpu(Space_";
				programFile << parentLpsName << ")" << stmtSeparator;
			}
			programFile << indent << "lpu->" << varName << " = space" << parentLpsName;
			programFile << "Lpu->" << varName << stmtSeparator;
			int dimensionality = array->getDimensionality();
			for (int d = 0; d < dimensionality; d++) {
				programFile << indent << "lpu->" << varName << "PartDims[" << d;
				programFile << "] = space" << parentLpsName;
				programFile << "Lpu->" << varName << "PartDims[" << d << "]" << stmtSeparator;
			}
			continue;
		}
		
		// retrieve the part configuration object for current array
		programFile << std::endl;
		programFile << indent << "DataPartitionConfig *" << varName << "Config = ";
		programFile << "partConfigMap->Lookup(";
		programFile << '"' << varName << "Space" << lpsName << "Config" << '"' << ")";
		programFile << stmtSeparator;

		// get the parent data structure reference holding LPS for the array; note that there is always a
		// parent/source reference for any structure due to the presence of the root LPS 
		DataStructure *source = array->getSource();
		Space *parentLps = source->getSpace();
		const char *parentLpsName = parentLps->getName();

		// if the LPU correspond to the parent LPS has not been retrieved previously then retrieve it
		if (!string_utils::contains(parentLpsNames, parentLpsName)) {
			parentLpsNames->Append(parentLpsName);
			programFile << indent << "Space" << parentLpsName;
			programFile << "_LPU *space" << parentLpsName << "Lpu = (Space" << parentLpsName; 
			programFile << "_LPU*) threadState->getCurrentLpu(Space_";
			programFile << parentLpsName << ")" << stmtSeparator;
		}
	
		// get the parent part dimension information
		programFile << indent << "PartDimension *" << varName << "ParentPartDims = ";
		programFile << "space" << parentLpsName << "Lpu->" << varName << "PartDims" << stmtSeparator; 	

		// set up the partition dimension information of the array part within the LPU
		programFile << indent << varName << "Config->updatePartDimensionInfo(";
		programFile << "lpuId" << paramSeparator;
		programFile << "lpuCounts" << paramSeparator;
		programFile << "lpu->" << varName << "PartDims" << paramSeparator; 
		programFile << varName << "ParentPartDims" << ")" << stmtSeparator;

		// if the variable is not accessed in the LPS then there is no need for updating the underlying
		// pointer reference to data
		if (!array->getUsageStat()->isAccessed()) continue;

		// TODO note that a segmented PPU may need to determine the LPU Ids of LPUs multiplexed to other 
		// segmented PPUs for the sake of communication. Now the process of retrieving LPU ids involves
		// generation of the LPUs -- only their metadata is needed though. Therefore, we can include a 
		// checking in the LPU generation process regarding whether the full LPU with all the data parts
		// references pointing to correct memory address is needed, as it will be for the LPUs that the
		// concerned segmented PPU will execute later, or a metadata only version of the LPU is sufficient.
		// For now, we are using the absense of a valid task-data reference as an indicator for metadata
		// only LPUs. This process should be changed alongside an update to the LPU generation process
		// that the future developers should investigate for optimization or even for a better design.
		programFile << indent << "if (taskData != NULL) {\n";

		// determine what LPS allocates the array; if it is different than the current LPS then determine the
		// number of steps need to be traced back to generate the ID for the allocated part
		Space *allocatorLps = array->getAllocator();
		const char *allocatorLpsName = allocatorLps->getName();	
		bool allocatedElsewhere = (allocatorLps != lps);
		int allocationJump = 0;
		if (allocatedElsewhere) {
			Space *currentSpace = lps;
			while (currentSpace != allocatorLps) {
				allocationJump++;
				currentSpace = currentSpace->getParent();
			}   
		}
		
		// retrieve the iterator reference for the part and from it a template part-Id object
		programFile << doubleIndent << "PartIterator *iterator = ";
		programFile << "threadState->getIterator(Space_" << allocatorLpsName << paramSeparator;
		programFile << "\"" << varName << "\")" << stmtSeparator;
		programFile << doubleIndent << "List<int*> *partId = ";
		programFile << "iterator->getPartIdTemplate()" << stmtSeparator;

		// retrieve the hierarchical part Id of the array for current LPU so that the storage instance can 
		// be identified
		if (!allocatedElsewhere) {
               		programFile << doubleIndent << varName;
			programFile << "Config->generatePartId(lpuIdChain" << paramSeparator;
			programFile << "partId)" << stmtSeparator;
		} else {
               		programFile << doubleIndent << varName;
			programFile << "Config->generateSuperPartId(lpuIdChain" << paramSeparator;
			programFile << allocationJump << paramSeparator;
			programFile << "partId)" << stmtSeparator;
		}

		// retrieve the data items list
		programFile << doubleIndent << "DataItems *" << varName << "Items = taskData->getDataItemsOfLps(";
		programFile << '"' << allocatorLpsName << '"' << paramSeparator;
		programFile << '"' << varName << '"' << ")" << stmtSeparator;
		
		// then retrieves the appropriate part from the item list
		programFile << doubleIndent << "DataPart *" << varName << "Part = ";
		programFile << varName << "Items->getDataPart(partId" << paramSeparator;
		programFile << "iterator)" << stmtSeparator;

		// populate storage dimension information into the LPU object from the part
		programFile << doubleIndent << varName << "Part->getMetadata()->updateStorageDimension(";		 		 
		programFile << "lpu->" << varName << "PartDims)" << stmtSeparator;

		// copy data from the data part object to the LPU object
		programFile << doubleIndent << "lpu->" << varName << " = ";
		programFile << "(" << elemType->getCType() << "*) " << varName << "Part->getData()";
		programFile << stmtSeparator;

		// if there are are multiple epoch dependent versions for current data then copy older versions 
		// in the LPU too
		int versionCount = array->getLocalVersionCount();
		for (int j = 1; j <= versionCount; j++) {
			programFile << doubleIndent << varName << "Part = ";
			programFile << varName << "Items->getDataPart(partId" << paramSeparator;
			programFile << j << paramSeparator;
			programFile << "iterator)" << stmtSeparator;
			programFile << doubleIndent << "lpu->" << varName << "_lag_" << j << " = ";
			programFile << "(" << elemType->getCType() << "*) ";
			programFile << varName << "Part->getData()" << stmtSeparator;
		}
		programFile << indent << "}\n";
	}
	
	programFile << "}\n";
}

void generateAllLpuConstructionFunctions(const char *headerFileName,
                const char *programFileName, 
		const char *initials, MappingNode *mappingRoot) {

	std::cout << "Generating LPU construction founctions" << std::endl;

	// if the output files cannot be opened then return
	std::ofstream programFile, headerFile;
	programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
	headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
  	if (!programFile.is_open() || !headerFile.is_open()) {
		std::cout << "Unable to open output header/program file";
		std::exit(EXIT_FAILURE);
	}

	// add a common comments for all these functions
	const char *message = "functions for generating LPUs given LPU Ids";
	decorator::writeSectionHeader(headerFile, message);
	headerFile << std::endl;
	decorator::writeSectionHeader(programFile, message);

	// iterate over all LPSes and generate a function for each LPS
	Hashtable<List<PartitionParameterConfig*>*> *paramTable = new Hashtable<List<PartitionParameterConfig*>*>;		
	std::deque<MappingNode*> nodeQueue;
	for (int i = 0; i < mappingRoot->children->NumElements(); i++) {
		nodeQueue.push_back(mappingRoot->children->Nth(i));
	}
	while (!nodeQueue.empty()) {
		MappingNode *node = nodeQueue.front();
		nodeQueue.pop_front();
		for (int i = 0; i < node->children->NumElements(); i++) {
			nodeQueue.push_back(node->children->Nth(i));
		}
		Space *lps = node->mappingConfig->LPS;
		generateLpuConstructionFunction(headerFile, programFile, initials, lps);	
	}

	headerFile.close();
	programFile.close();
}
