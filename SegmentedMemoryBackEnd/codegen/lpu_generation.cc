#include "lpu_generation.h"
#include "space_mapping.h"
#include "structure.h"
#include "../semantics/task_space.h"
#include "../utils/list.h"
#include "string.h"
#include "../utils/string_utils.h"
#include "../utils/hashtable.h"
#include "../utils/decorator_utils.h"	
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

	std::string paramSeparator = ", ";
	std::string stmtSeparator = ";\n";
	std::string indent = "\t";

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
				functionBody << paramSeparator << varName << dimensionParamName.str();
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

	std::string indent = "\t";
	std::string doubleIndent = "\t\t";
	std::string stmtSeparator = ";\n";
	std::string paramSeparator = ", ";

	// create a function header that takes three default arguments
	const char *lpsName = lps->getName();
	std::ostringstream functionHeader;
	functionHeader << "generateSpace" << lpsName << "Lpu(";
	functionHeader << "Space" << lpsName << "_LPU *lpu" << paramSeparator;
	functionHeader << '\n' << indent << doubleIndent;	
	functionHeader << "List<int*> *lpuIdList" << paramSeparator;
	functionHeader << '\n' << indent << doubleIndent;	
	functionHeader << "Hashtable<DataPartitionConfig*> *partConfigMap" << paramSeparator;
	functionHeader << '\n' <<  indent << doubleIndent;	
	functionHeader << "TaskData *taskData)";

	headerFile << "void " << functionHeader.str() << stmtSeparator;
	programFile << std::endl;
	programFile << "void " << initials << "::" << functionHeader.str() << " {\n";

	if (lps->getDimensionCount() > 0) {
		programFile << std::endl;
		programFile << indent << "int *lpuId = lpuIdList->Nth(lpuIdList->NumElements() - 1)";
		programFile << stmtSeparator;
		for (int i = 0; i < lps->getDimensionCount(); i++) {
			programFile << indent;
			programFile << "lpu->lpuId[" << i << "] = lpuId[" << i << "]";
			programFile << stmtSeparator;
		}
	}
	
	List<const char*> *localArrays = lps->getLocallyUsedArrayNames();
	for (int i = 0; i < localArrays->NumElements(); i++) {
		ArrayDataStructure *array = (ArrayDataStructure*) lps->getLocalStructure(localArrays->Nth(i));
		ArrayType *arrayType = (ArrayType*) array->getType();
		Type *elemType = arrayType->getTerminalElementType();
		const char *varName = array->getName();
		
		// retrieve the part configuration object for current array
		programFile << std::endl;
		programFile << indent << "DataPartitionConfig *" << varName << "Config = ";
		programFile << "partConfigMap->Lookup(";
		programFile << '"' << varName << "Space" << lpsName << "Config" << '"' << ")";
		programFile << stmtSeparator;
		
		// retrieve the part Id of the array for current LPU
		programFile << indent << "List<int*> *" << varName << "PartIdList = ";
		programFile << varName << "Config->generatePartIdList(lpuIdList)" << stmtSeparator;

		// set up the partition dimension information of the array part within the LPU
		programFile << indent << varName << "Config->updatePartDimensionInfo(";
		programFile << varName << "PartIdList" << paramSeparator;
		programFile << "&lpu->" << varName << "PartDims)" << stmtSeparator;

		// if the variable is not accessed in the LPS then there is no need for updating the underlying
		// pointer reference to data
		if (!array->getUsageStat()->isAccessed()) continue;

		// determine what LPS allocates the array
		Space *allocatorLps = array->getAllocator();
		bool allocatedElsewhere = (allocatorLps != lps);
		if (allocatedElsewhere) {
			// find how many step needs to be backtraced to reach the allocating LPS starting from 
			// the current LPS
			int steps = 0;
			Space *currentLps = lps;
			while (currentLps != allocatorLps) {
				currentLps = currentLps->getParent();
				steps++;
			}
			// find the partId list to locate the bigger data-part piece the current one is a part of
			programFile << indent << "List<int*> *" << varName << "AllocPartIdList = ";
			programFile << varName << "Config->generateSuperPartIdList(lpuIdList" << paramSeparator;
			programFile << steps << ")" << stmtSeparator;
		}

		// retrieve the data items list
		programFile << indent << "DataItems *" << varName << "Items = taskData->getDataItemsOfLps(";
		programFile << '"' << allocatorLps->getName() << '"' << paramSeparator;
		programFile << '"' << varName << '"' << ")" << stmtSeparator;
		
		// then retrieves the appropriate part from the item list
		programFile << indent << "DataPart *" << varName << "Part = ";
		programFile << varName << "Items->getDataPart(";
		if (allocatedElsewhere) programFile << varName << "AllocPartIdList)";
		else programFile << varName << "PartIdList)";
		programFile << stmtSeparator;

		// populate storage dimension information into the LPU object from the part
		programFile << indent << varName << "Part->getMetadata()->updateStorageDimension(";		 		 
		programFile << "&lpu->" << varName << "PartDims)" << stmtSeparator;

		// copy data from the data part object to the LPU object
		programFile << indent << "lpu->" << varName << " = ";
		programFile << "(" << elemType->getCType() << "*) " << varName << "Part->getData()";
		programFile << stmtSeparator;

		// if there are are multiple epoch dependent versions for current data then copy older versions 
		// in the LPU too
		int versionCount = array->getLocalVersionCount();
		for (int j = 1; j <= versionCount; j++) {
			programFile << indent << varName << "Part = ";
			programFile << varName << "Items->getDataPart(";
			if (allocatedElsewhere) programFile << varName << "AllocPartIdList";
			else programFile << varName << "PartIdList";
			programFile << paramSeparator << j << ")" << stmtSeparator;
			programFile << indent << "lpu->" << varName << "_lag_" << j << " = ";
			programFile << "(" << elemType->getCType() << "*) ";
			programFile << varName << "Part->getData()" << stmtSeparator;
		}
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
		generateLpuConstructionFunction(headerFile, programFile, initials, lps);	
	}

	headerFile.close();
	programFile.close();
}
