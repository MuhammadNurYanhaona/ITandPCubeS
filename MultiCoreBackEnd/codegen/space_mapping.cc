#include "space_mapping.h"
#include "structure.h"
#include "../semantics/task_space.h"
#include "../utils/list.h"
#include "string.h"
#include "../utils/string_utils.h"
#include "../utils/hashtable.h"
#include "../syntax/ast.h"
#include "../syntax/ast_expr.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <deque>

List<PPS_Definition*> *parsePCubeSDescription(const char *filePath) {

	List<PPS_Definition*> *list = new List<PPS_Definition*>;
	std::string line;
	std::ifstream pcubesfile(filePath);
	std::string separator1 = ":";
	std::string separator2 = " ";
	std::string separator3 = "(";
	List<std::string> *tokenList;

	if (!pcubesfile.is_open()) {
		std::cout << "could not open PCubeS specification file" << std::endl;
		std::exit(EXIT_FAILURE);
	}
	
	std::cout << "Parsing the PCubeS description-----------------------------------" << std::endl;

	while (std::getline(pcubesfile, line)) {
		// trim line and escape it if it is a comment	
		string_utils::trim(line);
		if (line.length() == 0) continue;
		std::string comments = "//";
		if (string_utils::startsWith(line, comments)) continue;

		// separate space number from its name and PPU count
		tokenList = string_utils::tokenizeString(line, separator1);
		std::string spaceNoStr = tokenList->Nth(0);
		std::string spaceNameStr = tokenList->Nth(1);

		// retrieve the space ID; also determine if current space represents CPU cores
		tokenList = string_utils::tokenizeString(spaceNoStr, separator2);
		std::string spaceNo = tokenList->Nth(1);
		int spaceId;
		bool coreSpace = false;
		if (string_utils::endsWith(spaceNo, '*')) {
			spaceNo = spaceNo.substr(0, spaceNo.length() - 1);
			spaceId = atoi(spaceNo.c_str());
			coreSpace = true;
		} else {
			spaceId = atoi(spaceNo.c_str());
		}

		// retrieve space name and PPU count
		tokenList = string_utils::tokenizeString(spaceNameStr, separator3);
		std::string spaceName = tokenList->Nth(0);
		std::string ppuCountStr = tokenList->Nth(1);
		int countEnd = ppuCountStr.find(')');
		int ppuCount = atoi(ppuCountStr.substr(0, countEnd).c_str());

		// create a PPS definition
		PPS_Definition *spaceDefinition = new PPS_Definition();
		spaceDefinition->id = spaceId;
		spaceDefinition->name = strdup(spaceName.c_str());
		spaceDefinition->units = ppuCount;
		spaceDefinition->coreSpace = coreSpace;
			
		// store the space definition in the list in top-down order
		int i = 0;	
		for (; i < list->NumElements(); i++) {
			if (list->Nth(i)->id > spaceId) continue;
			else break;	
		}
		list->InsertAt(spaceDefinition, i);
	}
	pcubesfile.close();

	for (int i = 0; i < list->NumElements(); i++) {
		printf("Space %s-%d-%d\n", list->Nth(i)->name, list->Nth(i)->id, list->Nth(i)->units);
	}
	return list;
}

MappingNode *parseMappingConfiguration(const char *taskName,
                const char *filePath,
                PartitionHierarchy *lpsHierarchy,
                List<PPS_Definition*> *pcubesConfig) {

	std::string line;
	std::ifstream mappingfile(filePath);
	std::string commentsDelimiter = "//";
	std::string newlineDelimiter = "\n";
	std::string mappingDelimiter = ":";
	List<std::string> *mappingList;
	List<std::string> *tokenList;
	std::string description;

	// open the mapping configuration file and read mapping configurations in a string
	if (mappingfile.is_open()) {
		while (std::getline(mappingfile, line)) {
			string_utils::trim(line);
			if (line.length() == 0) continue;
			if (string_utils::startsWith(line, commentsDelimiter)) continue;
			mappingList = string_utils::tokenizeString(line, commentsDelimiter);
			description.append(mappingList->Nth(0));
			description.append("\n");
		}
	} else {
		std::cout << "could not open the mapping file.\n";
		std::exit(EXIT_FAILURE);
	}
	std::cout << "Parsing the mapping configuration\n";

	// locate the mapping configuration of the mentioned task and extract it
	int taskConfigBegin = description.find(taskName);
	int mappingStart = description.find('{', taskConfigBegin);
	int mappingEnd = description.find('}', taskConfigBegin);
	std::string mapping = description.substr(mappingStart + 1, mappingEnd - mappingStart - 1);
	string_utils::trim(mapping);
	mappingfile.close();

	// create the root of the mapping hierarchy
	MapEntry *rootEntry = new MapEntry();
	Space *rootSpace = lpsHierarchy->getRootSpace();
	rootEntry->LPS = rootSpace;
	rootEntry->PPS = pcubesConfig->Nth(0);
	MappingNode *rootNode = new MappingNode();
	rootNode->parent = NULL;
	rootNode->mappingConfig = rootEntry;
	rootNode->children = new List<MappingNode*>;

	// parse individual lines and construct the mapping hierarchy
	mappingList = string_utils::tokenizeString(mapping, newlineDelimiter);
	Hashtable<MappingNode*> *mappingTable = new Hashtable<MappingNode*>;
	int i = 0;
	int mappingCount = mappingList->NumElements();
	int totalPPSes = pcubesConfig->NumElements();
	while (i < mappingCount) {
		// determine the LPS and PPS for the mapping
		std::string mapping = mappingList->Nth(i);
		tokenList = string_utils::tokenizeString(mapping, mappingDelimiter);
		std::string lpsStr = tokenList->Nth(0);
		char lpsId = lpsStr.at(lpsStr.length() - 1);
		Space *lps = lpsHierarchy->getSpace(lpsId);
		if (lps == NULL) {
			std::cout << "Logical space is not found in the code" << std::endl;
			std::exit(EXIT_FAILURE);
		}
		int ppsId = atoi(tokenList->Nth(1).c_str());
		PPS_Definition *pps = pcubesConfig->Nth(totalPPSes - ppsId);
		
		// create a mapping configuration object
		MapEntry *entry = new MapEntry();
		entry->LPS = lps;
		entry->PPS = pps;
		MappingNode *node = new MappingNode();
		node->parent = NULL;
		node->mappingConfig = entry;
		node->children = new List<MappingNode*>;
		mappingTable->Enter(lps->getName(), node, true);
		i++;
	
		// if the LPS is subpartitioned than map the subpartition into the same PPS
		if (lps->getSubpartition() != NULL) {
			MapEntry *subEntry = new MapEntry();
			subEntry->LPS = lps->getSubpartition();
			subEntry->PPS = pps;
			MappingNode *subNode = new MappingNode();
			subNode->parent = node;
			subNode->mappingConfig = subEntry;
			subNode->children = new List<MappingNode*>;
			mappingTable->Enter(subEntry->LPS->getName(), subNode, true);
		}
	}

	// correct the mapping hierarchy by setting parent and children references correctly
	MappingNode *currentNode = NULL;
	Iterator<MappingNode*> iterator = mappingTable->GetIterator();
	while ((currentNode = iterator.GetNextValue()) != NULL) {
		Space *parentLps = currentNode->mappingConfig->LPS->getParent();
		if (rootSpace == parentLps) {
			currentNode->parent = rootNode;
			rootNode->children->Append(currentNode);
		} else {
			MappingNode *parent = mappingTable->Lookup(parentLps->getName());
			currentNode->parent = parent;
			parent->children->Append(currentNode);
		}
	}

	// assign indexes to mapping nodes
	std::deque<MappingNode*> nodeQueue;
	nodeQueue.push_back(rootNode);
	int index = 0;
	while (!nodeQueue.empty()) {
		MappingNode *node = nodeQueue.front();
		node->index = index;
		index++;
		nodeQueue.pop_front();
		for (int i = 0; i < node->children->NumElements(); i++) {
			nodeQueue.push_back(node->children->Nth(i));
		}
	} 

	return rootNode;
}

void generateLPSConstants(const char *outputFile, MappingNode *mappingRoot) {
	std::ofstream programFile;
	programFile.open (outputFile, std::ofstream::out | std::ofstream::app);
  	if (programFile.is_open()) {
		programFile << "/*-----------------------------------------------------------------------------------" << std::endl;
		programFile << "constants for LPSes" << std::endl;
		programFile << "------------------------------------------------------------------------------------*/" << std::endl;
		std::deque<MappingNode*> nodeQueue;
		nodeQueue.push_back(mappingRoot);
		int spaceCount = 0;
		while (!nodeQueue.empty()) {
			spaceCount++;	
			MappingNode *node = nodeQueue.front();
			nodeQueue.pop_front();
			for (int i = 0; i < node->children->NumElements(); i++) {
				nodeQueue.push_back(node->children->Nth(i));
			}
			programFile << "const int Space_" << node->mappingConfig->LPS->getName();
			programFile << " = " << node->index << ';' << std::endl;	
		}
		programFile << "const int Space_Count = " << spaceCount << ';' << std::endl;
		programFile << std::endl; 
    		programFile.close();
  	} else {
		std::cout << "Unable to open output program file";
		std::exit(EXIT_FAILURE);
	}
}

void generatePPSCountConstants(const char *outputFile, List<PPS_Definition*> *pcubesConfig) {
	std::ofstream programFile;
	programFile.open (outputFile, std::ofstream::out | std::ofstream::app);
  	if (programFile.is_open()) {
		programFile << "/*-----------------------------------------------------------------------------------" << std::endl;
		programFile << "constants for PPS counts" << std::endl;
		programFile << "------------------------------------------------------------------------------------*/" << std::endl;
		PPS_Definition *pps = pcubesConfig->Nth(0);
		int prevSpaceId = pps->id;
		programFile << "const int Space_" << pps->id << "_PPUs";
		programFile << " = " << pps->units << ';' << std::endl;
		for (int i = 1; i < pcubesConfig->NumElements(); i++) {
			pps = pcubesConfig->Nth(i);
			programFile << "const int Space_" << pps->id;
			programFile << "_Par_" << prevSpaceId << "_PPUs";
			programFile << " = " << pps->units << ';' << std::endl;
			prevSpaceId = pps->id;
		}
		programFile << std::endl; 
    		programFile.close();
  	} else {
		std::cout << "Unable to open output program file";
		std::exit(EXIT_FAILURE);
	}
}

List<PartitionParameterConfig*> *generateLPUCountFunction(std::ofstream &headerFile, 
                std::ofstream &programFile, 
                const char *initials,
                Space *space, 
                List<Identifier*> *partitionArgs) {

	std::string defaultParameter = "ppuCount";
	std::string parameterSeparator = ", ";
	std::string statementSeparator = ";\n";
	std::string statementIndent = "\t";

	// This corresponds to the count functions definitions we have in backend_partition_mgmt file.
	// We have to find a better mechanism to store constants like these in the future.	
	std::string defaultPartitionFnSuffix = "_partitionCount(";

	List<PartitionParameterConfig*> *paramConfigList = new List<PartitionParameterConfig*>;
	CoordinateSystem *coordSys = space->getCoordinateSystem();
	int dimensionality = space->getDimensionCount();
	std::ostringstream functionHeader;
	functionHeader << "int " << defaultParameter;
	std::ostringstream functionBody;
	functionBody << statementIndent; 
	functionBody << "int *count = new int[" << dimensionality << "]";
	functionBody << statementSeparator;

	// iterate over the dimension of the space and for each dimension pick an array partition to
	// be used for partition-count calculation
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
			
			// add the dimension parameter in the function header
			functionHeader << parameterSeparator << "Dimension " << dimensionParamName.str();
			PartitionParameterConfig *paramConfig = new PartitionParameterConfig;
			paramConfig->arrayName = array->getName();
			paramConfig->dimensionNo = arrayDim;
			paramConfig->partitionArgsIndexes = new List<int>;
			paramConfigList->Append(paramConfig);
			
			PartitionFunctionConfig *partFn = array->getPartitionSpecForDimension(arrayDim);
			std::ostringstream fnCall;
			fnCall << partFn->getName() << defaultPartitionFnSuffix;
			
			// dimension and the ppu count are two default parameters in the cout routine
			// of any partition function.	
			fnCall << dimensionParamName.str() << parameterSeparator << defaultParameter;
		
			// Currently we only support one dividing argument par partition function. Later
			// we may lift this restriction. Then the logic here will be different
			Node *dividingArg = partFn->getArgsForDimension(arrayDim)->getDividingArg();
			if (dividingArg != NULL) {
				IntConstant *intConst = dynamic_cast<IntConstant*>(dividingArg);
				// if the argument is a constant just apply it as an argument
				if (intConst != NULL) {
					fnCall << parameterSeparator << intConst->getValue();
				// otherwise pass the argument and update the parameter config list 
				} else {
					const char *paramName = ((Identifier *) dividingArg)->getName();
					fnCall << parameterSeparator << paramName;
					functionHeader << parameterSeparator << "int " << paramName;
					// find the index of the parameter in partition section's argument
					// list and store this information
					bool paramFound = false;
					for (int k = 0; k < partitionArgs->NumElements(); k++) {
						const char *argName = partitionArgs->Nth(k)->getName();
						if (strcmp(argName, paramName) == 0) {
							paramConfig->partitionArgsIndexes->Append(k);
							paramFound = true;
							break;
						}
					} 
					if (!paramFound) {
						std::cout << "could not find matching argument in ";
						std::cout << "partition section." << std::endl;
					}	
				}
			}

			fnCall << ")";
			functionBody << statementIndent; 
			functionBody << "count[" << i - 1 << "] = " << fnCall.str();
			functionBody << statementSeparator;	
			break;
		}
	}
	functionBody << statementIndent << "return count" << statementSeparator;

	// write the function signature in the header file
	headerFile << "int *getLPUsCountOfSpace" << space->getName();
	headerFile << "(" << functionHeader.str() << ");\n"; 

	// write the function specification in the program file
	programFile << "int *" << initials << "::getLPUsCountOfSpace" << space->getName();
	programFile << "(" << functionHeader.str() << ") {\n"; 
	programFile << functionBody.str() << "}" << std::endl;
	
	return paramConfigList;
}

Hashtable<List<PartitionParameterConfig*>*> *generateLPUCountFunctions(const char *headerFileName,
                const char *programFileName, 
                const char *initials,
                MappingNode *mappingRoot, 
                List<Identifier*> *partitionArgs) {

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
	headerFile << "\n/*-----------------------------------------------------------------------------------\n";
	headerFile << "functions for retrieving partition counts in different LPSes\n";
	headerFile << "------------------------------------------------------------------------------------*/\n";
	programFile << "/*-----------------------------------------------------------------------------------\n";
	programFile << "functions for retrieving partition counts in different LPSes\n";
	programFile << "------------------------------------------------------------------------------------*/\n\n";

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
		List<PartitionParameterConfig*> *paramConfigList 
			= generateLPUCountFunction(headerFile, 
					programFile, initials, lps, partitionArgs);
		paramTable->Enter(lps->getName(), paramConfigList, true);
		programFile << std::endl;
	}
	
	headerFile.close();
	programFile.close(); 
	return paramTable;
}

List<int> *generateGetArrayPartForLPURoutine(Space *space, 
                ArrayDataStructure *array,
                std::ostream &headerFile,  
                std::ofstream &programFile, 
                const char *initials,
                List<Identifier*> *partitionArgs) {
	
	std::string parameterSeparator = ", ";
	std::string statementSeparator = ";\n";
	std::string statementIndent = "\t";
	
	List<int> *argIndexList = new List<int>;
	int dimensionCount = array->getDimensionality(); 
	const char *arrayName = array->getName();
	CoordinateSystem *coordSys = space->getCoordinateSystem();
	int dimensionality = space->getDimensionCount();
	List<const char*> *argNameList = new List<const char*>;
	

	// create two variables for parent LPU (i.e. the LPU of the parent space that will be passed as an 
	// argument in the function call) and one for the current LPU under concern that will be generated
	// and returned by the routine
	std::ostringstream parentVarStr;
	parentVarStr << arrayName << "ParentLpuDims";
	std::string parentVar = parentVarStr.str();
	std::ostringstream currentVarStr;
	currentVarStr << arrayName << "LpuDims";
	std::string currentVar = currentVarStr.str();
	
	// set the parameters default to all get-LPU routines	
	std::ostringstream functionHeader;
	functionHeader << "get" << arrayName << "PartForSpace" << space->getName() << "Lpu(";
	functionHeader << "PartDimension *" << currentVar << parameterSeparator;
	functionHeader << std::endl << statementIndent << statementIndent;
	functionHeader << "PartDimension *" << parentVar << parameterSeparator;
	functionHeader << std::endl << statementIndent << statementIndent;
	functionHeader << "int *lpuCount";
	functionHeader << parameterSeparator << "int *lpuId";
	std::ostringstream functionBody;
	functionBody << " {\n";

	for (int i = 0; i < dimensionCount; i++) {
		PartitionFunctionConfig *partConfig = array->getPartitionSpecForDimension(i + 1);
		// If the partition config along this dimension is null then this dimension is replicated
		// from the parent. Therefore, we can just copy parent's dimension information in current 
		// metadata object	
		if (partConfig == NULL) {
			functionBody << statementIndent;
			functionBody << currentVar << '[' << i << "] = " << parentVar << '[' << i << ']';
			functionBody << statementSeparator;
		// Otherwise, we need to allocate a new metadata variable for this dimension; invoke the
		// mentioned partition function; and set up other references properly.
		} else {
			// copy parent's storage dimension into current LPU's storage dimension
			functionBody << statementIndent;
			functionBody << currentVar << '[' << i << "].storage = ";
			functionBody << parentVar << '[' << i << "].storage";
			functionBody << statementSeparator;

			// assign the result of partition function invocation to current LPU's partition
			// dimension and register any partition argument needs to be passed
			functionBody << statementIndent;
			functionBody << currentVar << '[' << i << "].partition = ";
			functionBody << partConfig->getName() << "_getRange(";
			functionBody <<	parentVar << '[' << i << "].partition" << parameterSeparator;
			functionBody << std::endl << statementIndent << statementIndent << statementIndent;
		
			// determine which LPU-Count and ID should be used by determining the aligment of
			// concerned array's current dimension with the dimension of the space	
			int lpuDimIndex = 0;
			bool dimDidntMatch = true;
			for (int j = 1; j <= dimensionality; j++) {
				Coordinate *coord = coordSys->getCoordinate(j);
				Token *token = coord->getTokenForDataStructure(arrayName);
				if (token->getDimensionId() == i + 1) {
					lpuDimIndex = j - 1;
					dimDidntMatch = false;
					break;
				}
			}
			if (dimDidntMatch) std::cout << "problem generating LPUs in Space " 
					<< space->getName() << " due to variable " << arrayName << std::endl;

			// add two default parameters to the getRange function call
			functionBody << "lpuCount" << '[' << lpuDimIndex << ']';
			functionBody << parameterSeparator << "lpuId" << '[' << lpuDimIndex << ']';
			// check for dividing and padding arguments and append more parameters to the function
			// call accordingly
			DataDimensionConfig *argConfig = partConfig->getArgsForDimension(i + 1);
			Node *dividingArg = argConfig->getDividingArg();
			if (dividingArg != NULL) {
				IntConstant *intConst = dynamic_cast<IntConstant*>(dividingArg);
				if (intConst != NULL) {
					functionBody << parameterSeparator << intConst->getValue();
				} else {
					Identifier *arg = (Identifier*) dividingArg;
					argNameList->Append(arg->getName());
					functionBody << parameterSeparator << arg->getName();
					functionHeader << parameterSeparator << "int " << arg->getName();
				}
				if (partConfig->doesSupportGhostRegion()) {
					// process front padding arg
					Node *paddingArg = argConfig->getFrontPaddingArg();
					if (paddingArg == NULL) {
						// if no padding is provided then a ZERO must be passed to the
						// called function
						functionBody << parameterSeparator << "0";
					} else {
						IntConstant *intConst = dynamic_cast<IntConstant*>(paddingArg);
						if (intConst != NULL) {
							functionBody << parameterSeparator << intConst->getValue();
						} else {
							Identifier *arg = (Identifier*) paddingArg;
							argNameList->Append(arg->getName());
							functionBody << parameterSeparator << arg->getName();
							functionHeader << parameterSeparator << "int " << arg->getName();
						}
					}
					// process back padding arg in exact same manner
					paddingArg = argConfig->getBackPaddingArg();
					if (paddingArg == NULL) {
						functionBody << parameterSeparator << "0";
					} else {
						IntConstant *intConst = dynamic_cast<IntConstant*>(paddingArg);
						if (intConst != NULL) {
							functionBody << parameterSeparator << intConst->getValue();
						} else {
							Identifier *arg = (Identifier*) paddingArg;
							argNameList->Append(arg->getName());
							functionBody << parameterSeparator << arg->getName();
							functionHeader << parameterSeparator << "int " << arg->getName();
						}
					}
				}
			}
			functionBody << ")" << statementSeparator;
		}
	}
	functionHeader << ")";
	functionBody << "}\n";
	
	// write function signature in the header file and the specification in the program file
	headerFile << "void " << functionHeader.str() << statementSeparator;
	programFile << "void " << initials << "::" << functionHeader.str() << functionBody.str();

	// get the list of argument index from the used argument name list 
	for (int i = 0; i < argNameList->NumElements(); i++) {
		const char *argName = argNameList->Nth(i);
		bool paramFound = false;
		for (int k = 0; k < partitionArgs->NumElements(); k++) {
			const char *paramName = partitionArgs->Nth(k)->getName();
			if (strcmp(argName, paramName) == 0) {
				argIndexList->Append(k);
				paramFound = true;
				break;
			}
		} 
		if (!paramFound) std::cout << "some arguments in the getRange function is in error.";
	}

	return argIndexList;
}

Hashtable<List<int>*> *generateAllGetPartForLPURoutines(const char *headerFileName,
                const char *programFileName,
                const char *initials,
                MappingNode *mappingRoot,
                List<Identifier*> *partitionArgs) {
	
	std::cout << "Generating founctions for determining array parts" << std::endl;

	// if the output files cannot be opened then return
	std::ofstream programFile, headerFile;
	programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
	headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
  	if (!programFile.is_open() || !headerFile.is_open()) {
		std::cout << "Unable to open output header/program file";
		std::exit(EXIT_FAILURE);
	}

	// add a common comments for all these functions
	headerFile << "\n/*-----------------------------------------------------------------------------------\n";
	headerFile << "functions for getting data ranges along different dimensions of an LPU\n";
	headerFile << "-----------------------------------------------------------------------------------*/\n";
	programFile << "/*-----------------------------------------------------------------------------------\n";
	programFile << "functions for getting data ranges along different dimensions of an LPU\n";
	programFile << "-----------------------------------------------------------------------------------*/\n\n";

	Hashtable<List<int>*> *paramTable = new Hashtable<List<int>*>;		
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
		List<const char*> *arrayNameList = lps->getLocallyUsedArrayNames();
		for (int i = 0; i < arrayNameList->NumElements(); i++) {
			ArrayDataStructure *array = (ArrayDataStructure*) lps->getStructure(arrayNameList->Nth(i));
			// note that the second condition is for arrays inherited by a subpartitioned LPS
			if (!(array->isPartitioned() && array->getSpace() == lps)) continue;
			List<int> *argList = generateGetArrayPartForLPURoutine(lps, 
					array, headerFile, programFile, initials, partitionArgs);
			std::ostringstream entryName;
			entryName << lps->getName() << "_" << array->getName();
			paramTable->Enter(entryName.str().c_str(), argList, true);
			programFile << std::endl;
		}	
	}

	headerFile << std::endl;
	headerFile.close();
	programFile.close(); 
	
	return paramTable;
}
