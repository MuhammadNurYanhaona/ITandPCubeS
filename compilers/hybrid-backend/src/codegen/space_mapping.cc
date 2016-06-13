#include "space_mapping.h"
#include "../runtime/structure.h"
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

//--------------------------------------------------------------- PPS Definition ------------------------------------------------------/

PPS_Definition::PPS_Definition() {
	id = -1;
	name = NULL;
	units = 0;
	coreSpace = false;
	segmented = false;
	physicalUnit = false;
	gpuTransition = false;
	passive = false;
}

void PPS_Definition::print(int indentLevel) {
	std::ostringstream indent;
	for (int i = 0; i < indentLevel; i++) indent << '\t';
	std::cout << indent.str() << name << ": Space " << id << '\n';
	indent << '\t';
	std::cout << indent.str() << "Units: " << units << '\n';
	if (coreSpace) std::cout << indent.str() << "Computation Core\n";
	if (segmented) std::cout << indent.str() << "Segmented Memory\n";	
	if (physicalUnit) std::cout << indent.str() << "Physical Unit\n";
	if (gpuTransition) std::cout << indent.str() << "Top-most GPU Level\n";
}

//---------------------------------------------------------------- PCubeS Model -------------------------------------------------------/

PCubeSModel::PCubeSModel(int modelType) {
	this->modelType = modelType;
	ppsDefinitions = new List<PPS_Definition*>();
	modelName = NULL;
}

PCubeSModel::PCubeSModel(int modelType, List<PPS_Definition*> *ppsDefinitions) {
	this->modelType = modelType;
	this->ppsDefinitions = ppsDefinitions;
	modelName = NULL;
}

void PCubeSModel::setPassivePpsThreshold(int thresholdPpsId) {
	for (int i = 0; i < ppsDefinitions->NumElements(); i++) {
		PPS_Definition *pps = ppsDefinitions->Nth(i);
		pps->passive = (pps->id < thresholdPpsId);
	}
}

void PCubeSModel::print(int indentLevel) {
	for (int i = 0; i < indentLevel; i++) std::cout << '\t';
	if (modelName != NULL) {
		std::cout << "Model: " << modelName;
	} else {
		std::cout << "Model: Default Model";
	}
	std::cout << "\n";
	for (int i = 0; i < ppsDefinitions->NumElements(); i++) {
		ppsDefinitions->Nth(i)->print(indentLevel + 1);
	}
}

int PCubeSModel::getGpuTransitionSpaceId() {
	if (modelType != PCUBES_MODEL_TYPE_HYBRID) {
		std::cout << "Cannot get SM count from a host-only PCubeS model\n";
		std::exit(EXIT_FAILURE);
	}
	int gpuLevelIndex = ppsDefinitions->NumElements() - 3;
	return ppsDefinitions->Nth(gpuLevelIndex)->id;
}

int PCubeSModel::getSMCount() {
	if (modelType != PCUBES_MODEL_TYPE_HYBRID) {
		std::cout << "Cannot get SM count from a host-only PCubeS model\n";
		std::exit(EXIT_FAILURE);
	}
	int smLevelIndex = ppsDefinitions->NumElements() - 2;
	return ppsDefinitions->Nth(smLevelIndex)->units;
}

int PCubeSModel::getWarpCount() {
	if (modelType != PCUBES_MODEL_TYPE_HYBRID) {
		std::cout << "Cannot get WARP count from a host-only PCubeS model\n";
		std::exit(EXIT_FAILURE);
	}
	int warpLevelIndex = ppsDefinitions->NumElements() - 1;
	return ppsDefinitions->Nth(warpLevelIndex)->units;
}

//---------------------------------------------------------- Utility Library Functions ------------------------------------------------/

List<PCubeSModel*> *parsePCubeSDescription(const char *filePath) {

	List<PCubeSModel*> *modelList = new List<PCubeSModel*>;
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

	List<PPS_Definition*> *ppsList = new List<PPS_Definition*>;
	int currentModelNo = 0;
	std::string modelName("Default");
	bool hybridModel = false;

	while (std::getline(pcubesfile, line)) {

		// trim line and escape it if it is a comment	
		string_utils::trim(line);
		if (line.length() == 0) continue;
		std::string comments = "//";
		if (string_utils::startsWith(line, comments)) continue;
		
		// separate space number from its name and PPU count; alternatively the current line can be the beginning 
		// of a new PCubeS model
		tokenList = string_utils::tokenizeString(line, separator1);
		std::string firstToken = tokenList->Nth(0);
		if (strcmp(firstToken.c_str(), "Model") == 0) {
			if (currentModelNo > 0) {
				int modelType = hybridModel ? PCUBES_MODEL_TYPE_HYBRID : PCUBES_MODEL_TYPE_HOST_ONLY; 
				PCubeSModel *lastModel = new PCubeSModel(modelType, ppsList);
				lastModel->setModelName(strdup(modelName.c_str()));
				modelList->Append(lastModel);
			}
			hybridModel = false;
			modelName = tokenList->Nth(1);
			ppsList = new List<PPS_Definition*>;
			currentModelNo++;
			continue;
		}

		std::string spaceNoStr = tokenList->Nth(0);
		std::string spaceNameStr = tokenList->Nth(1);

		// retrieve the space ID; also read any additional attributes mentioned for the space
		tokenList = string_utils::tokenizeString(spaceNoStr, separator2);
		std::string spaceNo = tokenList->Nth(1);
		int spaceId;
		List<const char*> *attrList = string_utils::readAttributes(spaceNo);
		spaceNo = spaceNo.substr(0, 1);
		spaceId = atoi(spaceNo.c_str());
		bool coreSpace = string_utils::contains(attrList, "core");
		bool segmented = string_utils::contains(attrList, "segment");
		bool physicalUnit = string_utils::contains(attrList, "unit");
		bool gpuTransition = string_utils::contains(attrList, "gpu");

		// if there is a GPU transition PPS in the current PCubeS model then this is a hybrid model and we should 
		// record this information
		if (gpuTransition) hybridModel = true;

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
		spaceDefinition->segmented = segmented;
		spaceDefinition->physicalUnit = physicalUnit;
		spaceDefinition->gpuTransition = gpuTransition;
			
		// store the space definition in the list in top-down order
		int i = 0;	
		for (; i < ppsList->NumElements(); i++) {
			if (ppsList->Nth(i)->id > spaceId) continue;
			else break;	
		}
		ppsList->InsertAt(spaceDefinition, i);
	}
	pcubesfile.close();

	int modelType = hybridModel ? PCUBES_MODEL_TYPE_HYBRID : PCUBES_MODEL_TYPE_HOST_ONLY; 
	PCubeSModel *lastModel = new PCubeSModel(modelType, ppsList);
	lastModel->setModelName(strdup(modelName.c_str()));
	modelList->Append(lastModel);

	std::cout << "PCubeS Description(s) of the hardware\n";
	for (int i = 0; i < modelList->NumElements(); i++) {
		PCubeSModel *model = modelList->Nth(i);
		model->print(1);
	}
	return modelList;
}

MappingNode *parseMappingConfiguration(const char *taskName,
                const char *filePath,
                PartitionHierarchy *lpsHierarchy,
                List<PCubeSModel*> *pcubesModels) {

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
		
	// determine what PCubeS model is selected for the LPS to PPS mapping; if no model is specified then assume
	// that the default model has been intended
	mappingList = string_utils::tokenizeString(mapping, newlineDelimiter);
	int mappingCount = mappingList->NumElements();
	int lineNo = 0;
	std::string firstLine = mappingList->Nth(0);
	tokenList = string_utils::tokenizeString(firstLine, mappingDelimiter);
	List<PPS_Definition*> *pcubesConfig = NULL;
	if (strcmp("Model", tokenList->Nth(0).c_str()) == 0) {
		const char *modelName = tokenList->Nth(1).c_str();
		bool modelFound = false;
		for (int i = 0; i < pcubesModels->NumElements(); i++) {
			PCubeSModel *model = pcubesModels->Nth(i);
			if (strcmp(modelName, model->getModelName()) == 0) {
				modelFound = true;
				pcubesConfig = model->getPpsDefinitions();
				lpsHierarchy->setPCubeSModel(model);
			}
		}
		if (!modelFound) {
			std::cout << "could not find the specified model in the PCubeS description\n";
			std::exit(EXIT_FAILURE);	
		}
		lineNo++;
	} else {
		PCubeSModel *model = pcubesModels->Nth(0);
		pcubesConfig = model->getPpsDefinitions();
		lpsHierarchy->setPCubeSModel(model);
	}
	
	// create the root of the mapping hierarchy
	MapEntry *rootEntry = new MapEntry();
	Space *rootSpace = lpsHierarchy->getRootSpace();
	rootEntry->LPS = rootSpace;
	rootEntry->PPS = pcubesConfig->Nth(0);
	rootSpace->setPpsId(pcubesConfig->NumElements());
	MappingNode *rootNode = new MappingNode();
	rootNode->parent = NULL;
	rootNode->mappingConfig = rootEntry;
	rootNode->children = new List<MappingNode*>;

	// parse individual lines and construct the mapping hierarchy
	Hashtable<MappingNode*> *mappingTable = new Hashtable<MappingNode*>;
	int totalPPSes = pcubesConfig->NumElements();
	while (lineNo < mappingCount) {

		// determine the LPS and PPS for the mapping

		std::string mapping = mappingList->Nth(lineNo);
		tokenList = string_utils::tokenizeString(mapping, mappingDelimiter);
		int ppsId = atoi(tokenList->Nth(1).c_str());
		PPS_Definition *pps = pcubesConfig->Nth(totalPPSes - ppsId);

		// the mapping root is already defined and maintained separately; therefore, the remaining logic
		// can be skipped for the root
		std::string lpsStr = tokenList->Nth(0);
		std::string rootName(Space::RootSpaceName);
		std::size_t rootFound = lpsStr.rfind(rootName);
		if (rootFound != std::string::npos) {
			rootEntry->PPS = pps;
			rootSpace->setPpsId(ppsId);
			lineNo++;
			continue;
		}

		char lpsId = lpsStr.at(lpsStr.length() - 1);
		Space *lps = lpsHierarchy->getSpace(lpsId);
		if (lps == NULL) {
			std::cout << "Logical space \"" << lpsStr << "\" is not found in the code" << std::endl;
			std::exit(EXIT_FAILURE);
		}
		
		// create a mapping configuration object
		MapEntry *entry = new MapEntry();
		entry->LPS = lps;
		lps->setPpsId(ppsId);
		entry->PPS = pps;
		MappingNode *node = new MappingNode();
		node->parent = NULL;
		node->mappingConfig = entry;
		node->children = new List<MappingNode*>;
		mappingTable->Enter(lps->getName(), node, true);
		lineNo++;
	
		// if the LPS is subpartitioned then map the subpartition into the same PPS
		if (lps->getSubpartition() != NULL) {
			MapEntry *subEntry = new MapEntry();
			subEntry->LPS = lps->getSubpartition();
			subEntry->LPS->setPpsId(ppsId);
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
	std::string stmtSeparator = ";\n";
	std::ofstream programFile;
	programFile.open (outputFile, std::ofstream::out | std::ofstream::app);
  	if (programFile.is_open()) {
		const char *header = "constants for LPSes";
		decorator::writeSectionHeader(programFile, header);
		programFile << std::endl;
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
			programFile << " = " << node->index << stmtSeparator;	
		}
		programFile << "const int Space_Count = " << spaceCount << stmtSeparator;
    		programFile.close();
  	} else {
		std::cout << "Unable to open output program file";
		std::exit(EXIT_FAILURE);
	}
}

void generatePPSCountConstants(const char *outputFile, PCubeSModel *pcubesModel) {
	
	List<PPS_Definition*> *pcubesConfig = pcubesModel->getPpsDefinitions();
	std::string stmtSeparator = ";\n";
	std::ofstream programFile;
	programFile.open (outputFile, std::ofstream::out | std::ofstream::app);

  	if (programFile.is_open()) {
		
		decorator::writeSectionHeader(programFile, "constants for PPS counts");
		programFile << std::endl;
		PPS_Definition *pps = pcubesConfig->Nth(0);
		int prevSpaceId = pps->id;
		programFile << "const int Space_" << pps->id << "_PPUs";
		programFile << " = " << pps->units << stmtSeparator;
		for (int i = 1; i < pcubesConfig->NumElements(); i++) {
			pps = pcubesConfig->Nth(i);
			programFile << "const int Space_" << pps->id;
			programFile << "_Par_" << prevSpaceId << "_PPUs";
			programFile << " = " << pps->units << stmtSeparator;
			prevSpaceId = pps->id;
		}

		if (pcubesModel->getModelType() == PCUBES_MODEL_TYPE_HYBRID) {
			decorator::writeSectionHeader(programFile, "constants for GPU Configuration");		
			programFile << std::endl;
			programFile << "const int SM_COUNT = " << pcubesModel->getSMCount() << stmtSeparator;
			programFile << "const int WARP_COUNT = " << pcubesModel->getWarpCount() << stmtSeparator;
		}

    		programFile.close();
  	} else {
		std::cout << "Unable to open output program file";
		std::exit(EXIT_FAILURE);
	}
}

void generateProcessorOrderArray(const char *outputFile, const char *processorFile) {
	
	std::string line;
	std::ifstream inputFile(processorFile);
	std::string separator1 = ":";
	std::string separator2 = " ";
	List<std::string> *tokenList;
	List<std::string> *fieldList;
	if (!inputFile.is_open()) {
		std::cout << "could not open processor specification file" << std::endl;
		std::exit(EXIT_FAILURE);
	}	
	
	std::cout << "Parsing the prcoessor description" << std::endl;
	
	List<int> *processorIdList = new List<int>;
	List<int> *physicalIdList = new List<int>;

	// we are not sure if the core id information of a processor is important for ordering processor
	// nonetheless we are keeping this information in hand for possible later use 
	List<int> *coreIdList = new List<int>;

	// parse each line in the processor description file and get the different id of processors
	while (std::getline(inputFile, line)) {
		
		string_utils::trim(line);
		tokenList = string_utils::tokenizeString(line, separator1);
		string_utils::shrinkWhitespaces(line);
		
		std::string procNoStr = tokenList->Nth(1);
		fieldList = string_utils::tokenizeString(procNoStr, separator2);
		processorIdList->Append(atoi(fieldList->Nth(0).c_str()));
		
		std::string physicalIdStr = tokenList->Nth(2);
		fieldList = string_utils::tokenizeString(physicalIdStr, separator2);
		physicalIdList->Append(atoi(fieldList->Nth(0).c_str()));

		std::string coreIdStr = tokenList->Nth(3);
		fieldList = string_utils::tokenizeString(coreIdStr, separator2);
		coreIdList->Append(atoi(fieldList->Nth(0).c_str()));
	}
	inputFile.close();

	// sort the processor Id list based on their physical id
	List<int> *sortedProcessorIdList = new List<int>;
	List<int> *reorderedPhyIdList = new List<int>;
	sortedProcessorIdList->Append(processorIdList->Nth(0));
	reorderedPhyIdList->Append(physicalIdList->Nth(0));

	for (int i = 1; i < processorIdList->NumElements(); i++) {
		int processorId = processorIdList->Nth(i);
		int physicalId = physicalIdList->Nth(i);	
		bool inserted = false;
		for (int j = 0; j < sortedProcessorIdList->NumElements(); j++) {
			if (reorderedPhyIdList->Nth(j) <= physicalId) continue;
			reorderedPhyIdList->InsertAt(physicalId, j);
			sortedProcessorIdList->InsertAt(processorId, j);
			inserted = true;
			break;
		}
		if (!inserted) {
			reorderedPhyIdList->Append(physicalId);
			sortedProcessorIdList->Append(processorId);
		}
	}

	// write the reordered list as a constant array in the output fille
	std::string stmtSeparator = ";\n";
	std::string paramSeparator = ", ";
	std::ofstream headerFile;
	headerFile.open (outputFile, std::ofstream::out | std::ofstream::app);
  	if (!headerFile.is_open()) {
		std::cout << "Unable to open output header file to write processor order array";
		std::exit(EXIT_FAILURE);
	}
	const char *header = "processor ordering in the hardware";
	decorator::writeSectionHeader(headerFile, header);
	headerFile << std::endl;
	headerFile << "const int Processor_Order[" << sortedProcessorIdList->NumElements() << "]";
	headerFile << " = {";
	int count = 0;
	int remaining = sortedProcessorIdList->NumElements();
	for (int i = 0; i < sortedProcessorIdList->NumElements(); i++) {
		if (i > 0) headerFile << paramSeparator;
		headerFile << sortedProcessorIdList->Nth(i);
		count++;
		remaining--;
		if (count == 10 && remaining > 5) {
			headerFile << std::endl << "\t\t";
			count = 0;
		}
	}
	headerFile << "}" << stmtSeparator;
	headerFile.close();	
}
