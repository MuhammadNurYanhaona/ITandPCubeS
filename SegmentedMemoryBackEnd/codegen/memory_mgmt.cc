#include "memory_mgmt.h"
#include "../semantics/task_space.h"
#include "../memory-management/allocation.h"
#include "../memory-management/part_generation.h"
#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "../utils/decorator_utils.h"
#include "../utils/string_utils.h"
#include "../syntax/ast.h"
#include "../syntax/ast_expr.h"

#include <fstream>
#include <sstream>
#include <cstdlib>
#include <deque>

void genRoutineForDataPartConfig(std::ofstream &headerFile,
                std::ofstream &programFile,
                const char *initials,
                Space *lps,
                ArrayDataStructure *array) {
	
	std::string stmtSeparator = ";\n";
	std::string paramSeparator = ", ";
	std::string indent = "\t";
	std::string doubleIndent = "\t\t";

	std::ostringstream functionHeader;
	functionHeader << "get" << array->getName() << "ConfigForSpace" << lps->getName();
	functionHeader << "(ArrayMetadata *metadata" << paramSeparator;
	functionHeader << "\n" << doubleIndent;
	functionHeader << initials << "Partition partition" << paramSeparator;
	functionHeader << "\n" << doubleIndent;
	functionHeader << "int ppuCount)";

	headerFile << "DataPartitionConfig *" << functionHeader.str() << stmtSeparator;
	programFile << "DataPartitionConfig *" << string_utils::toLower(initials);
	programFile << "::" << functionHeader.str() << " {\n";

	// create a list of data dimension configs that will be used to create the final partition config
	// object
	int dimensionCount = array->getDimensionality();
	programFile << indent << "List<DimPartitionConfig*> *dimensionConfigs";
	programFile << " = new List<DimPartitionConfig*>" << stmtSeparator;

	for (int i = 0; i < dimensionCount; i++) {
		PartitionFunctionConfig *partitionConfig = array->getPartitionSpecForDimension(i + 1);
		// if the partition config allong a dimension is NULL then the structure is unpartitioned
		// along that dimension and we should add a replication config here
		if (partitionConfig == NULL) {
			programFile << indent << "dimensionConfigs->Append(";
			programFile << "new ReplicationConfig(metadata->" << array->getName();
			programFile << "Dims[" << i << "]))" << stmtSeparator;
		} else {
			// get the name of the configuration class for the partition function been used
			const char *configClassName = partitionConfig->getDimensionConfigClassName();
			// get dimension configuration
			DataDimensionConfig *partitionArgs = partitionConfig->getArgsForDimension(i + 1);
			// check if the function supports padding
			bool paddingSupported = partitionConfig->doesSupportGhostRegion();
			// if padding is supported then we need a two elements array to hold the padding
			// configurations for front and back of each partition
			if (paddingSupported) {
				programFile << indent << "int *dim" << i << "Paddings = new int[2]";
				programFile << stmtSeparator;
				programFile << indent << "dim" << i << "Paddings[0] = ";
				Node *frontPadding = partitionArgs->getFrontPaddingArg();
				if (frontPadding == NULL) {
					programFile << "0";
				} else {
					programFile << DataDimensionConfig::getArgumentString(
							frontPadding, "partition.");
				}
				programFile << stmtSeparator;		
				programFile << indent << "dim" << i << "Paddings[1] = ";
				Node *rearPadding = partitionArgs->getBackPaddingArg();
				if (rearPadding == NULL) {
					programFile << "0";
				} else {
					programFile << DataDimensionConfig::getArgumentString(
							rearPadding, "partition.");
				}
				programFile << stmtSeparator;		
			}
			// check if there is any partitioning argument needed by the function
			Node *dividingParam = partitionArgs->getDividingArg();
			bool hasParameters = dividingParam != NULL;
			// If the partition function supports parameters then configuration class instance
			// should get values of those parameters passed in a list. Note that currently we
			// have only single or no argument partition functions so code generation logic is
			// designed accordingly. In future, this restriction should be lifted and changes
			// should be made in semantics and code-generation modules to reflect that change.
			if (hasParameters) {
				programFile << indent << "int *dim" << i << "Arguments = new int";
				programFile << stmtSeparator << indent << "dim" << i << "Arguments[0] = ";
				programFile << DataDimensionConfig::getArgumentString(dividingParam, 
					"partition.");
				programFile << stmtSeparator;
			}

			// determines with what dimension of the LPS the current dimension of the array has
			// been aligned to
			CoordinateSystem *coordSys = lps->getCoordinateSystem();
			int matchingDim = i;
			int spaceDimensions = lps->getDimensionCount();
			int j = 0;
			for (; j < spaceDimensions; j++) {
				Coordinate *coordinate = coordSys->getCoordinate(j + 1);
				Token *token = coordinate->getTokenForDataStructure(array->getName());
				if (token != NULL && !token->isWildcard()) {
					if (token->getDimensionId() == i + 1) break;
				}
			}
			matchingDim = j;
			
			// create the dimension configuration object with all information found and add it in
			// the list
			programFile << indent << "dimensionConfigs->Append(";
			programFile << "new " << configClassName << "(";
			programFile << "metadata->" << array->getName() << "Dims[" << i << "]";
			programFile << paramSeparator << '\n' << indent << doubleIndent;
			if (hasParameters) {
				programFile << "dim" << i << "Arguments";
				programFile << paramSeparator;
			}
			if (paddingSupported) {
				programFile << "dim" << i << "Paddings";
				programFile << paramSeparator;
			}
			programFile << "ppuCount";
			programFile << paramSeparator << matchingDim;
			programFile << "))" << stmtSeparator;
		}
	}
	programFile << indent << "return new DataPartitionConfig(";
	programFile << dimensionCount << paramSeparator << "dimensionConfigs)" << stmtSeparator;
	programFile << "}\n";
}

void genRoutinesForTaskPartitionConfigs(const char *headerFileName,
                const char *programFileName,
                const char *initials,
                PartitionHierarchy *hierarchy) {
	
	std::string statementSeparator = ";\n";
        std::string statementIndent = "\t";
        std::ofstream programFile, headerFile;

        programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
        headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        if (!programFile.is_open() || !headerFile.is_open()) {
                std::cout << "Unable to open header/program file";
                std::exit(EXIT_FAILURE);
        }
	const char *header = "functions for generating partition configuration objects for data structures";
	decorator::writeSectionHeader(headerFile, header);
	decorator::writeSectionHeader(programFile, header);

	Space *root = hierarchy->getRootSpace();
	std::deque<Space*> lpsQueue;
	lpsQueue.push_back(root);
        while (!lpsQueue.empty()) {
                Space *lps = lpsQueue.front();
                lpsQueue.pop_front();
                List<Space*> *children = lps->getChildrenSpaces();
                for (int i = 0; i < children->NumElements(); i++) {
                        lpsQueue.push_back(children->Nth(i));
                }
                if (lps->getSubpartition() != NULL) lpsQueue.push_back(lps->getSubpartition());
		
		int generationCount = 0;
		List<const char*> *structureList = lps->getLocalDataStructureNames();
		for (int i = 0; i < structureList->NumElements(); i++) {
			DataStructure *structure = lps->getLocalStructure(structureList->Nth(i));
			ArrayDataStructure *array = dynamic_cast<ArrayDataStructure*>(structure);
			if (array == NULL) continue;
			else if (!array->getUsageStat()->isAllocated()) continue;

			if (generationCount == 0) {
				std::ostringstream message;
				message << "Space " << lps->getName();
				const char *c_message = message.str().c_str();
				decorator::writeSubsectionHeader(headerFile, c_message);
				decorator::writeSubsectionHeader(programFile, c_message);
			}
			programFile << std::endl;
			genRoutineForDataPartConfig(headerFile, programFile, initials, lps, array);
			generationCount++;
		}
	}
}
