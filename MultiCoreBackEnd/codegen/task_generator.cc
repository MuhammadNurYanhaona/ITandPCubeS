#include "task_generator.h"

#include <iostream>
#include <sstream>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>

#include "code_generator.h"
#include "thread_state_mgmt.h"
#include "space_mapping.h"
#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "../utils/string_utils.h"
#include "../syntax/ast.h"
#include "../syntax/ast_def.h"
#include "../syntax/errors.h"
#include "../syntax/ast_task.h"

TaskGenerator::TaskGenerator(TaskDef *taskDef,
                const char *outputDirectory,
                const char *mappingFile) {

	this->taskDef = taskDef;
	this->mappingFile = mappingFile;

	std::ostringstream headerFileStr;
	headerFileStr << outputDirectory;
	const char *taskName = string_utils::replaceChar(taskDef->getName(), ' ', '_');
	const char *taskNameLower = string_utils::toLower(taskName);
	headerFileStr << taskNameLower << ".h";
	headerFile = strdup(headerFileStr.str().c_str());

	std::ostringstream programFileStr;
	programFileStr << outputDirectory;
	programFileStr << taskNameLower << ".cc";
	programFile = strdup(programFileStr.str().c_str());

	initials = string_utils::getInitials(taskDef->getName());
	initials = string_utils::toLower(initials);
}

void TaskGenerator::generate(List<PPS_Definition*> *pcubesConfig) {

	initializeOutputFiles(headerFile, programFile, initials);

        PartitionHierarchy *lpsHierarchy = taskDef->getPartitionHierarchy();
        MappingNode *mappingConfig = parseMappingConfiguration(taskDef->getName(),
                        mappingFile, lpsHierarchy, pcubesConfig);
        
	// generate constansts needed for various reasons
        generateLPSConstants(headerFile, mappingConfig);
        generatePPSCountConstants(headerFile, pcubesConfig);
        generateThreadCountConstants(headerFile, mappingConfig, pcubesConfig);
        
	// generate library routines for LPUs management        
        List<Identifier*> *partitionArgs = taskDef->getPartitionArguments();
        Hashtable<List<PartitionParameterConfig*>*> *partitionFnParamConfigs
                        = generateLPUCountFunctions(headerFile, 
					programFile, initials, mappingConfig, partitionArgs);
        Hashtable<List<int>*> *lpuPartFnParamsConfigs
                        = generateAllGetPartForLPURoutines(headerFile, programFile, 
					initials, mappingConfig, partitionArgs);
        
	// generate task specific data structures 
        generateLpuDataStructures(headerFile, mappingConfig);
        generateArrayMetadataAndEnvLinks(headerFile, 
			mappingConfig, taskDef->getEnvironmentLinks());
        
	// generate thread management functions and classes
        generateFnForThreadIdsAllocation(headerFile, 
			programFile, initials, mappingConfig, pcubesConfig);
        generateThreadStateImpl(headerFile, programFile, mappingConfig,
                        partitionFnParamConfigs, lpuPartFnParamsConfigs);

	closeNameSpace(headerFile);
}


