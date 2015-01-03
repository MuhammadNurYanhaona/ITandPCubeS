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
}

void TaskGenerator::generate(List<PPS_Definition*> *pcubesConfig) {

	initializeOutputFile(programFile);

        PartitionHierarchy *lpsHierarchy = taskDef->getPartitionHierarchy();
        MappingNode *mappingConfig = parseMappingConfiguration(taskDef->getName(),
                        mappingFile, lpsHierarchy, pcubesConfig);
        
	// generate macro definitions needed for various reasons
        generateLPSMacroDefinitions(programFile, mappingConfig);
        generatePPSCountMacros(programFile, pcubesConfig);
        generateThreadCountMacros(programFile, mappingConfig, pcubesConfig);
        
	// generate library routines for LPUs management        
        List<Identifier*> *partitionArgs = taskDef->getPartitionArguments();
        Hashtable<List<PartitionParameterConfig*>*> *partitionFnParamConfigs
                        = generateLPUCountFunctions(programFile, mappingConfig, partitionArgs);
        Hashtable<List<int>*> *lpuPartFnParamsConfigs
                        = generateAllGetPartForLPURoutines(programFile, 
					mappingConfig, partitionArgs);
        
	// generate task specific data structures 
        generateLpuDataStructures(programFile, mappingConfig);
        generateArrayMetadataAndEnvLinks(programFile, 
			mappingConfig, taskDef->getEnvironmentLinks());
        
	// generate thread management functions and classes
        generateFnForThreadIdsAllocation(programFile, mappingConfig, pcubesConfig);
        generateThreadStateImpl(programFile, mappingConfig,
                        partitionFnParamConfigs, lpuPartFnParamsConfigs);

}


