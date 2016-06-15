#include "gpu_execution.h"
#include "space_mapping.h"
#include "../utils/list.h"
#include "../semantics/task_space.h"
#include "../static-analysis/gpu_execution_ctxt.h"
#include "../utils/decorator_utils.h"
#include "../utils/code_constant.h"
#include "../utils/string_utils.h"

#include <fstream>
#include <cstdlib>
#include <iostream>

void initializeCudaProgramFile(const char *initials, 
		const char *headerFileName, const char *programFileName) {
	        
	std::string line;
        std::ifstream commIncludeFile("config/default-cuda-includes.txt");
        std::ofstream programFile;
        programFile.open (programFileName, std::ofstream::out);
        if (!programFile.is_open()) {
                std::cout << "Unable to open output CUDA program file";
                std::exit(EXIT_FAILURE);
        }

        int taskNameIndex = string_utils::getLastIndexOf(headerFileName, '/') + 1;
        char *taskName = string_utils::substr(headerFileName, taskNameIndex, strlen(headerFileName));

        decorator::writeSectionHeader(programFile, "header file for the task");
        programFile << "#include \"" << taskName  << '"' << std::endl << std::endl;
        decorator::writeSectionHeader(programFile, "header files for different purposes");

        if (commIncludeFile.is_open()) {
                while (std::getline(commIncludeFile, line)) {
                        programFile << line << std::endl;
                }
                programFile << std::endl;
        } else {
                std::cout << "Unable to open common include file";
                std::exit(EXIT_FAILURE);
        }

        programFile << "using namespace " << string_utils::toLower(initials) << ";\n\n";

        commIncludeFile.close();
        programFile.close();
}

void generateBatchConfigurationConstants(const char *headerFileName, PCubeSModel *pcubesModel) {

	std::cout << "Generating GPU batch configuration constants\n";

	std::ofstream headerFile;
        headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        if (headerFile.is_open()) {
                const char *header = "GPU LPU Batch configuration constants";
                decorator::writeSectionHeader(headerFile, header);
		headerFile << "\n";
        } else {
                std::cout << "Unable to open output header file";
                std::exit(EXIT_FAILURE);
        }

	// currently we do not hold memory capacity information in the partial PCubeS description used by the
	// compiler; so we just set here a memory limit that is supported by most GPUs
	headerFile << "const long GPU_MAX_MEM_CONSUMPTION = 3 * 1024 * 1024 * 1024l" << stmtSeparator; // 3 GB

	// if the GPU context LPS has been mapped to the GPU then we will work on just 1 LPU at a time 
	// expecting the LPU computation is big enough to fill the entire card memory or it has enough work to
	// keep all the lower layer PPUs busy for a considerable time
	headerFile << "const int GPU_BATCH_SIZE_THRESHOLD = 1" << stmtSeparator;

	// for the SM and warp level mapping of the GPU context LPS the LPUs are expected to be small; so we
	// set the batch size considerably big; note that setting the batch size as some multiple of the PPU
	// count is ideal for doing proper load balancing
	int smCount = pcubesModel->getSMCount();
	int warpCount = pcubesModel->getWarpCount();
	int batchMultiplier = 100;
	headerFile << "const int SM_BATCH_SIZE_THRESHOLD = ";
	headerFile << smCount * batchMultiplier << stmtSeparator;
	headerFile << "const int WARP_BATCH_SIZE_THRESHOLD = ";
	headerFile << smCount * warpCount * batchMultiplier << stmtSeparator;

	headerFile.close();
}

void generateLpuBatchControllerForLps(Space *gpuContextLps, const char *initials,
                std::ofstream &headerFile, std::ofstream &programFile) {

	const char *lpsName = gpuContextLps->getName();	
        decorator::writeSubsectionHeader(headerFile, lpsName);
        decorator::writeSubsectionHeader(programFile, lpsName);
}

void generateLpuBatchControllers(List<GpuExecutionContext*> *gpuExecutionContextList,
                const char *initials,
                const char *headerFileName, const char *programFileName) {
	
	std::cout << "Generating GPU LPU stage-in stage-out controllers\n";

	std::ofstream programFile, headerFile;
        headerFile.open (headerFileName, std::ofstream::out);
        programFile.open (programFileName, std::ofstream::out);
        if (!programFile.is_open()) {
                std::cout << "Unable to open output program file";
                std::exit(EXIT_FAILURE);
        }
        if (!headerFile.is_open()) {
                std::cout << "Unable to open output header file";
                std::exit(EXIT_FAILURE);
        }

	const char *header = "LPU stage-in/stage-out controllers";
        decorator::writeSectionHeader(headerFile, header);
        decorator::writeSectionHeader(programFile, header);

	// different sub-flow execution contexts can share the same LPU batch controller if they enters the GPU
	// in the same LPS; thus the LPU batch controllers are LPS specific -- not context specific 
	List<const char*> *alreadyCoveredLpses = new List<const char*>;
	for (int i = 0; i < gpuExecutionContextList->NumElements(); i++) {
		GpuExecutionContext *context = gpuExecutionContextList->Nth(i);
		Space *contextLps = context->getContextLps();
		const char *lpsName = contextLps->getName();
		if (!string_utils::contains(alreadyCoveredLpses, lpsName)) {
			alreadyCoveredLpses->Append(lpsName);
			generateLpuBatchControllerForLps(contextLps, initials, headerFile, programFile);		
		}
	}

	headerFile.close();
	programFile.close();
}
