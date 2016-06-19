#include "gpu_execution.h"
#include "space_mapping.h"
#include "../utils/list.h"
#include "../semantics/task_space.h"
#include "../static-analysis/gpu_execution_ctxt.h"
#include "../utils/decorator_utils.h"
#include "../utils/code_constant.h"
#include "../utils/string_utils.h"

#include <fstream>
#include <sstream>
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

void generateLpuBatchControllerForLps(GpuExecutionContext *gpuContext, 
		PCubeSModel *pcubesModel, 
		const char *initials,
                std::ofstream &headerFile, std::ofstream &programFile) {

	const char *contextName = gpuContext->getContextName();	
        decorator::writeSubsectionHeader(headerFile, contextName);
        decorator::writeSubsectionHeader(programFile, contextName);

	std::ostringstream classNameStr;
	classNameStr << "Context" << gpuContext->getContextId() << "LpuBatchController";
	std::string className = classNameStr.str();
	
	// declare the class in the header file
	headerFile << "class " << className << " : public LpuBatchController {\n";
	headerFile << "  public:\n";
	headerFile << indent << className << "()" << stmtSeparator;
	headerFile << indent << "int calculateLpuMemoryRequirement(LPU *lpu)" << stmtSeparator;
	headerFile << indent << "void addLpuToTheCurrentBatch(LPU *lpu)" << stmtSeparator;
	headerFile << "}" << stmtSeparator;

	// then add implementation for the constructor and the two virtual functions inherited from the base 
	// class in the program file
	generateLpuBatchControllerConstructor(gpuContext, pcubesModel, initials, programFile);
	generateLpuBatchControllerLpuAdder(gpuContext, initials, programFile);
	generateLpuBatchControllerMemchecker(gpuContext, initials, programFile);
	
}

void generateLpuBatchControllers(List<GpuExecutionContext*> *gpuExecutionContextList,
                PCubeSModel *pcubesModel,
		const char *initials,
                const char *headerFileName, const char *programFileName) {
	
	std::cout << "Generating GPU LPU stage-in stage-out controllers\n";

	std::ofstream programFile, headerFile;
        headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
        if (!programFile.is_open()) {
                std::cout << "Unable to open output program file";
                std::exit(EXIT_FAILURE);
        }
        if (!headerFile.is_open()) {
                std::cout << "Unable to open output header file";
                std::exit(EXIT_FAILURE);
        }

	const char *header = "GPU LPU stage-in/stage-out controllers";
        decorator::writeSectionHeader(headerFile, header);
        decorator::writeSectionHeader(programFile, header);

	for (int i = 0; i < gpuExecutionContextList->NumElements(); i++) {
		GpuExecutionContext *context = gpuExecutionContextList->Nth(i);
		generateLpuBatchControllerForLps(context, 
				pcubesModel, initials, headerFile, programFile);		
	}

	headerFile.close();
	programFile.close();
}

void generateLpuBatchControllerConstructor(GpuExecutionContext *gpuContext, PCubeSModel *pcubesModel,
                const char *initials, std::ofstream &programFile) {
	
	// determine the batch size for executing LPUs in bunch in the GPU
	Space *gpuContextLps = gpuContext->getContextLps();
	int ppsId = gpuContextLps->getPpsId();
	int gpuPpsId = pcubesModel->getGpuTransitionSpaceId();
	std::string batchSize;	
	if (ppsId == gpuPpsId) {
		batchSize = std::string("GPU_BATCH_SIZE_THRESHOLD");
	} else if (ppsId == gpuPpsId - 1) {
		batchSize = std::string("SM_BATCH_SIZE_THRESHOLD");
	} else {
		batchSize = std::string("WARP_BATCH_SIZE_THRESHOLD");
	}

	std::ostringstream classNameStr;
	classNameStr << "Context" << gpuContext->getContextId() << "LpuBatchController";
	std::string className = classNameStr.str();

	programFile << std::endl;
	programFile << initials << "::" << className << "::" << className;
	programFile << "() : LpuBatchController() {\n";

	// Determine the list of arrays and how they are accessed within the current subflow. Note that scalar and
	// non-array collections are staged in and out separately. So here we are only concerned about arrays. 
	List<const char*> *arrayNames = gpuContextLps->getLocallyUsedArrayNames();
	List<const char*> *accessedArrays = string_utils::intersectLists(
			gpuContext->getVariableAccessList(), arrayNames);
	List<const char*> *modifiedArrays = string_utils::intersectLists(
			gpuContext->getModifiedVariableList(), arrayNames);
	List<const char*> *epochDependArrays = string_utils::intersectLists(
			gpuContext->getEpochDependentVariableList(), arrayNames);
	List<const char*> *epochIndArrays = string_utils::subtractList(accessedArrays, epochDependArrays);

	// create property lists inside the constructor for the arrays of the above
	programFile << std::endl;
	programFile << indent << "List<const char*> *propertyNames = new List<const char*>" << stmtSeparator;
	for (int i = 0; i < accessedArrays->NumElements(); i++) {
		const char *array = accessedArrays->Nth(i);
		programFile << indent << "propertyNames->Append(\"" << array << "\")" << stmtSeparator;
	}	
	programFile << indent << "List<const char*> *modifiedPropertyNames = new List<const char*>";
	programFile << stmtSeparator;
	for (int i = 0; i < modifiedArrays->NumElements(); i++) {
		const char *array = modifiedArrays->Nth(i);
		programFile << indent << "modifiedPropertyNames->Append(\"" << array << "\")" << stmtSeparator;
	}
	programFile << indent << "List<const char*> *multiversionProperties = new List<const char*>";
	programFile << stmtSeparator;
	for (int i = 0; i < epochDependArrays->NumElements(); i++) {
		const char *array = epochDependArrays->Nth(i);
		programFile << indent << "multiversionProperties->Append(\"" << array << "\")" << stmtSeparator;
	}	
	programFile << indent << "List<const char*> *versionlessProperties = new List<const char*>";
	programFile << stmtSeparator;
	for (int i = 0; i < epochIndArrays->NumElements(); i++) {
		const char *array = epochIndArrays->Nth(i);
		programFile << indent << "versionlessProperties->Append(\"" << array << "\")" << stmtSeparator;
	}

	// initialize the LPU Batch controller's internal buffer and data parts management structures
	programFile << std::endl;
	programFile << indent << "setBufferManager(new LpuDataBufferManager(";
	programFile << "versionlessProperties" << paramSeparator << "multiversionProperties))" << stmtSeparator;
	programFile << indent << "initialize(" << batchSize << paramSeparator << paramIndent;
	programFile << "GPU_MAX_MEM_CONSUMPTION" << paramSeparator << paramIndent;
	programFile << "propertyNames" << paramSeparator << "modifiedPropertyNames)" << stmtSeparator;	
		
	programFile << "}\n";
}

void generateLpuBatchControllerLpuAdder(GpuExecutionContext *gpuContext, 
		const char *initials, std::ofstream &programFile) {
	
	Space *gpuContextLps = gpuContext->getContextLps();

	// determine what multiversion and version-less arrays are accessed within the sub-flow
	List<const char*> *arrayNames = gpuContextLps->getLocallyUsedArrayNames();
	List<const char*> *epochDependArrays = string_utils::intersectLists(
			gpuContext->getEpochDependentVariableList(), arrayNames);
	List<const char*> *epochIndArrays = string_utils::intersectLists(
			gpuContext->getEpochIndependentVariableList(), arrayNames);

	std::ostringstream classNameStr;
	classNameStr << "Context" << gpuContext->getContextId() << "LpuBatchController";
	std::string className = classNameStr.str();

	programFile << std::endl;
	programFile << "void " << initials << "::" << className << "::addLpuToTheCurrentBatch(LPU *lpu) {\n\n";

	// get the LPS specific LPU reference
	const char *lpsName = gpuContextLps->getName();
	programFile << indent << "Space" << lpsName;
	programFile << "_LPU *typedLpu = (Space" << lpsName << "_LPU*) lpu" << stmtSeparator;

	// iterate over the versionless properties and create a normal LPU data part for each and add that to the
	// batch controller only when the data part has not already been included during the processing of some
	// earlier LPU
	programFile << indent << "bool redundantPart = false" << stmtSeparator;
	for (int i = 0; i < epochIndArrays->NumElements(); i++) {
		
		const char *varName = epochIndArrays->Nth(i);
		ArrayDataStructure *array = (ArrayDataStructure*) gpuContextLps->getStructure(varName);
		int dimensionality = array->getDimensionality();
		ArrayType *arrayType = (ArrayType*) array->getType();		
		Type *elementType = arrayType->getTerminalElementType();
	
		programFile << std::endl;
		programFile << indent << "LpuDataPart *" << varName << "Part = new LpuDataPart(";
		programFile << dimensionality << paramSeparator << paramIndent;
		programFile << "typedLpu->" << varName << "PartDims" << paramSeparator;
		programFile << "typedLpu->" << varName << paramSeparator << paramIndent;
		programFile << "sizeof(" << elementType->getCType() << ")" << paramSeparator;
		programFile << "typedLpu->" << varName << "PartId)" << stmtSeparator; 

		programFile << indent << "redundantPart = dataPartTracker->addDataPart(";
		programFile << varName << "Part" << paramSeparator;
		programFile << "\"" << varName << "\")" << stmtSeparator;
		programFile << indent << "if (redundantPart) delete " << varName << "Part" << stmtSeparator;
	}

	// for multiversion properties apply the same logic of the above but create versioned LPU data parts
	for (int i = 0; i < epochDependArrays->NumElements(); i++) {

		const char *varName = epochDependArrays->Nth(i);
		ArrayDataStructure *array = (ArrayDataStructure*) gpuContextLps->getStructure(varName);
		int dimensionality = array->getDimensionality();
		ArrayType *arrayType = (ArrayType*) array->getType();		
		Type *elementType = arrayType->getTerminalElementType();
		int versionCount = array->getVersionCount();
		
		programFile << std::endl;
		programFile << indent << "List<void*> *" << varName << "VersionList = ";
		programFile << "new List<void*>" << stmtSeparator;
		for (int j = 0; j <= versionCount; j++) {
			programFile << indent << varName << "VersionList->Append(";
			programFile << "typedLpu->" << varName;
			if (j > 0) programFile << "_lag_" << j;
			programFile << ")" << stmtSeparator;
		}

		programFile << std::endl;
		programFile << indent << "VersionedLpuDataPart *" << varName;
		programFile << "Part = new VersionedLpuDataPart(" << dimensionality << paramSeparator;
		programFile << paramIndent << "typedLpu->" << varName << "PartDims" << paramSeparator;
		programFile << varName << "VersionList" << paramSeparator << paramIndent;
		programFile << "sizeof(" << elementType->getCType() << ")" << paramSeparator;
		programFile << "typedLpu->" << varName << "PartId)" << stmtSeparator; 

		programFile << indent << "redundantPart = dataPartTracker->addDataPart(";
		programFile << varName << "Part" << paramSeparator;
		programFile << "\"" << varName << "\")" << stmtSeparator;
		programFile << indent << "if (redundantPart) delete " << varName << "Part" << stmtSeparator;
	}

	programFile << indent << "LpuBatchController::addLpuToTheCurrentBatch(lpu)" << stmtSeparator;
	programFile << "}\n";
}

void generateLpuBatchControllerMemchecker(GpuExecutionContext *gpuContext,
                const char *initials, std::ofstream &programFile) {
	
	Space *gpuContextLps = gpuContext->getContextLps();

	// determine what multiversion and version-less arrays are accessed within the sub-flow
	List<const char*> *arrayNames = gpuContextLps->getLocallyUsedArrayNames();
	List<const char*> *accessedArrays = string_utils::intersectLists(
			gpuContext->getVariableAccessList(), arrayNames);
	List<const char*> *epochDependArrays = string_utils::intersectLists(
			gpuContext->getEpochDependentVariableList(), arrayNames);

	std::ostringstream classNameStr;
	classNameStr << "Context" << gpuContext->getContextId() << "LpuBatchController";
	std::string className = classNameStr.str();

	programFile << std::endl;
	programFile << "int " << initials << "::" << className;
	programFile << "::calculateLpuMemoryRequirement(LPU *lpu) {\n\n";

	// get the LPS specific LPU reference
	const char *lpsName = gpuContextLps->getName();
	programFile << indent << "Space" << lpsName;
	programFile << "_LPU *typedLpu = (Space" << lpsName << "_LPU*) lpu" << stmtSeparator;

	// initialize the size tracker variable
	programFile << indent << "int size = 0" << stmtSeparator;

	// iterate over the arrays to be used in the GPU computation
	for (int i = 0; i < accessedArrays->NumElements(); i++) {
		
		const char *varName = accessedArrays->Nth(i);
		
		// if the current array has been already included in the batch as part of some other LPU, it does 
		// not add to the size calculation as we stage-in just one copy of replicated data parts
		programFile << indent << "if (!dataPartTracker->isAlreadyIncluded(";
		programFile << "typedLpu->" << varName << "PartId" << paramSeparator;
		programFile << "\"" << varName << "\""	<< ")) {\n";

		// determine the element type and dimensionality of the array
		ArrayDataStructure *array = (ArrayDataStructure*) gpuContextLps->getStructure(varName);
		int dimensionality = array->getDimensionality();
		ArrayType *arrayType = (ArrayType*) array->getType();		
		Type *elementType = arrayType->getTerminalElementType();

		// calculate the amount of additional GPU memory this part will consume
		programFile << doubleIndent << "int partSize = ";
		for (int j = 0; j < dimensionality; j++) {
			if (j > 0) {
				programFile << paramIndent << doubleIndent << "* ";
			}
			programFile << "typedLpu->" << varName << "PartDims[" << j << "].storage.getLength()";
		}
		programFile << paramIndent << doubleIndent << "* sizeof(" << elementType->getCType() << ")";
		if (string_utils::contains(epochDependArrays, varName)) {
			// version count storage starts from 0 so we need to add 1 here before applying it as a 
			// multiplication factor
			int versionCount = array->getVersionCount() + 1;
			programFile << " * " << versionCount;
		}
		programFile << stmtSeparator;

		// add the part size to the size tracker
		programFile << doubleIndent << "size += partSize" << stmtSeparator;
			
		programFile << indent << "}\n";
	}
	
	//return the size and close the program file
	programFile << indent << "return size" << stmtSeparator;
	programFile << "}\n";
}

void generateGpuCodeExecutorForContext(GpuExecutionContext *gpuContext,
                PCubeSModel *pcubesModel,
                const char *initials,
                std::ofstream &headerFile, std::ofstream &programFile) {

	const char *contextName = gpuContext->getContextName();	
        decorator::writeSubsectionHeader(headerFile, contextName);
        decorator::writeSubsectionHeader(programFile, contextName);
	
	std::ostringstream classNameStr;
	classNameStr << "Context" << gpuContext->getContextId() << "CodeExecutor";
	std::string className = classNameStr.str();

	// declare the class in the header file
	headerFile << "class " << className << " : public GpuCodeExecutor {\n";
	
	// all code executors should have the root level array metadata, partition information, and scalar variables
	// as their private properties
	headerFile << "  private:\n";
	headerFile << indent << initials << "::ArrayMetadata arrayMetadata" << stmtSeparator;
	const char *upperInitials = string_utils::toUpper(initials);
	headerFile << indent << upperInitials << "Partition partition" << stmtSeparator;

	// unlike the previous two, task-global and thread-local scalar variables are not read-only; so we maintain
	// pointer reference to them and have two versions: one for the host and the other for the GPU
	headerFile << indent << initials << "::TaskGlobals " << "*taskGlobalsHost";
	headerFile << paramSeparator << "*taskGlobalsGpu" << stmtSeparator;
	headerFile << indent << initials << "::ThreadLocals " << "*threadLocalsHost";
	headerFile << paramSeparator << "*threadLocalsGpu" << stmtSeparator;

	// define the constructor and three functions a context specific code executor should give implementation for
	headerFile << "  public:\n";
	headerFile << indent << className << "(LpuBatchController *lpuBatchController";
	headerFile << paramSeparator << paramIndent << initials << "::ArrayMetadata arrayMetadata";
	headerFile << paramSeparator << paramIndent << upperInitials << "Partition partition";
	headerFile << paramSeparator << paramIndent << initials << "::TaskGlobals *taskGlobals";
	headerFile << paramSeparator << paramIndent << initials << "::ThreadLocals *threadLocals)";
	headerFile << stmtSeparator;
	headerFile << indent << "void offloadFunction()" << stmtSeparator;
	headerFile << indent << "void initialize()" << stmtSeparator;
	headerFile << indent << "void cleanup()" << stmtSeparator;
	headerFile << "}" << stmtSeparator;

	// invoke auxiliary functions to generate implementations for the defined functions 
	generateGpuCodeExecutorConstructor(gpuContext, initials, programFile);
	generateGpuCodeExecutorInitializer(gpuContext, initials, programFile);
	generateGpuCodeExecutorOffloadFn(gpuContext,  pcubesModel, initials, programFile);
	generateGpuCodeExecutorCleanupFn(gpuContext, initials, programFile);
}

void generateGpuCodeExecutors(List<GpuExecutionContext*> *gpuExecutionContextList,
                PCubeSModel *pcubesModel,
                const char *initials,
                const char *headerFileName, const char *programFileName) {
	
	std::cout << "Generating GPU code executors\n";

	std::ofstream programFile, headerFile;
        headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
        if (!programFile.is_open()) {
                std::cout << "Unable to open output program file";
                std::exit(EXIT_FAILURE);
        }
        if (!headerFile.is_open()) {
                std::cout << "Unable to open output header file";
                std::exit(EXIT_FAILURE);
        }

	const char *header = "GPU code executors";
        decorator::writeSectionHeader(headerFile, header);
        decorator::writeSectionHeader(programFile, header);

	for (int i = 0; i < gpuExecutionContextList->NumElements(); i++) {
		GpuExecutionContext *context = gpuExecutionContextList->Nth(i);
		generateGpuCodeExecutorForContext(context, 
				pcubesModel, initials, headerFile, programFile);		
	}

	headerFile.close();
	programFile.close();
}

void generateGpuCodeExecutorConstructor(GpuExecutionContext *gpuContext,
                const char *initials, std::ofstream &programFile) {
	
	std::ostringstream classNameStr;
	classNameStr << "Context" << gpuContext->getContextId() << "CodeExecutor";
	std::string className = classNameStr.str();

	programFile << std::endl;
	programFile << initials << "::" << className << "::" << className;
	programFile << "(LpuBatchController *lpuBatchController";
	programFile << paramSeparator << paramIndent << initials << "::ArrayMetadata arrayMetadata";
	const char *upperInitials = string_utils::toUpper(initials);
	programFile << paramSeparator << paramIndent << upperInitials << "Partition partition";
	programFile << paramSeparator << paramIndent << initials << "::TaskGlobals *taskGlobals";
	programFile << paramSeparator << paramIndent << initials << "::ThreadLocals *threadLocals)";
	programFile << " : GpuCodeExecutor(lpuBatchController) {\n\n";

	programFile << indent << "this->arrayMetadata = arrayMetadata" << stmtSeparator;
	programFile << indent << "this->partition = partition" << stmtSeparator;
	programFile << indent << "this->taskGlobalsHost = taskGlobals" << stmtSeparator;
	programFile << indent << "this->taskGlobalsGpu = NULL" << stmtSeparator;
	programFile << indent << "this->threadLocalsHost = threadLocals" << stmtSeparator;
	programFile << indent << "this->threadLocalsGpu = NULL" << stmtSeparator;

	programFile << "}\n";
}

void generateGpuCodeExecutorInitializer(GpuExecutionContext *gpuContext,
                const char *initials, std::ofstream &programFile) {

	std::ostringstream classNameStr;
	classNameStr << "Context" << gpuContext->getContextId() << "CodeExecutor";
	std::string className = classNameStr.str();

	programFile << std::endl;
	programFile << "void " << initials << "::" << className << "::initialize() {\n\n";
	
	// invoke the super class's initialize function
	programFile << indent << "GpuCodeExecutor::initialize()" << stmtSeparator << '\n';

	// stage in the task global and thread local scalars into the GPU card memory
	programFile << indent << "int taskGlobalsSize = sizeof(*taskGlobalsHost)" << stmtSeparator;
	programFile << indent << "cudaMemcpy(taskGlobalsGpu" << paramSeparator;
	programFile << "taskGlobalsHost" << paramSeparator;
	programFile << "taskGlobalsSize" << paramSeparator;
	programFile << "cudaMemcpyHostToDevice)" << stmtSeparator;
	programFile << indent << "int threadLocalsSize = sizeof(*threadLocalsHost)" << stmtSeparator;
	programFile << indent << "cudaMemcpy(threadLocalsGpu" << paramSeparator;
	programFile << "threadLocalsHost" << paramSeparator;
	programFile << "threadLocalsSize" << paramSeparator;
	programFile << "cudaMemcpyHostToDevice)" << stmtSeparator;
	
	programFile << "}\n";
}

void generateGpuCodeExecutorOffloadFn(GpuExecutionContext *gpuContext,  
                PCubeSModel *pcubesModel,
                const char *initials, std::ofstream &programFile) {
	
	std::ostringstream classNameStr;
	classNameStr << "Context" << gpuContext->getContextId() << "CodeExecutor";
	std::string className = classNameStr.str();

	programFile << std::endl;
	programFile << "void " << initials << "::" << className << "::offloadFunction() {\n\n";
	
	programFile << indent << "// This is where the computation sub-flow should be implemented as a series\n";
	programFile << indent << "// of kernel calls. We do not have the GPU kernels yet. Need to discuss this.\n";	

	programFile << "}\n";
}

void generateGpuCodeExecutorCleanupFn(GpuExecutionContext *gpuContext,  
                const char *initials, std::ofstream &programFile) {
	
	std::ostringstream classNameStr;
	classNameStr << "Context" << gpuContext->getContextId() << "CodeExecutor";
	std::string className = classNameStr.str();

	programFile << std::endl;
	programFile << "void " << initials << "::" << className << "::cleanup() {\n\n";

	// retrieve whatever updates to scalar variables done in the GPU computation
	programFile << indent << "int taskGlobalsSize = sizeof(*taskGlobalsHost)" << stmtSeparator;
	programFile << indent << "cudaMemcpy(taskGlobalsHost" << paramSeparator;
	programFile << "taskGlobalsGpu" << paramSeparator;
	programFile << "taskGlobalsSize" << paramSeparator;
	programFile << "cudaMemcpyDeviceToHost)" << stmtSeparator;
	programFile << indent << "int threadLocalsSize = sizeof(*threadLocalsHost)" << stmtSeparator;
	programFile << indent << "cudaMemcpy(threadLocalsHost" << paramSeparator;
	programFile << "threadLocalsGpu" << paramSeparator;
	programFile << "threadLocalsSize" << paramSeparator;
	programFile << "cudaMemcpyDeviceToHost)" << stmtSeparator;

	// then invoke the superclass's cleanup function to tear down the GPU device context
	programFile << '\n' << indent << "GpuCodeExecutor::cleanup()" << stmtSeparator;

	programFile << "}\n"; 
}
