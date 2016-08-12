#include "gpu_execution.h"
#include "space_mapping.h"
#include "name_transformer.h"
#include "../utils/list.h"
#include "../syntax/ast_type.h"
#include "../semantics/task_space.h"
#include "../static-analysis/gpu_execution_ctxt.h"
#include "../utils/decorator_utils.h"
#include "../utils/code_constant.h"
#include "../utils/string_utils.h"

#include <fstream>
#include <sstream>
#include <cstdlib>
#include <iostream>
#include <stack>

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
	int batchMultiplier = 1000;
	headerFile << "const int SM_BATCH_SIZE_THRESHOLD = ";
	headerFile << smCount * batchMultiplier << stmtSeparator;
	headerFile << "const int WARP_BATCH_SIZE_THRESHOLD = ";
	headerFile << smCount * warpCount * batchMultiplier << stmtSeparator;

	headerFile.close();
}

void generateOffloadingMetadataStruct(Space *gpuContextLps, std::ofstream &headerFile) {
	
	// initially we are keeping properties for the LPU count and batch range only in the metadata struct
	headerFile << std::endl;
	const char *lpsName = gpuContextLps->getName();
	int lpsDimensionality = gpuContextLps->getDimensionCount();
	headerFile << "class Space" << lpsName << "GpuMetadata {\n";
	headerFile << "  public:\n";
	for (int i = 0; i < lpsDimensionality; i++) {
		headerFile << indent << "int lpuCount" << i + 1 << stmtSeparator;
	}
	headerFile << indent << "int batchRangeMin" << stmtSeparator;
	headerFile << indent << "int batchRangeMax" << stmtSeparator;
	
	// The GPU PPUs need to be able to identify where to start looking for data and metadata related to the
	// LPUs they will be processing. 
	headerFile << indent << "int batchStartIndex" << stmtSeparator;
	
	headerFile << "}" << stmtSeparator;
}

void generateMetadataAggregatorStruct(Space *gpuContextLps, 
		PCubeSModel *pcubesModel,  
		std::ofstream &headerFile) {
	
	// Determine the number of metadata entries the aggregator should have. The default is 1 for normal off-
	// loading context. If the LPS is subpartitioned then the context is a PPU location sensitive. In other
	// word, specific LPUs are assigned to specific PPUs and PPU-count number of metadata entries are needed. 
	int metadataEntryCount = 1;
	if (gpuContextLps->isSubpartitionSpace()) {
		int ppsId = gpuContextLps->getPpsId();
		int gpuPpsId = pcubesModel->getGpuTransitionSpaceId();
		std::string batchSize;	
		if (ppsId == gpuPpsId) {
			metadataEntryCount = 1;
		} else if (ppsId == gpuPpsId - 1) {
			metadataEntryCount = pcubesModel->getSMCount();
		} else {
			metadataEntryCount = pcubesModel->getSMCount() * pcubesModel->getWarpCount();
		}
	}
	
	const char *lpsName = gpuContextLps->getName();
	headerFile << std::endl;
	headerFile << "class Space" << lpsName << "GpuAggregateMetadata {\n";
	headerFile << "  public:\n";
	headerFile << indent << "Space" << lpsName << "GpuMetadata entries[" << metadataEntryCount << "]";
	headerFile << stmtSeparator;
	headerFile << "}" << stmtSeparator;	
}


void generateKernelLaunchMatadataStructFn(Space *gpuContextLps,
                PCubeSModel *pcubesModel,
                const char *initials,
                std::ofstream &headerFile, std::ofstream &programFile) {

	const char *lpsName = gpuContextLps->getName();

	std::ostringstream fnHeader;
	fnHeader << "getLaunchMetadata(std::vector<int*> *lpuCounts" << paramSeparator << paramIndent;
	fnHeader << "std::vector<Range> *lpuBatchRanges)";

	// write function signature in the header and the program files
	headerFile << std::endl << "Space" << lpsName << "GpuAggregateMetadata ";
	headerFile << fnHeader.str() << stmtSeparator;
	programFile << std::endl << "Space" << lpsName << "GpuAggregateMetadata ";
	programFile << initials << "::" << fnHeader.str() << " {\n\n";

	// instantiate an aggregator variable
	programFile << indent << "Space" << lpsName;
	programFile << "GpuAggregateMetadata metadata" << stmtSeparator;
	
	// instantiate a batch start index counter
	programFile << indent << "int currentBatchStartIndex = 0" << stmtSeparator;

	// iterate over the entries of the vector
	programFile << indent << "for (unsigned int i = 0; i < lpuCounts->size(); i++) {\n";
	programFile << doubleIndent << "int *lpuCount = lpuCounts->at(i)" << stmtSeparator;
	
	// if the count is NULL then there is no LPUs for the receiver PPU and all fields should be invalid in
	// the metadata structure entry for it
	int lpsDimensionality = gpuContextLps->getDimensionCount();
	programFile << doubleIndent << "if (lpuCount == NULL) {\n";
	for (int i = 0; i < lpsDimensionality; i++) {
		programFile << tripleIndent << "metadata.entries[i].lpuCount" << i + 1;
		programFile << " = INVALID_ID" << stmtSeparator;
	}
	programFile << tripleIndent << "metadata.entries[i].batchRangeMin = INVALID_ID" << stmtSeparator;	
	programFile << tripleIndent << "metadata.entries[i].batchRangeMax = INVALID_ID" << stmtSeparator;	
	programFile << tripleIndent << "metadata.entries[i].batchStartIndex = INVALID_ID" << stmtSeparator;
	
	programFile << doubleIndent << "} ";

	// if the LPU count is not NULL then we copy the count and batch ranges from the vector to the metadata
	// structure entry intended for the receiver PPU 
	programFile << "else {\n";
	for (int i = 0; i < lpsDimensionality; i++) {
		programFile << tripleIndent << "metadata.entries[i].lpuCount" << i + 1;
		programFile << " = lpuCount[" << i << "]" << stmtSeparator;
	}
	programFile << tripleIndent << "Range lpuRange = lpuBatchRanges->at(i)" << stmtSeparator;
	programFile << tripleIndent << "metadata.entries[i].batchRangeMin = lpuRange.min" << stmtSeparator;	
	programFile << tripleIndent << "metadata.entries[i].batchRangeMax = lpuRange.max" << stmtSeparator;
	programFile << tripleIndent << "metadata.entries[i].batchStartIndex = currentBatchStartIndex";
	programFile << stmtSeparator;
	
	// if the currnet batch range is a valid range then advance the batch starting index
	programFile << tripleIndent << "if (lpuRange.min != INVALID_ID) {\n";
	programFile << quadIndent << "currentBatchStartIndex += lpuRange.max - lpuRange.min + 1";
	programFile << stmtSeparator;
	programFile << tripleIndent << "}\n";	
	
	programFile << doubleIndent << "}\n";
 	
	programFile << indent << "}\n"; 

	// return result and close the function
	programFile << indent << "return metadata" << stmtSeparator;
	programFile << "}\n";
}

void generateSuperLpuConfigStruct(Space *gpuContextLps, std::ofstream &headerFile) {

	List<const char*> *accessedArrays = gpuContextLps->getLocallyUsedArrayNames();
	
	const char *lpsName = gpuContextLps->getName();
	headerFile << "\n" << "class Space" << lpsName << "AncestorLpuConfigs" << " {\n";
	headerFile << "  public:\n";

	// Note that there is no need for the linear ID of the LPU in this metadata structure. Nonetheless, we add
	// this property here and in the initializer function update it as the C++ compiler removes the initializer
	// function if there is no statement inside it. Then the CUDA compiler throw an incomplete type failure for
	// the entire structure
	headerFile << indent << "// a placeholder variable to avoid empty initializer function\n";
	headerFile << indent << "int lpuId" << stmtSeparator << std::endl; 

	for (int i = 0; i < accessedArrays->NumElements(); i++) {
		const char *arrayName = accessedArrays->Nth(i);
		ArrayDataStructure *array = (ArrayDataStructure *) gpuContextLps->getLocalStructure(arrayName);
		int dimensions = array->getDimensionality();
	
		// Determine how many host level super parts are there for this array; notice the special case for
		// subpartition where the array may be defined in the original LPS but not subpartitioned. In that
		// case the GPU is directly operating on parent LPS's LPU.
		int superPartCount = 0;
		DataStructure *parent = NULL;
		if (array->getSpace() != gpuContextLps) parent = array;
		else parent = array->getSource();
			
		while (!parent->getSpace()->isRoot()) {
			superPartCount++;
			parent = parent->getSource();
		}
		
		if (i > 0) {
			headerFile << std::endl;
		}

		// there should be a count, an ID, and a dimension range property for each parent link
		headerFile << indent << "// variable '" << arrayName << "' info\n";
		headerFile << indent << "int " << arrayName;
		headerFile << "PartsCount[" << superPartCount << "]";
		headerFile << "[" << dimensions << "]" << stmtSeparator; 
		headerFile << indent << "int " << arrayName << "PartIds[" << superPartCount << "]";
		headerFile << "[" << dimensions << "]" << stmtSeparator; 
		headerFile << indent << "int " << arrayName << "PartDims[" << superPartCount << "]";
		headerFile << "[" << dimensions << "][2]" << stmtSeparator; 
	}
	headerFile << " public:\n";
	headerFile << indent << "void initialize(LPU *lpu)" << stmtSeparator;
	headerFile << "}" << stmtSeparator; 
}

void generateSuperLpuConfigStructFn(Space *gpuContextLps, const char *initials, std::ofstream &programFile) {
	
	// generate the function signature
	const char *lpsName = gpuContextLps->getName();
	programFile << std::endl << "void " << initials << "::Space" << lpsName << "AncestorLpuConfigs";
	programFile << "::" << "initialize(LPU *lpu) ";

	// generate function body
	programFile << "{\n\n";
	
	// first cast the generic LPU to appropriate type
	programFile << indent << "Space" << lpsName << "_LPU *typedLpu = (Space" << lpsName << "_LPU*)";
	programFile << " lpu" << stmtSeparator;
	
	// If this function is optimized out by the C++ compiler because of lack of any assignments, the CUDA compiler
	// subsequently throws and incomplete type exception. So this statement is put here. 
	programFile << std::endl << indent;
	programFile << "// a placeholder statement to avoid the function to be optimized out\n";
	programFile << indent << "lpuId = typedLpu->id";
	programFile << stmtSeparator; 

	// process the individual arrays in sequence
	List<const char*> *accessedArrays = gpuContextLps->getLocallyUsedArrayNames();
	for (int i = 0; i < accessedArrays->NumElements(); i++) {
		const char *arrayName = accessedArrays->Nth(i);
		ArrayDataStructure *array = (ArrayDataStructure *) gpuContextLps->getLocalStructure(arrayName);
		int dimensions = array->getDimensionality();

		programFile << std::endl << indent << "// processing variable '" << arrayName << "'\n"; 
		
		// traverse the host level super parts for this array; notice the special case for subpartition where 
		// the array may be defined in the original LPS but not subpartitioned. In that case, the GPU is 
		// directly operating on parent LPS's LPU.
		DataStructure *parent = NULL;
		bool skipFirstLink = false;
		if (array->getSpace() != gpuContextLps) {
			parent = array;
			skipFirstLink = true;
		}
		else parent = array->getSource();

		int currentIndex = 0;
		while (!parent->getSpace()->isRoot()) {
				
			// iterate over the dimensions
			for (int j = 0; j < dimensions; j++) {
				
				// determine the reference point name for the proper metadata instance
				std::ostringstream metadata;
				metadata << "(&typedLpu->" << arrayName << "PartDims[" << j << "])";
				for (int k = 0; k <= currentIndex; k++) {
					if (k == 0 && skipFirstLink) continue;
					metadata << "->parent";
				}

				// assign properties from the part dimension object to proper fields of the of the 
				// current object
				programFile << indent << arrayName << "PartsCount[" << currentIndex << "]";
				programFile <<  "[" << j << "] = ";
				programFile << metadata.str() << "->count" << stmtSeparator;
				programFile << indent << arrayName << "PartIds[" << currentIndex << "]";
				programFile <<  "[" << j << "] = ";
				programFile << metadata.str() << "->index" << stmtSeparator;
				programFile << indent << arrayName << "PartDims[" << currentIndex << "]";
				programFile <<  "[" << j << "][0] = ";
				programFile << metadata.str() << "->partition.range.min" << stmtSeparator;
				programFile << indent << arrayName << "PartDims[" << currentIndex << "]";
				programFile <<  "[" << j << "][1] = ";
				programFile << metadata.str() << "->partition.range.max" << stmtSeparator;
			}

			parent = parent->getSource();
			currentIndex++;
		}
	}

	// close function body
	programFile << "}\n";
}

void generateSuperLpuConfigAggregatorStruct(Space *gpuContextLps,
                PCubeSModel *pcubesModel,
                std::ofstream &headerFile) {

	// Determine the number of metadata entries the aggregator should have. The default is 1 for normal off-
        // loading context. If the LPS is subpartitioned then the context is a PPU location sensitive. In other
        // word, specific LPUs are assigned to specific PPUs and PPU-count number of metadata entries are needed. 
        int metadataEntryCount = 1;
        if (gpuContextLps->isSubpartitionSpace()) {
                int ppsId = gpuContextLps->getPpsId();
                int gpuPpsId = pcubesModel->getGpuTransitionSpaceId();
                std::string batchSize;
                if (ppsId == gpuPpsId) {
                        metadataEntryCount = 1;
                } else if (ppsId == gpuPpsId - 1) {
                        metadataEntryCount = pcubesModel->getSMCount();
                } else {
                        metadataEntryCount = pcubesModel->getSMCount() * pcubesModel->getWarpCount();
                }
        }

        const char *lpsName = gpuContextLps->getName();
        headerFile << std::endl;
        headerFile << "class Space" << lpsName << "AncestorLpuConfigsAggregate {\n";
        headerFile << "  public:\n";
        headerFile << indent << "Space" << lpsName << "AncestorLpuConfigs entries[";
	headerFile << metadataEntryCount << "]" << stmtSeparator;
        headerFile << "}" << stmtSeparator;
}

void generateMaxPartSizeMetadataStruct(GpuExecutionContext *gpuContext, std::ofstream &headerFile) {
	
	decorator::writeSubsectionHeader(headerFile, gpuContext->getContextName());
	
	headerFile << std::endl;
	headerFile << "class " << gpuContext->getContextName() << "MaxPartSizes {\n";

	headerFile << "  public: \n";

	Space *gpuContextLps = gpuContext->getContextLps();	
	List<const char*> *arrayNames = gpuContextLps->getLocallyUsedArrayNames();
	List<const char*> *accessedArrays = string_utils::intersectLists(
			gpuContext->getVariableAccessList(), arrayNames);
	for (int i = 0; i < accessedArrays->NumElements(); i++) {
		const char *varName = accessedArrays->Nth(i);
		headerFile << indent << "int " << varName << "MaxPartSize" << stmtSeparator;
	}

	headerFile << "}" << stmtSeparator;
}

void generateAllLpuMetadataStructs(List<GpuExecutionContext*> *gpuExecutionContextList,
		PCubeSModel *pcubesModel, 
                const char *initials,
                const char *headerFileName, const char *programFileName) {

	std::cout << "Generating GPU LPU metadata reconstruction data structures\n";

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

	const char *header = "GPU LPU metadata reconstruction structures";
        decorator::writeSectionHeader(headerFile, header);
        decorator::writeSectionHeader(programFile, header);

	List<const char*> *coveredLpses = new List<const char*>;
	for (int i = 0; i < gpuExecutionContextList->NumElements(); i++) {

		// generate context specific maximum part sizes tracking metadata
		GpuExecutionContext *context = gpuExecutionContextList->Nth(i);
		generateMaxPartSizeMetadataStruct(context, headerFile);
		
		// determine if LPS specific LPU count and batch range metadata has already being generated
		Space *contextLps = context->getContextLps();
		const char *contextLpsName = contextLps->getName();
		if (string_utils::contains(coveredLpses, contextLpsName)) continue;
		coveredLpses->Append(contextLpsName);

		// generate LPU count and batch range metadata structures and function
		std::ostringstream header;
		header << "Space " << contextLpsName << " Offloading Contexts";
		decorator::writeSubsectionHeader(headerFile, header.str().c_str());
		decorator::writeSubsectionHeader(programFile, header.str().c_str());
		generateOffloadingMetadataStruct(contextLps, headerFile);
		generateMetadataAggregatorStruct(contextLps, pcubesModel, headerFile);
		generateKernelLaunchMatadataStructFn(contextLps, 
				pcubesModel, initials, headerFile, programFile);

		// generate host level ancestor LPUs' metadata holder structures and function
		generateSuperLpuConfigStruct(contextLps, headerFile);
		generateSuperLpuConfigStructFn(contextLps, initials, programFile);	
		generateSuperLpuConfigAggregatorStruct(contextLps, pcubesModel, headerFile);
	}

	headerFile.close();
	programFile.close();
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
	headerFile << indent << className;
	headerFile << "(Hashtable<DataPartitionConfig*> *partConfigMap)" << stmtSeparator;
	headerFile << indent << "int calculateLpuMemoryRequirement(LPU *lpu)" << stmtSeparator;
	headerFile << indent << "void addLpuToTheCurrentBatch(LPU *lpu" << paramSeparator;
	headerFile << "int ppuIndex)" << stmtSeparator;
	headerFile << indent << "int calculateSmMemReqForDataPart(";
	headerFile << "const char *varName" << paramSeparator;
	headerFile << "LpuDataPart *dataPart)" << stmtSeparator;
	headerFile << "}" << stmtSeparator;

	// then add implementation for the constructor and the three virtual functions inherited from the base 
	// class in the program file
	generateLpuBatchControllerConstructor(gpuContext, pcubesModel, initials, programFile);
	generateLpuBatchControllerLpuAdder(gpuContext, initials, programFile);
	generateLpuBatchControllerMemchecker(gpuContext, initials, programFile);	
	generateLpuBatchControllerSmMemReqFinder(gpuContext,pcubesModel,  initials, programFile);	
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
	programFile << "(" << paramIndent;
	programFile << "Hashtable<DataPartitionConfig*> *partConfigMap) : ";
	programFile << "LpuBatchController(partConfigMap) {\n";

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
	if (gpuContext->getContextType() == LOCATION_SENSITIVE_LPU_DISTR_CONTEXT) {
		programFile << "Space_" << gpuContextLps->getName();
		programFile << "_Threads_Per_Segment" << paramSeparator << paramIndent; 
	} else {
		programFile << '1' << paramSeparator;
	}
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
	programFile << "void " << initials << "::" << className << "::addLpuToTheCurrentBatch(LPU *lpu";
	programFile << paramSeparator << "int ppuIndex) {\n\n";

	// get the LPS specific LPU reference
	const char *lpsName = gpuContextLps->getName();
	programFile << indent << "Space" << lpsName;
	programFile << "_LPU *typedLpu = (Space" << lpsName << "_LPU*) lpu" << stmtSeparator;

	// iterate over the versionless properties and create a normal LPU data part for each and add that to the
	// batch controller only when the data part has not already been included during the processing of some
	// earlier LPU
	programFile << indent << "bool notRedundantPart = true" << stmtSeparator;
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

		programFile << indent << "notRedundantPart = dataPartTracker->addDataPart(";
		programFile << varName << "Part" << paramSeparator;
		programFile << "\"" << varName << "\"" << paramSeparator;
		programFile << "ppuIndex)" << stmtSeparator;
		programFile << indent << "if (notRedundantPart == false) delete ";
		programFile << varName << "Part" << stmtSeparator;
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

		programFile << indent << "notRedundantPart = dataPartTracker->addDataPart(";
		programFile << varName << "Part" << paramSeparator;
		programFile << "\"" << varName << "\"" << paramSeparator;
		programFile << "ppuIndex)" << stmtSeparator;
		programFile << indent << "if (notRedundantPart == false) delete ";
		programFile << varName << "Part" << stmtSeparator;
	}

	programFile << indent << "LpuBatchController::addLpuToTheCurrentBatch(lpu" << paramSeparator;
	programFile << "ppuIndex)" << stmtSeparator;
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
	
	// return the size and close the program file
	programFile << indent << "return size" << stmtSeparator;
	programFile << "}\n";
}

void generateLpuBatchControllerSmMemReqFinder(GpuExecutionContext *gpuContext,
        PCubeSModel *pcubesModel,        
	const char *initials, std::ofstream &programFile) {

	Space *gpuContextLps = gpuContext->getContextLps();
	List<GpuVarLocalitySpec*> *varLocalityInfoList = gpuContext->getVarAllocInstrList();

	std::ostringstream classNameStr;
        classNameStr << "Context" << gpuContext->getContextId() << "LpuBatchController";
        std::string className = classNameStr.str();

	// write the function signature
        programFile << std::endl;
        programFile << "int " << initials << "::" << className;
        programFile << "::calculateSmMemReqForDataPart(";
	programFile << "const char *varName" << paramSeparator;
	programFile << "LpuDataPart *dataPart) {\n\n";
	
	// generate a separate size determining condition block for each variable in the locality info list
	for (int i = 0; i < varLocalityInfoList->NumElements(); i++) {
		GpuVarLocalitySpec *varSpec =  varLocalityInfoList->Nth(i);
		const char *varName = varSpec->getVarName();
		Space *allocatorLps = varSpec->getAllocatingLps();
		bool canStoreInSm = varSpec->isSmLocalCopySupported();
		programFile << indent << "if(strcmp(varName" << paramSeparator << "\"";
		programFile << varName << "\") == 0) {\n";

		// if the variable cannot be stored in the SM memory then there is no additional memory overhead for it
		if (!canStoreInSm) {
			programFile << doubleIndent << "return 0" << stmtSeparator;

		// Otherwise several conditions should be checked
		} else {
			// if the allocator LPS and the GPU context LPSes are the same then the data part's get size 
			// function is sufficient as whole top level parts will be needed in the SM
			if (gpuContextLps == allocatorLps) {
				programFile << doubleIndent << "return dataPart->getSize()" << stmtSeparator;
			
			// Otherwise, the size needs to be determined using partition configuration information
			} else {
				// first retrieve the references of the variable in two LPSes 
				DataStructure *allocVar = allocatorLps->getLocalStructure(varName);
				DataStructure *topVar = gpuContextLps->getLocalStructure(varName);
				
				// need to retrieve all data partition specs from GPU context LPS to the allocator LPS
				// to determine the largest subpart at allocator LPS for the context LPS
				std::stack<Space*> partitionSpecs;
				while (allocVar != topVar) {
					partitionSpecs.push(allocVar->getSpace());
					allocVar = allocVar->getSource();
				}
				
				// SM allocation is an issue for arrays only; so cast the structure to an array and 
				// determine its dimensionality
				ArrayDataStructure *array = (ArrayDataStructure *) topVar;
				int dimensions = array->getDimensionality();
				
				// create an array to hold the submitted part's dimension lengths
				programFile << doubleIndent << "int partDimLengths[" << dimensions << "]";
				programFile << stmtSeparator;
				programFile << doubleIndent << "dataPart->copyDimLengths(partDimLengths)";
				programFile << stmtSeparator;

				// then iteratively derive the largest sub-part size for the allocating LPS using the
				// partition functions
				programFile << doubleIndent << "DataPartitionConfig *partConfig = NULL";
				programFile << stmtSeparator;
				while (!partitionSpecs.empty()) {
					Space *interimLps = partitionSpecs.top();
					partitionSpecs.pop();
					
					programFile << doubleIndent << "partConfig = partConfigMap->Lookup(";
					programFile << '"' << varName << "Space" << interimLps->getName();
					programFile << "Config\")" << stmtSeparator;
					programFile << doubleIndent << "partConfig->getLargestPartDimLengths";
					programFile << "(partDimLengths)" << stmtSeparator;
				}

				// finally, calculate the size requirements for the sub-part using a DataPart member
				// function and return it
				programFile << doubleIndent << "return dataPart->getSizeForSubpartsWithDimLengths";
				programFile << "(partDimLengths)" << stmtSeparator;
			}
		}
		programFile << indent << "}\n";
	}

	// if no condition has matched then this is a code generation anomally
	programFile << std::endl;
	programFile << indent << "std::cout << \"dont know how handle the variable in the GPU\\n\"";
	programFile << stmtSeparator;
	programFile << indent << "std::exit(EXIT_FAILURE)" << stmtSeparator;
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

	// a GPU code executor needs a host and a GPU instance of stage execution tracker object so that the host
	// can determine what compute stages have been executed inside the GPU (and how many times) and what did not
	headerFile << indent << initials << "::"; 
	headerFile << "Context" << gpuContext->getContextId() << "ExecutionTracker ";
	headerFile << "*stageExecutionTrackerHost, *stageExecutionTrackerGpu" << stmtSeparator;

	// finally, a GPU code executor should maintain a context specific metadata structure instance to propagate
	// host level LPU configuration information during kernel invocations; ancestor LPU metadata is needed for 
	// proper array index transformations
	const char *contextLpsName = gpuContext->getContextLps()->getName();
	headerFile << indent << initials << "::Space" << contextLpsName << "AncestorLpuConfigsAggregate ";
	headerFile << "hostLpuConfigs" << stmtSeparator; 

	// define the constructor and five functions a context specific code executor should give implementation for
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
	headerFile << indent << "int getExecutionCount(const char *stageCounterName)" << stmtSeparator;
	headerFile << indent << "void extractAncestorLpuConfigs(LPU *lpu" << paramSeparator;
	headerFile << "int ppuGroupIndex)" << stmtSeparator;
	headerFile << "}" << stmtSeparator;

	// Invoke an auxiliary function to generate all kernels releted to this execution context that will be needed
	// during the execution of the offload function. Note that, we need to generate the CUDA kernel definitions
	// before the offload function as they are not declared in the header file.
	generateGpuCodeExecutorKernelList(gpuContext, pcubesModel, initials, programFile);

	// invoke auxiliary functions to generate implementations for the defined functions 
	generateGpuCodeExecutorConstructor(gpuContext, initials, programFile);
	generateGpuCodeExecutorInitializer(gpuContext, initials, programFile);
	generateGpuCodeExecutorOffloadFn(gpuContext,  pcubesModel, initials, programFile);
	generateGpuCodeExecutorCleanupFn(gpuContext, initials, programFile);
	generateGpuCodeExecutorGetStageExecCountFn(gpuContext, initials, programFile);
	generateGpuCodeExecutorAncestorLpuExtractFn(gpuContext, initials, programFile);
}

void generateGpuCodeExecutors(List<GpuExecutionContext*> *gpuExecutionContextList,
                PCubeSModel *pcubesModel,
                const char *initials,
                const char *headerFileName, 
		const char *programFileName, const char *cudaProgramFileName) {
	
	std::cout << "Generating GPU code executors\n";

	std::ofstream headerFile, programFile, cudaProgramFile;
        headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
        cudaProgramFile.open (cudaProgramFileName, std::ofstream::out | std::ofstream::app);
        if (!headerFile.is_open()) {
                std::cout << "Unable to open output header file";
                std::exit(EXIT_FAILURE);
        }
        if (!programFile.is_open()) {
                std::cout << "Unable to open output program file";
                std::exit(EXIT_FAILURE);
        }
        if (!cudaProgramFile.is_open()) {
                std::cout << "Unable to open output CUDA program file";
                std::exit(EXIT_FAILURE);
        }

	const char *header = "GPU code executors";
        decorator::writeSectionHeader(headerFile, header);
        decorator::writeSectionHeader(programFile, header);
        decorator::writeSectionHeader(cudaProgramFile, header);

	for (int i = 0; i < gpuExecutionContextList->NumElements(); i++) {
		
		GpuExecutionContext *context = gpuExecutionContextList->Nth(i);

		// generate the compute stages's execution tracking structure and its member function
		generateStageExecutionTrackerStruct(context,
                		pcubesModel, initials, headerFile, programFile);  
		
		// generate the GPU code executor class and its member functions
		generateGpuCodeExecutorForContext(context, 
				pcubesModel, initials, headerFile, cudaProgramFile);		
	}

	headerFile.close();
	programFile.close();
	cudaProgramFile.close();
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
	programFile << paramIndent << ": GpuCodeExecutor(lpuBatchController" << paramSeparator;
	if (gpuContext->getContextType() == LOCATION_SENSITIVE_LPU_DISTR_CONTEXT) {
		programFile << "Space_" << gpuContext->getContextLps()->getName();
		programFile << "_Threads_Per_Segment";
	} else {
		programFile << "1";
	}
	programFile << ") {\n\n";

	programFile << indent << "this->arrayMetadata = arrayMetadata" << stmtSeparator;
	programFile << indent << "this->partition = partition" << stmtSeparator;
	programFile << indent << "this->taskGlobalsHost = taskGlobals" << stmtSeparator;
	programFile << indent << "this->taskGlobalsGpu = NULL" << stmtSeparator;
	programFile << indent << "this->threadLocalsHost = threadLocals" << stmtSeparator;
	programFile << indent << "this->threadLocalsGpu = NULL" << stmtSeparator;
	programFile << indent << "this->stageExecutionTrackerHost = new Context"; 
	programFile << gpuContext->getContextId() << "ExecutionTracker()" << stmtSeparator;
	programFile << indent << "this->stageExecutionTrackerGpu = NULL" << stmtSeparator;

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
	programFile << indent << "cudaMalloc((void **) &taskGlobalsGpu" << paramSeparator;
	programFile << "taskGlobalsSize)" << stmtSeparator;
	programFile << indent << "cudaMemcpy(taskGlobalsGpu" << paramSeparator;
	programFile << "taskGlobalsHost" << paramSeparator;
	programFile << "taskGlobalsSize" << paramSeparator;
	programFile << "cudaMemcpyHostToDevice)" << stmtSeparator;
	programFile << indent << "int threadLocalsSize = sizeof(*threadLocalsHost)" << stmtSeparator;
	programFile << indent << "cudaMalloc((void **) &threadLocalsGpu" << paramSeparator;
	programFile << "threadLocalsSize)" << stmtSeparator;
	programFile << indent << "cudaMemcpy(threadLocalsGpu" << paramSeparator;
	programFile << "threadLocalsHost" << paramSeparator;
	programFile << "threadLocalsSize" << paramSeparator;
	programFile << "cudaMemcpyHostToDevice)" << stmtSeparator;

	// reset the host instance of stage execution tracker and stage it into the GPU card memory
	programFile << std::endl;
	programFile << indent << "stageExecutionTrackerHost->resetCounters()" << stmtSeparator;
	programFile << indent << "int execTrackerSize = sizeof(*stageExecutionTrackerHost)";
	programFile << stmtSeparator;
	programFile << indent << "cudaMalloc((void **) &stageExecutionTrackerGpu" << paramSeparator;
	programFile << "execTrackerSize)" << stmtSeparator;
	programFile << indent << "cudaMemcpy(stageExecutionTrackerGpu" << paramSeparator;
	programFile << "stageExecutionTrackerHost" << paramSeparator;
	programFile << "execTrackerSize" << paramSeparator;
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

	// retrieve references for the arguments of the kernel function
	Space *contextLps = gpuContext->getContextLps();
	const char *lpsName = contextLps->getName();
	// generate a batch metadata configuration object
	programFile << indent << "// generating arguments for kernel function parameters\n";
	programFile << indent << initials << "::Space" << lpsName << "GpuAggregateMetadata "; 
	programFile << "launchMetadata" << paramIndent;
	programFile << indent << " = getLaunchMetadata(lpuCountVector" << paramSeparator;
	programFile << "lpuBatchRangeVector)" << stmtSeparator;
	// generate buffer references for all properties
	List<const char*> *arrayNames = contextLps->getLocallyUsedArrayNames();
	List<const char*> *accessedArrays = string_utils::intersectLists(
			gpuContext->getVariableAccessList(), arrayNames);
	for (int i = 0; i < accessedArrays->NumElements(); i++) {
		const char *arrayName = accessedArrays->Nth(i);
		programFile << indent << "GpuBufferReferences *" << arrayName << "Buffers = ";
		programFile << "lpuBatchController->getGpuBufferReferences(\"";
		programFile << arrayName << "\")" << stmtSeparator;
	}
	programFile << std::endl;

	// determine dynamic memory requirement for the kernels of this GPU execution context and initialize
	// a max part size metadata that will be used as an arguments for the CUDA kernels
	List<GpuVarLocalitySpec*> *varLocalitySpecList = gpuContext->getVarAllocInstrList();
	programFile << indent << "// determining dynamic shared memory requirements\n";	
	const char *contextName = gpuContext->getContextName();
	programFile << indent << "int dynamicSharedMemorySize = 0" << stmtSeparator;
	programFile << indent << contextName << "MaxPartSizes maxPartSizes" << stmtSeparator;
	for (int i = 0; i < varLocalitySpecList->NumElements(); i++) {
		GpuVarLocalitySpec *varSpec = varLocalitySpecList->Nth(i);
		const char *arrayName = varSpec->getVarName();
		programFile << indent << "int " << arrayName << "MaxSize = lpuBatchController->";
		programFile << "getMaxPartSizeForProperty(\"" << arrayName << "\")";
		programFile << stmtSeparator;
		programFile << indent << "maxPartSizes." << arrayName << "MaxPartSize = ";
		programFile << "getAlignedPartSize(" << arrayName << "MaxSize)" << stmtSeparator;
		if (varSpec->isSmLocalCopySupported()) {
			if (!varSpec->doesReqPerWarpInstances()) {
				programFile << indent << "dynamicSharedMemorySize += getAlignedPartSize(";
				programFile << arrayName << "MaxSize)" << stmtSeparator;
			} else {
				programFile << indent << "dynamicSharedMemorySize += getAlignedPartSize(";
				programFile << arrayName << "MaxSize)";
				programFile << " * WARP_COUNT" << stmtSeparator;
			}
		}
	}
	programFile << std::endl;

	// determine common kernel launch grid and block configuration
	programFile << indent << "// common kernel launch configuration\n";
	programFile << indent << "int gridConfig = SM_COUNT" << stmtSeparator;
	programFile << indent << "int blockConfig = WARP_SIZE * WARP_COUNT" << stmtSeparator;
	programFile << std::endl; 

	// invoke GPU context executor configuration to generate the logic of the subflow as sequences of kernel
	// invocations
	gpuContext->generateContextFlowImplementerCode(programFile, 1);

	// at the end cleanup the buffer reference pointers
	programFile << indent << "// cleaning up buffer reference pointers\n";	
	for (int i = 0; i < accessedArrays->NumElements(); i++) {
		const char *arrayName = accessedArrays->Nth(i);
		programFile << indent << "delete " << arrayName << "Buffers" << stmtSeparator;
	}
	
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

	programFile << indent << "int execTrackerSize = sizeof(*stageExecutionTrackerHost)";
	programFile << stmtSeparator;
	programFile << indent << "cudaMemcpy(stageExecutionTrackerHost" << paramSeparator;
	programFile << "stageExecutionTrackerGpu" << paramSeparator;
	programFile << "execTrackerSize" << paramSeparator;
	programFile << "cudaMemcpyDeviceToHost)" << stmtSeparator;

	// then invoke the superclass's cleanup function to tear down the GPU device context
	programFile << '\n' << indent << "GpuCodeExecutor::cleanup()" << stmtSeparator;

	programFile << "}\n"; 
}

void generateGpuCodeExecutorGetStageExecCountFn(GpuExecutionContext *gpuContext,   
                const char *initials, std::ofstream &programFile) {
	
	std::ostringstream classNameStr;
	classNameStr << "Context" << gpuContext->getContextId() << "CodeExecutor";
	std::string className = classNameStr.str();

	programFile << std::endl;
	programFile << "int " << initials << "::" << className;
	programFile << "::getExecutionCount(const char *stageCounterName) {\n";
	programFile << indent << "return stageExecutionTrackerHost->getExecutionCount(stageCounterName)";
	programFile << stmtSeparator;
	programFile << "}\n";
}

void generateGpuCodeExecutorAncestorLpuExtractFn(GpuExecutionContext *gpuContext,
                const char *initials, std::ofstream &programFile) {
	
	std::ostringstream classNameStr;
	classNameStr << "Context" << gpuContext->getContextId() << "CodeExecutor";
	std::string className = classNameStr.str();

	Space *contextLps = gpuContext->getContextLps();
	const char *lpsName = contextLps->getName();
	
	programFile << std::endl;
	programFile << "void " << initials << "::" << className; 
	programFile << "::extractAncestorLpuConfigs(LPU *lpu" << paramSeparator;
	programFile << "int ppuGroupIndex) {\n";
	programFile << indent << "hostLpuConfigs.entries[ppuGroupIndex].initialize(lpu)";
	programFile << stmtSeparator;
	programFile << "}\n";
}

void generateStageExecutionTrackerStruct(GpuExecutionContext *gpuContext,
                PCubeSModel *pcubesModel, 
                const char *initials,
                std::ofstream &headerFile, std::ofstream &programFile) {

	const char *contextName = gpuContext->getContextName();
	std::ostringstream subsectionHeader;
	subsectionHeader << contextName << " stage execution tracker";
	const char *subHeader =strdup(subsectionHeader.str().c_str());	
        decorator::writeSubsectionHeader(headerFile, subHeader);
        decorator::writeSubsectionHeader(programFile, subHeader);

	int topmostGpuPps = pcubesModel->getGpuTransitionSpaceId();
	List<ExecutionStage*> *executeStageList = gpuContext->getComputeStagesOfFlowContext();

	// declare the tracker structure in the header file
	std::ostringstream classNameStr;
	classNameStr << "Context" << gpuContext->getContextId() << "ExecutionTracker";
	std::string className = classNameStr.str();
	headerFile << "class " << className << "{";
	
	// their will be one execution tracker entry per execution stage per PPU that will execute that stage
	headerFile << "\n  public:\n";
	for (int i = 0; i < executeStageList->NumElements(); i++) {
		ExecutionStage *stage = executeStageList->Nth(i);
		Space *lps = stage->getSpace();
		int ppsId = lps->getPpsId();
		int ppuCount = 1;
		if (topmostGpuPps - ppsId > 0) {
			ppuCount *= pcubesModel->getSMCount();
		} 
		if (topmostGpuPps - ppsId > 1) {
			ppuCount *= pcubesModel->getWarpCount();
		}
		headerFile << indent << "int " << stage->getName() << "ExecutionCounter[" << ppuCount << "]";
		headerFile << stmtSeparator;
	}

	// declare a function to reset all the counters and another one to determine how many times a particular stage
	// has been executed
	headerFile << "  public:\n";
	headerFile << indent << "void resetCounters()" << stmtSeparator;
	headerFile << indent << "int getExecutionCount(const char *counterName)" << stmtSeparator;
	headerFile << "}" << stmtSeparator;

	// define the two functions in the program file
	generateStateExecTrackerCounterResetFn(gpuContext, pcubesModel, initials, programFile);  
	generateStateExecTrackerCounterGatherFn(gpuContext, pcubesModel, initials, programFile);  
}

void generateStateExecTrackerCounterResetFn(GpuExecutionContext *gpuContext,
                PCubeSModel *pcubesModel,
                const char *initials, std::ofstream &programFile) {
	
	int topmostGpuPps = pcubesModel->getGpuTransitionSpaceId();
	List<ExecutionStage*> *executeStageList = gpuContext->getComputeStagesOfFlowContext();
	
	std::ostringstream classNameStr;
	classNameStr << "Context" << gpuContext->getContextId() << "ExecutionTracker";
	std::string className = classNameStr.str();
	programFile << std::endl;
	programFile << "void " << initials << "::" << className << "::resetCounters() {\n";

	// iterate over all the execution stages of the GPU context's computation flow	
	for (int i = 0; i < executeStageList->NumElements(); i++) {
		
		// determine the name of the counter
		ExecutionStage *stage = executeStageList->Nth(i);
		std::ostringstream counterName;
		counterName << stage->getName() << "ExecutionCounter";
		
		// determine the number of entries the counter has
		Space *lps = stage->getSpace();
		int ppsId = lps->getPpsId();
		int ppuCount = 1;
		if (topmostGpuPps - ppsId > 0) {
			ppuCount *= pcubesModel->getSMCount();
		} 
		if (topmostGpuPps - ppsId > 1) {
			ppuCount *= pcubesModel->getWarpCount();
		}

		// set all entries of the counter to zero
		programFile << indent << "for (int i = 0; i < " << ppuCount << "; i++) ";
		programFile << counterName.str() << "[i] = 0" << stmtSeparator;
	}

	programFile << "}\n";
}
    
void generateStateExecTrackerCounterGatherFn(GpuExecutionContext *gpuContext,
                PCubeSModel *pcubesModel,
                const char *initials, std::ofstream &programFile) {
	
	int topmostGpuPps = pcubesModel->getGpuTransitionSpaceId();
	List<ExecutionStage*> *executeStageList = gpuContext->getComputeStagesOfFlowContext();
	
	std::ostringstream classNameStr;
	classNameStr << "Context" << gpuContext->getContextId() << "ExecutionTracker";
	std::string className = classNameStr.str();
	programFile << std::endl;
	programFile << "int " << initials << "::" << className;
	programFile << "::getExecutionCount(const char *counterName) {\n";

	// have a series of if-else block for different counters to compare the argument against
	for (int i = 0; i < executeStageList->NumElements(); i++) {

                // determine the name of the counter
                ExecutionStage *stage = executeStageList->Nth(i);
                std::ostringstream counterName;
                counterName << stage->getName() << "ExecutionCounter";

		// generate the if block
		programFile << indent;
		if (i > 0) programFile << " else ";
		programFile << "if (strcmp(\"" << counterName.str() << "\"" << paramSeparator;
		programFile << "counterName) == 0) {\n";
		
		// determine the number of entries in the counter
		Space *lps = stage->getSpace();
		int ppsId = lps->getPpsId();
		int ppuCount = 1;
		if (topmostGpuPps - ppsId > 0) {
			ppuCount *= pcubesModel->getSMCount();
		} 
		if (topmostGpuPps - ppsId > 1) {
			ppuCount *= pcubesModel->getWarpCount();
		}
	
		// sum up the execution counter's entries
		programFile << doubleIndent << "int total = 0" << stmtSeparator;
		programFile << doubleIndent << "for (int i = 0; i < " << ppuCount << "; i++) {\n";
		programFile << tripleIndent << "total += " << counterName.str() << "[i]" << stmtSeparator;
		programFile << doubleIndent << "}\n";	
		programFile << doubleIndent << "return total" << stmtSeparator;
		programFile << indent << "}";	 
	}
	programFile << "\n";

	programFile << indent << "return 0" << stmtSeparator;
	programFile << "}\n";
}

void generateGpuCodeExecutorKernel(CompositeStage *kernelDef,
                GpuExecutionContext *gpuContext,
                PCubeSModel *pcubesModel,
                const char *initials, std::ofstream &programFile) {

	const char *kernelName = kernelDef->getName();

	// define the kernel function signature
	programFile << std::endl << "__global__ void " << kernelName << "(";
	// first there are four default scalar parameters
	programFile << initials << "::ArrayMetadata arrayMetadata" << paramSeparator << paramIndent;
	programFile << string_utils::toUpper(initials) << "Partition partition" << paramSeparator;
	programFile << paramIndent << initials << "::TaskGlobals *taskGlobals" << paramSeparator;
	programFile << paramIndent << initials << "::ThreadLocals *threadLocals" << paramSeparator;
	// then add another parameter for the stage execution tracker GPU pointer
	programFile << paramIndent << initials << "::Context";
	programFile << gpuContext->getContextId() << "ExecutionTracker *stageExecutionTracker";
	programFile << paramSeparator;
	// then add another parameter for the launch configuration metadata 
	Space *contextLps = gpuContext->getContextLps();
	const char *lpsName = contextLps->getName();
	programFile << paramIndent << initials << "::Space" << lpsName << "GpuAggregateMetadata "; 
	programFile << "launchMetadata";
	// then add another parameter for host level LPUs metadata
	programFile << paramSeparator << paramIndent << initials << "::";
	programFile << "Space" << lpsName << "AncestorLpuConfigsAggregate hostLpuConfigs";
	// then add another parameter for the maximum sizes of data parts for different variables used in the kernels
	programFile << paramSeparator << paramIndent << initials << "::";
	programFile << gpuContext->getContextName() << "MaxPartSizes maxPartSizes";
	// then add buffer reference parameters for the arrays being used in the GPU offloading execution context
	List<const char*> *arrayNames = contextLps->getLocallyUsedArrayNames();
	List<const char*> *accessedArrays = string_utils::intersectLists(
			gpuContext->getVariableAccessList(), arrayNames);
	for (int i = 0; i < accessedArrays->NumElements(); i++) {
		const char *arrayName = accessedArrays->Nth(i);
		programFile << paramSeparator << paramIndent;
		programFile << "GpuBufferReferences " << arrayName << "Buffers";
	}

	// begin kernel function body
	programFile << ") {\n\n";

	// setup PPU identification
	programFile << indent << "// PPU and thread identification\n";
	programFile << indent << "int smId = blockIdx.x" << stmtSeparator;
	programFile << indent << "int warpId = threadIdx.x / WARP_SIZE" << stmtSeparator;
	programFile << indent << "int globalWarpId = smId * WARP_COUNT + warpId" << stmtSeparator;
	programFile << indent << "int threadId = threadIdx.x % WARP_SIZE" << stmtSeparator;
	programFile << std::endl;

	// declare the dynamic memory panel
	programFile << indent << "// Dynamic memory panel declaration\n";
	programFile << indent << "extern __shared__ char memoryPanel[]" << stmtSeparator;
	programFile << indent << "__shared__ int panelIndex" << stmtSeparator;
	programFile << std::endl;

	// then invoke the kernel generation logic of the GPU context class to fill in the rest of the GPU kernel
	gpuContext->generateGpuKernel(kernelDef, programFile, pcubesModel);
	
	// close kernel function body
	programFile << "}\n";
}

void generateGpuCodeExecutorKernelList(GpuExecutionContext *gpuContext,
                PCubeSModel *pcubesModel,
                const char *initials, std::ofstream &programFile) {

	// the name transformer needs to be set to GPU mode so that the variable names used inside GPU kernels get
	// prefix and suffixes that are appropriate for the GPU environment
	ntransform::HybridNameTransformer *transformer = (ntransform::HybridNameTransformer*) 
			ntransform::NameTransformer::transformer;
	transformer->setToGpuMode();	

	List<KernelGroupConfig*> *kernelGroupConfigList = gpuContext->getKernelConfigList();
	const char *contextName = gpuContext->getContextName();
	for (int i = 0; i < kernelGroupConfigList->NumElements(); i++) {
		
		KernelGroupConfig *groupConfig = kernelGroupConfigList->Nth(i);
		int groupId = groupConfig->getGroupId();
		List<CompositeStage*> *kernelDefList = groupConfig->getKernelDefinitions();

		for (int j = 0; j < kernelDefList->NumElements(); j++) {

			CompositeStage *kernelDef = kernelDefList->Nth(j);
			
			// generate a system-wide unique name for the CUDA kernel
			std::ostringstream kernelName;
			kernelName << initials << "_" << contextName;
			kernelName << "_Group" << groupId << "_Kernel" << j;
			kernelDef->setName(strdup(kernelName.str().c_str()));

			generateGpuCodeExecutorKernel(kernelDef, 
					gpuContext, pcubesModel, initials, programFile);
		}
	}
	
	// reset the name transformer to host mode
	transformer->setToHostMode();
}

void generateGpuExecutorMapFn(List<GpuExecutionContext*> *gpuExecutionContextList,
                const char *initials,
                const char *headerFileName, const char *programFileName) {
	
	std::cout << "Generating GPU code executor map for the batch PPU controller\n";

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

	const char *header = "GPU code executors setup";
        decorator::writeSubsectionHeader(headerFile, header);
        decorator::writeSubsectionHeader(programFile, header);

	// generate the function header
	std::ostringstream fnHeader;
	fnHeader << "getGpuCodeExecutorMap(";
	fnHeader << initials << "::ArrayMetadata arrayMetadata";
	const char *upperInitials = string_utils::toUpper(initials);
	fnHeader << paramSeparator << paramIndent << upperInitials << "Partition partition";
	fnHeader << paramSeparator << paramIndent << initials << "::TaskGlobals *taskGlobals";
	fnHeader << paramSeparator << paramIndent << initials << "::ThreadLocals *threadLocals";
	fnHeader << paramSeparator << paramIndent << "Hashtable<DataPartitionConfig*> *partConfigMap";
	fnHeader << paramSeparator << paramIndent << "std::ofstream &logFile)";

	// write the function header in the header and program files
	headerFile << "Hashtable<GpuCodeExecutor*> *" << fnHeader.str() << stmtSeparator;
	programFile << std::endl;
	programFile << "Hashtable<GpuCodeExecutor*> *" << initials << "::" << fnHeader.str();

	// starts function body by instantiating the GPU code executor map
	programFile << " {\n\n" << indent;
	programFile << "Hashtable<GpuCodeExecutor*> *executorMap = new Hashtable<GpuCodeExecutor*>";
	programFile << stmtSeparator;

	// iterate over the GPU execution contexts in the list and crate a LPU Batch controller and GPU code
	// executor for each context and put them in the map 
	for (int i = 0; i < gpuExecutionContextList->NumElements(); i++) {
		GpuExecutionContext *context = gpuExecutionContextList->Nth(i);
		int contextId = context->getContextId();
		programFile << std::endl;
		programFile << indent << "LpuBatchController *lpuBatchController" << contextId << " = ";
		programFile << "new Context" << contextId << "LpuBatchController(partConfigMap)";
		programFile << stmtSeparator;
		programFile << indent << "lpuBatchController" << contextId << "->setLogFile(&logFile)";
		programFile << stmtSeparator;
		programFile << indent << "GpuCodeExecutor *gpuCodeExecutor" << contextId << " = ";
		programFile << "new Context" << contextId << "CodeExecutor(";
		programFile << "lpuBatchController" << contextId << paramSeparator << paramIndent;
		programFile << "arrayMetadata" << paramSeparator << "partition" << paramSeparator;
		programFile << "taskGlobals" << paramSeparator << "threadLocals)" << stmtSeparator;
		programFile << indent << "gpuCodeExecutor" << contextId << "->setLogFile(&logFile)";
		programFile << stmtSeparator;
		programFile << indent << "executorMap->Enter(\"" << context->getContextName() << "\"";
		programFile << paramSeparator << "gpuCodeExecutor" << contextId << ")" << stmtSeparator;
	}

	// return the map and close the function
	programFile << indent << "return executorMap" << stmtSeparator;
	programFile << "}\n"; 

	headerFile.close();
	programFile.close();
}
