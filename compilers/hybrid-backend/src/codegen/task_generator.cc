#include "task_generator.h"

#include <iostream>
#include <sstream>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <deque>

#include "lpu_generation.h"
#include "code_generator.h"
#include "thread_state_mgmt.h"
#include "space_mapping.h"
#include "name_transformer.h"
#include "compute_translator.h"
#include "pthread_mgmt.h"
#include "sync_mgmt.h"
#include "task_invocation.h"
#include "memory_mgmt.h"
#include "file_io.h"
#include "communication.h"
#include "environment_mgmt.h"
#include "gpu_execution.h"

#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "../utils/string_utils.h"
#include "../utils/decorator_utils.h"
#include "../utils/code_constant.h"
#include "../utils/properties.h"
#include "../syntax/ast.h"
#include "../syntax/ast_def.h"
#include "../syntax/errors.h"
#include "../syntax/ast_task.h"
#include "../syntax/ast_type.h"
#include "../static-analysis/task_global.h"
#include "../static-analysis/sync_stat.h"
#include "../static-analysis/gpu_execution_ctxt.h"
#include "../semantics/task_space.h"

TaskGenerator::TaskGenerator(TaskDef *taskDef,
                const char *outputDirectory,
                const char *mappingFile,
		const char *processorFile) {

	this->taskDef = taskDef;
	this->mappingFile = mappingFile;
	this->processorFile = processorFile;

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

	std::ostringstream cudaProgramFileStr;
	cudaProgramFileStr << outputDirectory;
	cudaProgramFileStr << taskNameLower << "_offload.cu";
	cudaProgramFile = strdup(cudaProgramFileStr.str().c_str());

	initials = string_utils::getInitials(taskDef->getName());
	initials = string_utils::toLower(initials);

	mappingRoot = NULL;
	segmentedPPS = 0;
	syncManager = NULL;
}

const char *TaskGenerator::getHeaderFileName(TaskDef *taskDef) {
	std::ostringstream headerFileStr;
	const char *taskName = string_utils::replaceChar(taskDef->getName(), ' ', '_');
        const char *taskNameLower = string_utils::toLower(taskName);
        headerFileStr << taskNameLower << ".h";
        return strdup(headerFileStr.str().c_str());
}

const char *TaskGenerator::getNamespace(TaskDef *taskDef) {
	const char *initials = string_utils::getInitials(taskDef->getName());
	return string_utils::toLower(initials);
}

void TaskGenerator::generate(List<PCubeSModel*> *pcubesModels) {

	std::cout << "\n-----------------------------------------------------------------\n";
	std::cout << "Translating task: " << taskDef->getName();
	std::cout << "\n-----------------------------------------------------------------\n";

	initializeOutputFiles(headerFile, programFile, initials);

	// interpret mapping configuration
        PartitionHierarchy *lpsHierarchy = taskDef->getPartitionHierarchy();
        MappingNode *mappingConfig = parseMappingConfiguration(taskDef->getName(),
                        mappingFile, lpsHierarchy, pcubesModels);
	this->mappingRoot = mappingConfig;
	Space *rootLps = lpsHierarchy->getRootSpace();

	// retrieve the description of the PCubeS model that has been used for the LPS mapping of the current task

	PCubeSModel *pcubesModel = lpsHierarchy->getPCubeSModel();
	List<PPS_Definition*> *pcubesConfig = pcubesModel->getPpsDefinitions();

	// determine where memory segmentation happens in the hardware
	for (int i = 0; i < pcubesConfig->NumElements(); i++) {
		PPS_Definition *pps = pcubesConfig->Nth(i);
		if (pps->segmented) {
			segmentedPPS = pps->id;	
			break;
		} 
	}
	if (segmentedPPS == 0) {
		std::cout << "No segmentation observed in the PCubeS configuration\n";
		// assign the topmost PPS the segmentation marker to make the code work for non segmented memory 
		// architectures
		segmentedPPS = pcubesConfig->Nth(0)->id;
	}

	// do back-end architecture dependent static analyses of the task
	lpsHierarchy->performAllocationAnalysis(segmentedPPS);
	if (ReportError::NumErrors() > 0) {
		std::exit(EXIT_FAILURE);
	}

	// determine if the task is using the hybrid model and if YES what is the top-most LPS/PPS where host to
	// GPU LPU offloading will be done; below that PPS, PPUs will not be treated as independent entities from
	// the host and no state management will be done for them
	bool hybridMapping = taskDef->usingHybridModel();
	int passivePpsThreshold = -1;
	List<GpuExecutionContext*> *gpuContextList = NULL;
	if (hybridMapping) {
		std::cout << "Identifying parts of the computation to be offloaded to GPUs\n";
		gpuContextList = taskDef->getGpuExecutionContexts();
		for (int i = 0; i < gpuContextList->NumElements(); i++) {
			Space *offloadingLps = gpuContextList->Nth(i)->getContextLps();
			if (offloadingLps->getPpsId() < passivePpsThreshold 
					|| passivePpsThreshold == -1) {
				passivePpsThreshold = offloadingLps->getPpsId();
			}
		}
		// put the execution contexts from the list into a static map for later access during computation
		// flow translation process
		Hashtable<GpuExecutionContext*> *gpuContextMap = new Hashtable<GpuExecutionContext*>;
		for (int i = 0; i < gpuContextList->NumElements(); i++) {
			GpuExecutionContext *gpuContext = gpuContextList->Nth(i);
			// generate the GPU kernel configurations for each execution context at the same time
			gpuContext->generateKernelConfigs(pcubesModel);
			gpuContext->describe(1);
			gpuContextMap->Enter(gpuContext->getContextName(), gpuContext);
		}
 		GpuExecutionContext::gpuContextMap = gpuContextMap;
	}
	pcubesModel->setPassivePpsThreshold(passivePpsThreshold);

	// for host-only execution, we need to know how the physical cores of the machines are numbered to pin a
	// thread to a specific core
	if (!hybridMapping) {
		// generate a constant array for processor ordering in the hardware
		generateProcessorOrderArray(headerFile, processorFile);

	// for hybrid execution generate the batch configuration constants for offloading LPUs to the GPU and set
	// up a separate program file to hold the GPU/CUDA related code
	} else {
		generateBatchConfigurationConstants(headerFile, pcubesModel);
		initializeCudaProgramFile(initials, headerFile, cudaProgramFile);
	}

	// generate constansts needed for various reasons
        generateLPSConstants(headerFile, mappingConfig);
        generatePPSCountConstants(headerFile, pcubesModel);
        generateThreadCountConstants(headerFile, mappingConfig, pcubesModel);

	// generate data structures and functions for task environment management 
	generatePrintFnForLpuDataStructures(initials, programFile, mappingConfig);
        List<const char*> *envLinkList = generateArrayMetadataAndEnvLinks(headerFile, 
			mappingConfig, taskDef->getEnvironmentLinks());
	generateFnForMetadataAndEnvLinks(taskDef->getName(), 
			initials, programFile, mappingConfig, envLinkList);
        generateLpuDataStructures(headerFile, mappingConfig);
	List<TaskGlobalScalar*> *globalScalars 
			= TaskGlobalCalculator::calculateTaskGlobals(taskDef);
	generateClassesForGlobalScalars(headerFile, globalScalars, rootLps);

	// generate functions related to memory management
	const char *upperInitials = string_utils::getInitials(taskDef->getName());
	genRoutinesForTaskPartitionConfigs(headerFile, programFile, upperInitials, lpsHierarchy);
	genTaskMemoryConfigRoutine(taskDef, headerFile, programFile, upperInitials);
	
	// generate functions and classes for I/O
	generateReaderWriters(headerFile, initials, taskDef);
	generateReadersMap(headerFile, programFile, initials, taskDef);
	generateWritersMap(headerFile, programFile, initials, taskDef);

	// generate routines to contruct LPUs
	Hashtable<List<PartitionParameterConfig*>*> *partParamConfigMap 
		= generateLPUCountFunctions(headerFile, programFile, initials, mappingRoot);
	generateAllLpuConstructionFunctions(headerFile, programFile, initials, mappingRoot);

	// generate thread management functions and classes; these are needed in both hybrid and host only models
	// of execution as part of underlying state management infrastructure for LPU generation and traversal 
        generateFnForThreadIdsAllocation(headerFile, 
			programFile, initials, mappingConfig, pcubesConfig);
        generateFnForThreadIdsAdjustment(headerFile, programFile, initials, mappingConfig);
        generateThreadStateImpl(headerFile, programFile, mappingConfig, partParamConfigMap);

	// generate setup functions for hybrid execution management if hybrid model being used
	if (hybridMapping) {
		generateLpuBatchVectorSizeSetupRoutine(headerFile, 
				programFile, initials, mappingRoot);
		generateGpuExecutorMapFn(gpuContextList, initials, headerFile, programFile);
	}

	// generate synchronization primitives and their initialization functions; sync primitives are not needed
	// for hybrid mapping as host in that case works as a single threaded program
	if (!hybridMapping) {
		syncManager = new SyncManager(taskDef, headerFile, programFile, initials);
		syncManager->processSyncList();
		syncManager->generateSyncPrimitives();
		syncManager->generateSyncInitializerFn();
		syncManager->generateSyncStructureForThreads();
		syncManager->generateFnForThreadsSyncStructureInit();
	}

	// generate communication related data structures and functions
	std::cout << "Generating communication related data structures and functions\n";
	generateFnsForDistributionTrees(headerFile, programFile, taskDef, pcubesConfig);
	List<CommunicationCharacteristics*> *commCharacterList 
			= generateFnsForConfinementConstrConfigs(headerFile, 
					programFile, taskDef, pcubesConfig);
	int communicatorCount = commCharacterList->NumElements();
	generateAllDataExchangeFns(headerFile, programFile, taskDef, commCharacterList);
	if (communicatorCount > 0) {
		generateAllCommunicators(headerFile, 
				programFile, taskDef, commCharacterList);
		generateCommunicatorMapFn(headerFile, 
				programFile, taskDef, commCharacterList);
		generateCommunicationExcludeFn(headerFile, 
				programFile, taskDef, commCharacterList);
	}

	// generate environment management data structures and functions
	generateTaskEnvironmentClass(taskDef, initials, headerFile, programFile);
	generateFnToInitEnvLinksFromEnvironment(taskDef, initials, headerFile, programFile);
	generateFnToPreconfigureLpsAllocations(taskDef, initials, headerFile, programFile);
	generateFnToCopyBackNonArrayVars(taskDef, initials, headerFile, programFile);

	// generate task executor and associated functions
	generateTaskExecutor(this);

	// initialize the variable transformation map that would be used to translate the code inside initialize 
	// and compute blocks
	ntransform::NameTransformer::setTransformer(taskDef);	

	// translate the initialize block of the task into a function
	generateInitializeFunction(headerFile, programFile, initials, 
        		envLinkList, taskDef, mappingConfig->mappingConfig->LPS);

	// generate functions for all compute stages in the source code
	generateFnsForComputation(taskDef, headerFile, programFile, initials);

	// generate pthreads execution and management related functions for the host only task's LPS mapping
	if (!hybridMapping) {
		// generate run function for threads
		generateThreadRunFunction(taskDef, headerFile, 
				programFile, initials, mappingConfig, 
				syncManager->involvesSynchronization(), communicatorCount);

		// generate data structure and functions for Pthreads
		generateArgStructForPthreadRunFn(taskDef->getName(), headerFile);
		generatePThreadRunFn(headerFile, programFile, initials);
	
	// for the hybrid mapping, generate a function for simulating the task's computation flow in batches of 
	// LPUs and the implementation classes for all GPU sub-flow context executors
	} else {
		generateAllLpuMetadataStructs(gpuContextList,
                		pcubesModel, initials, headerFile, programFile);
		generateLpuBatchControllers(gpuContextList, 
				pcubesModel, initials, headerFile, programFile);
		
		// this part is using the cuda program file as it generates CUDA code 
		generateGpuCodeExecutors(gpuContextList, 
				pcubesModel, initials, headerFile, cudaProgramFile);

		generateBatchComputeFunction(taskDef, 
				headerFile, programFile, initials, communicatorCount);
	}

	closeNameSpace(headerFile);
}

List<const char*> *TaskGenerator::initiateEnvLinks(std::ofstream &stream) {
	
        PartitionHierarchy *lpsHierarchy = taskDef->getPartitionHierarchy();
	Space *rootLps = lpsHierarchy->getRootSpace();

	List<const char*> *externalEnvLinks = new List<const char*>;
	
	stream << indent << "// initializing variables that are environmental links \n";
	stream << indent << "std::cout << \"initializing environmental links\\n\"" << stmtSeparator;

	List<EnvironmentLink*> *envLinks = taskDef->getEnvironmentLinks();
	for (int i = 0; i < envLinks->NumElements(); i++) {
                EnvironmentLink *link = envLinks->Nth(i);
		
		// TODO instead of returning the isExternal clause should result in creating optional input prompts
		// for variables that might or might not be external as we support link-or-create flag for an
		// environment variable. Current logic only deals with linked variables spefified by the link flag.  
                if (!link->isExternal()) continue;

                const char *linkName = link->getVariable()->getName();
		externalEnvLinks->Append(linkName);
                DataStructure *structure = rootLps->getLocalStructure(linkName);
                ArrayDataStructure *array = dynamic_cast<ArrayDataStructure*>(structure);
                if (array != NULL) {
			
			ArrayType *arrayType = (ArrayType*) array->getType();
                        Type *elemType = arrayType->getTerminalElementType();
			int dimensionCount = array->getDimensionality();

			if (isUnsupportedInputType(elemType, linkName)) {
				stream << indent << "//TODO put custom initializing code for " << linkName << "\n";
			} else {	
				// first generate a prompt to determine if the user wants to read the array from a
				// file or to randomly initialize it
				stream << indent << "if (outprompt::getYesNoAnswer(\"Want to read array";
				stream << " \\\"" << linkName << "\\\" from a file?\"";
				stream << ")) {\n";

				// if the response is yes then generate a prompt for reading the array from a file 
				stream << doubleIndent;
				stream << "envLinks." << linkName << " = ";
				stream << "inprompt::readArrayFromFile ";
				stream << '<' << elemType->getName() << "> ";
				stream << "(\"" << linkName << "\"" << paramSeparator;
				stream << std::endl << doubleIndent << doubleIndent;
				stream << dimensionCount << paramSeparator;
				stream << "envLinks." << linkName << "Dims)" << stmtSeparator;
					
				// otherwise, generate code for randomly initialize the array
				stream << indent << "} else {\n";	
				// create a prompt to get the dimensions information for the variable under concern
				stream << doubleIndent;
				stream << "inprompt::readArrayDimensionInfo(\"" << linkName << "\"" << paramSeparator;
				stream << dimensionCount << paramSeparator;
				stream << "envLinks." << linkName << "Dims)" << stmtSeparator;
				// then allocate an array for the variable
				stream << doubleIndent;
				stream << "envLinks." << linkName << " = allocate::allocateArray ";
				stream << '<' << elemType->getName() << "> ";	
				stream << '(' << dimensionCount << paramSeparator;
				stream << "envLinks." << linkName << "Dims)" << stmtSeparator;
				// finally randomly initialize the array
				stream << doubleIndent << "allocate::randomFillPrimitiveArray ";
				stream << '<' << elemType->getName() << "> ";
				stream << "(envLinks." << linkName << paramSeparator;	
				stream << std::endl << doubleIndent << doubleIndent;
				stream << dimensionCount << paramSeparator;
				stream << "envLinks." << linkName << "Dims)" << stmtSeparator;
			
				stream << indent << "}\n";	
			}
		} else {
			Type *type = structure->getType();
			if (isUnsupportedInputType(type, linkName)) {
				stream << indent << "//TODO put custom initializing code for " << linkName << "\n";
			} else {
				stream << indent;
				stream << "envLinks." << linkName << " = ";
				if (type == Type::boolType) {
					stream << "inprompt::readBoolean";
				} else {	
					stream << "inprompt::readPrimitive";
					stream << " <" << type->getName() << "> "; 
				}
				stream <<"(\"" << linkName << "\")";
				stream << stmtSeparator;	
			}
		}
	}
	return externalEnvLinks;
}

bool TaskGenerator::isUnsupportedInputType(Type *type, const char *varName) {

	ListType *list = dynamic_cast<ListType*>(type);
	MapType *map = dynamic_cast<MapType*>(type);
        NamedType *object = dynamic_cast<NamedType*>(type);
        if (list != NULL || map != NULL || object != NULL) {
        	std::cout << "We still don't know how to input complex types from an external source ";
        	std::cout << "\nSo cannot initiate " << varName;
                std::cout << "\nModify the generated code to include your custom initializer\n";
		return true;
        } else {
		return false;
        }
}

void TaskGenerator::readPartitionParameters(std::ofstream &stream) {
	
	std::cout << "\tGenerating code for taking input partition parameters\n";

	List<Identifier*> *partitionArgs = taskDef->getPartitionArguments();
	
	stream << std::endl << indent << "// determining values of partition parameters\n";
	stream << indent << "std::cout << \"determining partition parameters\\n\"" << stmtSeparator;
	
	// Following variable is needed by the thread-state/LPU management library. We used this additional 
	// structure along with a partition variable to be able to use statically defined library code
	// for the most part.
	stream << indent << "int *partitionArgs = NULL" << stmtSeparator;
	int parameterCount = partitionArgs->NumElements();
	if (parameterCount > 0) {
		stream << indent << "partitionArgs = new int[" << parameterCount << "]" << stmtSeparator;
	}
	
	// Display prompt for partition parameters one by one and assign them in appropriate field of the
	// partition object of the task and in appropriate index within the partitionArgs array
	for (int i = 0; i < partitionArgs->NumElements(); i++) {
		const char *argName = partitionArgs->Nth(i)->getName();
		stream << indent << "partition." << argName << " = inprompt::readPrimitive <int> ";
		stream << "(\"" << argName << "\")" << stmtSeparator;
		stream << indent << "partitionArgs[" << i << "] = partition." << argName;
		stream << stmtSeparator; 
	} 	
}

void TaskGenerator::copyPartitionParameters(std::ofstream &stream) {
	List<Identifier*> *partitionArgs = taskDef->getPartitionArguments();
	stream << std::endl << indent << "// copying partitioning parameters into an array\n";
	stream << indent << "int *partitionArgs = NULL" << stmtSeparator;
	int parameterCount = partitionArgs->NumElements();
	if (parameterCount > 0) {
		stream << indent << "partitionArgs = new int[" << parameterCount << "]" << stmtSeparator;
	}
	for (int i = 0; i < partitionArgs->NumElements(); i++) {
		const char *argName = partitionArgs->Nth(i)->getName();
		stream << indent << "partitionArgs[" << i << "] = partition." << argName;
		stream << stmtSeparator; 
	} 	
}

void TaskGenerator::inovokeTaskInitializer(std::ofstream &stream, 
		List<const char*> *externalEnvLinks, 
		bool skipArgInitialization) {
	
	std::cout << "\tGenerating code for invoking the task initializer function\n";

	InitializeInstr *initSection = taskDef->getInitSection();
	List<const char*> *initArguments = NULL;
	List<Type*> *argumentTypes = NULL;
	if (initSection != NULL) {
		initArguments = initSection->getArguments();
		argumentTypes = initSection->getArgumentTypes();
	}
	
	bool initFunctionInvocable = true;

	// Argument initialization only needed when we automatically generate main function for the task; thus this 
	// checking is applied. 
	if (!skipArgInitialization) {
		stream << std::endl << indent << "// determining values of initialization parameters\n";
		stream << indent << "std::cout << \"determining initialization parameters\\n\"" << stmtSeparator;

		// If there are init arguments then we have to create local variables for them and generate prompts 
		// to read them from the console. We maintain a flag to indicate if there are init arguments that
		// we cannot currently read from external input (such as user defined objects and lists). If there
		// are such parameters then the programmer needs to update the generated main function to initialize
		// them and invoke the initialize function
		if (initArguments != NULL) {
			for (int i = 0; i < initArguments->NumElements(); i++) {
				const char *argName = initArguments->Nth(i);
				Type *argumentType = argumentTypes->Nth(i);
				stream << indent << argumentType->getCppDeclaration(argName) << stmtSeparator;
				if (isUnsupportedInputType(argumentType, argName)) {
					initFunctionInvocable = false;
					stream << indent << "//TODO initialize " << argName << " here\n";			
				} else {
					if (Type::boolType == argumentType) {
						stream << indent << argName << " = inprompt::readBoolean(\"";
						stream << argName << "\")" << stmtSeparator; 	
					} else {
						stream << indent << argName << " = inprompt::readPrimitive ";
						stream << "<" << argumentType->getName() << "> (\"";
						stream << argName << "\")" << stmtSeparator;
					}
				}
			}
		}
	}

	// Invoke the task initializer function if it is invocable or write a comment directing code modifications
	stream << std::endl << indent << "// invoking the initializer function\n";
	if (!initFunctionInvocable) {	
		stream << indent << "//TODO invoke the initializeTask function after making required changes\n";
		stream << indent << "//";			
	}
	stream << indent << "initializeTask(metadata, envLinks, &taskGlobals, &threadLocals, partition";
	if (initArguments != NULL) {
		for (int i = 0; i < initArguments->NumElements(); i++) {
			const char *argName = initArguments->Nth(i);
			stream << paramSeparator << argName;
		}
	}
	stream << ")" << stmtSeparator;

	// Update the length property of all dimensions of all arrays lest any has contents been modified within 
	// the initialize function 
	PartitionHierarchy *lpsHierarchy = taskDef->getPartitionHierarchy();
        Space *rootLps = lpsHierarchy->getRootSpace();
	List<const char*> *localArrays = rootLps->getLocallyUsedArrayNames();
        for (int i = 0; i < localArrays->NumElements(); i++) {
                ArrayDataStructure *array = (ArrayDataStructure*) rootLps->getLocalStructure(localArrays->Nth(i));
                int dimensions = array->getDimensionality();
                for (int d = 0; d < dimensions; d++) {
			stream << indent;
                	stream << "metadata->" << array->getName() << "Dims[" << d << "].setLength()";
                	stream << stmtSeparator;
		}
        }
	stream << indent << "logFile << \"\\ttask initialization is complete\\n\"" << stmtSeparator;
	stream << indent << "logFile.flush()" << stmtSeparator;
}

void TaskGenerator::initiateThreadStates(std::ofstream &stream) {
	
	std::cout << "\tGenerating state variables for threads\n";

	stream << std::endl << indent << "// declaring and initializing state variables for threads \n";	

	// declare an array of task-locals so that each thread can have its own copy for this for independent 
	// updates
	stream << indent << "ThreadLocals *threadLocalsList[Total_Threads]" << stmtSeparator;
	// generate a loop copying the initialized task-local variable into entries of the array
	stream << indent << "for (int i = 0; i < Total_Threads; i++) {\n";
	stream << indent << indent << "threadLocalsList[i] = new ThreadLocals" << stmtSeparator;
	stream << indent << indent << "*threadLocalsList[i] = threadLocals" << stmtSeparator;
	stream << indent << "}\n";

	// iterate over the mapping configuration and create an array of LPS dimensionality information
	stream << indent << "int lpsDimensions[Space_Count]" << stmtSeparator;
	std::deque<MappingNode*> nodeQueue;
        nodeQueue.push_back(mappingRoot);
        while (!nodeQueue.empty()) {
                MappingNode *node = nodeQueue.front();
                nodeQueue.pop_front();
                for (int i = 0; i < node->children->NumElements(); i++) {
                        nodeQueue.push_back(node->children->Nth(i));
                }
                Space *lps = node->mappingConfig->LPS;
                stream << indent << "lpsDimensions[Space_" << lps->getName() << "] = ";
		stream << lps->getDimensionCount() << stmtSeparator;
        }
	
	// create an array of thread IDs and initiate them
	stream << indent << "ThreadIds *threadIdsList[Total_Threads]" << stmtSeparator;
	stream << indent << "for (int i = 0; i < Total_Threads; i++) {\n";
	stream << indent << indent << "threadIdsList[i] = getPpuIdsForThread(i)" << stmtSeparator;
	stream << indent << indent << "adjustPpuCountsAndGroupSizes(threadIdsList[i])" << stmtSeparator;
	stream << indent << "}\n";

	// create a data partition configuration object
	stream << indent << "int *ppuCounts = threadIdsList[0]->getAllPpuCounts()" << stmtSeparator;
	stream << indent << "Hashtable<DataPartitionConfig*> *configMap = ";
	stream << '\n' << indent << indent << indent;
	stream << "getDataPartitionConfigMap(metadata, partition, ppuCounts)" << stmtSeparator;
	
	// finally create an array of Thread-State variables and initiate them	
	stream << indent << "ThreadStateImpl *threadStateList[Total_Threads]" << stmtSeparator;
	stream << indent << "for (int i = 0; i < Total_Threads; i++) {\n";
	stream << indent << indent << "threadStateList[i] = new ThreadStateImpl(Space_Count, ";
	stream << std::endl << indent << indent << indent << indent;
	stream << "lpsDimensions, partitionArgs, threadIdsList[i])" << stmtSeparator;
	stream << indent << indent << "threadStateList[i]->initializeLPUs()" << stmtSeparator;
	stream << indent << indent << "threadStateList[i]->setLpsParentIndexMap()" << stmtSeparator;
	stream << indent << indent << "threadStateList[i]->setPartConfigMap(configMap)" << stmtSeparator;	
	stream << indent << "}\n";
}

void TaskGenerator::performSegmentGrouping(std::ofstream &stream, bool segmentIdPassed) {
	
	std::cout << "\tGrouping threads into segments\n";

	stream << std::endl << indent << "// grouping threads into segments\n";	
	
	// declare a list of segments to hold the threads
	stream << indent << "List<SegmentState*> *segmentList = new List<SegmentState*>" << stmtSeparator;

	// add threads in proper segments
	stream << indent << "int segmentCount = Total_Threads / Threads_Per_Segment" << stmtSeparator;
	stream << indent << "int threadIndex = 0" << stmtSeparator;
	stream << indent << "for (int s = 0; s < segmentCount; s++) {\n";
	stream << doubleIndent << "SegmentState *segment = new SegmentState(s, s)" << stmtSeparator;
	stream << doubleIndent << "int participantCount = 0" << stmtSeparator;
	stream << doubleIndent << "while (participantCount < Threads_Per_Segment) {\n";
	stream << tripleIndent << "segment->addParticipant(threadStateList[threadIndex])" << stmtSeparator;
	stream << tripleIndent << "threadIndex++" << stmtSeparator;
	stream << tripleIndent << "participantCount++" << stmtSeparator;
	stream << doubleIndent << "}\n";
	stream << doubleIndent << "segmentList->Append(segment)" << stmtSeparator;
	stream << indent << "}\n";

	// determine the segment id of the current process and its participant range
	if (!segmentIdPassed) {
		stream << indent << "int segmentId = 0" << stmtSeparator;
		stream << indent << "if (argc > 1) segmentId = std::atoi(argv[1])" << stmtSeparator;
	}
	stream << indent << "SegmentState *mySegment = segmentList->Nth(segmentId)" << stmtSeparator;
	stream << indent << "int participantStart = segmentId * Threads_Per_Segment" << stmtSeparator;
	stream << indent << "int participantEnd = participantStart + Threads_Per_Segment - 1" << stmtSeparator;
	stream << indent << "logFile << \"\\tsegment grouping of threads is complete\\n\"" << stmtSeparator;
	stream << indent << "logFile.flush()" << stmtSeparator;
}

void TaskGenerator::initializeSegmentMemory(std::ofstream &stream) {
	
	std::cout << "\tGenerating code for initializing segment memory\n";

	stream << std::endl << indent << "// initializing segment memory\n";	
	
	// generate the task data object instance
	stream << indent << "TaskData *taskData = initializeTaskData(mySegment->getParticipantList()" << paramSeparator;
	stream << "\n" << indent << doubleIndent << "metadata" << paramSeparator;
	stream << "environment" << paramSeparator << "mySegment" << paramSeparator;
	stream << "partition" << paramSeparator << "ppuCounts)" << stmtSeparator;

	// set the task data property in each thread of the segment
	stream << indent << "for (int i = participantStart; i <= participantEnd; i++) {\n";
	stream << doubleIndent << "threadStateList[i]->setTaskData(taskData)" << stmtSeparator;

	// set up the part-iterator map whose elements will be used to search data-part Ids for LPUs
	stream << doubleIndent << "threadStateList[i]->setPartIteratorMap(";
	stream << "taskData->generatePartIteratorMap())" << stmtSeparator;
	
	// enable logging for the participant threads
	stream << doubleIndent << "threadStateList[i]->initiateLogFile(\"" << initials << "\")" << stmtSeparator;	
	stream << doubleIndent << "threadStateList[i]->enableLogging()" << stmtSeparator;

	stream << indent << "}\n";
	stream << indent << "delete[] ppuCounts" << stmtSeparator;
	stream << indent << "logFile << \"\\tmemory allocation is complete\\n\"" << stmtSeparator;
	stream << indent << "logFile.flush()" << stmtSeparator;
}

bool TaskGenerator::hasCommunicators() {
	List<const char*> *syncVars = taskDef->getComputation()->getVariablesNeedingCommunication(segmentedPPS);
	return syncVars != NULL && syncVars->NumElements() > 0;
}

bool TaskGenerator::generateCommunicators(std::ofstream &stream) {	

	// first check if communicator setup is needed for this task, if NO then return false
        List<const char*> *syncVars = taskDef->getComputation()->getVariablesNeedingCommunication(segmentedPPS);
	if (syncVars == NULL || syncVars->NumElements() == 0) {
		return false;
	}

	std::cout << "\tGenerating code for initializing communicators map\n";
	stream << std::endl << indent << "// initializing communicators map\n";	

	// create a communication statistics object to record time spent on different aspects of communication
	stream << indent << "CommStatistics *commStat = new CommStatistics()" << stmtSeparator;
	
	// first generate a distribution map for data shared among multiple segments
	stream << indent << "PartDistributionMap *distributionMap = generateDistributionMap(";
	stream << "segmentList" << paramSeparator << "configMap)" << stmtSeparator;

	// then use that map to create communicators for shared arrays; the same function creates communicator for scalars
	stream << indent << "Hashtable<Communicator*> *communicatorMap = generateCommunicators(";
	stream << "mySegment" << paramSeparator;
	stream << '\n' << indent << doubleIndent;
	stream << "segmentList" << paramSeparator << "taskData" << paramSeparator << "&taskGlobals";
	stream << paramSeparator << "configMap" << paramSeparator << "distributionMap"; 
	stream << paramSeparator << '\n' << indent << doubleIndent;
	stream << "commStat" << paramSeparator << "logFile)" << stmtSeparator;

	// finally assign the communicator map to the threads of the current segment
	stream << indent << "for (int i = participantStart; i <= participantEnd; i++) {\n";
	stream << doubleIndent << "threadStateList[i]->setCommunicatorMap(communicatorMap)" << stmtSeparator;
	stream << indent << "}\n"; 
	
	stream << indent << "logFile << \"\\tcommunicators have been created\\n\"" << stmtSeparator;
	stream << indent << "logFile.flush()" << stmtSeparator;

	return true;
}

void TaskGenerator::startThreads(std::ofstream &stream) {
	
	std::cout << "\tGenerating code for starting threads\n";

	stream << std::endl << indent << "// starting threads\n";
	stream << indent << "logFile << \"\\tlaunching threads\\n\"" << stmtSeparator;	
	stream << indent << "logFile.flush()" << stmtSeparator;
	
	// declare an array of thread IDs and another array of thread arguments
	stream << indent << "pthread_t threads[Total_Threads]" << stmtSeparator;
	stream << indent << "PThreadArg *threadArgs[Total_Threads]" << stmtSeparator;
	
	// initialize the argument list first
	stream << indent << "for (int i = 0; i < Total_Threads; i++) {\n";
	stream << indent << indent << "threadArgs[i] = new PThreadArg" << stmtSeparator;
	stream << indent << indent << "threadArgs[i]->taskName = \"" << taskDef->getName() << "\"" << stmtSeparator;
	stream << indent << indent << "threadArgs[i]->metadata = metadata" << stmtSeparator;
	stream << indent << indent << "threadArgs[i]->taskGlobals = &taskGlobals" << stmtSeparator;
	stream << indent << indent << "threadArgs[i]->threadLocals = threadLocalsList[i]" << stmtSeparator;
	stream << indent << indent << "threadArgs[i]->partition = partition" << stmtSeparator;
	stream << indent << indent << "threadArgs[i]->threadState = threadStateList[i]" << stmtSeparator;
	stream << indent << "}\n";
	
	// declare attributes that will be needed to set the thread affinity masks properly
	stream << indent << "pthread_attr_t attr" << stmtSeparator;
	stream << indent << "cpu_set_t cpus" << stmtSeparator;
	stream << indent << "pthread_attr_init(&attr)" << stmtSeparator;

	// then create the threads one by one
	stream << indent << "int state" << stmtSeparator;
	stream << indent << "for (int i = participantStart; i <= participantEnd; i++) {\n";
	// determine the cpu-id for the thread
	stream << indent << indent << "int cpuId = (i * Core_Jump / Threads_Per_Core) % Processors_Per_Phy_Unit";
	stream << stmtSeparator;
	stream << indent << indent << "int physicalId = Processor_Order[cpuId]" << stmtSeparator;

	// check if thread affinity is disabled in the deployment; by default threads are pinned to specific cores
        bool affinityEnabled = true;
        Properties *deploymentProps = PropertyReader::propertiesGroups->Lookup("deployment");
        if (deploymentProps != NULL) {
                const char *affinityKey = "thread.affinity.enabled";
                const char *affinitySetting = deploymentProps->getProperty(affinityKey);
                if (affinitySetting != NULL && strcmp(affinitySetting, "true") != 0) {
                        affinityEnabled = false;
                        std::cout << "\tGenerating multithreaded code without thread affinity\n";
                }
        }

        // then set the affinity attribute based on the CPU Id
        if (affinityEnabled) {
                stream << indent << indent << "CPU_ZERO(&cpus)" << stmtSeparator;
                stream << indent << indent << "CPU_SET(physicalId, &cpus)" << stmtSeparator;
                stream << indent << indent << "pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus)";
                stream << stmtSeparator;
        }

	// then create the thread
	stream << indent << indent << "state = pthread_create(&threads[i], &attr, runPThreads, (void *) threadArgs[i])";
	stream << stmtSeparator;
	// if thread creation fails then exit after writing an error message on the screen
	stream << indent << indent << "if (state) {\n";
	stream << indent << indent << indent << "std::cout << \"Could not start some PThread\" << std::endl";
	stream << stmtSeparator;
	stream << indent << indent << indent << "std::exit(EXIT_FAILURE)" << stmtSeparator;
	stream << indent << indent << "} else {\n";
	stream << tripleIndent << "logFile << \"\\t\\tlaunched thread #\" << i << \"\\n\"" << stmtSeparator;
	stream << tripleIndent << "logFile.flush()" << stmtSeparator;
	stream << doubleIndent << "}\n";
	stream << indent << "}\n";

	// finally make the main thread wait till all other threads finish execution	
	stream << indent << "for (int i = participantStart; i <= participantEnd; i++) {\n";
	stream << indent << indent << "pthread_join(threads[i], NULL)" << stmtSeparator;
	stream << indent << "}\n\n";
}

void TaskGenerator::writeResults(std::ofstream &stream) {

        PartitionHierarchy *lpsHierarchy = taskDef->getPartitionHierarchy();
	Space *rootLps = lpsHierarchy->getRootSpace();

	List<const char*> *externalEnvLinks = new List<const char*>;
	
	stream << indent << "// writing environment variables to files after task completion\n";
	stream << indent << "std::cout << \"writing results to output files\\n\"" << stmtSeparator;

	List<EnvironmentLink*> *envLinks = taskDef->getEnvironmentLinks();
	List<const char*> *scalarVarList = new List<const char*>;

	std::ostringstream structureRef;
	structureRef << "space" << rootLps->getName() << "Content";

	for (int i = 0; i < envLinks->NumElements(); i++) {
                
		EnvironmentLink *link = envLinks->Nth(i);
                const char *linkName = link->getVariable()->getName();

                DataStructure *structure = rootLps->getLocalStructure(linkName);
                ArrayDataStructure *array = dynamic_cast<ArrayDataStructure*>(structure);

		// we will output all scalar variables on the console as opposed to writing in a file 
		// as done in the case of arrays 
		if (array == NULL) {
			scalarVarList->Append(linkName);
			continue;
		}

		ArrayType *arrayType = (ArrayType*) array->getType();
                Type *elemType = arrayType->getTerminalElementType();
		if (isUnsupportedInputType(elemType, linkName)) {
			stream << indent << "//TODO put custom output code for " << linkName << "\n";
		} else {
			// first generate a prompt that will ask the user if he wants to write this
			// array to a file
			stream << indent << "if (outprompt::getYesNoAnswer(\"Want to save array";
			stream << " \\\"" << linkName << "\\\" in a file?\"";
			stream << ")) {\n";

			// then generate the prompt for writing the array to the file specified by
			// the user
			int dimensionCount = array->getDimensionality();
			stream << doubleIndent << "outprompt::writeArrayToFile ";
			stream << '<' << elemType->getName() << '>';
			stream << " (\"" << linkName << "\"" << paramSeparator;
			stream << std::endl << doubleIndent << doubleIndent;
			stream << structureRef.str() << '.' << linkName << paramSeparator;
			stream << std::endl << doubleIndent << doubleIndent;
			stream << dimensionCount << paramSeparator;
			stream << "metadata->" << linkName << "Dims)";
			stream << stmtSeparator;
			
			// close the if block at the end	
			stream << indent << "}\n";	
		}
	}

	if (scalarVarList->NumElements() > 0) {
		for (int i = 0; i < scalarVarList->NumElements(); i++) {
			const char *var = scalarVarList->Nth(i);
			stream << indent << "std::cout << \"" <<  var << ": \"";
			stream << " << taskGlobals." << var << " << '\\n'";
			stream << stmtSeparator;
		}
	}	
}
