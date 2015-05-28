#include "task_generator.h"

#include <iostream>
#include <sstream>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <deque>

#include "code_generator.h"
#include "thread_state_mgmt.h"
#include "space_mapping.h"
#include "name_transformer.h"
#include "compute_translator.h"
#include "pthread_mgmt.h"
#include "sync_mgmt.h"
#include "task_invocation.h"
#include "memory_mgmt.h"

#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "../utils/string_utils.h"
#include "../syntax/ast.h"
#include "../syntax/ast_def.h"
#include "../syntax/errors.h"
#include "../syntax/ast_task.h"
#include "../syntax/ast_type.h"
#include "../static-analysis/task_global.h"
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

	initials = string_utils::getInitials(taskDef->getName());
	initials = string_utils::toLower(initials);

	mappingRoot = NULL;
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

void TaskGenerator::generate(List<PPS_Definition*> *pcubesConfig) {

	std::cout << "\n-----------------------------------------------------------------\n";
	std::cout << "Translating task: " << taskDef->getName();
	std::cout << "\n-----------------------------------------------------------------\n";

	initializeOutputFiles(headerFile, programFile, initials);

	// interpret mapping configuration
        PartitionHierarchy *lpsHierarchy = taskDef->getPartitionHierarchy();
        MappingNode *mappingConfig = parseMappingConfiguration(taskDef->getName(),
                        mappingFile, lpsHierarchy, pcubesConfig);
	this->mappingRoot = mappingConfig;

	// determine where memory segmentation happens in the hardware
	int segmentedPPS = 0;
	for (int i = 0; i < pcubesConfig->NumElements(); i++) {
		PPS_Definition *pps = pcubesConfig->Nth(i);
		if (pps->segmented) {
			segmentedPPS = pps->id;	
			break;
		} 
	}
	if (segmentedPPS == 0) {
		std::cout << "No segmentation observed in the PCubeS configuration\n";
		// assign the topmost PPS the segmentation marker to make the code work for
		// non segmented memory architectures
		segmentedPPS = pcubesConfig->Nth(0)->id;
	}

	// do back-end architecture dependent static analyses of the task
	lpsHierarchy->performAllocationAnalysis(segmentedPPS);

	// generate a constant array for processor ordering in the hardware
	generateProcessorOrderArray(headerFile, processorFile);
        
	// generate constansts needed for various reasons
        generateLPSConstants(headerFile, mappingConfig);
        generatePPSCountConstants(headerFile, pcubesConfig);
        generateThreadCountConstants(headerFile, mappingConfig, pcubesConfig);
        
	// generate data structures and functions for task environment management 
	generatePrintFnForLpuDataStructures(initials, programFile, mappingConfig);
        List<const char*> *envLinkList = generateArrayMetadataAndEnvLinks(headerFile, 
			mappingConfig, taskDef->getEnvironmentLinks());
	generateFnForMetadataAndEnvLinks(taskDef->getName(), initials, 
			programFile, mappingConfig, envLinkList);

	// generate functions related to memory management
	const char *upperInitials = string_utils::getInitials(taskDef->getName());
	genRoutinesForTaskPartitionConfigs(headerFile, programFile, upperInitials, lpsHierarchy);

	List<TaskGlobalScalar*> *globalScalars 
			= TaskGlobalCalculator::calculateTaskGlobals(taskDef);
	generateClassesForGlobalScalars(headerFile, globalScalars);
	
	// generate functions to initialize LPS content references
	generateFnToInitiateRootLPSContent(headerFile, programFile, initials,
                	taskDef, mappingConfig, envLinkList);
	// TODO note that we commented out the original and accurate implementation for 
	// the sake of a simpler one as we are hard pressed with time. The latter version
	// uses memory allocations for structures within root LPS for all other LPSes.
	// There is no usage based allocations or reference redirections as done in the 
	// former implementation. Using a single LPS for all allocations make it easy to
	// generate code for LPS transition at the expense of performance. In the future
	// we must go back to the original implementation if we want to optimize the 
	// compiler. 
	// generateFnToInitiateLPSesContent(headerFile, 
	//		programFile, initials, mappingConfig);
	generateFnToInitiateLPSesContentSimple(headerFile, 
			programFile, initials, mappingConfig);
        
        
	// generate library routines for LPUs management        
        List<Identifier*> *partitionArgs = taskDef->getPartitionArguments();
        Hashtable<List<PartitionParameterConfig*>*> *partitionFnParamConfigs
                        = generateLPUCountFunctions(headerFile, 
					programFile, initials, mappingConfig, partitionArgs);
        Hashtable<List<int>*> *lpuPartFnParamsConfigs
                        = generateAllGetPartForLPURoutines(headerFile, programFile, 
					initials, mappingConfig, partitionArgs);
        generateLpuDataStructures(headerFile, mappingConfig);

	// generate thread management functions and classes
        generateFnForThreadIdsAllocation(headerFile, 
			programFile, initials, mappingConfig, pcubesConfig);
        generateThreadStateImpl(headerFile, programFile, mappingConfig,
                        partitionFnParamConfigs, lpuPartFnParamsConfigs);

	// generate synchronization primitives and their initialization functions
	syncManager = new SyncManager(taskDef, headerFile, programFile, initials);
	syncManager->processSyncList();
	syncManager->generateSyncPrimitives();
	syncManager->generateSyncInitializerFn();
	syncManager->generateSyncStructureForThreads();
	syncManager->generateFnForThreadsSyncStructureInit();

	// generate routines needed for supporting task invocation from the coordinator
	std::cout << "Generating task invocation related routines\n";
	generateFnToInitEnvLinksFromEnvironment(taskDef, 
			initials, envLinkList, headerFile, programFile);
	generateFnToInitTaskRootFromEnv(taskDef, initials, headerFile, programFile);
	generateTaskExecutor(this);

	// initialize the variable transformation map that would be used to translate
	// the code inside initialize and compute blocks
	ntransform::NameTransformer::setTransformer(taskDef);	

	// translate the initialize block of the task into a function
	generateInitializeFunction(headerFile, programFile, initials, 
        		envLinkList, taskDef, mappingConfig->mappingConfig->LPS);

	// generate functions for all compute stages in the source code
	generateFnsForComputation(taskDef, headerFile, programFile, initials);

	// generate run function for threads
	generateThreadRunFunction(taskDef, headerFile, 
			programFile, initials, mappingConfig, 
			syncManager->involvesSynchronization());

	// generate data structure and functions for Pthreads
	generateArgStructForPthreadRunFn(taskDef->getName(), headerFile);
	generatePThreadRunFn(headerFile, programFile, initials);

	closeNameSpace(headerFile);
}

void TaskGenerator::generateTaskMain() {
	
	std::cout << "\n-----------------------------------------------------------------\n";
	std::cout << "Generating a main function for task: " << taskDef->getName();
	std::cout << "\n-----------------------------------------------------------------\n";
	
	std::ofstream stream;
	stream.open(programFile, std::ofstream::out | std::ofstream::app);
	stream << "/*-----------------------------------------------------------------------------------\n";
        stream << "main function\n";
        stream << "------------------------------------------------------------------------------------*/\n";
	
	std::string indent = "\t";
	std::string stmtSeparator = ";\n";
	std::string paramSeparator = ", ";

	// write the function signature
	stream << "\nint main() {\n\n";
	stream << indent << "std::cout << \"Starting " << taskDef->getName() << " Task\\n\"" << stmtSeparator;
	stream << std::endl;
		
	// declares a list of default task related variables
	stream << indent << "// declaring common task related variables\n";
	stream << indent << "TaskGlobals taskGlobals" << stmtSeparator; 	
	stream << indent << "ThreadLocals threadLocals" << stmtSeparator;
	stream << indent << "EnvironmentLinks envLinks" << stmtSeparator;
	stream << indent << "ArrayMetadata *metadata = new ArrayMetadata" << stmtSeparator;
	const char *upperInitials = string_utils::getInitials(taskDef->getName());
	stream << indent << upperInitials << "Environment environment" << stmtSeparator;
	stream << indent << upperInitials << "Partition partition" << stmtSeparator;
	stream << std::endl;

	// create a log file for overall program log printing
	stream << indent << "// creating a program log file\n";
	stream << indent << "std::cout << \"Creating diagnostic log: it-program.log\\n\"" << stmtSeparator;
	stream << indent << "std::ofstream logFile" << stmtSeparator;
	stream << indent << "logFile.open(\"it-program.log\")" << stmtSeparator << std::endl;

	// generate prompts to read metadata of arrays and values of other structures in environment links
	List<const char*> *externalEnvLinks = initiateEnvLinks(stream);

	// read in partition parameters
	readPartitionParameters(stream);

	// if the task involves synchronization then initialize the global sync variables that would be used
	// by different threads
	if (syncManager->involvesSynchronization()) {
		stream << std::endl << indent << "// initializing sync primitives\n";
		stream << indent << "std::cout << \"Initializing sync primitives\\n\"" << stmtSeparator;
		stream << indent << "initializeSyncPrimitives()" << stmtSeparator;
	}

	// read any initialization parameter that are not already covered as environment links and invoke the
	// initialize function
	inovokeTaskInitializer(stream, externalEnvLinks);

	// assign the metadata created in the static arraymetadata object so that thread initialization can
	// be done
	stream << std::endl << indent << "// setting the global metadata variable\n";
	stream << indent << "arrayMetadata = *metadata" << stmtSeparator;
	stream << indent << "metadata->print(logFile)" << stmtSeparator;	

	// invoke functions to initialize array references in different LPSes
	stream << std::endl << indent << "// allocating memories for data structures\n";
	stream << indent << "std::cout << \"Allocating memories\\n\"" << stmtSeparator;
	stream << indent << initials << "::initializeRootLPSContent(&envLinks, metadata)" << stmtSeparator;
	stream << indent << initials << "::initializeLPSesContents(metadata)" << stmtSeparator;

	// generate thread-state objects for the intended number of threads and initialize their root LPUs
	initiateThreadStates(stream);

	// start execution time monitoring timer
	stream << std::endl << indent << "// starting execution timer clock\n";
	stream << indent << "struct timeval start" << stmtSeparator;
	stream << indent << "gettimeofday(&start, NULL)" << stmtSeparator;

	// start threads 
	startThreads(stream);

	// calculate running time
	stream << indent << "// calculating task running time\n";
	stream << indent << "struct timeval end" << stmtSeparator;
	stream << indent << "gettimeofday(&end, NULL)" << stmtSeparator;
	stream << indent << "double runningTime = ((end.tv_sec + end.tv_usec / 1000000.0)";
	stream << std::endl << indent << indent << indent;
	stream << "- (start.tv_sec + start.tv_usec / 1000000.0))" << stmtSeparator;
	stream << indent << "logFile << \"Execution Time: \" << runningTime << \" Seconds\" << std::endl";
	stream << stmtSeparator << std::endl;

	// write all environment variables into files
	writeResults(stream);
	
	// close the log file
	stream << std::endl << indent << "logFile.close()" << stmtSeparator;
	// display the running time on console
	stream << indent << "std::cout << \"Parallel Execution Time: \" << runningTime <<";
	stream << " \" Seconds\" << std::endl" << stmtSeparator;
	// then exit the function
	stream << indent << "return 0" << stmtSeparator;
	stream << "}\n";
	stream.close();
	
}

List<const char*> *TaskGenerator::initiateEnvLinks(std::ofstream &stream) {
	
        PartitionHierarchy *lpsHierarchy = taskDef->getPartitionHierarchy();
	Space *rootLps = lpsHierarchy->getRootSpace();

	List<const char*> *externalEnvLinks = new List<const char*>;
	std::string indent = "\t";
	std::string doubleIndent = "\t\t";
	std::string stmtSeparator = ";\n";
	std::string paramSeparator = ", ";
	
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
	
	std::cout << "Generating code for taking input partition parameters\n";

	List<Identifier*> *partitionArgs = taskDef->getPartitionArguments();
	std::string indent = "\t";
	std::string stmtSeparator = ";\n";
	std::string paramSeparator = ", ";
	
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
	std::string indent = "\t";
	std::string stmtSeparator = ";\n";
	std::string paramSeparator = ", ";
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
	
	std::cout << "Generating code for invoking the task initializer function\n";

	InitializeInstr *initSection = taskDef->getInitSection();
	List<const char*> *initArguments = NULL;
	List<Type*> *argumentTypes = NULL;
	if (initSection != NULL) {
		initArguments = initSection->getArguments();
		argumentTypes = initSection->getArgumentTypes();
	}
	
	std::string indent = "\t";
	std::string stmtSeparator = ";\n";
	std::string paramSeparator = ", ";	
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
	stream << indent << "//std::cout << \"invoking task initializer function\\n\"" << stmtSeparator;
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
}

void TaskGenerator::initiateThreadStates(std::ofstream &stream) {
	
	std::cout << "Generating state variables for threads\n";

	std::string indent = "\t";
	std::string stmtSeparator = ";\n";
	std::string paramSeparator = ", ";
	
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
	stream << indent << "//std::cout << \"generating PPU Ids for threads\\n\"" << stmtSeparator;
	stream << indent << "ThreadIds *threadIdsList[Total_Threads]" << stmtSeparator;
	stream << indent << "for (int i = 0; i < Total_Threads; i++) {\n";
	stream << indent << indent << "threadIdsList[i] = getPpuIdsForThread(i)" << stmtSeparator;
	stream << indent << indent << "threadIdsList[i]->print(logFile)" << stmtSeparator;
	stream << indent << "}\n";

	// finally create an array of Thread-State variables and initiate them	
	stream << indent << "//std::cout << \"initiating thread-states\\n\"" << stmtSeparator;
	stream << indent << "ThreadStateImpl *threadStateList[Total_Threads]" << stmtSeparator;
	stream << indent << "for (int i = 0; i < Total_Threads; i++) {\n";
	stream << indent << indent << "threadStateList[i] = new ThreadStateImpl(Space_Count, ";
	stream << std::endl << indent << indent << indent << indent;
	stream << "lpsDimensions, partitionArgs, threadIdsList[i])" << stmtSeparator;
	stream << indent << indent;
	stream << "threadStateList[i]->initiateLogFile(\"" << initials << "\")" << stmtSeparator;	
	stream << indent << indent << "threadStateList[i]->initializeLPUs()" << stmtSeparator;
	stream << indent << indent << "threadStateList[i]->setLpsParentIndexMap()" << stmtSeparator;	
	stream << indent << "}\n";
}

void TaskGenerator::startThreads(std::ofstream &stream) {
	
	std::cout << "Generating code for starting threads\n";

	std::string indent = "\t";
	std::string stmtSeparator = ";\n";
	std::string paramSeparator = ", ";
	
	stream << std::endl << indent << "// starting threads\n";	
	stream << indent << "//std::cout << \"starting threads\\n\"" << stmtSeparator;
	
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
	stream << indent << "for (int i = 0; i < Total_Threads; i++) {\n";
	// determine the cpu-id for the thread
	stream << indent << indent << "int cpuId = i * Core_Jump / Threads_Par_Core" << stmtSeparator;
	stream << indent << indent << "int physicalId = Processor_Order[cpuId]" << stmtSeparator;
	// then set the affinity attribute based on the CPU Id
	stream << indent << indent << "CPU_ZERO(&cpus)" << stmtSeparator;
	stream << indent << indent << "CPU_SET(physicalId, &cpus)" << stmtSeparator;
	stream << indent << indent << "pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus)" << stmtSeparator;
	// then create the thread
	stream << indent << indent << "state = pthread_create(&threads[i], &attr, runPThreads, (void *) threadArgs[i])";
	stream << stmtSeparator;
	// if thread creation fails then exit after writing an error message on the screen
	stream << indent << indent << "if (state) {\n";
	stream << indent << indent << indent << "std::cout << \"Could not start some PThread\" << std::endl";
	stream << stmtSeparator;
	stream << indent << indent << indent << "std::exit(EXIT_FAILURE)" << stmtSeparator;
	stream << indent << indent << "}\n";
	stream << indent << "}\n";

	// finally make the main thread wait till all other threads finish execution	
	stream << indent << "for (int i = 0; i < Total_Threads; i++) {\n";
	stream << indent << indent << "pthread_join(threads[i], NULL)" << stmtSeparator;
	stream << indent << "}\n\n";
}

void TaskGenerator::writeResults(std::ofstream &stream) {

        PartitionHierarchy *lpsHierarchy = taskDef->getPartitionHierarchy();
	Space *rootLps = lpsHierarchy->getRootSpace();

	List<const char*> *externalEnvLinks = new List<const char*>;
	std::string indent = "\t";
	std::string doubleIndent = "\t\t";
	std::string stmtSeparator = ";\n";
	std::string paramSeparator = ", ";
	
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
