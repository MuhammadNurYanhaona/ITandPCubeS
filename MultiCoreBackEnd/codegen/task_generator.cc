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

	mappingRoot = NULL;
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
        
	// generate task specific data structures and their functions 
        generateLpuDataStructures(headerFile, mappingConfig);
	generatePrintFnForLpuDataStructures(initials, programFile, mappingConfig);
        List<const char*> *envLinkList = generateArrayMetadataAndEnvLinks(headerFile, 
			mappingConfig, taskDef->getEnvironmentLinks());
	generateFnForMetadataAndEnvLinks(taskDef->getName(), initials, 
			programFile, mappingConfig, envLinkList);
	List<TaskGlobalScalar*> *globalScalars 
			= TaskGlobalCalculator::calculateTaskGlobals(taskDef);
	generateClassesForGlobalScalars(headerFile, globalScalars);

	// generate functions to initialize LPS content references
	generateFnToInitiateRootLPSContent(headerFile, programFile, initials,
                mappingConfig, envLinkList);
	generateFnToInitiateLPSesContent(headerFile, programFile, initials, mappingConfig);
        
	// generate thread management functions and classes
        generateFnForThreadIdsAllocation(headerFile, 
			programFile, initials, mappingConfig, pcubesConfig);
        generateThreadStateImpl(headerFile, programFile, mappingConfig,
                        partitionFnParamConfigs, lpuPartFnParamsConfigs);

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
			programFile, initials, mappingConfig);

	closeNameSpace(headerFile);
}

void TaskGenerator::generateTaskMain() {
	
	std::cout << "\n-----------------------------------------------------------------\n";
	std::cout << "Generating a main fcuntion for task: " << taskDef->getName();
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

	// start threads 
	startThreads(stream);	

	// close the log file and end the function definition
	stream << std::endl << indent << "logFile.close()" << stmtSeparator;
	stream << indent << "return 0" << stmtSeparator;
	stream << "}\n";
	stream.close();
}

List<const char*> *TaskGenerator::initiateEnvLinks(std::ofstream &stream) {
	
        PartitionHierarchy *lpsHierarchy = taskDef->getPartitionHierarchy();
	Space *rootLps = lpsHierarchy->getRootSpace();

	List<const char*> *externalEnvLinks = new List<const char*>;
	std::string indent = "\t";
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
			if (isUnsupportedInputType(elemType, linkName)) {
				stream << indent << "//TODO put custom initializing code for " << linkName << "\n";
			} else {
				// create a prompt to get the dimensions information for the variable under concern
				stream << indent;
				stream << "inprompt::readArrayDimensionInfo(\"" << linkName << "\"" << paramSeparator;
				int dimensionCount = array->getDimensionality();
				stream << dimensionCount << paramSeparator;
				stream << "envLinks." << linkName << "Dims)" << stmtSeparator;
				// then allocate an array for the variable
				stream << indent;
				stream << "envLinks." << linkName << " = allocate::allocateArray ";
				stream << '<' << elemType->getName() << "> ";	
				stream << '(' << dimensionCount << paramSeparator;
				stream << "envLinks." << linkName << "Dims)" << stmtSeparator;
				// finally randomly initialize the array
				stream << indent << "allocate::randomFillPrimitiveArray ";
				stream << '<' << elemType->getName() << "> ";
				stream << "(envLinks." << linkName << paramSeparator;	
				stream << dimensionCount << paramSeparator;
				stream << "envLinks." << linkName << "Dims)" << stmtSeparator;
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

void TaskGenerator::inovokeTaskInitializer(std::ofstream &stream, List<const char*> *externalEnvLinks) {
	
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
	
	stream << std::endl << indent << "// determining values of initialization parameters\n";
	stream << indent << "std::cout << \"determining initialization parameters\\n\"" << stmtSeparator;

	// If there are init arguments then we have to create local variables for them and generate prompts 
	// to read them from the console. We maintain a flag to indicate if there are init arguments that
	// we cannot currently read from external input (such as user defined objects and lists). If there
	// are such parameters then the programmer needs to update the generated main function to initialize
	// them and invoke the initialize function
	bool initFunctionInvocable = true;
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
					stream << indent << argName << " = readBoolean(\"";
					stream << argName << "\")" << stmtSeparator; 	
				} else {
					stream << indent << argName << " = readPrimitive ";
					stream << "<" << argumentType->getName() << "> (\"";
					stream << argName << "\")" << stmtSeparator;
				}
			}
		}
	}

	stream << std::endl << indent << "// invoking the initializer function\n";
	stream << indent << "std::cout << \"invoking task initializer function\\n\"" << stmtSeparator;
	// Invoke the initializer function if it is invocable or write a comment directing code modifications
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
	stream << indent << "std::cout << \"generating PPU Ids for threads\\n\"" << stmtSeparator;
	stream << indent << "ThreadIds *threadIdsList[Total_Threads]" << stmtSeparator;
	stream << indent << "for (int i = 0; i < Total_Threads; i++) {\n";
	stream << indent << indent << "threadIdsList[i] = getPpuIdsForThread(i)" << stmtSeparator;
	stream << indent << indent << "threadIdsList[i]->print(logFile)" << stmtSeparator;
	stream << indent << "}\n";

	// finally create an array of Thread-State variables and initiate them	
	stream << indent << "std::cout << \"initiating thread-states\\n\"" << stmtSeparator;
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
	stream << indent << "std::cout << \"starting threads\\n\"" << stmtSeparator;
	stream << indent << "for (int i = 0; i < Total_Threads; i++) {\n";
	stream << indent << indent << "run(metadata" << paramSeparator;
	stream << "&taskGlobals" << paramSeparator;
	stream << "threadLocalsList[i]" << paramSeparator;
	stream << "partition" << paramSeparator;
	stream << "threadStateList[i])" << stmtSeparator;

	stream << indent << "}\n";
}
