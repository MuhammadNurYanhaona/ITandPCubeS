#include "task_invocation.h"
#include "task_generator.h"
#include "code_generator.h"
#include "sync_mgmt.h"
#include "../syntax/ast_def.h"
#include "../syntax/ast_type.h"
#include "../semantics/scope.h"
#include "../semantics/task_space.h"
#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "../utils/string_utils.h"
#include "../utils/common_utils.h"
#include "../utils/decorator_utils.h"
#include "../utils/code_constant.h"
#include "name_transformer.h"

#include <sstream>
#include <fstream>
#include <iostream>
#include <cstdlib>

void initiateProgramHeaders(const char *headerFileName, const char *programFileName, ProgramDef *programDef) {

	std::cout << "Initializing header and program file\n";
       	std::string line;
        std::ifstream commIncludeFile("codegen/default-includes.txt");
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

        headerFile << "#ifndef _H_coordinator\n";
        headerFile << "#define _H_coordinator\n";

	// include the header file for coordinator program in its program file
        int programHeaderIndex = string_utils::getLastIndexOf(headerFileName, '/') + 1;
        char *programHeader = string_utils::substr(headerFileName, 
			programHeaderIndex, strlen(headerFileName));
	const char *message1 = "header file for the coordinator program";
	decorator::writeSectionHeader(programFile, message1);
        programFile << std::endl << "#include \"" << programHeader  << '"' << std::endl;

	// prepare an stream to write all header that should be included in both files
	std::ostringstream commonHeaders;

	// retrieve the list of default libraries from an external file
	const char *message2 = "header files included for different purposes";
	decorator::writeSectionHeader(commonHeaders, message2);
	commonHeaders << std::endl;
	if (commIncludeFile.is_open()) {
                while (std::getline(commIncludeFile, line)) {
                        commonHeaders << line << std::endl;
                }
        } else {
                std::cout << "Unable to open common include file";
                std::exit(EXIT_FAILURE);
        }
	headerFile << commonHeaders.str();

	// include headers for all tasks in the library list;
	const char *message3 = "header files for tasks";
	decorator::writeSectionHeader(commonHeaders, message3);
	commonHeaders << std::endl;
	List<TaskDef*> *taskList = programDef->getTasks();
	for (int i = 0; i < taskList->NumElements(); i++) {
		commonHeaders << "#include \"";
		commonHeaders << TaskGenerator::getHeaderFileName(taskList->Nth(i)) << "\"\n";
	}
	programFile << commonHeaders.str();

	commIncludeFile.close();
        programFile.close();
        headerFile.close();	
}

void generateRoutineToInitProgramArgs(TupleDef *programArg, const char *headerFileName, const char *programFileName) {
        
	std::cout << "Generating routine to get input program arguments\n";
	std::ofstream programFile, headerFile;
        headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
        if (!programFile.is_open() || !headerFile.is_open()) {
                std::cout << "Unable to open output header/program file";
                std::exit(EXIT_FAILURE);
        }

	std::ostringstream fnHeader;
        const char *comments = "function for initializing program arguments";
	decorator::writeSectionHeader(headerFile, comments);
	headerFile << std::endl;
	decorator::writeSectionHeader(programFile, comments);
	programFile << std::endl;

	// write function signature in header and program files
	const char *tupleName = programArg->getId()->getName();
	fnHeader << tupleName << " getProgramArgs(char *fileName" << paramSeparator << "std::ofstream &logFile)";
	headerFile << fnHeader.str() << stmtSeparator;
	programFile << fnHeader.str();

	// start function definition
	programFile << " {\n";

	// try to open the file
	programFile << indent << "std::ifstream argFile(fileName)" << stmtSeparator;
	programFile << indent << "if (!argFile.is_open()) {\n";
	programFile << doubleIndent << "logFile << \"could not open input arguments file\\n\"" << stmtSeparator;
	programFile << doubleIndent << "logFile.flush()" << stmtSeparator;
	programFile << doubleIndent << "std::exit(EXIT_FAILURE)" << stmtSeparator;
	programFile << indent << "}\n";
	
	// create a new instance of the program argument object
	programFile << indent << tupleName << " programArgs = " << tupleName << "()" << stmtSeparator;

	// iterate over the list of properties and generate prompt for initiating each of them
	List<VariableDef*> *propertyList = programArg->getComponents();
	for (int i = 0; i < propertyList->NumElements(); i++) {
		VariableDef *property = propertyList->Nth(i);
		const char *propertyName = property->getId()->getName();
		Type *propertyType = property->getType();
		if (propertyType == Type::stringType) {
			programFile << indent << "std::string " << propertyName << "Str" << stmtSeparator;
			programFile << indent << "outprompt::readNonEmptyLine(";
			programFile << propertyName << "Str" << paramSeparator;
			programFile << "argFile" << ")" << stmtSeparator;	
			programFile << indent << "programArgs." << propertyName << " = strdup(";
			programFile << propertyName << "Str.c_str())" << stmtSeparator;	
		} else {
			programFile << indent <<  "argFile >> programArgs." << propertyName << stmtSeparator;
		}
	}
	
	// return the initialized argument object
	programFile << indent << "return programArgs" << stmtSeparator;

	// close function definition;
	programFile << "}\n";

	headerFile.close();
	programFile.close();
}

void generateRoutineToReadProgramArgs(TupleDef *programArg,  const char *headerFileName,  const char *programFileName) {
	
	std::cout << "Generating routine to read program arguments\n";
	std::ofstream programFile, headerFile;
        headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
        if (!programFile.is_open() || !headerFile.is_open()) {
                std::cout << "Unable to open output header/program file";
                std::exit(EXIT_FAILURE);
        }

	std::ostringstream fnHeader;
        const char *comments = "function for reading program arguments";
	decorator::writeSectionHeader(headerFile, comments);
	headerFile << std::endl;
	decorator::writeSectionHeader(programFile, comments);
	programFile << std::endl;

	// write function signature in header and program files
	const char *tupleName = programArg->getId()->getName();
	fnHeader << tupleName << " readProgramArgs(int argc" << paramSeparator << "char *argv[])";
	headerFile << fnHeader.str() << stmtSeparator;
	programFile << fnHeader.str();

	// start function definition
	programFile << " {\n\n";

	// create a new instance of the program argument object
	programFile << indent << tupleName << " programArgs = " << tupleName << "()" << stmtSeparator;
	programFile << "\n";

	// go through the command line arguments one by one and separate each argument into a key and a value part
	programFile << indent << "for (int i = 1; i < argc; i++) {\n";
	programFile << doubleIndent << "std::string keyValue = std::string(argv[i])" << stmtSeparator;
	programFile << doubleIndent << "size_t separator = keyValue.find('=')" << stmtSeparator;
	programFile << doubleIndent << "if (separator == std::string::npos) {\n";
	programFile << tripleIndent << "std::cout << \"a command line parameter must be in the form of key=value\\n\"";
	programFile << stmtSeparator;
	programFile << tripleIndent << "std::exit(EXIT_FAILURE)" << stmtSeparator;
	programFile << doubleIndent << "}\n";
	programFile << doubleIndent << "std::string key = keyValue.substr(0, separator)" << stmtSeparator;
	programFile << doubleIndent << "std::string value = keyValue.substr(separator + 1)" << stmtSeparator;
	programFile << "\n";

	// identify the argument type by comparing the key with property names of the program argument tuple and
	// assign the value to the matching property
	List<VariableDef*> *propertyList = programArg->getComponents();
	for (int i = 0; i < propertyList->NumElements(); i++) {
		VariableDef *property = propertyList->Nth(i);
		const char *propertyName = property->getId()->getName();
		if (i > 0) {
			programFile << " else ";
		} else {
			programFile << doubleIndent;
		}
		programFile << "if (strcmp(";
		programFile << "\"" << propertyName << "\"" << paramSeparator;
		programFile << "key.c_str()) == 0) {\n";
		
		// cast the value into appropriate type and assign it to the matching property of the program argument
		Type *propertyType = property->getType();
		if (propertyType == Type::stringType) {
			programFile << tripleIndent << "programArgs." << propertyName << " = strdup(";
			programFile << "value.c_str())" << stmtSeparator;	
		} else {
			programFile << tripleIndent << "std::istringstream stream(" << "value)" << stmtSeparator;
			programFile << tripleIndent << "stream >> programArgs." << propertyName << stmtSeparator;
		}

		programFile << doubleIndent << "}";
	}
	programFile << " else {\n";
        programFile << tripleIndent << "std::cout << \"unrecognized command line parameter: \" << key";
        programFile << " << '\\n'" << stmtSeparator;
        programFile << tripleIndent << "std::exit(EXIT_FAILURE)" << stmtSeparator;
        programFile << doubleIndent << "}\n";

	programFile << indent << "}\n";
	
	// return the initialized argument object
	programFile << indent << "return programArgs" << stmtSeparator;

	// close function definition;
	programFile << "}\n";

	headerFile.close();
	programFile.close();
}

void generateFnToInitEnvLinksFromEnvironment(TaskDef *taskDef,
                const char *initials,
                List<const char*> *externalEnvLinks,
                const char *headerFileName,
                const char *programFileName) {
	
	std::cout << "Generating routine to initiate environment-link from environment reference\n";
	std::ofstream programFile, headerFile;
        headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
        if (!programFile.is_open() || !headerFile.is_open()) {
                std::cout << "Unable to open output header/program file";
                std::exit(EXIT_FAILURE);
        }

	std::ostringstream fnHeader;
        const char *comments = "function for initializing environment-links object";
	decorator::writeSectionHeader(headerFile, comments);
	headerFile << std::endl;
	decorator::writeSectionHeader(programFile, comments);
	programFile << std::endl;

	// write function signature in header and program files
	programFile << "EnvironmentLinks " << initials << "::";
	headerFile << "EnvironmentLinks ";
	fnHeader << "initiateEnvLinks";
	TupleDef *envTuple = taskDef->getEnvTuple();
	fnHeader << "(" << envTuple->getId()->getName() << " *environment)";
	programFile << fnHeader.str();
	headerFile << fnHeader.str() << stmtSeparator << std::endl;

	// open function definition
	programFile << " {\n";

	// declare a local environment-link variable that will bre returned at the end
	programFile << indent << "EnvironmentLinks envLinks" << stmtSeparator;

	// iterate over all environmental properties
	List<VariableDef*> *propertyList = envTuple->getComponents();
	for (int i = 0; i < propertyList->NumElements(); i++) {
		VariableDef *property = propertyList->Nth(i);
		const char *propertyName = property->getId()->getName();
		// determine if a property is a part of the environmental links
		bool external = false;
		for (int j = 0; j < externalEnvLinks->NumElements(); j++) {
			if (strcmp(propertyName, externalEnvLinks->Nth(j)) == 0) {
				external = true;
				break;
			}
		}
		// if it is a part of the environmental links then copy its value/reference from environment
		if (external) {
			programFile << indent;
			programFile << "envLinks." << propertyName; 
			programFile << " = environment->" << propertyName;
			programFile << stmtSeparator;

			// check if the property is of dynamic array type
			Type *type = property->getType();
			ArrayType *array = dynamic_cast<ArrayType*>(type);
			StaticArrayType *staticArray = dynamic_cast<StaticArrayType*>(type);

			// if the property is a dynamic array then copy dimension info from environemnt too
			if (array != NULL && staticArray == NULL) {
				int dimensions = array->getDimensions();
				for (int d = 0; d < dimensions; d++) {
					programFile << indent;
					programFile << "envLinks." << propertyName;
					programFile << "Dims[" << d << "]";
					programFile << " = environment->" << propertyName;
					programFile << "Dims[" << d << "].partition";
					programFile << stmtSeparator;
				}
			}	
		}
	}

	// return the populated environment link object
	programFile << indent << "return envLinks" << stmtSeparator;
	
	// close function definition
	programFile << "}\n";
	
	headerFile.close();
	programFile.close();
}

void generateTaskExecutor(TaskGenerator *taskGenerator) {
	
	std::cout << "Generating task execute routine \n";
	std::ofstream programFile, headerFile;
        headerFile.open (taskGenerator->getHeaderFile(), std::ofstream::out | std::ofstream::app);
        programFile.open (taskGenerator->getProgramFile(), std::ofstream::out | std::ofstream::app);
        if (!programFile.is_open() || !headerFile.is_open()) {
                std::cout << "Unable to open output header/program file";
                std::exit(EXIT_FAILURE);
        }

	TaskDef *taskDef = taskGenerator->getTaskDef();
	Space *rootLps = taskDef->getPartitionHierarchy()->getRootSpace();
	std::ostringstream fnHeader;
	const char *header = "task executor function";
	decorator::writeSectionHeader(headerFile, header);
	headerFile << std::endl;
	decorator::writeSectionHeader(programFile, header);
	programFile << std::endl;

	// generate the function header
	TupleDef *envTuple = taskDef->getEnvTuple();
	TupleDef *partitionTuple = taskDef->getPartitionTuple();
	fnHeader << "execute(";
	fnHeader << envTuple->getId()->getName() << " *environment";
	InitializeInstr *initSection = taskDef->getInitSection();
	if (initSection != NULL) {
		List<const char*> *arguments = initSection->getArguments();
		if (arguments != NULL) {
			List<Type*> *argTypes = initSection->getArgumentTypes();
			for (int i = 0; i < arguments->NumElements(); i++) {
				fnHeader << paramSeparator << std::endl << doubleIndent;
				Type *type = argTypes->Nth(i);
				const char *arg = arguments->Nth(i);
				fnHeader << type->getCppDeclaration(arg);
			}
		}
	}
	fnHeader << paramSeparator << std::endl << doubleIndent;
	fnHeader << partitionTuple->getId()->getName() << " partition";
	fnHeader << paramSeparator << std::endl << doubleIndent;
	fnHeader << "int segmentId";
	fnHeader << paramSeparator << std::endl << doubleIndent;
	fnHeader << "std::ofstream &logFile";
	fnHeader << ")";

	// write function signature in header and program files
	headerFile << "void " << fnHeader.str() << stmtSeparator;
	programFile << "void " << taskGenerator->getInitials() << "::" << fnHeader.str();

	// open function definition
	programFile << " {\n\n";

	// if the current segment has nothing to do about the task then control should return back to the main 
	// function from here without spending time in vein in any resource management computation
	programFile << indent << "if (segmentId >= Total_Segments) {\n";
	programFile << doubleIndent << "logFile << \"Current segment does not participate in: ";
	programFile << taskGenerator->getTaskName() << "\\n\"" << stmtSeparator;
	if (taskGenerator->hasCommunicators()) {
		programFile << doubleIndent << "excludeFromAllCommunication(";
		programFile << "segmentId" << paramSeparator << "logFile)" << stmtSeparator;
	}	
	programFile << doubleIndent << "return" << stmtSeparator;
	programFile << indent << "}\n\n";
	

	// create an instance of the environment-links object from the environment reference
	programFile << indent << "// initializing environment-links object\n";
	programFile << indent << "EnvironmentLinks envLinks = initiateEnvLinks(environment)" << stmtSeparator;
	programFile << std::endl;

	// declare task's common variables
	programFile << indent << "// declaring other task related common variables\n";
        programFile << indent << "TaskGlobals taskGlobals" << stmtSeparator;
        programFile << indent << "ThreadLocals threadLocals" << stmtSeparator;
        programFile << indent << "ArrayMetadata *metadata = new ArrayMetadata" << stmtSeparator;

	// create a start timer to record running time of different parts of the task
	programFile << std::endl;
	programFile << indent << "// declaring and initiating segment execution timer\n";
	programFile << indent << "struct timeval start" << stmtSeparator;
	programFile << indent << "gettimeofday(&start, NULL)" << stmtSeparator;

	// copy partition parameters into an array to later make them accessible for thread-state management
	taskGenerator->copyPartitionParameters(programFile); 

	// check if synchronization needed and initialize sync primitives if the answer is YES
	SyncManager *syncManager = taskGenerator->getSyncManager();	
        if (syncManager->involvesSynchronization()) {
                programFile << std::endl << indent << "// initializing sync primitives\n";
                programFile << indent << "initializeSyncPrimitives()" << stmtSeparator;
        }

	// get the list of external environment-links then invoke the task initializer function
	List<EnvironmentLink*> *envLinks = taskDef->getEnvironmentLinks();
	List<const char*> *externalEnvLinks = new List<const char*>;
        for (int i = 0; i < envLinks->NumElements(); i++) {
                EnvironmentLink *link = envLinks->Nth(i);
                if (link->isExternal()) {
                	const char *linkName = link->getVariable()->getName();
                	externalEnvLinks->Append(linkName);
		}
	}
	taskGenerator->inovokeTaskInitializer(programFile, externalEnvLinks, true);

	// generate thread-state objects for the intended number of threads and initialize their root LPUs
        taskGenerator->initiateThreadStates(programFile);

	// set up the root LPU reference on each all thread's state variable
        programFile << std::endl << indent;
	programFile << "// setting up root LPU reference in each thread's state\n";
        programFile << indent << "for (int i = 0; i < Total_Threads; i++) {\n";
        programFile << indent << indent;
	programFile << "threadStateList[i]->setRootLpu(metadata)" << stmtSeparator;
        programFile << indent << "}\n";

	// group threads into segments; then allocate and initialize the segment memory for current process
        taskGenerator->performSegmentGrouping(programFile, true);
        taskGenerator->initializeSegmentMemory(programFile);

	// log time spent on memory allocation
	programFile << std::endl;
	programFile << indent << "// calculating memory and threads preparation time\n";
	programFile << indent << "struct timeval end" << stmtSeparator;
	programFile << indent << "gettimeofday(&end, NULL)" << stmtSeparator;
        programFile << indent << "double allocationTime = ((end.tv_sec + end.tv_usec / 1000000.0)";
        programFile << std::endl << indent << indent << indent;
        programFile << "- (start.tv_sec + start.tv_usec / 1000000.0))" << stmtSeparator;
        programFile << indent << "logFile << \"Memory preparation time: \" << allocationTime";
	programFile << " << \" Seconds\" << std::endl" << stmtSeparator;
	programFile << indent << "double timeConsumedSoFar = allocationTime" << stmtSeparator;
	programFile << indent << "logFile.flush()" << stmtSeparator;

	// generate list of communicators that will be used for resolving data dependencies involving communications
	bool communicatorsGenerated = taskGenerator->generateCommunicators(programFile);
	if (communicatorsGenerated) {
		// log time spent on communicator setup
		programFile << std::endl;
		programFile << indent << "// calculating communicators setup time\n";
		programFile << indent << "gettimeofday(&end, NULL)" << stmtSeparator;
        	programFile << indent << "double communicatorTime = ((end.tv_sec + end.tv_usec / 1000000.0)";
        	programFile << std::endl << indent << indent << indent;
        	programFile << "- (start.tv_sec + start.tv_usec / 1000000.0)) - timeConsumedSoFar" << stmtSeparator;
        	programFile << indent << "logFile << \"Communicators setup time: \" << communicatorTime";
		programFile << " << \" Seconds\" << std::endl" << stmtSeparator;
		programFile << indent << "timeConsumedSoFar += communicatorTime" << stmtSeparator;
		programFile << indent << "logFile.flush()" << stmtSeparator << std::endl;
	}

	// read data structures from files if instructed by the coordinator program
	programFile << indent << "initializeEnvironment";
	programFile << "(" << "environment" << paramSeparator;
	programFile << "taskData" << paramSeparator;
	programFile << "configMap" << ")" << stmtSeparator; 
	programFile << indent << "logFile << \"\\tenvironment initialization is complete\\n\"" << stmtSeparator;
	programFile << indent << "logFile.flush()" << stmtSeparator;

	// log time spent on reading data from files
	programFile << std::endl;
	programFile << indent << "// calculating file reading time\n";
	programFile << indent << "gettimeofday(&end, NULL)" << stmtSeparator;
        programFile << indent << "double readingTime = ((end.tv_sec + end.tv_usec / 1000000.0)";
        programFile << std::endl << indent << indent << indent;
        programFile << "- (start.tv_sec + start.tv_usec / 1000000.0)) - timeConsumedSoFar" << stmtSeparator;
        programFile << indent << "logFile << \"Data reading time: \" << readingTime";
	programFile << " << \" Seconds\" << std::endl" << stmtSeparator;
	programFile << indent << "timeConsumedSoFar += readingTime" << stmtSeparator;
	programFile << indent << "logFile.flush()" << stmtSeparator << std::endl;
	
	// start threads and wait for them to finish execution of the task 
        taskGenerator->startThreads(programFile);

	// log time spent on task's computation
	programFile << std::endl;
	programFile << indent << "// calculating computation time\n";
	programFile << indent << "gettimeofday(&end, NULL)" << stmtSeparator;
        programFile << indent << "double computationTime = ((end.tv_sec + end.tv_usec / 1000000.0)";
        programFile << std::endl << indent << indent << indent;
        programFile << "- (start.tv_sec + start.tv_usec / 1000000.0)) - timeConsumedSoFar" << stmtSeparator;
	programFile << indent << "logFile << \"Computation time: \" << computationTime";
	programFile << " << \" Seconds\" << std::endl" << stmtSeparator;
	programFile << indent << "timeConsumedSoFar += computationTime" << stmtSeparator;
	programFile << indent << "logFile.flush()" << stmtSeparator << std::endl;
	
	// memory allocation and communicator setup time should be included in the actual computation time as for a hand-written
	// code those overheads should be insignifant -- the same is not true for file I/0, which should be proportionally costly
	programFile << indent << "double compAndOverheadTime = timeConsumedSoFar - readingTime" << stmtSeparator;
	programFile << indent << "logFile << \"Computation + overhead time: \" << compAndOverheadTime";
	programFile << " << \" Seconds\" << std::endl" << stmtSeparator;
	programFile << indent << "logFile.flush()" << stmtSeparator << std::endl;
	
	// if the task involves communications then log the detailed runtime of communication related activities
	if (communicatorsGenerated) {
		programFile << indent << "commStat->logStatistics(2" << paramSeparator << "logFile)" << stmtSeparator;
		programFile << indent << "logFile.flush()" << stmtSeparator << std::endl;
		programFile << indent << "double commTime = commStat->getTotalCommunicationTime()" << stmtSeparator;
		programFile << indent << "logFile << \"Total communication time: \" << commTime";
        	programFile << " << \" Seconds\" << std::endl" << stmtSeparator;
		programFile << indent << "logFile << \"Computation without communication time: \"";
		programFile << " << computationTime - commTime << \" Seconds\" << std::endl" << stmtSeparator;
		programFile << std::endl;
	}
	
	// write environmental data structures into output files if instructed by the coordinator program through output bindings
	programFile << indent << "// storing outputs in files\n"; 
	programFile << indent << "logFile << \"\\tgoing to write output to files\\n\"" << stmtSeparator;
	programFile << indent << "logFile.flush()" << stmtSeparator;
	programFile << indent << "storeEnvironment";
	programFile << "(" << "environment" << paramSeparator;
	programFile << "taskData" << paramSeparator;
	programFile << "mySegment" << paramSeparator;
	programFile << "configMap" << paramSeparator; 
	programFile << "logFile" << ")" << stmtSeparator; 
	programFile << indent << "logFile << \"\\tfile output is complete\\n\"" << stmtSeparator;
	programFile << indent << "logFile.flush()" << stmtSeparator;
	
	// close function definition
	programFile << "}\n\n";
	
	headerFile.close();
	programFile.close();
}

void generateMain(ProgramDef *programDef, const char *programFile) {
	
	std::cout << "Generating main function for the program\n";
	std::ofstream stream;
        stream.open (programFile, std::ofstream::out | std::ofstream::app);
        if (!stream.is_open()) {
                std::cout << "Unable to open output program file";
                std::exit(EXIT_FAILURE);
        }
	decorator::writeSectionHeader(stream, "main function");

	CoordinatorDef *coordDef = programDef->getProgramController();

        // write the function signature
        stream << "\nint main(int argc, char *argv[]) {\n\n";

	// do MPI initialization
	stream << indent << "MPI_Init(&argc, &argv)" << stmtSeparator << std::endl;

	// retrieve the segment id for the current process
        stream << indent << "// retreiving segmentation identifier\n";
	stream << indent << "int segmentId = 0" << stmtSeparator;
        stream << indent << "MPI_Comm_rank(MPI_COMM_WORLD, &segmentId)" << stmtSeparator << std::endl;

	// start execution time monitoring timer
        stream << indent << "// starting execution timer clock\n";
        stream << indent << "struct timeval start" << stmtSeparator;
        stream << indent << "gettimeofday(&start, NULL)" << stmtSeparator;

	// create a log file for overall program log printing
        stream << std::endl << indent << "// creating a program log file\n";
	stream << indent << "std::ostringstream logFileName" << stmtSeparator;
	stream << indent << "logFileName << " << "\"segment_\" << segmentId << \".log\"" << stmtSeparator;
        stream << indent << "std::ofstream logFile" << stmtSeparator;
        stream << indent << "logFile.open(logFileName.str().c_str())" << stmtSeparator << std::endl;

	// read all command line arguments as key, value pairs
	const char *argName = coordDef->getArgumentName();
	stream << indent << "// reading command line inputs\n";
	stream << indent << "ProgramArgs " << argName << stmtSeparator;
	stream << indent << argName << " = readProgramArgs(argc" << paramSeparator;
	stream << "argv)" << stmtSeparator; 

	// declare all local variables found in scope
	stream << std::endl << indent << "// declaring local variables\n";
	std::ostringstream declStream;
	coordDef->declareVariablesInScope(declStream, 1);
	stream << declStream.str() << std::endl;

	// reset the name transformer to avoid spill over of logic from task generation
	ntransform::NameTransformer::transformer->reset();
	
	// translate the code found inside the coordinator function
	std::ostringstream codeStream;
	coordDef->generateCode(codeStream, programDef->getScope());
	stream << codeStream.str() << std::endl;

	// calculate running time
        stream << indent << "// calculating task running time\n";
        stream << indent << "struct timeval end" << stmtSeparator;
        stream << indent << "gettimeofday(&end, NULL)" << stmtSeparator;
        stream << indent << "double runningTime = ((end.tv_sec + end.tv_usec / 1000000.0)";
        stream << std::endl << indent << indent << indent;
        stream << "- (start.tv_sec + start.tv_usec / 1000000.0))" << stmtSeparator;
        stream << indent << "logFile << \"Execution Time: \" << runningTime << \" Seconds\" << std::endl";
        stream << stmtSeparator;
	stream << indent << "logFile.flush()" << stmtSeparator << std::endl;
	
	// close the log file
        stream << indent << "logFile.close()" << stmtSeparator;
        // display the running time on console
        stream << indent << "std::cout << \"Parallel Execution Time: \" << runningTime <<";
        stream << " \" Seconds\" << std::endl" << stmtSeparator;
	// release MPI resources
	stream << indent << "MPI_Finalize()" << stmtSeparator;
	// then exit the function
        stream << indent << "return 0" << stmtSeparator;
        stream << "}\n";
        stream.close();
}

void processCoordinatorProgram(ProgramDef *programDef, const char *headerFile, const char *programFile) {
        
	std::cout << "\n-----------------------------------------------------------------\n";
        std::cout << "Handling Task-Invocator/Coordinator-Program";
        std::cout << "\n-----------------------------------------------------------------\n";
	
	CoordinatorDef *coordDef = programDef->getProgramController();

	// initializing the header and program files with appropriate include directives
	initiateProgramHeaders(headerFile, programFile, programDef);
	// generating routine for reading program arguments
	generateRoutineToReadProgramArgs(coordDef->getArgumentTuple(), headerFile, programFile);
	// generating the task executing main function
	generateMain(programDef, programFile);
        
	// closing header file definition
	std::ofstream header;
        header.open(headerFile, std::ofstream::app);
	header << "\n#endif\n";
	header.close();
}	
