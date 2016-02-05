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
        std::ifstream commIncludeFile("config/default-includes.txt");
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
        headerFile << "#define _H_coordinator\n\n";

	// include the header file for coordinator program in its program file
        int programHeaderIndex = string_utils::getLastIndexOf(headerFileName, '/') + 1;
        char *programHeader = string_utils::substr(headerFileName, 
			programHeaderIndex, strlen(headerFileName));
	const char *message1 = "header file for the coordinator program";
        decorator::writeSectionHeader(programFile, message1);
        programFile << "#include \"" << programHeader  << '"' << std::endl << std::endl;

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
	commonHeaders << std::endl;
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

	std::string indent = "\t";
	std::string stmtSeparator = ";\n";
	std::string paramSeparator = ", "; 
	std::ostringstream fnHeader;

	std::ostringstream  comments;
        comments << "/*-----------------------------------------------------------------------------------\n";
        comments << "function for initializing program arguments\n";
       	comments << "------------------------------------------------------------------------------------*/\n\n";
	headerFile << comments.str();
	programFile << comments.str();

	// write function signature in header and program files
	const char *tupleName = programArg->getId()->getName();
	fnHeader << tupleName << " getProgramArgs()";
	headerFile << fnHeader.str() << stmtSeparator;
	programFile << fnHeader.str();

	// start function definition
	programFile << " {\n";
	
	// create a new instance of the program argument object
	programFile << indent << tupleName << " programArgs = " << tupleName << "()" << stmtSeparator;

	// iterate over the list of properties and generate prompt for initiating each of them
	List<VariableDef*> *propertyList = programArg->getComponents();
	for (int i = 0; i < propertyList->NumElements(); i++) {
		VariableDef *property = propertyList->Nth(i);
		const char *propertyName = property->getId()->getName();
		programFile << indent << "programArgs." << propertyName << " = ";	
		Type *propertyType = property->getType();
		if (propertyType == Type::stringType) {
			programFile << "inprompt::readString";	
		} else if (propertyType == Type::boolType) {
			programFile << "inprompt::readBoolean";	
		} else {
			programFile << "inprompt::readPrimitive ";	
			programFile << "<" << propertyType->getCType() << "> ";
		}
		programFile << "(\"" << propertyName << "\")" ;
		programFile << stmtSeparator;
	}
	
	// return the initialized argument object
	programFile << indent << "return programArgs" << stmtSeparator;

	// close function definition;
	programFile << "}\n\n";

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
        programFile << "}\n\n";

        headerFile.close();
        programFile.close();
}


void generateFnToInitEnvLinksFromEnvironment(TaskDef *taskDef,
                const char *initials,
                List<const char*> *externalEnvLinks,
                const char *headerFileName,
                const char *programFileName) {
	
	std::cout << "\tGenerating routine to initiate environment-link from environment reference\n";
	std::ofstream programFile, headerFile;
        headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
        if (!programFile.is_open() || !headerFile.is_open()) {
                std::cout << "Unable to open output header/program file";
                std::exit(EXIT_FAILURE);
        }

	std::string indent = "\t";
	std::string stmtSeparator = ";\n";
	std::string paramSeparator = ", "; 
	std::ostringstream fnHeader;

	std::ostringstream  comments;
        comments << "/*-----------------------------------------------------------------------------------\n";
        comments << "function for initializing environment-links object\n";
       	comments << "------------------------------------------------------------------------------------*/\n\n";
	headerFile << comments.str();
	programFile << comments.str();

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
	programFile << "}\n\n";
	
	headerFile.close();
	programFile.close();
}

void generateFnToInitTaskRootFromEnv(TaskDef *taskDef,
                const char *initials,
                const char *headerFileName,
                const char *programFileName) {

	std::cout << "\tGenerating routine to initiate thread's root LPU from Environment reference \n";
	std::ofstream programFile, headerFile;
        headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
        if (!programFile.is_open() || !headerFile.is_open()) {
                std::cout << "Unable to open output header/program file";
                std::exit(EXIT_FAILURE);
        }

	Space *rootLps = taskDef->getPartitionHierarchy()->getRootSpace();
	std::string indent = "\t";
	std::string doubleIndent = "\t\t";
	std::string stmtSeparator = ";\n";
	std::string paramSeparator = ", "; 
	std::ostringstream fnHeader;

	std::ostringstream  comments;
        comments << "/*-----------------------------------------------------------------------------------\n";
        comments << "function for initializing root LPU from environment\n";
       	comments << "------------------------------------------------------------------------------------*/\n\n";
	headerFile << comments.str();
	programFile << comments.str();
	
	// write function signature in header and program files
	programFile << "Space" << rootLps->getName() << "_LPU *" << initials << "::";
	headerFile << "Space" << rootLps->getName() << "_LPU *";
	fnHeader << "initiateRootLpu";
	TupleDef *envTuple = taskDef->getEnvTuple();
	fnHeader << "(" << envTuple->getId()->getName() << " *environment";
	fnHeader << paramSeparator << "ArrayMetadata *metadata)";
	programFile << fnHeader.str();
	headerFile << fnHeader.str() << stmtSeparator << std::endl;

	// open function definition
	programFile << " {\n\n";

	// allocate a pointer reference for the root LPU instance
	programFile << indent;
	programFile << "Space" << rootLps->getName() << "_LPU *rootLpu = ";
	programFile << "new Space" << rootLps->getName() << "_LPU";
	programFile << stmtSeparator;
	
	// get the names of objects that are part of the environment
	List<VariableDef*> *propertyList = envTuple->getComponents();
	List<const char*> *envProperties = new List<const char*>;
	for (int i = 0; i < propertyList->NumElements(); i++) {
		VariableDef *property = propertyList->Nth(i);
		envProperties->Append(property->getId()->getName());
	}	

	// initialize each array present in the root LPU
        List<const char*> *localArrays = rootLps->getLocallyUsedArrayNames();
        for (int i = 0; i < localArrays->NumElements(); i++) {
                if (i > 0) programFile << std::endl;
                const char* arrayName = localArrays->Nth(i);
		ArrayDataStructure *array = (ArrayDataStructure*) rootLps->getLocalStructure(arrayName);
                int dimensionCount = array->getDimensionality();
                
		// copy the pointer reference to memory location from LPS content to the LPU
		programFile << indent << "rootLpu->" << arrayName << " = " ;
		programFile << "space" << rootLps->getName() << "Content.";
		programFile << arrayName << stmtSeparator;

		// check if the array is part of the environment
		bool partOfEnv = common_utils::isStringInList(arrayName, envProperties);
		// if it is part of the environment then try to initiate its metadata from the 
		// environment reference first
		std::string currentIndent = indent;
		if (partOfEnv) {
			programFile << indent << "if (environment->" << arrayName << " != NULL) {\n";
			currentIndent = doubleIndent;
			for (int d = 0; d < dimensionCount; d++) {
				programFile << currentIndent << "rootLpu->" << arrayName;
				programFile << "PartDims[" << d << "] = ";
				programFile << "environment->" << arrayName;
				programFile << "Dims[" << d << "]" << stmtSeparator;
			}
			programFile << indent << "} else {\n";
		}
		// if initialization from environment reference is unsuccessful or not applicable then
		// initiate the metadata based on the array metadata object
		std::ostringstream varName;
                varName << "rootLpu->" << arrayName << "PartDims";
                for (int j = 0; j < dimensionCount; j++) {
                        programFile << currentIndent << varName.str() << "[" << j << "] = ";
                        programFile << "PartDimension()" << stmtSeparator;
                        programFile << currentIndent << varName.str() << "[" << j << "].partition = ";
                        programFile << "metadata->" << arrayName << "Dims[" << j << "]";
                        programFile << stmtSeparator;
                        programFile << currentIndent << varName.str() << "[" << j << "].storage = ";
                        programFile << "metadata->" << arrayName << "Dims[" << j;
                        programFile << "].getNormalizedDimension()" << stmtSeparator;
                }	 
		
		if (partOfEnv) programFile << indent << "}\n";
	}

	// return the root LPU reference
	programFile << indent << "return rootLpu" << stmtSeparator;
	
	// close function definition
	programFile << "}\n\n";
	
	headerFile.close();
	programFile.close();
}

void generateTaskExecutor(TaskGenerator *taskGenerator) {
	
	std::cout << "\tGenerating task execute routine \n";
	std::ofstream programFile, headerFile;
        headerFile.open (taskGenerator->getHeaderFile(), std::ofstream::out | std::ofstream::app);
        programFile.open (taskGenerator->getProgramFile(), std::ofstream::out | std::ofstream::app);
        if (!programFile.is_open() || !headerFile.is_open()) {
                std::cout << "Unable to open output header/program file";
                std::exit(EXIT_FAILURE);
        }

	TaskDef *taskDef = taskGenerator->getTaskDef();
	Space *rootLps = taskDef->getPartitionHierarchy()->getRootSpace();
	std::string indent = "\t";
	std::string doubleIndent = "\t\t";
	std::string stmtSeparator = ";\n";
	std::string paramSeparator = ", "; 
	std::ostringstream fnHeader;

	std::ostringstream  comments;
        comments << "/*-----------------------------------------------------------------------------------\n";
        comments << "function for executing task\n";
       	comments << "------------------------------------------------------------------------------------*/\n\n";
	headerFile << comments.str();
	programFile << comments.str();

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
	fnHeader << "std::ofstream &logFile";
	fnHeader << ")";

	// write function signature in header and program files
	headerFile << "void " << fnHeader.str() << stmtSeparator;
	programFile << "void " << taskGenerator->getInitials() << "::" << fnHeader.str();

	// open function definition
	programFile << " {\n\n";

	// create an instance of the environment-links object from the environment reference
	programFile << indent << "// initializing environment-links object\n";
	programFile << indent << "EnvironmentLinks envLinks = initiateEnvLinks(environment)" << stmtSeparator;
	programFile << std::endl;

	// declare task's common variables
	programFile << indent << "// declaring other task related common variables\n";
        programFile << indent << "TaskGlobals taskGlobals" << stmtSeparator;
        programFile << indent << "ThreadLocals threadLocals" << stmtSeparator;
        programFile << indent << "ArrayMetadata *metadata = new ArrayMetadata" << stmtSeparator;

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

	// invoke functions to initialize array references in different LPSes
        programFile << std::endl << indent << "// allocating memories for data structures\n";
        programFile << indent << "initializeRootLPSContent(&envLinks, metadata)" << stmtSeparator;
        programFile << indent << "initializeLPSesContents(metadata)" << stmtSeparator;

	// initiate a root LPU reference based on the environment reference
	programFile << std::endl << indent << "// initializing the root LPU reference\n";
	programFile << indent << "Space" << rootLps->getName() << "_LPU *rootLpu = ";
	programFile << "initiateRootLpu(environment, metadata)" << stmtSeparator;

	// generate thread-state objects for the intended number of threads and initialize their 
	// root LPUs
        taskGenerator->initiateThreadStates(programFile);

	// set up the root LPU reference on each all thread's state variable
        programFile << std::endl << indent;
	programFile << "// setting up root LPU reference in each thread's state\n";
        programFile << indent << "for (int i = 0; i < Total_Threads; i++) {\n";
        programFile << indent << indent;
	programFile << "threadStateList[i]->setRootLpu(rootLpu)" << stmtSeparator;
        programFile << indent << "}\n";

	// start threads and wait for them to finish execution of the task 
        taskGenerator->startThreads(programFile);

	// copy updated environmental references from root-LPS content and task-global-scalar into
	// the original environment reference	
        programFile << indent << "// copying results of task execution into environment\n";
	List<VariableDef*> *propertyList = envTuple->getComponents();
	List<const char*> *propertyNames = new List<const char*>;
	for (int i = 0; i < propertyList->NumElements(); i++) {
		
		VariableDef *property = propertyList->Nth(i);
		Type *propertyType = property->getType();
		const char *propertyName = property->getId()->getName();
		propertyNames->Append(propertyName);	
		
		// if the data structure is a dynamic array, we copy it and its metadata from root LPS
		ArrayType *array = dynamic_cast<ArrayType*>(propertyType);
		StaticArrayType *staticArray = dynamic_cast<StaticArrayType*>(propertyType);
		if (array != NULL && staticArray == NULL) {
			programFile << indent;
			programFile << "environment->" << propertyName << " = ";
			programFile << "rootLpu->" << propertyName;
			programFile << stmtSeparator;
			int dimensions = array->getDimensions();
			for (int d = 0; d < dimensions; d++) {
				programFile << indent;
				programFile << "environment->" << propertyName;
				programFile << "Dims[" << d << "] = ";
				programFile << "rootLpu->" << propertyName << "PartDims[" << d << "]";
				programFile << stmtSeparator;
			}
		// otherwise, copy the property from task-global object into the environment 
		} else if (strcmp(propertyName, "name") != 0) {
			programFile << indent;
			programFile << "environment->" << propertyName << " = ";
			programFile << "taskGlobals." << propertyName;
			programFile << stmtSeparator;
		}	
	}	

	// remove all arrays from the execution context that are not part of the environment
        List<const char*> *localArrays = rootLps->getLocallyUsedArrayNames();
	bool removed = false;
	for (int i = 0; i < localArrays->NumElements(); i++) {
		const char *array = localArrays->Nth(i);
		if (!common_utils::isStringInList(array, propertyNames)) {
			if (!removed) {
				programFile << std::endl << indent;
				programFile << "// removing by-products of task execution\n";
			}
			removed = true;
			programFile << indent;
			programFile << "delete [] space" << rootLps->getName() << "Content." << array;
			programFile << stmtSeparator;
		}
	}	
	
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
        stream << "\nint main(int argc" << paramSeparator << "char *argv[]) {\n\n";

	// start execution time monitoring timer
        stream << indent << "// starting execution timer clock\n";
        stream << indent << "struct timeval start" << stmtSeparator;
        stream << indent << "gettimeofday(&start, NULL)" << stmtSeparator;

	// create a log file for overall program log printing
        stream << std::endl << indent << "// creating a program log file\n";
        stream << indent << "std::cout << \"Creating diagnostic log: it-program.log\\n\"" << stmtSeparator;
        stream << indent << "std::ofstream logFile" << stmtSeparator;
        stream << indent << "logFile.open(\"it-program.log\")" << stmtSeparator << std::endl;

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
        stream << stmtSeparator << std::endl;
	
	// close the log file
        stream << indent << "logFile.close()" << stmtSeparator;
        // display the running time on console
        stream << indent << "std::cout << \"Parallel Execution Time: \" << runningTime <<";
        stream << " \" Seconds\" << std::endl" << stmtSeparator;
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
	// generating routine for initializing program arguments
	generateRoutineToReadProgramArgs(coordDef->getArgumentTuple(), headerFile, programFile);
	// generating the task executing main function
	generateMain(programDef, programFile);
        
	// closing header file definition
	std::ofstream header;
        header.open(headerFile, std::ofstream::app);
	header << "\n#endif\n";
	header.close();
}	
