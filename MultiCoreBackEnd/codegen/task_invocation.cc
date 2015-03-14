#include "task_invocation.h"
#include "task_generator.h"
#include "code_generator.h"
#include "../syntax/ast_def.h"
#include "../syntax/ast_type.h"
#include "../semantics/scope.h"
#include "../semantics/task_space.h"
#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "../utils/string_utils.h"
#include "../utils/common_utils.h"

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
        headerFile << "#define _H_coordinator\n\n";

	// include the header file for coordinator program in its program file
        int programHeaderIndex = string_utils::getLastIndexOf(headerFileName, '/') + 1;
        char *programHeader = string_utils::substr(headerFileName, 
			programHeaderIndex, strlen(headerFileName));
        programFile << "/*-----------------------------------------------------------------------------------\n";
        programFile << "header file for the coordinator program" << std::endl;
        programFile << "------------------------------------------------------------------------------------*/\n\n";
        programFile << "#include \"" << programHeader  << '"' << std::endl << std::endl;

	// prepare an stream to write all header that should be included in both files
	std::ostringstream commonHeaders;

	// retrieve the list of default libraries from an external file
        commonHeaders << "/*-----------------------------------------------------------------------------------\n";
        commonHeaders << "header files included for different purposes" << std::endl;
       	commonHeaders << "------------------------------------------------------------------------------------*/\n\n";
	if (commIncludeFile.is_open()) {
                while (std::getline(commIncludeFile, line)) {
                        commonHeaders << line << std::endl;
                }
        } else {
                std::cout << "Unable to open common include file";
                std::exit(EXIT_FAILURE);
        }

	// include headers for all tasks in the library list;
        commonHeaders << "/*-----------------------------------------------------------------------------------\n";
        commonHeaders << "header files for tasks" << std::endl;
       	commonHeaders << "------------------------------------------------------------------------------------*/\n\n";
	List<TaskDef*> *taskList = programDef->getTasks();
	for (int i = 0; i < taskList->NumElements(); i++) {
		commonHeaders << "#include \"";
		commonHeaders << TaskGenerator::getHeaderFileName(taskList->Nth(i)) << "\"\n";
	}
	commonHeaders << std::endl;

	headerFile << commonHeaders.str();
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
	fnHeader << tupleName << " *getProgramArgs()";
	headerFile << fnHeader.str() << stmtSeparator;
	programFile << fnHeader.str();

	// start function definition
	programFile << " {\n";
	
	// create a new instance of the program argument object
	programFile << indent << tupleName << " programArgs = new " << tupleName << "()" << stmtSeparator;

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
	headerFile << fnHeader.str() << stmtSeparator;

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
                programFile << indent << "rootLpu->" << arrayName << " = NULL" << stmtSeparator;

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
				programFile << "Dims[" << d << "] = ";
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

void processCoordinatorProgram(ProgramDef *programDef, const char *headerFile, const char *programFile) {
        
	std::cout << "\n-----------------------------------------------------------------\n";
        std::cout << "Handling Task-Invocator/Coordinator-Program";
        std::cout << "\n-----------------------------------------------------------------\n";
	
	CoordinatorDef *coordDef = programDef->getProgramController();

	// initializing the header and program files with appropriate include directives
	initiateProgramHeaders(headerFile, programFile, programDef);
	// generating routine for initializing program arguments
	generateRoutineToInitProgramArgs(coordDef->getArgumentTuple(), headerFile, programFile);
        
	// closing header file definition
	std::ofstream header;
        header.open(headerFile, std::ofstream::app);
	header << "\n#endif\n";
	header.close();
}	
