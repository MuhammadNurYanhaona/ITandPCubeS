#include <iostream>
#include <sstream>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

#include "codegen/utils/space_mapping.h"
#include "codegen/utils/code_generator.h"
#include "codegen/utils/task_invocation.h"
#include "codegen/utils/fn_generator.h"

#include "../../common-libs/utils/list.h"
#include "../../common-libs/utils/properties.h"
#include "../../common-libs/domain-obj/constant.h"

#include "../../frontend/src/lex/scanner.h"
#include "../../frontend/src/yacc/parser.h"
#include "../../frontend/src/common/errors.h"
#include "../../frontend/src/syntax/ast.h"
#include "../../frontend/src/syntax/ast_def.h"
#include "../../frontend/src/syntax/ast_task.h"

int main(int argc, const char *argv[]) {

	//********************************************************* Command Line Arguments Reader
	// read the input arguments and keep track of the appropriate files
	const char *sourceFile, *pcubesFile, *processorFile, *mappingFile;
	if (argc < 5) {
		std::cout << "You need to pass four files as input arguments" << std::endl;
		std::cout << "\t" << "1. The input IT program file" << std::endl;
		std::cout << "\t" << "2. The PCubeS model of the machine" << std::endl;
		std::cout << "\t" << "3. A processor description file for the machine\n";
		std::cout << "\t" << "4. The mapping configuration file" << std::endl;
		return -1;
	} else {
		std::cout << "Compilation config---------------------------------------\n";
		sourceFile = argv[1];
		std::cout << "Source program: " << sourceFile << std::endl;
		pcubesFile = argv[2];
		std::cout << "PCubeS model: " << pcubesFile << std::endl;
		processorFile = argv[3];
		std::cout << "Processor configuration: " << processorFile << std::endl;	
		mappingFile = argv[4];
		std::cout << "Mapping Configuration: " << mappingFile << std::endl;	
	}
	// an additional optional parameter is supported to generate the C++ files for the
        // program in some directory other than the default directory 
        const char *buildSubDir;
        if (argc > 5) {
                buildSubDir = argv[5];
        } else buildSubDir = "tmp";
        // create a build subdirectory for the program
        const char *buildDir = "build/";
        std::ostringstream outputDirStr;
        outputDirStr << buildDir << buildSubDir << "/";
        mkdir(outputDirStr.str().c_str(), 0700);
	// redirect standard input to the source file for the front end compiler to work
	int fileDescriptor = open(sourceFile, O_RDONLY);
	if (fileDescriptor < 0) {
		std::cout << "Could not open the source program file" << std::endl;
		return -1;
	}
	dup2(fileDescriptor, STDIN_FILENO);
	close(fileDescriptor);
	// set the output directory and common header file parameters
        const char *outputDir = strdup(outputDirStr.str().c_str());
        std::ostringstream tupleHeaderStr;
        tupleHeaderStr << buildDir << buildSubDir << "/tuple.h";
        const char *tupleHeader = strdup(tupleHeaderStr.str().c_str());
        // set the output header and program files for the coordinator program
        std::ostringstream coordHeaderStr;
        coordHeaderStr << buildDir << buildSubDir << "/coordinator.h";
        const char *coordHeader = strdup(coordHeaderStr.str().c_str());
        std::ostringstream coordProgramStr;
        coordProgramStr << buildDir << buildSubDir << "/coordinator.cc";
        const char *coordProgram = strdup(coordProgramStr.str().c_str());
	// set the output header and program files for the user defined functions
        std::ostringstream fnHeaderStr;
        fnHeaderStr << buildDir << buildSubDir << "/function.h";
        const char *fnHeader = strdup(fnHeaderStr.str().c_str());
        std::ostringstream fnProgramStr;
        fnProgramStr << buildDir << buildSubDir << "/function.cc";
        const char *fnProgram = strdup(fnProgramStr.str().c_str());
	// set up the output text file for any external library linking during native code
	// compilation process
        std::ostringstream linkageListerStr;
        linkageListerStr << buildDir << buildSubDir << "/external_links.txt";
        const char *linkageListerFile = strdup(linkageListerStr.str().c_str());
	//***************************************************************************************


	//*********************************************************Compiler Property Files Reader
        const char *deploymentKey = "deployment";
        const char *deploymentPropertiesFile = "config/deployment.properties";
        PropertyReader::readPropertiesFile(deploymentPropertiesFile, deploymentKey);
        //***************************************************************************************

	
	//******************************************************************** Front End Compiler
	/* Entry point to the entire program. InitScanner() is used to set up the scanner.
         * InitParser() is used to set up the parser. The call to yyparse() will attempt to
         * parse a complete program from the input. 
         */
        InitScanner();
        InitParser();
        yyparse();
        if (ReportError::NumErrors() > 0) return -1;    //-------------exit after syntax analysis
        ProgramDef::program->performScopeAndTypeChecking();
        if (ReportError::NumErrors() > 0) return -1;    //-----------exit after semantic analysis
        ProgramDef::program->performStaticAnalysis();
        if (ReportError::NumErrors() > 0) return -1;    //-------------exit after static analysis
	//***************************************************************************************



	//********************************************************************* Back End Compiler
	// parse PCubeS description of the multicore hardware
        List<PPS_Definition*> *pcubesConfig = parsePCubeSDescription(pcubesFile);
	// iterate over list of tasks and generate code for each of them in separate files
	List<Definition*> *taskDefs = ProgramDef::program->getComponentsByType(TASK_DEF);
        for (int i = 0; i < taskDefs->NumElements(); i++) {
                TaskDef *taskDef = (TaskDef*) taskDefs->Nth(i);
                // update the static reference to get to the task definition from anywhere 
                // during code generation  
                TaskDef::currentTask = taskDef;
		// instanciate a task generator
		TaskGenerator *generator = new TaskGenerator(taskDef,
				outputDir, 
				mappingFile, processorFile);
		// generate all codes relevent to the current task	
                generator->generate(pcubesConfig);
	}
	// reset the current task reference before handling of user defined functions and 
	// the program coordinator function
        TaskDef::currentTask = NULL;
	// generate classes for the list of tuples present in the source in a header file
	List<TupleDef*> *classDefs = ProgramDef::program->getAllCustomTypes();
        generateClassesForTuples(tupleHeader, classDefs);
	// generate definitions for all user-defined IT functions
	List<Definition*> *functionDefs = ProgramDef::program->getComponentsByType(FN_DEF);
	generateFunctions(functionDefs, fnHeader, fnProgram);
        // invoke the library handling task-invocations to generate all routines needed for 
        // multi-task management and a the main function corresponds to the coordinator 
        // program definition
        processCoordinatorProgram(ProgramDef::program, coordHeader, coordProgram);
        // generate a text file that lists the external libraries to be linked with the 
        // program for successful compilation and execution of external code blocks
        generateExternLibraryLinkInfo(linkageListerFile);
	//***************************************************************************************
}

