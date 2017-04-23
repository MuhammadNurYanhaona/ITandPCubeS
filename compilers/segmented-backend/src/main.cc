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

	//********************************************************************* Compiler Configuration Reader
	const char *pcubesFile = "config/pcubes-description.ml";
	const char *processorFile = "config/pcubes-description.cn";
	// check if the two files exist by trying to open them
	int pcubesFileDescriptor = open(pcubesFile, O_RDONLY);
	int processorFileDescriptor = open(processorFile, O_RDONLY);
	if (pcubesFileDescriptor  < 0 || processorFileDescriptor < 0) {
		std::cerr << "The compiler has not been configured yet\n";
		std::cerr << "\tThe PCubeS description of the target hardware is needed\n";
		std::cerr << "\tCopy your pcubes model file in config/pcubes-description.ml\n";
		std::cerr << "\tThen copy a core numbering file in config/pcubes-description.cn\n";
		std::cerr << "Examples of these files are available in the sample directory\n";
		return -1;
	} else {
		close(pcubesFileDescriptor);
		close(processorFileDescriptor);
	}
	//***************************************************************************************************


	//********************************************************************* Command Line Arguments Reader
	// read the input arguments and keep track of the appropriate files
	const char *sourceFile, *mappingFile;
	if (argc < 3) {
		std::cerr << "You need to pass two files as input arguments" << std::endl;
		std::cerr << "\t" << "1. The input IT program file" << std::endl;
		std::cerr << "\t" << "2. The mapping configuration file" << std::endl;
		return -1;
	} else {
		std::cout << "Compilation config---------------------------------------\n";
		sourceFile = argv[1];
		std::cout << "Source program: " << sourceFile << std::endl;
		mappingFile = argv[2];
		std::cout << "Mapping Configuration: " << mappingFile << std::endl;	
	}
	// an additional optional parameter is supported to generate the C++ files for the program in some 
	// directory other than the default directory 
        const char *buildSubDir;
        if (argc > 3) {
                buildSubDir = argv[3];
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
	//***************************************************************************************************


	//******************************************************************Compilation Settings Files Reader
        const char *deploymentKey = "deployment";
        const char *deploymentPropertiesFile = "config/deployment.properties";
        PropertyReader::readPropertiesFile(deploymentPropertiesFile, deploymentKey);
        //***************************************************************************************************

	
	//******************************************************************************** Front End Compiler
        InitScanner();
        InitParser();
        yyparse();
        if (ReportError::NumErrors() > 0) return -1;    //-------------exit after syntax analysis
        ProgramDef::program->performScopeAndTypeChecking();
        if (ReportError::NumErrors() > 0) return -1;    //-----------exit after semantic analysis
        ProgramDef::program->performStaticAnalysis();
        if (ReportError::NumErrors() > 0) return -1;    //-------------exit after static analysis
        ProgramDef::program->prepareForCodegen();
        //***************************************************************************************************


	//********************************************************************************* Back End Compiler
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
        //***************************************************************************************************

	std::cout << "\n\n********************** Compilation Successful!!!\n\n";
}

