#include "fn_generator.h"

#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/string_utils.h"
#include "../../../../common-libs/utils/decorator_utils.h"
#include "../../../../frontend/src/syntax/ast_def.h"
#include "../../../../frontend/src/codegen-helper/extern_config.h"

#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>

void generateFnInstances(FunctionDef *fnDef, std::ofstream &headerFile, std::ofstream &programFile) {
	
	const char *fnName = fnDef->getId()->getName();
	std::ostringstream messageStream;
	messageStream << fnName << " instances";
	const char *message = strdup(messageStream.str().c_str());
	decorator::writeSectionHeader(headerFile, message);
	decorator::writeSectionHeader(programFile, message);
	headerFile << std::endl;
	programFile << std::endl;

	List<FunctionInstance*> *instanceList = fnDef->getInstanceList();
	for (int i = 0; i < instanceList->NumElements(); i++) {
		FunctionInstance *instance = instanceList->Nth(i);
		instance->generateCode(headerFile, programFile);
	}
}

void generateFunctions(List<Definition*> *fnDefList, 
		const char *headerFileName, 
		const char *programFileName) {

	std::string line;
        std::ifstream commIncludeFile("config/default-includes.txt");
        std::ofstream programFile, headerFile;
        headerFile.open (headerFileName, std::ofstream::out);
        programFile.open (programFileName, std::ofstream::out);
        if (!programFile.is_open()) {
                std::cout << "Unable to open output program file for user defined functions\n";
                std::exit(EXIT_FAILURE);
        }
        if (!headerFile.is_open()) {
                std::cout << "Unable to open output header file for user defined functions\n";
                std::exit(EXIT_FAILURE);
        }

        headerFile << "#ifndef _H_user_fn" << std::endl;
        headerFile << "#define _H_user_fn" << std::endl << std::endl;

	generateLibraryIncludes(fnDefList, headerFile, programFile);

	for (int i = 0; i < fnDefList->NumElements(); i++) {
		FunctionDef *fnDef = (FunctionDef*) fnDefList->Nth(i);
		generateFnInstances(fnDef, headerFile, programFile);
	}

	headerFile << std::endl << "#endif" << std::endl;
	
	headerFile.close();
	programFile.close();
}

void generateLibraryIncludes(List<Definition*> *fnDefList, 
                std::ofstream &headerFile, std::ofstream &programFile) {

	// include the header file for user defined type definition for the output header file
	decorator::writeSectionHeader(headerFile, "header file for user defined IT types");
	headerFile << std::endl << "#include \"tuple.h\"" << std::endl << std::endl;

	// include the common library header files in the output program file
	std::ifstream commIncludeFile("config/default-includes.txt");
        if (commIncludeFile.is_open()) {
		const char *message = "common header files for different purposes";
		decorator::writeSectionHeader(programFile, message);
		programFile << std::endl;
		std::string line;
                while (std::getline(commIncludeFile, line)) {
                        programFile << line << std::endl;
                }
                programFile << std::endl;
        } else {
                std::cout << "Unable to open common include file during user defined function generation\n";
                std::exit(EXIT_FAILURE);
        }
	
	// identify header files for all extern code blocks found in the user defined functions
	List<const char*> *headerIncludes = new List<const char*>;
	for (int i = 0; i < fnDefList->NumElements(); i++) {
                FunctionDef *fnDef = (FunctionDef*) fnDefList->Nth(i);
                IncludesAndLinksMap *externConfig = fnDef->getExternBlocksHeadersAndLibraries();

		// Since we are generating C++ code any external code block written in C and C++ can be directly 
		// placed within the generated code but we have to include the proper header files in the 
		// generated program to make this scheme work. So here we are filtering C and C++ headers
		if (externConfig->hasExternBlocksForLanguage("C++")) {
			LanguageIncludesAndLinks *cPlusHeaderAndLinks
					= externConfig->getIncludesAndLinksForLanguage("C++");
			string_utils::combineLists(headerIncludes, cPlusHeaderAndLinks->getHeaderIncludes());
		}
		if (externConfig->hasExternBlocksForLanguage("C")) {
			LanguageIncludesAndLinks *cHeaderAndLinks
					= externConfig->getIncludesAndLinksForLanguage("C");
			string_utils::combineLists(headerIncludes, cHeaderAndLinks->getHeaderIncludes());
		}
	}

	if (headerIncludes->NumElements() == 0) return;

	// include the extern code blocks' libraries in both output header and program files
	const char *message = "header files needed to execute external code blocks";
	decorator::writeSectionHeader(headerFile, message);
	decorator::writeSectionHeader(programFile, message);
	std::ostringstream includeStream; 
	includeStream << std::endl;
	for (int i = 0; i < headerIncludes->NumElements(); i++) {
		includeStream << "#include ";
		const char *headerInclude = headerIncludes->Nth(i);
		if (headerInclude[0] == '"') {
			includeStream << headerInclude << std::endl;
		} else {
			includeStream << '<' << headerInclude << '>' << std::endl;
		}
	}
	includeStream << std::endl;
	headerFile << includeStream.str();
	programFile << includeStream.str();
}
