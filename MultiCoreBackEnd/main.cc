/* File: main.cc
 * -------------
 * This file defines the main() routine for the program and not much else.
 */
 
#include <iostream>
#include <string.h>
#include <stdio.h>

#include "parser.h"
#include "scanner.h"
#include "backend_space_mapping.h"
#include "list.h"
#include "ast_def.h"
#include "errors.h"

int main(int argc, char *argv[]) {
	
	//*************************************************************** Front End Compiler
 	/* Entry point to the entire program. InitScanner() is used to set up the scanner.
	 * InitParser() is used to set up the parser. The call to yyparse() will attempt to
	 * parse a complete program from the input. 
	 */
    	InitScanner();
    	InitParser();
	yyparse();
	if (ReportError::NumErrors() > 0) return -1;
	//**********************************************************************************


	//**************************************************************** Back End Compiler
	List<PPS_Definition*> *pcubesConfig = parsePCubeSDescription("/home/yan/pcubes.ml");
	TaskDef *luTask = (TaskDef*) ProgramDef::program->getTaskDefinition("LU Factorization");
	if (luTask == NULL) std::cout << "could not find LU factorization task\n";
	PartitionHierarchy *lpsHierarchy = luTask->getPartitionHierarchy();
	MappingNode *mappingConfig = parseMappingConfiguration("LU Factorization",
        		"/home/yan/opteron-solver-mapping.map", lpsHierarchy, pcubesConfig);
	generateLPSMacroDefinitions("/home/yan/output.cpp", mappingConfig);
	generatePPSCountMacros("/home/yan/output.cpp", pcubesConfig);
}

