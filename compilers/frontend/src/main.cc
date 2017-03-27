#include <iostream>
#include <sstream>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

#include "lex/scanner.h"
#include "yacc/parser.h"
#include "syntax/ast.h"
#include "syntax/ast_def.h"
#include "common/errors.h"

int main(int argc, const char *argv[]) {

        //********************************************************* Command Line Arguments Reader
        // read the input argument to determine IT the source file
        const char *sourceFile;
        if (argc < 2) {
                std::cout << "Pass the IT source file as the command line argument" << std::endl;
                return -1;
        } else {
                sourceFile = argv[1];
                std::cout << "Compiling: " << sourceFile << std::endl;
        }

	// redirect standard input to the source file for the front end compiler to work
        int fileDescriptor = open(sourceFile, O_RDONLY);
        if (fileDescriptor < 0) {
                std::cout << "Could not open the source program file" << std::endl;
                return -1;
        }
	dup2(fileDescriptor, STDIN_FILENO);
        close(fileDescriptor);
        //***************************************************************************************

	
	//******************************************************************** Front End Compiler
        /* Entry point to the entire program. InitScanner() is used to set up the scanner.
         * InitParser() is used to set up the parser. The call to yyparse() will attempt to
         * parse a complete program from the input. 
         */
        InitScanner(); 
	InitParser(); 
	yyparse();
        if (ReportError::NumErrors() > 0) return -1;	//-------------exit after syntax analysis
        ProgramDef::program->performScopeAndTypeChecking();
        if (ReportError::NumErrors() > 0) return -1;	//-----------exit after semantic analysis
	ProgramDef::program->performStaticAnalysis();
        //***************************************************************************************
}

