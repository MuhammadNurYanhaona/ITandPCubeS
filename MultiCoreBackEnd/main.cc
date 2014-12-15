/* File: main.cc
 * -------------
 * This file defines the main() routine for the program and not much else.
 */
 
#include <string.h>
#include <stdio.h>
#include "parser.h"
#include "scanner.h"

#include "backend_space_mapping.h"
#include "list.h"

/* Function: main()
 * ----------------
 * Entry point to the entire program. InitScanner() is used to set up the scanner.
 * InitParser() is used to set up the parser. The call to yyparse() will attempt 
 * to parse a complete program from the input. 
 */
int main(int argc, char *argv[]) {
    	//InitScanner();
    	//InitParser();
	//yyparse();
	parsePCubeSDescription("/home/yan/pcubes.ml");
}

