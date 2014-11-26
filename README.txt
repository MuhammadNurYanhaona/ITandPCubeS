The sequence of phases for the compiler is
Front End
	1. Lexical Analysis
	2. Parsing
	3. AST Generation
	4. Scope and Type Checking
	5. Static Analysis
Back End
	1. Multi-core Backend 

We made changes as needed along the development process. So always look at the last phase 
to get most up-to-date status of the language syntax and its features.

Install the following tools to be able to generate the front end compiler
	1. bison
	2. flex
	3. g++
