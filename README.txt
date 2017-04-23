Compiler Install/uninstall instructions
---------------------------------------------------------------------------------------------
To install any specific IT compiler, go to the compiler directory and use the make command.
The front-end compiler has only one make file in its root directory. So just a 'make' would
do the compilation. In case of any back-end compiler, you will find two make files in its 
root directory: one for generating the compiler and another for generating the executable
from the native code the compiler produces by compiling an IT source code. Refer to the file
named MakeFile-Compiler using the syntax 
	
	make -f MakeFile-Compiler 

for the make utility to create the compiler.
To uninstall the compiler do:
 
	make -f MakeFile-Compiler clean

Note that any back-end compiler generation also generates object files for the front-end 
compiler codes. If you want to cleanup the front-end compiler's object files along with the
back-end compiler then also issue the following command:
	 
	make -f MakeFile-Compiler clean-front  

Directory structure of the project
---------------------------------------------------------------------------------------------
1. compilers: has source codes for released versions of IT compilers for the front-end and
   different back-ends
2. docs: has instructions for running codes, generating PCubeS descriptions of machines, etc.
3. machine-models: has some sample hardware's PCubeS descriptions
4. tools: has tools that can be used to generate random text/binary data files, compare file
   contents, etc; tools in this directory may come in handy during performance experiments

Contacts
---------------------------------------------------------------------------------------------
Yan: mny9md@virginia.edu
Andrew: grimshaw@virginia.edu
