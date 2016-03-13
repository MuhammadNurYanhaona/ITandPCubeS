#ifndef _H_file_io
#define _H_file_io

/* This library hosts code for generating data structure specific subclasses for reading and writing data parts
   to and from files. In addition, it generates routines to initialize a task's environment and store data from
   the environment to files when corresponding instructions are bound to the environment of task execution.   
*/

#include <fstream>
#include "../syntax/ast_task.h"
#include "../semantics/task_space.h"

/* function to generate partition specific data reader subclass for a data structure within an LPS */
void generatePartReaderForStructure(std::ofstream &headerFile, ArrayDataStructure *structure);

/* function to generate partition specific data writer subclass for a data structure within an LPS */
void generatePartWriterForStructure(std::ofstream &headerFile, ArrayDataStructure *structure);

/* function to generate all reader and writer subclasses for all arrays that an LPS allocates memory for */
void generateReaderWriterForLpsStructures(std::ofstream &headerFile, 
		const char *initials, Space *lps, 
		List<const char*> *envVariables);

/* this function calls the function above to generate classes for all LPSes */
void generateReaderWriters(const char *headerFile, const char *initials, TaskDef *taskDef);

/* Since data writers are statefull objects (they depend on existance of valid computations the current segment
   supposed to execute in an LPS, segment's position in the segment ordering, etc.) and at the same time needed
   to be executed from the outside-task environment management library, we create a map of preconfigured data 
   writers at task invocation. This function generates the method for creating the map.	*/
void generateWritersMap(const char *headerFile, 
		const char *programFile, 
		const char *initials, TaskDef *taskDef);

/* similar to the above, this generates a routine to create a map of Part-Readers for environment variables*/
void generateReadersMap(const char *headerFile, 
		const char *programFile, 
		const char *initials, TaskDef *taskDef);

#endif
