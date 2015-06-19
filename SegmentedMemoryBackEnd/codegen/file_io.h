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
                std::ofstream &programFile, 
                const char *initials, Space *lps);

/* this function calls the function above to generate classes for all LPSes */
void generateReaderWriters(const char *headerFile, 
		const char *programFile, 
		const char *initials, Space *rootLps);

/* this is a supporting routine that generates code for a part to actual data index in a file transformation
   that is needed by both reader and writer classes of a data structure. The transformation is often not staight-
   forward and we need a function for that due to the presence of data reordering partitions. */
void generateCodeForIndexTransformation(std::ofstream &headerFile, 
                std::ofstream &programFile, 
                const char *initials, ArrayDataStructure *structure);

/* function that generates a routine to initialize all parts of data structures in different LPSes of a task's 
   environment for which some input file has been specified during task invocation. */
void generateRoutineForDataInitialization(const char *headerFile, const char *programFile, TaskDef *taskDef);

/* function to generate a routine for writing the content of environment on files for data structures that have 
   been instructed to be saved in files at task invocation */
void generateRoutineForDataStorage(const char *headerFile, const char *programFile, TaskDef *taskDef);

#endif
