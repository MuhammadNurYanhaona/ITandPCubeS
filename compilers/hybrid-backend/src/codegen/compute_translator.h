#ifndef _H_compute_translator
#define _H_compute_translator

/* A lot of issues need to be tackled during translation of the compute section 
   of a task into a set of C++ compute functions and a thread::run routine to 
   execute them. Therefore we use this separate header file concerning compute 
   section translation alone. 
*/

#include <iostream>
#include <fstream>
#include "space_mapping.h"

class TaskDef;
class FlowStage;

/* function definitions for generating methods for compute stages */
int parseComputation(FlowStage *currentStage, const char *initialsLower, 
		const char *initialsUpper, 
		std::ofstream &headerFile,
		std::ofstream &programFile, 
		int currentFnNo, int topmostGpuPps);
void generateFnsForComputation(TaskDef *taskDef, const char *headerFile, 
		const char *programFile, const char *initials);

/* function definition for generating the thread::run function */
void generateThreadRunFunction(TaskDef *taskDef, const char *headerFile,
                const char *programFile,
                const char *initials,
                MappingNode *mappingRoot,
                bool involvesSynchronization,
                bool involvesReduction,
                int communicatorCount);

/* function definition for generating the computation flow executor in the batch
   execution mode (batch execution mode is used when the task has been mapped to
   the hybrid PCubeSModel of the target platform) */
void generateBatchComputeFunction(TaskDef *taskDef, const char *headerFile,
		const char *programFile, 
		const char *initials,
		bool involvesReduction, int communicatorCount);	

#endif
